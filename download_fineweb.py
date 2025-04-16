"""
FineWeb dataset (for srs pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb

example doc to highlight the structure of the dataset:
{
  "text": "Posted by mattsmith on 20th April 2012\nStraight from...",
  "id": "<urn:uuid:d853d453-196e-4488-a411-efc2b26c40d2>",
  "dump": "CC-MAIN-2013-20",
  "url": "http://nleastchatter.com/philliesphandom/tag/freddy-galvis/",
  "date": "2013-05-18T07:24:47Z",
  "file_path": "s3://commoncrawl/long.../path.../file.gz",
  "language": "en",
  "language_score": 0.9185474514961243,
  "token_count": 594
}
"""
import os
import argparse
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm
import argparse
import numpy as np
import pickle
from functools import partial

def write_datafile(filename, toks):
    """ 
    Saves token data as a .bin file, for reading in C.
    - First comes a header with 256 int32s
    - The tokens follow, each as a uint16
    """
    assert len(toks) < 2**31, "token count too large" # ~2.1B tokens
    # construct the header
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520 # magic
    header[1] = 1 # version
    header[2] = len(toks) # number of tokens after the 256*4 bytes of header (each 2 bytes as uint16)
    # construct the tokens numpy array, if not already
    if not isinstance(toks, np.ndarray) or not toks.dtype == np.uint16:
        # validate that no token exceeds a uint16
        maxtok = 2**16
        assert all(0 <= t < maxtok for t in toks), "token dictionary too large for uint16"
        toks_np = np.array(toks, dtype=np.uint16)
    else:
        toks_np = toks
    # write to file
    print(f"writing {len(toks):,} tokens to {filename}")
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(toks_np.tobytes())

def tokenize(doc, enc, eot):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    tokens = [eot] # the special <|endoftext|> token delimits all documents
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

def main():
    parser = argparse.ArgumentParser(description="FineWeb dataset preprocessing")
    parser.add_argument("-v", "--version", type=str, default="10Bedu", help="Which version of fineweb to use? 10B|100B|10Bedu|100Bedu (default 10Bedu)")
    parser.add_argument("-ss", "--shard_size", type=int, default=10**8, help="Size of each shard in tokens (default 100 million)")
    parser.add_argument("-ns", "--num_shards", type=int, default=None, help="Maximum number of shards to create (defaults to entire dataset)")
    parser.add_argument("-t", "--tokenizer", type=str, default="gpt4regex_v50256_n134217728", help="Filename of custom tokenizer (no default)")
    args = parser.parse_args()
    assert args.tokenizer[-4:] == ".pkl", f"tokenizer must be .pkl"

    # FineWeb has a few possible subsamples available
    assert args.version in ["10B", "100B", "10Bedu", "100Bedu"], "version must be one of 10B, 100B, 10Bedu, or 100Bedu"
    if args.version == "10B":
        hug_name = "HuggingFaceFW/fineweb"
        remote_name = "sample-10BT"
        savename = "fineweb"
    elif args.version == "100B":
        hug_name = "HuggingFaceFW/fineweb"
        remote_name = "sample-100BT"
        savename = "fineweb"
    elif args.version == "10Bedu":
        hug_name = "HuggingFaceFW/fineweb-edu"
        remote_name = "sample-10BT"
        savename = "finewebedu"
    elif args.version == "100Bedu":
        hug_name = "HuggingFaceFW/fineweb-edu"
        remote_name = "sample-100BT"
        savename = "finewebedu"
    local_dir = "data"

    # create the cache the local directory if it doesn't exist yet
    DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    # stream the dataset
    fw = load_dataset(hug_name, name=remote_name, split="train", streaming=True)

    # Load the tokenizer
    tokenizer_config = pickle.load(open(f"tokenizers/{args.tokenizer}", 'rb'))
    enc = tiktoken.Encoding(
        name=args.tokenizer[:-4], # :-4 to remove the .pkl
        pat_str=tokenizer_config['pat_str'],
        mergeable_ranks=tokenizer_config['mergeable_ranks'],
        special_tokens={
            "<|endoftext|>": len(tokenizer_config['mergeable_ranks']),
        }
    )
    eot = enc._special_tokens['<|endoftext|>'] # end of text token
    
    # Create a partial function with the enc and eot parameters
    tokenize_with_params = partial(tokenize, enc=enc, eot=eot)

    # tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
    nprocs = max(1, os.cpu_count() - 2) # don't hog the entire system
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        # preallocate buffer to hold current shard
        all_tokens_np = np.empty((args.shard_size,), dtype=np.uint16)
        token_count = 0
        progress_bar = None
        for tokens in pool.imap(tokenize_with_params, fw, chunksize=16):
            # is there enough space in the current shard for the new tokens?
            if token_count + len(tokens) < args.shard_size:
                # simply append tokens to current shard
                all_tokens_np[token_count:token_count+len(tokens)] = tokens
                token_count += len(tokens)
                # update progress bar
                if progress_bar is None:
                    progress_bar = tqdm(total=args.shard_size, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(len(tokens))
            else:
                # write the current shard and start a new one
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(DATA_CACHE_DIR, f"{savename}_{split}_{shard_index:06d}.bin")
                # split the document into whatever fits in this shard; the remainder goes to next one
                remainder = args.shard_size - token_count
                progress_bar.update(remainder)
                all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
                write_datafile(filename, all_tokens_np)
                shard_index += 1
                
                # Stop if we've reached the maximum number of shards
                if args.num_shards is not None and shard_index >= args.num_shards + 1: # +1 since 0 is val
                    print(f"Reached maximum number of shards ({args.num_shards}). Stopping.")
                    break
                    
                progress_bar = None
                # populate the next shard with the leftovers of the current doc
                all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
                token_count = len(tokens)-remainder

        # write any remaining tokens as the last shard if we haven't reached num_shards
        if token_count != 0 and (args.num_shards is None or shard_index < args.num_shards + 1):
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"{savename}_{split}_{shard_index:06d}.bin")
            write_datafile(filename, all_tokens_np[:token_count])

if __name__ == '__main__':
    mp.freeze_support()  # Support for freezing the process
    main()