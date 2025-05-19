"""
Built off of Tiktoken educational implementation 
https://github.com/openai/tiktoken/blob/main/tiktoken/_educational.py
and this pytorch implementation of karpathy's minbpe
https://github.com/kuprel/minbpe-pytorch/tree/main
for tutorial-ish repo see
https://github.com/evintunador/gpu_bpe
"""
from __future__ import annotations
import collections
import os
import shutil
import argparse
import pickle
from itertools import chain
from typing import List
import yaml

from tqdm import tqdm
import regex
import numpy as np
from datasets import load_dataset
import tiktoken
import torch
import torch.distributed as dist
from datasets.distributed import split_dataset_by_node

# set __test__ to False for any file in tokenizers/ that should not be tested
# (AKA purposely does not meet API requierments)
__test__ = False

# Check if environment variables are set by torchrun, otherwise default to single GPU
if "RANK" in os.environ and "WORLD_SIZE" in os.environ and "LOCAL_RANK" in os.environ:
    # Multi-GPU setup with torchrun
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
else:
    # Single GPU setup
    rank = 0
    world_size = 1
    local_rank = 0
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
assert torch.cuda.is_available()
device = torch.device("cuda", local_rank)
torch.cuda.set_device(device)

# Initialize distributed process group if using multiple GPUs
if world_size > 1:
    dist.init_process_group(backend="nccl", device_id=device)
    dist.barrier()
master_process = (rank == 0)  # this process will do logging, checkpointing etc.
if master_process: print(f"Running with {world_size} GPU(s)")


def visualise_tokens(token_values: list[bytes]) -> None:
    background = [f"\u001b[48;5;{i}m" for i in [167, 179, 185, 77, 80, 68, 134]]
    # If token boundaries do not occur at unicode character boundaries, it's unclear how best to
    # demo the token. Here, we'll just use the unicode replacement character to represent some
    # fraction of a character.
    unicode_token_values = [x.decode("utf-8", errors="replace") for x in token_values]

    running_length = 0
    last_color = None
    for token in unicode_token_values:
        color = background[running_length % len(background)]
        if color == last_color:
            color = background[(running_length + 1) % len(background)]
            assert color != last_color
        last_color = color
        running_length += len(token)
        print(color + token, end="")
    print("\u001b[0m")



def bpe_encode(
    mergeable_ranks: dict[bytes, int], input: bytes, demo: bool = False
) -> list[int]:
    parts = [bytes([b]) for b in input]
    while True:
        # See the intermediate merges play out!
        if demo: demo_tokens(parts)

        # Iterate over all pairs and find the pair we want to merge the most
        min_idx = None
        min_rank = None
        for i, pair in enumerate(zip(parts[:-1], parts[1:])):
            rank = mergeable_ranks.get(pair[0] + pair[1])
            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx = i
                min_rank = rank

        # If there were no pairs we could merge, we're done!
        if min_rank is None:
            break
        assert min_idx is not None

        # Otherwise, merge that pair and leave the rest unchanged. Then repeat.
        parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2 :]

    tokens = [mergeable_ranks[part] for part in parts]
    return tokens


def slow_merge(words, most_common_pair, token_bytes):
    new_words = []
    for word in words:
        new_word = []
        i = 0
        while i < len(word) - 1:
            if (word[i], word[i + 1]) == most_common_pair:
                # We found our pair! Merge it
                new_word.append(token_bytes)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        if i == len(word) - 1:
            new_word.append(word[i])
        new_words.append(new_word)
    return new_words


def nat2int(num: int):
    """
    converts natural numbers to integer counterparts for use in
    efficiently utilizing signed int datatypes
    (0, 1, 2, 3,...) -> (0, -1, 1, -2, 2,...)
    """
    if num % 2 == 0:  # even numbers map to positive
        return num // 2
    else: # odd numbers map to negative
        return -(num + 1) // 2
    
def int2nat(num: int):
    """
    converts integer numbers to natural counterparts for use in
    efficiently utilizing signed int datatypes
    (0, -1, 1, -2, 2,...) -> (0, 1, 2, 3,...)
    """
    if num >= 0:  # positive numbers map back to even
        return 2 * num
    else:  # negative numbers map back to odd
        return -2 * num - 1


def bpe_train(
    data: str, vocab_size: int, pat_str: str, demo: bool = False, k: int = 256
) -> dict[bytes, int]:
    # First, add tokens for each individual byte value
    if vocab_size < 2**8:
        raise ValueError("vocab_size must be at least 256, so we can encode all bytes")
    ranks = {}
    for i in range(2**8):
        ranks[bytes([i])] = i
    
    # choose efficient data type
    int_type = torch.int16 if vocab_size <= (2**16)-2 else torch.int32
    assert vocab_size <= (2**31)-2, f"bro why you making such a big tokenizer? {vocab_size}"
    # set indicator tokens for merging ops
    SEPARATOR_TOKEN = -32_768 if int_type == torch.int16 else -2_147_483_648
    REMOVE_TOKEN = 32_767 if int_type == torch.int16 else 2_147_483_647

    # Splinter up our data into lists of bytes
    words: list[list[bytes]] = [
        [bytes([b]) for b in word.encode("utf-8")] for word in regex.findall(pat_str, data)
    ]
    # Create a list to store numeric token IDs for tensor operations
    # Initially, these are just byte values (0-255)
    byte_ids_lofl = [[ranks[b] for b in word] for word in words]
    # convert from (0,1,2,3...,255) to (0,-1,1,-2,2,...,-127,127,-128) to saturate int16
    ids_lofl = [[nat2int(num) for num in sublist] for sublist in byte_ids_lofl]
    # turn data into parseable tensor - using the token IDs instead of raw bytes
    ids = torch.tensor(
        list(chain.from_iterable(word + [SEPARATOR_TOKEN] for word in ids_lofl))[:-1], 
        dtype=int_type, device=device)
        # shape (words_in_data * (avg_word_len + 1))\
    
    if master_process and demo:
        # Initialize demo text tokens outside the loop to track changes across iterations
        demo_text = (f"This is a test of our custom trained BPE tokenizer on FineWeb data.\n"
                    f"It should handle punctuation, numbers (like 42 and 3.14159), and special characters ($#@!) properly.\n"
                    f"Supercalifragilisticexpialidocious antidisestablishmentarianism!!!")
        demo_words = [[bytes([b]) for b in word.encode("utf-8")] for word in regex.findall(pat_str, demo_text)]

    # Now, use our data to figure out which merges we should make
    progress_bar = tqdm(total=vocab_size - 256, unit="merges")
    for j in range(256, vocab_size):
        # find frequency of all pairs
        pairs = torch.stack((ids[:-1], ids[1:]), dim=0) # (2, words_in_data * (avg_word_len + 1))
        unique, counts = torch.unique(pairs, return_counts=True, dim=1)
            # shapes (2, very_long) and (very_long)
            # where very_long < words_in_data * (avg_word_len + 1)
        
        # use separator token between words to ensure we follow regex
        valid_mask = torch.all(unique != SEPARATOR_TOKEN, dim=0) # (very_long)
        unique = unique[:, valid_mask] # (2, very_long)
        counts = counts[valid_mask] # (very_long)

        if world_size > 1:
            # select top k pairs to go into consideration
            counts, sort_idx = torch.sort(counts, descending=True) # (very_long) and (very_long)
            pairs_idx = sort_idx[:k] # shape (k)
            most_common_pairs_local = unique[:, pairs_idx] # (2, k)
            counts_local = counts[:k]# (k)
            
            # communicate between GPUs
            most_common_pairs_global = torch.zeros((2, k * world_size), dtype=torch.float32, device=device)
            counts_global = torch.zeros(k * world_size, dtype=torch.float32, device=device)
            most_common_pairs_global[:, rank * k : (rank + 1) * k] = most_common_pairs_local.to(torch.float32)
            counts_global[rank * k : (rank + 1) * k] = counts_local.to(torch.float32)
            dist.all_reduce(most_common_pairs_global, op=dist.ReduceOp.SUM)
            dist.all_reduce(counts_global, op=dist.ReduceOp.SUM)

            # get unique pairs and their counts from the combined data
            unique_pairs, inverse_indices = torch.unique(most_common_pairs_global.t(), dim=0, return_inverse=True)

            # Sum the counts for each unique pair
            sum_counts = torch.zeros(unique_pairs.size(0), dtype=torch.float, device=device)
            sum_counts.scatter_add_(0, inverse_indices, counts_global.float())

            # Count occurrences of each unique pair
            pair_occurrences = torch.bincount(inverse_indices)

            # Find the maximum occurrence count
            max_occurrence = torch.max(pair_occurrences)

            # Create a mask for pairs with the maximum occurrence count
            max_occurrence_mask = (pair_occurrences == max_occurrence)

            # Filter to only consider pairs with the maximum occurrence count
            filtered_sum_counts = sum_counts[max_occurrence_mask]
            filtered_unique_pairs = unique_pairs[max_occurrence_mask]

            # Find the pair with the largest count among the filtered pairs
            max_index = torch.argmax(filtered_sum_counts)
            best_pair = filtered_unique_pairs[max_index].cpu().numpy() # (2)
        else:
            pair_idx = torch.argmax(counts) # (1)
            best_pair = unique[:, pair_idx].cpu().numpy() # (2)
            
        # Map token IDs back to the corresponding byte sequences
        # Using the dictionary in reverse to get the bytes corresponding to these IDs
        best_bytes = [None, None]
        best_pair_0 = int2nat(best_pair[0])
        best_pair_1 = int2nat(best_pair[1])
        for bytes_token, id_token in ranks.items():
            if id_token == best_pair_0:
                best_bytes[0] = bytes_token
            if id_token == best_pair_1:
                best_bytes[1] = bytes_token
        token_bytes = best_bytes[0] + best_bytes[1]
        new_token_id = len(ranks)
        # Add the new token!
        ranks[token_bytes] = new_token_id

        # Now merge that most common pair in all the words
        pair_mask = (pairs[0] == best_pair[0]) & (pairs[1] == best_pair[1]) 
        ids[:-1][pair_mask] = nat2int(new_token_id)
        ids[1:][pair_mask] = REMOVE_TOKEN
        keep_mask = (ids != REMOVE_TOKEN)
        ids = ids[keep_mask]

        if master_process:
            progress_bar.update(1)

            if demo:
                # Also apply the same merge to our demo text
                demo_words = slow_merge(demo_words, tuple(best_bytes), token_bytes)

                # See the intermediate merges play out!
                if j % 1000 == 0 or j in [256, vocab_size - 1]:
                    print(f"\nThe most common pair {int2nat(best_pair[0])} + {int2nat(best_pair[1])} "
                            f"which makes {token_bytes} our {len(ranks)}th token")
                    # Flatten the demo words into a single list of tokens for visualization
                    demo_tokens = [token for word in demo_words for token in word]
                    visualise_tokens(demo_tokens)
    
    print(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
            f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB")

    return ranks


def fetch_fineweb_data(max_chars: int = 2**22):
    """Fetch data from FineWeb dataset for tokenizer training"""
    # Create a local cache directory
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Check for existing files that meet the size requirement
    def find_best_file():
        existing_files = []
        for file in os.listdir(data_dir):
            if file.startswith("tokenizer_training_data_") and file.endswith(".txt"):
                try:
                    file_size = int(file.split("_")[-1].split(".")[0])
                    existing_files.append((file, file_size))
                except ValueError:
                    continue
        
        # Find suitable existing files (>= max_chars)
        suitable_files = [(f, s) for f, s in existing_files if s >= max_chars]
        
        if suitable_files:
            # Use the smallest file that meets our requirements
            suitable_files.sort(key=lambda x: x[1])
            return suitable_files[0]
        return None, None
    
    # First check if we have a suitable file already
    best_file, best_size = find_best_file()
    
    # Only the master process handles downloading if needed
    if rank == 0 and best_file is None:
        # Clean up smaller files
        for file in os.listdir(data_dir):
            if file.startswith("tokenizer_training_data_") and file.endswith(".txt"):
                try:
                    file_size = int(file.split("_")[-1].split(".")[0])
                    if file_size < max_chars:
                        print(f"Removing smaller existing data file: {file} ({file_size:,} chars < {max_chars:,} chars)")
                        os.remove(os.path.join(data_dir, file))
                except ValueError:
                    continue
        
        # Download new data
        new_file_name = f"tokenizer_training_data_{max_chars}.txt"
        local_data_path = os.path.join(data_dir, new_file_name)
        print(f"Downloading FineWeb data to {local_data_path}...")
        
        dataset = load_dataset("HuggingFaceFW/fineweb", 
                              name="sample-10BT", 
                              split="train", 
                              streaming=True)
        
        text_data = []
        doc_lengths = []
        tot_len = 0
        for item in dataset:
            text_data.append(item["text"])
            doc_lengths.append(len(item["text"]))
            tot_len += len(item["text"])
            if tot_len >= max_chars:
                break
        
        # Show statistics
        print(f"\nDataset Statistics:"
            f"\nTotal documents: {len(text_data)}"
            f"\nTotal characters: {sum(doc_lengths):,}"
            f"\nAverage document length: {np.mean(doc_lengths):.1f} characters"
            f"\nMedian document length: {np.median(doc_lengths):.1f} characters"
            f"\nShortest document: {min(doc_lengths)} characters"
            f"\nLongest document: {max(doc_lengths):,} characters"
            f"\nStandard deviation: {np.std(doc_lengths):.1f} characters")
        
        # Save the combined text to a file
        with open(local_data_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(text_data))
        print(f"Data saved to {local_data_path}")
    
    # Make sure all processes wait until the data is downloaded
    if world_size > 1:
        dist.barrier()
    
    # After potential download, all ranks check again for the best file
    best_file, best_size = find_best_file()
    assert best_file is not None, f"No suitable data file found for size {max_chars}"
    
    local_data_path = os.path.join(data_dir, best_file)
    if rank == 0:
        print(f"All ranks using data from first {max_chars} characters of {local_data_path}")
    
    # All processes read from the local file and trim if needed
    with open(local_data_path, 'r', encoding='utf-8') as f:
        data = f.read()[:max_chars]
    
    # Trim data if it's larger than needed
    """if len(data) > max_chars:
        data = data[:max_chars]
        if rank == 0:
            print(f"Trimmed data to {max_chars:,} characters")"""
    
    # Shard the data for distributed processing if needed
    if world_size > 1:
        # Simple way to shard the data: split by character count
        shard_size = max_chars // world_size
        start = rank * shard_size
        end = start + shard_size if rank < world_size - 1 else max_chars
        data = data[start:end]
        print(f"Rank {rank}: Processing {len(data):,} characters ({start:,} to {end:,})")
    
    return data


def train_simple_encoding(sample_size: int, vocab_size: int, demo: bool = False, k: int = 256):
    """
    Train a custom BPE tokenizer using FineWeb data.
    
    Args:
        sample_size: maximum number of characters to include in data
        vocab_size: Size of the vocabulary to train
        demo: Visualization mode for BPE training process
    
    Returns:
        The trained tokenizer
    """
    data = fetch_fineweb_data(max_chars=sample_size)
    
    #gpt2_pattern = (r"""'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    gpt4_pattern = (
        r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
    )
    enc = SimpleBytePairEncoding.train(data, vocab_size=vocab_size, pat_str=gpt4_pattern, demo=demo, k=k)
    
    # Test the tokenizer with a simple example
    test_str = f"hello world rank={rank}"
    tokens = enc.encode(test_str)
    # Verify encoding-decoding roundtrips correctly
    decoded = enc.decode(tokens)
    assert decoded == test_str, f"Decoding failed: expected '{test_str}' but got '{decoded}'"
    decoded_bytes = enc.decode_bytes(tokens)
    assert decoded_bytes == test_str.encode('utf-8'), \
        f"Bytes decoding failed: expected {test_str.encode('utf-8')} but got {decoded_bytes}"
    
    return enc











class TokenizerClass:
    """
    This Tokenizer is just a wrapper around tiktoken which is written in Rust
    and therefore faster than anything we could do with Python
    """
    def __init__(self, tiktoken_encoding):
        """Initialize with a tiktoken Encoding object"""
        self.encoding = tiktoken_encoding
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs using the underlying tiktoken encoding"""
        return self.encoding.encode(text)
    
    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs back to text using the underlying tiktoken encoding"""
        return self.encoding.decode(tokens)


def train_tokenizer(dataloader, tokenizer_cfg) -> TokenizerClass:
    """Train a BPE tokenizer on some data!"""
    mergeable_ranks = bpe_train(
        dataloader=dataloader, 
        vocab_size=tokenizer_cfg['vocab_size'], 
        pat_str=tokenizer_cfg['regex'],  
        k=tokenizer_cfg['k']
    )
    enc = tiktoken.Encoding(
        name=tokenizer_cfg['nickname'],
        pat_str=tokenizer_cfg['regex'],
        mergeable_ranks=mergeable_ranks,
        special_tokens={
            tok: len(mergeable_ranks) + i
            for i, tok in enumerate(tokenizer_cfg['special_tokens'])
        }
    )
    return TokenizerClass(enc)


def demo_tokenizer(tokenizer, demo_text):
    pass


def save_tokenizer(tokenizer: TokenizerClass, tokenizer_cfg):
    save_dir = (f"tokenizers/trained/"
            f"/{tokenizer_cfg['nickname']}"
            f"_v{tokenizer_cfg['vocab_size']}"
            f"_n{tokenizer_cfg['sample_size']}")
    os.makedirs(save_dir, exist_ok=True)
    with open(f"{save_dir}/mergeable_ranks.pkl", 'wb') as f:
        pickle.dump(tokenizer.encoding.mergeable_ranks, f)
    # Save the tokenizer configuration as YAML
    with open(f"{save_dir}/config.yaml", 'w') as f:
        yaml.dump(tokenizer_cfg, f, default_flow_style=False)
    print(f"Tokenizer saved to {save_dir}")


def load_tokenizer(tokenizer_cfg) -> TokenizerClass:
    path = f"tokenizers/trained/{tokenizer_cfg['nickname']}/mergeable_ranks.pkl"
    with open(path, 'rb') as f:
        mergeable_ranks = pickle.load(f)
    enc = tiktoken.Encoding(
        name=tokenizer_cfg['nickname'],
        pat_str=tokenizer_cfg['regex'],
        mergeable_ranks=mergeable_ranks,
        special_tokens={
            tok: len(mergeable_ranks) + i
            for i, tok in enumerate(tokenizer_cfg['special_tokens'])
        }
    )
    return TokenizerClass(enc)

