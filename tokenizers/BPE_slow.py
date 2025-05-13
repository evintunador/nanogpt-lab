"""
API MUST CONTAIN:
train_tokenizer(datasets.Dataset, tokenizer_config: dict) -> TokenizerClass:
save_tokenizer(TokenizerClass, tokenizer_config: dict) -> None:
load_tokenizer(tokenizer_config: dict) -> TokenizerClass:
demo_tokenizer(TokenizerClass, demo_text) -> None (print to console)


"""
import os
import pickle
from pathlib import Path

from tqdm import tqdm
import numpy as np
import regex
from datasets import Dataset


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


class TokenizerClass:
    """
    A simple tokenizer class written in Python.
    """
    def __init__(self, *, pat_str: str, mergeable_ranks: dict[bytes, int]) -> None:
        """Creates an Encoding object."""
        # A regex pattern string that is used to split the input text
        self.pat_str = pat_str
        # A dictionary mapping token bytes to their ranks. The ranks correspond to merge priority
        self.mergeable_ranks = mergeable_ranks

        self._decoder = {token: token_bytes for token_bytes, token in mergeable_ranks.items()}
        self._pat = regex.compile(pat_str)

    def bpe_encode(self, mergeable_ranks: dict[bytes, int], input: bytes) -> list[int]:
        parts = [bytes([b]) for b in input]
        while True:
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

    def encode(self, text: str, demo: bool = False) -> list[int]:
        """Encodes a string into tokens.

        >>> enc.encode("hello world")
        [388, 372]
        """
        # Use the regex to split the text into (approximately) words
        words = self._pat.findall(text)
        tokens = []
        for word in words:
            # Turn each word into tokens, using the byte pair encoding algorithm
            word_bytes = word.encode("utf-8")
            word_tokens = self.bpe_encode(self.mergeable_ranks, word_bytes)
            tokens.extend(word_tokens)
        return tokens

    def decode_bytes(self, tokens: list[int]) -> bytes:
        """Decodes a list of tokens into bytes.

        >>> enc.decode_bytes([388, 372])
        b'hello world'
        """
        return b"".join(self._decoder[token] for token in tokens)

    def decode(self, tokens: list[int]) -> str:
        """Decodes a list of tokens into a string.

        Decoded bytes are not guaranteed to be valid UTF-8. In that case, we replace
        the invalid bytes with the replacement character "�".

        >>> enc.decode([388, 372])
        'hello world'
        """
        return self.decode_bytes(tokens).decode("utf-8", errors="replace")

    def decode_tokens_bytes(self, tokens: list[int]) -> list[bytes]:
        """Decodes a list of tokens into a list of bytes.

        Useful for visualising how a string is tokenised.

        >>> enc.decode_tokens_bytes([388, 372])
        [b'hello', b' world']
        """
        return [self._decoder[token] for token in tokens]
    
    def enc(self, text: str, demo: bool = False) -> list[int]:
        return self.encode(text=text, demo=demo)
    
    def dec(self, tokens: list[int]) -> str:
        return self.decode(tokens=tokens)
    

def merge_bytes(words, most_common_pair, token_bytes):
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


def get_stats(list_of_ids):
    """
    Given a list of lists of integers, return a dictionary of counts of consecutive pairs
    Example: [[1, 2, 3, 1, 2]] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    """
    counts = {}
    for word in list_of_ids:
        for pair in zip(word, word[1:]): # iterate consecutive elements
            counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge_ids(ids: list[list[int]], pair: tuple[int], idx: int):
    """
    In the list of integers (ids), replace all consecutive occurrences
    of pair with the new integer token idx
    Example: ids=[[1, 2, 3, 1, 2]], pair=(1, 2), idx=4 -> [[4, 3, 4]]
    """
    new_ids = []
    for word in ids: # word is a list of integers
        new_word = []
        i = 0
        while i < len(word):
            # if not at the very last position AND the pair matches, replace it
            # Use word[i] and len(word) instead of ids[i] and len(ids)
            if word[i] == pair[0] and i < len(word) - 1 and word[i+1] == pair[1]:
                new_word.append(idx)
                i += 2
            else:
                # Append the current token from the word
                new_word.append(word[i])
                i += 1
        new_ids.append(new_word)
    return new_ids


def bpe_train(data: str, vocab_size: int, pat_str: str) -> dict[bytes, int]:
    # First, add tokens for each individual byte value
    if vocab_size < 2**8:
        raise ValueError("vocab_size must be at least 256, so we can encode all bytes")
    ranks = {}
    for i in range(2**8):
        ranks[bytes([i])] = i
    
    # Splinter up our data into lists of bytes
    words: list[list[bytes]] = [
        [bytes([b]) for b in word.encode("utf-8")] for word in regex.findall(pat_str, data)
    ]
    # Create a list to store numeric token IDs for tensor operations
    # Initially, these are just byte values (0-255)
    ids = [[ranks[b] for b in word] for word in words]
    
    # Initialize demo text tokens outside the loop to track changes across iterations
    demo_text = (f"This is a test of our custom trained BPE tokenizer on FineWeb data.\n"
                f"It should handle punctuation, numbers (like 42 and 3.14159), and special characters ($#@!) properly.\n"
                f"Supercalifragilisticexpialidocious antidisestablishmentarianism!!!")
    demo_bytes = [[bytes([b]) for b in word.encode("utf-8")] for word in regex.findall(pat_str, demo_text)]

    # Now, use our data to figure out which merges we should make
    for j in tqdm(range(256, vocab_size), unit="merges"):
        stats = get_stats(ids)
        best_pair = max(stats, key=stats.get)
        
        # Map token IDs back to the corresponding byte sequences
        # Using the dictionary in reverse to get the bytes corresponding to these IDs
        best_bytes = [None, None]
        for bytes_token, id_token in ranks.items():
            if id_token == best_pair[0]:
                best_bytes[0] = bytes_token
            if id_token == best_pair[1]:
                best_bytes[1] = bytes_token
        token_bytes = best_bytes[0] + best_bytes[1]
        new_token_id = len(ranks)
        # Add the new token!
        ranks[token_bytes] = new_token_id

        # Now merge that most common pair in all the words
        ids = merge_ids(ids, best_pair, new_token_id)

        # Also apply the same merge to our demo text
        demo_bytes = merge_bytes(demo_bytes, tuple(best_bytes), token_bytes)

        # See the intermediate merges play out!
        if j % 1000 == 0 or j in [256, vocab_size - 1]:
            print(f"\nThe most common pair {best_pair[0]} + {best_pair[1]} "
                    f"which makes {token_bytes} our {len(ranks)}th token")
            # Flatten the demo words into a temporary list for visualization
            # Do not reassign back to demo_bytes
            flat_demo_bytes = [token for word in demo_bytes for token in word]
            visualise_tokens(flat_demo_bytes)

    return ranks


def fetch_fineweb_data(dataset, max_chars: int):
    """Fetch data from FineWeb dataset for tokenizer training"""
    # Create a local cache directory
    data_dir = os.path.join(os.path.dirname(__file__), "temp_dir")
    os.makedirs(data_dir, exist_ok=True)

    # Check for existing files that meet the size requirement
    existing_files = []
    for file in os.listdir(data_dir):
        if file.endswith(".txt") and file.startswith("tokenizer_training_data_"):
            try:
                # Extract size from filename like 'tokenizer_training_data_1000.txt'
                file_size = int(file.split("_")[-1].split(".")[0])
                existing_files.append((file, file_size))
            except (ValueError, IndexError):
                # Ignore files with unexpected naming format
                continue

    if existing_files:
        # Find suitable existing files (>= max_chars)
        suitable_files = [(f, s) for f, s in existing_files if s >= max_chars]

        if suitable_files:
            # Use the smallest file that meets our requirements
            suitable_files.sort(key=lambda x: x[1])
            filename, _ = suitable_files[0] # Get filename from the tuple
            filepath = os.path.join(data_dir, filename) # Construct the full path
            print(f"Using existing data file: {filepath}") # Inform the user
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read(max_chars) # Read up to max_chars from the file
            return content # Return the actual file content
    
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
    final_text = "\n".join(text_data)[:max_chars]
    with open(local_data_path, 'w', encoding='utf-8') as f:
        f.write(final_text)
        
    return final_text


def train_tokenizer(dataset: Dataset, tokenizer_config: dict) -> TokenizerClass:
    data = fetch_fineweb_data(dataset, max_chars=tokenizer_config['sample_size'])
    mergeable_ranks = bpe_train(
        data=data, 
        vocab_size=tokenizer_config['vocab_size'], 
        pat_str=tokenizer_config['regex_pattern']
    )
    izer = TokenizerClass(
        pat_str=tokenizer_config['regex_pattern'], 
        mergeable_ranks=mergeable_ranks
    )
    # Test the tokenizer with a simple example
    test_str = f"hello world"
    tokens = izer.encode(test_str)
    decoded = izer.decode(tokens)
    assert decoded == test_str, f"Decoding failed: expected '{test_str}' but got '{decoded}'"
    decoded_bytes = izer.decode_bytes(tokens)
    assert decoded_bytes == test_str.encode('utf-8'), \
        f"Bytes decoding failed: expected {test_str.encode('utf-8')} but got {decoded_bytes}"
    return izer


def save_tokenizer(tokenizer: TokenizerClass, save_dir: str | Path) -> None:
    """Saves the tokenizer to a file."""
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, f"tokenizer.pkl")
    with open(filepath, 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f"Tokenizer saved to {filepath}")


def load_tokenizer(save_dir: str | Path) -> TokenizerClass:
    """Loads a tokenizer from a file."""
    filepath = os.path.join(save_dir, f"tokenizer.pkl")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Tokenizer file not found: {filepath}")
    with open(filepath, 'rb') as f:
        tokenizer = pickle.load(f)
    print(f"Tokenizer loaded from {filepath}")
    return tokenizer


def demo_tokenizer(tokenizer: TokenizerClass, demo_text: str) -> None:
    """Demonstrates the tokenizer by visualizing tokenization of demo_text."""
    print(f"Original text: {demo_text}")
    tokens = tokenizer.encode(demo_text)
    print(f"Encoded tokens: {tokens}")
    token_bytes = tokenizer.decode_tokens_bytes(tokens)
    print("Visualized tokens:")
    visualise_tokens(token_bytes)