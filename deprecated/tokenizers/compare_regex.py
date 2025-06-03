import sys
import os

import regex

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deprecated.utils import visualise_tokens

# set __test__ to False for any file in tokenizers/ that should not be tested
# (AKA purposely does not meet API requierments)
__test__ = False

# gpt4's pattern
regex_pattern1 = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
# gpt4's pattern but with numerical digits separated
regex_pattern2 = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+| ?\d| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

test_strings = [
    "This is a test of our custom trained BPE tokenizer on FineWeb data.",
    "It should handle punctuation, numbers (like 42 and 3.14159), and special characters ($#@!) properly.",
    "Supercalifragilisticexpialidocious antidisestablishmentarianism!!!",
    "Here's an example: don't, can't, I'm, they're, we've.",
    "Numbers: 123 4567 89.0",
    "Whitespace    and\nnewlines\r\nare here."
]

pat1 = regex.compile(regex_pattern1)
pat2 = regex.compile(regex_pattern2)

for i, s in enumerate(test_strings, 1):
    print(f"\nExample {i}: {repr(s)}")
    tokens1 = pat1.findall(s)
    tokens2 = pat2.findall(s)
    print(" Pattern 1 tokens:")
    visualise_tokens([t.encode('utf-8') for t in tokens1])
    print(" Pattern 2 tokens:")
    visualise_tokens([t.encode('utf-8') for t in tokens2])
    # Visual diff
    if tokens1 != tokens2:
        print(f"  Token count difference: {len(tokens1)} (pattern1) vs {len(tokens2)} (pattern2) = {len(tokens1) - len(tokens2)}")
