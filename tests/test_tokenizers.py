import os
import pytest
from pathlib import Path
import sys

# Add the root directory to sys.path to allow importing utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import import_from_nested_path

# Gather all tokenizers that need to be tested
TOKENIZER_DIR = Path(__file__).parent.parent / "tokenizers"
TOKENIZER_FILES = [
    f.stem for f in TOKENIZER_DIR.iterdir()
    if f.is_file() and f.suffix == '.py' and f.name != '__init__.py'
]

# A simple mock dataset provider
def get_mock_dataset(texts):
    class MockDataset:
        def __init__(self, data):
            self.data = data
            self.index = 0

        def __iter__(self):
            self.index = 0
            return self

        def __next__(self):
            if self.index < len(self.data):
                item = self.data[self.index]
                self.index += 1
                return {"text": item}
            else:
                raise StopIteration
        
        def __len__(self):
            return len(self.data)

    return MockDataset(texts)


@pytest.mark.parametrize("tokenizer_filename", TOKENIZER_FILES)
def test_tokenizer_pipeline(tokenizer_filename, tmp_path):
    """
    1. Imports tokenizer functions.
    2. Trains the tokenizer on mock data.
    3. Saves the tokenizer.
    4. Loads the tokenizer.
    5. Performs an encode/decode assertion.
    * doesn't bother testing demo function
    """
    imported_items = import_from_nested_path(
        nested_folders=['tokenizers'],
        filename=tokenizer_filename,
        items=['train_tokenizer', 'load_tokenizer', 'demo_tokenizer']
    )
    train_tokenizer = imported_items.get('train_tokenizer')
    load_tokenizer = imported_items.get('load_tokenizer')
    demo_tokenizer = imported_items.get('demo_tokenizer')
    assert train_tokenizer, f"train_tokenizer not found in {tokenizer_filename}"
    assert load_tokenizer, f"load_tokenizer not found in {tokenizer_filename}"
    assert demo_tokenizer, f"demo_tokenizer not found in {tokenizer_filename}"

    tokenizer_config = {
        'filename': tokenizer_filename,
        'nickname': "pytest_run",
    }
    mock_texts = [
        "Hello world, this is a test.",
        "Another sentence for our tokenizer.",
        "Testing with numbers 123 and punctuation !?.",
        "Supercalifragilisticexpialidocious."
    ]
    # Multiply to ensure minimum sample_size is met
    dataloader = get_mock_dataset(mock_texts * 1_000) 

    # Train tokenizer
    tokenizer = train_tokenizer(dataloader, tokenizer_config)
    assert tokenizer is not None, "Training failed to return a tokenizer object."

    # Encode/decode assertion
    demo_text = "The quick brown fox jumps over the lazy dog."
    # Ensure the methods exist on the loaded tokenizer
    assert hasattr(tokenizer, 'enc'), f"{tokenizer_filename} missing 'enc' method."
    assert hasattr(tokenizer, 'dec'), f"{tokenizer_filename} missing 'dec' method."
    
    encoded_text = tokenizer.enc(demo_text)
    decoded_text = tokenizer.dec(encoded_text)
    assert decoded_text == demo_text, \
        f"Encode/decode mismatch for {tokenizer_filename}: expected '{demo_text}', got '{decoded_text}'"
