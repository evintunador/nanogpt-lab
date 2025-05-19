import os
import pytest
from pathlib import Path
import sys
import random
import string

# Add the root directory to sys.path to allow importing utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import import_from_nested_path

# Gather all tokenizers that need to be tested
TOKENIZER_DIR = Path(__file__).parent.parent / "tokenizers"
TOKENIZER_FILES = [
    f.stem for f in TOKENIZER_DIR.iterdir()
    if f.is_file() and f.suffix == '.py' and f.name != '__init__.py'
]
def check_testable(filename):
    try:
        imported_items = import_from_nested_path(
            nested_folders=['tokenizers'],
            filename=filename,
            items=['__test__']
        )
        __test__ = imported_items.get('__test__', True)
        return __test__
    except Exception as e:
        print(e)
        return True
TOKENIZER_FILES = [file for file in TOKENIZER_FILES if check_testable(file)]

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
    1. Imports tokenizer components (class and functions).
    2. Trains the tokenizer on mock data.
    3. Performs an encode/decode assertion.
    """
    imported_items = import_from_nested_path(
        nested_folders=['tokenizers'],
        filename=tokenizer_filename,
        items=['Tokenizer']
    )
    Tokenizer = imported_items.get('Tokenizer')

    assert Tokenizer, f"Tokenizer class not found in {tokenizer_filename}"
    for method in ['train', 'enc', 'dec', 'enc_bytes', 'dec_bytes']:
        assert hasattr(Tokenizer, method) and callable(getattr(Tokenizer, method, None)), \
        f"Tokenizer must have an method .{method}()"

    mock_config = {
        'filename': tokenizer_filename,
        'nickname': "pytest_run",
    }
    chars = string.ascii_letters + string.digits + ' '
    mock_texts = []
    while len(mock_texts) < 1_000:
        length = random.randint(50, 500)
        text = ''.join(random.choice(chars) for _ in range(length))
        mock_texts.append(text)
    dataloader = get_mock_dataset(mock_texts) 

    # Train tokenizer
    tokenizer = Tokenizer.train(dataloader, mock_config)
    assert tokenizer is not None, "Training failed to return a tokenizer object."

    # Encode/decode assertions
    demo_text = "The quick brown fox jumps over the lazy dog."
    encoded_text = tokenizer.enc(demo_text)
    decoded_text = tokenizer.dec(encoded_text)
    assert decoded_text == demo_text, \
        f"instance methods encode/decode mismatch for {tokenizer_filename}: expected '{demo_text}', got '{decoded_text}'"
    encoded_text = tokenizer.enc_bytes(demo_text.encode('utf-8'))
    decoded_text = "".join(tokenizer.dec_bytes(encoded_text))
    assert decoded_text == demo_text, \
        f"instance methods encode_bytes/decode_bytes mismatch for {tokenizer_filename}: expected '{demo_text}', got '{decoded_text}'"