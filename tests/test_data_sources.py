import os
import pytest
from pathlib import Path
import sys
import random

# Add the root directory to sys.path to allow importing utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import import_from_nested_path

# Gather all datasets that need to be tested
DATASET_DIR = Path(__file__).parent.parent / "data_sources"
DATASET_FILES = [
    f.stem for f in DATASET_DIR.iterdir()
    if f.is_file() and f.suffix == '.py' and f.name != '__init__.py'
]
def check_testable(filename):
    """Checks if a dataset module is marked as testable (i.e., __test__ is not False)."""
    try:
        imported_items = import_from_nested_path(
            nested_folders=['data_sources'],
            filename=filename,
            items=['__test__']
        )
        __test__ = imported_items.get('__test__', True) # Default to True if __test__ is not found
        return __test__
    except Exception:
        # If importing __test__ fails for any reason, assume it's testable
        return True
DATASET_FILES = [file for file in DATASET_FILES if check_testable(file)]

@pytest.mark.parametrize("dataset_filename", DATASET_FILES)
def test_dataset_api(dataset_filename):
    """
    Tests the API contract for dataset modules:
    1. Imports `get_dataset` function from the dataset module.
    2. Calls `get_dataset` with a mock configuration.
    3. Asserts that the returned dataset is iterable.
    4. Asserts that dataset items are dictionaries containing a 'text' key with a string value.
    """
    imported_items = import_from_nested_path(
        nested_folders=['data_sources'],
        filename=dataset_filename,
        items=['get_dataset']
    )
    get_dataset_func = imported_items.get('get_dataset')

    assert get_dataset_func, f"`get_dataset` function not found in {dataset_filename}.py"
    assert callable(get_dataset_func), f"`get_dataset` in {dataset_filename}.py is not a callable function."

    # each dataset should handle default values for any of its optional config arguments
    config = {'filename': dataset_filename}
    try:
        dataset = get_dataset_func(cfg=config, split='train', seed=42)
    except Exception as e:
        pytest.fail(f"Failed to get dataset from {dataset_filename}.py with config {config}. Error: {e}")

    assert hasattr(dataset, '__iter__'), f"Dataset from {dataset_filename}.py is not iterable."

    try:
        iterator = iter(dataset)
        first_item = next(iterator)

    except Exception as e:
        pytest.fail(f"Failed to iterate or get first item from dataset {dataset_filename}.py. Error: {e}")

    assert isinstance(first_item, dict), \
        f"First item from dataset {dataset_filename}.py is not a dictionary. Got: {type(first_item)}"
    
    assert 'text' in first_item, \
        f"First item from dataset {dataset_filename}.py does not contain a 'text' key. Keys: {list(first_item.keys())}"
    
    assert isinstance(first_item['text'], str), \
        f"'text' value in first item from dataset {dataset_filename}.py is not a string. Got: {type(first_item['text'])}"
