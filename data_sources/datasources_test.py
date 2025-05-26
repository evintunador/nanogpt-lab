from pathlib import Path

import pytest
from pyarrow.dataset import dataset

from utils import import_from_nested_path
from data_sources.datasource import DataSourceConfig, DataSource

# Gather all datasets that need to be tested
DATASET_DIR = Path(__file__)
DATASET_FILES = [
    f.stem for f in DATASET_DIR.iterdir()
    if f.is_file() and f.suffix == '.py' and f.name != 'datasources_test.py'
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
        # If importing __test__ fails for any reason, assume it's testable so that an error gets thrown
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
        items=['__config_name__', '__datasource_name__']
    )
    config_name = imported_items.get('__config_name__')
    datasource_name = imported_items.get('__datasource_name__')
    imported_items = import_from_nested_path(
        nested_folders=['data_sources'],
        filename=dataset_filename,
        items=[config_name, datasource_name]
    )
    config_cls = imported_items.get('config_name')
    datasource_cls = imported_items.get('datasource_name')

    assert isinstance(config_cls, DataSourceConfig), \
        f"DatSourceConfig class not found in {dataset_filename}.py"
    assert isinstance(datasource_cls, DataSource), \
        f"DataSource class not found in {dataset_filename}.py"
    assert hasattr(datasource_cls, 'get_datasource'), \
        f"DataSource class in {dataset_filename}.py does not have a `get_datasource` method."
    assert callable(datasource_cls.get_datasource), \
        f"DataSource class in {dataset_filename}.py has a `get_datasource` attribute that is not callable."

    # each dataset should handle default values for any of its optional config arguments
    minimal_config = {'filename': dataset_filename}
    try:
        datasource = datasource_cls.get_datasource(config=minimal_config)
    except Exception as e:
        pytest.fail(f"Failed to get dataset from {dataset_filename}.py with config {minimal_config}. Error: {e}")

    assert hasattr(datasource, '__getitem__'), \
        f"Datasource from {dataset_filename}.py has no __getitem__ method."
    assert hasattr(datasource, '__len__'), \
        f"Datasource from {dataset_filename}.py has no __len__ method."
    assert hasattr(datasource, '__iter__'), \
        f"Datasource from {dataset_filename}.py has no __iter__ method."

    try:
        iterator = iter(dataset)
        first_item_from_iter = next(iterator)
        first_item_from_get = dataset[0]"
    except Exception as e:
        pytest.fail(f"Failed to iterate or get first item from dataset {dataset_filename}.py. Error: {e}")

    assert first_item_from_iter == first_item_from_get, \
        f"First item from dataset {dataset_filename}.py is not equal to the first item from dataset iterator. Got: {first_item_from_iter} != {first_item_from_get}