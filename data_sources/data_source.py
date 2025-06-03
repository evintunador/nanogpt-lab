from abc import ABC, abstractmethod

from data_sources.datatypes import CustomDataType, CustomDataVal


class DataSourceConfig(ABC):
    def __init__(self, python_filename: str):
        assert isinstance(python_filename, str), f"python_filename must be a string but got {type(python_filename)}"
        self._python_filename = python_filename

    @property
    def python_filename(self) -> str:
        return self._python_filename

    @abstractmethod
    def save_to_dir(self, dir: str) -> None:
        raise NotImplementedError

    @abstractmethod
    @classmethod
    def load_from_dir(cls, dir: str) -> "DataSourceConfig":
        raise NotImplementedError


class DataSource(ABC):
    def __init__(self, config: DataSourceConfig):
        assert isinstance(config, DataSourceConfig), f"config must be a DataSourceConfig but got {type(config)}"
        self._config = config

    @property
    def config(self) -> DataSourceConfig:
        return self._config

    @abstractmethod
    def get_data_generator(self) -> Generator[CustomDataVal, None, None]:
        raise NotImplementedError

    @abstractmethod
    @classmethod
    def initialize(cls, config: DataSourceConfig) -> "DataSource":
        raise NotImplementedError

    @abstractmethod
    def save_to_dir(self, dir: str) -> None:
        raise NotImplementedError

    @abstractmethod
    @classmethod
    def load_from_dir(cls, dir: str) -> "DataSource":
        raise NotImplementedError

