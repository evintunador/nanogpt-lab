"""
API MUST CONTAIN:
DataSourceConfig
DataSource.get_datasource() -> DataSource

and the datasets.Dataset should be useable like
item = next(dataset)['text']
for getting the raw text, with other optional dict keys
being for metadata
"""
from abc import ABC, abstractmethod

__test__ = False

class DataSourceConfig:
    def __init__(self, filename: str):
        self._filename = filename

    @property
    @abstractmethod
    def filename(self):
        return self._filename

    @abstractmethod
    def to_dict(self) -> :
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_dict(cls, d: dict):


class DataSource(ABC):
    @staticmethod
    @abstractmethod
    def get_datasource(cls) -> DataSource:
        raise NotImplementedError

    def __init__(self, config: DataSourceConfig):
        self._config = config

    @property
    def config(self):
        return self._config

    @abstractmethod
    def __getitem__(self, key):
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError

    @abstractmethod
    def __next__(self):
        raise NotImplementedError


