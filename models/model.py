from abc import ABC, abstractmethod
from typing import Generator

from data_sources.datatypes import CustomDataVal, CustomDataType


class ModelConfig(ABC):
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
    def load_from_dir(cls, dir: str) -> "ModelConfig":
        raise NotImplementedError


class Model(ABC):
    def __init__(self, config: ModelConfig):
        assert isinstance(config, ModelConfig), f"config must be a ModelConfig but got {type(config)}"
        self._config = config

    @property
    def config(self) -> ModelConfig:
        return self._config

    @abstractmethod
    def save_to_dir(self, dir: str) -> None:
        raise NotImplementedError

    @abstractmethod
    @classmethod
    def load_from_dir(cls, dir: str) -> "Model":
        raise NotImplementedError

    @abstractmethod
    @classmethod # is this redundant on the __init__ method?
    def instantiate(cls, config: ModelConfig) -> "Model":
        raise NotImplementedError

    @abstractmethod
    def forward(self, input: Generator[CustomDataVal, None, None]) -> Generator[CustomDataVal, None, None]:
        raise NotImplementedError

    @abstractmethod
    @property
    def forward_input_type(self) -> CustomDataType:
        raise NotImplementedError

    @abstractmethod
    @property
    def forward_output_type(self) -> CustomDataType:
        raise NotImplementedError