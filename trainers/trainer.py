from abc import ABC, abstractmethod

from data_sources.data_source import DataSource
from data_sources.datatypes import CustomDataType
from models.model import Model


class TrainerConfig(ABC):
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
    def load_from_dir(cls, dir: str) -> "TrainerConfig":
        raise NotImplementedError


class Trainer(ABC):
    def __init__(self, config: TrainerConfig, model: Model, input_data_source: DataSource, target_data_source: DataSource):
        assert isinstance(config, TrainerConfig), f"config must be a TrainerConfig but got {type(config)}"
        assert isinstance(model, Model), \
            f"Trainer expects model to be of type models.model.Model but got {type(model)}"
        assert isinstance(input_data_source, DataSource), \
            f"Trainer expects input data source to be of type data_sources.data_source.DataSource but got {type(input_data_source)}"
        assert isinstance(target_data_source, DataSource), \
            f"Trainer expects input data source to be of type data_sources.data_source.DataSource but got {type(input_data_source)}"
        self._config = config
        self._model = model
        self._input_data_source = input_data_source
        self._target_data_source = target_data_source

    @property
    def model(self) -> Model:
        return self._model

    @property
    def input_data_source(self) -> DataSource:
        return self._input_data_source

    @property
    def target_data_source(self) -> DataSource:
        return self._target_data_source

    @abstractmethod
    def train(self) -> Model:
        raise NotImplementedError

    @abstractmethod
    @property
    def train_input_type(self) -> CustomDataType:
        raise NotImplementedError

    @abstractmethod
    @property
    def train_output_type(self) -> CustomDataType:
        raise NotImplementedError

    @abstractmethod
    @property
    def train_target_type(self) -> CustomDataType:
        raise NotImplementedError

    @abstractmethod
    def save_to_dir(self, dir: str) -> None:
        raise NotImplementedError

    @abstractmethod
    @classmethod
    def load_from_dir(cls, dir: str) -> "Trainer":
        raise NotImplementedError

    @abstractmethod
    @classmethod
    def instantiate(
            cls, config: TrainerConfig, model: Model, input_data_source: DataSource, target_data_source: DataSource
    ) -> "Trainer":
        # assert input/target custom data types of this trainer, model, input_data_source & output_data_source match
        # also setting up blank optimizers and whatever else that's specific to a fresh instantiation could go in here
        # return cls(config=config, model=model, input_data_source=input_data_source, target_data_source=target_data_source)
        raise NotImplementedError

    @abstractmethod
    @classmethod
    def load_checkpoint(cls, dir: str) -> "Trainer":
        # similar to load_from_dir (might even call load_from_dir) except we've gotta do checkpoint-specific setup
        #  like iterating our dataloaders until they get to the point where they left off
        raise NotImplementedError

