from abc import ABC, abstractmethod
from typing import Generator

import torch

from data_sources.datatypes import CustomDataType, CustomDataVal
from models.model import ModelConfig, Model


available_devices = ['cpu']
if torch.backends.mps.is_available():
    available_devices.append('mps')
if torch.cuda.is_available():
    if torch.version.hip is None:
        available_devices.append('cuda')
        available_devices.extend([f'cuda:{i}' for i in range(torch.cuda.device_count())])
    else:
        available_devices.append('hip')
        available_devices.extend([f'hip:{i}' for i in range(torch.cuda.device_count())])

class PyTorchModelConfig(ModelConfig, ABC):
    def __init__(self, python_filename: str, device: str = "cpu"):
        super().__init__(python_filename=python_filename)
        assert device in available_devices, f"device must be {available_devices} but got {device}"
        self._device = device

    @property
    def device(self) -> str:
        return self._device

    @abstractmethod
    def save_to_dir(self, dir: str) -> None:
        raise NotImplementedError

    @abstractmethod
    @classmethod
    def load_from_dir(cls, dir: str) -> "PyTorchModelConfig":
        raise NotImplementedError


class PyTorchModel(Model, ABC):
    def __init__(self, config: PyTorchModelConfig, model: torch.nn.Module):
        super().__init__(config=config)
        assert isinstance(model, torch.nn.Module), f"model must be a torch.nn.Module but got {type(model)}"
        self._model = model.to(self.config.device)

    @property
    def model(self) -> torch.nn.Module:
        return self._model

    @abstractmethod
    def forward(self, input: Generator[CustomDataVal, None, None]) -> Generator[CustomDataVal, None, None]:
        # the .forward() method needs to be the inner part that supports both training and inference
        # for example, in LLMs it would input Tokens and output Tensor (the sized [b,s,v] logits)
        raise NotImplementedError

    @abstractmethod
    @property
    def forward_input_type(self) -> CustomDataType:
        raise NotImplementedError

    @abstractmethod
    @property
    def forward_output_type(self) -> CustomDataType:
        raise NotImplementedError

    @abstractmethod
    def inference(self, input: Generator[CustomDataVal, None, None]) -> Generator[CustomDataVal, None, None]:
        # the .inference() method should wrap around the .forward() method for actual every-day use of the model
        # for example, in LLMs it input would be Text and output would be Text
        raise NotImplementedError

    @abstractmethod
    @property
    def inference_input_type(self) -> CustomDataType:
        raise NotImplementedError

    @abstractmethod
    @property
    def inference_output_type(self) -> CustomDataType:
        raise NotImplementedError

    @abstractmethod
    def save_to_dir(self, dir: str) -> None:
        raise NotImplementedError

    @abstractmethod
    @classmethod
    def load_from_dir(cls, dir: str) -> "PyTorchModel":
        raise NotImplementedError

    @abstractmethod
    def save_to_cloud(self, api_key: str, addr: str) -> None:
        # TODO: general huggingface model save
        raise NotImplementedError

    @abstractmethod
    @classmethod
    def load_from_cloud(cls, config: "PyTorchModelConfig", api_key: str, addr: str) -> "PyTorchModel":
        # TODO: general huggingface model load
        raise NotImplementedError

    @abstractmethod
    @classmethod # is this redundant on the __init__ method?
    def instantiate(cls, config: PyTorchModelConfig) -> "PyTorchModel":
        raise NotImplementedError