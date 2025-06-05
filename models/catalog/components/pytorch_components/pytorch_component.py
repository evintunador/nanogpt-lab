from abc import ABC, abstractmethod
from typing import Tuple, Union, Dict, List, Callable

import torch
import torch.nn as nn


FLOAT_DTYPES = {torch.float16, torch.float32, torch.float64, torch.bfloat16}
INT_DTYPES = {torch.int8, torch.int16, torch.int32, torch.int64}
UINT_DTYPES = {torch.uint8}
SIGNED_DTYPES = INT_DTYPES | FLOAT_DTYPES
# example usage:
#assert my_dtype in UINT_DTYPES


class TensorInfo:
    def __init__(self, shape: Tuple[int, ...], dtype: torch.dtype, assertions: List[Callable[[torch.Tensor], bool]]):
        self._shape = shape # TODO: how do we make shape flexible? None for flexible dims? Ratios for related ones? new class?
        self._dtype = dtype
        self._assertions = assertions

    @property
    def shape(self) -> Tuple[int]:
        return self._shape

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def assertions(self) -> List[Callable[[torch.Tensor], bool]]:
        return self._assertions

    def __eq__(self, other) -> bool:
        if not isinstance(other, TensorInfo):
            raise NotImplementedError
        # TODO: How can we check for assertion equality? do they need to be their own classes?
        return (self._shape == other._shape
                and self._dtype == other._dtype)


class TorchComponentConfig(ABC):
    def __init__(self, has_triton: bool, use_triton: bool = True):
        self._has_triton = has_triton
        self._use_triton = use_triton if has_triton else False

    @property
    def has_triton(self) -> bool:
        return self._has_triton

    @property
    def use_triton(self) -> bool:
        return self._use_triton

    @abstractmethod
    def __eq__(self, other) -> bool:
        raise NotImplementedError


class TorchComponent(ABC, nn.Module):
    def __init__(self, config: TorchComponentConfig):
        super().__init__()
        self._config = config

    @property
    def config(self) -> TorchComponentConfig:
        return self._config

    @abstractmethod
    def input_info(self) -> Union[TensorInfo, Tuple[TensorInfo]]:
        # I recommend making this a function of self._config rather than hard-coding
        raise NotImplementedError

    @abstractmethod
    def output_info(self) -> Union[TensorInfo, Tuple[TensorInfo]]:
        raise NotImplementedError