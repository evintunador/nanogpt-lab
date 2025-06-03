from abc import abstractmethod, ABC
from typing import List, Tuple, Union, Dict

import numpy as np


class CustomType(ABC):
    def __init__(self, nullable: bool = False):
        self._nullable = nullable

    @property
    def nullable(self) -> bool:
        return self._nullable

    @abstractmethod
    def coerce(self, val: str) -> "CustomDataVal":
        raise NotImplementedError

    @abstractmethod
    def __eq__(self) -> bool:
        raise NotImplementedError


class TextType(CustomType):
    def __init__(self, nullable: bool = False):
        super().__init__(nullable=nullable)

    def coerce(self, val: str) -> "Text":
        try:
            val = str(val)
        except TypeError:
            raise NotImplementedError
        return Text(val=val, nullable=self._nullable)

    def __eq__(self, other) -> bool:
        if not isinstance(other, TextType):
            raise NotImplementedError
        return self._nullable == other._nullable

    @abstractmethod
    def val(self) -> str:
        raise NotImplementedError

class Text(TextType):
    def __init__(self, val: str, nullable: bool = False):
        super().__init__(nullable=nullable)
        if val is None:
            if not self._nullable:
                raise ValueError("val cannot be None")
        else:
            assert type(val) is str, f"value must be str but got {type(val)}"
        self._val = val

    @property
    def val(self) -> str:
        return self._val

    def __eq__(self, other) -> bool:
        if not isinstance(other, Text):
            raise NotImplementedError
        if self._nullable != other._nullable:
            return False
        if self._val is None and other._val is None:
            return True
        if self._val is None or other._val is None:
            return False
        return self._val == other._val


class TokenType(CustomType):
    def __init__(self, vocab_len: int, dtype: np.dtype, nullable: bool = False):
        super().__init__(nullable=nullable)
        assert isinstance(vocab_len, int), f"vocab_len {vocab_len} must be an integer"
        assert vocab_len > 0, f"vocab_len={vocab_len} must be > 0",
        acceptable_dtypes = (np.uint8, np.uint16, np.uint32)
        assert isinstance(dtype, acceptable_dtypes), f"dtype {dtype} must be a numpy unsigned int dtype but got {dtype}"
        if dtype == np.uint8:
            assert vocab_len <= 2**8, f"vocab_len={vocab_len} must be <= {2**8} for dtype {dtype}"
        elif dtype == np.uint16:
            assert vocab_len <= 2**16, f"vocab_len={vocab_len} must be <= {2**16} for dtype {dtype}"
        elif dtype == np.uint32:
            assert vocab_len <= 2**32, f"vocab_len={vocab_len} must be <= {2**32} for dtype {dtype}"
        self._vocab_len = vocab_len
        self._dtype = dtype

    @property
    def vocab_len(self) -> int:
        return self._vocab_len

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    def coerce(self, val: np.ndarray) -> "TokenSequence":
        if not isinstance(val, np.ndarray):
            try:
                val = np.array(val, dtype=self._dtype)
            except:
                raise NotImplementedError
        if val.ndim == 1:
            return TokenSequence(
                val=val, vocab_len=self._vocab_len, dtype=self._dtype, nullable=self._nullable
            )
        elif val.ndim == 2:
            return BatchedTokenSequences(
                val=val, vocab_len=self._vocab_len, dtype=self._dtype, nullable=self._nullable
            )
        else:
            raise NotImplementedError

        def __eq__(self, other) -> bool:
            if not isinstance(other, TokenType):
                raise NotImplementedError
            return (self._nullable == other._nullable
                    and self._vocab_len == other._vocab_len
                    and self._dtype == other._dtype)

class Tokens(TokenType):
    def __init__(selfself, val: np.ndarray, vocab_len: int, dtype: np.dtype, nullable: bool = False):
        super().__init__(vocab_len=vocab_len, dtype=dtype, nullable=nullable)
        assert val.dtype == self.dtype, f"token sequence dtype {val.dtype} does not match {self.dtype}"
        assert np.max(val) < self.vocab_len, f"token {np.max(val)} is out of range for vocab_len={self.vocab_len}"
        self._val = val

    @property
    def val(self) -> np.ndarray:
        return self._val

    def __eq__(self, other) -> bool:
        if not isinstance(other, TokenSequence):
            raise NotImplementedError
        if (self._nullable != other._nullable
            or self._vocab_len != other._vocab_len
            or self._dtype != other._dtype):
            return False
        return bool(np.all(self._val == other._val))


class TensorType(CustomType):
    def __init__(self, shape: Tuple[int], dtype: np.dtype, nullable: bool = False):
        super().__init__(nullable=nullable)
        assert isinstance(shape, tuple), f"shape {shape} must be a tuple"
        for dim in shape:
            assert isinstance(dim, int), f"shape {shape} must be a tuple of integers"
            assert dim > 0, f"shape {shape} must have positive dimensions"
        self._shape = shape
        self._dtype = dtype

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def shape(self) -> Tuple[int]:
        return self._shape

    def coerce(self, val: np.ndarray) -> "Tensor":
        if not isinstance(val, np.ndarray):
            try:
                val = np.array(val, dtype=self._dtype)
            except:
                raise NotImplementedError
        return Tensor(val=val, dtype=self._dtype, shape=self._shape, nullable=self._nullable)

    def __eq__(self, other) -> bool:
        if not isinstance(other, TensorType):
            raise NotImplementedError
        return (self._nullable == other._nullable
                and self._shape == other._shape
                and self._dtype == other._dtype)

class Tensor(TensorType):
    def __init__(self, val: np.ndarray, dtype: np.dtype, shape: Tuple[int], nullable: bool = False):
        super().__init__(shape=shape, dtype=dtype, nullable=nullable)
        assert val.dtype == self._dtype, f"tensor dtype {val.dtype} does not match {self.dtype}"
        if self.shape is not None:
            assert val.shape == self.shape, f"tensor shape {val.shape} does not match {self.shape}"
        self._val = val

    @property
    def val(self) -> np.ndarray:
        return self._val

    def __eq__(self, other) -> bool:
        if not isinstance(other, Tensor):
            raise NotImplementedError
        if (self._nullable != other._nullable
            or self._shape != other._shape
            or self._dtype != other._dtype):
            return False
        return bool(np.all(self._val == other._val))


CustomDataType = Union[TextType, TokenType, TensorType]
CustomDataVal = Union[Text, Tokens, Tensor]