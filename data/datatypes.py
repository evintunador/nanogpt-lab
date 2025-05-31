from abc import abstractmethod, ABC
from typing import List, Tuple, Union, Dict

import numpy as np


class ColumnType(ABC):
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

class TextType(ColumnType):
    def __init__(self, nullable: bool = False):
        super().__init__(nullable=nullable)

    def coerce(self, val: str) -> "Text":
        # TODO -> BatchedText
        try:
            val = str(val)
        except TypeError:
            raise NotImplementedError
        return Text(val=val)

    def __eq__(self, other) -> bool:
        if not isinstance(other, TextType):
            raise NotImplementedError
        return self._nullable == other._nullable:

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
    def val(self):
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

class BatchedText(TextType):
    def __init__(self, val: List[Text], nullable: bool = False):
        super().__init__(nullable=nullable)
        if val is None:
            if not self._nullable:
                raise ValueError("val cannot be None")
        else:
            if not isinstance(val, list):
                raise TypeError(f"val must be a list but got {type(val)}")
            if not all(isinstance(x, Text) for x in val):
                raise TypeError(f"text sequences in batch must be of type Text but got {[type(x) for x in val]}")
        self._val = val

    @property
    def val(self) -> List[Text]:
        return self._val


class TokensType(ABC):
    def __init__(self, vocab_len: int, dtype: np.dtype):
        assert isinstance(vocab_len, int), f"vocab_len {vocab_len} must be an integer"
        assert vocab_len > 0, f"vocab_len={vocab_len} must be > 0",
        acceptable_dtypes = (np.uint8, np.uint16, np.uint32)
        assert isinstance(dtype, acceptable_dtypes), f"dtype {dtype} must be a numpy unsighed int dtype but got {dtype}"
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

    def coerce(self, val: np.ndarray) -> Union["TokenSequence", "BatchedTokenSequences"]:
        if not isinstance(val, np.ndarray):
            try:
                val = np.array(val, dtype=self._dtype)
            except:
                raise NotImplementedError
        if val.ndim == 1:
            return TokenSequence(val=val, vocab_len=self._vocab_len, dtype=self._dtype)
        elif val.ndim == 2:
            return BatchedTokenSequences(val=val, vocab_len=self._vocab_len, dtype=self._dtype)
        else:
            raise NotImplementedError

    @abstractmethod
    def val(self):
        raise NotImplementedError

class TokenSequence(TokensType):
    def __init__(self, val: np.ndarray, vocab_len: int, dtype: np.dtype):
        super().__init__(vocab_len=vocab_len, dtype=dtype)
        assert val.ndim == 1, f"token sequence must have 1 dimension but got ndim={val.ndim}"
        assert val.dtype == self.dtype, f"token sequence dtype {val.dtype} does not match {self.dtype}"
        assert np.all(val >= 0), f"tokens must be non-negative but got minimum={np.min(val)}"
        assert np.all(val <= self.vocab_len), \
            f"tokens must be less than {self.vocab_len} but got maximum={np.max(val)}"
        self._val = val

    @property
    def val(self) -> np.ndarray:
        return self._val

class BatchedTokenSequences(TokensType):
    def __init__(self, val: np.ndarray, vocab_len: int, dtype: np.dtype):
        super().__init__(vocab_len=vocab_len, dtype=dtype)
        assert val.ndim == 2, f"batched token sequence must have 2 dimensions but got ndim={val.ndim}"
        assert val.dtype == self.dtype, f"token sequence dtype {val.dtype} does not match {self.dtype}"
        assert np.all(val >= 0), f"tokens must be non-negative but got minimum={np.min(val)}"
        assert np.all(val <= self.vocab_len), \
            f"tokens must be less than {self.vocab_len} but got maximum={np.max(val)}"
        self._val = val

    @property
    def val(self) -> List[List[int]]:
        return self._val


class TensorType(ABC):
    def __init__(self, dtype: np.dtype, shape: Tuple[int] = None):
        assert isinstance(dtype, np.dtype), f"dtype must be np.dtype but got {type(dtype)}"
        self._dtype = dtype
        self._shape = shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def shape(self) -> Tuple[int]:
        return self._shape

    def coerce(self, val: np.ndarray) -> "TensorType":
        if not isinstance(val, np.ndarray):
            try:
                val = np.array(val, dtype=self._dtype)
            except:
                raise NotImplementedError
        if val.shape != self.shape:
            try:
                val = np.broadcast_to(val, self._shape)
            except:
                raise NotImplementedError
        return Tensor(val=val, dtype=self._dtype, shape=self._shape)

    @abstractmethod
    def val(self):
        raise NotImplementedError

class Tensor(TensorType):
    def __init__(self, val: np.ndarray, dtype: np.dtype, shape: Tuple[int]):
        super().__init__(dtype=dtype, shape=shape)
        assert isinstance(val, np.ndarray), f"val must be np.ndarray but got {type(val)}"
        assert val.dtype == self.dtype, f"dtype must be {self.dtype} but got {val.dtype}"
        if self.shape is not None:
            assert val.shape == self.shape, f"val shape must be {self.shape} but got {val.shape}"
        self._val = val

    @property
    def val(self) -> np.ndarray:
        return self._val


class ClassType(ABC):
    def __init__(self, classes: List[str]):
        assert isinstance(classes, list), f"classes must be a list of strings but got {classes}"
        assert all(isinstance(c, str) for c in classes), f"classes must be a list of strings but got {classes}"
        self._classes = classes

    @property
    def classes(self) -> List[str]:
        return self._classes

    def coerce(self, val: str) -> "Class":
        # TODO: if dict w/ keys in categories & vals summing to 1 then return ClassProbabilitiesType
        if not isinstance(val, str):
            try:
                val = str(val)
            except:
                raise NotImplementedError
        if val not in self.classes:
            raise NotImplementedError
        return Class(val=val, classes=self._classes)

    @abstractmethod
    def val(self):
        raise NotImplementedError

class Class(ClassType):
    def __init__(self, val: str, classes: List[str]):
        super().__init__(classes=classes)
        assert isinstance(val, str), f"val must be str but got {type(val)}"
        assert val in self._classes, f"val={val} not in {self._classes}"
        self._val = val

    @property
    def val(self) -> str:
        return self._val

class ClassProbabilities(ClassType):
    def __init__(self, val: Dict[str, float], classes: List[str]):
        super().__init__(classes=classes)
        assert isinstance(val, dict), f"val must be dict but got {type(val)}"
        assert all(c in self._classes for c in val.keys()), \
            f"val.keys()={val.keys()} contains classes not in {self._classes}"
        assert all(c in val.keys() for c in self._classes), \
            f"val.keys()={val.keys()} missing classes in {self._classes}"
        assert all(isinstance(p, float) for p in val.values()), \
            f"val.values()={val.values()} contains values that are not of type float"
        assert all(0 <= p <= 1 for p in val.values()), \
            f"val.values()={val.values()} contains values that are not in [0, 1]"
        assert sum(val.values()) == 1, f"probabilities should sum to 1, but got {sum(val.values())}"
        self._val = val

    @property
    def val(self) -> Dict[str, float]:
        return self._val


class ReferenceType(ABC):
    def __init__(self, root_dir: str):
        self._root_dir = root_dir

    @property
    def root_dir(self) -> str:
        return self._root_dir

    def coerce(self, val: str) -> "ReferenceType":
        raise NotImplementedError

    @abstractmethod
    def val(self):
        raise NotImplementedError

class LocalReference(ReferenceType):
    def __init__(self, val: str, root_dir: str):
        super().__init__(root_dir=root_dir)
        self._val = val

        @property
        def val(self) -> str:
            return self._val
# TODO -> do i specify filetype for references? do they load themselves? Should I have a BinReference type?


type CustomDataType = Union[TextType, TensorType, ClassType, ReferenceType]
type CustomDataVal = Union[Text, BatchedText, TextNode, Tensor, Class, ClassProbabilities, LocalReference]