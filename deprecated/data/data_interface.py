from abc import ABC, abstractmethod
from typing import List, Dict, Union

from datatypes import CustomDataType, CustomDataVal

class DataInterface(ABC):
    @abstractmethod
    def initialize_columns(self columns: Dict[str, CustomDataType]) -> None:
        raise NotImplementedError

    @abstractmethod
    def add_columns(self, columns: Dict[str, CustomDataVal]) -> None:
        raise NotImplementedError

    @abstractmethod
    def del_columns(self, columns: Union[str, List[str]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def build_columns(self, func, input_columns: Dict[str, CustomDataType], output_columns: Dict[str, CustomDataType]) -> None:
        # calls add_columns on output_columns
        # loops over rows, using func(input_vals)->output_vals to fill in output_columns
        raise NotImplementedError

    @abstractmethod
    def add_row(self, row: Dict[str, CustomDataVal]) -> None:
        raise NotImplementedError

    @abstractmethod
    def add_rows(self, rows: List[Dict[str, CustomDataVal]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def del_row(self, idx: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def del_rows(self, rows: List[int]) -> None:
        raise NotImplementedError

    @abstractmethod
    def to_bytes(self) -> bytes:
        raise NotImplementedError

    @abstractmethod
    def from_bytes(self, b: bytes) -> None:
        raise NotImplementedError