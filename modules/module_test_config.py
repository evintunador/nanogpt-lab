from typing import List, Dict, Optional, Type
from dataclasses import dataclass

import torch.nn as nn

@dataclass
class ModuleTestConfig:
    """
    A dataclass to hold the full testing configuration for a module.
    """
    # The pure PyTorch nn.Module class.
    module_class: Type[nn.Module]
    # A list of dictionaries, where each dict is a self-contained test case.
    test_cases: List[Dict]
    # The optional kernel-enabled nn.Module class, which must inherit from module_class.
    kernel_class: Optional[Type[nn.Module]] = None