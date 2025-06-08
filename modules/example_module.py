from typing import List, Union, Sequence

import torch
import torch.nn as nn

from modules.module_test_config import ModuleTestConfig


##################################################
############# PRIMARY PYTORCH MODULE #############
##################################################

class ExampleModule(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(size=(dim,)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight


##################################################
####### (OPTIONAL) CUSTOM KERNEL VERSION #########
##################################################

try:
    from .example_kernel import _KernelFn

    class ExampleKernelModule(ExampleModule):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return _KernelFn.apply(x, self.weight)
        
except (ImportError, ModuleNotFoundError):
    ExampleKernelModule = None
    

##################################################
#################### TESTING ####################
##################################################

def example_pytorch_output_validator(module: nn.Module, inputs: Sequence[torch.Tensor], output: Union[torch.Tensor, Sequence[torch.Tensor]]):
    # The output of a module might be a single tensor or a tuple of tensors.
    # We wrap single tensors in a tuple for consistent processing.
    if not isinstance(output, tuple):
        output = (output,)
    
    assert output[0].shape == inputs[0].shape
    assert output[0].shape[-1] == module.weight.shape[-1]
    
def example_kernel_run_filter(inputs: List[torch.Tensor]) -> bool:
    if 'cuda' not in str(inputs[0].device): return False
    # some more examples of what checks for a kernel would look like:
    #if inputs[0].dtype not in [torch.float16, torch.bfloat16]: return False
    #if inputs[0].shape[-1] % 32 != 0: return False
    return True

__test_config__ = ModuleTestConfig(
    module_class=ExampleModule,
    kernel_class=ExampleKernelModule,
    test_cases=[
        {
            'init_args': {'dim': dim},
            'input_args': lambda dev, d=dim, dt=dt: (torch.randn(1, d, device=dev, dtype=dt, requires_grad=True),),
            'output_validator': example_pytorch_output_validator,       # Mandatory
            'kernel_run_filter': example_kernel_run_filter,             # Optional
            'tolerances': {'atol': 1e-3, 'rtol': 1e-2},                 # Optional
            'excluded_devices': [],                     # Optional; of all available devices, which should not be tested
                                                        # eg. ['mps'] if mps doesn't support an operation you use
        }
        for dim in [32, 8192]
        for dt in [torch.float16, torch.float32, torch.bfloat16]
    ]
)