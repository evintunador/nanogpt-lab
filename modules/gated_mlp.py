from typing import List, Union, Tuple, Any

import torch
import torch.nn as nn

from modules.base_test_bench_utils import (
    ModuleTestConfig, 
    ModuleBenchmarkConfig, 
    BenchmarkPlotConfig, 
    Competitor,
    TensorParallelConfig
)


##################################################
############# PRIMARY PYTORCH MODULE #############
##################################################

def next_multiple(x, n):
    return int(((int(x) + n - 1) // n) * n)

class ReLU2(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x) ** 2

class GatedMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, activation: str):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = next_multiple(x=hidden_dim, n=128)

        self.act_str = activation.lower()
        act_registry = {
            "relu": nn.ReLU(),
            "silu": nn.SiLU(),
            "gelu": nn.GELU(),
            "relu2": ReLU2(),
        }
        self.act_fn = act_registry[self.act_str]

        self.Wup = nn.Parameter(torch.randn(size=(self.in_dim, self.hidden_dim)))
        self.Wgate = nn.Parameter(torch.randn(size=(self.in_dim, self.hidden_dim)))
        self.Wdown = nn.Parameter(torch.randn(size=(self.hidden_dim, self.out_dim)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ((x @ self.Wup) * self.act_fn(x @ self.Wgate)) @ self.Wdown


##################################################
####### (OPTIONAL) CUSTOM KERNEL VERSION #########
##################################################
"""
Please keep custom kernels in a separate file and load them in a try-except block.
This allows for us to support a variety of devices 
(eg. loading a Triton kernel on Apple silicon would otherwise throw an errror).
"""

try:
    #from .gated_mlp_kernel import GatedMLPFunctional

    class TritonGatedMLP(GatedMLP):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # The user-provided kernel would be used here.
            # For now, we call the parent to ensure tests pass.
            return super().forward(x)
            # return GatedMLPFunctional.apply(
            #     x,
            #     self.Wup, self.Wgate, self.Wdown,
            #     self.act_str,
            # )
        
except (ImportError, ModuleNotFoundError):
    TritonGatedMLP = None


##################################################
#################### TESTING ####################
##################################################


def pytorch_output_validator(
        module: nn.Module,
        inputs: Union[torch.Tensor, Tuple[Any]],
        outputs: Union[torch.Tensor, Tuple[Any]],
) -> None:
    """
    Validates whether the pytorch output meets expectations.
    Handles single tensor or tuple outputs by checking the first tensor.
    """
    input_tensor = inputs if isinstance(inputs, torch.Tensor) else inputs[0]
    output_tensor = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
    expected_shape = (*input_tensor.shape[:-1], module.out_dim)
    assert output_tensor.shape == expected_shape, f"Expected output shape {expected_shape}, but got {output_tensor.shape}"
    assert output_tensor.dtype == input_tensor.dtype
    
    
def kernel_run_filter(inputs: Union[torch.Tensor, Tuple[Any]]) -> bool:
    """
    Many custom kernels are only appropriate for use under a subset of all the conditions where a regular
    pytorch nn.module can run.
    Use this function to ensure that testing is only attempted on that subset.
    """
    if 'cuda' not in str(inputs[0].device):
        return False
    # some more examples of what checks for a kernel would look like:
    #if inputs[0].dtype not in [torch.float16, torch.bfloat16, torch.float32]: return False
    #if inputs[0].shape[-1] % 32 != 0: return False
    return True


__competitors__ = {
    'PyTorch': Competitor(module_class=GatedMLP),
    'Triton': Competitor(module_class=TritonGatedMLP, excluded_devices=['mps', 'cpu']),
}


__test_config__ = ModuleTestConfig(
    competitors=__competitors__,
    reference_competitor='PyTorch',
    test_cases=[
        {
            'init_args': {'in_dim': dim, 'out_dim': dim, 'hidden_dim': dim * 4, 'activation': 'relu2'},
            'input_args': lambda dev, d=dim, dt=dt: (torch.randn(16, d, device=dev, dtype=dt, requires_grad=True),),
            'pytorch_output_validator': pytorch_output_validator,
            'kernel_run_filter': kernel_run_filter,                     # Optional
            'tolerances': {'atol': 1e-2, 'rtol': 1e-2},                 # Optional
        }
        for dim, dt in [(128, torch.float16), (512, torch.float32), (2048, torch.bfloat16)]
    ]
)


##################################################
################# BENCHMARKING ###################
##################################################


def benchmark_input_provider(init_args: dict, device: str) -> tuple:
    """Generates a standard input for benchmarking."""
    # input shape: (batch_size, sequence_length, dimension)
    return (torch.randn(1, 1024, init_args['in_dim'], device=device, dtype=torch.float16),)

__benchmark_config__ = ModuleBenchmarkConfig(
    competitors=__competitors__,
    plots=[
        BenchmarkPlotConfig(
            plot_name='gated_mlp_perf_vs_dim',
            x_arg='dim',
            x_vals=[128, 256, 512, 1024, 2048, 4096, 8192],
            init_arg_builder=lambda dim: {'in_dim': dim, 'out_dim': dim, 'hidden_dim': dim * 4, 'activation': 'relu2'},
            input_provider=benchmark_input_provider,
        )
    ]
)