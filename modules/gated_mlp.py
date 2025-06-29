from typing import List, Union, Tuple, Any
import math

import torch
import torch.nn as nn

from modules.base_test_bench_utils import (
    ModuleTestConfig, 
    BenchmarkConfig, 
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
        return torch.relu(x).clamp(max=255.0).square()

class GatedMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, activation: str, 
                 dtype: torch.dtype = torch.float32, device: str = 'cpu'):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = next_multiple(x=hidden_dim, n=128)

        self.act_str = activation.lower()
        act_registry = {
            "relu": nn.ReLU(),
            "silu": nn.SiLU(),
            "relu2": ReLU2(),
        }
        self.act_fn = act_registry[self.act_str]

        self.Wup = nn.Parameter(torch.empty(size=(self.in_dim, self.hidden_dim), dtype=dtype, device=device))
        self.Wgate = nn.Parameter(torch.empty(size=(self.in_dim, self.hidden_dim), dtype=dtype, device=device))
        self.Wdown = nn.Parameter(torch.empty(size=(self.hidden_dim, self.out_dim), dtype=dtype, device=device))
        
        nn.init.kaiming_uniform_(self.Wup,  a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.Wgate, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.Wdown, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ((x @ self.Wup) * self.act_fn(x @ self.Wgate)) @ self.Wdown


##################################################
# EXAMPLE ALTERNATIVE IMPLEMENTATION FOR TESTING #
##################################################

@torch.compile
def fwd(inp, w_up, w_gate, w_down, act_fn):
    return ((inp @ w_up) * act_fn(inp @ w_gate)) @ w_down

class CompiledGatedMLP(GatedMLP):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fwd(x, self.Wup, self.Wgate, self.Wdown, self.act_fn)
    

def compiled_run_filter(inputs: Union[torch.Tensor, Tuple[Any]]) -> bool:
    """
    Many custom modules are only appropriate for use under a subset of all the conditions where a regular pytorch nn.module can run.
    Use this function to ensure that testing is only attempted on that subset.
    Here, for example, our CompiledGatedMLP should only be run on a GPU since it uses torch.compile.
    """
    if 'cpu' in str(inputs[0].device):
        return False
    return True


##################################################
######## EXAMPLE CUSTOM KERNEL VERSION ##########
##################################################
"""
Please keep custom kernels in a separate file and load them in a try-except block.
This allows for us to support a variety of devices 
(eg. loading a Triton kernel on Apple silicon would otherwise throw an error).
"""

try:
    from modules.gated_mlp_kernel import _GatedMLPKernelFn

    class TritonGatedMLP(GatedMLP):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return _GatedMLPKernelFn.apply(
                x,
                self.Wup, self.Wgate, self.Wdown,
                self.act_str,
            )

except (ImportError, ModuleNotFoundError):
    TritonGatedMLP = None


def triton_run_filter(inputs: Union[torch.Tensor, Tuple[Any]]) -> bool:
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


##################################################
#################### TESTING ####################
##################################################


def output_validator(
        module: nn.Module,
        inputs: Tuple[Any],
        outputs: Tuple[Any],
) -> None:
    """
    Validates whether the base module output meets expectations.
    Testing framework always passes in tuples even if there's only one input/output tensor
    """
    input_tensor = inputs[0] 
    output_tensor = outputs[0]
    expected_shape = (*input_tensor.shape[:-1], module.out_dim)
    assert output_tensor.shape == expected_shape, f"Expected output shape {expected_shape}, but got {output_tensor.shape}"
    assert output_tensor.dtype == input_tensor.dtype
    

__competitors__ = {
    'GatedMLP': Competitor(module_class=GatedMLP),
    'CompiledGatedMLP': Competitor(module_class=CompiledGatedMLP, run_filter=compiled_run_filter),
    'TritonGatedMLP': Competitor(module_class=TritonGatedMLP, run_filter=triton_run_filter),
}


__test_config__ = ModuleTestConfig(
    competitors=__competitors__,
    reference_competitor='GatedMLP',
    test_cases=[
        {
            'init_args': {'in_dim': dim, 'out_dim': dim, 'hidden_dim': dim * 4, 'activation': act, 'dtype': dt},
            'input_args': lambda dev, d=dim, dt=dt: (torch.randn(16, d, device=dev, dtype=dt, requires_grad=True),),
            'output_validator': output_validator,
            'tolerances': {'atol': 5e-3, 'rtol': 5e-3},          # Optional
            'case_descriptor': f'dim={dim}_dt={dt}_act={act}',
        }
        for dim, dt in [(128, torch.float16), (512, torch.float32), (2048, torch.bfloat16)]
        for act in ['relu', 'relu2', 'silu']
    ]
)


##################################################
################# BENCHMARKING ###################
##################################################


def benchmark_input_provider(init_args: dict, device: str) -> tuple:
    """Generates a standard input for benchmarking."""
    # input shape: (batch_size, sequence_length, dimension)
    dtype = init_args.get('dtype', torch.float32)
    return (torch.randn(1, 1, init_args['in_dim'], device=device, dtype=dtype),)

__benchmark_config__ = BenchmarkConfig(
    module_name='GatedMLP',
    competitors=__competitors__,
    parameter_space={
        'dim': [32, 64, 128, 512, 1024, 2048, 4096],
        'activation': ['relu', 'silu', 'relu2'],
        'dtype': [torch.float16, torch.bfloat16, torch.float32],
    },
    init_arg_builder=lambda params: {
        'in_dim': params['dim'],
        'out_dim': params['dim'],
        'hidden_dim': params['dim'] * 4,
        'activation': params['activation'],
        'dtype': params['dtype']
    },
    input_provider=benchmark_input_provider,
)