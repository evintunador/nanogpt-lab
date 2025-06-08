from typing import List, Union, Sequence

import torch
import torch.nn as nn
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    # This allows the file to be imported and inspected on systems without Triton.
    TRITON_AVAILABLE = False

from modules.bulk_testing_utils import ComponentTestConfig


##################################################
############# PRIMARY PYTORCH MODULE #############
##################################################
"""
Nothing interesting to see here, just a regular old pytorch module.
This bulk testing framework does not require you do anything special; integration
comes later and is completely optional
"""

class ExampleModule(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(size=(dim,)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight


##################################################
####### (OPTIONAL) CUSTOM KERNEL VERSION #########
##################################################
"""
You may or may not want to implement a custom GPU kernel to speed up the runtime and/or decrease the VRAM utilization. 

- If you DO NOT, that is perfectly alright. 
Just skip this section and the corresponding parts of the next section when writing your own modules.

- If you DO, then there's only one special thing you have to do here beyond what you'd normally do when writing a custom kernel.
When setting up the nn.Module that calls the kernel, instead of inheriting directly from nn.Module and setting up its own __init__, you should directly inherit from your pytorch module version and let it handle the entire config. 
We do it this way in order to ensure that the configuration and state dict of both modules exactly match during testing later.
"""
if TRITON_AVAILABLE:
    
    @triton.autotune([triton.Config({"BLOCK_SIZE": 1024}, num_stages=3, num_warps=8,)], key=["n_elements", "loop_stride"])
    @triton.jit
    def elementwise_mul_forward(
        x_ptr, y_ptr,               
        output_ptr,                 
        n_elements,                 
        loop_stride,                
        OP: tl.constexpr,           
        BLOCK_SIZE: tl.constexpr,   
    ):   
        program_id = tl.program_id(axis=0) 
        block_start_x = program_id * BLOCK_SIZE
        block_start_y = block_start_x % loop_stride # the looping is how we handle broadcasting
        offsets_x = block_start_x + tl.arange(0, BLOCK_SIZE)
        offsets_y = (block_start_y + tl.arange(0, BLOCK_SIZE)) % loop_stride
        mask_x = offsets_x < n_elements
        mask_y = offsets_y < loop_stride
        x = tl.load(x_ptr + offsets_x, mask = mask_x)
        y = tl.load(y_ptr + offsets_y, mask = mask_y)
        out = x * y
        tl.store(output_ptr + offsets_x, out, mask = mask_x)


    @triton.autotune([triton.Config({"BLOCK_SIZE": 1024}, num_stages=3, num_warps=8,)], key=["n_elements", "loop_stride"])
    @triton.jit
    def elementwise_mul_backward_dLdx(
        y_ptr,               
        dx_ptr,             
        do_ptr,                     
        n_elements,                 
        loop_stride,                
        OP: tl.constexpr,           
        BLOCK_SIZE: tl.constexpr,   
    ):
        pid = tl.program_id(axis=0)
        block_start_x = pid * BLOCK_SIZE
        offsets_x = block_start_x + tl.arange(0, BLOCK_SIZE)
        mask_x = offsets_x < n_elements
        do = tl.load(do_ptr + offsets_x, mask=mask_x)
        dx = tl.load(dx_ptr + offsets_x, mask=mask_x)
        block_start_y = block_start_x % loop_stride 
        offsets_y = (block_start_y + tl.arange(0, BLOCK_SIZE)) % loop_stride
        mask_y = offsets_y < loop_stride
        y_val = tl.load(y_ptr + offsets_y, mask=mask_y)
        dx += do * y_val
        tl.store(dx_ptr + offsets_x, dx, mask=mask_x)
        

    @triton.autotune([triton.Config({"BLOCK_SIZE": 1024}, num_stages=3, num_warps=8,)], key=["n_elements", "loop_stride"])
    @triton.jit
    def elementwise_mul_backward_dLdy(
        x_ptr, y_ptr,              
        dLdy_ptr,            
        dLdz_ptr,                    
        n_elements,                
        loop_stride,               
        BLOCK_SIZE: tl.constexpr,  
    ):
        pid = tl.program_id(axis=0)
        block_start_x = pid * BLOCK_SIZE
        block_start_y = block_start_x % loop_stride 
        offsets_x = block_start_x + tl.arange(0, BLOCK_SIZE)
        offsets_y = (block_start_y + tl.arange(0, BLOCK_SIZE)) % loop_stride
        mask_x = offsets_x < n_elements
        mask_y = offsets_y < loop_stride
        dLdz = tl.load(dLdz_ptr + offsets_x, mask=mask_x)
        x_val = tl.load(x_ptr + offsets_x, mask=mask_x)
        tl.atomic_add(dLdy_ptr + offsets_y, x_val * dLdz, mask=mask_y)


    class _KernelFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, y):
            assert x.device == y.device, \
                f'tensors must be on same device but got self.device: {x.device}, other.device: {y.device}'
            assert x.data.is_contiguous() and y.data.is_contiguous()
            try:
                torch.broadcast_shapes(x.shape, y.shape)
            except RuntimeError:
                raise ValueError(f"Tensors with shapes {x.shape} and {y.shape} are not broadcast compatible.")

            n_elements = x.numel()
            loop_stride = y.numel()
            assert n_elements >= loop_stride, "for multiplication, the first input must have more than or as many entries as the second"
            assert n_elements % loop_stride == 0, "the number of entries in the first input must be a multiple of the second"
            
            z = torch.empty_like(x)
            grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
            elementwise_mul_forward[grid](x, y, z, n_elements, loop_stride)
            
            ctx.save_for_backward(x, y, z)
            ctx.n_elements = n_elements
            ctx.loop_stride = loop_stride
            ctx.grid = grid

            return z
        
        @staticmethod
        def backward(ctx, dLdz):
            x, y, z = ctx.saved_tensors
            dLdx = torch.empty_like(x) if x.requires_grad else None
            dLdy = torch.empty_like(y) if y.requires_grad else None
            if x.requires_grad:
                elementwise_mul_backward_dLdx[ctx.grid](y, dLdx, dLdz, ctx.n_elements, ctx.loop_stride)
            if y.requires_grad:
                elementwise_mul_backward_dLdy[ctx.grid](x, y, dLdy, dLdz, ctx.n_elements, ctx.loop_stride)
            return dLdx, dLdy


    class ExampleCustomKernelModule(ExampleModule):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return _KernelFn.apply(x, self.weight)
else:
    # If Triton is not available, the kernel module class is None.
    ExampleCustomKernelModule = None
    

##################################################
#################### TESTING ####################
##################################################
"""
Here is where we inform our bulk testing framework how to integrate with the modules we've defined above.
If you don't want to integrate with the framework when writing your own modules that's cool, just don't do any of this and your module will be skipped/ignored. 
Please write your own tests rather than not testing at all though.

Instructions:
1. Define your output validator function. 
The purpose here is to ensure that a given input to your pytorch nn.module successfully results in ouput with whatever characteristics you expect. 
Usually this just means making simple shape and dtype assertions, but you might want some more complicated criteria such as ensuring that the output to a LayerNorm module has entries with a mean of zero.
2. IF you wrote a custom kernel, define your kernel filter function.
Many times custom kernels have restrictions on what devices, datatypes, tensor sizes, etc that they can run on; this function allows you to specify those restrictions.
For a specific set of inputs, this function should returns True if the kernel is compatible (and therefore should be tested) and False if incompatible.
3. Define the dunder variable '__test_config__' (avoid typos!) as an instance of the class modules.bulk_testing_utils.ComponentTestConfig
Args:
    module_class: set equal to the pytorch module you've written
    kernel_module_class: set equal to the kernel class you've written if you have, and if not then set to None
    test_cases: A list of dictionaries, each of which describes a test case. Its keys are:
        init_args: values for args during the initialization of the module
        input_args: values for args when calling .forward() on the module
            - should be a lambda with input device type since the main testing loop will only test available devices
        output_validator: set equal to your output validator function
*TODO: come back; realized I should probably finish code before writing documentation
"""

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

__test_config__ = ComponentTestConfig(
    module_class=ExampleModule,
    kernel_module_class=ExampleCustomKernelModule,
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