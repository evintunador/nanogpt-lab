import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune([triton.Config({"BLOCK_SIZE": 1024}, num_stages=3, num_warps=8,)], key=["n_elements", "loop_stride"])
@triton.jit
def elementwise_mul_forward(
    x_ptr, y_ptr,               
    output_ptr,                 
    n_elements,                 
    loop_stride,
    BLOCK_SIZE: tl.constexpr,   
):   
    program_id = tl.program_id(axis=0) 
    block_start_x = program_id * BLOCK_SIZE
    block_start_y = block_start_x % loop_stride
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
    x_ptr,
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
            elementwise_mul_backward_dLdy[ctx.grid](x, dLdy, dLdz, ctx.n_elements, ctx.loop_stride)
        return dLdx, dLdy