import torch
import triton
import triton.language as tl

@triton.jit
def _silu_mul_kernel(
    X_ptr,          # pointer to input X (which will have SiLU applied)
    Y_ptr,          # pointer to input Y (which will be multiplied)
    Out_ptr,        # pointer to output
    num_elements,   # total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Determine offset
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    # Load data
    x = tl.load(X_ptr + offsets, mask=mask)
    y = tl.load(Y_ptr + offsets, mask=mask)

    # Compute SiLU: x * sigmoid(x)
    # Using tl.sigmoid
    sigmoid_x = tl.sigmoid(x.to(tl.float32))
    silu_x = x.to(tl.float32) * sigmoid_x

    # Output = SiLU(X) * Y
    out = silu_x * y.to(tl.float32)

    # Store back
    tl.store(Out_ptr + offsets, out.to(x.dtype), mask=mask)

def silu_mul_forward(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
    """
    Computes: out = SiLU(x) * y
    Often used in SwiGLU MLP variants (like Llama, Qwen).
    """
    assert x.shape == y.shape
    assert x.is_contiguous() and y.is_contiguous()

    if out is None:
        out = torch.empty_like(x)
    else:
        assert out.shape == x.shape
        assert out.is_contiguous()

    num_elements = x.numel()
    
    # Triton configuration
    BLOCK_SIZE = 4096
    grid = (triton.cdiv(num_elements, BLOCK_SIZE),)

    _silu_mul_kernel[grid](
        x, y, out,
        num_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out

