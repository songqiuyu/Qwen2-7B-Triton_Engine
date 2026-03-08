import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16),
    ],
    key=['num_elements'],
)
@triton.jit
def _silu_mul_kernel(
    X_ptr,          # pointer to gate input (SiLU applied)
    Y_ptr,          # pointer to up input (multiplied)
    Out_ptr,        # pointer to output
    num_elements,   # total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    # Load gate and up values
    x = tl.load(X_ptr + offsets, mask=mask).to(tl.float32)
    y = tl.load(Y_ptr + offsets, mask=mask).to(tl.float32)

    # Fused SiLU(x) * y = (x * sigmoid(x)) * y
    sigmoid_x = tl.sigmoid(x)
    out = x * sigmoid_x * y

    tl.store(Out_ptr + offsets, out.to(tl.float16), mask=mask)


def silu_mul_forward(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
    """
    Computes: out = SiLU(x) * y  (Fused SwiGLU activation)
    Autotuned for optimal BLOCK_SIZE on RTX 4080.
    """
    assert x.shape == y.shape
    assert x.is_contiguous() and y.is_contiguous()

    if out is None:
        out = torch.empty_like(x)
    else:
        assert out.shape == x.shape
        assert out.is_contiguous()

    num_elements = x.numel()

    grid = lambda META: (triton.cdiv(num_elements, META['BLOCK_SIZE']),)

    _silu_mul_kernel[grid](
        x, y, out,
        num_elements,
    )

    return out
