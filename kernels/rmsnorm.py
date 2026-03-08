import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_forward_kernel(
    X_ptr,          # pointer to input (num_rows, hidden_size)
    W_ptr,          # pointer to weights (hidden_size,)  
    Y_ptr,          # pointer to output (num_rows, hidden_size)
    stride_x_row,
    stride_y_row,
    N,              # hidden_size
    eps,            # variance epsilon
    BLOCK_SIZE: tl.constexpr,
):
    """
    RMSNorm: y = x * rsqrt(mean(x^2) + eps) * w
    
    Each program processes exactly one row. BLOCK_SIZE must cover
    the full hidden dimension because we do a row-wise reduction.
    """
    row_idx = tl.program_id(0)

    x_ptr_start = X_ptr + row_idx * stride_x_row
    y_ptr_start = Y_ptr + row_idx * stride_y_row

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N

    # Load in FP32 for numerical stability
    x = tl.load(x_ptr_start + col_offsets, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)

    # Row-wise variance
    variance = tl.sum(x * x, axis=0) / N
    rrms = tl.math.rsqrt(variance + eps)
    y = x * rrms * w

    tl.store(y_ptr_start + col_offsets, y.to(tl.float16), mask=mask)


def rmsnorm_forward(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    RMSNorm forward using Triton.
    BLOCK_SIZE is set to next_power_of_2(hidden_size) since the kernel
    does a full-row reduction (variance), it must cover the entire row.
    """
    assert x.dtype in [torch.float16, torch.bfloat16, torch.float32]
    assert weight.shape[-1] == x.shape[-1]

    x_flatten = x.reshape(-1, x.shape[-1])
    n_rows, hidden_size = x_flatten.shape

    y_flatten = torch.empty_like(x_flatten)

    # BLOCK_SIZE must be >= hidden_size for the reduction
    BLOCK_SIZE = triton.next_power_of_2(hidden_size)
    
    # Optimal num_warps for RTX 4080 
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
    else:
        num_warps = 4

    grid = (n_rows,)

    _rmsnorm_forward_kernel[grid](
        x_flatten,
        weight,
        y_flatten,
        x_flatten.stride(0),
        y_flatten.stride(0),
        hidden_size,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    return y_flatten.view_as(x)
