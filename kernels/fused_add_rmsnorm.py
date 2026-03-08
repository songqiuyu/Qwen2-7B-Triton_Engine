import torch
import triton
import triton.language as tl


@triton.jit
def _fused_add_rmsnorm_kernel(
    Residual_ptr,   # (num_rows, hidden_size) — updated in-place
    X_ptr,          # (num_rows, hidden_size) — input to add
    W_ptr,          # (hidden_size,) — RMSNorm weights
    Y_ptr,          # (num_rows, hidden_size) — normalized output
    stride_r_row,
    stride_x_row,
    stride_y_row,
    N,              # hidden_size
    eps,            # RMSNorm epsilon
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused: residual = residual + x; y = RMSNorm(residual)
    
    BLOCK_SIZE must cover the full hidden dimension because we do
    a row-wise reduction (variance computation).
    """
    row_idx = tl.program_id(0)
    
    r_start = Residual_ptr + row_idx * stride_r_row
    x_start = X_ptr + row_idx * stride_x_row
    y_start = Y_ptr + row_idx * stride_y_row
    
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N
    
    # Load residual, input, and weights
    r = tl.load(r_start + col_offsets, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_start + col_offsets, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Fused add
    h = r + x
    
    # Store updated residual in-place
    tl.store(r_start + col_offsets, h.to(tl.float16), mask=mask)
    
    # RMSNorm on the fused result
    variance = tl.sum(h * h, axis=0) / N
    rrms = tl.math.rsqrt(variance + eps)
    y = h * rrms * w
    
    tl.store(y_start + col_offsets, y.to(tl.float16), mask=mask)


def fused_add_rmsnorm_forward(
    residual: torch.Tensor,
    x: torch.Tensor, 
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Fused residual addition + RMSNorm.
    
    residual is updated IN-PLACE: residual = residual + x
    Returns: RMSNorm(residual) with the updated residual
    
    BLOCK_SIZE is set to cover the full hidden dimension.
    """
    assert x.shape == residual.shape
    assert weight.shape[-1] == x.shape[-1]
    
    r_flat = residual.reshape(-1, residual.shape[-1])
    x_flat = x.reshape(-1, x.shape[-1])
    n_rows, hidden_size = r_flat.shape
    
    y_flat = torch.empty_like(r_flat)
    
    BLOCK_SIZE = triton.next_power_of_2(hidden_size)
    
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
    else:
        num_warps = 4
    
    grid = (n_rows,)
    
    _fused_add_rmsnorm_kernel[grid](
        r_flat, x_flat, weight, y_flat,
        r_flat.stride(0),
        x_flat.stride(0),
        y_flat.stride(0),
        hidden_size,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    
    return y_flat.view_as(residual)
