import torch
import triton
import triton.language as tl

@triton.jit
def _rmsnorm_forward_kernel(
    X_ptr,          # pointer to input (batch_size * seq_len, hidden_size)
    W_ptr,          # pointer to weights (hidden_size,)
    Y_ptr,          # pointer to output (batch_size * seq_len, hidden_size)
    stride_x_row,   # stride.
    stride_y_row,   # stride.
    N,              # hidden_size
    eps,            # variance epsilon
    BLOCK_SIZE: tl.constexpr, # block size (e.g., 4096)
):
    # Map each program to a row
    row_idx = tl.program_id(0)

    # pointers to the starting row
    x_ptr_start = X_ptr + row_idx * stride_x_row
    y_ptr_start = Y_ptr + row_idx * stride_y_row

    # Create pointer offsets up to block_size
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N

    # Load row data and weights
    x = tl.load(x_ptr_start + col_offsets, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)

    # Compute variance
    x_var = tl.sum(x * x, axis=0) / N
    
    # Compute output
    rsqrt = tl.math.rsqrt(x_var + eps)
    y = x * rsqrt * w

    # Store result (cast to fp16 normally matching Y_ptr datatype)
    tl.store(y_ptr_start + col_offsets, y, mask=mask)

def rmsnorm_forward(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Args:
        x: Input tensor of shape (..., hidden_size)
        weight: Learnable weights of shape (hidden_size,)
        eps: Epsilon for numerical stability
    Returns:
        Output tensor of the same shape and dtype as input
    """
    assert x.dtype in [torch.float16, torch.bfloat16, torch.float32]
    assert weight.shape[-1] == x.shape[-1]

    # Flatten input to 2D
    x_flatten = x.view(-1, x.shape[-1])
    n_rows, hidden_size = x_flatten.shape

    # Pre-allocate output
    y_flatten = torch.empty_like(x_flatten)

    # Find the next power of 2 for block size
    BLOCK_SIZE = triton.next_power_of_2(hidden_size)
    # RTX 4080 has max 1024 threads per block natively, but Triton manages it up to limit. 
    # Qwen2 7B hidden_size is 3584 -> Next power of 2 is 4096.

    # 1 program ID per row
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
    )

    return y_flatten.view_as(x)

