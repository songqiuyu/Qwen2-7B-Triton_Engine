import torch
import triton
import triton.language as tl



@triton.jit
def _rope_forward_kernel(
    Q_ptr,              # (batch_size * seq_len, num_heads, head_dim)
    K_ptr,              # (batch_size * seq_len, num_kv_heads, head_dim)
    Cos_ptr,            # (seq_len, head_dim) / actually (seq_len, head_dim // 2) depending on convention
    Sin_ptr,            # (seq_len, head_dim)
    Pos_ptr,            # pointer to 1D position ids array (batch_size * seq_len,)
    stride_q_token,     # stride
    stride_q_head,      # stride
    stride_k_token,     # stride
    stride_k_head,      # stride
    stride_cos_seq,
    num_seqs_x_len,     # batch_size * seq_len
    num_heads,
    num_kv_heads,
    head_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, # Number of heads to process per program
):
    token_idx = tl.program_id(0) # each program handles one token (row)
    if token_idx >= num_seqs_x_len:
        return
        
    head_start = tl.program_id(1) * BLOCK_SIZE
    
    # Load the position id for this token
    pos = tl.load(Pos_ptr + token_idx)
    
    # Head Dimension offsets (assuming head_dim is small and power of 2 like 128)
    head_offsets = tl.arange(0, head_dim)
    
    # We load Cos and Sin for the specific position
    cos = tl.load(Cos_ptr + pos * stride_cos_seq + head_offsets)
    sin = tl.load(Sin_ptr + pos * stride_cos_seq + head_offsets)
    
    # To apply RoPE, we rotate half arrays. e.g. x1, x2 -> x1 * cos - x2 * sin, x2 * cos + x1 * sin
    # The Qwen2 RoPE typically interleaves elements, or splits by half.
    # Standard split-by-half layout for RoPE: [x0, x1, ... x_{d/2-1}, x_{d/2}, x_{d/2+1}, ...]
    # For simplicity, we assume `cos` and `sin` are expanded to `head_dim` like `[cos0, cos1... cos_{d/2-1}, cos0, cos1... cos_{d/2-1}]`.
    # And we just negate the first half and swap for the second term.
    half_dim = head_dim // 2
    rotate_half_indices = tl.where(head_offsets < half_dim, head_offsets + half_dim, head_offsets - half_dim)
    # Mask for negating: first half is negative, second half is positive
    # Actually wait: standard RoPE is:
    # rotated_x[:d/2] = -x[d/2:]
    # rotated_x[d/2:] = x[:d/2]
    # So if x_rot_idx < d/2, we load x[idx + d/2] and negate it.
    
    # Let's process Q
    head_block_offsets = head_start + tl.arange(0, BLOCK_SIZE)
    q_mask = head_block_offsets < num_heads
    
    for h in range(BLOCK_SIZE):
        h_idx = head_start + h
        if h_idx < num_heads:
            q_ptr_h = Q_ptr + token_idx * stride_q_token + h_idx * stride_q_head
            q = tl.load(q_ptr_h + head_offsets)
            
            # Apply rotation
            q_rotated = tl.load(q_ptr_h + rotate_half_indices)
            # Create negative mask for first half
            neg_mask = tl.where(head_offsets < half_dim, -1.0, 1.0)
            q_rotated = q_rotated * neg_mask
            
            q_out = q * cos + q_rotated * sin
            tl.store(q_ptr_h + head_offsets, q_out)
            
    # Process K (only if we are on valid KV heads)
    for h in range(BLOCK_SIZE):
        h_idx = head_start + h
        if h_idx < num_kv_heads:
            k_ptr_h = K_ptr + token_idx * stride_k_token + h_idx * stride_k_head
            k = tl.load(k_ptr_h + head_offsets)
            
            k_rotated = tl.load(k_ptr_h + rotate_half_indices)
            neg_mask = tl.where(head_offsets < half_dim, -1.0, 1.0)
            k_rotated = k_rotated * neg_mask
            
            k_out = k * cos + k_rotated * sin
            tl.store(k_ptr_h + head_offsets, k_out)

def apply_rope_inplace(
    q: torch.Tensor, 
    k: torch.Tensor, 
    cos: torch.Tensor, 
    sin: torch.Tensor, 
    position_ids: torch.Tensor
):
    """
    Applies RoPE to q and k *in-place*.
    q shape: (..., num_heads, head_dim)
    k shape: (..., num_kv_heads, head_dim)
    cos/sin shape: (max_seq_len, head_dim) -- Precomputed
    position_ids shape: (...)
    """
    assert q.shape[-1] == k.shape[-1]
    head_dim = q.shape[-1]
    
    original_q_shape = q.shape
    num_heads = q.shape[-2]
    num_kv_heads = k.shape[-2]
    
    q_flat = q.view(-1, num_heads, head_dim)
    k_flat = k.view(-1, num_kv_heads, head_dim)
    pos_flat = position_ids.view(-1)
    
    num_tokens = q_flat.shape[0]
    
    BLOCK_SIZE_HEADS = 4
    grid = (num_tokens, triton.cdiv(num_heads, BLOCK_SIZE_HEADS)) # 头内进行并行，一共HEADS个头
    
    _rope_forward_kernel[grid](
        q_flat, k_flat, cos, sin, pos_flat,
        q_flat.stride(0), q_flat.stride(1),
        k_flat.stride(0), k_flat.stride(1),
        cos.stride(0),
        num_tokens, num_heads, num_kv_heads,
        head_dim=head_dim,
        BLOCK_SIZE=BLOCK_SIZE_HEADS
    )
    
    return q, k

# Helper to precompute cos/sin
def precompute_freqs_cis(dim: int, end: int, theta: float = 1000000.0, device="cuda"):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32) 
    freqs = torch.outer(t, freqs).float()  # type: ignore
    
    # We duplicate to format [cos_0, cos_1 ... cos_{d/2-1}, cos_0 ... cos_{d/2-1}]
    # matching the split half format used in kernel.
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    
    cos = torch.cat([cos, cos], dim=-1).to(device)
    sin = torch.cat([sin, sin], dim=-1).to(device)
    return cos, sin

