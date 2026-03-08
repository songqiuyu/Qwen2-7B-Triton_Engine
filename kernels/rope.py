import torch
import triton
import triton.language as tl


@triton.jit
def _rope_forward_kernel(
    Q_ptr,              # (num_tokens, num_heads, head_dim)
    K_ptr,              # (num_tokens, num_kv_heads, head_dim)
    Cos_ptr,            # (max_seq_len, head_dim) 
    Sin_ptr,            # (max_seq_len, head_dim)
    Pos_ptr,            # (num_tokens,) position ids
    stride_q_token,
    stride_q_head,
    stride_k_token,
    stride_k_head,
    stride_cos_seq,
    num_tokens,
    num_heads,
    num_kv_heads,
    head_dim: tl.constexpr,
):
    """
    Fully vectorized RoPE kernel.
    Each program handles one (token, head) pair.
    Grid: (num_tokens, max(num_heads, num_kv_heads))
    
    No more serial for-loop over heads — each head is a separate program.
    """
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    
    if token_idx >= num_tokens:
        return
    
    # Load position id for this token
    pos = tl.load(Pos_ptr + token_idx)
    
    # Head dimension offsets (head_dim is power-of-2 like 128)
    d = tl.arange(0, head_dim)
    half_dim = head_dim // 2
    
    # Load cos/sin for this position
    cos = tl.load(Cos_ptr + pos * stride_cos_seq + d)
    sin = tl.load(Sin_ptr + pos * stride_cos_seq + d)
    
    # Compute rotate_half indices and negation mask
    # rotate_half: first half gets second half (negated), second half gets first half
    rotate_idx = tl.where(d < half_dim, d + half_dim, d - half_dim)
    neg_mask = tl.where(d < half_dim, -1.0, 1.0).to(tl.float32)
    
    # Process Q head (if within Q head range)
    if head_idx < num_heads:
        q_ptr = Q_ptr + token_idx * stride_q_token + head_idx * stride_q_head
        q = tl.load(q_ptr + d).to(tl.float32)
        q_rot = tl.load(q_ptr + rotate_idx).to(tl.float32) * neg_mask
        q_out = q * cos + q_rot * sin
        tl.store(q_ptr + d, q_out.to(tl.float16))
    
    # Process K head (if within KV head range)
    if head_idx < num_kv_heads:
        k_ptr = K_ptr + token_idx * stride_k_token + head_idx * stride_k_head
        k = tl.load(k_ptr + d).to(tl.float32)
        k_rot = tl.load(k_ptr + rotate_idx).to(tl.float32) * neg_mask
        k_out = k * cos + k_rot * sin
        tl.store(k_ptr + d, k_out.to(tl.float16))


def apply_rope_inplace(
    q: torch.Tensor, 
    k: torch.Tensor, 
    cos: torch.Tensor, 
    sin: torch.Tensor, 
    position_ids: torch.Tensor
):
    """
    Applies RoPE to q and k *in-place* with fully vectorized Triton kernel.
    
    q shape: (..., num_heads, head_dim)
    k shape: (..., num_kv_heads, head_dim)
    cos/sin shape: (max_seq_len, head_dim)
    position_ids shape: (...)
    """
    assert q.shape[-1] == k.shape[-1]
    head_dim = q.shape[-1]
    
    num_heads = q.shape[-2]
    num_kv_heads = k.shape[-2]
    
    q_flat = q.reshape(-1, num_heads, head_dim)
    k_flat = k.reshape(-1, num_kv_heads, head_dim)
    pos_flat = position_ids.reshape(-1)
    
    num_tokens = q_flat.shape[0]
    
    # Each program handles one (token, head) pair — fully parallel
    max_heads = max(num_heads, num_kv_heads)
    grid = (num_tokens, max_heads)
    
    _rope_forward_kernel[grid](
        q_flat, k_flat, cos, sin, pos_flat,
        q_flat.stride(0), q_flat.stride(1),
        k_flat.stride(0), k_flat.stride(1),
        cos.stride(0),
        num_tokens, num_heads, num_kv_heads,
        head_dim=head_dim,
    )
    
    return q, k


def precompute_freqs_cis(dim: int, end: int, theta: float = 1000000.0, device="cuda"):
    """Precompute cos/sin frequency tables for RoPE."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs).float()
    
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    
    # Duplicate to full head_dim: [cos_0..cos_{d/2-1}, cos_0..cos_{d/2-1}]
    cos = torch.cat([cos, cos], dim=-1).to(device).to(torch.float32)
    sin = torch.cat([sin, sin], dim=-1).to(device).to(torch.float32)
    return cos, sin
