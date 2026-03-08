import torch
import triton
import triton.language as tl
import math


@triton.jit
def _flash_attn_fwd_kernel(
    Q, K, V, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    num_heads: tl.constexpr,
    seq_len_q,
    seq_len_kv,
    scale,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
):
    """
    FlashAttention-2 forward kernel.
    
    Each program processes one Q-block for one (batch, head) pair.
    Uses online softmax to avoid materializing the full attention matrix.
    
    Grid: (cdiv(seq_len_q, BLOCK_M), batch * num_heads)
    """
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    off_z = off_hz // num_heads   # batch index
    off_h = off_hz % num_heads    # head index
    
    # Base offsets for this batch/head
    q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    k_offset = off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
    v_offset = off_z.to(tl.int64) * stride_vz + off_h.to(tl.int64) * stride_vh
    o_offset = off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh
    
    # Index offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)        # Q row indices
    offs_d = tl.arange(0, BLOCK_HEADDIM)                       # head_dim indices
    
    # Load Q block: (BLOCK_M, BLOCK_HEADDIM)
    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q_mask = offs_m[:, None] < seq_len_q
    q = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float16)
    
    # Initialize online softmax accumulators
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)   # running max
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)                 # running sum
    o_i = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)  # output accumulator
    
    # KV iteration range — for causal, skip K blocks past the Q diagonal  
    end_n = seq_len_kv
    if IS_CAUSAL:
        end_n = tl.minimum((start_m + 1) * BLOCK_M, seq_len_kv)
    
    # Iterate over KV blocks
    for start_n in range(0, end_n, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        kv_mask = offs_n[:, None] < seq_len_kv
        
        # Load K block: (BLOCK_N, BLOCK_HEADDIM)
        k_ptrs = K + k_offset + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
        k = tl.load(k_ptrs, mask=kv_mask, other=0.0).to(tl.float16)
        
        # Compute attention scores: S = Q @ K^T * scale
        # (BLOCK_M, BLOCK_HEADDIM) @ (BLOCK_HEADDIM, BLOCK_N) -> (BLOCK_M, BLOCK_N)
        s = tl.dot(q, tl.trans(k)) * scale
        
        # Mask out-of-bounds K positions
        s = tl.where(offs_n[None, :] < seq_len_kv, s, float('-inf'))
        
        # Causal mask: Q position i can only attend to K positions <= i
        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            s = tl.where(causal_mask, s, float('-inf'))
        
        # === Online softmax (FlashAttention-2 algorithm) ===
        # New row-wise max
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        # Rescaling factor for previous accumulator
        alpha = tl.exp(m_i - m_new)
        # Softmax numerator for current block
        p = tl.exp(s - m_new[:, None])
        
        # Load V block: (BLOCK_N, BLOCK_HEADDIM)
        v_ptrs = V + v_offset + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
        v = tl.load(v_ptrs, mask=kv_mask, other=0.0).to(tl.float16)
        
        # Update accumulators
        # O = O * exp(m_old - m_new) + P @ V
        l_i = l_i * alpha + tl.sum(p, axis=1)
        o_i = o_i * alpha[:, None] + tl.dot(p.to(tl.float16), v)
        m_i = m_new
    
    # Final normalization: O = O / l
    o_i = o_i / l_i[:, None]
    
    # Store output
    o_ptrs = Out + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    o_mask = offs_m[:, None] < seq_len_q
    tl.store(o_ptrs, o_i.to(tl.float16), mask=o_mask)


def flash_attention_forward(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor, 
    is_causal: bool = False,
) -> torch.Tensor:
    """
    Triton FlashAttention-2 forward pass.
    
    Uses online softmax to compute attention without materializing
    the full N×N attention matrix, reducing memory from O(N²) to O(N).
    
    Args:
        q: (batch, num_heads, seq_len_q, head_dim) float16
        k: (batch, num_heads, seq_len_kv, head_dim) float16
        v: (batch, num_heads, seq_len_kv, head_dim) float16
        is_causal: whether to apply causal (lower-triangular) mask
    Returns:
        out: (batch, num_heads, seq_len_q, head_dim) float16
    """
    batch, num_heads, seq_len_q, head_dim = q.shape
    _, _, seq_len_kv, _ = k.shape
    
    assert head_dim in {64, 128, 256}, f"head_dim={head_dim} not supported, must be 64/128/256"
    
    out = torch.empty_like(q)
    scale = 1.0 / math.sqrt(head_dim)
    
    # Tile sizes tuned for RTX 4080 (101KB shared memory) with head_dim=128
    # Shared mem per tile: Q(BLOCK_M*128*2) + K(BLOCK_N*128*2) + V(BLOCK_N*128*2)
    if head_dim <= 64:
        BLOCK_M, BLOCK_N = 64, 64
        num_warps = 4
    elif head_dim <= 128:
        BLOCK_M, BLOCK_N = 64, 32   # Q=16KB + K=8KB + V=8KB = 32KB per stage
        num_warps = 4
    else:
        BLOCK_M, BLOCK_N = 32, 32
        num_warps = 4
    
    grid = (triton.cdiv(seq_len_q, BLOCK_M), batch * num_heads)
    
    _flash_attn_fwd_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        num_heads=num_heads,
        seq_len_q=seq_len_q,
        seq_len_kv=seq_len_kv,
        scale=scale,
        IS_CAUSAL=is_causal,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_HEADDIM=head_dim,
        num_warps=num_warps,
        num_stages=2,
    )
    
    return out
