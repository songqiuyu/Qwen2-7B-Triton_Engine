import torch
import torch.nn.functional as F
import math

# Baseline RMSNorm
def rmsnorm_forward(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return weight * x

# Baseline Fused Add RMSNorm
def fused_add_rmsnorm_forward(residual: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    residual.add_(x)
    return rmsnorm_forward(residual, weight, eps)

# Baseline RoPE
def apply_rope_inplace(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: torch.Tensor):
    # This might not be 100% exactly the same in-place semantics if we create new tensors, 
    # but we can return the modified tensors.
    def rotate_half(x):
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    cos_q = cos[position_ids].view(cos.shape[0] if position_ids.dim() == 2 else 1, -1, 1, cos.shape[-1])
    sin_q = sin[position_ids].view(sin.shape[0] if position_ids.dim() == 2 else 1, -1, 1, sin.shape[-1])
    
    q_embed = (q * cos_q) + (rotate_half(q) * sin_q)
    k_embed = (k * cos_q) + (rotate_half(k) * sin_q)
    
    q.copy_(q_embed)
    k.copy_(k_embed)
    return q, k

# Baseline AWQ GEMM
def awq_gemm_forward(x: torch.Tensor, qweight: torch.Tensor, qzeros: torch.Tensor, scales: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    in_shape = x.shape
    x_flat = x.reshape(-1, x.shape[-1])
    
    K, packed_N = qweight.shape
    N = packed_N * 8
    
    # Dequantize
    w_fp16 = torch.zeros((K, N), dtype=torch.float16, device=x.device)
    for i in range(N):
        packed_n = i // 8
        element_idx = i % 8
        shift = ((element_idx // 2) + (element_idx % 2) * 4) * 4
        
        w_int4 = (qweight[:, packed_n] >> shift) & 0xF
        
        group_idx = torch.arange(K, device=x.device) // group_size
        z_int4 = (qzeros[group_idx, packed_n] >> shift) & 0xF
        
        scale = scales[group_idx, i]
        
        w_fp16[:, i] = (w_int4.half() - z_int4.half()) * scale
        
    out = torch.matmul(x_flat, w_fp16)
    return out.view(*in_shape[:-1], N)

# Baseline SiLU Mul
def silu_mul_forward(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return F.silu(gate) * up

# Baseline Flash Attention
def flash_attention_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
    batch, num_heads, seq_len_q, head_dim = q.shape
    _, _, seq_len_kv, _ = k.shape
    
    scale = 1.0 / math.sqrt(head_dim)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    
    if is_causal and seq_len_q > 1:
        mask = torch.ones(seq_len_q, seq_len_kv, device=q.device, dtype=torch.bool).tril(diagonal=seq_len_kv - seq_len_q)
        scores.masked_fill_(~mask, float('-inf'))
        
    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn.to(q.dtype), v)
    return out
