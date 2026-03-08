import torch
import torch.nn.functional as F
import sys
sys.path.append('.')
from kernels.rmsnorm import rmsnorm_forward
from kernels.rope import apply_rope_inplace, precompute_freqs_cis

def test_rmsnorm():
    print("Testing RMSNorm Kernel...")
    batch, seqlen, hidden = 2, 512, 3584
    x = torch.randn(batch, seqlen, hidden, dtype=torch.float16, device="cuda")
    weight = torch.ones(hidden, dtype=torch.float16, device="cuda")
    
    # PyTorch Output
    torch_out = x.to(torch.float32)
    var = torch_out.pow(2).mean(-1, keepdim=True)
    torch_out = torch_out * torch.rsqrt(var + 1e-6)
    torch_out = (weight.to(torch.float32) * torch_out).to(torch.float16)
    
    # Triton Output
    triton_out = rmsnorm_forward(x.clone(), weight)
    # print(triton_out)
    # print(torch_out)
    diff = (triton_out - torch_out).abs().max()
    print(f"RMSNorm Max Diff: {diff:.6f}")
    assert diff < 1e-2, "RMSNorm Triton kernel output deviates too much from PyTorch!"
    print("RMSNorm Kernel passed!\n")

def test_rope():
    print("Testing RoPE Kernel...")
    seq_len = 10
    head_dim = 128
    num_heads = 28
    num_kv_heads = 4
    
    q = torch.randn(1, seq_len, num_heads, head_dim, dtype=torch.float32, device="cuda")
    k = torch.randn(1, seq_len, num_kv_heads, head_dim, dtype=torch.float32, device="cuda")
    pos_ids = torch.arange(seq_len, dtype=torch.int32, device="cuda")
    
    cos, sin = precompute_freqs_cis(head_dim, 1024, device="cuda")
    
    # Save original for PyTorch comparison
    q_pt, k_pt = q.clone(), k.clone()
    
    # Run PyTorch equivalent
    half_dim = head_dim // 2
    
    def rotate_half(x):
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]
        return torch.cat((-x2, x1), dim=-1)
        
    # The format of our precomputed `cos` and `sin` is duplicated along the last dimension.
    # PyTorch implementation (matching standard HF transformers)
    cos_pt = cos[pos_ids, :].unsqueeze(1) # shape: (1, seq_len, 1, head_dim)
    sin_pt = sin[pos_ids, :].unsqueeze(1)
    
    # Note: `cos` and `sin` in our implementation is [cos_0...cos_{d/2-1}, cos_0...cos_{d/2-1}]
    # PyTorch needs to multiply this duplicated version matching `rotate_half`
    
    q_pt_out = (q_pt * cos_pt) + (rotate_half(q_pt) * sin_pt)
    k_pt_out = (k_pt * cos_pt) + (rotate_half(k_pt) * sin_pt)
    
    # Run Triton kernel (In-place)
    apply_rope_inplace(q, k, cos, sin, pos_ids)
    diff_q = (q - q_pt_out).abs().max()
    diff_k = (k - k_pt_out).abs().max()
    print(f"RoPE Q Max Diff: {diff_q:.6f}")
    print(f"RoPE K Max Diff: {diff_k:.6f}")
    assert diff_q < 1e-3 and diff_k < 1e-3, "RoPE Triton kernel failed"
    print("RoPE Kernel passed!")

if __name__ == "__main__":
    test_rmsnorm()
    test_rope()
    print("\nAll kernel tests passed!")
