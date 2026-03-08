import torch
import torch.nn.functional as F
import sys
sys.path.append('.')
from kernels.rmsnorm import rmsnorm_forward
from kernels.rope import apply_rope_inplace, precompute_freqs_cis
from kernels.silu_mul import silu_mul_forward
from kernels.fused_add_rmsnorm import fused_add_rmsnorm_forward

def test_rmsnorm():
    print("Testing RMSNorm Kernel...")
    batch, seqlen, hidden = 2, 512, 3584
    x = torch.randn(batch, seqlen, hidden, dtype=torch.float16, device="cuda")
    weight = torch.ones(hidden, dtype=torch.float16, device="cuda")
    
    # PyTorch reference
    torch_out = x.to(torch.float32)
    var = torch_out.pow(2).mean(-1, keepdim=True)
    torch_out = torch_out * torch.rsqrt(var + 1e-6)
    torch_out = (weight.to(torch.float32) * torch_out).to(torch.float16)
    
    # Triton
    triton_out = rmsnorm_forward(x.clone(), weight)
    
    diff = (triton_out - torch_out).abs().max()
    print(f"  RMSNorm Max Diff: {diff:.6f}")
    assert diff < 1e-2, f"RMSNorm failed! Max diff: {diff}"
    print("  ✅ RMSNorm passed!\n")

def test_rope():
    print("Testing RoPE Kernel...")
    seq_len = 10
    head_dim = 128
    num_heads = 28
    num_kv_heads = 4
    
    q = torch.randn(1, seq_len, num_heads, head_dim, dtype=torch.float16, device="cuda")
    k = torch.randn(1, seq_len, num_kv_heads, head_dim, dtype=torch.float16, device="cuda")
    pos_ids = torch.arange(seq_len, dtype=torch.int32, device="cuda")
    
    cos, sin = precompute_freqs_cis(head_dim, 1024, device="cuda")
    
    q_pt, k_pt = q.clone(), k.clone()
    
    # PyTorch reference
    half_dim = head_dim // 2
    def rotate_half(x):
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]
        return torch.cat((-x2, x1), dim=-1)
    
    cos_pt = cos[pos_ids, :].unsqueeze(0).unsqueeze(2)
    sin_pt = sin[pos_ids, :].unsqueeze(0).unsqueeze(2)
    
    q_pt_out = (q_pt.float() * cos_pt) + (rotate_half(q_pt.float()) * sin_pt)
    k_pt_out = (k_pt.float() * cos_pt) + (rotate_half(k_pt.float()) * sin_pt)
    
    # Triton (in-place)
    apply_rope_inplace(q, k, cos, sin, pos_ids)
    
    diff_q = (q.float() - q_pt_out).abs().max()
    diff_k = (k.float() - k_pt_out).abs().max()
    print(f"  RoPE Q Max Diff: {diff_q:.6f}")
    print(f"  RoPE K Max Diff: {diff_k:.6f}")
    assert diff_q < 1e-2 and diff_k < 1e-2, f"RoPE failed! Q diff: {diff_q}, K diff: {diff_k}"
    print("  ✅ RoPE passed!\n")

def test_silu_mul():
    print("Testing SiLU*Mul Kernel...")
    size = (2, 512, 18944)  # Qwen2-7B intermediate_size
    gate = torch.randn(size, dtype=torch.float16, device="cuda")
    up = torch.randn(size, dtype=torch.float16, device="cuda")
    
    # PyTorch reference
    torch_out = F.silu(gate.float()) * up.float()
    torch_out = torch_out.to(torch.float16)
    
    # Triton
    triton_out = silu_mul_forward(gate.clone(), up.clone())
    
    diff = (triton_out - torch_out).abs().max()
    print(f"  SiLU*Mul Max Diff: {diff:.6f}")
    assert diff < 0.1, f"SiLU*Mul failed! Max diff: {diff}"
    print("  ✅ SiLU*Mul passed!\n")

def test_fused_add_rmsnorm():
    print("Testing Fused Add-RMSNorm Kernel...")
    batch, seqlen, hidden = 2, 512, 3584
    residual = torch.randn(batch, seqlen, hidden, dtype=torch.float16, device="cuda")
    x = torch.randn(batch, seqlen, hidden, dtype=torch.float16, device="cuda")
    weight = torch.ones(hidden, dtype=torch.float16, device="cuda")
    eps = 1e-6
    
    # PyTorch reference
    residual_pt = residual.clone()
    residual_pt = residual_pt + x
    h = residual_pt.to(torch.float32)
    var = h.pow(2).mean(-1, keepdim=True)
    h = h * torch.rsqrt(var + eps)
    torch_out = (weight.to(torch.float32) * h).to(torch.float16)
    
    # Triton (residual is modified in-place)
    residual_triton = residual.clone()
    triton_out = fused_add_rmsnorm_forward(residual_triton, x, weight, eps)
    
    # Check residual was updated
    diff_residual = (residual_triton - (residual + x)).abs().max()
    diff_norm = (triton_out - torch_out).abs().max()
    print(f"  Residual Update Max Diff: {diff_residual:.6f}")
    print(f"  Norm Output Max Diff: {diff_norm:.6f}")
    assert diff_residual < 1e-2, f"Residual update failed! Max diff: {diff_residual}"
    assert diff_norm < 1e-2, f"Fused norm output failed! Max diff: {diff_norm}"
    print("  ✅ Fused Add-RMSNorm passed!\n")

def test_awq_dequant():
    """Test AWQ GEMM kernel against PyTorch reference dequantization."""
    print("Testing AWQ GEMM Kernel...")
    from kernels.awq_gemm import awq_gemm_forward
    
    K, N = 3584, 3584
    group_size = 128
    M = 4  # Small batch
    
    # Create test inputs
    x = torch.randn(M, K, dtype=torch.float16, device="cuda")
    qweight = torch.randint(0, 2**31, (K, N // 8), dtype=torch.int32, device="cuda")
    qzeros = torch.randint(0, 2**31, (K // group_size, N // 8), dtype=torch.int32, device="cuda")
    scales = torch.randn(K // group_size, N, dtype=torch.float16, device="cuda") * 0.01
    
    # PyTorch reference (from original code)
    pack_num = 8
    qw = qweight.view(K, N // 8, 1).expand(K, N // 8, pack_num)
    order = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], device=x.device, dtype=torch.int32)
    shifts = (order * 4).view(1, 1, pack_num)
    w_fp = ((qw >> shifts) & 0xF).flatten(1, 2).view(K, N).to(torch.float16)
    
    qz = qzeros.view(K // group_size, N // 8, 1).expand(K // group_size, N // 8, pack_num)
    z_fp = ((qz >> shifts) & 0xF).flatten(1, 2).view(scales.shape[0], N).to(torch.float16)
    
    w_fp_dq = w_fp.view(K // group_size, group_size, N)
    z_fp_dq = z_fp.view(K // group_size, 1, N)
    s_fp_dq = scales.view(K // group_size, 1, N)
    w_fp_dq = (w_fp_dq - z_fp_dq) * s_fp_dq
    w_fp_dq = w_fp_dq.view(K, N)
    
    torch_out = torch.matmul(x.float(), w_fp_dq.float()).to(torch.float16)
    
    # Triton
    triton_out = awq_gemm_forward(x, qweight, qzeros, scales, group_size)
    
    # Check relative error
    abs_diff = (triton_out.float() - torch_out.float()).abs()
    rel_diff = abs_diff / (torch_out.float().abs() + 1e-6)
    max_abs_diff = abs_diff.max().item()
    max_rel_diff = rel_diff.max().item()
    mean_rel_diff = rel_diff.mean().item()
    
    print(f"  AWQ GEMM Max Abs Diff: {max_abs_diff:.6f}")
    print(f"  AWQ GEMM Max Rel Diff: {max_rel_diff:.6f}")
    print(f"  AWQ GEMM Mean Rel Diff: {mean_rel_diff:.6f}")
    
    # Allow some tolerance due to FP16 accumulation differences
    assert mean_rel_diff < 0.1, f"AWQ GEMM mean relative error too high: {mean_rel_diff}"
    print("  ✅ AWQ GEMM passed!\n")


if __name__ == "__main__":
    print("=" * 50)
    print("  Triton Kernel Correctness Tests")
    print("=" * 50 + "\n")
    
    test_rmsnorm()
    test_rope()
    test_silu_mul()
    test_fused_add_rmsnorm()
    test_awq_dequant()
    
    print("=" * 50)
    print("  ✅ All kernel tests passed!")
    print("=" * 50)
