import torch
import sys
sys.path.append('.')
from kernels.awq_gemm import awq_gemm_forward

def test_awq():
    print("Testing AWQ Matrix Multiplication...")
    K = 4096
    N = 4096
    M = 2  # batch * seq_len
    group_size = 128
    
    x = torch.randn(M, K, dtype=torch.float16, device="cuda")
    
    # Fake packed weights
    qweight = torch.randint(0, 2**31 - 1, (K // 8, N), dtype=torch.int32, device="cuda")
    qzeros = torch.randint(0, 2**31 - 1, (K // group_size, N // 8), dtype=torch.int32, device="cuda")
    scales = torch.randn(K // group_size, N, dtype=torch.float16, device="cuda")
    
    out = awq_gemm_forward(x, qweight, qzeros, scales, group_size)
    print("Success! Output shape:", out.shape)
    
if __name__ == "__main__":
    test_awq()
