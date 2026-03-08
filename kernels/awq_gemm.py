import torch
import triton
import triton.language as tl

# AutoAWQ bit-field interleaving order: [0, 4, 1, 5, 2, 6, 3, 7]
# Each INT32 packs 8 x 4-bit weights. The unpacking shift amounts are:
# weight_i = (packed >> (order[i] * 4)) & 0xF
# order[i] = (i // 2) + (i % 2) * 4

@triton.autotune(
    configs=[
        # Conservative configs to fit RTX 4080's 101KB shared memory
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _awq_gemm_kernel(
    # Pointers
    A_ptr,          # (M, K) fp16 activations
    QW_ptr,         # (K, N // 8) int32 packed weights
    QZ_ptr,         # (K // group_size, N // 8) int32 packed zeros
    Scales_ptr,     # (K // group_size, N) fp16 scales
    C_ptr,          # (M, N) fp16 output
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_qw_k, stride_qw_n,
    stride_qz_k, stride_qz_n,
    stride_s_k, stride_s_n,
    stride_cm, stride_cn,
    # Quantization config
    group_size: tl.constexpr,
    # Tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Fused AWQ W4A16 GEMM: Dequantize INT4 weights on-the-fly and multiply with FP16 activations.
    
    AutoAWQ packs 8 x 4-bit weights per INT32 with interleaved order [0,4,1,5,2,6,3,7].
    Zeros use the same packing format.
    
    This kernel fuses: unpack INT4 -> subtract zeros -> multiply scales -> GEMM accumulate
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    # L2 cache-friendly swizzle
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Offsets for the M and N dimensions
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # For packed weights: N columns are stored as N//8 int32 columns
    packed_n = offs_n // 8           # which int32 column
    element_idx = offs_n % 8         # which element within the int32
    
    # AutoAWQ interleaving shift: order[i] = (i // 2) + (i % 2) * 4
    shift_amount = ((element_idx // 2) + (element_idx % 2) * 4) * 4
    
    # Accumulator in FP32 for precision
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # K-dimension loop
    for k_start in range(0, K, BLOCK_SIZE_K):
        offs_k = k_start + tl.arange(0, BLOCK_SIZE_K)
        
        # Load A tile: (BLOCK_SIZE_M, BLOCK_SIZE_K)
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak, 
                     mask=a_mask, other=0.0)
        
        # Load packed weights: (BLOCK_SIZE_K, BLOCK_SIZE_N)
        qw_ptrs = QW_ptr + offs_k[:, None] * stride_qw_k + packed_n[None, :] * stride_qw_n
        qw_mask = (offs_k[:, None] < K) & (packed_n[None, :] < (N // 8))
        qw_packed = tl.load(qw_ptrs, mask=qw_mask, other=0)
        
        # Extract 4-bit weights
        w_int4 = (qw_packed >> shift_amount[None, :]) & 0xF
        
        # Load zeros
        group_idx = offs_k // group_size
        qz_ptrs = QZ_ptr + group_idx[:, None] * stride_qz_k + packed_n[None, :] * stride_qz_n
        qz_mask = (group_idx[:, None] < (K // group_size)) & (packed_n[None, :] < (N // 8))
        qz_packed = tl.load(qz_ptrs, mask=qz_mask, other=0)
        
        # Extract 4-bit zeros
        z_int4 = (qz_packed >> shift_amount[None, :]) & 0xF
        
        # Load scales
        s_ptrs = Scales_ptr + group_idx[:, None] * stride_s_k + offs_n[None, :] * stride_s_n
        s_mask = (group_idx[:, None] < (K // group_size)) & (offs_n[None, :] < N)
        scales = tl.load(s_ptrs, mask=s_mask, other=0.0)
        
        # Dequantize: w_fp = (w_int4 - z_int4) * scales
        w_fp = (w_int4.to(tl.float16) - z_int4.to(tl.float16)) * scales
        
        # Accumulate: acc += A_tile @ W_tile
        acc += tl.dot(a.to(tl.float16), w_fp.to(tl.float16))
    
    # Store output
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


def awq_gemm_forward(x: torch.Tensor, qweight: torch.Tensor, qzeros: torch.Tensor, scales: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    """
    Fused AWQ W4A16 GEMM using Triton.
    
    Args:
        x: (..., K) fp16 inputs
        qweight: (K, N//8) int32 packed weights
        qzeros: (K//group_size, N//8) int32 packed zeros  
        scales: (K//group_size, N) fp16 scales
    Returns:
        out: (..., N) fp16 
    """
    in_shape = x.shape
    x_flat = x.reshape(-1, x.shape[-1]).contiguous()
    M, K = x_flat.shape
    N = scales.shape[1]
    
    out = torch.empty((M, N), dtype=torch.float16, device=x.device)
    
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    
    _awq_gemm_kernel[grid](
        x_flat, qweight, qzeros, scales, out,
        M, N, K,
        x_flat.stride(0), x_flat.stride(1),
        qweight.stride(0), qweight.stride(1),
        qzeros.stride(0), qzeros.stride(1),
        scales.stride(0), scales.stride(1),
        out.stride(0), out.stride(1),
        group_size=group_size,
    )
    
    return out.view(*in_shape[:-1], N)
