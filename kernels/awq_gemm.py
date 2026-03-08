import torch
import triton
import triton.language as tl

# AutoAWQ bit-field interleaving order: [0, 4, 1, 5, 2, 6, 3, 7]
# order[i] = (i // 2) + (i % 2) * 4

# =============================================================================
# Dequantization Kernel — Unpack INT4 → FP16 for cuBLAS matmul
# =============================================================================
# Pre-decode strategy: fast Triton dequant to FP16, then cuBLAS GEMV.
# cuBLAS achieves near-optimal bandwidth (60-80%) vs our GEMV kernel (~7%).

@triton.jit
def _dequant_awq_kernel(
    QW_ptr,         # (K, N//8) int32 packed weights
    QZ_ptr,         # (K//gs, N//8) int32 packed zeros
    S_ptr,          # (K//gs, N) fp16 scales
    Out_ptr,        # (K, N) fp16 output
    K, N,
    stride_qw_k, stride_qw_n,
    stride_qz_k, stride_qz_n,
    stride_s_k, stride_s_n,
    stride_o_k, stride_o_n,
    group_size: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Pure dequantization: INT4 → FP16. No matmul, just memory transform."""
    pid_k = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    packed_n = offs_n // 8
    element_idx = offs_n % 8
    shift_amount = ((element_idx // 2) + (element_idx % 2) * 4) * 4
    
    k_mask = offs_k < K
    n_mask = offs_n < N
    
    # Load packed weights
    qw = tl.load(QW_ptr + offs_k[:, None] * stride_qw_k + packed_n[None, :] * stride_qw_n,
                 mask=k_mask[:, None] & (packed_n[None, :] < N // 8), other=0)
    w_int4 = (qw >> shift_amount[None, :]) & 0xF
    
    # Load zeros
    group_idx = offs_k // group_size
    qz = tl.load(QZ_ptr + group_idx[:, None] * stride_qz_k + packed_n[None, :] * stride_qz_n,
                 mask=(group_idx[:, None] < K // group_size) & (packed_n[None, :] < N // 8), other=0)
    z_int4 = (qz >> shift_amount[None, :]) & 0xF
    
    # Load scales
    scales = tl.load(S_ptr + group_idx[:, None] * stride_s_k + offs_n[None, :] * stride_s_n,
                     mask=(group_idx[:, None] < K // group_size) & n_mask[None, :], other=0.0)
    
    # Dequantize to FP16
    w_fp = (w_int4.to(tl.float16) - z_int4.to(tl.float16)) * scales
    
    # Store
    tl.store(Out_ptr + offs_k[:, None] * stride_o_k + offs_n[None, :] * stride_o_n,
             w_fp.to(tl.float16), mask=k_mask[:, None] & n_mask[None, :])


def _dequant_to_fp16(qweight, qzeros, scales, out_buf, group_size=128):
    """Fast Triton dequantization: AWQ INT4 packed weights → FP16 dense matrix."""
    K, packed_N = qweight.shape
    N = packed_N * 8
    
    BLOCK_K = 32
    BLOCK_N = 64
    grid = (triton.cdiv(K, BLOCK_K), triton.cdiv(N, BLOCK_N))
    
    _dequant_awq_kernel[grid](
        qweight, qzeros, scales, out_buf,
        K, N,
        qweight.stride(0), qweight.stride(1),
        qzeros.stride(0), qzeros.stride(1),
        scales.stride(0), scales.stride(1),
        out_buf.stride(0), out_buf.stride(1),
        group_size=group_size,
        BLOCK_K=BLOCK_K,
        BLOCK_N=BLOCK_N,
        num_warps=4,
    )
    return out_buf


# =============================================================================
# GEMV Kernel — Fallback for cases where dequant+cuBLAS isn't applicable
# =============================================================================

@triton.jit
def _awq_gemv_kernel(
    X_ptr, QW_ptr, QZ_ptr, S_ptr, Out_ptr,
    M, N, K,
    stride_x_m, stride_x_k,
    stride_qw_k, stride_qw_n,
    stride_qz_k, stride_qz_n,
    stride_s_k, stride_s_n,
    stride_o_m, stride_o_n,
    group_size: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """AWQ W4A16 GEMV for decode (kept as fallback)."""
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    packed_n = offs_n // 8
    element_idx = offs_n % 8
    shift_amount = ((element_idx // 2) + (element_idx % 2) * 4) * 4
    
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K
        
        x = tl.load(X_ptr + pid_m * stride_x_m + offs_k * stride_x_k,
                     mask=k_mask, other=0.0).to(tl.float32)
        
        qw_ptrs = QW_ptr + offs_k[:, None] * stride_qw_k + packed_n[None, :] * stride_qw_n
        qw_mask = (offs_k[:, None] < K) & (packed_n[None, :] < N // 8)
        qw = tl.load(qw_ptrs, mask=qw_mask, other=0)
        w_int4 = (qw >> shift_amount[None, :]) & 0xF
        
        group_idx = offs_k // group_size
        qz_ptrs = QZ_ptr + group_idx[:, None] * stride_qz_k + packed_n[None, :] * stride_qz_n
        qz_mask = (group_idx[:, None] < K // group_size) & (packed_n[None, :] < N // 8)
        qz = tl.load(qz_ptrs, mask=qz_mask, other=0)
        z_int4 = (qz >> shift_amount[None, :]) & 0xF
        
        s_ptrs = S_ptr + group_idx[:, None] * stride_s_k + offs_n[None, :] * stride_s_n
        s_mask = (group_idx[:, None] < K // group_size) & (offs_n[None, :] < N)
        scales = tl.load(s_ptrs, mask=s_mask, other=0.0)
        
        w_fp = (w_int4.to(tl.float32) - z_int4.to(tl.float32)) * scales.to(tl.float32)
        acc += tl.sum(x[:, None] * w_fp, axis=0)
    
    out_mask = offs_n < N
    tl.store(Out_ptr + pid_m * stride_o_m + offs_n * stride_o_n,
             acc.to(tl.float16), mask=out_mask)


# =============================================================================
# GEMM Kernel — For Prefill Phase (M > threshold)
# =============================================================================

@triton.autotune(
    configs=[
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
    A_ptr, QW_ptr, QZ_ptr, Scales_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_qw_k, stride_qw_n,
    stride_qz_k, stride_qz_n,
    stride_s_k, stride_s_n,
    stride_cm, stride_cn,
    group_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Fused AWQ W4A16 GEMM for prefill phase (M > 4)."""
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    packed_n = offs_n // 8
    element_idx = offs_n % 8
    shift_amount = ((element_idx // 2) + (element_idx % 2) * 4) * 4
    
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k_start in range(0, K, BLOCK_SIZE_K):
        offs_k = k_start + tl.arange(0, BLOCK_SIZE_K)
        
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak, 
                     mask=a_mask, other=0.0)
        
        qw_ptrs = QW_ptr + offs_k[:, None] * stride_qw_k + packed_n[None, :] * stride_qw_n
        qw_mask = (offs_k[:, None] < K) & (packed_n[None, :] < (N // 8))
        qw_packed = tl.load(qw_ptrs, mask=qw_mask, other=0)
        w_int4 = (qw_packed >> shift_amount[None, :]) & 0xF
        
        group_idx = offs_k // group_size
        qz_ptrs = QZ_ptr + group_idx[:, None] * stride_qz_k + packed_n[None, :] * stride_qz_n
        qz_mask = (group_idx[:, None] < (K // group_size)) & (packed_n[None, :] < (N // 8))
        qz_packed = tl.load(qz_ptrs, mask=qz_mask, other=0)
        z_int4 = (qz_packed >> shift_amount[None, :]) & 0xF
        
        s_ptrs = Scales_ptr + group_idx[:, None] * stride_s_k + offs_n[None, :] * stride_s_n
        s_mask = (group_idx[:, None] < (K // group_size)) & (offs_n[None, :] < N)
        scales = tl.load(s_ptrs, mask=s_mask, other=0.0)
        
        w_fp = (w_int4.to(tl.float16) - z_int4.to(tl.float16)) * scales
        acc += tl.dot(a.to(tl.float16), w_fp.to(tl.float16))
    
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


# =============================================================================
# Pre-allocated dequantization buffer (reused across all projections)
# =============================================================================
_dequant_buffer = None


def awq_gemm_forward(x: torch.Tensor, qweight: torch.Tensor, qzeros: torch.Tensor, scales: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    """
    AWQ W4A16 GEMM/GEMV with pre-decode optimization.
    
    For decode (M<=4): Triton dequant INT4→FP16 + cuBLAS matmul
    For prefill (M>4): Fused Triton GEMM kernel
    """
    in_shape = x.shape
    x_flat = x.reshape(-1, x.shape[-1]).contiguous()
    M, K = x_flat.shape
    N = scales.shape[1]
    
    out = torch.empty((M, N), dtype=torch.float16, device=x.device)
    
    if M <= 4:
        # === Pre-decode + cuBLAS path ===
        # Step 1: Fast Triton dequant INT4 → FP16 (pure memory transform)
        # Step 2: cuBLAS FP16 matmul (near-optimal bandwidth utilization)
        global _dequant_buffer
        K_w = qweight.shape[0]
        buf_size = K_w * N
        
        if _dequant_buffer is None or _dequant_buffer.numel() < buf_size:
            _dequant_buffer = torch.empty(buf_size, dtype=torch.float16, device=x.device)
        
        w_fp16 = _dequant_buffer[:buf_size].view(K_w, N)
        _dequant_to_fp16(qweight, qzeros, scales, w_fp16, group_size)
        
        # cuBLAS GEMV — highly optimized by NVIDIA
        out = torch.matmul(x_flat, w_fp16)
    else:
        # === GEMM path: tiled GEMM for prefill ===
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
