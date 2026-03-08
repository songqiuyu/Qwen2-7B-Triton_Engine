import torch
import sys
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('.')
from transformers import AutoTokenizer
from model.loader import load_config, load_weights
from model.config import Qwen2Config
from model.qwen2 import Qwen2ForCausalLM
from inference.engine import TritonInferenceEngine

# Import kernel modules for monkeypatching
import kernels.rmsnorm
import kernels.rope
import kernels.silu_mul
import kernels.fused_add_rmsnorm
import kernels.flash_attention

# ============================================================
# Pure PyTorch Fallback Implementations (Baseline)
# ============================================================
def torch_rmsnorm(x, weight, eps):
    input_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return (weight.to(torch.float32) * x).to(input_dtype)

def torch_apply_rope(q, k, cos, sin, position_ids):
    head_dim = q.shape[-1]
    half_dim = head_dim // 2
    
    def rotate_half(x):
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]
        return torch.cat((-x2, x1), dim=-1)
    
    pos_flat = position_ids.reshape(-1)
    cos_seq = cos[pos_flat, :].view(1, -1, 1, head_dim)
    sin_seq = sin[pos_flat, :].view(1, -1, 1, head_dim)
    
    q_out = (q.float() * cos_seq) + (rotate_half(q.float()) * sin_seq)
    k_out = (k.float() * cos_seq[:, :, :, :]) + (rotate_half(k.float()) * sin_seq[:, :, :, :])
    
    q.copy_(q_out.to(q.dtype))
    k.copy_(k_out.to(k.dtype))
    return q, k

def torch_silu_mul(gate, up, out=None):
    result = torch.nn.functional.silu(gate.float()) * up.float()
    result = result.to(gate.dtype)
    if out is not None:
        out.copy_(result)
        return out
    return result

def torch_fused_add_rmsnorm(residual, x, weight, eps=1e-6):
    """PyTorch equivalent of fused add + rmsnorm."""
    input_dtype = residual.dtype
    residual.add_(x)
    # RMSNorm
    h = residual.to(torch.float32)
    variance = h.pow(2).mean(-1, keepdim=True)
    h = h * torch.rsqrt(variance + eps)
    return (weight.to(torch.float32) * h).to(input_dtype)

# AWQ PyTorch dequant fallback (from original code)
def torch_awq_gemm(x, qweight, qzeros, scales, group_size=128):
    in_shape = x.shape
    x = x.view(-1, x.shape[-1])
    M, K = x.shape
    pack_num = 8
    N = scales.shape[1]
    
    qw = qweight.view(K, N // 8, 1).expand(K, N // 8, pack_num)
    order = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], device=x.device, dtype=torch.int32)
    shifts = (order * 4).view(1, 1, pack_num)
    w_fp = ((qw >> shifts) & 0xF).flatten(1, 2).view(K, N).to(torch.float16)
    
    qz = qzeros.view(K // group_size, N // 8, 1).expand(K // group_size, N // 8, pack_num)
    z_fp = ((qz >> shifts) & 0xF).flatten(1, 2).view(scales.shape[0], N).to(torch.float16)
    
    w_fp = w_fp.view(K // group_size, group_size, N)
    z_fp = z_fp.view(K // group_size, 1, N)
    s_fp = scales.view(K // group_size, 1, N)
    
    w_fp = (w_fp - z_fp) * s_fp
    w_fp = w_fp.view(K, N)
    
    out = torch.matmul(x, w_fp)  # FP16 × FP16, same precision as Triton tl.dot
    return out.view(*in_shape[:-1], N)


def run_benchmark(model, tokenizer, use_triton=True, num_runs=3, max_tokens=150):
    import kernels.awq_gemm
    
    # PyTorch SDPA fallback for attention
    def torch_flash_attention(q, k, v, is_causal=False):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
    
    if use_triton:
        # Use Triton kernels
        kernels.rmsnorm.rmsnorm_forward = _original_rmsnorm
        kernels.rope.apply_rope_inplace = _original_rope
        kernels.silu_mul.silu_mul_forward = _original_silu_mul
        kernels.fused_add_rmsnorm.fused_add_rmsnorm_forward = _original_fused_add_rmsnorm
        kernels.awq_gemm.awq_gemm_forward = _original_awq_gemm
        kernels.flash_attention.flash_attention_forward = _original_flash_attn
        mode = "🚀 Triton Optimized"
    else:
        # Use pure PyTorch
        kernels.rmsnorm.rmsnorm_forward = torch_rmsnorm
        kernels.rope.apply_rope_inplace = torch_apply_rope
        kernels.silu_mul.silu_mul_forward = torch_silu_mul
        kernels.fused_add_rmsnorm.fused_add_rmsnorm_forward = torch_fused_add_rmsnorm
        kernels.awq_gemm.awq_gemm_forward = torch_awq_gemm
        kernels.flash_attention.flash_attention_forward = torch_flash_attention
        mode = "🐢 Pure PyTorch Baseline"

    engine = TritonInferenceEngine(model, tokenizer, max_seq_len=2048, device="cuda")
    prompt = "<|im_start|>user\n你好，请介绍一下西安<|im_end|>\n<|im_start|>assistant\n"
    
    print(f"\n{'='*50}")
    print(f"[{mode}]")
    print(f"{'='*50}")
    
    # Warm up (JIT compile + cache warm)
    print("Warming up...")
    engine.generate(prompt, max_new_tokens=10, print_stream=False)
    engine.generate(prompt, max_new_tokens=10, print_stream=False)
    
    # Multi-run benchmark
    all_tps = []
    all_prefill_tps = []
    best_out = ""
    
    for run_idx in range(num_runs):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        
        start_t = time.time()
        out_text = engine.generate(prompt, max_new_tokens=max_tokens, temperature=0.0, print_stream=False)
        torch.cuda.synchronize()
        total_time = time.time() - start_t
        
        gen_tokens = len(tokenizer.encode(out_text))
        tps = gen_tokens / total_time
        all_tps.append(tps)
        best_out = out_text
        print(f"  Run {run_idx+1}: {tps:.1f} tokens/s ({gen_tokens} tokens in {total_time:.2f}s)")
    
    max_mem = torch.cuda.max_memory_allocated() / 1024**3
    avg_tps = np.mean(all_tps)
    best_tps = np.max(all_tps)
    
    print(f"\n  Average: {avg_tps:.1f} tokens/s | Best: {best_tps:.1f} tokens/s | Peak VRAM: {max_mem:.2f} GB")
    print(f"  Output: {best_out[:80]}...")
    
    return avg_tps, best_tps, max_mem, best_out


def main():
    model_dir = "models/qwen/Qwen2___5-7B-Instruct-AWQ"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    config_dict = load_config(model_dir)
    config = Qwen2Config(**{k: v for k, v in config_dict.items() if hasattr(Qwen2Config, k)})
    model = Qwen2ForCausalLM(config).to("cuda", dtype=torch.float16)
    
    weights_gen = load_weights(model_dir, device="cuda")
    model.load_awq_weights(weights_gen)
    
    # Save original Triton kernel references
    import kernels.awq_gemm
    global _original_rmsnorm, _original_rope, _original_silu_mul, _original_fused_add_rmsnorm, _original_awq_gemm, _original_flash_attn
    _original_rmsnorm = kernels.rmsnorm.rmsnorm_forward
    _original_rope = kernels.rope.apply_rope_inplace
    _original_silu_mul = kernels.silu_mul.silu_mul_forward
    _original_fused_add_rmsnorm = kernels.fused_add_rmsnorm.fused_add_rmsnorm_forward
    _original_awq_gemm = kernels.awq_gemm.awq_gemm_forward
    _original_flash_attn = kernels.flash_attention.flash_attention_forward

    print("\n" + "=" * 60)
    print("  BENCHMARK: TRITON OPTIMIZED vs PURE PYTORCH BASELINE  ")
    print("  RTX 4080 | Qwen2.5-7B-Instruct-AWQ | INT4 Quantized  ")
    print("=" * 60)
    
    # Run PyTorch baseline first
    avg_torch, best_torch, mem_torch, out_torch = run_benchmark(model, tokenizer, use_triton=False, num_runs=3)
    
    # Run Triton optimized
    avg_triton, best_triton, mem_triton, out_triton = run_benchmark(model, tokenizer, use_triton=True, num_runs=3)
    
    # Calculate speedup
    speedup = avg_triton / avg_torch if avg_torch > 0 else 0
    
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Pure PyTorch: {avg_torch:.1f} tokens/s (best: {best_torch:.1f})")
    print(f"  Triton:       {avg_triton:.1f} tokens/s (best: {best_triton:.1f})")
    print(f"  Speedup:      {speedup:.2f}x")
    print(f"  VRAM (PyTorch): {mem_torch:.2f} GB | VRAM (Triton): {mem_triton:.2f} GB")
    print("=" * 60)
    
    # ============================================
    # Generate comparison chart 
    # ============================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    labels = ['Pure PyTorch', 'Triton Optimized']
    colors = ['#E74C3C', '#27AE60']
    
    # Throughput bar chart
    ax1 = axes[0]
    bars1 = ax1.bar(labels, [avg_torch, avg_triton], color=colors, width=0.6, edgecolor='white', linewidth=1.5)
    ax1.set_ylabel('Throughput (Tokens/s)', fontweight='bold', fontsize=12)
    ax1.set_title('Inference Speed', fontweight='bold', fontsize=14)
    for bar, val in zip(bars1, [avg_torch, avg_triton]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f"{val:.1f} t/s", ha='center', va='bottom', fontweight='bold', fontsize=11)
    ax1.set_ylim(0, max(avg_torch, avg_triton) * 1.3)
    ax1.grid(axis='y', alpha=0.3)
    
    # VRAM bar chart
    ax2 = axes[1]
    bars2 = ax2.bar(labels, [mem_torch, mem_triton], color=colors, width=0.6, edgecolor='white', linewidth=1.5)
    ax2.set_ylabel('Peak VRAM (GB)', fontweight='bold', fontsize=12)
    ax2.set_title('Memory Usage', fontweight='bold', fontsize=14)
    for bar, val in zip(bars2, [mem_torch, mem_triton]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                f"{val:.2f} GB", ha='center', va='bottom', fontweight='bold', fontsize=11)
    ax2.set_ylim(0, max(mem_torch, mem_triton) * 1.3)
    ax2.grid(axis='y', alpha=0.3)
    
    fig.suptitle(f'Performance: Triton vs PyTorch ({speedup:.2f}x speedup)\nQwen2.5-7B-Instruct-AWQ on RTX 4080', 
                 fontweight='bold', fontsize=15, y=1.02)
    fig.tight_layout()
    plt.savefig('test/benchmark_chart.png', dpi=300, bbox_inches='tight')
    print("\nBenchmark chart saved to: test/benchmark_chart.png")
    
    print("\n--- Output Comparison ---")
    print(f"PyTorch: {out_torch[:120]}...")
    print(f"Triton:  {out_triton[:120]}...")


if __name__ == "__main__":
    main()
