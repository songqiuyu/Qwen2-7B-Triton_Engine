import torch
import sys
import time
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('.')
from transformers import AutoTokenizer
from model.loader import load_config, load_weights
from model.config import Qwen2Config
from model.qwen2 import Qwen2ForCausalLM
from inference.engine import TritonInferenceEngine

# To test baseline, we'll patch the Triton kernels back to pure PyTorch inside the script.
import kernels.rmsnorm
import kernels.rope
import kernels.silu_mul

# Pure PyTorch Implementations
def torch_rmsnorm(x, weight, eps):
    input_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return weight * x.to(input_dtype)

def torch_apply_rope(q, k, cos, sin, position_ids):
    # PyTorch non-inplace RoPE
    # Standard implementation allocating new memory
    head_dim = q.shape[-1]
    half_dim = head_dim // 2
    
    def rotate_half(x):
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]
        return torch.cat((-x2, x1), dim=-1)
        
    seq_len = q.shape[1]
    cos_seq = cos[:seq_len, :].unsqueeze(0).unsqueeze(2) # (1, seq, 1, dim)
    sin_seq = sin[:seq_len, :].unsqueeze(0).unsqueeze(2)
    
    q_out = (q * cos_seq) + (rotate_half(q) * sin_seq)
    k_out = (k * cos_seq) + (rotate_half(k) * sin_seq)
    
    q.copy_(q_out)
    k.copy_(k_out)
    return q, k

def torch_silu_mul(gate, up):
    return torch.nn.functional.silu(gate) * up

def run_benchmark(model, tokenizer, use_triton=True):
    if use_triton:
        # Restore Triton
        kernels.rmsnorm.rmsnorm_forward = _original_rmsnorm
        kernels.rope.apply_rope_inplace = _original_rope
        kernels.silu_mul.silu_mul_forward = _original_silu_mul
        mode = "Triton + Optimizations"
    else:
        # Patch to PyTorch
        kernels.rmsnorm.rmsnorm_forward = torch_rmsnorm
        kernels.rope.apply_rope_inplace = torch_apply_rope
        kernels.silu_mul.silu_mul_forward = torch_silu_mul
        mode = "Pure PyTorch Baseline"

    engine = TritonInferenceEngine(model, tokenizer, max_seq_len=2048, device="cuda")
    prompt = "<|im_start|>user\n你好，请介绍一下西安<|im_end|>\n<|im_start|>assistant\n"
    
    print(f"\n[{mode}] Warming up...")
    engine.generate(prompt, max_new_tokens=5, print_stream=False)
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    print(f"\n[{mode}] Generating...")
    torch.cuda.synchronize()
    start_t = time.time()
    
    out_text = engine.generate(prompt, max_new_tokens=150, temperature=0.7, print_stream=False)
    
    torch.cuda.synchronize()
    total_time = time.time() - start_t
    max_mem = torch.cuda.max_memory_allocated() / 1024**3
    
    # Approx tokens generated
    gen_tokens = len(tokenizer.encode(out_text)) - len(tokenizer.encode(prompt))
    if gen_tokens <= 0:
        gen_tokens = 150 # Fallback 
        
    tps = gen_tokens / total_time
    print(f"Output preview: {out_text[:60]}...")
    print(f"Time: {total_time:.2f}s | Speed: {tps:.2f} tokens/s | Peak VRAM: {max_mem:.2f} GB")
    
    return tps, max_mem, out_text

def main():
    model_dir = "models/qwen/Qwen2___5-7B-Instruct-AWQ"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    config_dict = load_config(model_dir)
    config = Qwen2Config(**{k: v for k, v in config_dict.items() if hasattr(Qwen2Config, k)})
    model = Qwen2ForCausalLM(config).to("cuda", dtype=torch.float16)
    
    weights_gen = load_weights(model_dir, device="cuda")
    model.load_awq_weights(weights_gen)
    
    # Save original tritons
    global _original_rmsnorm, _original_rope, _original_silu_mul
    _original_rmsnorm = kernels.rmsnorm.rmsnorm_forward
    _original_rope = kernels.rope.apply_rope_inplace
    _original_silu_mul = kernels.silu_mul.silu_mul_forward

    print("\n" + "="*50)
    print("STARTING BENCHMARK: TRITON VS PYTORCH")
    print("="*50)
    
    tps_torch, mem_torch, out_torch = run_benchmark(model, tokenizer, use_triton=False)
    tps_triton, mem_triton, out_triton = run_benchmark(model, tokenizer, use_triton=True)
    
    # Plot results
    labels = ['Pure PyTorch', 'Triton Optimized']
    tps = [tps_torch, tps_triton]
    mem = [mem_torch, mem_triton]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(8, 6))
    
    color = 'tab:blue'
    ax1.set_ylabel('Throughput (Tokens/s)', color=color, fontweight='bold')
    bars1 = ax1.bar(x - width/2, tps, width, color=color, label='Tokens/s')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Peak VRAM (GB)', color=color, fontweight='bold')
    bars2 = ax2.bar(x + width/2, mem, width, color=color, label='VRAM (GB)')
    ax2.tick_params(axis='y', labelcolor=color)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontweight='bold')
    plt.title('Performance Comparison: Triton Engine vs PyTorch Baseline\n(Qwen2.5-7B-Instruct-AWQ)', pad=20, fontweight='bold')
    
    # Add values on top of bars
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f"{yval:.1f} t/s", ha='center', va='bottom', fontweight='bold')
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f"{yval:.2f} GB", ha='center', va='bottom', fontweight='bold')
        
    fig.tight_layout()
    plt.savefig('test/benchmark_chart.png', dpi=300)
    print("\nBenchmark chart saved to: test/benchmark_chart.png")
    
    print("\n--- Final Outputs Check ---")
    print(f"PyTorch Output: {out_torch[:100]}...")
    print(f"Triton Output:  {out_triton[:100]}...")

if __name__ == "__main__":
    main()
