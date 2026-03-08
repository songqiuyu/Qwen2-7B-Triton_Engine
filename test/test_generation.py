import torch
import sys
import time
sys.path.append('.')
from transformers import AutoTokenizer
from model.loader import load_config, load_weights
from model.config import Qwen2Config
from model.qwen2 import Qwen2ForCausalLM
from inference.engine import TritonInferenceEngine

def main():
    model_dir = "models/qwen/Qwen2___5-7B-Instruct-AWQ"
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    print("Loading config...")
    config_dict = load_config(model_dir)
    config = Qwen2Config(**{k: v for k, v in config_dict.items() if hasattr(Qwen2Config, k)})
    
    print("Initializing model architecture...")
    model = Qwen2ForCausalLM(config).to("cuda", dtype=torch.float16)
    
    print("Loading safetensors weights into PyTorch buffers...")
    start_t = time.time()
    weights_gen = load_weights(model_dir, device="cuda")
    model.load_awq_weights(weights_gen)
    print(f"Weight loading took: {time.time() - start_t:.2f} s")
    
    # ---------------------------------------------------------
    # Custom Backend (Triton RoPE + SiLU + pure PyTorch AWQ Fallback + SDPA)
    # ---------------------------------------------------------
    engine = TritonInferenceEngine(model, tokenizer, max_seq_len=2048, device="cuda")
    
    prompts = [
        "<|im_start|>user\nHello! How are you?<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWrite a quick Python script to calculate Fibonacci sequence.<|im_end|>\n<|im_start|>assistant\n"
    ]
    
    # Warmup
    print("Warming up JIT kernels...")
    engine.generate(prompts[0], max_new_tokens=10, print_stream=False)
    
    print("\n" + "="*50)
    print("BENCHMARKING GENERATION")
    print("="*50)
    
    for idx, prompt in enumerate(prompts):
        print(f"\n[Prompt {idx+1}]")
        torch.cuda.synchronize()
        start_t = time.time()
        
        # We handle timing internally inside generate, but let's track total here too
        engine.generate(prompt, max_new_tokens=256, temperature=0.7)
        
        torch.cuda.synchronize()
        total_time = time.time() - start_t
        print(f"\nTotal Wall Time: {total_time:.2f} s")
        # Ensure VRAM tracks
        print(f"Max VRAM Allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

if __name__ == "__main__":
    main()
