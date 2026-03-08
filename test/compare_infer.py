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
    weights_gen = load_weights(model_dir, device="cuda")
    model.load_awq_weights(weights_gen)
    
    # Enable PyTorch RMSNorm fallback and compare
    # We will dynamically monkeypatch the model's forward passes for comparison if needed
    
    print("\n" + "="*50)
    prompt = "<|im_start|>user\n你好，请你介绍一下OpenAI Triton。<|im_end|>\n<|im_start|>assistant\n"
    print(f"Prompt: {prompt}")
    print("="*50 + "\n")
    
    # ---------------------------------------------------------
    # RUN 1: Custom engine (Currently using Triton RoPE + SiLU + pure PyTorch AWQ Fallback + PyTorch SDPA)
    # ---------------------------------------------------------
    engine = TritonInferenceEngine(model, tokenizer, max_seq_len=2048, device="cuda")
    
    print("--- Running Custom Inference Engine (PyTorch Fallback AWQ + Triton RoPE/SiLU) ---")
    torch.cuda.synchronize()
    start_t = time.time()
    out_text = engine.generate(prompt, max_new_tokens=100, temperature=0.7)
    torch.cuda.synchronize()
    total_time = time.time() - start_t
    print(f"\nTime taken: {total_time:.2f} s")
    print(f"Final Output:\n{out_text}")
    print("-------------------------------------------------------\n")

if __name__ == "__main__":
    main()
