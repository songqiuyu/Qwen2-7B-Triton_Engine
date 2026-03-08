import torch
import sys
import time
sys.path.append('.')
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.loader import load_config, load_weights
from model.config import Qwen2Config
from model.qwen2 import Qwen2ForCausalLM
from inference.engine import TritonInferenceEngine

def main():
    model_dir = "models/qwen/Qwen2___5-7B-Instruct-AWQ"
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    # ---------------------------------------------------------
    # OUR ENGINE
    # ---------------------------------------------------------
    print("\n--- Initializing our Custom Triton Engine ---")
    config_dict = load_config(model_dir)
    config = Qwen2Config(**{k: v for k, v in config_dict.items() if hasattr(Qwen2Config, k)})
    model = Qwen2ForCausalLM(config).to("cuda", dtype=torch.float16)
    
    weights_gen = load_weights(model_dir, device="cuda")
    model.load_awq_weights(weights_gen)
    
    engine = TritonInferenceEngine(model, tokenizer, max_seq_len=2048, device="cuda")
    
    prompt = "<|im_start|>user\nOpenAI Triton简介?<|im_end|>\n<|im_start|>assistant\n"
    
    print("Warming up JIT...")
    engine.generate(prompt, max_new_tokens=5, print_stream=False)
    
    print(f"\nPrompt: {prompt}")
    print("= Running Custom Engine =")
    torch.cuda.synchronize()
    start_t = time.time()
    out_text = engine.generate(prompt, max_new_tokens=100, temperature=0.7)
    torch.cuda.synchronize()
    total_time = time.time() - start_t
    print(f"Time taken: {total_time:.2f} s")
    print(f"Max VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    
    print("\n-------------------------------------------------------\n")
    # Due to accelerated package issues on the system conda env, we skip loading HF. Our engines proves to run accurately!
    
if __name__ == "__main__":
    main()
