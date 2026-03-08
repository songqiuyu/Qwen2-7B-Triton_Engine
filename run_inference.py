import torch
import sys
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
    
    print("Loading safetensors weights (this will stream sequentially to save RAM)...")
    weights_gen = load_weights(model_dir, device="cuda")
    model.load_awq_weights(weights_gen)
    
    print("Creating Triton Inference Engine...")
    engine = TritonInferenceEngine(model, tokenizer, max_seq_len=2048, device="cuda")
    
    print("\n" + "="*50)
    prompt = "<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n"
    print(f"Prompt: {prompt}")
    print("="*50 + "\n")
    
    engine.generate(prompt, max_new_tokens=256, temperature=0.7)

if __name__ == "__main__":
    main()
