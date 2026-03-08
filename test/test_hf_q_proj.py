import torch
import sys
sys.path.append('.')
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.loader import load_weights
from model.config import Qwen2Config
from kernels.awq_gemm import awq_gemm_forward

model_dir = "models/qwen/Qwen2___5-7B-Instruct-AWQ"

print("Loading reference HF model...")
hf_model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="cuda", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

prompt = "Hello"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    hidden = hf_model.model.embed_tokens(inputs.input_ids)
    print("HF Embeddings shape:", hidden.shape)
    print("HF Embeddings max:", hidden.max().item())

    # HF Q proj forward
    layer_0 = hf_model.model.layers[0]
    # To get just the Q projection output
    norm_hidden = layer_0.input_layernorm(hidden)
    
    # In HF Qwen2 with AWQ, the q_proj is likely a WQLinear or similar
    hf_q_out = layer_0.self_attn.q_proj(norm_hidden)
    print("HF q_proj out max:", hf_q_out.max().item())
    
    # -------------------------
    # Now our Custom Decoded
    # -------------------------
    print("\n--- Our Custom AWQ Dequantizer ---")
    weights_gen = load_weights(model_dir, device="cuda")
    w_dict = {k: v for k, v in weights_gen if "layers.0.self_attn.q_proj" in k}
    
    qweight = w_dict["model.layers.0.self_attn.q_proj.qweight"]
    qzeros = w_dict["model.layers.0.self_attn.q_proj.qzeros"]
    scales = w_dict["model.layers.0.self_attn.q_proj.scales"]
    bias = w_dict["model.layers.0.self_attn.q_proj.bias"]
    
    print("qweight shape:", qweight.shape)
    print("qzeros shape:", qzeros.shape)
    
    our_q_out = awq_gemm_forward(norm_hidden, qweight, qzeros, scales, group_size=128)
    our_q_out += bias
    
    print("Our q_proj out max:", our_q_out.max().item())
    diff = torch.abs(our_q_out - hf_q_out).max().item()
    print("Max Absolute Diff:", diff)
