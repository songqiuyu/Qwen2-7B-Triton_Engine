import sys
sys.path.append('.')
from model.loader import load_weights

model_dir = "models/qwen/Qwen2___5-7B-Instruct-AWQ"
weights_gen = load_weights(model_dir, device="cpu")

print("Key shapes from safetensors:")
for k, v in weights_gen:
    if "embed_tokens" in k or "lm_head" in k:
        print(f"{k}: {v.shape} | {v.dtype}")
    if "layers.0.self_attn.q_proj" in k:
        print(f"{k}: {v.shape} | {v.dtype}")
    if "layers.0.mlp.gate_proj" in k:
        print(f"{k}: {v.shape} | {v.dtype}")

