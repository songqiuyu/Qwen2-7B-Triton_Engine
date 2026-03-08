import sys
sys.path.append('.')
from model.loader import load_weights
model_dir = "models/qwen/Qwen2___5-7B-Instruct-AWQ"
weights = load_weights(model_dir, device="cpu")
for k, v in weights:
    if "layers.0" in k and "proj" in k:
        print(f"{k}: {v.shape}")
