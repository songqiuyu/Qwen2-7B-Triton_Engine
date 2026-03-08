import sys
import torch
sys.path.append('.')
from model.loader import load_weights
from model.config import Qwen2Config

model_dir = "models/qwen/Qwen2___5-7B-Instruct-AWQ"

# We just want to inspect the structure of the attention layer 0 parameters
weights = load_weights(model_dir, device="cpu")
for key, tensor in weights:
    if "layers.0" in key:
        print(f"{key}: {tensor.shape} ({tensor.dtype})")
