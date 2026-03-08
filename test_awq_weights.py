import sys
sys.path.append('.')
from model.loader import load_weights

model_dir = "models/qwen/Qwen2___5-7B-Instruct-AWQ"
weights = load_weights(model_dir, device="cpu")

# Print first 20 weight shapes to see how AWQ linear layer looks
print("First 20 weight tensor shapes:")
for i, (key, tensor) in enumerate(weights):
    if i < 20:
        print(f"  {key}: {tensor.shape} ({tensor.dtype})")
    else:
        break
