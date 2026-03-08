import sys
sys.path.append('.')
from model.loader import load_config, load_weights
from model.config import Qwen2Config

model_dir = "models/qwen/Qwen2___5-7B-Instruct-AWQ"
config_dict = load_config(model_dir)
config = Qwen2Config(**{k: v for k, v in config_dict.items() if hasattr(Qwen2Config, k)})
print(f"Loaded config: hidden_size={config.hidden_size}, layers={config.num_hidden_layers}")

# Test yielding first weight
weights = load_weights(model_dir, device="cpu")
first_key, first_tensor = next(weights)
print(f"First weight layer: {first_key}, shape: {first_tensor.shape}, dtype: {first_tensor.dtype}")
