import sys
sys.path.append('.')
from model.loader import load_weights
from model.config import Qwen2Config
from model.qwen2 import Qwen2ForCausalLM

model_dir = "models/qwen/Qwen2___5-7B-Instruct-AWQ"
weights_gen = load_weights(model_dir, device="cpu")

config = Qwen2Config()
model = Qwen2ForCausalLM(config)
model_keys = set(model.state_dict().keys())

safetensors_keys = set([name for name, _ in weights_gen])

uninitialized = model_keys - safetensors_keys
print(f"Uninitialized parameters ({len(uninitialized)}):")
for k in sorted(list(uninitialized)):
    print(k)

# Also check tie_word_embeddings
print("Config tie_word_embeddings:", config.tie_word_embeddings)
