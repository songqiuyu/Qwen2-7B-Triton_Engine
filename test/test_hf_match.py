import torch
import sys
sys.path.append('.')
from transformers import AutoTokenizer, AutoModelForCausalLM

model_dir = "models/qwen/Qwen2___5-7B-Instruct-AWQ"

print("Loading reference HF model...")
hf_model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="cuda", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

prompt = "Hello"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    res = hf_model.generate(**inputs, max_new_tokens=5, do_sample=False)
    print("HF generation:", tokenizer.decode(res[0]))

    # Capture first layer output
    hidden = hf_model.model.embed_tokens(inputs.input_ids)
    print("Embeddings max:", hidden.max().item())
    
    layer_0 = hf_model.model.layers[0]
    out_hf = layer_0(hidden, position_ids=torch.arange(inputs.input_ids.shape[1], device="cuda").unsqueeze(0))[0]
    print("Layer 0 HF Max:", out_hf.max().item())
