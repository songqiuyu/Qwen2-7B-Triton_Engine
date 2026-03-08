import os
import json
import glob
from safetensors import safe_open
from typing import Dict, Any

def load_config(model_dir: str) -> Dict[str, Any]:
    """Load configuration from config.json"""
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def find_safetensor_files(model_dir: str) -> list[str]:
    """Find all safetensor files in the directory"""
    files = glob.glob(os.path.join(model_dir, "*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No .safetensors files found in {model_dir}")
    return sorted(files)

def load_weights(model_dir: str, device: str = "cuda:0"):
    """
    Generator that yields (tensor_name, tensor) by reading all safetensors files.
    This helps in loading weights one by one to avoid CPU RAM spikes.
    """
    files = find_safetensor_files(model_dir)
    print(f"Found {len(files)} safetensors files.")
    
    for file_path in files:
        # print(f"Loading weights from {os.path.basename(file_path)}...")
        with safe_open(file_path, framework="pt", device=device) as f:
            for key in f.keys():
                yield key, f.get_tensor(key)

