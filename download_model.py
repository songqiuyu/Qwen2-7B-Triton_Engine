import os
try:
    from modelscope import snapshot_download
except ImportError:
    print("Please install modelscope first: pip install modelscope")
    exit(1)

def main():
    model_id = 'qwen/Qwen2.5-7B-Instruct-AWQ'
    cache_dir = './models'
    
    print(f"Starting to download {model_id} from ModelScope...")
    print("This might take a while depending on your network speed (approx. 4.5 GB).")
    
    model_dir = snapshot_download(model_id, cache_dir=cache_dir)
    print(f"\nModel successfully downloaded to: {model_dir}")

if __name__ == "__main__":
    main()
