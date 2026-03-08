import torch
from transformers import AutoTokenizer
import time
from typing import Optional, List, Dict, Any

from inference.kv_cache import KVCacheManager


class TritonInferenceEngine:
    def __init__(self, model: torch.nn.Module, tokenizer, max_seq_len: int = 4096, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_seq_len = max_seq_len
        self.batch_size = 1 
        
        self.kv_manager = KVCacheManager(
            num_layers=model.config.num_hidden_layers,
            num_kv_heads=model.config.num_key_value_heads,
            head_dim=model.config.hidden_size // model.config.num_attention_heads,
            max_seq_len=max_seq_len,
            batch_size=self.batch_size,
            dtype=torch.float16,
            device=device
        )
        
        # Pre-allocate decode-phase tensors to avoid per-step allocation
        self._decode_input_ids = torch.zeros((1, 1), dtype=torch.long, device=device)
        self._decode_position_ids = torch.zeros((1, 1), dtype=torch.long, device=device)

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.7, top_p: float = 0.9, print_stream=True):
        self.model.eval()
        self.kv_manager.reset()
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)
        seq_len = input_ids.shape[1]
        
        if seq_len + max_new_tokens > self.max_seq_len:
            raise ValueError("Prompt + max_new_tokens exceeds KV cache max_seq_len")
            
        if print_stream:
            print(f"\n[Prefill Stage] Prompt length: {seq_len} tokens")
        start_time = time.time()
        
        generated_tokens = []
        
        # --- Prefill Phase ---
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=self.device).unsqueeze(0)
        
        logits = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            kv_caches=self.kv_manager
        )
        self.kv_manager.advance(seq_len)
        
        next_token_logits = logits[:, -1, :]
        next_token = self._sample(next_token_logits, temperature, top_p)
        generated_tokens.append(next_token.item())
        
        if print_stream:
            print(self.tokenizer.decode([next_token.item()]), end="", flush=True)
            
        prefill_time = time.time() - start_time
        if print_stream:
            print(f"\nPrefill latency: {prefill_time:.3f} s ({seq_len / prefill_time:.1f} tokens/s context processing)")
        
        # --- Decode Phase ---
        decode_start_time = time.time()

        cur_token_val = next_token.item()
        
        for _ in range(max_new_tokens - 1):
            # Use pre-allocated tensors to avoid per-step allocation
            self._decode_input_ids[0, 0] = cur_token_val
            self._decode_position_ids[0, 0] = self.kv_manager.current_seq_len
            
            logits = self.model(
                input_ids=self._decode_input_ids,
                position_ids=self._decode_position_ids,
                kv_caches=self.kv_manager
            )
            self.kv_manager.advance(1)
            
            next_token_logits = logits[:, -1, :]
            next_token = self._sample(next_token_logits, temperature, top_p)
            
            token_val = next_token.item()
            generated_tokens.append(token_val)
            
            if token_val == self.tokenizer.eos_token_id:
                break
                
            if print_stream:
                print(self.tokenizer.decode([token_val]), end="", flush=True)
                
            cur_token_val = token_val
            
        decode_time = time.time() - decode_start_time
        num_generated = len(generated_tokens)
        
        if print_stream:
            print(f"\n\n[Decode Stage] Generated {num_generated} tokens in {decode_time:.3f} s ({num_generated / decode_time:.1f} tokens/s)")
            
        return self.tokenizer.decode(generated_tokens)

    def _sample(self, logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
        """Sample next token from logits."""
        if temperature <= 1e-5:
            return torch.argmax(logits, dim=-1)
            
        probs = torch.softmax(logits / temperature, dim=-1)
        
        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            probs[indices_to_remove] = 0.0
            probs = probs / probs.sum(dim=-1, keepdim=True)
            
        next_token = torch.multinomial(probs, num_samples=1)
        return next_token.squeeze(1)
