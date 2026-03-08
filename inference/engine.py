import torch
from transformers import AutoTokenizer
import time
from typing import Optional, List, Dict, Any

from inference.kv_cache import KVCacheManager
# from model.qwen2 import Qwen2ForCausalLM
# We will tie the engine once model loading script is injected 

class TritonInferenceEngine:
    def __init__(self, model: torch.nn.Module, tokenizer, max_seq_len: int = 4096, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_seq_len = max_seq_len
        
        # We assume single batch inference for this custom engine
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

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.7, top_p: float = 0.9, print_stream=True):
        self.model.eval()
        self.kv_manager.reset()
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device) # (1, seq_len)
        seq_len = input_ids.shape[1]
        
        if seq_len + max_new_tokens > self.max_seq_len:
            raise ValueError("Prompt + max_new_tokens exceeds KV cache max_seq_len")
            
        print(f"\n[Prefill Stage] Prompt length: {seq_len} tokens")
        start_time = time.time()
        
        # Track output tokens
        generated_tokens = []
        
        # --- Prefill Phase ---
        # 1st Pass: input all prompt tokens
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=self.device).unsqueeze(0)
        
        logits = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            kv_caches=self.kv_manager
        )
        # Advance KV manager tracking
        self.kv_manager.advance(seq_len)
        
        # Get next token
        next_token_logits = logits[:, -1, :] # (1, vocab_size)
        next_token = self._sample(next_token_logits, temperature, top_p)
        generated_tokens.append(next_token.item())
        
        if print_stream:
            print(self.tokenizer.decode([next_token.item()]), end="", flush=True)
            
        prefill_time = time.time() - start_time
        print(f"\nPrefill latency: {prefill_time:.3f} s ({seq_len / prefill_time:.1f} tokens/s context processing)")
        
        # --- Decode Phase ---
        decode_start_time = time.time()
        
        # Current token to feed in autoregressively
        cur_token_id = next_token.unsqueeze(0) # (1, 1)
        
        for _ in range(max_new_tokens - 1):
            # Position is just the current sequence length
            position_ids = torch.tensor([[self.kv_manager.current_seq_len]], dtype=torch.long, device=self.device)
            
            logits = self.model(
                input_ids=cur_token_id,
                position_ids=position_ids,
                kv_caches=self.kv_manager
            )
            # Advance exactly 1 step
            self.kv_manager.advance(1)
            
            next_token_logits = logits[:, -1, :]
            next_token = self._sample(next_token_logits, temperature, top_p)
            
            token_val = next_token.item()
            generated_tokens.append(token_val)
            
            # Stop condition
            if token_val == self.tokenizer.eos_token_id:
                break
                
            if print_stream:
                print(self.tokenizer.decode([token_val]), end="", flush=True)
                
            cur_token_id = next_token.unsqueeze(0)
            
        decode_time = time.time() - decode_start_time
        num_generated = len(generated_tokens)
        
        if print_stream:
            print(f"\n\n[Decode Stage] Generated {num_generated} tokens in {decode_time:.3f} s ({num_generated / decode_time:.1f} tokens/s)")
            
        return self.tokenizer.decode(generated_tokens)

    def _sample(self, logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
        """Sample next token from logits."""
        # Simple greedy if temp == 0
        if temperature <= 1e-5:
            return torch.argmax(logits, dim=-1)
            
        # Optional: Implement Top-P sampling. For now, fallback to greedy or simplistic categorical
        # A simple implementation:
        probs = torch.softmax(logits / temperature, dim=-1)
        
        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # Scatter back
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            probs[indices_to_remove] = 0.0
            probs = probs / probs.sum(dim=-1, keepdim=True)
            
        next_token = torch.multinomial(probs, num_samples=1)
        return next_token.squeeze(1)

