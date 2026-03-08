import torch
from typing import Tuple

class KVCacheManager:
    """
    Manages static pre-allocated KV Cache for the Qwen2 attention layers.
    This avoids dynamic tensor concatenation which causes memory fragmentation and overhead.
    """
    def __init__(self, num_layers: int, num_kv_heads: int, head_dim: int, max_seq_len: int, batch_size: int = 1, dtype=torch.float16, device="cuda"):
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        
        # We pre-allocate maximum required memory
        # Shape: (num_layers, batch_size, max_seq_len, num_kv_heads, head_dim)
        # Using [batch, seq_len, head, dim] format matching typical Triton implementations
        self.k_cache = torch.zeros(
            (num_layers, batch_size, max_seq_len, num_kv_heads, head_dim),
            dtype=dtype,
            device=device
        )
        self.v_cache = torch.zeros(
            (num_layers, batch_size, max_seq_len, num_kv_heads, head_dim),
            dtype=dtype,
            device=device
        )
        
        # Track current length
        self.current_seq_len = 0

    def reset(self):
        """Reset sequence length counter for a new generation"""
        self.current_seq_len = 0

    def update(self, layer_idx: int, k_state: torch.Tensor, v_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the KV cache with new token states for a specific layer.
        
        Args:
            layer_idx: integer representing the transformer layer block
            k_state: (batch_size, seq_len, num_kv_heads, head_dim)
            v_state: (batch_size, seq_len, num_kv_heads, head_dim)
            
        Returns:
            The full Key and Value cache up to the current sequence length.
        """
        batch_size, seq_len, num_kv_heads, head_dim = k_state.shape
        start_pos = self.current_seq_len
        end_pos = start_pos + seq_len
        
        if end_pos > self.max_seq_len:
            raise ValueError(f"Exceeded maximum sequence length of {self.max_seq_len}")
        
        # Inject new shapes
        self.k_cache[layer_idx, :, start_pos:end_pos, :, :] = k_state
        self.v_cache[layer_idx, :, start_pos:end_pos, :, :] = v_state
        
        # Note: We don't advance self.current_seq_len here because this is called per layer.
        # The Engine should call `advance(seq_len)` after all layers complete a discrete step.

        # Return a view of the cache up to the end position
        # shape: (batch_size, current_total_len, num_kv_heads, head_dim)
        return self.k_cache[layer_idx, :, :end_pos, :, :], self.v_cache[layer_idx, :, :end_pos, :, :]
        
    def advance(self, seq_len: int):
        """Advance the sequence length step after a full forward pass"""
        self.current_seq_len += seq_len
