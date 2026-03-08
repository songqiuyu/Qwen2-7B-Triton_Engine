from dataclasses import dataclass
from typing import Optional

@dataclass
class Qwen2Config:
    vocab_size: int = 152064
    hidden_size: int = 3584
    intermediate_size: int = 18944
    num_hidden_layers: int = 28
    num_attention_heads: int = 28
    num_key_value_heads: int = 4 # GQA
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    tie_word_embeddings: bool = False

    # quant information
    quant_method: str = "awq" # AWQ或者GPTQ
    weight_bits: int = 4
    group_size: int = 128

    max_batch_size: int = 1
    max_seq_len: int = 4096

    def __post_init__(self):
        assert self.hidden_size % self.num_attention_heads == 0
        self.head_dim = self.hidden_size // self.num_attention_heads

        # the group of GQA
        self.num_attention_groups = self.num_attention_heads // self.num_key_value_heads