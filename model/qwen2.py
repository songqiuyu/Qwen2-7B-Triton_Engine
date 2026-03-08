import torch
import torch.nn as nn
from typing import Optional, Tuple
from model.config import Qwen2Config
# Module-level imports: functions are called via module reference
# so that benchmark monkeypatching (kernels.xxx.func = new_func) works correctly.
import kernels.rmsnorm
import kernels.fused_add_rmsnorm
import kernels.rope
import kernels.awq_gemm
import kernels.silu_mul
import kernels.flash_attention
import math

class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return kernels.rmsnorm.rmsnorm_forward(hidden_states, self.weight, self.variance_epsilon)

class LinearAWQ(nn.Module):
    """
    Custom module to hold the packed AWQ weights and biases and dispatch to our Triton GEMM kernel.
    """
    def __init__(self, in_features, out_features, group_size=128, has_bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        
        self.register_buffer("qweight", torch.zeros((in_features, out_features // 8), dtype=torch.int32))
        self.register_buffer("qzeros", torch.zeros((in_features // group_size, out_features // 8), dtype=torch.int32))
        self.register_buffer("scales", torch.zeros((in_features // group_size, out_features), dtype=torch.float16))
        
        if has_bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float16))
        else:
            self.bias = None

    def forward(self, x):
        out = kernels.awq_gemm.awq_gemm_forward(x, self.qweight, self.qzeros, self.scales, self.group_size)
        if self.bias is not None:
            out = out + self.bias
        return out

class Qwen2Attention(nn.Module):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_key_value_heads * self.head_dim

        # AWQ Quantized Projections (individual, for weight loading)
        self.q_proj = LinearAWQ(self.hidden_size, self.q_size, group_size=config.group_size, has_bias=True)
        self.k_proj = LinearAWQ(self.hidden_size, self.kv_size, group_size=config.group_size, has_bias=True)
        self.v_proj = LinearAWQ(self.hidden_size, self.kv_size, group_size=config.group_size, has_bias=True)
        self.o_proj = LinearAWQ(self.q_size, self.hidden_size, group_size=config.group_size, has_bias=False)
        
        # Fused QKV weights (created post-loading)
        self._qkv_qweight = None
        self._qkv_qzeros = None
        self._qkv_scales = None
        self._qkv_bias = None
        
        # Precomputed RoPE frequencies lazily initialized
        self.cos_cached = None
        self.sin_cached = None

    def fuse_qkv(self):
        """Concatenate Q/K/V weights into single projection. Call after weight loading."""
        self._qkv_qweight = torch.cat([self.q_proj.qweight, self.k_proj.qweight, self.v_proj.qweight], dim=1)
        self._qkv_qzeros = torch.cat([self.q_proj.qzeros, self.k_proj.qzeros, self.v_proj.qzeros], dim=1)
        self._qkv_scales = torch.cat([self.q_proj.scales, self.k_proj.scales, self.v_proj.scales], dim=1)
        if self.q_proj.bias is not None:
            self._qkv_bias = torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias], dim=0)

    def _update_rope_cache(self, device):
        if self.cos_cached is None:
            self.cos_cached, self.sin_cached = kernels.rope.precompute_freqs_cis(
                self.head_dim, self.config.max_seq_len, theta=self.config.rope_theta, device=device
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        kv_caches = None,
    ):
        batch_size, seq_len, _ = hidden_states.shape
        self._update_rope_cache(hidden_states.device)

        if self._qkv_qweight is not None:
            # === Fused QKV path: ONE kernel call instead of THREE ===
            qkv = kernels.awq_gemm.awq_gemm_forward(
                hidden_states, self._qkv_qweight, self._qkv_qzeros, 
                self._qkv_scales, self.q_proj.group_size
            )
            if self._qkv_bias is not None:
                qkv = qkv + self._qkv_bias
            
            q = qkv[..., :self.q_size]
            k = qkv[..., self.q_size:self.q_size + self.kv_size]
            v = qkv[..., self.q_size + self.kv_size:]
        else:
            # Fallback: separate projections
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)

        # Apply RoPE using vectorized Triton Kernel (In-place)
        kernels.rope.apply_rope_inplace(q, k, self.cos_cached, self.sin_cached, position_ids)

        # Handle KV Cache updates
        if kv_caches is not None:
            k, v = kv_caches.update(self.layer_idx, k, v)
        
        # GQA expand: use expand+reshape instead of repeat_interleave (zero-copy)
        if self.num_key_value_groups > 1:
            k = k.unsqueeze(3).expand(
                batch_size, k.shape[1], self.num_key_value_heads, self.num_key_value_groups, self.head_dim
            ).reshape(batch_size, k.shape[1], self.num_heads, self.head_dim)
            v = v.unsqueeze(3).expand(
                batch_size, v.shape[1], self.num_key_value_heads, self.num_key_value_groups, self.head_dim
            ).reshape(batch_size, v.shape[1], self.num_heads, self.head_dim)

        # Transpose to (batch, num_heads, seq_len, head_dim) 
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        is_causal = seq_len > 1
        
        # Triton FlashAttention-2
        attn_output = kernels.flash_attention.flash_attention_forward(
            q, k, v,
            is_causal=is_causal
        )
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        
        output = self.o_proj(attn_output)
        return output

class Qwen2MLP(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.config = config
        self.intermediate_size = config.intermediate_size
        
        self.gate_proj = LinearAWQ(config.hidden_size, config.intermediate_size, group_size=config.group_size, has_bias=False)
        self.up_proj = LinearAWQ(config.hidden_size, config.intermediate_size, group_size=config.group_size, has_bias=False)
        self.down_proj = LinearAWQ(config.intermediate_size, config.hidden_size, group_size=config.group_size, has_bias=False)
        
        # Fused gate+up weights (created post-loading)
        self._gate_up_qweight = None
        self._gate_up_qzeros = None
        self._gate_up_scales = None

    def fuse_gate_up(self):
        """Concatenate gate+up weights into single projection. Call after weight loading."""
        self._gate_up_qweight = torch.cat([self.gate_proj.qweight, self.up_proj.qweight], dim=1)
        self._gate_up_qzeros = torch.cat([self.gate_proj.qzeros, self.up_proj.qzeros], dim=1)
        self._gate_up_scales = torch.cat([self.gate_proj.scales, self.up_proj.scales], dim=1)

    def forward(self, x):
        if self._gate_up_qweight is not None:
            # === Fused gate+up path: ONE kernel call instead of TWO ===
            gate_up = kernels.awq_gemm.awq_gemm_forward(
                x, self._gate_up_qweight, self._gate_up_qzeros,
                self._gate_up_scales, self.gate_proj.group_size
            )
            gate = gate_up[..., :self.intermediate_size].contiguous()
            up = gate_up[..., self.intermediate_size:].contiguous()
        else:
            gate = self.gate_proj(x)
            up = self.up_proj(x)
        
        # Fused SwiGLU: SiLU(gate) * up via Triton kernel
        intermediate = kernels.silu_mul.silu_mul_forward(gate, up)
        
        down = self.down_proj(intermediate)
        return down

class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.self_attn = Qwen2Attention(config, layer_idx)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        kv_caches = None,
    ):
        # === Attention block ===
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_ids, kv_caches)
        
        # === Fused residual-add + RMSNorm for MLP block ===
        hidden_states = kernels.fused_add_rmsnorm.fused_add_rmsnorm_forward(
            residual, hidden_states,
            self.post_attention_layernorm.weight,
            self.post_attention_layernorm.variance_epsilon
        )
        
        # MLP block
        mlp_out = self.mlp(hidden_states)
        hidden_states = residual + mlp_out

        return hidden_states

class Qwen2Model(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen2DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor, kv_caches=None):
        hidden_states = self.embed_tokens(input_ids)
        
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_ids, kv_caches)
            
        hidden_states = self.norm(hidden_states)
        return hidden_states

class Qwen2ForCausalLM(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.config = config
        self.model = Qwen2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def load_awq_weights(self, weights_generator):
        """
        Loads the AWQ safetensors into the module buffers sequentially,
        then fuses QKV and gate+up projections for faster decode.
        """
        print("Mapping quantized weights to engine buffers...")
        state_dict = self.state_dict()
        
        matched = 0
        for name, tensor in weights_generator:
            if name in state_dict:
                state_dict[name].copy_(tensor)
                matched += 1
                
        print(f"Loaded {matched} weight tensors.")
        
        # Fuse projections for faster inference
        print("Fusing QKV and Gate+Up projections...")
        for layer in self.model.layers:
            layer.self_attn.fuse_qkv()
            layer.mlp.fuse_gate_up()
        print("Fusion complete.")

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor, kv_caches=None):
        hidden_states = self.model(input_ids, position_ids, kv_caches)
        logits = self.lm_head(hidden_states)
        return logits
