"""Monkey-patching utility for TurboQuant 2-bit compressed KV cache inference.

Intercepts SarvamAttention and LlamaAttention modules to route autoregressive decode
steps through compressed_attention_decode, drastically reducing KV cache memory.
"""

from __future__ import annotations

from contextlib import contextmanager
import types
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from turboquant.attention import compressed_attention_decode
from turboquant.cache import TurboQuantCache
from turboquant.quantizer import DynamicKVQuantizer


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of the input tensor for RoPE."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Position Embedding to query and key tensors."""
    if cos.ndim == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot, k_rot


def _format_decode_attention_mask(
    mask: Optional[torch.Tensor],
    target_dtype: torch.dtype,
    target_device: torch.device,
) -> Optional[torch.Tensor]:
    """Format attention mask into 4D tensor suitable for compressed decode logits."""
    if mask is None:
        return None
    if mask.ndim == 2:
        if mask.dtype == torch.bool:
            out_mask = torch.zeros(
                (mask.shape[0], 1, 1, mask.shape[1]),
                dtype=target_dtype,
                device=target_device,
            )
            return out_mask.masked_fill(~mask[:, None, None, :], float("-inf"))
        elif torch.all((mask == 0) | (mask == 1)):
            out_mask = torch.zeros(
                (mask.shape[0], 1, 1, mask.shape[1]),
                dtype=target_dtype,
                device=target_device,
            )
            return out_mask.masked_fill(mask[:, None, None, :] == 0, float("-inf"))
        else:
            return mask[:, None, None, :].to(dtype=target_dtype, device=target_device)
    elif mask.ndim == 3:
        return mask[:, None, :, :].to(dtype=target_dtype, device=target_device)
    return mask.to(dtype=target_dtype, device=target_device)


def turboquant_attention_forward(
    self: nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[Any] = None,
    cache_position: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    *args: Any,
    **kwargs: Any,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Custom attention forward implementation maintaining a 2-bit compressed KV cache.

    - Prefill phase (S > 1): Evaluates attention via PyTorch SDPA and quantizes
      resulting prompt Keys and Values into TurboQuantCache.
    - Decode phase (S = 1): Quantizes the single new token's Keys and Values,
      appends to TurboQuantCache, and executes compressed_attention_decode.
    """
    if past_key_values is None and "past_key_value" in kwargs:
        past_key_values = kwargs["past_key_value"]
    if position_ids is None and "position_ids" in kwargs:
        position_ids = kwargs["position_ids"]
    if not output_attentions and kwargs.get("output_attentions", False):
        output_attentions = True
    if position_embeddings is None and "position_embeddings" in kwargs:
        position_embeddings = kwargs["position_embeddings"]

    input_shape = hidden_states.shape[:-1]
    head_dim = getattr(self, "head_dim", None)
    if head_dim is None:
        config = getattr(self, "config", None)
        if config is not None:
            head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        else:
            head_dim = 128

    # 1. Linear projections
    query_states = self.q_proj(hidden_states).view(*input_shape, -1, head_dim).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(*input_shape, -1, head_dim).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(*input_shape, -1, head_dim).transpose(1, 2)

    layer_idx = getattr(self, "layer_idx", 0)

    # 2. Rotary Position Embeddings (RoPE)
    if position_embeddings is None:
        rot = getattr(self, "rotary_emb", None)
        if rot is None and hasattr(self, "config"):
            try:
                from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

                if not hasattr(self, "_auto_rotary_emb") or self._auto_rotary_emb is None:
                    self._auto_rotary_emb = LlamaRotaryEmbedding(self.config).to(
                        device=hidden_states.device
                    )
                rot = self._auto_rotary_emb
            except Exception:
                rot = None

        if rot is not None:
            seq_len = query_states.shape[-2]
            past_seen = (
                past_key_values.get_seq_length(layer_idx)
                if past_key_values is not None
                else 0
            )
            if position_ids is None:
                position_ids = torch.arange(
                    past_seen, past_seen + seq_len, device=hidden_states.device
                ).unsqueeze(0)
            cos, sin = rot(value_states, position_ids=position_ids)
            position_embeddings = (cos, sin)

    if position_embeddings is not None:
        cos, sin = position_embeddings
        query_states, key_states = _apply_rotary_pos_emb(query_states, key_states, cos, sin)

    seq_len = query_states.shape[-2]

    # Resolve quantizer
    quantizer = getattr(self, "_turboquant_quantizer", None)
    if quantizer is None and past_key_values is not None:
        quantizer = getattr(past_key_values, "quantizer", None)
    if quantizer is None:
        quantizer = DynamicKVQuantizer(
            head_dim=head_dim,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        self._turboquant_quantizer = quantizer

    num_q_heads = query_states.shape[1]
    num_kv_heads = key_states.shape[1]
    gqa_ratio = num_q_heads // num_kv_heads

    # 3. Route based on sequence length S = Q.shape[-2]
    if seq_len > 1:
        # 📥 Prefill Phase (S > 1)
        past_seen = (
            past_key_values.get_seq_length(layer_idx)
            if past_key_values is not None and isinstance(past_key_values, TurboQuantCache)
            else 0
        )

        if past_seen > 0 and isinstance(past_key_values, TurboQuantCache):
            # Multi-token chunk attending to previously cached compressed chunks
            k_cache, v_cache = past_key_values.update(
                key_states, value_states, layer_idx=layer_idx, quantizer=quantizer
            )
            total_k = past_seen + seq_len

            if attention_mask is not None:
                formatted_mask = _format_decode_attention_mask(
                    attention_mask, query_states.dtype, query_states.device
                )
            else:
                formatted_mask = None

            if (
                formatted_mask is not None
                and formatted_mask.shape[-1] == total_k
                and formatted_mask.shape[-2] == seq_len
            ):
                chunk_mask = formatted_mask
            else:
                q_idx = torch.arange(seq_len, device=query_states.device)[:, None]
                k_idx = torch.arange(total_k, device=query_states.device)[None, :]
                chunk_mask = torch.where(
                    k_idx <= (past_seen + q_idx),
                    torch.tensor(0.0, dtype=query_states.dtype, device=query_states.device),
                    torch.tensor(float("-inf"), dtype=query_states.dtype, device=query_states.device),
                )[None, None, :, :]
                if formatted_mask is not None:
                    chunk_mask = chunk_mask + formatted_mask

            decode_res = compressed_attention_decode(
                query=query_states,
                k_cache=k_cache,
                v_cache=v_cache,
                quantizer=quantizer,
                attention_mask=chunk_mask,
                return_attention_weights=output_attentions,
            )

            if output_attentions:
                attn_output, attn_weights = decode_res
            else:
                attn_output = decode_res
                attn_weights = None
        else:
            k_sdpa = (
                torch.repeat_interleave(key_states, repeats=gqa_ratio, dim=1)
                if gqa_ratio > 1
                else key_states
            )
            v_sdpa = (
                torch.repeat_interleave(value_states, repeats=gqa_ratio, dim=1)
                if gqa_ratio > 1
                else value_states
            )

            is_causal = attention_mask is None
            attn_output = F.scaled_dot_product_attention(
                query_states,
                k_sdpa,
                v_sdpa,
                attn_mask=attention_mask,
                is_causal=is_causal,
            )

            # Store prompt K and V into TurboQuantCache
            if past_key_values is not None:
                if isinstance(past_key_values, TurboQuantCache):
                    past_key_values.update(
                        key_states, value_states, layer_idx=layer_idx, quantizer=quantizer
                    )
                elif hasattr(past_key_values, "update"):
                    past_key_values.update(key_states, value_states, layer_idx)

            attn_weights = None
    else:
        # ⏱️ Decode Phase (S == 1)
        if past_key_values is not None and isinstance(past_key_values, TurboQuantCache):
            # Quantize new token's K, V and append to TurboQuantCache
            k_cache, v_cache = past_key_values.update(
                key_states, value_states, layer_idx=layer_idx, quantizer=quantizer
            )

            # Format mask if present
            formatted_mask = _format_decode_attention_mask(
                attention_mask, query_states.dtype, query_states.device
            )

            # Execute compressed attention decode without de-rotating past keys
            decode_res = compressed_attention_decode(
                query=query_states,
                k_cache=k_cache,
                v_cache=v_cache,
                quantizer=quantizer,
                attention_mask=formatted_mask,
                return_attention_weights=output_attentions,
            )

            if output_attentions:
                attn_output, attn_weights = decode_res
            else:
                attn_output = decode_res
                attn_weights = None
        else:
            # Fallback if no TurboQuantCache provided
            if past_key_values is not None and hasattr(past_key_values, "update"):
                key_states, value_states = past_key_values.update(
                    key_states, value_states, layer_idx
                )
            k_sdpa = (
                torch.repeat_interleave(key_states, repeats=gqa_ratio, dim=1)
                if gqa_ratio > 1
                else key_states
            )
            v_sdpa = (
                torch.repeat_interleave(value_states, repeats=gqa_ratio, dim=1)
                if gqa_ratio > 1
                else value_states
            )
            attn_output = F.scaled_dot_product_attention(
                query_states,
                k_sdpa,
                v_sdpa,
                attn_mask=attention_mask,
            )
            attn_weights = None

    # 4. Output projection
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    return attn_output, attn_weights


def _is_attention_module(m: Any) -> bool:
    """Check if module has attention projection layers."""
    if not isinstance(m, nn.Module):
        return False
    return (
        hasattr(m, "q_proj")
        and hasattr(m, "k_proj")
        and hasattr(m, "v_proj")
        and hasattr(m, "o_proj")
    )


def _find_attention_modules(model: nn.Module) -> List[nn.Module]:
    """Find all self-attention modules within a model, layer, or attention block."""
    if _is_attention_module(model):
        return [model]

    attention_modules: List[nn.Module] = []
    seen = set()

    for m in model.modules():
        if hasattr(m, "self_attn") and _is_attention_module(m.self_attn):
            if id(m.self_attn) not in seen:
                seen.add(id(m.self_attn))
                attention_modules.append(m.self_attn)
        elif _is_attention_module(m) and m is not model:
            if id(m) not in seen:
                seen.add(id(m))
                attention_modules.append(m)

    return attention_modules


def apply_turboquant(
    model: nn.Module,
    quantizer: Optional[DynamicKVQuantizer] = None,
) -> nn.Module:
    """In-place swap of layer.self_attn.forward to maintain 2-bit compressed KV cache.

    Args:
        model: HuggingFace model, decoder layer, or attention module.
        quantizer: Optional DynamicKVQuantizer instance to use across attention layers.

    Returns:
        The patched model.
    """
    targets = _find_attention_modules(model)
    for attn in targets:
        if getattr(attn, "_is_turboquant_patched", False):
            continue
        attn._original_forward = attn.forward
        if quantizer is not None:
            attn._turboquant_quantizer = quantizer
        elif not hasattr(attn, "_turboquant_quantizer"):
            d_h = getattr(attn, "head_dim", 128)
            attn._turboquant_quantizer = DynamicKVQuantizer(head_dim=d_h)
        attn.forward = types.MethodType(turboquant_attention_forward, attn)
        attn._is_turboquant_patched = True
    return model


def restore_turboquant(model: nn.Module) -> nn.Module:
    """Restore original forward methods on patched attention modules.

    Args:
        model: Patched model or module.

    Returns:
        The restored model.
    """
    targets = _find_attention_modules(model)
    for attn in targets:
        if getattr(attn, "_is_turboquant_patched", False):
            attn.forward = attn._original_forward
            del attn._original_forward
            attn._is_turboquant_patched = False
            if hasattr(attn, "_turboquant_quantizer"):
                del attn._turboquant_quantizer
            if hasattr(attn, "_auto_rotary_emb"):
                del attn._auto_rotary_emb
    return model


@contextmanager
def turboquant_context(
    model: nn.Module,
    quantizer: Optional[DynamicKVQuantizer] = None,
):
    """Context manager that applies TurboQuant monkey patch on entry and restores on exit.

    Args:
        model: HuggingFace model or attention module to patch.
        quantizer: Optional DynamicKVQuantizer instance.
    """
    apply_turboquant(model, quantizer=quantizer)
    try:
        yield model
    finally:
        restore_turboquant(model)


__all__ = [
    "apply_turboquant",
    "restore_turboquant",
    "turboquant_attention_forward",
    "turboquant_context",
]
