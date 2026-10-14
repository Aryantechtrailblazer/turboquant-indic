"""Compressed attention decoding for LLM inference using TurboQuant.

Exploits orthogonal rotational invariance (R @ R.T == I) to evaluate Grouped-Query Attention (GQA)
directly in rotated space against compressed 2-bit Keys without de-rotating them.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple, Union
import torch

from turboquant.packing import unpack_2bit
from turboquant.quantizer import DynamicKVQuantizer, QJLData, QuantizedKVTensor

# Default 2-bit quantization centroids for range [-1.0, 1.0]
DEFAULT_CENTROIDS = torch.tensor([-0.75, -0.25, 0.25, 0.75], dtype=torch.float32)


def _unpack_kv_cache(
    cache: Union[QuantizedKVTensor, Dict[str, Any], Tuple[Any, ...]],
) -> Tuple[torch.Tensor, torch.Tensor, Optional[Any]]:
    """Extract packed_codes, scales, and qjl_data from various container formats."""
    if isinstance(cache, dict):
        return cache["packed_codes"], cache["scales"], cache.get("qjl_data")
    elif hasattr(cache, "packed_codes"):
        return cache.packed_codes, cache.scales, cache.qjl_data
    elif isinstance(cache, (tuple, list)):
        if len(cache) == 3:
            return cache[0], cache[1], cache[2]
        elif len(cache) == 2:
            return cache[0], cache[1], None
    raise ValueError(f"Unsupported cache container format: {type(cache)}")


def _extract_qjl_components(
    qjl_data: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract signs and scale from QJLData, dict, or tuple."""
    if isinstance(qjl_data, dict):
        return qjl_data["signs"], qjl_data["scale"]
    elif hasattr(qjl_data, "signs"):
        return qjl_data.signs, qjl_data.scale
    elif isinstance(qjl_data, (tuple, list)):
        return qjl_data[0], qjl_data[1]
    raise TypeError(f"Unsupported qjl_data format: {type(qjl_data)}")


def compressed_attention_decode(
    query: torch.Tensor,
    k_cache: Union[QuantizedKVTensor, Dict[str, Any], Tuple[Any, ...], torch.Tensor],
    v_cache: Optional[Union[QuantizedKVTensor, Dict[str, Any], Tuple[Any, ...], torch.Tensor]] = None,
    rotation_matrix: Optional[torch.Tensor] = None,
    quantizer: Optional[DynamicKVQuantizer] = None,
    centroids: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    attention_mask: Optional[torch.Tensor] = None,
    return_attention_weights: bool = False,
    *args: Any,
    **kwargs: Any,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Evaluate Grouped-Query Attention (GQA) directly against compressed Key-Value caches.

    Exploits orthogonal rotational invariance:
        Q @ K^T == (Q @ R) @ (K_rot)^T
    bypassing the need to de-rotate cached Keys across the sequence dimension.

    Args:
        query: Query tensor Q of shape [B, H_q, 1, d_h] (or [B, H_q, L_q, d_h]).
        k_cache: Compressed Keys (QuantizedKVTensor, tuple of (k_packed, k_scales, k_qjl), or dict).
        v_cache: Compressed Values (QuantizedKVTensor, tuple of (v_packed, v_scales, v_qjl), or dict).
        rotation_matrix: Orthogonal rotation matrix R of shape [d_h, d_h] (optional if quantizer provided).
        quantizer: Optional DynamicKVQuantizer instance from which R and centroids are derived.
        centroids: Optional 4-element tensor of centroid values (defaults to [-0.75, -0.25, 0.25, 0.75]).
        scale: Optional logit scaling factor (default: 1.0 / sqrt(d_h)).
        attention_mask: Optional attention mask of shape [B, 1, 1, L_k] or broadcastable.
        return_attention_weights: Whether to also return the attention weights tensor A.

    Returns:
        torch.Tensor of shape [B, H_q, 1, d_h] representing attention output O,
        or tuple (O, A) if return_attention_weights is True.
    """
    # 0. Handle argument variations if k_packed, k_scales, etc. were passed flat
    if len(args) >= 5 and isinstance(k_cache, torch.Tensor) and isinstance(v_cache, torch.Tensor):
        k_packed = k_cache
        k_scales = v_cache
        k_qjl = args[0]
        v_packed = args[1]
        v_scales = args[2]
        v_qjl = args[3]
        if rotation_matrix is None and len(args) >= 5:
            rotation_matrix = args[4]
    else:
        k_packed, k_scales, k_qjl = _unpack_kv_cache(k_cache)
        if v_cache is None:
            raise ValueError("v_cache must be provided.")
        v_packed, v_scales, v_qjl = _unpack_kv_cache(v_cache)

    # Resolve rotation matrix R
    if rotation_matrix is None:
        if quantizer is not None:
            rotation_matrix = quantizer.R
        else:
            raise ValueError("Either rotation_matrix or quantizer must be provided.")

    # Resolve centroids
    if centroids is None:
        if quantizer is not None:
            centroids = quantizer.centroids
        else:
            centroids = DEFAULT_CENTROIDS

    d_h = query.shape[-1]
    b, h_q, l_q, _ = query.shape
    h_kv = k_scales.shape[1]

    if h_q % h_kv != 0:
        raise ValueError(
            f"Query heads H_q ({h_q}) must be divisible by KV heads H_kv ({h_kv}) for GQA."
        )
    gqa_ratio = h_q // h_kv

    # 1. Rotated Query Projection: Q_rot = Q @ R
    r_mat = rotation_matrix.to(device=query.device, dtype=query.dtype)
    q_rot = torch.matmul(query, r_mat)  # [B, H_q, L_q, d_h]

    # 2. Reconstruct Keys directly in rotated space (bypassing R^T de-rotation)
    k_unpacked = unpack_2bit(k_packed, target_dtype=torch.long)
    centroids_dev = centroids.to(device=query.device, dtype=query.dtype)
    k_rot_hat = centroids_dev[k_unpacked]  # [B, H_kv, L_k, d_h]

    if k_qjl is not None:
        k_signs, k_qjl_scale = _extract_qjl_components(k_qjl)
        k_rot_hat = (
            k_rot_hat
            + k_qjl_scale.to(device=query.device, dtype=query.dtype)
            * k_signs.to(device=query.device, dtype=query.dtype)
        )

    # 3. GQA Head Alignment (repeat Key & Value heads to match H_q)
    if gqa_ratio > 1:
        k_rot_hat = torch.repeat_interleave(k_rot_hat, repeats=gqa_ratio, dim=1)
        k_scales_aligned = torch.repeat_interleave(k_scales, repeats=gqa_ratio, dim=1)
    else:
        k_scales_aligned = k_scales

    # 4. Attention Scoring:
    # scores = (Q_rot @ K_rot_hat^T) * K_scales
    # Q_rot: [B, H_q, L_q, d_h]
    # K_rot_hat: [B, H_q, L_k, d_h] -> transpose(-1, -2): [B, H_q, d_h, L_k]
    # k_scales_aligned: [B, H_q, L_k, 1] -> transpose(-1, -2): [B, H_q, 1, L_k]
    raw_dot = torch.matmul(q_rot, k_rot_hat.transpose(-1, -2))  # [B, H_q, L_q, L_k]
    scores = raw_dot * k_scales_aligned.to(device=query.device, dtype=query.dtype).transpose(-1, -2)

    # Scale logits by 1.0 / sqrt(d_h)
    scale_factor = scale if scale is not None else (1.0 / math.sqrt(d_h))
    logits = scores * scale_factor

    if attention_mask is not None:
        logits = logits + attention_mask.to(device=logits.device, dtype=logits.dtype)

    attention_weights = torch.softmax(logits, dim=-1)  # [B, H_q, L_q, L_k]

    # 5. Reconstruct Values: V_hat = (V_rot_hat @ R^T) * V_scales
    if quantizer is not None:
        v_hat = quantizer.dequantize(v_packed, v_scales, v_qjl)
    else:
        v_unpacked = unpack_2bit(v_packed, target_dtype=torch.long)
        v_rot_hat = centroids_dev[v_unpacked]
        if v_qjl is not None:
            v_signs, v_qjl_scale = _extract_qjl_components(v_qjl)
            v_rot_hat = (
                v_rot_hat
                + v_qjl_scale.to(device=query.device, dtype=query.dtype)
                * v_signs.to(device=query.device, dtype=query.dtype)
            )
        v_unit_hat = torch.matmul(v_rot_hat, r_mat.T)
        v_hat = v_unit_hat * v_scales.to(device=query.device, dtype=query.dtype)

    # Align Value heads for GQA
    if gqa_ratio > 1:
        v_hat = torch.repeat_interleave(v_hat, repeats=gqa_ratio, dim=1)

    # 6. Aggregate Output: O = A @ V_hat
    output = torch.matmul(attention_weights, v_hat)  # [B, H_q, L_q, d_h]

    if return_attention_weights:
        return output, attention_weights
    return output


__all__ = ["compressed_attention_decode"]
