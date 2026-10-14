"""TurboQuant 2-bit Key-Value Cache implementation compatible with HuggingFace transformers.

Maintains 2-bit compressed keys and values in RAM during LLM autoregressive inference,
drastically reducing KV cache memory footprints.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import torch

try:
    from transformers.cache_utils import Cache
except ImportError:
    class Cache:  # type: ignore[no-redef]
        """Fallback base Cache class when transformers is not installed."""
        pass

from turboquant.quantizer import DynamicKVQuantizer, QJLData, QuantizedKVTensor


class TurboQuantCacheLayer:
    """View into a single layer's compressed KV cache buffers."""

    def __init__(self, cache: "TurboQuantCache", layer_idx: int) -> None:
        self._cache = cache
        self.layer_idx = layer_idx

    @property
    def k_packed(self) -> Optional[torch.Tensor]:
        return self._cache.k_packed[self.layer_idx]

    @property
    def k_scales(self) -> Optional[torch.Tensor]:
        return self._cache.k_scales[self.layer_idx]

    @property
    def k_qjl(self) -> Optional[QJLData]:
        return self._cache.k_qjl[self.layer_idx]

    @property
    def v_packed(self) -> Optional[torch.Tensor]:
        return self._cache.v_packed[self.layer_idx]

    @property
    def v_scales(self) -> Optional[torch.Tensor]:
        return self._cache.v_scales[self.layer_idx]

    @property
    def v_qjl(self) -> Optional[QJLData]:
        return self._cache.v_qjl[self.layer_idx]

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Invalid key '{key}' for TurboQuantCacheLayer")

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def get_seq_length(self) -> int:
        return self._cache.get_seq_length(self.layer_idx)

    def get_max_length(self) -> int:
        return -1

    def crop(self, max_length: int) -> None:
        self._cache.crop_layer(self.layer_idx, max_length)


class TurboQuantCache(Cache):
    """Dynamic 2-bit compressed Key-Value cache for TurboQuant inference.

    Inherits from transformers Cache, maintaining compressed buffers per layer:
        - k_packed: uint8 tensor of shape [B, H_kv, S, d_h // 4]
        - k_scales: float tensor of shape [B, H_kv, S, 1]
        - k_qjl: optional 1-bit QJL error correction data
        - v_packed: uint8 tensor of shape [B, H_kv, S, d_h // 4]
        - v_scales: float tensor of shape [B, H_kv, S, 1]
        - v_qjl: optional 1-bit QJL error correction data
    """

    def __init__(
        self,
        quantizer: Optional[DynamicKVQuantizer] = None,
        head_dim: Optional[int] = None,
        use_qjl: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize TurboQuantCache.

        Args:
            quantizer: Optional pre-configured DynamicKVQuantizer instance.
            head_dim: Per-head vector dimension (e.g. 128).
            use_qjl: Whether to enable 1-bit QJL residual correction.
        """
        if issubclass(TurboQuantCache, Cache) and Cache is not object:
            try:
                super().__init__(layers=[])
            except Exception:
                pass

        self.quantizer = quantizer
        self.head_dim = head_dim or (quantizer.head_dim if quantizer is not None else None)
        self.use_qjl = use_qjl or (quantizer.use_qjl if quantizer is not None else False)

        # Per-layer compressed storage buffers
        self.k_packed: List[Optional[torch.Tensor]] = []
        self.k_scales: List[Optional[torch.Tensor]] = []
        self.k_qjl: List[Optional[QJLData]] = []
        self.v_packed: List[Optional[torch.Tensor]] = []
        self.v_scales: List[Optional[torch.Tensor]] = []
        self.v_qjl: List[Optional[QJLData]] = []

    @property
    def is_croppable(self) -> bool:
        return True

    @property
    def layers(self) -> List[TurboQuantCacheLayer]:
        """List of layer views for compatibility with Cache abstractions."""
        return [TurboQuantCacheLayer(self, i) for i in range(len(self.k_packed))]

    @property
    def key_cache(self) -> List[Optional[torch.Tensor]]:
        """Compatibility property for key packed buffers."""
        return self.k_packed

    @property
    def value_cache(self) -> List[Optional[torch.Tensor]]:
        """Compatibility property for value packed buffers."""
        return self.v_packed

    def __len__(self) -> int:
        return len(self.k_packed)

    def __getitem__(self, layer_idx: int) -> TurboQuantCacheLayer:
        if layer_idx < 0 or layer_idx >= len(self.k_packed):
            raise IndexError(f"Layer index {layer_idx} out of range (num layers: {len(self.k_packed)})")
        return TurboQuantCacheLayer(self, layer_idx)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Return sequence length stored for the specified layer."""
        if layer_idx < len(self.k_packed) and self.k_packed[layer_idx] is not None:
            return self.k_packed[layer_idx].shape[-2]
        return 0

    def get_max_length(self, layer_idx: Optional[int] = None) -> int:
        """Return maximum sequence length supported (-1 denotes unbounded dynamic)."""
        return -1

    def get_max_cache_shape(self) -> int:
        return -1

    def get_mask_sizes(self, query_length: int, layer_idx: int = 0) -> Tuple[int, int]:
        """Return length and offset of cache used to generate causal attention mask."""
        kv_offset = 0
        kv_length = self.get_seq_length(layer_idx) + query_length
        return kv_length, kv_offset

    def reset(self) -> None:
        """Clear all cached buffers across all layers."""
        self.k_packed.clear()
        self.k_scales.clear()
        self.k_qjl.clear()
        self.v_packed.clear()
        self.v_scales.clear()
        self.v_qjl.clear()

    def crop(self, max_length: int) -> None:
        """Slice all stored buffers in-place along the sequence dimension (dim=-2) to [:max_length].

        Zero-copy slicing allows subsequent decode/generation steps to reuse the exact
        cached prefix without deep-copying memory.

        Args:
            max_length: Maximum sequence length to keep.
        """
        max_len = max(0, int(max_length))
        for layer_idx in range(len(self.k_packed)):
            self.crop_layer(layer_idx, max_len)
        if hasattr(self, "_seen_tokens"):
            self._seen_tokens = max_len

    def crop_layer(self, layer_idx: int, max_length: int) -> None:
        """Slice stored buffers for layer_idx in-place along dim=-2 to [:max_length]."""
        if layer_idx >= len(self.k_packed):
            return
        max_len = max(0, int(max_length))

        if self.k_packed[layer_idx] is not None:
            self.k_packed[layer_idx] = self.k_packed[layer_idx][:, :, :max_len, :]
        if self.k_scales[layer_idx] is not None:
            self.k_scales[layer_idx] = self.k_scales[layer_idx][:, :, :max_len, :]
        if self.k_qjl[layer_idx] is not None:
            qjl = self.k_qjl[layer_idx]
            signs = qjl.signs[:, :, :max_len, :] if qjl.signs is not None else None
            scale = qjl.scale[:, :, :max_len, :] if qjl.scale is not None else None
            self.k_qjl[layer_idx] = QJLData(signs=signs, scale=scale)

        if self.v_packed[layer_idx] is not None:
            self.v_packed[layer_idx] = self.v_packed[layer_idx][:, :, :max_len, :]
        if self.v_scales[layer_idx] is not None:
            self.v_scales[layer_idx] = self.v_scales[layer_idx][:, :, :max_len, :]
        if self.v_qjl[layer_idx] is not None:
            qjl = self.v_qjl[layer_idx]
            signs = qjl.signs[:, :, :max_len, :] if qjl.signs is not None else None
            scale = qjl.scale[:, :, :max_len, :] if qjl.scale is not None else None
            self.v_qjl[layer_idx] = QJLData(signs=signs, scale=scale)

    def get_cache_size_bytes(self) -> int:
        """Calculate total memory consumed by compressed KV buffers across all layers in bytes."""
        total = 0
        for buf_list in (self.k_packed, self.k_scales, self.v_packed, self.v_scales):
            for t in buf_list:
                if t is not None:
                    total += t.numel() * t.element_size()
        for qjl_list in (self.k_qjl, self.v_qjl):
            for qjl in qjl_list:
                if qjl is not None:
                    if qjl.signs is not None:
                        total += qjl.signs.numel() * qjl.signs.element_size()
                    if qjl.scale is not None:
                        total += qjl.scale.numel() * qjl.scale.element_size()
        return total

    @property
    def cache_bytes(self) -> int:
        """Total memory consumed by compressed KV buffers across all layers in bytes."""
        return self.get_cache_size_bytes()

    def get_k(self, layer_idx: int) -> QuantizedKVTensor:
        """Return complete QuantizedKVTensor for keys at layer_idx."""
        if layer_idx >= len(self.k_packed) or self.k_packed[layer_idx] is None:
            raise IndexError(f"No key cache found for layer {layer_idx}")
        return QuantizedKVTensor(
            packed_codes=self.k_packed[layer_idx],
            scales=self.k_scales[layer_idx],
            qjl_data=self.k_qjl[layer_idx],
        )

    def get_v(self, layer_idx: int) -> QuantizedKVTensor:
        """Return complete QuantizedKVTensor for values at layer_idx."""
        if layer_idx >= len(self.v_packed) or self.v_packed[layer_idx] is None:
            raise IndexError(f"No value cache found for layer {layer_idx}")
        return QuantizedKVTensor(
            packed_codes=self.v_packed[layer_idx],
            scales=self.v_scales[layer_idx],
            qjl_data=self.v_qjl[layer_idx],
        )

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int = 0,
        cache_kwargs: Optional[Dict[str, Any]] = None,
        quantizer: Optional[DynamicKVQuantizer] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[QuantizedKVTensor, QuantizedKVTensor]:
        """Compress incoming K, V states and append along sequence dimension S.

        Args:
            key_states: Incoming Keys of shape [B, H_kv, S, d_h].
            value_states: Incoming Values of shape [B, H_kv, S, d_h].
            layer_idx: Target transformer layer index.
            cache_kwargs: Optional metadata or keyword arguments.
            quantizer: Optional DynamicKVQuantizer overriding default quantizer.

        Returns:
            Tuple of (QuantizedKVTensor, QuantizedKVTensor) containing the updated
            cumulative compressed cache for layer_idx.
        """
        # Ensure capacity up to layer_idx
        while len(self.k_packed) <= layer_idx:
            self.k_packed.append(None)
            self.k_scales.append(None)
            self.k_qjl.append(None)
            self.v_packed.append(None)
            self.v_scales.append(None)
            self.v_qjl.append(None)

        # Resolve active quantizer
        active_quantizer = quantizer or self.quantizer
        if active_quantizer is None:
            head_dim = self.head_dim or key_states.shape[-1]
            active_quantizer = DynamicKVQuantizer(
                head_dim=head_dim,
                use_qjl=self.use_qjl,
                device=key_states.device,
                dtype=key_states.dtype,
            )
            self.quantizer = active_quantizer
            self.head_dim = head_dim

        # Compress new incoming states
        k_quant = active_quantizer.quantize(key_states)
        v_quant = active_quantizer.quantize(value_states)

        # Append along sequence dimension S (dim = -2)
        if self.k_packed[layer_idx] is None:
            self.k_packed[layer_idx] = k_quant.packed_codes
            self.k_scales[layer_idx] = k_quant.scales
            self.k_qjl[layer_idx] = k_quant.qjl_data

            self.v_packed[layer_idx] = v_quant.packed_codes
            self.v_scales[layer_idx] = v_quant.scales
            self.v_qjl[layer_idx] = v_quant.qjl_data
        else:
            self.k_packed[layer_idx] = torch.cat(
                [self.k_packed[layer_idx], k_quant.packed_codes], dim=-2
            )
            self.k_scales[layer_idx] = torch.cat(
                [self.k_scales[layer_idx], k_quant.scales], dim=-2
            )

            self.v_packed[layer_idx] = torch.cat(
                [self.v_packed[layer_idx], v_quant.packed_codes], dim=-2
            )
            self.v_scales[layer_idx] = torch.cat(
                [self.v_scales[layer_idx], v_quant.scales], dim=-2
            )

            if k_quant.qjl_data is not None and self.k_qjl[layer_idx] is not None:
                old_k_qjl = self.k_qjl[layer_idx]
                k_signs = torch.cat([old_k_qjl.signs, k_quant.qjl_data.signs], dim=-2)
                k_scale = torch.cat([old_k_qjl.scale, k_quant.qjl_data.scale], dim=-2)
                self.k_qjl[layer_idx] = QJLData(signs=k_signs, scale=k_scale)
            else:
                self.k_qjl[layer_idx] = k_quant.qjl_data

            if v_quant.qjl_data is not None and self.v_qjl[layer_idx] is not None:
                old_v_qjl = self.v_qjl[layer_idx]
                v_signs = torch.cat([old_v_qjl.signs, v_quant.qjl_data.signs], dim=-2)
                v_scale = torch.cat([old_v_qjl.scale, v_quant.qjl_data.scale], dim=-2)
                self.v_qjl[layer_idx] = QJLData(signs=v_signs, scale=v_scale)
            else:
                self.v_qjl[layer_idx] = v_quant.qjl_data

        k_out = QuantizedKVTensor(
            packed_codes=self.k_packed[layer_idx],
            scales=self.k_scales[layer_idx],
            qjl_data=self.k_qjl[layer_idx],
        )
        v_out = QuantizedKVTensor(
            packed_codes=self.v_packed[layer_idx],
            scales=self.v_scales[layer_idx],
            qjl_data=self.v_qjl[layer_idx],
        )
        return k_out, v_out


__all__ = [
    "TurboQuantCache",
    "TurboQuantCacheLayer",
]
