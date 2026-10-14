"""Dynamic KV cache quantizer for LLM inference in TurboQuant."""

from __future__ import annotations

from typing import Any, Dict, NamedTuple, Optional, Tuple, Union
import torch

from turboquant.packing import pack_2bit, unpack_2bit
from turboquant.qjl import compute_residual, extract_qjl_signs
from turboquant.rotation import random_orthogonal_matrix


class _QJLDataTuple(NamedTuple):
    signs: torch.Tensor
    scale: torch.Tensor


class QJLData(_QJLDataTuple):
    """Container for 1-bit QJL error correction data.

    Attributes:
        signs: 1-bit signs tensor in {-1, +1} matching rotated coordinate shape.
        scale: Residual scale factor tensor of shape [..., 1].
    """

    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, str):
            if item == "signs":
                return self.signs
            elif item == "scale":
                return self.scale
            raise KeyError(f"Invalid key '{item}' for QJLData")
        return tuple.__getitem__(self, item)

    def get(self, key: str, default: Any = None) -> Any:
        if key == "signs":
            return self.signs
        elif key == "scale":
            return self.scale
        return default


class _QuantizedKVTensorTuple(NamedTuple):
    packed_codes: torch.Tensor
    scales: torch.Tensor
    qjl_data: Optional[Union[QJLData, Dict[str, torch.Tensor]]] = None


class QuantizedKVTensor(_QuantizedKVTensorTuple):
    """Container for 2-bit packed KV cache representations.

    Supports tuple unpacking: ``packed_codes, scales, qjl_data = res``,
    attribute access: ``res.packed_codes``, and dictionary indexing: ``res['packed_codes']``.

    Attributes:
        packed_codes: Packed uint8 tensor of shape [..., head_dim // 4].
        scales: Euclidean norms tensor of shape [..., 1].
        qjl_data: Optional QJL error correction data (QJLData or None).
    """

    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, str):
            if item == "packed_codes":
                return self.packed_codes
            elif item == "scales":
                return self.scales
            elif item == "qjl_data":
                return self.qjl_data
            raise KeyError(f"Invalid key '{item}' for QuantizedKVTensor")
        return tuple.__getitem__(self, item)

    def get(self, key: str, default: Any = None) -> Any:
        if key == "packed_codes":
            return self.packed_codes
        elif key == "scales":
            return self.scales
        elif key == "qjl_data":
            return self.qjl_data
        return default


class DynamicKVQuantizer:
    """Dynamic 2-bit quantizer for streaming Key and Value tensors in LLM attention.

    Compresses head vectors (e.g. d_h = 128) using:
        1. Per-vector Euclidean norm extraction (scale = ||x||_2)
        2. Normalization to unit sphere (x_unit = x / (scale + 1e-8))
        3. Fixed orthogonal Haar rotation (x_rot = x_unit @ R)
        4. 2-bit centroid bucketization and physical uint8 packing (32 bytes / head)
        5. Optional 1-bit QJL error correction for residual estimation
    """

    def __init__(
        self,
        head_dim: int = 128,
        use_qjl: bool = False,
        device: Optional[Union[torch.device, str]] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initialize DynamicKVQuantizer.

        Args:
            head_dim: Per-head attention dimension (default: 128, must be divisible by 4).
            use_qjl: Whether to enable 1-bit QJL residual error correction.
            device: Torch device for precomputed tensors (default: None / CPU).
            dtype: Floating point precision for computations (default: torch.float32).
        """
        if head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {head_dim}")
        if head_dim % 4 != 0:
            raise ValueError(f"head_dim must be divisible by 4 for 2-bit packing, got {head_dim}")

        self.head_dim = head_dim
        self.use_qjl = use_qjl
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype

        # Precompute and register fixed orthogonal rotation matrix R in R^{d_h x d_h}
        self.R: torch.Tensor = random_orthogonal_matrix(
            dim=head_dim,
            device=self.device,
            dtype=self.dtype,
        )

        # Decision boundaries [-0.5, 0.0, 0.5] and centroids [-0.75, -0.25, 0.25, 0.75]
        self.boundaries: torch.Tensor = torch.tensor(
            [-0.5, 0.0, 0.5],
            device=self.device,
            dtype=self.dtype,
        )
        self.centroids: torch.Tensor = torch.tensor(
            [-0.75, -0.25, 0.25, 0.75],
            device=self.device,
            dtype=self.dtype,
        )

    def to(
        self,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> DynamicKVQuantizer:
        """Move quantizer buffers to specified device and/or dtype."""
        if device is not None:
            self.device = torch.device(device) if isinstance(device, str) else device
        if dtype is not None:
            self.dtype = dtype

        self.R = self.R.to(device=self.device, dtype=self.dtype)
        self.boundaries = self.boundaries.to(device=self.device, dtype=self.dtype)
        self.centroids = self.centroids.to(device=self.device, dtype=self.dtype)
        return self

    def quantize(self, x: torch.Tensor) -> QuantizedKVTensor:
        """Compress input tensor of shape [..., head_dim] into packed 2-bit representation.

        Args:
            x: Input tensor of shape [..., head_dim] (e.g. [batch, num_kv_heads, seq_len, head_dim]).

        Returns:
            QuantizedKVTensor with:
                - packed_codes: torch.uint8 tensor of shape [..., head_dim // 4]
                - scales: Euclidean norm tensor of shape [..., 1]
                - qjl_data: QJLData with 1-bit signs and scale if use_qjl=True, else None
        """
        if x.shape[-1] != self.head_dim:
            raise ValueError(
                f"Expected last dimension to be head_dim={self.head_dim}, got {x.shape[-1]}"
            )

        # 1. Extract Euclidean norm per vector
        scale = torch.norm(x, dim=-1, keepdim=True)

        # 2. Normalize to unit sphere
        x_unit = x / (scale + 1e-8)

        # 3. Rotate: x_rot = x_unit @ R
        r_mat = self.R.to(device=x.device, dtype=x.dtype)
        x_rot = torch.matmul(x_unit, r_mat)

        # 4. Bucketize x_rot against boundaries [-0.5, 0.0, 0.5] -> discrete codes {0, 1, 2, 3}
        boundaries = self.boundaries.to(device=x.device, dtype=x.dtype)
        codes = torch.bucketize(x_rot.contiguous(), boundaries)
        codes = torch.clamp(codes, 0, len(self.centroids) - 1)

        # 5. Pack discrete codes into torch.uint8 using pack_2bit -> shape [..., head_dim // 4]
        packed_codes = pack_2bit(codes)

        # 6. Optional 1-bit QJL error correction
        qjl_data: Optional[QJLData] = None
        if self.use_qjl:
            centroids = self.centroids.to(device=x.device, dtype=x.dtype)
            x_rot_hat = centroids[codes]
            residual = compute_residual(x_rot, x_rot_hat)
            signs = extract_qjl_signs(residual)
            qjl_scale = torch.mean(torch.abs(residual), dim=-1, keepdim=True)
            qjl_data = QJLData(signs=signs, scale=qjl_scale)

        return QuantizedKVTensor(
            packed_codes=packed_codes,
            scales=scale,
            qjl_data=qjl_data,
        )

    def dequantize(
        self,
        packed_codes: Union[torch.Tensor, QuantizedKVTensor, Dict[str, Any]],
        scales: Optional[torch.Tensor] = None,
        qjl_data: Optional[Union[QJLData, Dict[str, Any], Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> torch.Tensor:
        """Reconstruct original vectors from packed 2-bit representation.

        Args:
            packed_codes: Packed torch.uint8 tensor, or QuantizedKVTensor / dict container.
            scales: Per-vector scale tensor of shape [..., 1] (optional if container provided).
            qjl_data: Optional QJL correction data (optional if container provided).

        Returns:
            Reconstructed tensor of shape [..., head_dim].
        """
        # Unpack container if provided as first argument
        if isinstance(packed_codes, dict):
            scales = packed_codes["scales"]
            qjl_data = packed_codes.get("qjl_data")
            packed_codes = packed_codes["packed_codes"]
        elif hasattr(packed_codes, "packed_codes"):
            scales = packed_codes.scales
            qjl_data = packed_codes.qjl_data
            packed_codes = packed_codes.packed_codes

        if scales is None:
            raise ValueError("scales tensor must be provided for dequantization.")

        # 1. Unpack 2-bit codes into discrete indices {0, 1, 2, 3}
        unpacked_codes = unpack_2bit(packed_codes, target_dtype=torch.long)

        # 2. Map indices to centroid coordinates
        target_dtype = scales.dtype if scales is not None else self.dtype
        centroids = self.centroids.to(device=unpacked_codes.device, dtype=target_dtype)
        x_rot_hat = centroids[unpacked_codes]

        # 3. Add QJL correction if present
        if qjl_data is not None:
            if isinstance(qjl_data, dict):
                signs = qjl_data["signs"]
                qjl_scale = qjl_data["scale"]
            elif hasattr(qjl_data, "signs"):
                signs = qjl_data.signs
                qjl_scale = qjl_data.scale
            elif isinstance(qjl_data, (tuple, list)):
                signs, qjl_scale = qjl_data
            else:
                raise TypeError(f"Unsupported qjl_data type: {type(qjl_data)}")

            signs = signs.to(device=x_rot_hat.device, dtype=x_rot_hat.dtype)
            qjl_scale = qjl_scale.to(device=x_rot_hat.device, dtype=x_rot_hat.dtype)
            x_rot_hat = x_rot_hat + qjl_scale * signs

        # 4. De-rotate: x_unit_hat = x_rot_hat @ R.T
        r_mat = self.R.to(device=x_rot_hat.device, dtype=x_rot_hat.dtype)
        x_unit_hat = torch.matmul(x_rot_hat, r_mat.T)

        # 5. Rescale: x_hat = x_unit_hat * scales
        x_hat = x_unit_hat * scales

        return x_hat


__all__ = [
    "DynamicKVQuantizer",
    "QJLData",
    "QuantizedKVTensor",
]
