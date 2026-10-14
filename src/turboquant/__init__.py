"""TurboQuant package."""

from turboquant.attention import compressed_attention_decode
from turboquant.cache import TurboQuantCache, TurboQuantCacheLayer
from turboquant.embed import IndicEmbeddingExtractor, extract_indic_embeddings
from turboquant.index import SearchResult, TurboQuantIndex, quantize_embeddings_2bit
from turboquant.ingest import (
    DocumentPrefixEngine,
    PrefixState,
    ingest_document_prefix,
)
from turboquant.packing import (
    pack_2bit,
    unpack_2bit,
)
from turboquant.patch import (
    apply_turboquant,
    restore_turboquant,
    turboquant_attention_forward,
    turboquant_context,
)
from turboquant.polar import (
    cartesian_to_polar,
    dequantize_angles,
    polar_to_cartesian,
    quantize_angles,
)
from turboquant.qa import (
    DEFAULT_INDIC_QA_TEMPLATE,
    IndicDocumentQAEngine,
    QAResult,
)
from turboquant.qjl import (
    compute_residual,
    estimate_dot_product,
    extract_qjl_signs,
)
from turboquant.quantizer import (
    DynamicKVQuantizer,
    QJLData,
    QuantizedKVTensor,
)
from turboquant.rotation import random_orthogonal_matrix

__all__ = [
    "DocumentPrefixEngine",
    "DynamicKVQuantizer",
    "IndicDocumentQAEngine",
    "IndicEmbeddingExtractor",
    "PrefixState",
    "QAResult",
    "QJLData",
    "QuantizedKVTensor",
    "SearchResult",
    "TurboQuantCache",
    "TurboQuantCacheLayer",
    "TurboQuantIndex",
    "apply_turboquant",
    "cartesian_to_polar",
    "compressed_attention_decode",
    "compute_residual",
    "dequantize_angles",
    "estimate_dot_product",
    "extract_indic_embeddings",
    "extract_qjl_signs",
    "ingest_document_prefix",
    "pack_2bit",
    "polar_to_cartesian",
    "quantize_angles",
    "quantize_embeddings_2bit",
    "random_orthogonal_matrix",
    "restore_turboquant",
    "turboquant_attention_forward",
    "turboquant_context",
    "unpack_2bit",
]
