"""IndicBERT embedding extraction module with masked mean pooling and L2 normalization."""

from __future__ import annotations

from typing import Any, Optional, Sequence, Union

import torch
import torch.nn.functional as F


def extract_indic_embeddings(
    texts: Sequence[str],
    model: Any,
    tokenizer: Any,
    device: Optional[Union[torch.device, str]] = None,
    max_length: Optional[int] = None,
) -> torch.Tensor:
    """Extract normalized sentence embeddings using masked mean pooling and L2 normalization.

    Parameters
    ----------
    texts : Sequence[str]
        Input texts to embed.
    model : Any
        Transformer model or compatible model returning ``last_hidden_state``.
    tokenizer : Any
        Tokenizer supporting padding, truncation, and PyTorch tensor output.
    device : Optional[Union[torch.device, str]], default=None
        Target device for computation. Inferred from model parameters if None.
    max_length : Optional[int], default=None
        Optional maximum sequence length.

    Returns
    -------
    torch.Tensor
        L2-normalized sentence embeddings of shape ``[len(texts), 768]`` (or ``[len(texts), hidden_dim]``).
    """
    if len(texts) == 0:
        hidden_dim = getattr(getattr(model, "config", None), "hidden_size", 768)
        return torch.empty((0, hidden_dim), dtype=torch.float32)

    texts_list = list(texts)

    # 1. Tokenize texts with padding=True, truncation=True, return_tensors="pt"
    tokenize_kwargs: dict[str, Any] = {
        "padding": True,
        "truncation": True,
        "return_tensors": "pt",
    }
    if max_length is not None:
        tokenize_kwargs["max_length"] = max_length

    inputs = tokenizer(texts_list, **tokenize_kwargs)

    # Infer device if not explicitly provided
    if device is None and hasattr(model, "parameters"):
        try:
            device = next(model.parameters()).device
        except (StopIteration, AttributeError):
            device = None

    if device is not None:
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    # 2. Pass tokens through model to get last_hidden_state
    with torch.no_grad():
        try:
            outputs = model(**inputs)
        except TypeError:
            # Fall back to passing only essential keys if model has restricted signature
            filtered_inputs = {
                k: v for k, v in inputs.items() if k in ("input_ids", "attention_mask")
            }
            outputs = model(**filtered_inputs)

    if hasattr(outputs, "last_hidden_state"):
        last_hidden_state = outputs.last_hidden_state
    elif isinstance(outputs, (tuple, list)):
        last_hidden_state = outputs[0]
    elif isinstance(outputs, dict) and "last_hidden_state" in outputs:
        last_hidden_state = outputs["last_hidden_state"]
    else:
        last_hidden_state = outputs

    # 3. Apply masked mean pooling:
    #    * Multiply hidden states by attention_mask.unsqueeze(-1)
    #    * Sum along sequence dimension (dim=1)
    #    * Divide by torch.clamp(attention_mask.sum(dim=1, keepdim=True), min=1e-9)
    attention_mask = inputs["attention_mask"]
    input_mask_expanded = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(attention_mask.sum(dim=1, keepdim=True), min=1e-9)
    embeddings = sum_embeddings / sum_mask

    # 4. Apply L2 normalization using torch.nn.functional.normalize(embeddings, p=2, dim=-1)
    normalized_embeddings = F.normalize(embeddings, p=2, dim=-1)

    return normalized_embeddings


class IndicEmbeddingExtractor:
    """Class wrapper for extracting embeddings with IndicBERT models."""

    def __init__(
        self,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        model_name_or_path: str = "ai4bharat/indic-bert",
        device: Optional[Union[torch.device, str]] = None,
    ):
        self.device = device
        if model is None or tokenizer is None:
            try:
                from transformers import AutoModel, AutoTokenizer
            except ImportError as e:
                raise ImportError(
                    "The 'transformers' package is required to load models automatically. "
                    "Please install it or supply model and tokenizer instances directly."
                ) from e

            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            if model is None:
                model = AutoModel.from_pretrained(model_name_or_path)

        self.model = model
        self.tokenizer = tokenizer

        if self.device is not None and hasattr(self.model, "to"):
            self.model.to(self.device)
        if hasattr(self.model, "eval"):
            self.model.eval()

    def extract(
        self,
        texts: Sequence[str],
        max_length: Optional[int] = None,
    ) -> torch.Tensor:
        """Extract L2-normalized sentence embeddings."""
        return extract_indic_embeddings(
            texts=texts,
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            max_length=max_length,
        )

    def __call__(
        self,
        texts: Sequence[str],
        max_length: Optional[int] = None,
    ) -> torch.Tensor:
        """Callable interface forwarding to extract."""
        return self.extract(texts=texts, max_length=max_length)


__all__ = ["IndicEmbeddingExtractor", "extract_indic_embeddings"]
