"""Unit tests for IndicBERT embedding extraction with masked mean pooling and L2 normalization."""

from __future__ import annotations

from typing import Optional

import pytest
import torch
from transformers import BertConfig, BertModel

from turboquant.embed import IndicEmbeddingExtractor, extract_indic_embeddings


class DummyTokenizer:
    """Self-contained tokenizer for deterministic testing without external network access."""

    def __init__(self, pad_token_id: int = 0) -> None:
        self.pad_token_id = pad_token_id

    def __call__(
        self,
        texts: list[str],
        padding: bool = True,
        truncation: bool = True,
        return_tensors: str = "pt",
        max_length: Optional[int] = None,
    ) -> dict[str, torch.Tensor]:
        tokenized_seqs: list[list[int]] = []
        for text in texts:
            words = text.strip().split()
            # Map words deterministically to token IDs > 0
            seq = [abs(hash(w)) % 800 + 10 for w in words] or [10]
            if truncation and max_length is not None:
                seq = seq[:max_length]
            tokenized_seqs.append(seq)

        max_len = max((len(s) for s in tokenized_seqs), default=0) if padding else 0

        input_ids = []
        attention_mask = []
        for seq in tokenized_seqs:
            seq_len = len(seq)
            if padding and max_len > seq_len:
                padded_seq = seq + [self.pad_token_id] * (max_len - seq_len)
                mask = [1] * seq_len + [0] * (max_len - seq_len)
            else:
                padded_seq = seq
                mask = [1] * seq_len
            input_ids.append(padded_seq)
            attention_mask.append(mask)

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }
        return {"input_ids": input_ids, "attention_mask": attention_mask}


@pytest.fixture
def dummy_tokenizer() -> DummyTokenizer:
    return DummyTokenizer()


@pytest.fixture
def bert_768_model() -> BertModel:
    """In-memory 768-dim BERT model with random weights."""
    torch.manual_seed(42)
    config = BertConfig(
        vocab_size=1000,
        hidden_size=768,
        num_attention_heads=4,
        num_hidden_layers=2,
        intermediate_size=512,
    )
    model = BertModel(config)
    model.eval()
    return model


def test_extract_indic_embeddings_output_shape(
    bert_768_model: BertModel, dummy_tokenizer: DummyTokenizer
) -> None:
    """Verify that extract_indic_embeddings returns tensor of shape [N, 768]."""
    texts = [
        "नमस्ते",
        "TurboQuant is fast and memory efficient",
        "Short text",
    ]
    embeddings = extract_indic_embeddings(texts, bert_768_model, dummy_tokenizer)

    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape == (len(texts), 768)

    # Test single sentence
    single_emb = extract_indic_embeddings(["Single sentence"], bert_768_model, dummy_tokenizer)
    assert single_emb.shape == (1, 768)

    # Test empty list
    empty_emb = extract_indic_embeddings([], bert_768_model, dummy_tokenizer)
    assert empty_emb.shape == (0, 768)


def test_extract_indic_embeddings_l2_norm(
    bert_768_model: BertModel, dummy_tokenizer: DummyTokenizer
) -> None:
    """Verify that every returned vector has an L2 norm approximately equal to 1.0."""
    texts = [
        "भारत एक महान देश है",
        "Vector quantization enables fast search",
        "A",
        "This is a significantly longer sentence designed to test pooling across multiple tokens",
    ]
    embeddings = extract_indic_embeddings(texts, bert_768_model, dummy_tokenizer)

    # Compute L2 norms along dimension -1
    norms = torch.norm(embeddings, p=2, dim=-1)

    assert norms.shape == (len(texts),)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), (
        f"L2 norms deviate from 1.0: {norms}"
    )


def test_extract_indic_embeddings_padding_invariance(
    bert_768_model: BertModel, dummy_tokenizer: DummyTokenizer
) -> None:
    """Ensure a sentence embedded alone yields the exact same embedding vector as when batched and padded."""
    short_sentence = "TurboQuant"
    long_sentence = (
        "TurboQuant is an extreme throughput vector quantization library "
        "designed for fast semantic search and scalable neural retrieval in Indic languages."
    )

    # 1. Embed short sentence alone
    emb_alone = extract_indic_embeddings([short_sentence], bert_768_model, dummy_tokenizer)

    # 2. Embed short sentence batched alongside much longer sentence (as first element)
    emb_batched_first = extract_indic_embeddings(
        [short_sentence, long_sentence], bert_768_model, dummy_tokenizer
    )

    # 3. Embed short sentence batched alongside much longer sentence (as second element)
    emb_batched_second = extract_indic_embeddings(
        [long_sentence, short_sentence], bert_768_model, dummy_tokenizer
    )

    # Verify short sentence embedding is identical despite differing padding in batch
    diff_first = torch.max(torch.abs(emb_alone[0] - emb_batched_first[0])).item()
    diff_second = torch.max(torch.abs(emb_alone[0] - emb_batched_second[1])).item()

    assert torch.allclose(emb_alone[0], emb_batched_first[0], atol=1e-5), (
        f"Padding invariance failed when short sentence was first in batch. Max diff: {diff_first}"
    )
    assert torch.allclose(emb_alone[0], emb_batched_second[1], atol=1e-5), (
        f"Padding invariance failed when short sentence was second in batch. Max diff: {diff_second}"
    )


def test_indic_embedding_extractor_wrapper(
    bert_768_model: BertModel, dummy_tokenizer: DummyTokenizer
) -> None:
    """Verify that IndicEmbeddingExtractor wrapper class extracts identical embeddings."""
    extractor = IndicEmbeddingExtractor(
        model=bert_768_model,
        tokenizer=dummy_tokenizer,
    )
    texts = ["Testing extractor wrapper class", "Another sentence"]

    emb_helper = extract_indic_embeddings(texts, bert_768_model, dummy_tokenizer)
    emb_method = extractor.extract(texts)
    emb_callable = extractor(texts)

    assert torch.allclose(emb_helper, emb_method, atol=1e-6)
    assert torch.allclose(emb_helper, emb_callable, atol=1e-6)
    assert emb_method.shape == (2, 768)
    assert torch.allclose(torch.norm(emb_method, p=2, dim=-1), torch.ones(2), atol=1e-5)
