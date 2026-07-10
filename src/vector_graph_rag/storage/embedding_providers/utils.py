"""Shared helpers for embedding providers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import List, Union

import numpy as np


def ensure_text_list(texts: Union[str, List[str]]) -> List[str]:
    """Normalize a single text or text list into a list."""
    if isinstance(texts, str):
        return [texts]
    return texts


def filter_empty_texts(texts: List[str]) -> tuple[List[int], List[str]]:
    """Return indexes and values for non-empty texts."""
    valid_indices = [i for i, text in enumerate(texts) if text and text.strip()]
    return valid_indices, [texts[i] for i in valid_indices]


def restore_empty_embeddings(
    valid_embeddings: np.ndarray,
    valid_indices: List[int],
    total_count: int,
    dimension: int,
) -> np.ndarray:
    """Restore zero vectors for empty texts after embedding only valid inputs."""
    if len(valid_indices) == total_count:
        return valid_embeddings

    embeddings = np.zeros((total_count, dimension), dtype=float)
    for idx, valid_idx in enumerate(valid_indices):
        embeddings[valid_idx] = valid_embeddings[idx]
    return embeddings


def batched(items: List[str], batch_size: int) -> Iterable[List[str]]:
    """Yield batches from a list of strings."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize embeddings while preserving zero rows."""
    if embeddings.size == 0:
        return embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return np.divide(embeddings, norms, out=np.zeros_like(embeddings), where=norms != 0)
