"""Local sentence-transformers embedding provider."""

from __future__ import annotations

import io
import os
import sys
from typing import List, Literal, Union

import numpy as np

from vector_graph_rag.storage.embedding_providers.utils import (
    ensure_text_list,
    filter_empty_texts,
    restore_empty_embeddings,
)


def _detect_device() -> str:
    """Detect best available device."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class LocalEmbedding:
    """Local embedding provider backed by sentence-transformers."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        *,
        batch_size: int = 0,
    ) -> None:
        del batch_size

        prev_tqdm = os.environ.get("TQDM_DISABLE")
        os.environ["TQDM_DISABLE"] = "1"
        old_stderr = sys.stderr
        try:
            sys.stderr = io.StringIO()
            from sentence_transformers import SentenceTransformer

            self._st_model = SentenceTransformer(
                model_name, device=_detect_device(), trust_remote_code=True
            )
        finally:
            sys.stderr = old_stderr
            if prev_tqdm is None:
                os.environ.pop("TQDM_DISABLE", None)
            else:
                os.environ["TQDM_DISABLE"] = prev_tqdm

        self._model_name = model_name
        self._dimension = self._st_model.get_sentence_embedding_dimension() or 384

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True,
        text_type: Literal["query", "document"] = "query",
    ) -> np.ndarray:
        """Encode texts to embeddings."""
        del text_type
        texts = ensure_text_list(texts)
        valid_indices, valid_texts = filter_empty_texts(texts)
        if not valid_texts:
            return np.zeros((len(texts), self.dimension))

        embeddings = np.array(
            self._st_model.encode(valid_texts, normalize_embeddings=normalize), dtype=float
        )
        return restore_empty_embeddings(embeddings, valid_indices, len(texts), self.dimension)
