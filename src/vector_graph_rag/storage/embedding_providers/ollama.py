"""Ollama embedding provider."""

from __future__ import annotations

from typing import List, Literal, Optional, Union

import numpy as np

from vector_graph_rag.storage.embedding_providers.utils import (
    ensure_text_list,
    filter_empty_texts,
    normalize_embeddings,
    restore_empty_embeddings,
)


class OllamaEmbedding:
    """Ollama embedding provider for local models."""

    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        *,
        batch_size: int = 0,
        base_url: Optional[str] = None,
    ) -> None:
        del batch_size
        import ollama

        self._client = ollama.Client(host=base_url) if base_url else ollama.Client()
        self._model_name = model_name
        trial = self._client.embed(model=model_name, input=["dim"])
        self._dimension = len(trial["embeddings"][0])

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

        result = self._client.embed(model=self._model_name, input=valid_texts)
        embeddings = np.array(result["embeddings"], dtype=float)
        if normalize:
            embeddings = normalize_embeddings(embeddings)
        return restore_empty_embeddings(embeddings, valid_indices, len(texts), self.dimension)
