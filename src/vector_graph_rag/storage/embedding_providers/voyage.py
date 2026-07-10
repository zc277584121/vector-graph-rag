"""Voyage AI embedding provider."""

from __future__ import annotations

from typing import List, Literal, Optional, Union

import numpy as np

from vector_graph_rag.storage.embedding_providers.utils import (
    ensure_text_list,
    filter_empty_texts,
    normalize_embeddings,
    restore_empty_embeddings,
)

_KNOWN_DIMENSIONS: dict[str, int] = {
    "voyage-4-lite": 1024,
    "voyage-4": 1024,
    "voyage-4-large": 1024,
    "voyage-3-lite": 512,
    "voyage-3": 1024,
    "voyage-code-3": 1024,
}


class VoyageEmbedding:
    """Voyage AI embedding provider."""

    def __init__(
        self,
        model_name: str = "voyage-3-lite",
        *,
        batch_size: int = 0,
        api_key: Optional[str] = None,
    ) -> None:
        del batch_size
        import voyageai

        self._client = voyageai.Client(api_key=api_key) if api_key else voyageai.Client()
        self._model_name = model_name
        self._dimension = _detect_dimension(self._client, model_name)

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

        result = self._client.embed(valid_texts, model=self._model_name)
        embeddings = np.array(result.embeddings, dtype=float)
        if normalize:
            embeddings = normalize_embeddings(embeddings)
        return restore_empty_embeddings(embeddings, valid_indices, len(texts), self.dimension)


def _detect_dimension(client, model_name: str) -> int:
    """Return the embedding dimension for a model."""
    if model_name in _KNOWN_DIMENSIONS:
        return _KNOWN_DIMENSIONS[model_name]
    trial = client.embed(["dim"], model=model_name)
    return len(trial.embeddings[0])
