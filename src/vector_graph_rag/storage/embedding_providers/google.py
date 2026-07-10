"""Google Gemini embedding provider."""

from __future__ import annotations

import os
from typing import List, Literal, Optional, Union

import numpy as np

from vector_graph_rag.storage.embedding_providers.utils import (
    ensure_text_list,
    filter_empty_texts,
    normalize_embeddings,
    restore_empty_embeddings,
)

_KNOWN_DIMENSIONS: dict[str, int] = {
    "gemini-embedding-001": 768,
    "gemini-embedding-2-preview": 768,
    "text-embedding-005": 768,
    "text-embedding-004": 768,
}


class GoogleEmbedding:
    """Google Generative AI embedding provider."""

    def __init__(
        self,
        model_name: str = "gemini-embedding-001",
        *,
        batch_size: int = 0,
        api_key: Optional[str] = None,
    ) -> None:
        del batch_size
        from google import genai

        use_vertex_ai = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").strip().lower() == "true"
        kwargs = {"vertexai": use_vertex_ai}
        if api_key:
            kwargs["api_key"] = api_key
        self._client = genai.Client(**kwargs)
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
        from google.genai import types

        texts = ensure_text_list(texts)
        valid_indices, valid_texts = filter_empty_texts(texts)
        if not valid_texts:
            return np.zeros((len(texts), self.dimension))

        result = self._client.models.embed_content(
            model=self._model_name,
            contents=valid_texts,
            config=types.EmbedContentConfig(output_dimensionality=self._dimension),
        )
        embeddings = np.array([e.values for e in result.embeddings], dtype=float)
        if normalize:
            embeddings = normalize_embeddings(embeddings)
        return restore_empty_embeddings(embeddings, valid_indices, len(texts), self.dimension)


def _detect_dimension(client, model_name: str) -> int:
    """Return the embedding dimension for a model."""
    if model_name in _KNOWN_DIMENSIONS:
        return _KNOWN_DIMENSIONS[model_name]
    result = client.models.embed_content(model=model_name, contents=["dim"])
    return len(result.embeddings[0].values)
