"""OpenAI and OpenAI-compatible embedding provider."""

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
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbedding:
    """OpenAI-compatible embedding model wrapper."""

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        *,
        batch_size: int = 0,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        from openai import OpenAI
        from tenacity import retry, stop_after_attempt, wait_exponential

        self._model_name = model_name
        self._batch_size = batch_size
        self.client = OpenAI(api_key=api_key, base_url=base_url)

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=10),
        )
        def _call_api(texts: List[str]):
            return self.client.embeddings.create(model=self._model_name, input=texts)

        self._call_api = _call_api
        self._dimension: Optional[int] = _KNOWN_DIMENSIONS.get(model_name)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            response = self._call_api(["test"])
            self._dimension = len(response.data[0].embedding)
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

        response = self._call_api(valid_texts)
        sorted_data = sorted(response.data, key=lambda x: x.index)
        valid_embeddings = np.array([item.embedding for item in sorted_data], dtype=float)

        if normalize:
            valid_embeddings = normalize_embeddings(valid_embeddings)

        return restore_empty_embeddings(valid_embeddings, valid_indices, len(texts), self.dimension)
