"""Jina AI embedding provider."""

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

_API_URL = "https://api.jina.ai/v1/embeddings"
_KNOWN_DIMENSIONS: dict[str, int] = {
    "jina-embeddings-v4": 2048,
    "jina-embeddings-v3": 1024,
    "jina-embeddings-v2-base-en": 768,
    "jina-embeddings-v2-base-code": 768,
}


class JinaEmbedding:
    """Jina AI embedding provider."""

    _TIMEOUT_SECONDS = 60.0

    def __init__(
        self,
        model_name: str = "jina-embeddings-v4",
        *,
        batch_size: int = 0,
        api_key: Optional[str] = None,
        task: Optional[str] = None,
        dimensions: Optional[int] = None,
    ) -> None:
        del batch_size
        import httpx
        from tenacity import retry, stop_after_attempt, wait_exponential

        self._api_key = api_key or os.environ.get("JINA_API_KEY")
        if not self._api_key:
            raise RuntimeError("JINA_API_KEY is required for the Jina embedding provider")

        self._model_name = model_name
        self._task = task
        self._dimension = (
            dimensions if dimensions is not None else _KNOWN_DIMENSIONS.get(model_name, 2048)
        )
        self._client = httpx.Client(timeout=self._TIMEOUT_SECONDS)

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            reraise=True,
        )
        def _post_embeddings(body: dict) -> dict:
            response = self._client.post(
                _API_URL,
                json=body,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
            )
            response.raise_for_status()
            return response.json()

        self._post_embeddings = _post_embeddings

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
        texts = ensure_text_list(texts)
        valid_indices, valid_texts = filter_empty_texts(texts)
        if not valid_texts:
            return np.zeros((len(texts), self.dimension))

        task = self._task or ("retrieval.query" if text_type == "query" else "retrieval.passage")
        body: dict = {
            "model": self._model_name,
            "input": valid_texts,
            "task": task,
            "dimensions": self._dimension,
        }

        payload = self._post_embeddings(body)
        embeddings = np.array([item["embedding"] for item in payload["data"]], dtype=float)
        if normalize:
            embeddings = normalize_embeddings(embeddings)
        return restore_empty_embeddings(embeddings, valid_indices, len(texts), self.dimension)
