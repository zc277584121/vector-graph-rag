"""Embedding model facade for vector representations."""

from __future__ import annotations

import warnings
from typing import List, Literal, Optional

from tqdm import tqdm

from vector_graph_rag.config import Settings, get_settings
from vector_graph_rag.observability import start_span
from vector_graph_rag.storage.embedding_providers import (
    DEFAULT_MODELS,
    canonicalize_provider,
    default_model_for_provider,
    get_provider,
)
from vector_graph_rag.storage.embedding_providers.huggingface import (
    INSTRUCTION_TEMPLATES,
    HuggingFaceEmbedding,
)
from vector_graph_rag.storage.embedding_providers.openai import OpenAIEmbedding

_SETTINGS_DEFAULT_EMBEDDING_MODEL = Settings.model_fields["embedding_model"].default


def _is_openai_model(model_name: str) -> bool:
    """Check whether a model name is a known OpenAI embedding model."""
    openai_models = {
        "text-embedding-3-small",
        "text-embedding-3-large",
        "text-embedding-ada-002",
    }
    return model_name in openai_models or model_name.startswith("text-embedding")


def _infer_legacy_provider(settings: Settings, model_name: str) -> str:
    """Infer provider for compatibility when embedding_provider is omitted."""
    if _is_openai_model(model_name):
        return "openai"
    if settings.openai_base_url or settings.embedding_base_url:
        return "openai"
    if "/" in model_name:
        return "huggingface"
    return "openai"


def _resolve_model_name(
    *,
    settings: Settings,
    provider_name: str,
    model_override: Optional[str],
    provider_was_explicit: bool,
) -> str:
    """Resolve the effective model name for a provider."""
    if model_override:
        return model_override
    if (
        provider_was_explicit
        and provider_name != "openai"
        and settings.embedding_model == _SETTINGS_DEFAULT_EMBEDDING_MODEL
    ):
        return default_model_for_provider(provider_name)
    return settings.embedding_model or DEFAULT_MODELS[provider_name]


class EmbeddingModel:
    """
    Unified embedding model wrapper.

    The preferred configuration is explicit provider selection:

        >>> model = EmbeddingModel(
        ...     settings=Settings(
        ...         embedding_provider="openai",
        ...         embedding_model="text-embedding-3-small",
        ...     )
        ... )

    Supported providers: openai, huggingface, google/gemini, voyage, jina,
    mistral, ollama, local, and onnx.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        model: Optional[str] = None,
        instruction: Optional[str] = None,
        instruction_template: Optional[str] = None,
        provider: Optional[str] = None,
    ):
        """
        Initialize the embedding model.

        Args:
            settings: Configuration settings.
            model: Override embedding model from settings.
            instruction: Custom instruction for query encoding.
            instruction_template: Template style, such as "qwen3" or "bge".
            provider: Override embedding provider from settings.
        """
        self.settings = settings or get_settings()
        provider_value = provider or self.settings.embedding_provider
        provider_was_explicit = provider_value is not None

        legacy_model_name = model or self.settings.embedding_model
        if provider_value is None:
            provider_value = _infer_legacy_provider(self.settings, legacy_model_name)
            warnings.warn(
                "Inferring the embedding provider from embedding_model is deprecated and "
                "will be removed in v1.0.0. Set embedding_provider explicitly.",
                DeprecationWarning,
                stacklevel=2,
            )

        self.provider_name = canonicalize_provider(provider_value)
        self.model_name = _resolve_model_name(
            settings=self.settings,
            provider_name=self.provider_name,
            model_override=model,
            provider_was_explicit=provider_was_explicit,
        )
        self.instruction = instruction
        self.instruction_template = instruction_template

        if self.provider_name == "openai":
            self.settings.validate_settings()

        api_key = self.settings.embedding_api_key
        if self.provider_name == "openai" and api_key is None:
            api_key = self.settings.openai_api_key

        base_url = self.settings.embedding_base_url
        if self.provider_name == "openai" and base_url is None:
            base_url = self.settings.openai_base_url

        self._backend = get_provider(
            self.provider_name,
            model_name=self.model_name,
            batch_size=self.settings.batch_size,
            api_key=api_key,
            base_url=base_url,
            instruction=instruction,
            instruction_template=instruction_template,
        )
        self._dimension: Optional[int] = None

    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        if self._dimension is None:
            backend_dimension = getattr(self._backend, "dimension", None)
            if backend_dimension is not None:
                self._dimension = int(backend_dimension)
            else:
                test_embedding = self.embed("test")
                self._dimension = len(test_embedding)
        return self._dimension

    def embed(
        self,
        text: str,
        text_type: Literal["query", "document"] = "query",
    ) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: The text to embed.
            text_type: Type of text - "query" or "document".

        Returns:
            Embedding vector as a list of floats.
        """
        with start_span(
            "vgrag.embedding.embed",
            {
                "vgrag.embedding_provider": self.provider_name,
                "vgrag.embedding_model": self.model_name,
                "vgrag.text_type": text_type,
                "vgrag.text_count": 1,
                "vgrag.text_length": len(text),
            },
        ):
            embeddings = self._backend.encode(text, text_type=text_type)
            return embeddings[0].tolist()

    def embed_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = False,
        text_type: Literal["query", "document"] = "query",
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts with batching.

        Args:
            texts: List of texts to embed.
            batch_size: Number of texts per batch.
            show_progress: Whether to show progress bar.
            text_type: Type of text - "query" or "document".

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        batch_size = batch_size or self.settings.batch_size
        all_embeddings = []

        batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
        with start_span(
            "vgrag.embedding.batch",
            {
                "vgrag.embedding_provider": self.provider_name,
                "vgrag.embedding_model": self.model_name,
                "vgrag.text_type": text_type,
                "vgrag.text_count": len(texts),
                "vgrag.batch_size": batch_size,
                "vgrag.batch_count": len(batches),
            },
        ):
            iterator = tqdm(batches, desc="Generating embeddings") if show_progress else batches

            for batch in iterator:
                embeddings = self._backend.encode(batch, text_type=text_type)
                all_embeddings.extend(embeddings.tolist())

        return all_embeddings


__all__ = [
    "EmbeddingModel",
    "HuggingFaceEmbedding",
    "INSTRUCTION_TEMPLATES",
    "OpenAIEmbedding",
]
