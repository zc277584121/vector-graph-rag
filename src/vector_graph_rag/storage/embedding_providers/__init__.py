"""Embedding provider registry."""

from __future__ import annotations

import importlib
from typing import Any, Literal, Protocol, runtime_checkable

EmbeddingProviderName = Literal[
    "openai",
    "huggingface",
    "google",
    "gemini",
    "voyage",
    "jina",
    "mistral",
    "ollama",
    "local",
    "onnx",
]


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Minimal sync interface every embedding provider implements."""

    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def encode(self, texts, normalize: bool = True, text_type: str = "query"): ...


_ALIASES: dict[str, str] = {
    "gemini": "google",
}

_PROVIDERS: dict[str, tuple[str, str]] = {
    "openai": ("vector_graph_rag.storage.embedding_providers.openai", "OpenAIEmbedding"),
    "huggingface": (
        "vector_graph_rag.storage.embedding_providers.huggingface",
        "HuggingFaceEmbedding",
    ),
    "google": ("vector_graph_rag.storage.embedding_providers.google", "GoogleEmbedding"),
    "voyage": ("vector_graph_rag.storage.embedding_providers.voyage", "VoyageEmbedding"),
    "jina": ("vector_graph_rag.storage.embedding_providers.jina", "JinaEmbedding"),
    "mistral": ("vector_graph_rag.storage.embedding_providers.mistral", "MistralEmbedding"),
    "ollama": ("vector_graph_rag.storage.embedding_providers.ollama", "OllamaEmbedding"),
    "local": ("vector_graph_rag.storage.embedding_providers.local", "LocalEmbedding"),
    "onnx": ("vector_graph_rag.storage.embedding_providers.onnx", "OnnxEmbedding"),
}

DEFAULT_MODELS: dict[str, str] = {
    "openai": "text-embedding-3-small",
    "huggingface": "BAAI/bge-large-en-v1.5",
    "google": "gemini-embedding-001",
    "gemini": "gemini-embedding-001",
    "voyage": "voyage-3-lite",
    "jina": "jina-embeddings-v4",
    "mistral": "mistral-embed",
    "ollama": "nomic-embed-text",
    "local": "all-MiniLM-L6-v2",
    "onnx": "gpahal/bge-m3-onnx-int8",
}

_INSTALL_HINTS: dict[str, str] = {
    "openai": "uv sync",
    "huggingface": "uv sync --extra hf",
    "google": "uv sync --extra google",
    "voyage": "uv sync --extra voyage",
    "jina": "uv sync --extra jina",
    "mistral": "uv sync --extra mistral",
    "ollama": "uv sync --extra ollama",
    "local": "uv sync --extra local",
    "onnx": "uv sync --extra onnx",
}

_API_KEY_PROVIDERS = {"openai", "google", "voyage", "jina", "mistral"}
_BASE_URL_PROVIDERS = {"openai", "ollama"}


def canonicalize_provider(name: str) -> str:
    """Return the canonical provider name, resolving aliases."""
    normalized = name.strip().lower()
    normalized = _ALIASES.get(normalized, normalized)
    if normalized not in _PROVIDERS:
        raise ValueError(
            f"Unknown embedding provider {name!r}. "
            f"Available providers: {', '.join(supported_providers())}"
        )
    return normalized


def supported_providers() -> list[str]:
    """Return supported provider names, including aliases."""
    return sorted([*_PROVIDERS.keys(), *_ALIASES.keys()])


def default_model_for_provider(name: str) -> str:
    """Return the recommended default model for a provider."""
    return DEFAULT_MODELS[canonicalize_provider(name)]


def get_provider(
    name: str,
    *,
    model_name: str | None = None,
    batch_size: int = 0,
    api_key: str | None = None,
    base_url: str | None = None,
    instruction: str | None = None,
    instruction_template: str | None = None,
) -> EmbeddingProvider:
    """Instantiate an embedding provider by name."""
    provider_name = canonicalize_provider(name)
    module_path, class_name = _PROVIDERS[provider_name]
    hint = _INSTALL_HINTS.get(provider_name, f"install the {provider_name} dependencies")

    try:
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
    except ImportError as exc:
        raise ImportError(
            f"Embedding provider {name!r} is not installed: {exc}. Install with: {hint}"
        ) from exc

    kwargs: dict[str, Any] = {
        "model_name": model_name or DEFAULT_MODELS[provider_name],
    }
    if batch_size > 0:
        kwargs["batch_size"] = batch_size
    if api_key and provider_name in _API_KEY_PROVIDERS:
        kwargs["api_key"] = api_key
    if base_url and provider_name in _BASE_URL_PROVIDERS:
        kwargs["base_url"] = base_url
    if provider_name == "huggingface":
        kwargs["instruction"] = instruction
        kwargs["instruction_template"] = instruction_template

    try:
        return cls(**kwargs)
    except ImportError as exc:
        raise ImportError(
            f"Embedding provider {name!r} is not installed: {exc}. Install with: {hint}"
        ) from exc


__all__ = [
    "DEFAULT_MODELS",
    "EmbeddingProvider",
    "EmbeddingProviderName",
    "canonicalize_provider",
    "default_model_for_provider",
    "get_provider",
    "supported_providers",
]
