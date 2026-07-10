"""Tests for embedding provider selection."""

from __future__ import annotations

import numpy as np
import pytest

from vector_graph_rag.config import Settings
from vector_graph_rag.storage import embeddings as embeddings_module
from vector_graph_rag.storage.embedding_providers import (
    canonicalize_provider,
    default_model_for_provider,
    supported_providers,
)
from vector_graph_rag.storage.embedding_providers.utils import (
    filter_empty_texts,
    restore_empty_embeddings,
)
from vector_graph_rag.storage.embeddings import EmbeddingModel


class DummyProvider:
    """Small provider used to avoid real SDK calls in routing tests."""

    model_name = "dummy"
    dimension = 3

    def encode(self, texts, normalize=True, text_type="query"):
        if isinstance(texts, str):
            texts = [texts]
        return np.array([[float(len(text)), 1.0, 0.0] for text in texts])


def patch_provider_factory(monkeypatch):
    """Patch the provider factory and return captured calls."""
    calls = []

    def fake_get_provider(name, **kwargs):
        calls.append({"name": name, **kwargs})
        return DummyProvider()

    monkeypatch.setattr(embeddings_module, "get_provider", fake_get_provider)
    return calls


def test_supported_provider_registry():
    """Registry exposes all supported provider names and aliases."""
    providers = supported_providers()

    assert canonicalize_provider("gemini") == "google"
    assert default_model_for_provider("gemini") == "gemini-embedding-001"
    assert {
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
    }.issubset(set(providers))


def test_unknown_provider_raises():
    """Unknown providers fail with a clear configuration error."""
    with pytest.raises(ValueError, match="Unknown embedding provider"):
        canonicalize_provider("cohere")


def test_empty_embedding_rows_are_restored():
    """Provider helpers restore zero vectors for empty text rows."""
    valid_indices, valid_texts = filter_empty_texts(["alpha", "", "  ", "beta"])
    valid_embeddings = np.array([[1.0, 2.0], [3.0, 4.0]])

    restored = restore_empty_embeddings(valid_embeddings, valid_indices, 4, 2)

    assert valid_texts == ["alpha", "beta"]
    assert restored.tolist() == [[1.0, 2.0], [0.0, 0.0], [0.0, 0.0], [3.0, 4.0]]


def test_explicit_provider_is_passed_to_factory(monkeypatch):
    """Explicit provider selection is the primary routing path."""
    calls = patch_provider_factory(monkeypatch)
    settings = Settings(
        openai_api_key="openai-key",
        embedding_provider="jina",
        embedding_model="jina-embeddings-v4",
        embedding_api_key="jina-key",
    )

    model = EmbeddingModel(settings=settings)

    assert model.provider_name == "jina"
    assert model.model_name == "jina-embeddings-v4"
    assert calls[0]["name"] == "jina"
    assert calls[0]["api_key"] == "jina-key"
    assert model.embed("hello") == [5.0, 1.0, 0.0]


def test_explicit_provider_uses_provider_default_model(monkeypatch):
    """Provider defaults are used when the caller only specifies provider."""
    calls = patch_provider_factory(monkeypatch)
    settings = Settings(
        openai_api_key="openai-key",
        embedding_provider="ollama",
        embedding_model=Settings.model_fields["embedding_model"].default,
    )

    model = EmbeddingModel(settings=settings)

    assert model.provider_name == "ollama"
    assert model.model_name == "nomic-embed-text"
    assert calls[0]["model_name"] == "nomic-embed-text"


def test_openai_provider_uses_embedding_overrides(monkeypatch):
    """Embedding-specific key/base URL override the OpenAI LLM settings."""
    calls = patch_provider_factory(monkeypatch)
    settings = Settings(
        openai_api_key="llm-key",
        openai_base_url="https://llm.example/v1",
        embedding_provider="openai",
        embedding_model="custom-embed",
        embedding_api_key="embedding-key",
        embedding_base_url="https://embedding.example/v1",
    )

    EmbeddingModel(settings=settings)

    assert calls[0]["api_key"] == "embedding-key"
    assert calls[0]["base_url"] == "https://embedding.example/v1"


def test_legacy_inference_prefers_huggingface_for_repo_ids(monkeypatch):
    """Legacy inference remains for compatibility but warns."""
    calls = patch_provider_factory(monkeypatch)
    settings = Settings(openai_api_key="openai-key", embedding_model="BAAI/bge-large-en-v1.5")

    with pytest.deprecated_call(match="embedding_provider"):
        model = EmbeddingModel(settings=settings)

    assert model.provider_name == "huggingface"
    assert calls[0]["name"] == "huggingface"


def test_legacy_inference_prefers_openai_compatible_base_url(monkeypatch):
    """OpenAI-compatible base URLs route no-slash models to OpenAI."""
    calls = patch_provider_factory(monkeypatch)
    settings = Settings(
        openai_api_key="openai-key",
        embedding_model="nomic-embed-text",
        embedding_base_url="http://localhost:11434/v1",
    )

    with pytest.deprecated_call(match="embedding_provider"):
        model = EmbeddingModel(settings=settings)

    assert model.provider_name == "openai"
    assert calls[0]["name"] == "openai"
    assert calls[0]["base_url"] == "http://localhost:11434/v1"
