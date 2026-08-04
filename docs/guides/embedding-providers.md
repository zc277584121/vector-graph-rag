# Embedding Providers

Vector Graph RAG supports explicit embedding provider selection. New applications should set both `embedding_provider` and `embedding_model` instead of relying on model-name inference.

## Supported Providers

| Provider | Extra | Typical use |
|---|---|---|
| `openai` | built in | OpenAI or OpenAI-compatible embedding endpoints. |
| `huggingface` | `hf` | Local HuggingFace transformers models. |
| `google` / `gemini` | `google` | Google Gemini embeddings. |
| `voyage` | `voyage` | Voyage AI embeddings. |
| `jina` | `jina` | Jina AI embeddings. |
| `mistral` | `mistral` | Mistral embeddings. |
| `ollama` | `ollama` | Local Ollama embedding server. |
| `local` | `local` | Sentence Transformers style local embeddings. |
| `onnx` | `onnx` | ONNX Runtime embedding models. |

Install only the provider extras you need:

```bash
uv add "vector-graph-rag[jina]"
uv add "vector-graph-rag[ollama]"
uv add "vector-graph-rag[all]"
```

## Configure A Provider

```python
from vector_graph_rag import VectorGraphRAG

rag = VectorGraphRAG(
    embedding_provider="openai",
    embedding_model="text-embedding-3-small",
)
```

For providers that need a separate key or endpoint, use `embedding_api_key` and `embedding_base_url`:

```python
rag = VectorGraphRAG(
    embedding_provider="jina",
    embedding_model="jina-embeddings-v4",
    embedding_api_key="jina_...",
)
```

```python
rag = VectorGraphRAG(
    embedding_provider="ollama",
    embedding_model="nomic-embed-text",
    embedding_base_url="http://localhost:11434",
)
```

## Dimension Consistency

All entities, relations, and passages in one graph must use the same embedding model and dimension. If you change embedding providers or embedding dimensions, rebuild that graph or use a new `collection_prefix`.

```python
rag = VectorGraphRAG(
    collection_prefix="finance_openai_small",
    embedding_provider="openai",
    embedding_model="text-embedding-3-small",
)
```

## Legacy Provider Inference

If `embedding_provider` is omitted, Vector Graph RAG keeps a legacy compatibility path that infers the provider from `embedding_model`. That path is deprecated and planned for removal in v1.0.0. Prefer explicit configuration:

```python
rag = VectorGraphRAG(
    embedding_provider="openai",
    embedding_model="text-embedding-3-small",
)
```

See the [Python API Reference](../reference/python-api.md#embedding-providers) for constructor parameters and provider examples.
