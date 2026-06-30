# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Vector Graph RAG implements Graph RAG using **pure vector search in Milvus** — no graph database (Neo4j etc.). Knowledge-graph entities and relations are stored as vectors; the graph's adjacency is encoded as ID lists in each record's metadata, and "graph traversal" is implemented as repeated `query`/`search` calls against Milvus by those IDs. The key algorithmic bet is replacing iterative LLM agent loops (IRCoT, multi-step reflection) with a **single-pass LLM reranking** step.

## Commands

This is a `uv` + `pyproject.toml` project. Use `uv` directly — never `pip` or `uv pip`.

```bash
uv sync --extra dev          # install with dev deps (pytest, ruff)
uv sync --extra all          # everything (dev + api + hf)

uv run pytest                # run all tests
uv run pytest tests/test_graph.py            # single file
uv run pytest tests/test_graph.py::test_name # single test
uv run pytest -k metadata_filter             # by keyword

uv run ruff check .          # lint
uv run ruff format .         # format

# API server (canonical entrypoint — used by README and Dockerfile)
uv sync --extra api
uv run uvicorn vector_graph_rag.api.app:app --host 0.0.0.0 --port 8000

# Frontend (React + Vite)
cd frontend && npm install && npm run dev

# Evaluation (multi-hop QA benchmarks)
uv run python evaluation/evaluate.py --dataset musique --method graph
```

Tests mock OpenAI and use Milvus Lite (a temp `.db` file per test via the `temp_milvus_uri` fixture in `tests/conftest.py`), so they run offline with no API key. `asyncio_mode = "auto"`, so async tests need no decorator.

## Architecture

### Indexing → Query pipeline

```
add_documents:  Documents → TripletExtractor (LLM) → GraphBuilder → embeddings → MilvusStore
query:          Question → EntityExtractor → vector search → SubGraph.expand → LLMReranker → AnswerGenerator
```

`VectorGraphRAG` (`src/vector_graph_rag/rag.py`) is the user-facing facade that wires these components together. Read it first — its `add_documents` and `query` methods are the spine of the whole system.

### Three Milvus collections

`MilvusStore` (`storage/milvus.py`) manages **three collections** — `entities`, `relations`, `passages` (prefixed by `collection_prefix`). The graph structure lives entirely in metadata:

- **Entity** metadata → `relation_ids`, `passage_ids`
- **Relation** metadata → `entity_ids` (head, tail), `passage_ids`, plus structured `subject`/`predicate`/`object`
- **Passage** metadata → `entity_ids`, `relation_ids`, **plus arbitrary user metadata** (used for the `filter` feature)

Subgraph expansion (`graph/knowledge_graph.py`, `SubGraph`) is **lazy**: it holds IDs and fetches neighbor records from Milvus on demand during `expand(degree=...)` rather than loading the full graph into memory.

### Important data-flow detail

`add_documents` **drops and recreates all collections every call** (`rag.py` ~line 403) — it is a full rebuild, not an append. The retriever (`self._retriever`) is reset to `None` after any add and lazily rebuilt on the next query via `_ensure_retriever()`.

### Retrieval specifics (`graph/retriever.py`)

- Entity and relation searches each apply a **similarity threshold** (`entity_similarity_threshold` default `0.9`; relation default `-1.0` = keep all). Scores are kept when `score > threshold`.
- **Eviction strategy**: if expanded relations exceed `relation_number_threshold` (default 1000), a vector search re-ranks them down to the threshold. Below it, relations are sorted by ID to match HippoRAG's deterministic behavior.
- The `filter` parameter (a Milvus filter expression on passage metadata) flows through retrieval by first resolving allowed passage IDs, then filtering relations to those touching an allowed passage.

### Component map

- `llm/extractor.py` — `TripletExtractor` (doc → triplets) and `EntityExtractor` (query → entities); `processing_phrases` normalizes entity strings. Has an NER cache for evaluation (HippoRAG TSV format).
- `llm/reranker.py` — `LLMReranker` (single-pass relation reranking) and `AnswerGenerator`.
- `llm/cache.py` — disk cache for LLM responses (`use_llm_cache`).
- `graph/builder.py` — `GraphBuilder` turns triplets into entity/relation/passage records + adjacency maps.
- `graph/graph.py` — `Graph`, a higher-level CRUD interface over `MilvusStore` (used mainly by the API layer); entity/relation methods are private, users work at the passage level.
- `storage/embeddings.py` — `EmbeddingModel`, supports OpenAI and (optional `[hf]` extra) HuggingFace models.

### Configuration

`config.py` `Settings` is a pydantic-settings `BaseSettings`. All settings read from env with the **`VGRAG_` prefix** (e.g. `VGRAG_LLM_MODEL`) or a `.env` file. `VectorGraphRAG.__init__` kwargs override env. The OpenAI key falls back to plain `OPENAI_API_KEY`. Note the default `embedding_model` is `text-embedding-3-large` (dimension 3072) in `Settings`, while the `create_rag` factory and docstrings default to `text-embedding-3-small` — keep `embedding_dimension` consistent with whatever model you set.

### Two API directories — do not confuse them

- `src/vector_graph_rag/api/app.py` — **the canonical, packaged API.** This is what the README, Dockerfile, and `uvicorn vector_graph_rag.api.app:app` use. Edit this one.
- `api/` (repo root) — a separate standalone variant with a `sys.path` hack and its own `schemas.py`. Not the packaged entrypoint; treat as legacy unless a task explicitly concerns it.

## Conventions

- Code and comments in English. User-facing replies follow the user's language.
- Methods prefixed with `_` on `MilvusStore`/`Graph` (e.g. `_insert_entities`, `_search_relations`) are internal entity/relation operations — the intended public surface is passage-level.
- `Document` is `langchain_core.documents.Document`; pre-extracted triplets are passed via `metadata["triplets"]` as `[subject, predicate, object]` lists.
- PyTorch install index is backend-specific (CPU on macOS/arm Linux, CUDA 12.4 on x86 Linux) via `[tool.uv.sources]`; the Dockerfile rewrites this based on the `TORCH_BACKEND` build arg.
