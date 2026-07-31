# Observability

Vector Graph RAG can emit OpenTelemetry traces for ingestion, retrieval, LLM, embedding, loader, and Milvus operations. This is optional instrumentation: the library creates spans when OpenTelemetry is installed, but your application remains responsible for configuring the SDK and exporter.

## Installation

Install the observability extra:

```bash
pip install "vector-graph-rag[observability]"
# or
uv add "vector-graph-rag[observability]"
```

For local development or tests, install an OpenTelemetry SDK/exporter in the application environment. For example:

```bash
uv add opentelemetry-sdk
```

## Configure OpenTelemetry

Configure OpenTelemetry once in your application before creating or using `VectorGraphRAG`:

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)
```

You can replace `ConsoleSpanExporter` with any exporter or distribution supported by your deployment. For managed observability platforms, configure the platform's OpenTelemetry package in the host application; Vector Graph RAG does not add platform-specific dependencies.

## Add Request Context

Use `observability_context()` to attach safe request-level attributes to all spans created inside a call chain:

```python
from vector_graph_rag import VectorGraphRAG, observability_context

rag = VectorGraphRAG(collection_prefix="finance")

with observability_context(
    request_id="req-123",
    tenant_id="tenant-a",
    graph_name="finance",
    source="file-123",
    attributes={
        "app.import_job_id": "job-456",
        "app.parser": "custom-parser",
    },
):
    rag.upsert_documents_by_source(
        chunks,
        source="file-123",
        extract_triplets=False,
    )
```

The standard context fields are:

| Field | Purpose |
|---|---|
| `request_id` | Correlate application requests with Vector Graph RAG spans. |
| `tenant_id` | Identify the tenant or workspace for multi-tenant deployments. |
| `graph_name` | Identify the knowledge base, graph, or collection prefix. |
| `source` | Identify the external source being imported, updated, or deleted. |
| `attributes` | Add application-specific low-sensitivity attributes. |

Custom `attributes` should use a namespace such as `app.import_job_id` or `app.pipeline`. Attribute values are normalized to OpenTelemetry-compatible scalar or scalar-list values.

## Span Coverage

The built-in spans cover the main pipeline stages:

| Area | Example spans |
|---|---|
| Ingestion | `vgrag.rebuild_documents`, `vgrag.upsert_documents_by_source`, `vgrag.delete_documents_by_source` |
| Loading | `vgrag.import_sources`, `vgrag.import_source`, `vgrag.chunk_text`, `vgrag.convert_document`, `vgrag.fetch_url` |
| Extraction | `vgrag.extract_triplets.batch`, `vgrag.extract_triplets`, `vgrag.extract_query_entities` |
| Embeddings | `vgrag.embedding.batch`, `vgrag.embedding.embed` |
| Storage | `vgrag.milvus.insert`, `vgrag.milvus.upsert`, `vgrag.milvus.search`, `vgrag.milvus.query`, `vgrag.milvus.delete` |
| Retrieval | `vgrag.retrieve.graph`, `vgrag.retrieve.entities`, `vgrag.retrieve.relations`, `vgrag.subgraph.expand` |
| Generation | `vgrag.rerank`, `vgrag.answer` |

The packaged REST API also attaches request context from `x-request-id` or `x-correlation-id`, `x-tenant-id`, and the graph name when available.

## Data Safety

By default, Vector Graph RAG records operational metadata such as counts, booleans, model names, collection names, operation names, and safe context attributes. It does not record document text, prompt text, query text, generated answers, filters, or full URLs as span attributes.

Use stable internal identifiers for `tenant_id`, `source`, and custom attributes when those values could expose sensitive information.

## No-Op Behavior

If OpenTelemetry is not installed, `observability_context()` still keeps local context for the active block and `start_span()` becomes a no-op. This keeps the core package lightweight and avoids requiring observability dependencies in small deployments.
