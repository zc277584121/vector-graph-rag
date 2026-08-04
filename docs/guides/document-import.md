# Document Import

Vector Graph RAG includes a lightweight document import layer for turning files, URLs, and text into LangChain `Document` chunks. The importer is intentionally pluggable: simple workflows can use the default converter, while more demanding document pipelines can provide a stronger parser adapter.

## Installation

Install loader dependencies when you need URL fetching or local file conversion:

```bash
uv add "vector-graph-rag[loaders]"
```

Optional parser adapters have their own extras:

```bash
uv add "vector-graph-rag[docling]"
uv add "vector-graph-rag[mineru]"
```

## Import URLs And Files

```python
from vector_graph_rag import VectorGraphRAG
from vector_graph_rag.loaders import DocumentImporter

importer = DocumentImporter(chunk_size=1000, chunk_overlap=200)
result = importer.import_sources(
    [
        "https://en.wikipedia.org/wiki/Albert_Einstein",
        "/path/to/document.pdf",
        "/path/to/report.docx",
    ]
)

rag = VectorGraphRAG(collection_prefix="demo")
rag.rebuild_documents(result.documents, extract_triplets=True)
```

The default local-file converter uses MarkItDown for common document formats. URL content is fetched and converted to text before chunking.

## Use Docling

Use Docling when you want a document parser with richer PDF layout handling:

```python
from vector_graph_rag.loaders import DoclingConverter, DocumentImporter

importer = DocumentImporter(
    converter=DoclingConverter(),
    chunk_size=1000,
    chunk_overlap=200,
)

result = importer.import_sources(["/path/to/report.pdf"])
```

Docling may download parser models on first use. In offline or restricted environments, pre-download those assets and pass a preconfigured Docling converter through `converter_kwargs`.

## Use MinerU

Use MinerU when your parsing pipeline already depends on MinerU or when you need its document processing accuracy:

```python
from vector_graph_rag.loaders import DocumentImporter, MinerUConverter

importer = DocumentImporter(
    converter=MinerUConverter(
        output_dir="./mineru-output",
        backend="pipeline",
    ),
    chunk_size=1000,
    chunk_overlap=200,
)

result = importer.import_sources(["/path/to/report.pdf"])
```

Follow MinerU's setup guide for parser models and runtime backends. Vector Graph RAG reads the Markdown output and returns standard LangChain `Document` chunks.

## Connect Import To Incremental Updates

For source-level updates, every chunk produced from the same external object should carry the same stable source value.

```python
result = importer.import_sources(["/path/to/report.pdf"])

rag.upsert_documents_by_source(
    result.documents,
    source="file:report-123",
    extract_triplets=True,
)
```

If your parser extracts triplets itself, store them in each chunk's `metadata["triplets"]` and disable LLM extraction:

```python
rag.upsert_documents_by_source(
    result.documents,
    source="file:report-123",
    extract_triplets=False,
)
```

See [Incremental Updates](incremental-updates.md) for source key design, retry behavior, and delete flows.

## API Reference

The importer and converters are documented in the [Python API Reference](../reference/python-api.md#documentimporter):

- [`DocumentImporter`](../reference/python-api.md#documentimporter)
- [`DoclingConverter`](../reference/python-api.md#doclingconverter)
- [`MinerUConverter`](../reference/python-api.md#mineruconverter)
