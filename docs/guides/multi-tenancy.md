# Multi-Tenancy

Vector Graph RAG supports practical dataset isolation patterns today, but it does not yet implement Zilliz partition-key-level tenancy. Choose the isolation level based on tenant count, operational needs, and query isolation requirements.

## Collection Prefix Isolation

Use `collection_prefix` when you want separate entity, relation, and passage collections for each graph, workspace, customer, or use case.

```python
from vector_graph_rag import VectorGraphRAG

legal_rag = VectorGraphRAG(
    milvus_uri="./data.db",
    collection_prefix="legal",
)

finance_rag = VectorGraphRAG(
    milvus_uri="./data.db",
    collection_prefix="finance",
)
```

With `collection_prefix="finance"`, the three collections are named like:

```text
finance_vgrag_entities
finance_vgrag_relations
finance_vgrag_passages
```

This is the clearest option for a small or moderate number of isolated graphs.

## Milvus Database Isolation

Use `milvus_db` when your Milvus deployment uses database-level separation:

```python
rag = VectorGraphRAG(
    milvus_uri="http://localhost:19530",
    milvus_db="tenant_a",
    collection_prefix="kb",
)
```

Database-level isolation is useful when the Milvus deployment and operations model already maps tenants or environments to databases.

## API Graph Names

The packaged REST API uses `graph_name` as a graph selector. Internally, it maps to separate `VectorGraphRAG` / `Graph` instances and collection prefixes.

```bash
curl -X POST "http://localhost:8000/query?graph_name=finance" \
  -H "content-type: application/json" \
  -d '{"question":"What changed this quarter?"}'
```

Use this when exposing multiple knowledge bases through the same API process.

## Metadata Filtering

You can store tenant or workspace metadata on passages and use Milvus filter expressions during query:

```python
rag.upsert_documents_by_source(
    documents=chunks,
    source="file:file-123",
    metadata={
        "tenant_id": "tenant-a",
        "workspace_id": "finance",
    },
)

result = rag.query(
    "What changed this quarter?",
    filter='tenant_id == "tenant-a" and workspace_id == "finance"',
)
```

Metadata filtering is flexible, but it is not the same as physical isolation. Use collection or database separation when tenants require stronger operational boundaries.

## Current Limitations

Vector Graph RAG does not currently provide:

- Zilliz partition key-level tenant routing
- tenant-aware schema fields that are fixed across all collections
- tenant lifecycle helpers for create/delete/migrate
- tenant-level authorization enforcement

Those can be layered in an application today, but they are not first-class storage features in the package yet.

## Recommendation

For a small number of tenants or use cases, start with `collection_prefix` or `milvus_db`. For many tenants, metadata filtering can reduce collection sprawl, but application-side authorization and careful filter construction become critical. Partition-key-level tenancy is a future storage design item rather than a completed feature.
