"""End-to-end tests for incremental source updates in VectorGraphRAG."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from vector_graph_rag.config import Settings
from vector_graph_rag.graph.retriever import GraphRetriever
from vector_graph_rag.models import Document
from vector_graph_rag.rag import VectorGraphRAG


class FakeEmbeddingModel:
    """Small deterministic embedding model for Milvus Lite tests."""

    dimension = 4

    def embed(self, text: str, text_type: str = "query") -> list[float]:
        text = text.lower()
        if "alpha" in text or "blue" in text or "green" in text:
            return [1.0, 0.0, 0.0, 0.0]
        if "beta" in text or "red" in text:
            return [0.0, 1.0, 0.0, 0.0]
        if "gamma" in text or "yellow" in text:
            return [0.0, 0.0, 1.0, 0.0]
        return [0.5, 0.5, 0.0, 0.0]

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int | None = None,
        show_progress: bool = False,
        text_type: str = "query",
    ) -> list[list[float]]:
        return [self.embed(text, text_type=text_type) for text in texts]


class FakeEntityExtractor:
    """Deterministic entity extractor for query-time graph retrieval."""

    def extract(self, question: str) -> list[str]:
        return ["alpha"]


def create_test_rag(milvus_uri: str, collection_prefix: str) -> VectorGraphRAG:
    """Create a VectorGraphRAG instance with fake embeddings and no LLM calls."""
    settings = Settings(
        milvus_uri=milvus_uri,
        openai_api_key="test-api-key",
        embedding_provider="openai",
        collection_prefix=collection_prefix,
        final_top_k=3,
    )
    fake_embedding = FakeEmbeddingModel()

    with patch("vector_graph_rag.rag.EmbeddingModel", return_value=fake_embedding):
        rag = VectorGraphRAG(settings=settings)

    rag._answer_generator.generate = MagicMock(
        side_effect=lambda question, passages: "\n".join(passages)
    )
    return rag


def with_temp_rag(collection_prefix: str):
    """Create a temporary Milvus Lite backed RAG instance."""
    temp_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    temp_file.close()
    rag = create_test_rag(temp_file.name, collection_prefix)
    return rag, temp_file.name


def remove_temp_milvus_file(milvus_uri: str) -> None:
    """Remove a temporary Milvus Lite file if it exists."""
    if os.path.exists(milvus_uri):
        os.unlink(milvus_uri)


def doc(
    text: str,
    triplets: list[list[str]],
    id: str | None = None,
    metadata: dict | None = None,
) -> Document:
    """Build a test document with pre-extracted triplets."""
    return Document(
        page_content=text,
        metadata={"triplets": triplets, **(metadata or {})},
        id=id,
    )


def test_upsert_documents_by_source_adds_without_rebuilding_existing_sources():
    """Upsert a new source without removing sources already in the graph."""
    rag, milvus_uri = with_temp_rag("incremental_add")

    try:
        rag.upsert_documents_by_source(
            [
                doc(
                    "Alpha owns the blue database.",
                    [["Alpha", "owns", "blue database"]],
                )
            ],
            source="file_alpha",
            extract_triplets=False,
            show_progress=False,
        )
        rag.upsert_documents_by_source(
            [
                doc(
                    "Beta owns the red database.",
                    [["Beta", "owns", "red database"]],
                )
            ],
            source="file_beta",
            extract_triplets=False,
            show_progress=False,
        )

        alpha_passages = rag._store.get_passages_by_source("file_alpha")
        beta_passages = rag._store.get_passages_by_source("file_beta")

        assert [p["text"] for p in alpha_passages] == ["Alpha owns the blue database."]
        assert [p["text"] for p in beta_passages] == ["Beta owns the red database."]
    finally:
        remove_temp_milvus_file(milvus_uri)


def test_upsert_documents_by_source_infers_source_and_sets_chunk_metadata():
    """Infer source from metadata and persist source/chunk metadata."""
    rag, milvus_uri = with_temp_rag("incremental_infer_source")

    try:
        result = rag.upsert_documents_by_source(
            [
                doc(
                    "Alpha owns the blue database.",
                    [["Alpha", "owns", "blue database"]],
                    id="alpha_chunk_0",
                    metadata={"source": "file_alpha", "page": 3},
                )
            ],
            extract_triplets=False,
            show_progress=False,
        )

        passages = rag._store.get_passages_by_ids(
            ["alpha_chunk_0"],
            output_fields=[
                "id",
                "text",
                "source",
                "page",
                "chunk_index",
                "document_id",
                "chunk_id",
            ],
        )

        assert result.documents[0].id == "alpha_chunk_0"
        assert result.documents[0].metadata["source"] == "file_alpha"
        assert result.documents[0].metadata["chunk_index"] == 0
        assert passages == [
            {
                "id": "alpha_chunk_0",
                "text": "Alpha owns the blue database.",
                "source": "file_alpha",
                "page": 3,
                "chunk_index": 0,
            }
        ]
    finally:
        remove_temp_milvus_file(milvus_uri)


def test_upsert_documents_by_source_replaces_only_target_source_and_cleans_orphans():
    """Replace one source and clean graph records that only belonged to it."""
    rag, milvus_uri = with_temp_rag("incremental_replace")

    try:
        rag.upsert_documents_by_source(
            [
                doc(
                    "Alpha owns the blue database.",
                    [["Alpha", "owns", "blue database"]],
                )
            ],
            source="file_alpha",
            extract_triplets=False,
            show_progress=False,
        )
        alpha_passage_id = rag._store.get_passages_by_source("file_alpha")[0]["id"]

        rag.upsert_documents_by_source(
            [
                doc(
                    "Alpha owns the red database.",
                    [["Alpha", "owns", "red database"]],
                )
            ],
            source="file_beta",
            extract_triplets=False,
            show_progress=False,
        )

        rag.upsert_documents_by_source(
            [
                doc(
                    "Alpha owns the green database.",
                    [["Alpha", "owns", "green database"]],
                )
            ],
            source="file_alpha",
            extract_triplets=False,
            show_progress=False,
        )

        alpha_passages = rag._store.get_passages_by_source("file_alpha")
        beta_passages = rag._store.get_passages_by_source("file_beta")
        assert [p["text"] for p in alpha_passages] == ["Alpha owns the green database."]
        assert [p["text"] for p in beta_passages] == ["Alpha owns the red database."]
        assert alpha_passages[0]["id"] == alpha_passage_id

        relations = rag._store._get_relations_by_texts(
            [
                "alpha owns blue database",
                "alpha owns red database",
                "alpha owns green database",
            ]
        )
        assert "alpha owns blue database" not in relations
        assert set(relations) == {
            "alpha owns red database",
            "alpha owns green database",
        }

        alpha_entity = rag._store._get_entities_by_texts(["alpha"])["alpha"]
        assert set(alpha_entity["passage_ids"]) == {alpha_passages[0]["id"], beta_passages[0]["id"]}
        assert set(alpha_entity["relation_ids"]) == {
            relations["alpha owns red database"]["id"],
            relations["alpha owns green database"]["id"],
        }
    finally:
        remove_temp_milvus_file(milvus_uri)


def test_delete_documents_by_source_cascades_and_preserves_shared_graph_records():
    """Delete one source while preserving shared relation/entity records."""
    rag, milvus_uri = with_temp_rag("incremental_delete")

    try:
        shared_triplet = [["Alpha", "founded", "Acme"]]
        rag.upsert_documents_by_source(
            [doc("Alpha founded Acme in source A.", shared_triplet)],
            source="file_alpha",
            extract_triplets=False,
            show_progress=False,
        )
        rag.upsert_documents_by_source(
            [doc("Alpha founded Acme in source B.", shared_triplet)],
            source="file_beta",
            extract_triplets=False,
            show_progress=False,
        )

        beta_passage_id = rag._store.get_passages_by_source("file_beta")[0]["id"]
        relation = rag._store._get_relations_by_texts(["alpha founded acme"])["alpha founded acme"]
        assert len(relation["passage_ids"]) == 2

        assert rag.delete_documents_by_source("file_alpha") is True
        assert rag._store.get_passages_by_source("file_alpha") == []

        relation = rag._store._get_relations_by_texts(["alpha founded acme"])["alpha founded acme"]
        assert relation["passage_ids"] == [beta_passage_id]
        alpha_entity = rag._store._get_entities_by_texts(["alpha"])["alpha"]
        assert alpha_entity["passage_ids"] == [beta_passage_id]
        assert alpha_entity["relation_ids"] == [relation["id"]]

        assert rag.delete_documents_by_source("file_beta") is True
        assert rag._store._get_relations_by_texts(["alpha founded acme"]) == {}
        assert rag._store._get_entities_by_texts(["alpha"]) == {}
        assert rag.delete_documents_by_source("file_missing") is False
    finally:
        remove_temp_milvus_file(milvus_uri)


def test_incremental_exact_text_lookups_are_batched():
    """Batch exact entity/relation lookups instead of querying one text at a time."""
    rag, milvus_uri = with_temp_rag("incremental_batched_lookup")
    rag.settings.batch_size = 2

    try:
        rag._store._insert_entities(
            ["alpha", "beta", "gamma"],
            ids=["entity_alpha", "entity_beta", "entity_gamma"],
            embeddings=[
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
        )
        rag._store._insert_relations(
            [
                "alpha owns beta",
                "beta owns gamma",
                "gamma owns alpha",
            ],
            ids=["relation_alpha_beta", "relation_beta_gamma", "relation_gamma_alpha"],
            embeddings=[
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
        )

        with patch.object(rag._store.client, "query", wraps=rag._store.client.query) as query_spy:
            entities = rag._store._get_entities_by_texts(
                ["alpha", "beta", "alpha", "gamma", "missing"]
            )

        assert set(entities) == {"alpha", "beta", "gamma"}
        assert query_spy.call_count == 2
        assert all("text in [" in call.kwargs["filter"] for call in query_spy.call_args_list)

        with patch.object(rag._store.client, "query", wraps=rag._store.client.query) as query_spy:
            relations = rag._store._get_relations_by_texts(
                [
                    "alpha owns beta",
                    "beta owns gamma",
                    "alpha owns beta",
                    "gamma owns alpha",
                    "missing relation",
                ]
            )

        assert set(relations) == {
            "alpha owns beta",
            "beta owns gamma",
            "gamma owns alpha",
        }
        assert query_spy.call_count == 2
        assert all("text in [" in call.kwargs["filter"] for call in query_spy.call_args_list)
    finally:
        remove_temp_milvus_file(milvus_uri)


def test_upsert_documents_by_source_batches_shared_graph_metadata_updates():
    """Reuse shared graph records without per-record ID lookups on incremental insert."""
    rag, milvus_uri = with_temp_rag("incremental_batched_upsert")

    try:
        shared_triplet = [["Alpha", "founded", "Acme"]]
        rag.upsert_documents_by_source(
            [doc("Alpha founded Acme in source A.", shared_triplet)],
            source="file_alpha",
            extract_triplets=False,
            show_progress=False,
        )

        with (
            patch.object(
                rag._store,
                "_get_entities_by_ids",
                wraps=rag._store._get_entities_by_ids,
            ) as entity_lookup_spy,
            patch.object(
                rag._store,
                "_get_relations_by_ids",
                wraps=rag._store._get_relations_by_ids,
            ) as relation_lookup_spy,
            patch.object(rag._store.client, "upsert", wraps=rag._store.client.upsert) as upsert_spy,
        ):
            rag.upsert_documents_by_source(
                [doc("Alpha founded Acme in source B.", shared_triplet)],
                source="file_beta",
                extract_triplets=False,
                show_progress=False,
            )

        assert entity_lookup_spy.call_count == 0
        assert relation_lookup_spy.call_count == 0

        entity_upserts = [
            call
            for call in upsert_spy.call_args_list
            if call.kwargs["collection_name"] == rag._store.entity_collection
        ]
        relation_upserts = [
            call
            for call in upsert_spy.call_args_list
            if call.kwargs["collection_name"] == rag._store.relation_collection
        ]
        assert len(entity_upserts) == 1
        assert len(entity_upserts[0].kwargs["data"]) == 2
        assert len(relation_upserts) == 1
        assert len(relation_upserts[0].kwargs["data"]) == 1
    finally:
        remove_temp_milvus_file(milvus_uri)


def test_upsert_documents_by_source_supports_metadata_filter_query_end_to_end():
    """Use upserted source metadata to filter graph query results."""
    rag, milvus_uri = with_temp_rag("incremental_query_filter")

    try:
        rag.upsert_documents_by_source(
            [
                doc(
                    "Alpha owns the blue database.",
                    [["Alpha", "owns", "blue database"]],
                )
            ],
            source="file_alpha",
            metadata={"tenant_id": "team_a"},
            extract_triplets=False,
            show_progress=False,
        )
        rag.upsert_documents_by_source(
            [
                doc(
                    "Alpha owns the red database.",
                    [["Alpha", "owns", "red database"]],
                )
            ],
            source="file_beta",
            metadata={"tenant_id": "team_b"},
            extract_triplets=False,
            show_progress=False,
        )
        rag._retriever = GraphRetriever(
            store=rag._store,
            graph_builder=rag._graph_builder,
            settings=rag.settings,
            embedding_model=rag._embedding_model,
            entity_extractor=FakeEntityExtractor(),
        )

        result = rag.query(
            "What database does Alpha own?",
            use_reranking=False,
            filter='tenant_id == "team_a" and source == "file_alpha"',
        )

        assert result.passages == ["Alpha owns the blue database."]
        assert result.retrieved_passages == ["Alpha owns the blue database."]
        assert "red database" not in result.answer
    finally:
        remove_temp_milvus_file(milvus_uri)


def test_upsert_documents_by_source_supports_custom_source_field():
    """Use a custom source metadata field for update and delete."""
    rag, milvus_uri = with_temp_rag("incremental_custom_source_field")

    try:
        rag.upsert_documents_by_source(
            [
                doc(
                    "Gamma owns the yellow database.",
                    [["Gamma", "owns", "yellow database"]],
                    metadata={"file_id": "file_gamma"},
                )
            ],
            source_field="file_id",
            extract_triplets=False,
            show_progress=False,
        )

        passages = rag._store.get_passages_by_source("file_gamma", source_field="file_id")
        assert [p["text"] for p in passages] == ["Gamma owns the yellow database."]
        assert rag.delete_documents_by_source("file_gamma", source_field="file_id") is True
        assert rag._store.get_passages_by_source("file_gamma", source_field="file_id") == []
    finally:
        remove_temp_milvus_file(milvus_uri)


def test_upsert_documents_by_source_validates_source_contract():
    """Reject source-less, multi-source, conflicting, and unsafe source fields."""
    rag, milvus_uri = with_temp_rag("incremental_source_contract")

    try:
        with pytest.raises(ValueError, match='metadata\\["source"\\] or source'):
            rag.upsert_documents_by_source(
                [doc("Alpha owns blue.", [["Alpha", "owns", "blue"]])],
                extract_triplets=False,
                show_progress=False,
            )

        with pytest.raises(ValueError, match="expects one source per call"):
            rag.upsert_documents_by_source(
                [
                    doc(
                        "Alpha owns blue.",
                        [["Alpha", "owns", "blue"]],
                        metadata={"source": "file_alpha"},
                    ),
                    doc(
                        "Beta owns red.",
                        [["Beta", "owns", "red"]],
                        metadata={"source": "file_beta"},
                    ),
                ],
                extract_triplets=False,
                show_progress=False,
            )

        with pytest.raises(ValueError, match="differ from source"):
            rag.upsert_documents_by_source(
                [
                    doc(
                        "Alpha owns blue.",
                        [["Alpha", "owns", "blue"]],
                        metadata={"source": "file_beta"},
                    )
                ],
                source="file_alpha",
                extract_triplets=False,
                show_progress=False,
            )

        with pytest.raises(ValueError, match="simple metadata field name"):
            rag.upsert_documents_by_source(
                [doc("Alpha owns blue.", [["Alpha", "owns", "blue"]])],
                source="file_alpha",
                source_field="source.field",
                extract_triplets=False,
                show_progress=False,
            )

        with pytest.raises(ValueError, match="non-empty string"):
            rag.delete_documents_by_source(" ")
    finally:
        remove_temp_milvus_file(milvus_uri)


def test_legacy_incremental_document_apis_raise_migration_errors():
    """Legacy document_id incremental APIs fail with explicit migration guidance."""
    rag, milvus_uri = with_temp_rag("incremental_legacy_errors")

    try:
        with pytest.raises(RuntimeError, match="upsert_documents_by_source"):
            rag.upsert_documents(
                document_id="file_alpha",
                documents=[doc("Alpha owns blue.", [["Alpha", "owns", "blue"]])],
                extract_triplets=False,
            )

        with pytest.raises(RuntimeError, match="delete_documents_by_source"):
            rag.delete_documents("file_alpha")
    finally:
        remove_temp_milvus_file(milvus_uri)
