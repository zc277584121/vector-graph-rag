"""End-to-end tests for incremental document updates in VectorGraphRAG."""

import os
import tempfile
from unittest.mock import MagicMock, patch

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


def doc(text: str, triplets: list[list[str]], id: str | None = None) -> Document:
    """Build a test document with pre-extracted triplets."""
    return Document(page_content=text, metadata={"triplets": triplets}, id=id)


def test_upsert_document_adds_without_rebuilding_existing_documents():
    """Upsert a new document without removing documents already in the graph."""
    rag, milvus_uri = with_temp_rag("incremental_add")

    try:
        rag.upsert_document(
            "file_alpha",
            [
                doc(
                    "Alpha owns the blue database.",
                    [["Alpha", "owns", "blue database"]],
                )
            ],
            extract_triplets=False,
            show_progress=False,
        )
        rag.upsert_document(
            "file_beta",
            [
                doc(
                    "Beta owns the red database.",
                    [["Beta", "owns", "red database"]],
                )
            ],
            extract_triplets=False,
            show_progress=False,
        )

        alpha_passages = rag._store.get_passages_by_document_id("file_alpha")
        beta_passages = rag._store.get_passages_by_document_id("file_beta")

        assert [p["text"] for p in alpha_passages] == ["Alpha owns the blue database."]
        assert [p["text"] for p in beta_passages] == ["Beta owns the red database."]
    finally:
        remove_temp_milvus_file(milvus_uri)


def test_upsert_document_replaces_only_target_document_and_cleans_orphans():
    """Replace one document and clean graph records that only belonged to it."""
    rag, milvus_uri = with_temp_rag("incremental_replace")

    try:
        rag.upsert_document(
            "file_alpha",
            [
                doc(
                    "Alpha owns the blue database.",
                    [["Alpha", "owns", "blue database"]],
                )
            ],
            extract_triplets=False,
            show_progress=False,
        )
        alpha_passage_id = rag._store.get_passages_by_document_id("file_alpha")[0]["id"]

        rag.upsert_document(
            "file_beta",
            [
                doc(
                    "Alpha owns the red database.",
                    [["Alpha", "owns", "red database"]],
                )
            ],
            extract_triplets=False,
            show_progress=False,
        )

        rag.upsert_document(
            "file_alpha",
            [
                doc(
                    "Alpha owns the green database.",
                    [["Alpha", "owns", "green database"]],
                )
            ],
            extract_triplets=False,
            show_progress=False,
        )

        alpha_passages = rag._store.get_passages_by_document_id("file_alpha")
        beta_passages = rag._store.get_passages_by_document_id("file_beta")
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


def test_delete_document_cascades_and_preserves_shared_graph_records():
    """Delete one document while preserving shared relation/entity records."""
    rag, milvus_uri = with_temp_rag("incremental_delete")

    try:
        shared_triplet = [["Alpha", "founded", "Acme"]]
        rag.upsert_document(
            "file_alpha",
            [doc("Alpha founded Acme in source A.", shared_triplet)],
            extract_triplets=False,
            show_progress=False,
        )
        rag.upsert_document(
            "file_beta",
            [doc("Alpha founded Acme in source B.", shared_triplet)],
            extract_triplets=False,
            show_progress=False,
        )

        beta_passage_id = rag._store.get_passages_by_document_id("file_beta")[0]["id"]
        relation = rag._store._get_relations_by_texts(["alpha founded acme"])["alpha founded acme"]
        assert len(relation["passage_ids"]) == 2

        assert rag.delete_document("file_alpha") is True
        assert rag._store.get_passages_by_document_id("file_alpha") == []

        relation = rag._store._get_relations_by_texts(["alpha founded acme"])["alpha founded acme"]
        assert relation["passage_ids"] == [beta_passage_id]
        alpha_entity = rag._store._get_entities_by_texts(["alpha"])["alpha"]
        assert alpha_entity["passage_ids"] == [beta_passage_id]
        assert alpha_entity["relation_ids"] == [relation["id"]]

        assert rag.delete_document("file_beta") is True
        assert rag._store._get_relations_by_texts(["alpha founded acme"]) == {}
        assert rag._store._get_entities_by_texts(["alpha"]) == {}
        assert rag.delete_document("file_missing") is False
    finally:
        remove_temp_milvus_file(milvus_uri)


def test_upsert_document_supports_metadata_filter_query_end_to_end():
    """Use upserted document metadata to filter graph query results."""
    rag, milvus_uri = with_temp_rag("incremental_query_filter")

    try:
        rag.upsert_document(
            "file_alpha",
            [
                doc(
                    "Alpha owns the blue database.",
                    [["Alpha", "owns", "blue database"]],
                )
            ],
            metadata={"tenant_id": "team_a"},
            extract_triplets=False,
            show_progress=False,
        )
        rag.upsert_document(
            "file_beta",
            [
                doc(
                    "Alpha owns the red database.",
                    [["Alpha", "owns", "red database"]],
                )
            ],
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
            filter='tenant_id == "team_a"',
        )

        assert result.passages == ["Alpha owns the blue database."]
        assert result.retrieved_passages == ["Alpha owns the blue database."]
        assert "red database" not in result.answer
    finally:
        remove_temp_milvus_file(milvus_uri)
