"""Candidate assembly and document scoring regression tests."""

from unittest.mock import MagicMock

import pytest

from vector_graph_rag import Settings
from vector_graph_rag.llm.jev import JevReranker, merge_passage_candidates
from vector_graph_rag.rag import VectorGraphRAG


def test_round_robin_respects_database_order_and_filter():
    rag = object.__new__(VectorGraphRAG)
    rag._store = MagicMock()
    rag._store._get_relations_by_ids.return_value = [
        {"id": "b", "passage_ids": ["b1", "shared"]},
        {"id": "a", "passage_ids": ["a1", "shared", "a3"]},
    ]
    rag._store.get_passages_by_ids.return_value = [
        {"id": x, "text": x} for x in ["a3", "shared", "b1", "a1"]
    ]
    ids, _ = rag._get_passages_from_relations(["a", "b"], filter='tenant == "x"', round_robin=True)
    assert ids == ["a1", "b1", "shared", "a3"]
    rag._store.get_passages_by_ids.assert_called_once_with(ids, filter='tenant == "x"')


def test_candidate_merge_preserves_source_order_and_bounds():
    assert merge_passage_candidates(["a", "b", "c"], ["b", "d", "a"]) == ["a", "b", "d", "c"]
    result = merge_passage_candidates([f"g{i}" for i in range(100)], [f"d{i}" for i in range(20)])
    assert result == [x for i in range(8) for x in (f"g{i}", f"d{i}")]
    assert merge_passage_candidates([], [str(i) for i in range(20)]) == [str(i) for i in range(10)]


def test_passage_scores_ties_full_text_and_limits():
    ranker = JevReranker(Settings(reranker_model="jev", jev_api_key="test", use_llm_cache=False))
    docs = ["prefix " * 300 + "required bridge", "answer", "noise"]
    payload = ranker.build_passage_request("question", docs)
    assert payload["state"]["excerpts"]["d0"] == docs[0]
    ranker._score_payloads = MagicMock(return_value={"d0": 0.8, "d1": 0.8, "d2": 0.1})
    assert ranker.rerank_passages("question", docs) == docs
    assert ranker.rerank_passages("question", []) == []
    with pytest.raises(ValueError, match="at most 16"):
        ranker.build_passage_request("q", ["x"] * 17)
    with pytest.raises(ValueError, match="too large"):
        ranker.build_passage_request("q", ["x " * 25000])


def test_top64_keeps_bridge_even_below_old_threshold():
    ranker = JevReranker(Settings(jev_api_key="test", use_llm_cache=False))
    ranker.score_relations = MagicMock(return_value={str(i): 0.01 for i in range(70)})
    ids = [str(i) for i in range(70)]
    assert ranker.rerank("q", ids, ids)[0] == ids[:64]


def test_empty_graph_still_scores_dense_and_propagates_errors():
    rag = object.__new__(VectorGraphRAG)
    rag._get_passages_from_relations = MagicMock(return_value=([], []))
    retriever = MagicMock()
    retriever.retrieve_passages_naive.return_value = ["dense noise", "dense evidence"]
    rag._reranker = MagicMock()
    rag._reranker.rerank_passages.return_value = ["dense evidence", "dense noise"]
    assert rag._jev_passages("q", [], retriever, 1, "tenant == 1") == ["dense evidence"]
    retriever.retrieve_passages_naive.assert_called_once_with("q", top_k=10, filter="tenant == 1")
    rag._reranker.rerank_passages.side_effect = RuntimeError("API failure")
    with pytest.raises(RuntimeError, match="API failure"):
        rag._jev_passages("q", [], retriever, 1, "tenant == 1")
