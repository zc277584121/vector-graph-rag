"""Task routing, credential boundaries and compatibility."""

from unittest.mock import patch

import pytest

from vector_graph_rag import ModelConfig, Settings
from vector_graph_rag.llm.extractor import EntityExtractor, TripletExtractor
from vector_graph_rag.llm.reranker import AnswerGenerator, LLMReranker


def test_defaults_and_task_overrides():
    s = Settings(
        openai_api_key="shared",
        llm_model="default",
        extractor_model="extract",
        answer_model="answer",
        reranker_model="jev",
        jev_api_key="decision",
    )
    assert s.task_model("extractor").model == "extract"
    assert s.task_model("answer").model == "answer"
    assert s.task_model("reranker").model == "jev-1.13.0"
    assert s.task_model("reranker").api_key.get_secret_value() == "decision"
    assert Settings(llm_model="old").task_model("answer").model == "old"


def test_custom_connections_and_redaction():
    s = Settings(
        openai_api_key="old",
        llm_model=ModelConfig(
            model="default", api_key="private", base_url="https://default.example/v1"
        ),
        answer_model=ModelConfig(
            model="answer", api_key="answer-key", base_url="https://answer.example/v1"
        ),
        reranker_model="jev",
        jev_api_key="jev-key",
    )
    assert s.for_task("extractor").openai_base_url == "https://default.example/v1"
    assert s.for_task("answer").openai_api_key == "answer-key"
    assert s.task_model("reranker").base_url == "https://api.typesafe.ai/v1"
    assert "private" not in s.model_dump_json()
    assert "answer-key" not in repr(s)
    s.answer_model = ModelConfig(model="answer", base_url="https://another.example/v1")
    with pytest.raises(ValueError, match="API key"):
        s.for_task("answer")


@pytest.mark.parametrize("task", ["extractor_model", "answer_model", "llm_model"])
def test_unsupported_jev_tasks(task):
    with pytest.raises(ValueError, match="does not support Jev"):
        Settings(**{task: "jev"})


def test_conflicts_and_legacy_configuration():
    assert (
        Settings(reranker_provider="jev", jev_model="jev-test").task_model("reranker").model
        == "jev-test"
    )
    with pytest.raises(ValueError, match="conflicts"):
        Settings(reranker_provider="llm", reranker_model="jev")
    with pytest.raises(ValueError, match="conflicts"):
        Settings(reranker_model="jev-one", jev_model="jev-two")


def test_components_use_their_task_connection():
    s = Settings(
        openai_api_key="base",
        llm_model="default",
        extractor_model="extract",
        reranker_model="rank",
        answer_model=ModelConfig(
            model="answer", api_key="local", base_url="https://answer.example/v1"
        ),
        use_llm_cache=False,
    )
    with (
        patch("vector_graph_rag.llm.extractor.OpenAI"),
        patch("vector_graph_rag.llm.reranker.OpenAI") as client,
    ):
        assert TripletExtractor(s).model == "extract"
        assert EntityExtractor(s).model == "extract"
        assert LLMReranker(s).model == "rank"
        answer = AnswerGenerator(s)
        assert answer.model == "answer"
        client.assert_called_with(api_key="local", base_url="https://answer.example/v1")
    assert s.llm_model == "default"


def test_endpoint_cache_isolation():
    s = Settings(openai_base_url="https://one.example/v1")
    t = Settings(openai_base_url="https://two.example/v1")
    assert s.cache_model_name("same") != t.cache_model_name("same")
    assert Settings().cache_model_name("same") == "same"


def test_environment_task_configuration(monkeypatch):
    monkeypatch.setenv("VGRAG_RERANKER_MODEL", "jev")
    monkeypatch.setenv(
        "VGRAG_ANSWER_MODEL",
        '{"model":"custom","base_url":"https://x.example/v1","api_key":"secret"}',
    )
    s = Settings(_env_file=None)
    assert s.uses_jev
    assert s.task_model("answer").model == "custom"


def test_facade_accepts_task_models_without_global_generation_key():
    from vector_graph_rag import VectorGraphRAG

    with (
        patch("vector_graph_rag.rag.EmbeddingModel"),
        patch("vector_graph_rag.rag.MilvusStore"),
        patch("vector_graph_rag.rag.GraphBuilder"),
        patch("vector_graph_rag.rag.TripletExtractor"),
        patch("vector_graph_rag.rag.AnswerGenerator"),
    ):
        rag = VectorGraphRAG(
            settings=Settings(
                openai_api_key=None,
                extractor_model=ModelConfig(model="extract", api_key="extract-key"),
                answer_model=ModelConfig(model="answer", api_key="answer-key"),
                reranker_model=ModelConfig(model="jev", api_key="decision-key"),
            )
        )
        assert rag._reranker.connection.api_key.get_secret_value() == "decision-key"


def test_constructor_shortcuts_and_factory():
    from vector_graph_rag import VectorGraphRAG, create_rag

    with (
        patch("vector_graph_rag.rag.EmbeddingModel"),
        patch("vector_graph_rag.rag.MilvusStore"),
        patch("vector_graph_rag.rag.GraphBuilder"),
        patch("vector_graph_rag.rag.TripletExtractor"),
        patch("vector_graph_rag.rag.AnswerGenerator"),
        patch("vector_graph_rag.rag.LLMReranker"),
    ):
        for factory in (VectorGraphRAG, create_rag):
            rag = factory(
                openai_api_key="test",
                extractor_model="extract",
                reranker_model="rank",
                answer_model="answer",
            )
            assert rag.settings.task_model("answer").model == "answer"
            assert rag.settings.task_model("reranker").model == "rank"
