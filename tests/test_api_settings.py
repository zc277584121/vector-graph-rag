"""Settings response compatibility without database or model calls."""

import pytest
from fastapi.testclient import TestClient

from vector_graph_rag.api.app import create_app
from vector_graph_rag.config import ModelConfig, Settings


@pytest.mark.parametrize(
    "model",
    [
        "custom-model",
        ModelConfig(
            model="custom-model",
            base_url="https://example.com/v1",
            api_key="private-model-key",
        ),
    ],
)
def test_settings_returns_model_name_without_credentials(model):
    settings = Settings(llm_model=model, openai_api_key="private-shared-key", _env_file=None)
    client = TestClient(create_app(settings))
    response = client.get("/settings")
    assert response.status_code == 200
    assert response.json()["llm_model"] == "custom-model"
    assert "private-model-key" not in response.text
    assert "private-shared-key" not in response.text
