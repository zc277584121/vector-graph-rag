"""HuggingFace transformers embedding provider."""

from __future__ import annotations

from typing import List, Literal, Optional, Union

import numpy as np

from vector_graph_rag.storage.embedding_providers.utils import (
    ensure_text_list,
    filter_empty_texts,
    restore_empty_embeddings,
)

INSTRUCTION_TEMPLATES = {
    "qwen3": {
        "query": "Instruct: {instruction}\nQuery: {text}",
        "document": "{text}",
        "default_instruction": "Given a question, retrieve passages that contain the answer",
    },
    "bge": {
        "query": "{instruction}: {text}",
        "document": "{text}",
        "default_instruction": "Represent this sentence for searching relevant passages",
    },
}


def _get_model_family(model_name: str) -> Optional[str]:
    """Detect model family from model name for instruction templates."""
    model_lower = model_name.lower()
    if "qwen" in model_lower and "embed" in model_lower:
        return "qwen3"
    if "bge" in model_lower:
        return "bge"
    return None


def _mean_pooling(token_embeddings, attention_mask):
    """Mean pooling with attention mask."""
    token_embeddings = token_embeddings.masked_fill(~attention_mask[..., None].bool(), 0.0)
    sentence_embeddings = token_embeddings.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    return sentence_embeddings


class HuggingFaceEmbedding:
    """HuggingFace embedding model wrapper."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        *,
        batch_size: int = 0,
        device: Optional[str] = None,
        instruction: Optional[str] = None,
        instruction_template: Optional[str] = None,
    ) -> None:
        del batch_size
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "HuggingFace embedding models require optional dependencies. "
                "Install with: uv sync --extra hf."
            ) from exc

        self._model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._torch = torch
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model.eval()

        self.model_family = _get_model_family(model_name)
        self.instruction = instruction
        self.instruction_template = instruction_template
        if self.instruction and not self.instruction_template and self.model_family:
            self.instruction_template = self.model_family
        self._dimension: Optional[int] = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._dimension = len(self.encode("test")[0])
        return self._dimension

    def _apply_instruction(
        self, texts: List[str], text_type: Literal["query", "document"] = "query"
    ) -> List[str]:
        """Apply instruction template to texts if configured."""
        if not self.instruction or not self.instruction_template:
            return texts

        template_config = INSTRUCTION_TEMPLATES.get(self.instruction_template)
        if not template_config:
            return texts

        template = template_config.get(text_type, "{text}")
        instruction = self.instruction or template_config.get("default_instruction", "")
        return [template.format(instruction=instruction, text=t) for t in texts]

    def encode(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True,
        text_type: Literal["query", "document"] = "query",
    ) -> np.ndarray:
        """Encode texts to embeddings."""
        texts = ensure_text_list(texts)
        valid_indices, valid_texts = filter_empty_texts(texts)
        if not valid_texts:
            return np.zeros((len(texts), self.dimension))

        processed_texts = self._apply_instruction(valid_texts, text_type)
        torch = self._torch

        with torch.no_grad():
            inputs = self.tokenizer(
                processed_texts, padding=True, truncation=True, return_tensors="pt", max_length=512
            ).to(self.device)
            outputs = self.model(**inputs)
            embeddings = _mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])

            if normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            valid_embeddings = embeddings.float().cpu().numpy()
            return restore_empty_embeddings(
                valid_embeddings, valid_indices, len(texts), valid_embeddings.shape[1]
            )
