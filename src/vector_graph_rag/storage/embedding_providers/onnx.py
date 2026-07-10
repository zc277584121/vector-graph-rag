"""ONNX Runtime embedding provider."""

from __future__ import annotations

import json
import os
from contextlib import suppress
from pathlib import Path
from typing import List, Literal, Union

import numpy as np

from vector_graph_rag.storage.embedding_providers.utils import (
    ensure_text_list,
    filter_empty_texts,
    normalize_embeddings,
    restore_empty_embeddings,
)


def _infer_max_length(session, default: int = 8192) -> int:
    """Infer max sequence length from an ONNX session."""
    try:
        for inp in session.get_inputs():
            if inp.name != "input_ids":
                continue
            shape = getattr(inp, "shape", None) or []
            if len(shape) > 1 and isinstance(shape[1], int) and shape[1] > 0:
                return min(default, shape[1])
    except Exception:
        pass
    return default


def _tokenizer_config_max_length(path: str | None, default: int) -> int:
    """Read tokenizer max length when available."""
    if not path:
        return default
    try:
        value = json.loads(Path(path).read_text()).get("model_max_length")
    except Exception:
        return default
    if isinstance(value, int) and 0 < value < 1_000_000_000:
        return min(default, value)
    return default


class OnnxEmbedding:
    """ONNX Runtime embedding provider."""

    def __init__(
        self,
        model_name: str = "gpahal/bge-m3-onnx-int8",
        *,
        batch_size: int = 0,
    ) -> None:
        del batch_size
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "ONNX embedding provider requires optional dependencies. "
                "Install with: uv sync --extra onnx."
            ) from exc

        from huggingface_hub import hf_hub_download, list_repo_files
        from tokenizers import Tokenizer

        self._cache_dir = (
            Path(os.environ.get("VGRAG_HOME") or (Path.home() / ".vector_graph_rag")) / "onnx-cache"
        )
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        tok_path, model_path, tok_cfg_path = self._download_model_files(
            model_name, hf_hub_download, list_repo_files
        )

        self._tokenizer = Tokenizer.from_file(tok_path)
        self._tokenizer.enable_padding(pad_id=1, pad_token="<pad>")

        self._session = ort.InferenceSession(model_path)
        self._input_names = [i.name for i in self._session.get_inputs()]
        self._output_names = [o.name for o in self._session.get_outputs()]
        max_length = _tokenizer_config_max_length(tok_cfg_path, _infer_max_length(self._session))
        self._tokenizer.enable_truncation(max_length=max_length)
        self._has_dense_vecs = "dense_vecs" in self._output_names
        self._model_name = model_name

        probe = self._encode(["hello"], normalize=True)
        self._dimension = len(probe[0])

    def _download_model_files(self, model_name, hf_hub_download, list_repo_files):
        """Download tokenizer and ONNX model, preferring cached files."""
        kwargs = {"cache_dir": str(self._cache_dir)}

        try:
            tok_path = hf_hub_download(
                model_name, "tokenizer.json", local_files_only=True, **kwargs
            )
            model_path = None
            onnx_file = None
            for candidate in ("model_quantized.onnx", "model.onnx"):
                try:
                    model_path = hf_hub_download(
                        model_name, candidate, local_files_only=True, **kwargs
                    )
                    onnx_file = candidate
                    break
                except Exception:
                    continue
            if model_path is None:
                raise FileNotFoundError("No cached ONNX model found")
            tok_cfg_path = None
            with suppress(Exception):
                hf_hub_download(model_name, onnx_file + "_data", local_files_only=True, **kwargs)
            with suppress(Exception):
                tok_cfg_path = hf_hub_download(
                    model_name, "tokenizer_config.json", local_files_only=True, **kwargs
                )
            return tok_path, model_path, tok_cfg_path
        except Exception:
            pass

        tok_path = hf_hub_download(model_name, "tokenizer.json", **kwargs)
        tok_cfg_path = None
        with suppress(Exception):
            tok_cfg_path = hf_hub_download(model_name, "tokenizer_config.json", **kwargs)
        repo_files = list_repo_files(model_name)
        onnx_files = [f for f in repo_files if f.endswith(".onnx")]
        if not onnx_files:
            raise ValueError(f"No .onnx files found in {model_name}")
        if "model_quantized.onnx" in onnx_files:
            onnx_file = "model_quantized.onnx"
        elif "model.onnx" in onnx_files:
            onnx_file = "model.onnx"
        else:
            onnx_file = onnx_files[0]
        data_file = onnx_file + "_data"
        if data_file in repo_files:
            hf_hub_download(model_name, data_file, **kwargs)
        model_path = hf_hub_download(model_name, onnx_file, **kwargs)
        return tok_path, model_path, tok_cfg_path

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True,
        text_type: Literal["query", "document"] = "query",
    ) -> np.ndarray:
        """Encode texts to embeddings."""
        del text_type
        texts = ensure_text_list(texts)
        valid_indices, valid_texts = filter_empty_texts(texts)
        if not valid_texts:
            return np.zeros((len(texts), self.dimension))
        embeddings = np.array(self._encode(valid_texts, normalize=normalize), dtype=float)
        return restore_empty_embeddings(embeddings, valid_indices, len(texts), self.dimension)

    def _encode(self, texts: List[str], normalize: bool) -> List[List[float]]:
        """Run the ONNX model."""
        encoded = self._tokenizer.encode_batch(texts)
        input_ids = np.array([e.ids for e in encoded])
        attention_mask = np.array([e.attention_mask for e in encoded])
        feed = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = np.zeros_like(input_ids)

        outputs = self._session.run(None, feed)
        if self._has_dense_vecs:
            idx = self._output_names.index("dense_vecs")
            embeddings = outputs[idx]
        else:
            idx = self._output_names.index("last_hidden_state")
            embeddings = outputs[idx][:, 0, :]

        if normalize:
            embeddings = normalize_embeddings(embeddings)
        return embeddings.tolist()
