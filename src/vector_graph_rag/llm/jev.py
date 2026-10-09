"""Two-stage Jev scoring with shared relation and passage context."""

import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import tiktoken

from vector_graph_rag.config import Settings
from vector_graph_rag.llm.cache import get_llm_cache

INSTRUCTIONS = (
    "Could this candidate relation supply a specific fact needed to answer the question, "
    "including a necessary intermediate fact in a multi-hop answer? "
    "Use the other candidate relations in state as context."
)
CRITERIA = {
    "true": "The relation supplies the requested fact or a necessary intermediate link that "
    "identifies the entity needed for the next step.",
    "false": "The relation only shares a topic or entity with the question and does not supply "
    "a fact needed to resolve it.",
}


DOCUMENT_INSTRUCTIONS = (
    "Would this source excerpt be useful evidence to retrieve for answering the original question? "
    "It may establish the requested fact or identify an intermediate entity needed to connect other evidence. "
    "Use the other excerpts as context but require support in THIS excerpt. "
    "Distinguish relevant identity or relation evidence from incidental name overlap and unrelated attributes. "
    "Do not assume a complete answer is available. Judge usefulness, not whether this excerpt alone answers everything."
)


class JevReranker:
    """Score each relation independently against shared query/graph context.

    Scores are sorted stably; the top 64 relations seed passage candidates.
    Candidate merging belongs to the RAG layer, followed by passage scoring.
    The token estimator is a conservative proxy, not the provider's tokenizer.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        if not self.settings.uses_jev and self.settings.reranker_model is not None:
            raise ValueError("JevReranker requires a Jev reranker_model.")
        resolved = (
            self.settings
            if self.settings.uses_jev
            else self.settings.model_copy(update={"reranker_provider": "jev"})
        )
        self.connection = resolved.task_model("reranker")
        if not self.connection.api_key:
            raise ValueError("Set TYPESAFE_API_KEY or VGRAG_JEV_API_KEY to use Jev.")
        import httpx

        self._httpx = httpx
        self._encoding = tiktoken.get_encoding("cl100k_base")
        # Moving aliases must not reuse results produced by an older model release.
        moving_alias = self.connection.model in {"jev-latest", "jev-preview"}
        self._cache = get_llm_cache() if self.settings.use_llm_cache and not moving_alias else None

    def _tokens(self, value: object) -> int:
        return len(
            self._encoding.encode(json.dumps(value, ensure_ascii=False), disallowed_special=())
        )

    def build_requests(
        self, query: str, relation_ids: List[str], relation_texts: List[str]
    ) -> list:
        """Build complete payloads without making requests or truncating candidates."""
        if len(relation_ids) != len(relation_texts):
            raise ValueError("Relation IDs and texts must have equal lengths.")
        ids = [str(rid) for rid in relation_ids]
        if len(set(ids)) != len(ids):
            raise ValueError("Relation IDs must be unique.")
        if not ids:
            return []
        state = {
            "question": query,
            "candidate_graph": "\n".join(
                f"[{rid}] {text}" for rid, text in zip(ids, relation_texts)
            ),
        }
        questions = {
            f"r{rid}": {
                "type": "noul",
                "instructions": INSTRUCTIONS
                + f"\nTarget relation [{rid}]: {text}"
                + " Labeled demonstrations, if present, illustrate the rule; judge only the actual question and target relation.",
                "criteria": dict(CRITERIA),
            }
            for rid, text in zip(ids, relation_texts)
        }
        state_tokens = self._tokens(state)
        if state_tokens + max(self._tokens(q) for q in questions.values()) >= 23000:
            raise ValueError("Jev shared context is too large; reduce candidate retrieval limits.")
        batches = []
        current = {}
        used = state_tokens + 1024
        for key, question in questions.items():
            cost = self._tokens({key: question}) + 16
            if current and used + cost > 38000:
                batches.append(current)
                current = {}
                used = state_tokens + 1024
            current[key] = question
            used += cost
        if current:
            batches.append(current)
        payloads = [
            {"model": self.connection.model, "state": state, "questions": batch}
            for batch in batches
        ]
        if any(self._tokens(payload) >= 40000 for payload in payloads):
            raise ValueError("Jev request is too large; reduce candidate retrieval limits.")
        return payloads

    @staticmethod
    def _validate_response(payload: dict, response: dict) -> Dict[str, float]:
        answers = response.get("answers")
        if not isinstance(answers, dict) or set(answers) != set(payload["questions"]):
            raise ValueError("Jev returned missing or unexpected relation scores.")
        scores = {}
        for key, answer in answers.items():
            value = answer.get("noul") if isinstance(answer, dict) else None
            if (
                not isinstance(answer, dict)
                or answer.get("type") != "noul"
                or isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError("Jev returned an invalid noul score.")
            scores[key] = float(value)
        return scores

    def _request(self, client, payload: dict) -> dict:
        for attempt in range(3):
            try:
                response = client.post(
                    self.connection.base_url.rstrip("/") + "/systemone", json=payload
                )
                if response.status_code in {429, 500, 502, 503, 504, 529} and attempt < 2:
                    time.sleep(2**attempt)
                    continue
                response.raise_for_status()
                result = response.json()
                self._validate_response(payload, result)
                return result
            except self._httpx.TransportError:
                if attempt == 2:
                    raise
                time.sleep(2**attempt)
        raise RuntimeError("Jev request attempts exhausted.")

    def score_relations(
        self, query: str, relation_ids: List[str], relation_texts: List[str]
    ) -> Dict[str, float]:
        """Return all scores; API failures are errors, never empty successful selections."""
        scores = self._score_payloads(self.build_requests(query, relation_ids, relation_texts))
        return {str(rid): scores[f"r{rid}"] for rid in relation_ids}

    def _score_payloads(self, payloads: list) -> Dict[str, float]:
        """Share validation, retries and caching across both scoring stages."""
        scores = {}
        pending = []
        for payload in payloads:
            key = json.dumps(payload, ensure_ascii=False)
            if self.connection.base_url.rstrip("/") != "https://api.typesafe.ai/v1":
                key = self.connection.base_url + "\n" + key
            cached = self._cache.get(self.connection.model, key) if self._cache else None
            if cached is not None:
                scores.update(self._validate_response(payload, json.loads(cached)))
            else:
                pending.append((key, payload))
        if pending:
            with self._httpx.Client(
                headers={"Authorization": "Bearer " + self.connection.api_key.get_secret_value()},
                timeout=self.settings.jev_timeout,
            ) as client:
                with ThreadPoolExecutor(max_workers=self.settings.jev_max_concurrency) as pool:
                    responses = pool.map(lambda item: self._request(client, item[1]), pending)
                    # Serialize cache writes; the existing file cache is not thread-safe.
                    for (key, payload), response in zip(pending, responses):
                        scores.update(self._validate_response(payload, response))
                        if self._cache:
                            self._cache.set(self.connection.model, key, json.dumps(response))
        return scores

    def rerank(
        self, query: str, relation_ids: List[str], relation_texts: List[str]
    ) -> Tuple[List[str], List[str]]:
        scores = self.score_relations(query, relation_ids, relation_texts)
        ordered = sorted(zip(relation_ids, relation_texts), key=lambda pair: -scores[str(pair[0])])
        selected = ordered[:64]
        return [rid for rid, _ in selected], [text for _, text in selected]

    def build_passage_request(self, query: str, passages: List[str]) -> dict:
        """Build the frozen document-stage prompt without silently truncating text."""
        if len(passages) > 16:
            raise ValueError("The two-stage recipe accepts at most 16 candidate passages.")
        payload = {
            "model": self.connection.model,
            "state": {
                "question": query,
                "excerpts": {f"d{i}": text for i, text in enumerate(passages)},
            },
            "questions": {
                f"d{i}": {
                    "type": "noul",
                    "instructions": DOCUMENT_INSTRUCTIONS + f"\nTarget excerpt: d{i}.",
                    "criteria": {
                        "true": "Contains a supported answer fact or a necessary intermediate identity or connection.",
                        "false": "Irrelevant, wrong entity, unsupported connection, or merely related background.",
                    },
                }
                for i in range(len(passages))
            },
        }
        if self._tokens(payload["state"]) >= 23000 or self._tokens(payload) >= 47000:
            raise ValueError("Jev passage context is too large; use smaller source chunks.")
        return payload

    def rerank_passages(self, query: str, passages: List[str]) -> List[str]:
        if not passages:
            return []
        scores = self._score_payloads([self.build_passage_request(query, passages)])
        order = sorted(range(len(passages)), key=lambda i: -scores[f"d{i}"])
        return [passages[i] for i in order]


def merge_passage_candidates(graph: List[str], dense: List[str]) -> List[str]:
    """Alternate graph and dense candidates, deduplicate exact text, cap at 16."""
    dense = dense[:10]
    result = []
    seen = set()
    for i in range(max(len(graph), len(dense))):
        for source in (graph, dense):
            if i < len(source) and source[i] not in seen:
                seen.add(source[i])
                result.append(source[i])
                if len(result) == 16:
                    return result
    return result
