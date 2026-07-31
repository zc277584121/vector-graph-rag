"""Optional OpenTelemetry helpers for Vector Graph RAG."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Iterator, Mapping, Optional

try:  # pragma: no cover - exercised when the optional dependency is installed.
    from opentelemetry import trace
except ImportError:  # pragma: no cover - the no-op path is covered by unit tests.
    trace = None  # type: ignore[assignment]


AttributeValue = str | bool | int | float | list[str] | list[bool] | list[int] | list[float]

_TRACER_NAME = "vector_graph_rag"
_OBSERVABILITY_CONTEXT: ContextVar[Dict[str, AttributeValue]] = ContextVar(
    "vector_graph_rag_observability_context",
    default={},
)


def _is_supported_scalar(value: Any) -> bool:
    """Return whether a value is safe for OpenTelemetry span attributes."""
    return isinstance(value, (str, bool, int, float))


def _normalize_attribute_value(value: Any) -> Optional[AttributeValue]:
    """Normalize a value to an OpenTelemetry-compatible attribute value."""
    if value is None:
        return None
    if _is_supported_scalar(value):
        return value
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list) and all(_is_supported_scalar(item) for item in value):
        return value
    return str(value)


def _clean_attributes(attributes: Optional[Mapping[str, Any]]) -> Dict[str, AttributeValue]:
    """Return attributes with empty keys and None values removed."""
    cleaned: Dict[str, AttributeValue] = {}
    if not attributes:
        return cleaned

    for key, value in attributes.items():
        if not isinstance(key, str) or not key.strip():
            continue
        normalized = _normalize_attribute_value(value)
        if normalized is not None:
            cleaned[key.strip()] = normalized
    return cleaned


def _standard_context_attributes(
    request_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
    graph_name: Optional[str] = None,
    source: Optional[str] = None,
) -> Dict[str, AttributeValue]:
    """Build namespaced observability attributes for common request context."""
    attributes: Dict[str, AttributeValue] = {}
    if request_id:
        attributes["vgrag.request_id"] = request_id
    if tenant_id:
        attributes["vgrag.tenant_id"] = tenant_id
    if graph_name:
        attributes["vgrag.graph_name"] = graph_name
    if source:
        attributes["vgrag.source"] = source
    return attributes


def get_observability_attributes() -> Dict[str, AttributeValue]:
    """Return the current observability context attributes."""
    return dict(_OBSERVABILITY_CONTEXT.get())


@contextmanager
def observability_context(
    request_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
    graph_name: Optional[str] = None,
    source: Optional[str] = None,
    attributes: Optional[Mapping[str, Any]] = None,
) -> Iterator[None]:
    """Attach low-sensitivity attributes to spans created inside the context."""
    context_attributes = {
        **_standard_context_attributes(
            request_id=request_id,
            tenant_id=tenant_id,
            graph_name=graph_name,
            source=source,
        ),
        **_clean_attributes(attributes),
    }
    current = get_observability_attributes()
    token = _OBSERVABILITY_CONTEXT.set({**current, **context_attributes})
    try:
        yield
    finally:
        _OBSERVABILITY_CONTEXT.reset(token)


def set_span_attributes(span: Any, attributes: Optional[Mapping[str, Any]]) -> None:
    """Set supported attributes on a span-like object."""
    if span is None:
        return
    for key, value in _clean_attributes(attributes).items():
        span.set_attribute(key, value)


@contextmanager
def start_span(name: str, attributes: Optional[Mapping[str, Any]] = None) -> Iterator[Any]:
    """Start an OpenTelemetry span if the optional dependency is installed."""
    if trace is None:
        yield None
        return

    tracer = trace.get_tracer(_TRACER_NAME)
    with tracer.start_as_current_span(name) as span:
        set_span_attributes(span, get_observability_attributes())
        set_span_attributes(span, attributes)
        yield span


__all__ = [
    "get_observability_attributes",
    "observability_context",
    "set_span_attributes",
    "start_span",
]
