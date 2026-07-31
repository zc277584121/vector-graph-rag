"""
Docling converter integration.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from vector_graph_rag.observability import observability_context, start_span

from .converter import ConversionResult

try:
    from docling.document_converter import DocumentConverter as DoclingDocumentConverter

    HAS_DOCLING = True
except ImportError:
    DoclingDocumentConverter = None
    HAS_DOCLING = False


class DoclingConverter:
    """
    Convert local files to Markdown through Docling.

    Docling is used through its public Python API and returns Markdown text that
    can flow through the existing DocumentImporter, chunker, and RAG ingestion
    pipeline.
    """

    supported_extensions = {
        ".pdf",
        ".docx",
        ".pptx",
        ".xlsx",
        ".doc",
        ".ppt",
        ".xls",
        ".odt",
        ".odp",
        ".ods",
        ".epub",
        ".png",
        ".jpg",
        ".jpeg",
        ".tiff",
        ".tif",
        ".bmp",
    }

    def __init__(
        self,
        converter: Optional[Any] = None,
        converter_kwargs: Optional[Dict[str, Any]] = None,
        export_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the Docling converter.

        Args:
            converter: Optional preconfigured Docling DocumentConverter.
            converter_kwargs: Keyword arguments used to create Docling's
                DocumentConverter when converter is omitted.
            export_kwargs: Keyword arguments passed to export_to_markdown().
        """
        if converter is not None:
            self.converter = converter
        else:
            if not HAS_DOCLING or DoclingDocumentConverter is None:
                raise ImportError(
                    "docling is not installed. Install with: uv add 'vector-graph-rag[docling]'"
                )
            self.converter = DoclingDocumentConverter(**(converter_kwargs or {}))

        self.export_kwargs = dict(export_kwargs or {})

    def convert(self, source: str) -> ConversionResult:
        """
        Convert a single file to Markdown with Docling.

        Args:
            source: File path to convert.

        Returns:
            ConversionResult with one Markdown document or conversion errors.
        """
        path = Path(source)
        with observability_context(source=str(path)):
            with start_span(
                "vgrag.convert_document",
                {
                    "vgrag.parser": "docling",
                    "vgrag.source_type": path.suffix.lower().lstrip("."),
                },
            ):
                if not path.exists():
                    return ConversionResult(documents=[], errors=[f"File not found: {source}"])

                ext = path.suffix.lower()
                if ext not in self.supported_extensions:
                    return ConversionResult(
                        documents=[], errors=[f"Unsupported file type for Docling: {ext}"]
                    )

                try:
                    result = self.converter.convert(str(path))
                    docling_document = getattr(result, "document", None)
                    if docling_document is None:
                        return ConversionResult(
                            documents=[],
                            errors=[
                                f"Docling completed but no document output was found for {source}"
                            ],
                        )

                    content = docling_document.export_to_markdown(**self.export_kwargs).strip()
                    if not content:
                        return ConversionResult(
                            documents=[],
                            errors=[f"Docling produced an empty Markdown document for {source}"],
                        )

                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": str(path),
                            "source_type": ext[1:],
                            "parser": "docling",
                        },
                    )
                    return ConversionResult(documents=[doc])

                except Exception as exc:
                    return ConversionResult(
                        documents=[],
                        errors=[f"Failed to convert {source} with Docling: {str(exc)}"],
                    )

    def convert_batch(self, sources: List[str]) -> ConversionResult:
        """Convert multiple files."""
        all_documents = []
        all_errors = []

        for source in sources:
            result = self.convert(source)
            all_documents.extend(result.documents)
            all_errors.extend(result.errors)

        return ConversionResult(documents=all_documents, errors=all_errors)
