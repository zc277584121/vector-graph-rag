"""
Unified document importer for Vector Graph RAG.

Supports importing from:
- Local files: PDF, DOCX, TXT, MD, HTML
- URLs: Web pages (fetched and converted)

Focus: Text documents only.
"""

from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from pydantic import BaseModel, ConfigDict

from vector_graph_rag.observability import observability_context, start_span

from .chunker import TextChunker
from .converter import ConversionResult, DocumentConverter, DocumentConverterProtocol
from .docling import DoclingConverter
from .mineru import MinerUConverter
from .url_fetcher import URLFetcher


class LoaderResult(BaseModel):
    """Result of document loading."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    documents: List[Document]
    errors: List[str] = []


class DocumentImporter:
    """
    Unified document importer for text documents.

    Supported formats:
    - PDF (via MarkItDown)
    - DOCX (via MarkItDown)
    - URLs (via trafilatura)
    - TXT, MD, HTML (passthrough)

    Example:
        importer = DocumentImporter(chunk_documents=True)
        result = importer.import_sources([
            "/path/to/document.pdf",
            "/path/to/report.docx",
            "https://example.com/article",
        ])
        for doc in result.documents:
            print(doc.page_content)
    """

    TEXT_EXTENSIONS = {".txt", ".md", ".html", ".htm"}

    def __init__(
        self,
        chunk_documents: bool = True,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        converter: Optional[DocumentConverterProtocol] = None,
    ):
        self.converter = converter or DocumentConverter()
        self.url_fetcher = URLFetcher(converter=self.converter)
        self.chunker = TextChunker(chunk_size, chunk_overlap) if chunk_documents else None

    @property
    def supported_extensions(self) -> set[str]:
        """Return file extensions supported by passthrough readers or converter."""
        converter_extensions = getattr(self.converter, "supported_extensions", set())
        normalized_converter_extensions = {
            extension if extension.startswith(".") else f".{extension}"
            for extension in converter_extensions
        }
        return self.TEXT_EXTENSIONS | normalized_converter_extensions

    def import_sources(
        self,
        sources: List[str],
    ) -> LoaderResult:
        """
        Import documents from multiple sources.

        Args:
            sources: List of file paths or URLs

        Returns:
            LoaderResult with Documents and any errors
        """
        with start_span(
            "vgrag.import_sources",
            {
                "vgrag.source_count": len(sources),
                "vgrag.chunk_documents": self.chunker is not None,
            },
        ):
            all_documents = []
            all_errors = []

            for source in sources:
                with observability_context(source=source):
                    result = self._import_single(source)
                all_documents.extend(result.documents)
                all_errors.extend(result.errors)

            # Apply chunking if enabled
            if self.chunker and all_documents:
                with start_span(
                    "vgrag.chunk_documents",
                    {
                        "vgrag.document_count": len(all_documents),
                        "vgrag.chunk_size": self.chunker.chunk_size,
                        "vgrag.chunk_overlap": self.chunker.chunk_overlap,
                    },
                ):
                    all_documents = self.chunker.chunk_batch(all_documents)

            return LoaderResult(documents=all_documents, errors=all_errors)

    def _import_single(self, source: str) -> ConversionResult:
        """Import a single source (file or URL)."""
        with start_span(
            "vgrag.import_source",
            {
                "vgrag.source_type": "url"
                if source.startswith(("http://", "https://"))
                else Path(source).suffix.lower().lstrip("."),
            },
        ):
            # Check if URL
            if source.startswith(("http://", "https://")):
                return self.url_fetcher.fetch(source)

            # Check if file exists
            path = Path(source)
            if not path.exists():
                return ConversionResult(documents=[], errors=[f"File not found: {source}"])

            # Check if supported
            ext = path.suffix.lower()
            if ext not in self.supported_extensions:
                return ConversionResult(documents=[], errors=[f"Unsupported file type: {ext}"])

            # Handle different file types
            if ext in self.TEXT_EXTENSIONS:
                # Direct passthrough for text files
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        content = f.read()
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": str(path),
                            "source_type": ext[1:],  # Remove dot
                        },
                    )
                    return ConversionResult(documents=[doc])
                except Exception as e:
                    return ConversionResult(
                        documents=[],
                        errors=[f"Failed to read {source}: {str(e)}"],
                    )
            else:
                return self.converter.convert(source)

    def import_text(self, text: str, source: str = "text_input") -> LoaderResult:
        """
        Import raw text directly.

        Args:
            text: Raw text content
            source: Source identifier for metadata

        Returns:
            LoaderResult with Document
        """
        with observability_context(source=source):
            with start_span(
                "vgrag.import_text",
                {
                    "vgrag.text_length": len(text),
                    "vgrag.chunk_documents": self.chunker is not None,
                },
            ):
                doc = Document(
                    page_content=text, metadata={"source": source, "source_type": "text"}
                )

                documents = [doc]
                if self.chunker:
                    with start_span(
                        "vgrag.chunk_documents",
                        {
                            "vgrag.document_count": len(documents),
                            "vgrag.chunk_size": self.chunker.chunk_size,
                            "vgrag.chunk_overlap": self.chunker.chunk_overlap,
                        },
                    ):
                        documents = self.chunker.chunk_batch(documents)

                return LoaderResult(documents=documents)


# Convenience exports
__all__ = [
    "DocumentImporter",
    "DocumentConverter",
    "DocumentConverterProtocol",
    "DoclingConverter",
    "MinerUConverter",
    "URLFetcher",
    "TextChunker",
    "LoaderResult",
    "ConversionResult",
]
