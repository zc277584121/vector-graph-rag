"""
MinerU converter integration.
"""

import os
import shutil
import signal
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document

from vector_graph_rag.observability import observability_context, start_span

from .converter import ConversionResult


class MinerUConverter:
    """
    Convert local files to Markdown through the MinerU CLI.

    MinerU is intentionally treated as an external parser process. This keeps
    Vector Graph RAG independent of MinerU's internal Python APIs while still
    allowing higher-fidelity document parsing when MinerU is installed.
    """

    supported_extensions = {
        ".pdf",
        ".png",
        ".jpg",
        ".jpeg",
        ".docx",
        ".pptx",
        ".xlsx",
    }

    def __init__(
        self,
        command: str = "mineru",
        output_dir: Optional[str] = None,
        timeout: Optional[float] = None,
        extra_args: Optional[List[str]] = None,
        check_command: bool = True,
    ):
        """
        Initialize the MinerU converter.

        Args:
            command: MinerU executable name or absolute path.
            output_dir: Optional persistent output directory. A temporary
                directory is used when omitted.
            timeout: Optional CLI timeout in seconds.
            extra_args: Extra CLI arguments appended after ``-p`` and ``-o``.
            check_command: Validate the command at initialization time.
        """
        self.command = command
        self.output_dir = Path(output_dir) if output_dir else None
        self.timeout = timeout
        self.extra_args = list(extra_args or [])

        if check_command:
            self._resolve_command()

    def _resolve_command(self) -> str:
        """Resolve the MinerU CLI command."""
        resolved = shutil.which(self.command)
        if not resolved:
            raise ImportError(
                "MinerU CLI is not installed or not available on PATH. "
                "Install MinerU support with: uv add 'vector-graph-rag[mineru]'"
            )
        return resolved

    def convert(self, source: str) -> ConversionResult:
        """
        Convert a single file to Markdown with MinerU.

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
                    "vgrag.parser": "mineru",
                    "vgrag.source_type": path.suffix.lower().lstrip("."),
                },
            ):
                if not path.exists():
                    return ConversionResult(documents=[], errors=[f"File not found: {source}"])

                ext = path.suffix.lower()
                if ext not in self.supported_extensions:
                    return ConversionResult(
                        documents=[], errors=[f"Unsupported file type for MinerU: {ext}"]
                    )

                temp_dir = None
                if self.output_dir is None:
                    temp_dir = tempfile.TemporaryDirectory()
                    output_root = Path(temp_dir.name)
                else:
                    output_root = self.output_dir
                    output_root.mkdir(parents=True, exist_ok=True)

                try:
                    command = [
                        self._resolve_command(),
                        "-p",
                        str(path),
                        "-o",
                        str(output_root),
                        *self.extra_args,
                    ]
                    result = self._run_command(command)
                    if result.returncode != 0:
                        details = (result.stderr or result.stdout or "").strip()
                        message = f"MinerU failed to convert {source}"
                        if details:
                            message = f"{message}: {details}"
                        return ConversionResult(documents=[], errors=[message])

                    markdown_path = self._find_markdown_output(output_root, path)
                    if markdown_path is None:
                        return ConversionResult(
                            documents=[],
                            errors=[
                                f"MinerU completed but no Markdown output was found for {source}"
                            ],
                        )

                    content = markdown_path.read_text(encoding="utf-8").strip()
                    if not content:
                        return ConversionResult(
                            documents=[],
                            errors=[f"MinerU produced an empty Markdown document for {source}"],
                        )

                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": str(path),
                            "source_type": ext[1:],
                            "parser": "mineru",
                        },
                    )
                    return ConversionResult(documents=[doc])

                except subprocess.TimeoutExpired:
                    return ConversionResult(
                        documents=[], errors=[f"MinerU timed out converting {source}"]
                    )
                except Exception as exc:
                    return ConversionResult(
                        documents=[], errors=[f"Failed to convert {source}: {str(exc)}"]
                    )
                finally:
                    if temp_dir is not None:
                        temp_dir.cleanup()

    def _run_command(self, command: List[str]) -> subprocess.CompletedProcess[str]:
        """Run MinerU and clean up child processes on timeout."""
        existing_server_pids = self._list_mineru_server_pids()
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=(os.name == "posix"),
        )
        try:
            stdout, stderr = process.communicate(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            self._terminate_process(process)
            self._terminate_new_mineru_servers(existing_server_pids)
            raise

        return subprocess.CompletedProcess(
            command,
            process.returncode,
            stdout=stdout,
            stderr=stderr,
        )

    @staticmethod
    def _terminate_process(process: subprocess.Popen[str]) -> None:
        """Terminate a MinerU process and its children when possible."""
        if process.poll() is not None:
            return

        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
        except ProcessLookupError:
            return

        try:
            process.wait(timeout=10)
            return
        except subprocess.TimeoutExpired:
            try:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            except ProcessLookupError:
                return
            process.wait()

    @staticmethod
    def _list_mineru_server_pids() -> set[int]:
        """List local MinerU API server processes on Linux."""
        proc_root = Path("/proc")
        if not proc_root.exists():
            return set()

        pids: set[int] = set()
        for entry in proc_root.iterdir():
            if not entry.name.isdigit():
                continue
            try:
                cmdline = (entry / "cmdline").read_bytes().replace(b"\x00", b" ")
            except OSError:
                continue
            if b"mineru.cli.fast_api" in cmdline:
                pids.add(int(entry.name))
        return pids

    @classmethod
    def _terminate_new_mineru_servers(cls, existing_pids: set[int]) -> None:
        """Terminate MinerU API server processes started by the timed-out command."""
        new_pids = cls._list_mineru_server_pids() - existing_pids
        for pid in sorted(new_pids):
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                continue

        for pid in sorted(new_pids):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                continue
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                continue

    def convert_batch(self, sources: List[str]) -> ConversionResult:
        """Convert multiple files."""
        all_documents = []
        all_errors = []

        for source in sources:
            result = self.convert(source)
            all_documents.extend(result.documents)
            all_errors.extend(result.errors)

        return ConversionResult(documents=all_documents, errors=all_errors)

    def _find_markdown_output(self, output_root: Path, source_path: Path) -> Optional[Path]:
        """Find the Markdown file produced by MinerU for one source."""
        candidates = [path for path in output_root.rglob("*.md") if path.is_file()]
        if not candidates:
            return None

        source_stem = source_path.stem.lower()

        def sort_key(path: Path) -> tuple[int, int, int, str]:
            path_stem = path.stem.lower()
            exact_name = 0 if path_stem == source_stem else 1
            contains_name = 0 if source_stem in path_stem else 1
            size = -path.stat().st_size
            return (exact_name, contains_name, size, str(path))

        return sorted(candidates, key=sort_key)[0]
