"""Parse uploaded files into pages, then split pages into fixed-size chunks."""

import hashlib
import io
import re
from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".html", ".htm"}

# Rough English average; avoids pulling in a tokenizer just to size chunks.
WORDS_PER_TOKEN = 0.75


@dataclass
class Page:
    text: str
    page: int | None = None  # 1-indexed; None when the format has no pages


@dataclass
class Chunk:
    text: str
    chunk_index: int
    page: int | None = None
    metadata: dict = field(default_factory=dict)


class UnsupportedFileError(ValueError):
    pass


def content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse(filename: str, data: bytes) -> list[Page]:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return _parse_pdf(data)
    if ext == ".docx":
        return _parse_docx(data)
    if ext in {".txt", ".md"}:
        return [Page(_decode(data))]
    if ext in {".html", ".htm"}:
        return [Page(_strip_html(_decode(data)))]
    raise UnsupportedFileError(
        f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
    )


def _decode(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def _parse_pdf(data: bytes) -> list[Page]:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    return [Page(page.extract_text() or "", page=i) for i, page in enumerate(reader.pages, start=1)]


def _parse_docx(data: bytes) -> list[Page]:
    import docx

    document = docx.Document(io.BytesIO(data))
    parts = [p.text for p in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            parts.append(" | ".join(cell.text for cell in row.cells))
    return [Page("\n".join(parts))]


def _strip_html(html: str) -> str:
    html = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", html)
    return re.sub(r"<[^>]+>", " ", html)


def clean(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_pages(pages: list[Page], chunk_tokens: int, overlap: float) -> list[Chunk]:
    """Fixed-size word windows with overlap. Chunks never span pages, so each
    chunk keeps an exact page number for citations."""
    size = max(1, int(chunk_tokens * WORDS_PER_TOKEN))
    step = max(1, size - int(size * overlap))
    chunks: list[Chunk] = []
    for page in pages:
        words = clean(page.text).split(" ")
        words = [w for w in words if w]
        if not words:
            continue
        for start in range(0, len(words), step):
            window = words[start : start + size]
            chunks.append(Chunk(" ".join(window), chunk_index=len(chunks), page=page.page))
            if start + size >= len(words):
                break
    return chunks
