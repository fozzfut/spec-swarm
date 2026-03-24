"""Document parser -- extracts text from various file formats.

Supports PDF (via pymupdf, optional), plain text, markdown, reStructuredText, and CSV.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path


def parse_document(path: str) -> dict:
    """Parse a document and return structured text.

    Returns:
        {"text": str, "pages": int, "format": str, "metadata": dict}
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    suffix = p.suffix.lower()

    if suffix == ".pdf":
        return _parse_pdf(p)
    elif suffix == ".csv":
        return _parse_csv(p)
    elif suffix in (".txt", ".md", ".rst", ".text", ".markdown"):
        return _parse_text(p, suffix)
    else:
        # Try as plain text
        return _parse_text(p, suffix)


def _parse_pdf(path: Path) -> dict:
    """Parse PDF using pymupdf (fitz). Falls back with helpful error if not installed."""
    try:
        import fitz  # pymupdf
    except ImportError:
        raise ImportError(
            "PDF support requires pymupdf. Install with: pip install spec-swarm-ai[pdf]"
        )

    doc = fitz.open(str(path))
    try:
        pages_text: list[str] = []
        metadata: dict = {}

        # Extract document metadata
        doc_meta = doc.metadata
        if doc_meta:
            for key in ("title", "author", "subject", "keywords", "creator"):
                val = doc_meta.get(key, "")
                if val:
                    metadata[key] = val

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            if text.strip():
                pages_text.append(text)
    finally:
        doc.close()

    full_text = "\n\n--- Page Break ---\n\n".join(pages_text)

    return {
        "text": full_text,
        "pages": len(pages_text),
        "format": "pdf",
        "metadata": metadata,
    }


def _parse_text(path: Path, suffix: str) -> dict:
    """Parse plain text, markdown, or reStructuredText files."""
    text = path.read_text(encoding="utf-8", errors="replace")

    format_map = {
        ".md": "markdown",
        ".markdown": "markdown",
        ".rst": "restructuredtext",
        ".txt": "plaintext",
        ".text": "plaintext",
    }
    fmt = format_map.get(suffix, "plaintext")

    # Count logical pages (separated by form feeds or large gaps)
    pages = text.count("\f") + 1 if "\f" in text else 1

    return {
        "text": text,
        "pages": pages,
        "format": fmt,
        "metadata": {"filename": path.name},
    }


def _parse_csv(path: Path) -> dict:
    """Parse CSV files into structured text suitable for spec extraction.

    Converts CSV rows into a readable tabular format that the spec extractor
    can parse for register maps, pin tables, etc.
    """
    text_parts: list[str] = []
    row_count = 0

    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        # Sniff dialect
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except csv.Error:
            dialect = csv.excel

        reader = csv.reader(f, dialect)
        headers: list[str] = []

        for i, row in enumerate(reader):
            if not any(cell.strip() for cell in row):
                continue
            row_count += 1

            if i == 0:
                headers = [cell.strip() for cell in row]
                text_parts.append("| " + " | ".join(headers) + " |")
                text_parts.append("|" + "|".join("---" for _ in headers) + "|")
            else:
                # Pad row to match headers length
                while len(row) < len(headers):
                    row.append("")
                text_parts.append("| " + " | ".join(cell.strip() for cell in row[:len(headers)]) + " |")

    return {
        "text": "\n".join(text_parts),
        "pages": 1,
        "format": "csv",
        "metadata": {"filename": path.name, "rows": row_count},
    }
