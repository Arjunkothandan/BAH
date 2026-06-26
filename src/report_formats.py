"""Report format constants and normalization (no heavy dependencies)."""

from __future__ import annotations

SUPPORTED_FORMATS = ("pdf", "html", "md", "txt")

FORMAT_ALIASES = {
    "pdf": "pdf",
    "html": "html",
    "htm": "html",
    "md": "md",
    "markdown": "md",
    "txt": "txt",
    "text": "txt",
    "plain": "txt",
}

FORMAT_EXTENSIONS = {
    "pdf": "pdf",
    "html": "html",
    "md": "md",
    "txt": "txt",
}

FORMAT_MIME_TYPES = {
    "pdf": "application/pdf",
    "html": "text/html",
    "md": "text/markdown",
    "txt": "text/plain",
}


def normalize_format(format: str) -> str:
    """Map UI / user-facing format strings to a canonical backend format key."""
    key = (format or "").strip().lower()
    normalized = FORMAT_ALIASES.get(key)
    if normalized is None:
        aliases = ", ".join(sorted(FORMAT_ALIASES))
        supported = ", ".join(SUPPORTED_FORMATS)
        raise ValueError(
            f"Unsupported report format: {format}. "
            f"Supported formats: [{supported}]. "
            f"Accepted aliases: [{aliases}]"
        )
    return normalized
