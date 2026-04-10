"""
scanner.py
----------
Recursively scan a root directory and return all image file paths.
Supports the most common raster image formats.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Supported image extensions (lowercase)
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".jpg", ".jpeg",
        ".png",
        ".bmp",
        ".gif",
        ".tiff", ".tif",
        ".webp",
        ".heic", ".heif",
        ".jfif",
    }
)


def scan_images(root: str | Path, recursive: bool = True) -> list[Path]:
    """
    Walk *root* and collect all image files.

    Parameters
    ----------
    root : str | Path
        Top-level directory to scan.
    recursive : bool
        If True (default) descend into sub-directories.

    Returns
    -------
    list[Path]
        Sorted list of image paths found under *root*.
    """
    root = Path(root).expanduser().resolve()

    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Expected a directory, got: {root}")

    pattern = "**/*" if recursive else "*"
    found: list[Path] = []

    for p in root.glob(pattern):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            found.append(p)

    found.sort()
    logger.info(f"Scanned '{root}': {len(found)} image(s) found.")
    return found

