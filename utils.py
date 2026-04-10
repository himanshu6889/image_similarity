"""
utils.py
--------
Utility helpers:
  - Device detection (CUDA / MPS / CPU)
  - Embedding cache  (save / load via pickle + numpy)
  - Logging configuration
  - Pretty result printing
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device() -> str:
    """
    Return the best available torch device string:
      - 'cuda'  if an NVIDIA GPU is available
      - 'mps'   if an Apple-Silicon GPU is available
      - 'cpu'   otherwise
    """
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        logger.info(f"Using device: {device}")
        return device
    except ImportError:
        return "cpu"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a clean formatter."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Embedding cache  (in-memory dict  +  optional disk persistence)
# ---------------------------------------------------------------------------

DEFAULT_CACHE_PATH = Path(".image_embedding_cache.pkl")


def load_cache(cache_path: Optional[str | Path] = None) -> dict[str, np.ndarray]:
    """
    Load an embedding cache from disk.
    Returns an empty dict if the file does not exist or is corrupt.
    """
    cache_path = Path(cache_path or DEFAULT_CACHE_PATH)
    if not cache_path.exists():
        logger.info(f"No existing cache at '{cache_path}'. Starting fresh.")
        return {}
    try:
        with open(cache_path, "rb") as fh:
            cache: dict = pickle.load(fh)
        logger.info(f"Loaded {len(cache)} cached embeddings from '{cache_path}'.")
        return cache
    except Exception as exc:
        logger.warning(f"Could not load cache ({exc}). Starting fresh.")
        return {}


def save_cache(
    cache: dict[str, np.ndarray],
    cache_path: Optional[str | Path] = None,
) -> None:
    """Persist the embedding cache to disk."""
    cache_path = Path(cache_path or DEFAULT_CACHE_PATH)
    try:
        with open(cache_path, "wb") as fh:
            pickle.dump(cache, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Saved {len(cache)} embeddings to '{cache_path}'.")
    except Exception as exc:
        logger.error(f"Failed to save cache: {exc}")


def update_cache(
    cache: dict[str, np.ndarray],
    new_embeddings: dict[str, np.ndarray],
    cache_path: Optional[str | Path] = None,
    autosave: bool = True,
) -> dict[str, np.ndarray]:
    """
    Merge *new_embeddings* into *cache* and optionally persist to disk.
    Returns the updated cache.
    """
    cache.update(new_embeddings)
    if autosave:
        save_cache(cache, cache_path)
    return cache


def filter_uncached(
    all_paths: list[Path],
    cache: dict[str, np.ndarray],
) -> list[Path]:
    """Return only paths that are NOT already present in the cache."""
    return [p for p in all_paths if str(p) not in cache]


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------

def print_results(results: list[dict], query_path: str | Path) -> None:
    """Pretty-print search results to stdout."""
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  Query : {query_path}")
    print(f"  Found : {len(results)} result(s)")
    print(f"{sep}")
    for rank, r in enumerate(results, start=1):
        bar_len = int(r["similarity"] * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  {rank:>3}.  [{bar}]  {r['similarity']:.4f}  {r['path']}")
    print(f"{sep}\n")


def format_results_json(results: list[dict], indent: int = 2) -> str:
    """Serialise results list to a JSON string."""
    return json.dumps(results, indent=indent, ensure_ascii=False)
