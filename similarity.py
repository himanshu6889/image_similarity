"""
similarity.py
-------------
Cosine similarity computation and Top-K ranking.

All vectors are expected to be **already L2-normalised** (as produced by
embedder.py), so cosine similarity reduces to a plain dot product – O(N·D)
with a single numpy matrix multiply, which is fast even for large indexes.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Core similarity
# ---------------------------------------------------------------------------

def cosine_similarity(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between *query_vec* and every row of *matrix*.

    Parameters
    ----------
    query_vec : np.ndarray, shape (D,)
        L2-normalised query embedding.
    matrix : np.ndarray, shape (N, D)
        L2-normalised index embeddings stacked as rows.

    Returns
    -------
    np.ndarray, shape (N,)
        Similarity score in [-1, 1] for each indexed image.
    """
    # Safety: re-normalise in case caller passed un-normalised vectors
    q = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
    m = matrix / norms
    return m @ q   # dot product = cosine sim when both are unit vectors


# ---------------------------------------------------------------------------
# Top-K retrieval
# ---------------------------------------------------------------------------

def top_k(
    query_vec: np.ndarray,
    index_vectors: np.ndarray,
    index_paths: list[str],
    k: int = 10,
    threshold: float = 0.0,
) -> list[dict]:
    """
    Find the Top-K most similar images to *query_vec*.

    Parameters
    ----------
    query_vec : np.ndarray, shape (D,)
        Embedding of the query image.
    index_vectors : np.ndarray, shape (N, D)
        Embeddings of all indexed images (one row per image).
    index_paths : list[str]
        File paths corresponding to each row in *index_vectors*.
    k : int
        Number of results to return.
    threshold : float
        Minimum similarity score to include in results (default 0 = no filter).

    Returns
    -------
    list[dict]
        Sorted list (highest similarity first) of
        ``{"path": str, "similarity": float}``.
    """
    if len(index_paths) == 0:
        return []

    scores = cosine_similarity(query_vec, index_vectors)  # (N,)

    # Clip scores to [-1, 1] to fix float32 rounding (e.g. 1.0000001)
    scores = np.clip(scores, -1.0, 1.0)

    # Cap threshold just below 1.0 — cosine similarity of identical images
    # is never exactly 1.0 in float32 math (typically 0.9999998...), so a
    # threshold of 1.0 would silently drop all matches.
    if threshold >= 1.0:
        threshold = 0.9999

    # Apply threshold filter before sorting for efficiency
    if threshold > 0.0:
        valid_mask = scores >= threshold
        scores = scores[valid_mask]
        filtered_paths = [p for p, m in zip(index_paths, valid_mask) if m]
    else:
        filtered_paths = index_paths

    if len(filtered_paths) == 0:
        return []

    # argsort descending, take top-k
    top_indices = np.argsort(scores)[::-1][: k]

    results = [
        {
            "path": filtered_paths[i],
            "similarity": float(round(float(scores[i]), 6)),
        }
        for i in top_indices
    ]
    return results


# ---------------------------------------------------------------------------
# Build a matrix from an embedding cache (dict → ndarray)
# ---------------------------------------------------------------------------

def build_index_matrix(
    embedding_cache: dict[str, np.ndarray],
) -> tuple[np.ndarray, list[str]]:
    """
    Convert the embedding cache dict into a (matrix, paths) pair
    suitable for passing to :func:`top_k`.

    Parameters
    ----------
    embedding_cache : dict[str, np.ndarray]
        Maps image file path → 1-D embedding vector.

    Returns
    -------
    matrix : np.ndarray, shape (N, D)
    paths  : list[str]
    """
    if not embedding_cache:
        return np.empty((0, 0), dtype=np.float32), []

    paths = list(embedding_cache.keys())
    vecs = [embedding_cache[p] for p in paths]

    # Pad shorter vectors to the maximum dimension (handles clip-only vs fused)
    max_dim = max(v.shape[0] for v in vecs)
    padded = []
    for v in vecs:
        if v.shape[0] < max_dim:
            v = np.concatenate([v, np.zeros(max_dim - v.shape[0], dtype=v.dtype)])
        padded.append(v)

    matrix = np.stack(padded, axis=0).astype(np.float32)
    return matrix, paths
