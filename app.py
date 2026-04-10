"""
app.py
------
High-accuracy local image similarity search.

CLI usage:
    python app.py --query path/to/query.jpg --root D:/images --topk 10

    # With threshold filter (only return similarity >= 0.75):
    python app.py --query img.png --root ./photos --topk 20 --threshold 0.75

    # Force re-index even if cache exists:
    python app.py --query img.png --root ./photos --reindex

    # Save / load cache from a custom path:
    python app.py --query img.png --root ./photos --cache my_cache.pkl

    # Output JSON to a file:
    python app.py --query img.png --root ./photos --output results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from embedder import embed_batch, embed_single
from scanner import scan_images
from similarity import build_index_matrix, top_k
from utils import (
    filter_uncached,
    format_results_json,
    get_device,
    load_cache,
    print_results,
    save_cache,
    setup_logging,
    update_cache,
)


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="app.py",
        description="High-accuracy multi-modal local image similarity search.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--query", required=True, metavar="PATH",
        help="Path to the query image.",
    )
    p.add_argument(
        "--root", required=True, metavar="DIR",
        help="Root directory to scan for candidate images.",
    )
    p.add_argument(
        "--topk", type=int, default=10, metavar="K",
        help="Number of top results to return (default: 10).",
    )
    p.add_argument(
        "--threshold", type=float, default=0.0, metavar="SCORE",
        help="Minimum cosine similarity to include a result (default: 0 = no filter).",
    )
    p.add_argument(
        "--cache", default=".image_embedding_cache.pkl", metavar="FILE",
        help="Path to the embedding cache file (default: .image_embedding_cache.pkl).",
    )
    p.add_argument(
        "--reindex", action="store_true",
        help="Ignore existing cache and re-embed all images.",
    )
    p.add_argument(
        "--batch-size", type=int, default=16, metavar="N",
        help="Number of images per embedding batch (default: 16).",
    )
    p.add_argument(
        "--output", default=None, metavar="FILE",
        help="Write JSON results to this file (default: print to stdout).",
    )
    p.add_argument(
        "--no-recursive", action="store_true",
        help="Do not recurse into sub-directories.",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG logging.",
    )
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    logger = logging.getLogger("app")

    device = get_device()

    # ------------------------------------------------------------------
    # 1. Validate query image
    # ------------------------------------------------------------------
    query_path = Path(args.query).expanduser().resolve()
    if not query_path.exists():
        logger.error(f"Query image not found: {query_path}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Scan index directory
    # ------------------------------------------------------------------
    logger.info(f"Scanning '{args.root}' …")
    all_image_paths = scan_images(args.root, recursive=not args.no_recursive)
    if not all_image_paths:
        logger.error("No images found in the root directory.")
        sys.exit(1)
    logger.info(f"Found {len(all_image_paths)} image(s) to index.")

    # ------------------------------------------------------------------
    # 3. Load / build embedding cache
    # ------------------------------------------------------------------
    cache: dict = {} if args.reindex else load_cache(args.cache)

    # Determine which images still need embedding
    to_embed = filter_uncached(all_image_paths, cache) if not args.reindex else all_image_paths

    if to_embed:
        logger.info(f"Embedding {len(to_embed)} new image(s) …")
        new_embeddings = embed_batch(
            to_embed,
            device=device,
            batch_size=args.batch_size,
        )
        cache = update_cache(cache, new_embeddings, cache_path=args.cache)
    else:
        logger.info("All images already cached – skipping embedding step.")

    # ------------------------------------------------------------------
    # 4. Build the similarity index matrix
    # ------------------------------------------------------------------
    # Keep only paths that appear in cache (some may have failed)
    valid_paths = [str(p) for p in all_image_paths if str(p) in cache]
    logger.info(f"Building index from {len(valid_paths)} embedded image(s).")

    restricted_cache = {p: cache[p] for p in valid_paths}
    index_matrix, index_paths = build_index_matrix(restricted_cache)

    # ------------------------------------------------------------------
    # 5. Embed the query image
    # ------------------------------------------------------------------
    logger.info(f"Embedding query image: {query_path}")
    query_vec = embed_single(query_path, device=device)

    # Pad query to match index dimension if needed
    idx_dim = index_matrix.shape[1] if index_matrix.ndim > 1 else query_vec.shape[0]
    if query_vec.shape[0] < idx_dim:
        query_vec = np.concatenate(
            [query_vec, np.zeros(idx_dim - query_vec.shape[0], dtype=query_vec.dtype)]
        )

    # ------------------------------------------------------------------
    # 6. Search
    # ------------------------------------------------------------------
    results = top_k(
        query_vec=query_vec,
        index_vectors=index_matrix,
        index_paths=index_paths,
        k=args.topk,
        threshold=args.threshold,
    )

    # ------------------------------------------------------------------
    # 7. Output
    # ------------------------------------------------------------------
    json_output = format_results_json(results)

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json_output, encoding="utf-8")
        logger.info(f"Results written to '{out_path}'.")
    else:
        print_results(results, query_path)
        print(json_output)


if __name__ == "__main__":
    main()
