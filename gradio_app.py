"""
gradio_app.py
-------------
Gradio interface for the Image Similarity Search project, built for
deployment on Hugging Face Spaces (Gradio SDK, free tier).

Reuses the existing core pipeline unchanged:
  - embedder.py    (CLIP + BGE-M3 fused embeddings)
  - scanner.py     (folder scanning, kept for CLI use)
  - similarity.py  (cosine similarity + top-k)
  - utils.py       (device detection, caching)

Flow:
  1. User uploads a folder of images (multiple files) -> "Build Index"
  2. User uploads a query image -> "Search"
  3. Gallery shows the most similar images with similarity scores
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np

from embedder import embed_batch, embed_single
from similarity import build_index_matrix, top_k
from utils import get_device, setup_logging

setup_logging(logging.INFO)
logger = logging.getLogger("gradio_app")

DEVICE = get_device()

# In-memory index (per Space instance; fine for a single-user demo)
_index_matrix: np.ndarray | None = None
_index_paths: list[str] = []


def build_index(files: list[str] | None, batch_size: int = 16):
    """
    files: list of temp file paths from gr.Files upload.
    Embeds each uploaded image and builds the searchable index.
    """
    global _index_matrix, _index_paths

    if not files:
        return "⚠️ Please upload at least one image to index.", gr.update(interactive=False)

    image_paths = [f.name if hasattr(f, "name") else f for f in files]

    logger.info(f"Embedding {len(image_paths)} uploaded image(s) …")
    embeddings = embed_batch(image_paths, device=DEVICE, batch_size=batch_size)

    if not embeddings:
        return "❌ No valid images could be embedded.", gr.update(interactive=False)

    _index_matrix, _index_paths = build_index_matrix(embeddings)
    msg = f"✅ Indexed {len(_index_paths)} image(s). Ready to search."
    return msg, gr.update(interactive=True)


def search(query_image_path: str | None, topk: int = 10, threshold: float = 0.0):
    """
    query_image_path: path from gr.Image(type='filepath').
    Returns a list of (image_path, caption) tuples for gr.Gallery.
    """
    global _index_matrix, _index_paths

    if _index_matrix is None or len(_index_paths) == 0:
        return [], "⚠️ Build an index first (upload a folder of images above)."

    if query_image_path is None:
        return [], "⚠️ Please upload a query image."

    query_vec = embed_single(query_image_path, device=DEVICE)

    idx_dim = _index_matrix.shape[1]
    if query_vec.shape[0] < idx_dim:
        query_vec = np.concatenate(
            [query_vec, np.zeros(idx_dim - query_vec.shape[0], dtype=query_vec.dtype)]
        )
    elif query_vec.shape[0] > idx_dim:
        query_vec = query_vec[:idx_dim]

    results = top_k(
        query_vec=query_vec,
        index_vectors=_index_matrix,
        index_paths=_index_paths,
        k=int(topk),
        threshold=float(threshold),
    )

    gallery_items = [
        (r["path"], f"similarity: {r['similarity']:.4f}") for r in results
    ]
    status = f"Found {len(results)} match(es)." if results else "No matches above the similarity threshold."
    return gallery_items, status


with gr.Blocks(title="Image Similarity Search") as demo:
    gr.Markdown(
        "# 🔍 Image Similarity Search\n"
        "Upload a folder of images to build an index, then upload a query image "
        "to find the most visually similar matches (CLIP + BGE-M3 fused embeddings)."
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Build Index")
            index_files = gr.Files(
                label="Upload images to index", file_types=["image"], file_count="multiple"
            )
            batch_size_input = gr.Slider(
                minimum=1, maximum=64, value=16, step=1, label="Batch size"
            )
            index_btn = gr.Button("⚡ Build Index", variant="primary")
            index_status = gr.Markdown("")

            gr.Markdown("### 2. Search")
            query_image = gr.Image(label="Query image", type="filepath")
            topk_input = gr.Slider(minimum=1, maximum=50, value=10, step=1, label="Top K")
            threshold_input = gr.Slider(
                minimum=0.0, maximum=1.0, value=0.0, step=0.05, label="Min similarity"
            )
            search_btn = gr.Button("🚀 Find Similar", variant="secondary")
            search_status = gr.Markdown("")

        with gr.Column(scale=2):
            gr.Markdown("### Results")
            results_gallery = gr.Gallery(
                label="Similar images", columns=4, height="auto", object_fit="cover"
            )

    index_btn.click(
        fn=build_index,
        inputs=[index_files, batch_size_input],
        outputs=[index_status, search_btn],
    )

    search_btn.click(
        fn=search,
        inputs=[query_image, topk_input, threshold_input],
        outputs=[results_gallery, search_status],
    )


demo.queue()

if __name__ == "__main__":
    demo.launch()
    