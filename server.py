"""
server.py
---------
FastAPI localhost server for image similarity search.

Run:
    python server.py
Then open http://localhost:8000 in your browser.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from PIL import Image

from embedder import embed_batch, embed_single
from scanner import scan_images
from similarity import build_index_matrix, top_k
from utils import (
    filter_uncached,
    get_device,
    load_cache,
    save_cache,
    setup_logging,
)

setup_logging(logging.INFO)
logger = logging.getLogger("server")

# ─── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(title="Image Similarity Search", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Globals ─────────────────────────────────────────────────────────────────
DEVICE = get_device()
CACHE_PATH = Path(".image_embedding_cache.pkl")

embedding_cache: dict[str, np.ndarray] = {}   # ALL ever-indexed embeddings (persisted)
index_matrix: Optional[np.ndarray] = None     # matrix for CURRENT root only
index_paths: list[str] = []                   # paths for CURRENT root only
current_root: str = ""                        # last successfully indexed root

index_status: dict = {
    "running": False,
    "total": 0,
    "done": 0,
    "message": "Not started",
    "error": None,
    "root": "",
}

# ─── Startup ─────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    global embedding_cache
    embedding_cache = load_cache(CACHE_PATH)
    logger.info(f"Loaded {len(embedding_cache)} cached embeddings.")


# ─── Helpers ─────────────────────────────────────────────────────────────────
def _rebuild_index_for_paths(paths: list[str]) -> None:
    """
    Build index_matrix / index_paths from embedding_cache
    restricted to only the given paths list.
    This is the KEY fix: index always reflects the CURRENT root, not all-time cache.
    """
    global index_matrix, index_paths
    scoped_cache = {p: embedding_cache[p] for p in paths if p in embedding_cache}
    index_matrix, index_paths = build_index_matrix(scoped_cache)
    logger.info(f"Index ready: {len(index_paths)} images from current root.")


def _run_indexing(root: str, batch_size: int = 16) -> None:
    """Background thread: scan root, embed new images, rebuild scoped index."""
    global embedding_cache, current_root, index_status

    try:
        index_status.update({
            "running": True, "done": 0, "error": None,
            "message": "Scanning directory …", "root": root,
        })

        # 1. Scan the NEW root
        all_paths = scan_images(root, recursive=True)
        all_path_strs = [str(p) for p in all_paths]

        if not all_paths:
            index_status.update({
                "running": False,
                "message": "⚠️ No images found in that directory.",
            })
            return

        # 2. Only embed images not already in cache
        to_embed = filter_uncached(all_paths, embedding_cache)

        index_status["total"] = len(all_paths)
        index_status["done"]  = len(all_paths) - len(to_embed)
        index_status["message"] = (
            f"Found {len(all_paths)} images — "
            f"embedding {len(to_embed)} new ones …"
        )

        # 3. Embed in batches
        if to_embed:
            from tqdm import tqdm
            for start in tqdm(range(0, len(to_embed), batch_size), desc="Embedding"):
                batch = to_embed[start: start + batch_size]
                new_embs = embed_batch(batch, device=DEVICE, batch_size=batch_size)
                embedding_cache.update(new_embs)
                index_status["done"] += len(new_embs)
                index_status["message"] = (
                    f"Embedded {index_status['done']} / {index_status['total']} …"
                )

            save_cache(embedding_cache, CACHE_PATH)

        # 4. ✅ KEY FIX: rebuild index scoped ONLY to this root's paths
        _rebuild_index_for_paths(all_path_strs)
        current_root = root

        index_status.update({
            "running": False,
            "message": f"✅ Indexed {len(index_paths)} images from: {root}",
        })

    except Exception as exc:
        logger.exception("Indexing failed")
        index_status.update({
            "running": False,
            "error": str(exc),
            "message": f"❌ Error: {exc}",
        })


# ─── Thumbnail ───────────────────────────────────────────────────────────────
def _make_thumbnail_b64(path: str, size: tuple = (400, 400)) -> str:
    """
    Generate a base64-encoded JPEG thumbnail.
    Returns empty string on failure (caller falls back to /api/image).
    """
    try:
        p = Path(path)
        if not p.exists():
            logger.warning(f"Thumbnail: file not found: {path}")
            return ""
        img = Image.open(p)
        # Handle all modes: RGBA, P (palette), L (grayscale), CMYK, etc.
        if img.mode == "P":
            img = img.convert("RGBA")
        if img.mode in ("RGBA", "LA"):
            background = Image.new("RGB", img.size, (18, 18, 26))
            mask = img.split()[-1] if img.mode == "RGBA" else img.split()[-1]
            background.paste(img.convert("RGB"), mask=mask)
            img = background
        else:
            img = img.convert("RGB")
        img.thumbnail(size, Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85, optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return "data:image/jpeg;base64," + b64
    except Exception as exc:
        logger.warning(f"Thumbnail failed for {path}: {exc}")
        return ""


# ─── API endpoints ────────────────────────────────────────────────────────────

@app.post("/api/index")
async def start_index(root: str = Form(...), batch_size: int = Form(16)):
    """Start background indexing. Clears current index immediately on root change."""
    global index_matrix, index_paths, current_root

    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise HTTPException(400, f"Directory not found: {root}")

    resolved = str(root_path)

    if index_status["running"]:
        return JSONResponse({"status": "already_running"})

    # If root changed, clear the active index so stale results can't be served
    if resolved != current_root:
        logger.info(f"Root changed: '{current_root}' → '{resolved}'. Clearing index.")
        index_matrix = None
        index_paths = []
        current_root = ""

    thread = threading.Thread(
        target=_run_indexing,
        args=(resolved,),
        kwargs={"batch_size": batch_size},
        daemon=True,
    )
    thread.start()
    return JSONResponse({"status": "started", "root": resolved})


@app.get("/api/index/status")
async def get_index_status():
    return JSONResponse(index_status)


@app.post("/api/search")
async def search(
    file: UploadFile = File(...),
    topk: int = Form(10),
    threshold: float = Form(0.0),
):
    """Search for similar images against the CURRENT root's index."""
    if index_matrix is None or len(index_paths) == 0:
        raise HTTPException(
            400,
            "Index is empty. Please set the root path and click Build Index first."
        )

    suffix = Path(file.filename or "query.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        query_vec = embed_single(tmp_path, device=DEVICE)

        # Align dimension to index
        idx_dim = index_matrix.shape[1]
        if query_vec.shape[0] < idx_dim:
            query_vec = np.concatenate(
                [query_vec, np.zeros(idx_dim - query_vec.shape[0], dtype=query_vec.dtype)]
            )
        elif query_vec.shape[0] > idx_dim:
            query_vec = query_vec[:idx_dim]

        results = top_k(
            query_vec=query_vec,
            index_vectors=index_matrix,
            index_paths=index_paths,
            k=topk,
            threshold=threshold,
        )

        # Attach thumbnails — every result must have either b64 or a fallback flag
        enriched = []
        for r in results:
            thumb = _make_thumbnail_b64(r["path"])
            enriched.append({
                **r,
                "thumbnail": thumb,
                "has_thumb": bool(thumb),
            })

        return JSONResponse({
            "results": enriched,
            "query_filename": file.filename,
            "index_root": current_root,
            "total_indexed": len(index_paths),
        })

    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


@app.get("/api/image")
async def serve_image(path: str):
    """
    Serve a full-resolution image.
    Accepts the raw Windows/Linux path passed as a query param.
    """
    p = Path(path)
    if not p.exists():
        # Try URL-decoded path (browser may double-encode backslashes)
        from urllib.parse import unquote
        p = Path(unquote(path))
    if not p.exists():
        raise HTTPException(404, f"Image not found: {path}")

    # Determine MIME type
    ext = p.suffix.lower()
    media_map = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".gif": "image/gif",
        ".webp": "image/webp", ".bmp": "image/bmp",
        ".tiff": "image/tiff", ".tif": "image/tiff",
    }
    media_type = media_map.get(ext, "image/jpeg")
    return FileResponse(str(p), media_type=media_type)


# ─── Serve single-page UI ────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    ui_path = Path(__file__).parent / "ui.html"
    if ui_path.exists():
        return HTMLResponse(ui_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>ui.html not found — make sure ui.html is in the same folder.</h1>", status_code=404)


# ─── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "═" * 55)
    print("  🔍  Image Similarity Search — localhost server")
    print("  Open http://localhost:8000 in your browser")
    print("  Press Ctrl+C to stop")
    print("═" * 55 + "\n")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
