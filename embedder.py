"""
embedder.py
-----------
Multi-modal embedding engine.

Pipeline:
  1. Detect image type (pure visual / image+text / text-only)
  2. Generate visual embedding via CLIP (ViT-L/14) or DINOv2
  3. Run OCR to extract text (EasyOCR primary, pytesseract fallback)
  4. Generate text embedding via BGE-M3 (sentence-transformers)
  5. Fuse: 0.4 * img_vec + 0.6 * txt_vec  (or img_vec only if no text)
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy-loaded global models (loaded once, reused across all calls)
# ---------------------------------------------------------------------------
_clip_model = None
_clip_processor = None
_text_embedder = None
_ocr_reader = None   # EasyOCR reader


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def _load_clip(device: str) -> tuple:
    """Load CLIP ViT-L/14 via HuggingFace transformers."""
    global _clip_model, _clip_processor
    if _clip_model is None:
        from transformers import CLIPModel, CLIPProcessor
        logger.info("Loading CLIP ViT-L/14 …")
        model_id = "openai/clip-vit-large-patch14"
        _clip_processor = CLIPProcessor.from_pretrained(model_id)
        _clip_model = CLIPModel.from_pretrained(model_id).to(device)
        _clip_model.eval()
        logger.info("CLIP loaded.")
    return _clip_model, _clip_processor


def _load_text_embedder():
    """Load BGE-M3 text embedding model via sentence-transformers."""
    global _text_embedder
    if _text_embedder is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading BGE-M3 text embedder …")
        # BAAI/bge-m3 supports 100+ languages and long context
        _text_embedder = SentenceTransformer("BAAI/bge-m3")
        logger.info("BGE-M3 loaded.")
    return _text_embedder


def _load_ocr():
    """Load EasyOCR reader (English + common scripts)."""
    global _ocr_reader
    if _ocr_reader is None:
        try:
            import easyocr
            logger.info("Loading EasyOCR …")
            _ocr_reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
            logger.info("EasyOCR loaded.")
        except ImportError:
            logger.warning("EasyOCR not installed; will fall back to pytesseract.")
            _ocr_reader = "pytesseract"
    return _ocr_reader


# ---------------------------------------------------------------------------
# Pre-processing helpers
# ---------------------------------------------------------------------------

TARGET_SIZE = (336, 336)   # CLIP ViT-L/14 native input size


def preprocess_image(pil_img: Image.Image) -> Image.Image:
    """
    Resize to TARGET_SIZE with LANCZOS resampling.
    Converts palette / RGBA images to RGB.
    """
    if pil_img.mode not in ("RGB", "L"):
        pil_img = pil_img.convert("RGB")
    elif pil_img.mode == "L":
        pil_img = pil_img.convert("RGB")
    pil_img = pil_img.resize(TARGET_SIZE, Image.LANCZOS)
    return pil_img


def detect_text_ratio(pil_img: Image.Image) -> float:
    """
    Heuristic: use the fraction of near-black/near-white pixel runs to
    guess if the image is text-heavy without running full OCR.
    Returns a value in [0, 1].  > 0.15 → likely contains significant text.
    """
    import cv2  # opencv-python
    img_np = np.array(pil_img.convert("L"))
    # Binarise
    _, binary = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Connected components
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(~binary, connectivity=8)
    if n_labels <= 1:
        return 0.0
    # Text-like components: small aspect-ratio-close-to-1, small area
    h, w = img_np.shape
    total_area = h * w
    text_area = 0
    for i in range(1, n_labels):
        cw, ch, ca = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_AREA]
        if ca == 0:
            continue
        aspect = max(cw, ch) / (min(cw, ch) + 1e-6)
        if ca < (total_area * 0.05) and aspect < 10:
            text_area += ca
    return min(text_area / (total_area + 1e-6), 1.0)


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------

def extract_text(pil_img: Image.Image) -> str:
    """
    Run OCR on the image and return the extracted text.
    Tries EasyOCR first; falls back to pytesseract.
    """
    reader = _load_ocr()
    try:
        if reader == "pytesseract":
            import pytesseract
            text = pytesseract.image_to_string(pil_img)
        else:
            results = reader.readtext(np.array(pil_img), detail=0, paragraph=True)
            text = " ".join(results)
        # Basic cleanup
        text = re.sub(r"\s+", " ", text).strip()
        return text
    except Exception as exc:
        logger.warning(f"OCR failed: {exc}")
        return ""


# ---------------------------------------------------------------------------
# Individual embeddings
# ---------------------------------------------------------------------------

def embed_image_clip(pil_imgs: list[Image.Image], device: str) -> np.ndarray:
    """
    Produce L2-normalised CLIP image embeddings for a batch of PIL images.
    Returns ndarray of shape (N, D).

    Handles both transformers versions:
      - New: get_image_features() returns a plain Tensor
      - Old: returns BaseModelOutputWithPooling  → extract .pooler_output
    """
    model, processor = _load_clip(device)

    # CLIPProcessor only needs `images=` — drop any stray text keys
    inputs = processor(images=pil_imgs, return_tensors="pt")
    # Move only tensor values to device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items()}

    with torch.no_grad():
        # Pass only pixel_values; get_image_features does not accept
        # attention_mask or input_ids from a vision-only call
        pixel_values = inputs.get("pixel_values")
        feats = model.get_image_features(pixel_values=pixel_values)

    # --- Robustly extract tensor from whatever the model returned ---
    if isinstance(feats, torch.Tensor):
        tensor = feats
    elif hasattr(feats, "image_embeds"):          # CLIPOutput
        tensor = feats.image_embeds
    elif hasattr(feats, "pooler_output"):          # BaseModelOutputWithPooling
        tensor = feats.pooler_output
    elif hasattr(feats, "last_hidden_state"):      # use CLS token
        tensor = feats.last_hidden_state[:, 0, :]
    else:
        raise TypeError(
            f"Unexpected CLIP output type: {type(feats)}. "
            "Please update transformers: pip install -U transformers"
        )

    arr = tensor.cpu().float().numpy()
    # L2 normalise
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-10
    return arr / norms


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Produce L2-normalised BGE-M3 text embeddings.
    Returns ndarray of shape (N, D).
    """
    model = _load_text_embedder()
    # BGE-M3 uses dense_vecs by default
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,   # already L2-normalised
        convert_to_numpy=True,
    )
    return embeddings


# ---------------------------------------------------------------------------
# Dimension alignment for fusion
# ---------------------------------------------------------------------------

def _pad_or_project(vec: np.ndarray, target_dim: int) -> np.ndarray:
    """
    Make two vectors the same dimensionality by zero-padding the shorter one.
    (Simple but avoids training a projection layer.)
    """
    d = vec.shape[0]
    if d == target_dim:
        return vec
    if d < target_dim:
        return np.concatenate([vec, np.zeros(target_dim - d, dtype=vec.dtype)])
    # Truncate (fallback – shouldn't happen with standard models)
    return vec[:target_dim]


# ---------------------------------------------------------------------------
# Fused embedding (public API)
# ---------------------------------------------------------------------------

IMG_WEIGHT = 0.4
TXT_WEIGHT = 0.6
TEXT_RATIO_THRESHOLD = 0.08   # run OCR if heuristic score exceeds this


def embed_single(
    image_path: str | Path,
    device: str = "cpu",
    force_ocr: bool = False,
) -> np.ndarray:
    """
    Compute the fused multi-modal embedding for a single image.

    Returns a 1-D float32 numpy array (L2-normalised).
    """
    path = Path(image_path)
    pil_img = Image.open(path)
    pil_img = preprocess_image(pil_img)

    # --- Visual embedding (always computed) ---
    img_vec = embed_image_clip([pil_img], device)[0]   # shape (D_clip,)

    # --- Decide whether to run OCR ---
    text_ratio = detect_text_ratio(pil_img)
    run_ocr = force_ocr or (text_ratio >= TEXT_RATIO_THRESHOLD)

    if run_ocr:
        text = extract_text(pil_img)
    else:
        text = ""

    if text.strip():
        txt_vec = embed_texts([text])[0]           # shape (D_bge,)
        # Align dimensions
        max_dim = max(img_vec.shape[0], txt_vec.shape[0])
        img_vec_a = _pad_or_project(img_vec, max_dim)
        txt_vec_a = _pad_or_project(txt_vec, max_dim)
        fused = IMG_WEIGHT * img_vec_a + TXT_WEIGHT * txt_vec_a
    else:
        fused = img_vec   # pure visual

    # Final L2 normalise
    norm = np.linalg.norm(fused) + 1e-10
    return (fused / norm).astype(np.float32)


def embed_batch(
    image_paths: list[str | Path],
    device: str = "cpu",
    batch_size: int = 16,
) -> dict[str, np.ndarray]:
    """
    Compute embeddings for a list of image paths in batches.
    Returns dict mapping str(path) → embedding vector.
    """
    from tqdm import tqdm

    results: dict[str, np.ndarray] = {}
    paths = [Path(p) for p in image_paths]

    for start in tqdm(range(0, len(paths), batch_size), desc="Embedding batches"):
        batch_paths = paths[start : start + batch_size]
        pil_imgs: list[Image.Image] = []
        valid_paths: list[Path] = []

        for p in batch_paths:
            try:
                img = Image.open(p)
                pil_imgs.append(preprocess_image(img))
                valid_paths.append(p)
            except Exception as exc:
                logger.warning(f"Cannot open {p}: {exc}")

        if not pil_imgs:
            continue

        # CLIP batch
        img_vecs = embed_image_clip(pil_imgs, device)   # (N, D_clip)

        # OCR + text embed per image
        for idx, (p, img, img_vec) in enumerate(zip(valid_paths, pil_imgs, img_vecs)):
            text_ratio = detect_text_ratio(img)
            text = extract_text(img) if text_ratio >= TEXT_RATIO_THRESHOLD else ""

            if text.strip():
                txt_vec = embed_texts([text])[0]
                max_dim = max(img_vec.shape[0], txt_vec.shape[0])
                img_vec_a = _pad_or_project(img_vec, max_dim)
                txt_vec_a = _pad_or_project(txt_vec, max_dim)
                fused = IMG_WEIGHT * img_vec_a + TXT_WEIGHT * txt_vec_a
            else:
                fused = img_vec

            norm = np.linalg.norm(fused) + 1e-10
            results[str(p)] = (fused / norm).astype(np.float32)

    return results
