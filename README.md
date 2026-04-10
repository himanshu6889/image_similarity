# 🔍 Image Similarity Search

A local, **privacy-first** image similarity search tool. Point it at a folder of images, upload a query image, and it finds the most visually similar matches — no cloud, no API keys needed.

Uses **CLIP ViT-L/14** for visual embeddings, **BGE-M3** for text (via OCR on image text), and cosine similarity for fast retrieval.

---

## Features

- **Web UI** — drag-and-drop interface via a local browser (FastAPI + single-page HTML)
- **CLI** — scriptable command-line mode for automation
- **Smart caching** — embeddings are cached to disk so re-runs are instant
- **Multi-modal** — automatically detects text in images and blends visual + text embeddings
- **GPU support** — runs on CUDA, Apple MPS, or CPU automatically

---

## Requirements

- Python **3.10+**
- pip

> GPU is optional but strongly recommended for large libraries. On CPU, indexing will be slow for thousands of images.

---

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/image-similarity.git
cd image-similarity

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate # Mac/Linux
venv\Scripts\activate # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

> **First run** will download model weights automatically (~1-2 GB total for CLIP + BGE-M3). This only happens once.

---

## Usage

### Option A — Web UI (recommended)

```bash
python server.py
```

Then open **http://localhost:8000** in your browser.

**Steps in the UI:**
1. Enter the path to your images folder (e.g. `D:\Photos` or `/home/user/pics`)
2. Click **Build Index** — progress is shown live
3. Upload any query image and click **Search**
4. Results appear with similarity scores and thumbnails

---

## Project Structure

```
image_similarity/
├── server.py          # FastAPI web server + REST API
├── app.py             # CLI entry point
├── embedder.py        # CLIP + OCR + BGE-M3 embedding pipeline
├── scanner.py         # Directory scanner (finds image files)
├── similarity.py      # Cosine similarity + Top-K search
├── utils.py           # Caching, logging, device detection, output
├── ui.html            # Single-page browser UI
└── requirements.txt   # Python dependencies
```

---

## How It Works

1. **Scan** — finds all images under the root directory
2. **Embed** — each image gets a combined embedding:
   - CLIP ViT-L/14 produces a visual embedding
   - If the image contains text (detected via a fast heuristic), EasyOCR extracts it and BGE-M3 encodes it
   - The two vectors are fused: `0.4 × visual + 0.6 × text`
3. **Cache** — embeddings are saved to `.image_embedding_cache.pkl` so repeat runs skip already-processed images
4. **Search** — the query image is embedded the same way, then compared to the index using cosine similarity

---

## Notes

- The cache file (`.image_embedding_cache.pkl`) is excluded from git. It will be created automatically on first run.
- No `.env` or API keys are needed — everything runs locally.
- Supported image formats: JPG, PNG, BMP, GIF, TIFF, WEBP, HEIC, JFIF
