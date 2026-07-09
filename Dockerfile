FROM python:3.10-slim

# System deps needed by opencv-python-headless and Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Hugging Face Spaces (Docker SDK) expects the app on port 7860
ENV PORT=7860
EXPOSE 7860

CMD ["python", "server.py"]
