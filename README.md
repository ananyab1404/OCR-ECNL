# OCR FastAPI Boilerplate

A small FastAPI starter for OCR endpoints. It exposes a health check and upload/URL extraction endpoints that send PDFs or images through a swappable OCR service.

## Features

- FastAPI app with a `/health` route
- `/ocr/extract` endpoint for image or PDF uploads
- `/ocr/extract-from-url` endpoint to fetch a PDF/image from a URL and extract text
- `/papers/sources` and `/papers/ingest` endpoints for course-wise past-paper sync
- OCR service layer that defaults to Tesseract through `pytesseract`
- Clean package layout that can be extended for PDFs, async jobs, and storage later

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -e .
```

3. Install the Tesseract binary on your machine if you want the default OCR backend to work.
4. Run the app:

```bash
uvicorn app.main:app --reload
```

## Example request

```bash
curl -X POST "http://127.0.0.1:8000/ocr/extract" \
  -F "file=@sample.png"

curl -X POST "http://127.0.0.1:8000/ocr/extract-from-url" \
  -H "Content-Type: application/json" \
  -d "{\"url\": \"https://example.com/question-paper.pdf\"}"
```

## Notes

- The default OCR service uses `pytesseract`, so the system `tesseract` executable must be available on PATH.
- The app uses `Database_URL` / `DATABASE_URL` for the database connection and accepts `DB_HOST` as a host override.
- The course-wise embedding table stores OpenAI embeddings using `text-embedding-3-small` by default.
- If your Postgres instance has pgvector enabled, make sure the `vector` extension is available.
- You can replace the service implementation later with EasyOCR, PaddleOCR, or a cloud OCR provider without changing the API shape.
