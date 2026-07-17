# OCR FastAPI Boilerplate

A small FastAPI starter for OCR endpoints. It exposes a health check and a single image upload endpoint that sends the file through a swappable OCR service.

## Features

- FastAPI app with a `/health` route
- `/ocr/extract` endpoint for image uploads
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
```

## Notes

- The default OCR service uses `pytesseract`, so the system `tesseract` executable must be available on PATH.
- You can replace the service implementation later with EasyOCR, PaddleOCR, or a cloud OCR provider without changing the API shape.
