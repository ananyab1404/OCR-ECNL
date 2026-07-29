# OCR-ECNL

A local-first FastAPI service for extracting text and document structure from
PDFs and images. The production path preserves native PDF text, sends only
raster pages through OCR, exposes coordinates and confidence, and keeps
Tesseract as a dependable fallback.

## Pipeline

1. Validate file size, page count, encryption, and rendered pixel limits.
2. Open a PDF once with PyMuPDF.
3. Keep native text when it is usable. This is both the fastest and the most
   accurate route for normal PDFs, code, and digitally generated equations.
4. Render only scanned or image-dominant pages. Truly blank pages are detected
   before OCR and return an empty result.
5. On Windows, run RapidOCR with ONNX Runtime DirectML. All detector,
   classifier, and recognizer sessions must report `DmlExecutionProvider`
   first; otherwise the service falls back to Tesseract instead of silently
   running a slow CPU configuration.
   A cheap ruled-line signal plus RapidLayout confirmation routes pathological
   full-page tables to Tesseract PSM 6, avoiding the observed long-running
   DirectML table case.
6. With `include_layout=true`, add document regions from RapidLayout, native
   tables as Markdown, image bounds, OCR line bounds, and formula blocks with a
   `latex` representation when the PDF contains reliable native math text.

Repeated requests are cached in a bounded in-memory TTL cache keyed by the file
hash and all output-affecting settings.

## Setup

Python 3.12 or newer is required.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Install the Tesseract executable as well. On Windows the service automatically
checks the two standard `Program Files` locations; elsewhere put `tesseract` on
`PATH` or set `TESSERACT_CMD`.

Run the API:

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

The OCR and health routes start without PostgreSQL or cloud credentials. The
pre-existing paper-ingestion routes are optional: enable them with
`APP_ENABLE_PAPERS=true`, configure `DATABASE_URL`, an OpenAI key, and the
HTTPS source allowlist in `PAPERS_REMOTE_ALLOWED_HOSTS`, and set a strong
`PAPERS_INGEST_API_KEY`. `POST /papers/ingest` then requires
`Authorization: Bearer <key>`. Automatic table creation is off by default; use
migrations in production, or explicitly set `APP_AUTO_CREATE_TABLES=true` for a
controlled development environment. The current database schema requires
`OPENAI_EMBEDDING_DIMENSIONS=1536`; startup fails closed if this does not match.

For stable low latency, initialize the ONNX sessions during process startup:

```powershell
$env:OCR_PRELOAD_MODELS = "true"
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

The first model initialization is intentionally visible as cold-start cost.
On a 4 GB GPU, use one application worker; every worker owns its own model
sessions and memory.

## API

Legacy text-only clients can keep using the default response:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/ocr/extract" `
  -F "file=@sample.pdf"
```

Request structured document output:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/ocr/extract?include_layout=true" `
  -F "file=@sample.pdf"
```

Each structured page can contain:

- the extraction method and page processing time;
- text, image, table, and formula blocks;
- page-coordinate bounding boxes;
- OCR confidence and actual inference provider;
- native tables serialized as Markdown;
- native formula text plus normalized LaTeX;
- warnings when an optional backend is unavailable and a fallback was used.

`POST /ocr/extract-from-url` accepts HTTPS URLs only for hosts explicitly listed
in `OCR_REMOTE_ALLOWED_HOSTS`. Remote fetching is therefore disabled by default.
Redirects are not followed, proxy environment variables are ignored, non-default
ports and private/link-local/loopback targets are rejected, DNS resolution is
bounded, and the streamed response is capped before it is retained in memory.
Production deployments should still apply an outbound-network allowlist.

## Important settings

| Variable | Default | Purpose |
|---|---:|---|
| `OCR_MAX_FILE_BYTES` | 104857600 | Maximum upload or remote response |
| `OCR_MAX_PAGES` | 500 | Maximum PDF pages |
| `OCR_MAX_IMAGE_PIXELS` | 25000000 | Render/decompression-bomb guard |
| `OCR_MAX_DOCUMENT_RENDERED_PIXELS` | 1000000000 | Cumulative render-work budget per request |
| `OCR_MAX_DOCUMENT_SECONDS` | 300 | Cooperative end-to-end processing deadline |
| `OCR_MAX_OUTPUT_CHARACTERS` | 10000000 | Maximum textual payload across the response |
| `OCR_RENDER_DPI` | 144 | Primary raster resolution |
| `OCR_RETRY_DPI` | 216 | Low-confidence Tesseract retry |
| `OCR_ENABLE_RAPIDOCR` | Windows: true | Accelerated scan OCR |
| `OCR_RAPIDOCR_USE_DIRECTML` | Windows: true | Use the DirectML provider |
| `OCR_RAPIDOCR_REQUIRE_ACCELERATOR` | Windows: true | Reject silent CPU fallback |
| `OCR_RAPIDOCR_MIN_CONFIDENCE` | 55 | Fall back instead of accepting weak accelerated OCR |
| `OCR_RAPIDOCR_MAX_SIDE` | 2048 | Bound detector input while preserving readable text |
| `OCR_RAPIDOCR_ISOLATE_PROCESS` | Windows: true | Run DirectML OCR in a killable persistent worker |
| `OCR_RAPIDOCR_INFERENCE_TIMEOUT_SECONDS` | 15 | Hard worker inference timeout before Tesseract fallback |
| `OCR_RAPIDOCR_RECYCLE_AFTER_CALLS` | 100 | Restart the DirectML worker to bound long-run state |
| `OCR_ENABLE_DENSE_TABLE_GUARD` | true | Route confirmed full-page ruled tables to Tesseract |
| `OCR_DENSE_TABLE_LAYOUT_CONFIDENCE` | 0.80 | Minimum table-layout confidence for the guard |
| `OCR_DENSE_TABLE_MIN_AREA_RATIO` | 0.50 | Minimum page area covered by the confirmed table |
| `OCR_DENSE_TABLE_TESSERACT_PSM` | 6 | Tesseract segmentation mode for guarded tables |
| `OCR_ENABLE_DOCUMENT_LAYOUT` | true | Add layout regions when requested |
| `OCR_PRELOAD_MODELS` | false | Pay model initialization at startup |
| `OCR_MAX_CONCURRENCY` | 2 | API requests admitted to OCR work |
| `OCR_QUEUE_TIMEOUT_SECONDS` | 2 | Maximum wait for an OCR execution slot |
| `OCR_CACHE_MAX_BYTES` | 67108864 | Bounded result-cache size |
| `OCR_CACHE_MAX_ENTRIES` | 512 | Maximum number of cached results |
| `OCR_CACHE_TTL_SECONDS` | 900 | Result-cache lifetime |
| `OCR_REMOTE_ALLOWED_HOSTS` | empty | Comma-separated remote download allowlist |
| `OCR_REMOTE_CONNECT_TIMEOUT_SECONDS` | 10 | DNS-resolution/connect timeout |
| `OCR_REMOTE_TOTAL_TIMEOUT_SECONDS` | 60 | Absolute remote-transfer deadline |
| `APP_ENABLE_PAPERS` | false | Enable optional DB/OpenAI paper-ingestion routes |
| `APP_AUTO_CREATE_TABLES` | false | Development-only schema creation at startup |
| `APP_ENV` | development | Set to `production` to disable API docs |
| `APP_TRUSTED_HOSTS` | empty | Required comma-separated host allowlist in production |
| `PAPERS_REMOTE_ALLOWED_HOSTS` | empty | Required HTTPS download-host allowlist when paper routes are enabled |
| `PAPERS_INGEST_API_KEY` | empty | Bearer key required by the ingestion action |

Set `OCR_ENABLE_RAPIDOCR=false` to force Tesseract. On non-Windows systems,
RapidOCR is disabled by default unless explicitly enabled because the measured
CPU path was slower than Tesseract on this corpus.

The document deadline is a hard wall timeout for the isolated RapidOCR worker
and for Tesseract itself. It remains cooperative around PyMuPDF and optional
RapidLayout calls because those native libraries do not expose safe per-call
cancellation. Deploy untrusted workloads with the documented page/pixel/output
limits and an outer container or job-level kill deadline.

## Verification

```powershell
python -m compileall -q app
ruff check app
```

## Accuracy boundary

Printed text and native PDFs are the strong path. Handwriting, hand-drawn
schematics, and raster mathematics remain intrinsically harder: the service
returns recognized text, confidence, coordinates, and detected regions, but it
does not invent a semantic description when the local models cannot support
one. Native equations are preserved and normalized to LaTeX; raster-equation
models must be separately quality-gated before being enabled in production.
