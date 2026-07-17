from fastapi import FastAPI

from app.api.routes.health import router as health_router
from app.api.routes.ocr import router as ocr_router

app = FastAPI(title="OCR FastAPI Boilerplate", version="0.1.0")

app.include_router(health_router)
app.include_router(ocr_router, prefix="/ocr", tags=["ocr"])
