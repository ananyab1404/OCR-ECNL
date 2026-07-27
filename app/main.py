from fastapi import FastAPI

from app.api.routes.health import router as health_router
from app.api.routes.papers import router as papers_router
from app.api.routes.ocr import router as ocr_router
from app.db.base import Base
from app.db.session import engine
from app.db import models  # noqa: F401

app = FastAPI(title="OCR FastAPI Boilerplate", version="0.1.0")


@app.on_event("startup")
def create_tables() -> None:
	Base.metadata.create_all(bind=engine)


app.include_router(health_router)
app.include_router(ocr_router, prefix="/ocr", tags=["ocr"])
app.include_router(papers_router, prefix="/papers", tags=["papers"])
