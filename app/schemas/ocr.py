from typing import Literal

from pydantic import BaseModel, Field


class OCRBoundingBox(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float


class OCRBlock(BaseModel):
    kind: Literal["text", "image", "table", "formula", "unknown"]
    bbox: OCRBoundingBox
    text: str | None = None
    latex: str | None = None
    confidence: float | None = None
    source: Literal[
        "native",
        "tesseract",
        "rapidocr",
        "rapidlayout",
        "formula",
    ]
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class OCRPage(BaseModel):
    page_number: int
    text: str
    width: float | None = None
    height: float | None = None
    blocks: list[OCRBlock] | None = None
    confidence: float | None = None
    method: str | None = None
    warnings: list[str] | None = None
    processing_seconds: float | None = None


class OCRResponse(BaseModel):
    filename: str
    text: str
    confidence: float | None = None
    pages: list[OCRPage] = Field(default_factory=list)
    source_type: str = "image"
    processing_seconds: float | None = None
    cache_hit: bool | None = None
