from pydantic import BaseModel, Field


class OCRPage(BaseModel):
    page_number: int
    text: str


class OCRResponse(BaseModel):
    filename: str
    text: str
    confidence: float | None = None
    pages: list[OCRPage] = Field(default_factory=list)
    source_type: str = "image"
