from pydantic import BaseModel


class OCRResponse(BaseModel):
    filename: str
    text: str
    confidence: float | None = None
