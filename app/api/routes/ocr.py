from fastapi import APIRouter, File, HTTPException, UploadFile

from app.schemas.ocr import OCRResponse
from app.services.ocr import OCRService

router = APIRouter()
ocr_service = OCRService()


@router.post("/extract", response_model=OCRResponse)
async def extract_text(file: UploadFile = File(...)) -> OCRResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image uploads are supported")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    result = ocr_service.extract_text(content)
    return OCRResponse(filename=file.filename or "upload", text=result.text, confidence=result.confidence)
