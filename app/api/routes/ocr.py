import httpx

from fastapi import APIRouter, Body, File, UploadFile

from app.schemas.ocr import OCRResponse
from app.services.ocr import OCRService

router = APIRouter()
ocr_service = OCRService()


@router.post("/extract", response_model=OCRResponse)
async def extract_text(file: UploadFile = File(...)) -> OCRResponse:
    content = await file.read()
    if not content:
        from fastapi import HTTPException

        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    result = ocr_service.extract_text(content, filename=file.filename)
    return OCRResponse(
        filename=file.filename or "upload",
        text=result.text,
        confidence=result.confidence,
        pages=[{"page_number": page_number, "text": text} for page_number, text in (result.pages or [])],
        source_type=result.source_type,
    )


@router.post("/extract-from-url", response_model=OCRResponse)
async def extract_from_url(url: str = Body(..., embed=True)) -> OCRResponse:
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.get(url)
        response.raise_for_status()

    filename = url.rsplit("/", maxsplit=1)[-1] or "remote-file"
    result = ocr_service.extract_text(response.content, filename=filename)
    return OCRResponse(
        filename=filename,
        text=result.text,
        confidence=result.confidence,
        pages=[{"page_number": page_number, "text": text} for page_number, text in (result.pages or [])],
        source_type=result.source_type,
    )
