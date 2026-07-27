from dataclasses import dataclass
from io import BytesIO

from PIL import Image
import fitz
from pypdf import PdfReader
import pytesseract


@dataclass(slots=True)
class OCRResult:
    text: str
    confidence: float | None = None
    pages: list[tuple[int, str]] | None = None
    source_type: str = "image"


class OCRService:
    def extract_text(self, file_bytes: bytes, filename: str | None = None) -> OCRResult:
        if self._looks_like_pdf(file_bytes, filename):
            return self._extract_pdf_text(file_bytes)

        image = Image.open(BytesIO(file_bytes))
        text = pytesseract.image_to_string(image).strip()
        return OCRResult(text=text, pages=[(1, text)] if text else [], source_type="image")

    def _extract_pdf_text(self, pdf_bytes: bytes) -> OCRResult:
        reader = PdfReader(BytesIO(pdf_bytes))
        extracted_pages: list[tuple[int, str]] = []

        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                text = self._ocr_pdf_page(pdf_bytes, page_number - 1).strip()
            extracted_pages.append((page_number, text))

        combined_text = "\n\n".join(page_text for _, page_text in extracted_pages if page_text)
        return OCRResult(text=combined_text.strip(), pages=extracted_pages, source_type="pdf")

    def _ocr_pdf_page(self, pdf_bytes: bytes, page_index: int) -> str:
        document = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = document.load_page(page_index)
        pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        image = Image.open(BytesIO(pixmap.tobytes("png")))
        return pytesseract.image_to_string(image)

    def _looks_like_pdf(self, file_bytes: bytes, filename: str | None) -> bool:
        if file_bytes.startswith(b"%PDF"):
            return True

        if filename:
            return filename.lower().endswith(".pdf")

        return False
