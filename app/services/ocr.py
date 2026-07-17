from dataclasses import dataclass
from io import BytesIO

from PIL import Image
import pytesseract


@dataclass(slots=True)
class OCRResult:
    text: str
    confidence: float | None = None


class OCRService:
    def extract_text(self, image_bytes: bytes) -> OCRResult:
        image = Image.open(BytesIO(image_bytes))
        text = pytesseract.image_to_string(image)
        return OCRResult(text=text.strip())
