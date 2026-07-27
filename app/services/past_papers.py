from collections.abc import Iterable
from dataclasses import dataclass

import httpx
from openai import OpenAI
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import CoursePastPaper, PastPaper
from app.services.ocr import OCRService


@dataclass(slots=True)
class IngestStats:
    processed: int = 0
    skipped: int = 0
    stored: int = 0


class PastPaperIngestionService:
    def __init__(self) -> None:
        settings = get_settings()
        api_key = settings["openai_api_key"]
        if not api_key:
            raise RuntimeError("OPEN_AI or OPENAI_API_KEY is required")

        self.embedding_model = str(settings["embedding_model"])
        self.client = OpenAI(api_key=api_key)
        self.ocr_service = OCRService()

    def list_source_papers(self, db: Session) -> list[PastPaper]:
        stmt = select(PastPaper).where(PastPaper.file_url.isnot(None))
        return list(db.scalars(stmt))

    def ingest_all(self, db: Session, papers: Iterable[PastPaper]) -> IngestStats:
        stats = IngestStats()
        for paper in papers:
            stats.processed += 1
            if not paper.course_id or not paper.year or not paper.file_url:
                stats.skipped += 1
                continue

            existing = db.scalar(
                select(CoursePastPaper).where(CoursePastPaper.past_paper_id == paper.id)
            )

            paper_bytes = self._download_pdf(paper.file_url)
            ocr_result = self.ocr_service.extract_text(paper_bytes, filename=paper.file_url)
            embedding = self._embed_text(ocr_result.text)
            page_map = [
                {"page_number": page_number, "text": text}
                for page_number, text in (ocr_result.pages or [])
            ]

            if existing:
                existing.course_id = paper.course_id
                existing.year = paper.year
                existing.slot = paper.slot
                existing.source_url = paper.file_url
                existing.ocr_text = ocr_result.text
                existing.ocr_pages = {"pages": page_map}
                existing.embedding = embedding
            else:
                db.add(
                    CoursePastPaper(
                        course_id=paper.course_id,
                        past_paper_id=paper.id,
                        year=paper.year,
                        slot=paper.slot,
                        source_url=paper.file_url,
                        ocr_text=ocr_result.text,
                        ocr_pages={"pages": page_map},
                        embedding=embedding,
                    )
                )
                stats.stored += 1

            db.commit()

        return stats

    def _download_pdf(self, url: str) -> bytes:
        response = httpx.get(url, timeout=60)
        response.raise_for_status()
        return response.content

    def _embed_text(self, text: str) -> list[float]:
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text[:12000],
        )
        return list(response.data[0].embedding)