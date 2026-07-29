import hmac
import os
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy.orm import Session

from app.db.models import CoursePastPaper
from app.db.session import get_db
from app.schemas.papers import IngestResult, SourcePaperOut
from app.services.past_papers import PastPaperIngestionService

router = APIRouter()
ingestion_service = PastPaperIngestionService()


def require_ingest_key(authorization: Annotated[str | None, Header()] = None) -> None:
    expected = os.getenv("PAPERS_INGEST_API_KEY")
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="Paper ingestion is disabled until PAPERS_INGEST_API_KEY is configured",
        )
    scheme, separator, supplied = (authorization or "").partition(" ")
    if (
        not separator
        or scheme.lower() != "bearer"
        or not hmac.compare_digest(supplied, expected)
    ):
        raise HTTPException(
            status_code=401,
            detail="Valid bearer credentials are required",
            headers={"WWW-Authenticate": "Bearer"},
        )


@router.get("/sources", response_model=list[SourcePaperOut])
def list_source_papers(db: Session = Depends(get_db)) -> list[SourcePaperOut]:
    papers = ingestion_service.list_source_papers(db)
    return [
        SourcePaperOut(
            id=paper.id,
            course_id=paper.course_id,
            year=paper.year,
            slot=paper.slot,
            title=paper.title,
            file_url=paper.file_url,
        )
        for paper in papers
    ]


@router.post(
    "/ingest",
    response_model=IngestResult,
    dependencies=[Depends(require_ingest_key)],
)
def ingest_papers(db: Session = Depends(get_db)) -> IngestResult:
    papers = ingestion_service.list_source_papers(db)
    stats = ingestion_service.ingest_all(db, papers)
    return IngestResult(processed=stats.processed, skipped=stats.skipped, stored=stats.stored)


@router.get("/indexed-count")
def indexed_count(db: Session = Depends(get_db)) -> dict[str, int]:
    return {"count": db.query(CoursePastPaper).count()}
