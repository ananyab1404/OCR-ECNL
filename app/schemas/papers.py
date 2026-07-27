from pydantic import BaseModel


class SourcePaperOut(BaseModel):
    id: str
    course_id: str | None = None
    year: int | None = None
    slot: str | None = None
    title: str
    file_url: str


class IngestResult(BaseModel):
    processed: int
    skipped: int
    stored: int
