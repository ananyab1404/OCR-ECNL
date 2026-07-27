from datetime import datetime
from uuid import uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


def create_uuid() -> str:
    return str(uuid4())


class Course(Base):
    __tablename__ = "Course"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=create_uuid)
    code: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)

    papers: Mapped[list["PastPaper"]] = relationship(back_populates="course")


class PastPaper(Base):
    __tablename__ = "PastPaper"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=create_uuid)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    file_url: Mapped[str] = mapped_column("fileUrl", Text, nullable=False)
    course_id: Mapped[str | None] = mapped_column(
        "courseId", String(36), ForeignKey("Course.id", onupdate="CASCADE", ondelete="SET NULL")
    )
    year: Mapped[int | None] = mapped_column(Integer)
    slot: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime | None] = mapped_column(
        "createdAt", DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        "updatedAt", DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    course: Mapped[Course | None] = relationship(back_populates="papers")


class CoursePastPaper(Base):
    __tablename__ = "CoursePastPaper"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=create_uuid)
    course_id: Mapped[str] = mapped_column(
        "courseId", String(36), ForeignKey("Course.id", onupdate="CASCADE", ondelete="CASCADE"), nullable=False
    )
    past_paper_id: Mapped[str] = mapped_column(
        "pastPaperId", String(36), ForeignKey("PastPaper.id", onupdate="CASCADE", ondelete="CASCADE"), unique=True, nullable=False
    )
    year: Mapped[int] = mapped_column(Integer, nullable=False)
    slot: Mapped[str | None] = mapped_column(String(64))
    source_url: Mapped[str] = mapped_column(Text, nullable=False)
    ocr_text: Mapped[str] = mapped_column(Text, nullable=False)
    ocr_pages: Mapped[dict | None] = mapped_column(JSONB)
    embedding: Mapped[list[float]] = mapped_column(Vector(1536), nullable=False)
    created_at: Mapped[datetime | None] = mapped_column(
        "createdAt", DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        "updatedAt", DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
