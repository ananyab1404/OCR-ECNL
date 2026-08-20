from __future__ import annotations

import ipaddress
import os
import socket
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import httpx
from openai import OpenAI
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import CoursePastPaper, PastPaper

if TYPE_CHECKING:
    from app.services.ocr import OCRService


PAPER_EMBEDDING_DIMENSIONS = 1536
PAPER_ALLOWED_HOSTS_ENV = "PAPERS_REMOTE_ALLOWED_HOSTS"
_MAX_URL_LENGTH = 2_048
_DOWNLOAD_TIMEOUT_SECONDS = 60.0
_CONNECT_TIMEOUT_SECONDS = 10.0


class PaperConfigurationError(RuntimeError):
    pass


class PaperDownloadError(RuntimeError):
    pass


class PaperDownloadLimitError(PaperDownloadError):
    pass


@dataclass(slots=True)
class IngestStats:
    processed: int = 0
    skipped: int = 0
    stored: int = 0


def _normalize_allowed_hosts(hosts: Iterable[str]) -> frozenset[str]:
    return frozenset(
        host.strip().lower().rstrip(".")
        for host in hosts
        if host.strip()
    )


def _configured_allowed_hosts() -> frozenset[str]:
    return _normalize_allowed_hosts(os.getenv(PAPER_ALLOWED_HOSTS_ENV, "").split(","))


def _validate_paper_configuration(
    settings: dict[str, str | int | None],
    allowed_hosts: frozenset[str],
) -> None:
    if not str(settings["openai_api_key"] or "").strip():
        raise PaperConfigurationError(
            "OPEN_AI or OPENAI_API_KEY is required when APP_ENABLE_PAPERS=true"
        )
    if int(settings["embedding_dimensions"] or 0) != PAPER_EMBEDDING_DIMENSIONS:
        raise PaperConfigurationError(
            "OPENAI_EMBEDDING_DIMENSIONS must be 1536 to match CoursePastPaper.embedding"
        )
    if not allowed_hosts:
        raise PaperConfigurationError(
            f"{PAPER_ALLOWED_HOSTS_ENV} is required when APP_ENABLE_PAPERS=true"
        )


def validate_paper_startup_configuration() -> None:
    _validate_paper_configuration(get_settings(), _configured_allowed_hosts())


class PastPaperIngestionService:
    def __init__(
        self,
        *,
        ocr_service: OCRService | None = None,
        client: OpenAI | None = None,
        allowed_hosts: Iterable[str] | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        settings = get_settings()
        api_key = str(settings["openai_api_key"] or "").strip()
        normalized_hosts = (
            _configured_allowed_hosts()
            if allowed_hosts is None
            else _normalize_allowed_hosts(allowed_hosts)
        )
        _validate_paper_configuration(settings, normalized_hosts)

        if ocr_service is None:
            # The public OCR route owns the process-wide model stack. Reusing it
            # prevents optional paper ingestion from loading duplicate models.
            from app.api.routes.ocr import ocr_service as shared_ocr_service

            ocr_service = shared_ocr_service
        self.embedding_model = str(settings["embedding_model"])
        self.embedding_dimensions = int(settings["embedding_dimensions"] or 0)
        self.client = client if client is not None else OpenAI(api_key=api_key)
        self.ocr_service = ocr_service
        self.allowed_hosts = normalized_hosts
        self.max_download_bytes = ocr_service.settings.max_file_bytes
        self.transport = transport

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
        self._validate_download_url(url)
        content = bytearray()
        try:
            with (
                httpx.Client(
                    timeout=httpx.Timeout(
                        _DOWNLOAD_TIMEOUT_SECONDS,
                        connect=_CONNECT_TIMEOUT_SECONDS,
                    ),
                    follow_redirects=False,
                    trust_env=False,
                    transport=self.transport,
                ) as client,
                client.stream("GET", url) as response,
            ):
                if response.is_redirect:
                    raise PaperDownloadError("Paper download redirects are not allowed")
                response.raise_for_status()

                content_length = response.headers.get("content-length")
                if content_length:
                    try:
                        declared_bytes = int(content_length)
                    except ValueError:
                        declared_bytes = 0
                    if declared_bytes > self.max_download_bytes:
                        raise PaperDownloadLimitError(
                            f"Paper exceeds {self.max_download_bytes} bytes"
                        )

                for chunk in response.iter_bytes():
                    content.extend(chunk)
                    if len(content) > self.max_download_bytes:
                        raise PaperDownloadLimitError(
                            f"Paper exceeds {self.max_download_bytes} bytes"
                        )
        except PaperDownloadError:
            raise
        except httpx.HTTPError as exc:
            raise PaperDownloadError("Paper download failed") from exc

        return bytes(content)

    def _validate_download_url(self, url: str) -> None:
        if len(url) > _MAX_URL_LENGTH or any(ord(character) < 32 for character in url):
            raise PaperDownloadError("Paper URL is invalid")

        try:
            parsed = urlparse(url)
            parsed_hostname = parsed.hostname
            port = parsed.port or 443
        except ValueError as exc:
            raise PaperDownloadError("Paper URL is invalid") from exc

        if (
            parsed.scheme.lower() != "https"
            or not parsed_hostname
            or parsed.username
            or parsed.password
        ):
            raise PaperDownloadError("Paper URL must be an HTTPS URL without credentials")

        hostname = parsed_hostname.lower().rstrip(".")
        if hostname not in self.allowed_hosts:
            raise PaperDownloadError("Paper URL host is not allowed")
        if port != 443:
            raise PaperDownloadError("Paper URL must use the default HTTPS port")

        try:
            addresses = socket.getaddrinfo(
                hostname,
                port,
                type=socket.SOCK_STREAM,
            )
        except socket.gaierror as exc:
            raise PaperDownloadError("Paper URL host could not be resolved") from exc

        if not addresses:
            raise PaperDownloadError("Paper URL host could not be resolved")
        for address in addresses:
            ip = ipaddress.ip_address(address[4][0].split("%", maxsplit=1)[0])
            if not ip.is_global:
                raise PaperDownloadError("Private network paper URLs are not allowed")

    def _embed_text(self, text: str) -> list[float]:
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text[:12000],
            dimensions=self.embedding_dimensions,
        )
        embedding = list(response.data[0].embedding)
        if len(embedding) != self.embedding_dimensions:
            raise RuntimeError(
                "Embedding provider returned an unexpected vector dimension"
            )
        return embedding
