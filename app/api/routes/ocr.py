from __future__ import annotations

import asyncio
import ipaddress
import logging
import socket
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from urllib.parse import unquote, urlparse

import httpx
from fastapi import APIRouter, Body, File, HTTPException, Query, UploadFile

from app.core.ocr_config import OCRSettings
from app.schemas.ocr import OCRBlock, OCRBoundingBox, OCRPage, OCRResponse
from app.services.ocr import (
    OCRDependencyError,
    OCRInputError,
    OCRLimitError,
    OCRResult,
    OCRService,
    OCRTimeoutError,
)

router = APIRouter()
logger = logging.getLogger(__name__)
ocr_settings = OCRSettings.from_env()
ocr_service = OCRService(settings=ocr_settings)
ocr_slots = asyncio.Semaphore(ocr_settings.max_concurrency)
request_slots = asyncio.Semaphore(ocr_settings.max_concurrency)
download_slots = asyncio.Semaphore(ocr_settings.max_concurrency)


@router.post(
    "/extract",
    response_model=OCRResponse,
    response_model_exclude_none=True,
)
async def extract_text(
    file: UploadFile = File(...),
    include_layout: bool = Query(False),
) -> OCRResponse:
    async with _request_admission():
        content = await file.read(ocr_settings.max_file_bytes + 1)
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        if len(content) > ocr_settings.max_file_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Uploaded file exceeds {ocr_settings.max_file_bytes} bytes",
            )

        result = await _extract(content, file.filename, include_layout)
        return _to_response(
            result,
            filename=file.filename or "upload",
            include_layout=include_layout,
        )


@router.post(
    "/extract-from-url",
    response_model=OCRResponse,
    response_model_exclude_none=True,
)
async def extract_from_url(
    url: str = Body(..., embed=True),
    include_layout: bool = Query(False),
) -> OCRResponse:
    content = bytearray()
    async with _download_admission():
        await _validate_remote_url(url)
        try:
            async with asyncio.timeout(ocr_settings.remote_total_timeout_seconds):
                async with (
                    httpx.AsyncClient(
                        timeout=httpx.Timeout(
                            ocr_settings.remote_total_timeout_seconds,
                            connect=ocr_settings.remote_connect_timeout_seconds,
                        ),
                        follow_redirects=False,
                        trust_env=False,
                    ) as client,
                    client.stream("GET", url) as response,
                ):
                    response.raise_for_status()
                    content_length = response.headers.get("content-length")
                    if content_length:
                        try:
                            declared_bytes = int(content_length)
                        except ValueError:
                            declared_bytes = 0
                        if declared_bytes > ocr_settings.max_file_bytes:
                            raise HTTPException(
                                status_code=413,
                                detail=(
                                    f"Remote file exceeds "
                                    f"{ocr_settings.max_file_bytes} bytes"
                                ),
                            )
                    async for chunk in response.aiter_bytes():
                        content.extend(chunk)
                        if len(content) > ocr_settings.max_file_bytes:
                            raise HTTPException(
                                status_code=413,
                                detail=(
                                    f"Remote file exceeds "
                                    f"{ocr_settings.max_file_bytes} bytes"
                                ),
                            )
        except HTTPException:
            raise
        except TimeoutError as exc:
            raise HTTPException(status_code=504, detail="Remote download timed out") from exc
        except httpx.HTTPError as exc:
            logger.warning("Remote OCR download failed: %s", exc)
            raise HTTPException(status_code=502, detail="Remote download failed") from exc

        # Keep the remote-memory admission slot until OCR completes. This caps
        # fully buffered remote documents even while the OCR execution queue is
        # saturated, without blocking local upload admission.
        filename = (
            unquote(urlparse(url).path.rsplit("/", maxsplit=1)[-1])
            or "remote-file"
        )
        result = await _extract(bytes(content), filename, include_layout)
        return _to_response(
            result,
            filename=filename,
            include_layout=include_layout,
        )


async def _extract(
    content: bytes,
    filename: str | None,
    include_layout: bool,
) -> OCRResult:
    await _acquire_slot(ocr_slots)
    try:
        worker = asyncio.create_task(
            asyncio.to_thread(
                ocr_service.extract_text,
                content,
                filename,
                include_layout,
            )
        )
    except BaseException:
        ocr_slots.release()
        raise

    worker.add_done_callback(_ocr_worker_finished)
    try:
        return await asyncio.shield(worker)
    except OCRInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except OCRLimitError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except OCRTimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except OCRDependencyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


async def _acquire_slot(semaphore: asyncio.Semaphore) -> None:
    try:
        await asyncio.wait_for(
            semaphore.acquire(),
            timeout=ocr_settings.queue_timeout_seconds,
        )
    except TimeoutError as exc:
        raise HTTPException(
            status_code=429,
            detail="OCR service is busy; retry shortly",
            headers={"Retry-After": "1"},
        ) from exc


def _ocr_worker_finished(worker: asyncio.Task[OCRResult]) -> None:
    ocr_slots.release()
    if worker.cancelled():
        return
    # Retrieve an exception even when the client disconnected and no coroutine
    # remains to await the shielded task.
    worker.exception()


@asynccontextmanager
async def _request_admission() -> AsyncIterator[None]:
    await _acquire_slot(request_slots)
    try:
        yield
    finally:
        request_slots.release()


@asynccontextmanager
async def _download_admission() -> AsyncIterator[None]:
    await _acquire_slot(download_slots)
    try:
        yield
    finally:
        download_slots.release()


def _to_response(
    result: OCRResult,
    *,
    filename: str,
    include_layout: bool,
) -> OCRResponse:
    details_by_page = {detail.page_number: detail for detail in (result.page_details or [])}
    pages: list[OCRPage] = []
    for page_number, text in result.pages or []:
        detail = details_by_page.get(page_number)
        blocks = None
        if include_layout and detail is not None:
            blocks = [
                OCRBlock(
                    kind=block.kind,
                    bbox=OCRBoundingBox(
                        x0=block.bbox.x0,
                        y0=block.bbox.y0,
                        x1=block.bbox.x1,
                        y1=block.bbox.y1,
                    ),
                    text=block.text,
                    latex=block.latex,
                    confidence=block.confidence,
                    source=block.source,
                    metadata=block.metadata,
                )
                for block in detail.blocks
            ]
        pages.append(
            OCRPage(
                page_number=page_number,
                text=text,
                width=detail.width if include_layout and detail else None,
                height=detail.height if include_layout and detail else None,
                blocks=blocks,
                confidence=detail.confidence if include_layout and detail else None,
                method=detail.method if include_layout and detail else None,
                warnings=detail.warnings if include_layout and detail else None,
                processing_seconds=(
                    detail.processing_seconds if include_layout and detail else None
                ),
            )
        )

    return OCRResponse(
        filename=filename,
        text=result.text,
        confidence=result.confidence,
        pages=pages,
        source_type=result.source_type,
        processing_seconds=result.processing_seconds if include_layout else None,
        cache_hit=result.cache_hit if include_layout else None,
    )


async def _validate_remote_url(url: str) -> None:
    if len(url) > 2_048 or any(ord(character) < 32 for character in url):
        raise HTTPException(status_code=400, detail="URL is invalid")

    try:
        parsed = urlparse(url)
        parsed_hostname = parsed.hostname
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="URL is invalid") from exc
    allowed_schemes = {"https"}
    if ocr_settings.allow_http_urls:
        allowed_schemes.add("http")
    if parsed.scheme.lower() not in allowed_schemes:
        raise HTTPException(
            status_code=400,
            detail=f"URL scheme must be one of: {', '.join(sorted(allowed_schemes))}",
        )
    if not parsed_hostname or parsed.username or parsed.password:
        raise HTTPException(status_code=400, detail="URL host is invalid")

    hostname = parsed_hostname.lower().rstrip(".")
    if hostname not in ocr_settings.remote_allowed_hosts:
        raise HTTPException(
            status_code=403,
            detail="Remote URL host is not allowed",
        )

    default_port = 443 if parsed.scheme.lower() == "https" else 80
    try:
        port = parsed.port or default_port
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="URL port is invalid") from exc
    if port != default_port:
        raise HTTPException(status_code=400, detail="Non-default URL ports are not allowed")

    try:
        addresses = await asyncio.wait_for(
            asyncio.to_thread(
                socket.getaddrinfo,
                hostname,
                port,
                type=socket.SOCK_STREAM,
            ),
            timeout=ocr_settings.remote_connect_timeout_seconds,
        )
    except (TimeoutError, socket.gaierror) as exc:
        raise HTTPException(status_code=400, detail="URL host could not be resolved") from exc

    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            raise HTTPException(status_code=400, detail="Private network URLs are not allowed")
