from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.api.routes.health import router as health_router
from app.api.routes.ocr import router as ocr_router

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _database_configured() -> bool:
    return bool(os.getenv("Database_URL") or os.getenv("DATABASE_URL"))


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    if _env_bool("APP_ENABLE_PAPERS", False) and _env_bool(
        "APP_AUTO_CREATE_TABLES",
        False,
    ):
        from app.db import models  # noqa: F401
        from app.db.base import Base
        from app.db.session import engine

        await asyncio.to_thread(Base.metadata.create_all, bind=engine)
    try:
        yield
    finally:
        from app.api.routes.ocr import ocr_service

        ocr_service.close()


is_production = os.getenv("APP_ENV", "development").strip().lower() == "production"
app = FastAPI(
    title="OCR-ECNL",
    version="0.2.0",
    docs_url=None if is_production else "/docs",
    redoc_url=None if is_production else "/redoc",
    openapi_url=None if is_production else "/openapi.json",
    lifespan=lifespan,
)

trusted_hosts = [
    host.strip()
    for host in os.getenv("APP_TRUSTED_HOSTS", "").split(",")
    if host.strip()
]
if is_production and not trusted_hosts:
    raise RuntimeError("APP_TRUSTED_HOSTS is required when APP_ENV=production")
if trusted_hosts:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=trusted_hosts)


@app.middleware("http")
async def add_security_headers(request: Request, call_next) -> Response:
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    return response


app.include_router(health_router)
app.include_router(ocr_router, prefix="/ocr", tags=["ocr"])

if _env_bool("APP_ENABLE_PAPERS", False):
    if not _database_configured():
        raise RuntimeError("DATABASE_URL is required when APP_ENABLE_PAPERS=true")
    from app.services.past_papers import validate_paper_startup_configuration

    validate_paper_startup_configuration()
    from app.api.routes.papers import router as papers_router

    app.include_router(papers_router, prefix="/papers", tags=["papers"])
else:
    logger.info("Optional paper-ingestion routes are disabled")
