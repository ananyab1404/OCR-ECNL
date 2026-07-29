from __future__ import annotations

import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    value = os.getenv(name)
    parsed = default if value is None else int(value)
    if parsed < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return parsed


def _env_float(name: str, default: float, *, minimum: float = 0) -> float:
    value = os.getenv(name)
    parsed = default if value is None else float(value)
    if not math.isfinite(parsed) or parsed < minimum:
        raise ValueError(f"{name} must be finite and at least {minimum}")
    return parsed


def _env_csv(name: str) -> tuple[str, ...]:
    value = os.getenv(name, "")
    return tuple(
        sorted(
            {
                item.strip().lower().rstrip(".")
                for item in value.split(",")
                if item.strip()
            }
        )
    )


def resolve_tesseract_cmd(configured: str | None = None) -> str | None:
    candidates: list[str] = []
    if configured:
        candidates.append(configured)

    discovered = shutil.which("tesseract")
    if discovered:
        candidates.append(discovered)

    if os.name == "nt":
        candidates.extend(
            [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            ]
        )

    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path.is_file():
            return str(path.resolve())
    return None


@dataclass(frozen=True, slots=True)
class OCRSettings:
    tesseract_cmd: str | None
    tesseract_lang: str = "eng"
    tesseract_oem: int = 1
    tesseract_psm: int = 3
    tesseract_timeout_seconds: int = 30
    tesseract_thread_limit: int = 1
    render_dpi: int = 144
    retry_dpi: int = 216
    retry_confidence_threshold: float = 45.0
    native_text_min_chars: int = 24
    native_text_min_words: int = 4
    image_dominant_ratio: float = 0.55
    max_file_bytes: int = 100 * 1024 * 1024
    max_pages: int = 500
    max_image_pixels: int = 25_000_000
    max_document_rendered_pixels: int = 1_000_000_000
    max_document_seconds: float = 300.0
    max_output_characters: int = 10_000_000
    max_concurrency: int = 2
    queue_timeout_seconds: float = 2.0
    cache_max_bytes: int = 64 * 1024 * 1024
    cache_max_entries: int = 512
    cache_ttl_seconds: int = 900
    allow_http_urls: bool = False
    remote_allowed_hosts: tuple[str, ...] = ()
    remote_connect_timeout_seconds: float = 10.0
    remote_total_timeout_seconds: float = 60.0
    enable_rapidocr: bool = True
    rapidocr_use_directml: bool = False
    rapidocr_require_accelerator: bool = False
    rapidocr_min_confidence: float = 55.0
    rapidocr_max_side: int = 2_048
    rapidocr_model_root: str | None = None
    rapidocr_isolate_process: bool = False
    rapidocr_inference_timeout_seconds: float = 15.0
    rapidocr_recycle_after_calls: int = 100
    enable_dense_table_guard: bool = True
    dense_table_layout_confidence: float = 0.80
    dense_table_min_area_ratio: float = 0.50
    dense_table_tesseract_psm: int = 6
    preload_models: bool = False
    enable_document_layout: bool = True
    enable_formula_ocr: bool = True
    layout_model_type: str = "pp_layout_cdla"
    layout_model_path: str | None = None
    layout_confidence_threshold: float = 0.35

    @classmethod
    def from_env(cls) -> OCRSettings:
        directml_default = os.name == "nt"
        return cls(
            tesseract_cmd=resolve_tesseract_cmd(os.getenv("TESSERACT_CMD")),
            tesseract_lang=os.getenv("TESSERACT_LANG", "eng"),
            tesseract_oem=_env_int("TESSERACT_OEM", 1),
            tesseract_psm=_env_int("TESSERACT_PSM", 3),
            tesseract_timeout_seconds=_env_int("TESSERACT_TIMEOUT_SECONDS", 30, minimum=1),
            tesseract_thread_limit=_env_int("TESSERACT_THREAD_LIMIT", 1, minimum=1),
            render_dpi=_env_int("OCR_RENDER_DPI", 144, minimum=72),
            retry_dpi=_env_int("OCR_RETRY_DPI", 216, minimum=72),
            retry_confidence_threshold=_env_float("OCR_RETRY_CONFIDENCE_THRESHOLD", 45.0),
            native_text_min_chars=_env_int("OCR_NATIVE_TEXT_MIN_CHARS", 24),
            native_text_min_words=_env_int("OCR_NATIVE_TEXT_MIN_WORDS", 4),
            image_dominant_ratio=_env_float("OCR_IMAGE_DOMINANT_RATIO", 0.55),
            max_file_bytes=_env_int("OCR_MAX_FILE_BYTES", 100 * 1024 * 1024, minimum=1),
            max_pages=_env_int("OCR_MAX_PAGES", 500, minimum=1),
            max_image_pixels=_env_int("OCR_MAX_IMAGE_PIXELS", 25_000_000, minimum=1),
            max_document_rendered_pixels=_env_int(
                "OCR_MAX_DOCUMENT_RENDERED_PIXELS",
                1_000_000_000,
                minimum=1,
            ),
            max_document_seconds=_env_float(
                "OCR_MAX_DOCUMENT_SECONDS",
                300.0,
                minimum=1.0,
            ),
            max_output_characters=_env_int(
                "OCR_MAX_OUTPUT_CHARACTERS",
                10_000_000,
                minimum=1,
            ),
            max_concurrency=_env_int("OCR_MAX_CONCURRENCY", 2, minimum=1),
            queue_timeout_seconds=_env_float(
                "OCR_QUEUE_TIMEOUT_SECONDS",
                2.0,
                minimum=0.1,
            ),
            cache_max_bytes=_env_int("OCR_CACHE_MAX_BYTES", 64 * 1024 * 1024),
            cache_max_entries=_env_int("OCR_CACHE_MAX_ENTRIES", 512, minimum=1),
            cache_ttl_seconds=_env_int("OCR_CACHE_TTL_SECONDS", 900),
            allow_http_urls=_env_bool("OCR_ALLOW_HTTP_URLS", False),
            remote_allowed_hosts=_env_csv("OCR_REMOTE_ALLOWED_HOSTS"),
            remote_connect_timeout_seconds=_env_float(
                "OCR_REMOTE_CONNECT_TIMEOUT_SECONDS",
                10.0,
                minimum=0.1,
            ),
            remote_total_timeout_seconds=_env_float(
                "OCR_REMOTE_TOTAL_TIMEOUT_SECONDS",
                60.0,
                minimum=1.0,
            ),
            enable_rapidocr=_env_bool("OCR_ENABLE_RAPIDOCR", directml_default),
            rapidocr_use_directml=_env_bool("OCR_RAPIDOCR_USE_DIRECTML", directml_default),
            rapidocr_require_accelerator=_env_bool(
                "OCR_RAPIDOCR_REQUIRE_ACCELERATOR",
                directml_default,
            ),
            rapidocr_min_confidence=_env_float(
                "OCR_RAPIDOCR_MIN_CONFIDENCE",
                55.0,
            ),
            rapidocr_max_side=_env_int(
                "OCR_RAPIDOCR_MAX_SIDE",
                2_048,
                minimum=736,
            ),
            rapidocr_model_root=os.getenv("OCR_RAPIDOCR_MODEL_ROOT"),
            rapidocr_isolate_process=_env_bool(
                "OCR_RAPIDOCR_ISOLATE_PROCESS",
                directml_default,
            ),
            rapidocr_inference_timeout_seconds=_env_float(
                "OCR_RAPIDOCR_INFERENCE_TIMEOUT_SECONDS",
                15.0,
                minimum=1.0,
            ),
            rapidocr_recycle_after_calls=_env_int(
                "OCR_RAPIDOCR_RECYCLE_AFTER_CALLS",
                100,
                minimum=1,
            ),
            enable_dense_table_guard=_env_bool(
                "OCR_ENABLE_DENSE_TABLE_GUARD",
                True,
            ),
            dense_table_layout_confidence=_env_float(
                "OCR_DENSE_TABLE_LAYOUT_CONFIDENCE",
                0.80,
            ),
            dense_table_min_area_ratio=_env_float(
                "OCR_DENSE_TABLE_MIN_AREA_RATIO",
                0.50,
            ),
            dense_table_tesseract_psm=_env_int(
                "OCR_DENSE_TABLE_TESSERACT_PSM",
                6,
                minimum=1,
            ),
            preload_models=_env_bool("OCR_PRELOAD_MODELS", False),
            enable_document_layout=_env_bool(
                "OCR_ENABLE_DOCUMENT_LAYOUT",
                True,
            ),
            enable_formula_ocr=_env_bool("OCR_ENABLE_FORMULA_OCR", True),
            layout_model_type=os.getenv("OCR_LAYOUT_MODEL_TYPE", "pp_layout_cdla"),
            layout_model_path=os.getenv("OCR_LAYOUT_MODEL_PATH"),
            layout_confidence_threshold=_env_float(
                "OCR_LAYOUT_CONFIDENCE_THRESHOLD",
                0.35,
            ),
        )
