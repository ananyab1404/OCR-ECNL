from __future__ import annotations

import json
import logging
import math
import os
import re
import time
import unicodedata
from contextlib import redirect_stdout
from dataclasses import dataclass, field, replace
from hashlib import sha256
from io import BytesIO, StringIO
from pathlib import Path
from typing import Literal

import fitz
import pytesseract
from PIL import Image, ImageOps, UnidentifiedImageError
from pytesseract import Output

from app.core.ocr_config import OCRSettings
from app.services.ocr_cache import MemoryTTLCache, OCRCache
from app.services.rapid_layout_backend import (
    RapidLayoutBackend,
    RapidLayoutBackendError,
)
from app.services.rapid_ocr_backend import (
    RapidOCRBackend,
    RapidOCRBackendError,
)

PIPELINE_VERSION = "pymupdf-rapidocr-layout-v4"
logger = logging.getLogger(__name__)
BlockKind = Literal["text", "image", "table", "formula", "unknown"]
BlockSource = Literal[
    "native",
    "tesseract",
    "rapidocr",
    "rapidlayout",
    "formula",
]


class OCRError(RuntimeError):
    """Base class for errors safe to map at the API boundary."""


class OCRInputError(OCRError):
    pass


class OCRLimitError(OCRError):
    pass


class OCRTimeoutError(OCRError):
    pass


class OCRDependencyError(OCRError):
    pass


@dataclass(frozen=True, slots=True)
class OCRBoundingBox:
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass(frozen=True, slots=True)
class OCRBlockResult:
    kind: BlockKind
    bbox: OCRBoundingBox
    text: str | None = None
    latex: str | None = None
    confidence: float | None = None
    source: BlockSource = "native"
    metadata: dict[str, str | int | float | bool | None] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OCRPageDetail:
    page_number: int
    text: str
    width: float
    height: float
    blocks: list[OCRBlockResult] = field(default_factory=list)
    confidence: float | None = None
    method: str = "native"
    warnings: list[str] = field(default_factory=list)
    processing_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class OCRResult:
    text: str
    confidence: float | None = None
    pages: list[tuple[int, str]] | None = None
    source_type: str = "image"
    page_details: list[OCRPageDetail] | None = None
    processing_seconds: float | None = None
    cache_hit: bool = False


@dataclass(frozen=True, slots=True)
class _RasterOCRResult:
    text: str
    confidence: float | None
    blocks: list[OCRBlockResult]
    dpi: int
    backend: Literal["tesseract", "rapidocr"]
    psm: int | None = None
    provider: str | None = None
    blank: bool = False
    warnings: tuple[str, ...] = ()


@dataclass(slots=True)
class _WorkBudget:
    deadline: float
    max_rendered_pixels: int
    rendered_pixels: int = 0

    def check_deadline(self) -> None:
        if time.perf_counter() > self.deadline:
            raise OCRTimeoutError("OCR document processing deadline exceeded")

    def charge_render(self, width: int, height: int) -> None:
        self.check_deadline()
        self.rendered_pixels += width * height
        if self.rendered_pixels > self.max_rendered_pixels:
            raise OCRLimitError(
                "OCR document rendered-pixel budget exceeded: "
                f"{self.rendered_pixels} > {self.max_rendered_pixels}"
            )

    def ensure_render_fits(self, width: int, height: int) -> None:
        self.check_deadline()
        projected_total = self.rendered_pixels + width * height
        if projected_total > self.max_rendered_pixels:
            raise OCRLimitError(
                "OCR document rendered-pixel budget would be exceeded: "
                f"{projected_total} > {self.max_rendered_pixels}"
            )


class OCRService:
    def __init__(
        self,
        settings: OCRSettings | None = None,
        cache: OCRCache[OCRResult] | None = None,
        rapidocr_backend: RapidOCRBackend | None = None,
        layout_backend: RapidLayoutBackend | None = None,
    ) -> None:
        self.settings = settings or OCRSettings.from_env()
        self._rapidocr_model_fingerprint = self._fingerprint_model_root(
            self.settings.rapidocr_model_root
        )
        self.cache = cache or MemoryTTLCache[OCRResult](
            max_bytes=self.settings.cache_max_bytes,
            ttl_seconds=self.settings.cache_ttl_seconds,
            max_entries=self.settings.cache_max_entries,
        )
        self.rapidocr_backend = rapidocr_backend
        if self.rapidocr_backend is None and self.settings.enable_rapidocr:
            self.rapidocr_backend = RapidOCRBackend(
                use_directml=self.settings.rapidocr_use_directml,
                require_accelerator=self.settings.rapidocr_require_accelerator,
                max_side=self.settings.rapidocr_max_side,
                model_root=self.settings.rapidocr_model_root,
                isolate_process=self.settings.rapidocr_isolate_process,
                inference_timeout_seconds=(
                    self.settings.rapidocr_inference_timeout_seconds
                ),
                recycle_after_calls=self.settings.rapidocr_recycle_after_calls,
            )
        self.layout_backend = layout_backend
        if self.layout_backend is None and self.settings.enable_document_layout:
            self.layout_backend = RapidLayoutBackend(
                model_type=self.settings.layout_model_type,
                model_path=self.settings.layout_model_path,
                confidence_threshold=self.settings.layout_confidence_threshold,
                use_directml=self.settings.rapidocr_use_directml,
                require_accelerator=self.settings.rapidocr_require_accelerator,
            )
        self._configure_tesseract()
        if self.settings.preload_models:
            if self.rapidocr_backend is not None:
                try:
                    self.rapidocr_backend.preload()
                except RapidOCRBackendError as exc:
                    logger.warning(
                        "RapidOCR preload failed; Tesseract remains available: %s",
                        exc,
                    )
            if self.layout_backend is not None:
                try:
                    self.layout_backend.preload()
                except RapidLayoutBackendError as exc:
                    logger.warning(
                        "Document layout preload failed; basic blocks remain available: %s",
                        exc,
                    )

    def close(self) -> None:
        closer = getattr(self.rapidocr_backend, "close", None)
        if callable(closer):
            closer()

    def extract_text(
        self,
        file_bytes: bytes,
        filename: str | None = None,
        include_layout: bool = False,
    ) -> OCRResult:
        started = time.perf_counter()
        self._validate_file_size(file_bytes)
        budget = _WorkBudget(
            deadline=started + self.settings.max_document_seconds,
            max_rendered_pixels=self.settings.max_document_rendered_pixels,
        )
        cache_key = self._cache_key(file_bytes, filename, include_layout)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return replace(
                cached,
                cache_hit=True,
                processing_seconds=time.perf_counter() - started,
            )

        if self._looks_like_pdf(file_bytes, filename):
            result = self._extract_pdf_text(
                file_bytes,
                include_layout=include_layout,
                budget=budget,
            )
        else:
            result = self._extract_image_text(
                file_bytes,
                include_layout=include_layout,
                budget=budget,
            )

        budget.check_deadline()
        self._validate_result_output(result)
        elapsed = time.perf_counter() - started
        result = replace(result, processing_seconds=elapsed)
        self.cache.set(cache_key, result, self._estimate_result_bytes(result))
        return result

    def _extract_pdf_text(
        self,
        pdf_bytes: bytes,
        *,
        include_layout: bool,
        budget: _WorkBudget,
    ) -> OCRResult:
        try:
            document = fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception as exc:
            raise OCRInputError(f"Unable to open PDF: {exc}") from exc

        with document:
            if document.needs_pass:
                raise OCRInputError("Encrypted PDFs are not supported")
            if document.page_count > self.settings.max_pages:
                raise OCRLimitError(
                    f"PDF contains {document.page_count} pages; limit is {self.settings.max_pages}"
                )

            extracted_pages: list[tuple[int, str]] = []
            page_details: list[OCRPageDetail] = []
            ocr_confidences: list[float] = []
            output_characters = 0

            for page_number, page in enumerate(document, start=1):
                budget.check_deadline()
                page_started = time.perf_counter()
                detail = self._extract_pdf_page(
                    document,
                    page,
                    page_number,
                    include_layout=include_layout,
                    budget=budget,
                )
                detail = replace(
                    detail,
                    processing_seconds=time.perf_counter() - page_started,
                )
                output_characters += len(detail.text)
                if output_characters > self.settings.max_output_characters:
                    raise OCRLimitError(
                        "OCR output character limit exceeded: "
                        f"{output_characters} > {self.settings.max_output_characters}"
                    )
                extracted_pages.append((page_number, detail.text))
                page_details.append(detail)
                if detail.confidence is not None:
                    ocr_confidences.append(detail.confidence)

        combined_text = "\n\n".join(page_text for _, page_text in extracted_pages if page_text)
        confidence = (
            round(sum(ocr_confidences) / len(ocr_confidences), 2) if ocr_confidences else None
        )
        return OCRResult(
            text=combined_text.strip(),
            confidence=confidence,
            pages=extracted_pages,
            source_type="pdf",
            page_details=page_details,
        )

    def _extract_pdf_page(
        self,
        document: fitz.Document,
        page: fitz.Page,
        page_number: int,
        *,
        include_layout: bool,
        budget: _WorkBudget,
    ) -> OCRPageDetail:
        native_blocks = self._native_text_blocks(page)
        native_text = "\n".join(
            block.text.strip() for block in native_blocks if block.text and block.text.strip()
        ).strip()
        image_blocks = self._image_blocks(document, page)
        image_ratio = self._covered_page_ratio(page, image_blocks)

        if self._native_text_is_sufficient(native_text, image_ratio):
            warnings: list[str] = []
            blocks: list[OCRBlockResult] = []
            if include_layout:
                layout_blocks, layout_warnings = self._layout_pdf_page(
                    page,
                    budget=budget,
                )
                warnings.extend(layout_warnings)
                supplemental_ocr_blocks: list[OCRBlockResult] = []
                if self._needs_supplemental_image_ocr(image_ratio):
                    supplemental = self._ocr_pdf_page(page, budget=budget)
                    warnings.extend(supplemental.warnings)
                    supplemental_ocr_blocks = [
                        replace(
                            block,
                            metadata={
                                **block.metadata,
                                "supplemental": True,
                            },
                        )
                        for block in supplemental.blocks
                    ]
                    if not supplemental.text and not supplemental.blank:
                        warnings.append("supplemental_image_ocr_returned_no_text")
                blocks = [
                    *native_blocks,
                    *self._native_table_blocks(page),
                    *image_blocks,
                    *layout_blocks,
                    *supplemental_ocr_blocks,
                ]
            return OCRPageDetail(
                page_number=page_number,
                text=native_text,
                width=float(page.rect.width),
                height=float(page.rect.height),
                blocks=blocks,
                method="native",
                warnings=warnings,
            )

        warnings: list[str] = []
        if native_text:
            warnings.append(
                "native_text_merged_with_page_ocr"
                if self._native_text_is_plausible(native_text)
                else "native_text_failed_quality_gate"
            )

        ocr_result = self._ocr_pdf_page(page, budget=budget)
        warnings.extend(ocr_result.warnings)
        page_text = ocr_result.text
        if (
            native_text
            and self._native_text_is_plausible(native_text)
            and self._normalized_comparison_text(native_text)
            not in self._normalized_comparison_text(ocr_result.text)
        ):
            page_text = "\n".join(part for part in (native_text, ocr_result.text) if part)
        blocks: list[OCRBlockResult] = []
        if include_layout:
            layout_blocks, layout_warnings = self._layout_pdf_page(
                page,
                budget=budget,
            )
            warnings.extend(layout_warnings)
            blocks = [
                *native_blocks,
                *image_blocks,
                *layout_blocks,
                *ocr_result.blocks,
            ]
        if not ocr_result.text and not ocr_result.blank:
            warnings.append("ocr_returned_no_text")

        return OCRPageDetail(
            page_number=page_number,
            text=page_text,
            width=float(page.rect.width),
            height=float(page.rect.height),
            blocks=blocks,
            confidence=ocr_result.confidence,
            method=self._ocr_method(ocr_result),
            warnings=warnings,
        )

    def _extract_image_text(
        self,
        file_bytes: bytes,
        *,
        include_layout: bool,
        budget: _WorkBudget,
    ) -> OCRResult:
        try:
            image = Image.open(BytesIO(file_bytes))
        except (UnidentifiedImageError, OSError) as exc:
            raise OCRInputError(f"Unsupported or malformed image: {exc}") from exc

        self._validate_image_pixels(image.width, image.height)
        try:
            image.load()
        except OSError as exc:
            raise OCRInputError(f"Unsupported or malformed image: {exc}") from exc
        image = ImageOps.exif_transpose(image)
        self._validate_image_pixels(image.width, image.height)
        budget.charge_render(image.width, image.height)
        if image.mode not in {"L", "RGB"}:
            image = image.convert("RGB")

        result = self._run_primary_raster_ocr(
            image,
            page_width=float(image.width),
            page_height=float(image.height),
            pixels_per_page_unit=1.0,
            dpi=self.settings.render_dpi,
        )
        if len(result.text) > self.settings.max_output_characters:
            raise OCRLimitError(
                "OCR output character limit exceeded: "
                f"{len(result.text)} > {self.settings.max_output_characters}"
            )
        page_blocks = result.blocks if include_layout else []
        layout_warnings: list[str] = []
        if include_layout:
            layout_blocks, layout_warnings = self._layout_image_blocks(
                image,
                page_width=float(image.width),
                page_height=float(image.height),
                pixels_per_page_unit=1.0,
            )
            page_blocks = [
                OCRBlockResult(
                    kind="image",
                    bbox=OCRBoundingBox(0, 0, float(image.width), float(image.height)),
                    source="native",
                    metadata={
                        "width": image.width,
                        "height": image.height,
                        "format": image.format,
                    },
                ),
                *layout_blocks,
                *page_blocks,
            ]

        page_detail = OCRPageDetail(
            page_number=1,
            text=result.text,
            width=float(image.width),
            height=float(image.height),
            blocks=page_blocks,
            confidence=result.confidence,
            method=self._ocr_method(result),
            warnings=[
                *result.warnings,
                *layout_warnings,
                *(
                    []
                    if result.text or result.blank
                    else ["ocr_returned_no_text"]
                ),
            ],
        )
        return OCRResult(
            text=result.text,
            confidence=result.confidence,
            pages=[(1, result.text)] if result.text else [],
            source_type="image",
            page_details=[page_detail],
        )

    def _ocr_pdf_page(
        self,
        page: fitz.Page,
        *,
        budget: _WorkBudget,
    ) -> _RasterOCRResult:
        image, scale, effective_dpi = self._render_page(
            page,
            self.settings.render_dpi,
            grayscale=self.rapidocr_backend is None,
            budget=budget,
        )
        if self._dense_table_guard_matches(image):
            guarded = self._run_tesseract(
                image if image.mode == "L" else ImageOps.grayscale(image),
                page_width=float(page.rect.width),
                page_height=float(page.rect.height),
                pixels_per_page_unit=scale,
                dpi=effective_dpi,
                psm=self.settings.dense_table_tesseract_psm,
            )
            return replace(
                guarded,
                warnings=("dense_ruled_table_tesseract_guard",),
            )
        first = self._run_primary_raster_ocr(
            image,
            page_width=float(page.rect.width),
            page_height=float(page.rect.height),
            pixels_per_page_unit=scale,
            dpi=effective_dpi,
        )

        if first.backend == "rapidocr" or first.blank:
            return first
        if not self._should_retry_tesseract(first):
            return first

        retry_image, retry_scale, retry_dpi = self._render_page(
            page,
            self.settings.retry_dpi,
            grayscale=True,
            budget=budget,
        )
        retry = self._run_tesseract(
            retry_image,
            page_width=float(page.rect.width),
            page_height=float(page.rect.height),
            pixels_per_page_unit=retry_scale,
            dpi=retry_dpi,
            psm=6,
        )
        best = max((first, retry), key=self._tesseract_score)
        return replace(best, warnings=first.warnings)

    def _render_page(
        self,
        page: fitz.Page,
        requested_dpi: int,
        *,
        grayscale: bool,
        budget: _WorkBudget | None = None,
    ) -> tuple[Image.Image, float, int]:
        if budget is not None:
            budget.check_deadline()
        scale = requested_dpi / 72.0
        projected_pixels = page.rect.width * scale * page.rect.height * scale
        if projected_pixels > self.settings.max_image_pixels:
            scale *= math.sqrt(self.settings.max_image_pixels / projected_pixels)

        pixmap: fitz.Pixmap | None = None
        for _ in range(4):
            if budget is not None:
                budget.ensure_render_fits(
                    max(1, math.ceil(page.rect.width * scale)),
                    max(1, math.ceil(page.rect.height * scale)),
                )
            matrix = fitz.Matrix(scale, scale)
            pixmap = page.get_pixmap(
                matrix=matrix,
                colorspace=fitz.csGRAY if grayscale else fitz.csRGB,
                alpha=False,
            )
            if budget is not None:
                budget.charge_render(pixmap.width, pixmap.height)
            actual_pixels = pixmap.width * pixmap.height
            if actual_pixels <= self.settings.max_image_pixels:
                break
            # MuPDF rounds output dimensions independently, so the analytical
            # scale can land a few pixels above the limit. Back off using the
            # measured raster size and render again.
            scale *= math.sqrt(self.settings.max_image_pixels / actual_pixels) * 0.999
        else:  # pragma: no cover - defensive guard for unusual page transforms
            raise OCRLimitError("Unable to render page within the configured pixel limit")

        if pixmap is None:  # pragma: no cover - loop always executes
            raise OCRLimitError("Unable to render page")
        mode = "L" if grayscale else "RGB"
        image = Image.frombytes(mode, (pixmap.width, pixmap.height), pixmap.samples)
        return image, scale, max(72, round(scale * 72))

    def _dense_table_guard_matches(self, image: Image.Image) -> bool:
        if (
            not self.settings.enable_dense_table_guard
            or not self.settings.tesseract_cmd
            or self.layout_backend is None
            or not self._is_ruled_table_candidate(image)
        ):
            return False
        try:
            result = self.layout_backend.analyze(image)
        except RapidLayoutBackendError as exc:
            logger.warning("Dense-table layout confirmation unavailable: %s", exc)
            return False

        page_area = max(1.0, float(image.width * image.height))
        minimum_confidence = self.settings.dense_table_layout_confidence * 100.0
        for region in result.regions:
            if region.label.casefold() != "table":
                continue
            x0, y0, x1, y1 = region.bbox
            region_area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
            if (
                region.confidence >= minimum_confidence
                and region_area / page_area >= self.settings.dense_table_min_area_ratio
            ):
                return True
        return False

    def _is_ruled_table_candidate(self, image: Image.Image) -> bool:
        """Cheaply identify pathological full-page ruled tables before OCR."""
        try:
            import cv2
            import numpy as np
        except ImportError:
            logger.warning("Dense-table guard requires OpenCV and numpy")
            return False

        grayscale = np.asarray(ImageOps.grayscale(image))
        height, width = grayscale.shape
        max_side = max(height, width)
        if max_side > 1_684:
            scale = 1_684 / max_side
            grayscale = cv2.resize(
                grayscale,
                (
                    max(1, round(width * scale)),
                    max(1, round(height * scale)),
                ),
                interpolation=cv2.INTER_AREA,
            )

        binary = cv2.threshold(
            grayscale,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )[1]
        ink_ratio = float(np.count_nonzero(binary)) / max(1, binary.size)
        if not 0.01 <= ink_ratio <= 0.20:
            return False

        horizontal = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1)),
        )
        vertical = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40)),
        )
        _, _, horizontal_stats, _ = cv2.connectedComponentsWithStats(horizontal)
        _, _, vertical_stats, _ = cv2.connectedComponentsWithStats(vertical)
        long_horizontal = sum(
            int(component[cv2.CC_STAT_WIDTH]) >= 100
            and int(component[cv2.CC_STAT_HEIGHT]) <= 8
            for component in horizontal_stats[1:]
        )
        long_vertical = sum(
            int(component[cv2.CC_STAT_HEIGHT]) >= 100
            and int(component[cv2.CC_STAT_WIDTH]) <= 8
            for component in vertical_stats[1:]
        )
        if long_horizontal < 3 or long_vertical < 3:
            return False

        row_projection = np.count_nonzero(binary, axis=1) / binary.shape[1]
        column_projection = np.count_nonzero(binary, axis=0) / binary.shape[0]
        dense_rows = int(np.count_nonzero(row_projection >= 0.60))
        dense_columns = int(np.count_nonzero(column_projection >= 0.50))
        return dense_rows >= 2 and dense_columns >= 2

    def _is_blank_image(self, image: Image.Image) -> bool:
        """Conservatively skip OCR when a rendered page contains no meaningful ink."""
        grayscale = image if image.mode == "L" else ImageOps.grayscale(image)
        pixel_count = max(1, grayscale.width * grayscale.height)
        histogram = grayscale.histogram()
        visibly_dark_pixels = sum(histogram[:245])
        return visibly_dark_pixels / pixel_count < 0.0002

    def _run_primary_raster_ocr(
        self,
        image: Image.Image,
        *,
        page_width: float,
        page_height: float,
        pixels_per_page_unit: float,
        dpi: int,
    ) -> _RasterOCRResult:
        if self._is_blank_image(image):
            return _RasterOCRResult(
                text="",
                confidence=None,
                blocks=[],
                dpi=dpi,
                backend="tesseract",
                psm=self.settings.tesseract_psm,
                blank=True,
            )

        fallback_warnings: list[str] = []
        accelerated_candidate: _RasterOCRResult | None = None
        if self.rapidocr_backend is not None:
            try:
                accelerated = self._run_rapidocr(
                    image,
                    page_width=page_width,
                    page_height=page_height,
                    pixels_per_page_unit=pixels_per_page_unit,
                    dpi=dpi,
                )
                if self._rapidocr_result_is_acceptable(accelerated):
                    return accelerated
                accelerated_candidate = accelerated
                fallback_warnings.append(
                    "rapidocr_below_acceptance_threshold"
                    if accelerated.text
                    else "rapidocr_returned_no_text"
                )
            except RapidOCRBackendError as exc:
                logger.warning("RapidOCR unavailable; using Tesseract: %s", exc)
                fallback_warnings.append("rapidocr_unavailable_tesseract_fallback")

        if not self.settings.tesseract_cmd and accelerated_candidate is not None:
            return replace(
                accelerated_candidate,
                warnings=tuple(fallback_warnings),
            )

        grayscale = image if image.mode == "L" else ImageOps.grayscale(image)
        tesseract_result = self._run_tesseract(
            grayscale,
            page_width=page_width,
            page_height=page_height,
            pixels_per_page_unit=pixels_per_page_unit,
            dpi=dpi,
            psm=self.settings.tesseract_psm,
        )
        return replace(tesseract_result, warnings=tuple(fallback_warnings))

    def _run_rapidocr(
        self,
        image: Image.Image,
        *,
        page_width: float,
        page_height: float,
        pixels_per_page_unit: float,
        dpi: int,
    ) -> _RasterOCRResult:
        if self.rapidocr_backend is None:  # pragma: no cover - guarded by caller
            raise RapidOCRBackendError("RapidOCR backend is disabled")
        result = self.rapidocr_backend.recognize(image)
        scale = max(pixels_per_page_unit, 1e-9)
        blocks: list[OCRBlockResult] = []
        confidence_total = 0.0
        confidence_weight = 0
        line_texts: list[str] = []
        for line in result.lines:
            if not line.quadrilateral:
                continue
            xs = [point[0] for point in line.quadrilateral]
            ys = [point[1] for point in line.quadrilateral]
            bbox = OCRBoundingBox(
                max(0.0, min(xs) / scale),
                max(0.0, min(ys) / scale),
                min(page_width, max(xs) / scale),
                min(page_height, max(ys) / scale),
            )
            line_texts.append(line.text)
            weight = max(1, len(line.text))
            confidence_total += line.confidence * weight
            confidence_weight += weight
            blocks.append(
                OCRBlockResult(
                    kind=(
                        "formula"
                        if self._looks_like_formula(line.text)
                        else "text"
                    ),
                    bbox=bbox,
                    text=line.text,
                    latex=(
                        self._formula_to_latex(line.text)
                        if self.settings.enable_formula_ocr
                        and self._looks_like_formula(line.text)
                        else None
                    ),
                    confidence=round(line.confidence, 2),
                    source="rapidocr",
                    metadata={
                        "provider": result.provider,
                        "engine_seconds": result.engine_seconds,
                    },
                )
            )
        confidence = (
            round(confidence_total / confidence_weight, 2)
            if confidence_weight
            else None
        )
        return _RasterOCRResult(
            text="\n".join(line_texts).strip(),
            confidence=confidence,
            blocks=blocks,
            dpi=dpi,
            backend="rapidocr",
            provider=result.provider,
        )

    def _layout_pdf_page(
        self,
        page: fitz.Page,
        *,
        budget: _WorkBudget,
    ) -> tuple[list[OCRBlockResult], list[str]]:
        if self.layout_backend is None:
            return [], []
        image, scale, _ = self._render_page(
            page,
            self.settings.render_dpi,
            grayscale=False,
            budget=budget,
        )
        return self._layout_image_blocks(
            image,
            page_width=float(page.rect.width),
            page_height=float(page.rect.height),
            pixels_per_page_unit=scale,
        )

    def _layout_image_blocks(
        self,
        image: Image.Image,
        *,
        page_width: float,
        page_height: float,
        pixels_per_page_unit: float,
    ) -> tuple[list[OCRBlockResult], list[str]]:
        if self.layout_backend is None:
            return [], []
        try:
            result = self.layout_backend.analyze(image)
        except RapidLayoutBackendError as exc:
            logger.warning("Document layout unavailable: %s", exc)
            return [], ["document_layout_unavailable"]

        scale = max(pixels_per_page_unit, 1e-9)
        blocks: list[OCRBlockResult] = []
        for region in result.regions:
            x0, y0, x1, y1 = region.bbox
            label = region.label.lower()
            if "equation" in label or "formula" in label:
                kind: BlockKind = "formula"
            elif "table" in label:
                kind = "table"
            elif "figure" in label or label in {"image", "chart"}:
                kind = "image"
            elif label in {
                "text",
                "title",
                "figure_caption",
                "table_caption",
                "header",
                "footer",
                "reference",
            }:
                kind = "text"
            else:
                kind = "unknown"
            blocks.append(
                OCRBlockResult(
                    kind=kind,
                    bbox=OCRBoundingBox(
                        max(0.0, x0 / scale),
                        max(0.0, y0 / scale),
                        min(page_width, x1 / scale),
                        min(page_height, y1 / scale),
                    ),
                    confidence=round(region.confidence, 2),
                    source="rapidlayout",
                    metadata={
                        "label": region.label,
                        "provider": result.provider,
                        "engine_seconds": result.engine_seconds,
                    },
                )
            )
        return blocks, []

    def _run_tesseract(
        self,
        image: Image.Image,
        *,
        page_width: float,
        page_height: float,
        pixels_per_page_unit: float,
        dpi: int,
        psm: int,
    ) -> _RasterOCRResult:
        if not self.settings.tesseract_cmd:
            raise OCRDependencyError(
                "Tesseract is required for scanned pages but was not found. "
                "Set TESSERACT_CMD to the executable path."
            )

        config = f"--oem {self.settings.tesseract_oem} --psm {psm} --dpi {dpi}"
        try:
            data = pytesseract.image_to_data(
                image,
                lang=self.settings.tesseract_lang,
                config=config,
                output_type=Output.DICT,
                timeout=self.settings.tesseract_timeout_seconds,
            )
        except pytesseract.TesseractNotFoundError as exc:
            raise OCRDependencyError(str(exc)) from exc
        except RuntimeError as exc:
            if "timeout" in str(exc).lower():
                raise OCRTimeoutError(
                    f"Tesseract exceeded {self.settings.tesseract_timeout_seconds}s"
                ) from exc
            raise OCRError(f"Tesseract failed: {exc}") from exc
        except pytesseract.TesseractError as exc:
            raise OCRError(f"Tesseract failed: {exc}") from exc

        blocks, text, confidence = self._tsv_to_blocks(
            data,
            page_width=page_width,
            page_height=page_height,
            pixels_per_page_unit=pixels_per_page_unit,
        )
        return _RasterOCRResult(
            text=text,
            confidence=confidence,
            blocks=blocks,
            dpi=dpi,
            backend="tesseract",
            psm=psm,
        )

    def _tsv_to_blocks(
        self,
        data: dict[str, list[object]],
        *,
        page_width: float,
        page_height: float,
        pixels_per_page_unit: float,
    ) -> tuple[list[OCRBlockResult], str, float | None]:
        lines: dict[tuple[int, int, int], list[dict[str, float | str]]] = {}
        confidence_total = 0.0
        confidence_weight = 0

        for index, raw_text in enumerate(data.get("text", [])):
            text = str(raw_text).strip()
            if not text:
                continue
            try:
                confidence = float(data["conf"][index])
            except (KeyError, TypeError, ValueError, IndexError):
                confidence = -1
            if confidence < 0:
                continue

            key = (
                int(data["block_num"][index]),
                int(data["par_num"][index]),
                int(data["line_num"][index]),
            )
            left = float(data["left"][index])
            top = float(data["top"][index])
            width = float(data["width"][index])
            height = float(data["height"][index])
            lines.setdefault(key, []).append(
                {
                    "text": text,
                    "confidence": confidence,
                    "left": left,
                    "top": top,
                    "right": left + width,
                    "bottom": top + height,
                }
            )
            weight = max(1, len(text))
            confidence_total += confidence * weight
            confidence_weight += weight

        blocks: list[OCRBlockResult] = []
        line_texts: list[str] = []
        scale = max(pixels_per_page_unit, 1e-9)
        for words in lines.values():
            line_text = " ".join(str(word["text"]) for word in words)
            line_texts.append(line_text)
            line_confidence = sum(
                float(word["confidence"]) * max(1, len(str(word["text"]))) for word in words
            ) / sum(max(1, len(str(word["text"]))) for word in words)
            x0 = max(0.0, min(float(word["left"]) for word in words) / scale)
            y0 = max(0.0, min(float(word["top"]) for word in words) / scale)
            x1 = min(page_width, max(float(word["right"]) for word in words) / scale)
            y1 = min(page_height, max(float(word["bottom"]) for word in words) / scale)
            blocks.append(
                OCRBlockResult(
                    kind=(
                        "formula"
                        if self._looks_like_formula(line_text)
                        else "text"
                    ),
                    bbox=OCRBoundingBox(x0, y0, x1, y1),
                    text=line_text,
                    latex=(
                        self._formula_to_latex(line_text)
                        if self.settings.enable_formula_ocr
                        and self._looks_like_formula(line_text)
                        else None
                    ),
                    confidence=round(line_confidence, 2),
                    source="tesseract",
                )
            )

        confidence = round(confidence_total / confidence_weight, 2) if confidence_weight else None
        return blocks, "\n".join(line_texts).strip(), confidence

    def _native_text_blocks(self, page: fitz.Page) -> list[OCRBlockResult]:
        flags = fitz.TEXTFLAGS_BLOCKS & ~fitz.TEXT_PRESERVE_IMAGES
        blocks: list[OCRBlockResult] = []
        for raw_block in page.get_text("blocks", sort=True, flags=flags):
            if len(raw_block) < 7 or int(raw_block[6]) != 0:
                continue
            text = str(raw_block[4]).strip()
            if not text:
                continue
            blocks.append(
                OCRBlockResult(
                    kind=(
                        "formula"
                        if self._looks_like_formula(text)
                        else "text"
                    ),
                    bbox=OCRBoundingBox(
                        float(raw_block[0]),
                        float(raw_block[1]),
                        float(raw_block[2]),
                        float(raw_block[3]),
                    ),
                    text=text,
                    latex=(
                        self._formula_to_latex(text)
                        if self.settings.enable_formula_ocr
                        and self._looks_like_formula(text)
                        else None
                    ),
                    source="native",
                )
            )
        return blocks

    def _native_table_blocks(self, page: fitz.Page) -> list[OCRBlockResult]:
        try:
            # Some PyMuPDF versions print an optional-package suggestion from
            # find_tables(); keep request logs clean.
            with redirect_stdout(StringIO()):
                finder = page.find_tables()
            tables = list(finder.tables)
        except Exception as exc:  # noqa: BLE001 - table detection is optional
            logger.debug("Unable to inspect native PDF tables: %s", exc)
            return []

        blocks: list[OCRBlockResult] = []
        for table in tables:
            try:
                rows = table.extract()
                markdown = table.to_markdown().strip()
                bbox = table.bbox
            except Exception as exc:  # noqa: BLE001 - preserve all other blocks
                logger.debug("Unable to serialize native PDF table: %s", exc)
                continue
            column_count = max((len(row) for row in rows), default=0)
            blocks.append(
                OCRBlockResult(
                    kind="table",
                    bbox=OCRBoundingBox(
                        float(bbox[0]),
                        float(bbox[1]),
                        float(bbox[2]),
                        float(bbox[3]),
                    ),
                    text=markdown,
                    source="native",
                    metadata={
                        "format": "markdown",
                        "rows": len(rows),
                        "columns": column_count,
                    },
                )
            )
        return blocks

    def _looks_like_formula(self, text: str) -> bool:
        compact = " ".join(text.split())
        if not compact or len(compact) > 300:
            return False
        math_markers = set("=≈±×÷√∑∫∞≤≥≠∂∇∈∉⊂⊆∪∩")
        unicode_scripts = set("₀₁₂₃₄₅₆₇₈₉₊₋₌⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼")
        greek = set("αβγδεζηθικλμνξοπρστυφχψωΓΔΘΛΞΠΣΦΨΩπ")
        marker_count = sum(
            character in math_markers
            or character in unicode_scripts
            or character in greek
            for character in compact
        )
        has_operand = any(character.isdigit() for character in compact)
        has_operator = any(character in math_markers for character in compact)
        return marker_count >= 2 and (has_operand or has_operator)

    def _formula_to_latex(self, text: str) -> str:
        substitutions = {
            "≈": r"\approx ",
            "±": r"\pm ",
            "×": r"\times ",
            "÷": r"\div ",
            "·": r"\cdot ",
            "≤": r"\le ",
            "≥": r"\ge ",
            "≠": r"\ne ",
            "∞": r"\infty ",
            "∑": r"\sum ",
            "∫": r"\int ",
            "∂": r"\partial ",
            "∇": r"\nabla ",
            "∈": r"\in ",
            "∉": r"\notin ",
            "⊂": r"\subset ",
            "⊆": r"\subseteq ",
            "∪": r"\cup ",
            "∩": r"\cap ",
            "π": r"\pi ",
            "ω": r"\omega ",
            "Ω": r"\Omega ",
            "α": r"\alpha ",
            "β": r"\beta ",
            "γ": r"\gamma ",
            "δ": r"\delta ",
            "ε": r"\epsilon ",
            "θ": r"\theta ",
            "λ": r"\lambda ",
            "μ": r"\mu ",
            "ρ": r"\rho ",
            "σ": r"\sigma ",
            "φ": r"\phi ",
        }
        subscript_map = str.maketrans("₀₁₂₃₄₅₆₇₈₉₊₋₌", "0123456789+-=")
        superscript_map = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼", "0123456789+-=")
        subscript_chars = "₀₁₂₃₄₅₆₇₈₉₊₋₌"
        superscript_chars = "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼"

        formula = " ".join(text.split())
        formula = re.sub(
            rf"([{re.escape(subscript_chars)}]+)",
            lambda match: "_{" + match.group(1).translate(subscript_map) + "}",
            formula,
        )
        formula = re.sub(
            rf"([{re.escape(superscript_chars)}]+)",
            lambda match: "^{" + match.group(1).translate(superscript_map) + "}",
            formula,
        )
        formula = re.sub(
            r"√\s*\[([^\]]+)\]",
            lambda match: r"\sqrt{" + match.group(1) + "}",
            formula,
        )
        formula = re.sub(
            r"√\s*\(([^)]+)\)",
            lambda match: r"\sqrt{" + match.group(1) + "}",
            formula,
        )
        for source, target in substitutions.items():
            formula = formula.replace(source, target)
        formula = formula.replace("=>", r"\Rightarrow ")
        formula = re.sub(
            r"\b([A-Za-z])_([A-Za-z0-9]+)\b",
            lambda match: f"{match.group(1)}_{{{match.group(2)}}}",
            formula,
        )
        formula = re.sub(
            r"\b([A-Za-z])(\d+)\b",
            lambda match: f"{match.group(1)}_{{{match.group(2)}}}",
            formula,
        )
        formula = re.sub(r"\s+", " ", formula).strip()
        return formula

    def _image_blocks(
        self,
        document: fitz.Document,
        page: fitz.Page,
    ) -> list[OCRBlockResult]:
        blocks: list[OCRBlockResult] = []
        try:
            # Asking PyMuPDF to resolve xrefs hashes every image against every
            # shared PDF resource. A 95-page corpus document measured 7.4s per
            # page with xrefs versus 0.009s for the displayed-image index.
            image_info = page.get_image_info(hashes=False, xrefs=False)
        except (RuntimeError, ValueError) as exc:
            logger.debug("Unable to inspect displayed PDF images: %s", exc)
            return []

        seen: set[tuple[int, float, float, float, float]] = set()
        for item in image_info:
            rect = fitz.Rect(item["bbox"])
            if rect.is_empty or rect.is_infinite:
                continue
            image_number = int(item.get("number", -1))
            key = (
                image_number,
                round(rect.x0, 3),
                round(rect.y0, 3),
                round(rect.x1, 3),
                round(rect.y1, 3),
            )
            if key in seen:
                continue
            seen.add(key)
            blocks.append(
                OCRBlockResult(
                    kind="image",
                    bbox=OCRBoundingBox(rect.x0, rect.y0, rect.x1, rect.y1),
                    source="native",
                    metadata={
                        "image_number": image_number,
                        "width": int(item.get("width", 0)),
                        "height": int(item.get("height", 0)),
                        "bits_per_component": int(item.get("bpc", 0)),
                        "colorspace": str(item.get("cs-name", "")),
                        "x_resolution": int(item.get("xres", 0)),
                        "y_resolution": int(item.get("yres", 0)),
                        "encoded_bytes": int(item.get("size", 0)),
                        "has_mask": bool(item.get("has-mask", False)),
                        "document_pages": document.page_count,
                    },
                )
            )
        return blocks

    def _covered_page_ratio(
        self,
        page: fitz.Page,
        image_blocks: list[OCRBlockResult],
    ) -> float:
        page_area = max(1.0, float(page.rect.get_area()))
        image_area = 0.0
        for block in image_blocks:
            rect = fitz.Rect(
                block.bbox.x0,
                block.bbox.y0,
                block.bbox.x1,
                block.bbox.y1,
            )
            rect &= page.rect
            image_area += max(0.0, float(rect.get_area()))
        return min(1.0, image_area / page_area)

    def _native_text_is_sufficient(self, text: str, image_ratio: float) -> bool:
        if not self._native_text_is_plausible(text):
            return False
        alphanumeric_count = sum(character.isalnum() for character in text)
        word_count = len(re.findall(r"\b[\w'-]+\b", text, flags=re.UNICODE))
        enough_text = (
            alphanumeric_count >= self.settings.native_text_min_chars
            and word_count >= self.settings.native_text_min_words
        )
        native_only_page = alphanumeric_count > 0 and image_ratio < 0.1
        return native_only_page or enough_text

    def _needs_supplemental_image_ocr(self, image_ratio: float) -> bool:
        return image_ratio >= self.settings.image_dominant_ratio

    def _native_text_is_plausible(self, text: str) -> bool:
        non_whitespace = [character for character in text if not character.isspace()]
        if not non_whitespace:
            return False
        invalid = sum(
            unicodedata.category(character) in {"Cc", "Cs"}
            or character == "\ufffd"
            for character in non_whitespace
        )
        return invalid / len(non_whitespace) <= 0.02

    def _normalized_comparison_text(self, text: str) -> str:
        return "".join(character.casefold() for character in text if character.isalnum())

    def _rapidocr_result_is_acceptable(self, result: _RasterOCRResult) -> bool:
        alphanumeric_count = sum(character.isalnum() for character in result.text)
        return (
            bool(result.text)
            and alphanumeric_count >= 3
            and result.confidence is not None
            and result.confidence >= self.settings.rapidocr_min_confidence
            and self._native_text_is_plausible(result.text)
        )

    def _should_retry_tesseract(self, result: _RasterOCRResult) -> bool:
        if result.backend != "tesseract":
            return False
        if result.psm == 6:
            return False
        if not result.text:
            return True
        return (
            result.confidence is not None
            and result.confidence < self.settings.retry_confidence_threshold
        )

    def _tesseract_score(self, result: _RasterOCRResult) -> tuple[float, int]:
        return (result.confidence or 0.0, len(result.text))

    def _ocr_method(self, result: _RasterOCRResult) -> str:
        if result.blank:
            return "blank"
        if result.backend == "rapidocr":
            provider = (result.provider or "unknown").replace("ExecutionProvider", "").lower()
            return f"rapidocr-{result.dpi}dpi-{provider}"
        return f"tesseract-{result.dpi}dpi-psm{result.psm}"

    def _configure_tesseract(self) -> None:
        if self.settings.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = self.settings.tesseract_cmd
        os.environ.setdefault(
            "OMP_THREAD_LIMIT",
            str(self.settings.tesseract_thread_limit),
        )

    def _validate_file_size(self, file_bytes: bytes) -> None:
        if not file_bytes:
            raise OCRInputError("Uploaded file is empty")
        if len(file_bytes) > self.settings.max_file_bytes:
            raise OCRLimitError(
                f"File is {len(file_bytes)} bytes; limit is {self.settings.max_file_bytes} bytes"
            )

    def _validate_image_pixels(self, width: int, height: int) -> None:
        pixels = width * height
        if pixels > self.settings.max_image_pixels:
            raise OCRLimitError(
                f"Rendered image contains {pixels} pixels; limit is "
                f"{self.settings.max_image_pixels}"
            )

    def _cache_key(
        self,
        file_bytes: bytes,
        filename: str | None,
        include_layout: bool,
    ) -> str:
        configuration = {
            "pipeline": PIPELINE_VERSION,
            "input_type": (
                "pdf"
                if self._looks_like_pdf(file_bytes, filename)
                else "image"
            ),
            "include_layout": include_layout,
            "lang": self.settings.tesseract_lang,
            "oem": self.settings.tesseract_oem,
            "psm": self.settings.tesseract_psm,
            "timeout": self.settings.tesseract_timeout_seconds,
            "dpi": self.settings.render_dpi,
            "retry_dpi": self.settings.retry_dpi,
            "retry_confidence": self.settings.retry_confidence_threshold,
            "native_chars": self.settings.native_text_min_chars,
            "native_words": self.settings.native_text_min_words,
            "image_ratio": self.settings.image_dominant_ratio,
            "max_pixels": self.settings.max_image_pixels,
            "max_document_pixels": self.settings.max_document_rendered_pixels,
            "max_document_seconds": self.settings.max_document_seconds,
            "max_output_characters": self.settings.max_output_characters,
            "rapidocr": self.settings.enable_rapidocr,
            "rapidocr_dml": self.settings.rapidocr_use_directml,
            "rapidocr_required": self.settings.rapidocr_require_accelerator,
            "rapidocr_min_confidence": self.settings.rapidocr_min_confidence,
            "rapidocr_max_side": self.settings.rapidocr_max_side,
            "rapidocr_model_root": self.settings.rapidocr_model_root,
            "rapidocr_model_fingerprint": self._rapidocr_model_fingerprint,
            "rapidocr_isolate_process": self.settings.rapidocr_isolate_process,
            "rapidocr_inference_timeout_seconds": (
                self.settings.rapidocr_inference_timeout_seconds
            ),
            "rapidocr_recycle_after_calls": self.settings.rapidocr_recycle_after_calls,
            "dense_table_guard": self.settings.enable_dense_table_guard,
            "dense_table_layout_confidence": (
                self.settings.dense_table_layout_confidence
            ),
            "dense_table_min_area_ratio": self.settings.dense_table_min_area_ratio,
            "dense_table_tesseract_psm": self.settings.dense_table_tesseract_psm,
            "formula": self.settings.enable_formula_ocr,
            "layout": self.settings.enable_document_layout,
            "layout_type": self.settings.layout_model_type,
            "layout_model": self.settings.layout_model_path,
            "layout_confidence": self.settings.layout_confidence_threshold,
        }
        digest = sha256(file_bytes).hexdigest()
        encoded = json.dumps(configuration, sort_keys=True, separators=(",", ":"))
        return sha256(f"{digest}:{encoded}".encode()).hexdigest()

    def _estimate_result_bytes(self, result: OCRResult) -> int:
        size = 1_024 + len(result.text.encode("utf-8"))
        for _, page_text in result.pages or []:
            size += 128 + len(page_text.encode("utf-8"))
        for detail in result.page_details or []:
            size += 512 + len(detail.text.encode("utf-8"))
            size += sum(len(warning.encode("utf-8")) + 64 for warning in detail.warnings)
            for block in detail.blocks:
                size += 512
                size += len((block.text or "").encode("utf-8"))
                size += len((block.latex or "").encode("utf-8"))
                size += len(
                    json.dumps(
                        block.metadata,
                        ensure_ascii=False,
                        sort_keys=True,
                        default=str,
                    ).encode("utf-8")
                )
        return max(1, size)

    def _fingerprint_model_root(self, configured_root: str | None) -> str | None:
        if not configured_root:
            return None
        root = Path(configured_root).expanduser().resolve()
        if not root.is_dir():
            return f"missing:{root}"

        digest = sha256()
        for path in sorted(
            (candidate for candidate in root.rglob("*") if candidate.is_file()),
            key=lambda candidate: candidate.relative_to(root).as_posix(),
        ):
            relative = path.relative_to(root).as_posix()
            digest.update(relative.encode("utf-8"))
            try:
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
            except OSError as exc:
                raise OCRDependencyError(
                    f"Unable to fingerprint RapidOCR model file {path}: {exc}"
                ) from exc
        return digest.hexdigest()

    def _validate_result_output(self, result: OCRResult) -> None:
        characters = len(result.text) + 256
        for _, page_text in result.pages or []:
            characters += len(page_text) + 64
        for detail in result.page_details or []:
            characters += len(detail.text) + len(detail.method) + 256
            characters += sum(len(warning) + 32 for warning in detail.warnings)
            for block in detail.blocks:
                characters += 256
                characters += len(block.text or "") + len(block.latex or "")
                characters += len(block.kind) + len(block.source)
                characters += len(
                    json.dumps(
                        block.metadata,
                        ensure_ascii=False,
                        sort_keys=True,
                        default=str,
                    )
                )
        if characters > self.settings.max_output_characters:
            raise OCRLimitError(
                "OCR structured output character limit exceeded: "
                f"{characters} > {self.settings.max_output_characters}"
            )

    def _looks_like_pdf(self, file_bytes: bytes, filename: str | None) -> bool:
        if file_bytes.startswith(b"%PDF"):
            return True
        return bool(filename and filename.lower().endswith(".pdf"))
