from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock

from PIL import Image


class RapidLayoutBackendError(RuntimeError):
    """Raised when the optional layout backend cannot run safely."""


@dataclass(frozen=True, slots=True)
class LayoutRegion:
    label: str
    bbox: tuple[float, float, float, float]
    confidence: float


@dataclass(frozen=True, slots=True)
class RapidLayoutBackendResult:
    regions: list[LayoutRegion]
    provider: str
    engine_seconds: float | None = None


class RapidLayoutBackend:
    def __init__(
        self,
        *,
        model_type: str,
        model_path: str | None,
        confidence_threshold: float,
        use_directml: bool,
        require_accelerator: bool,
        engine_factory: Callable[..., object] | None = None,
    ) -> None:
        self.model_type = model_type
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.use_directml = use_directml
        self.require_accelerator = require_accelerator
        self._engine_factory = engine_factory
        self._engine: object | None = None
        self._failure: RapidLayoutBackendError | None = None
        self._provider = "uninitialized"
        self._initialization_lock = Lock()
        self._inference_lock = Lock()

    @property
    def provider(self) -> str:
        return self._provider

    def preload(self) -> None:
        self._get_engine()

    def analyze(self, image: Image.Image) -> RapidLayoutBackendResult:
        engine = self._get_engine()
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - rapid-layout depends on numpy
            raise RapidLayoutBackendError("rapid-layout requires numpy") from exc

        rgb = image.convert("RGB")
        bgr = np.asarray(rgb)[:, :, ::-1].copy()
        try:
            with self._inference_lock:
                output = engine(bgr)
        except Exception as exc:
            raise RapidLayoutBackendError(
                f"Document layout inference failed: {exc}"
            ) from exc

        raw_boxes = getattr(output, "boxes", None)
        raw_labels = getattr(output, "class_names", None)
        raw_scores = getattr(output, "scores", None)
        boxes = list(raw_boxes) if raw_boxes is not None else []
        labels = list(raw_labels) if raw_labels is not None else []
        scores = list(raw_scores) if raw_scores is not None else []
        regions = [
            LayoutRegion(
                label=str(label),
                bbox=tuple(float(value) for value in box),
                confidence=float(score) * 100.0,
            )
            for box, label, score in zip(boxes, labels, scores, strict=False)
            if len(box) == 4
        ]
        engine_seconds = getattr(output, "elapse", None)
        return RapidLayoutBackendResult(
            regions=regions,
            provider=self._provider,
            engine_seconds=(
                float(engine_seconds)
                if isinstance(engine_seconds, (int, float))
                else None
            ),
        )

    def _get_engine(self) -> object:
        if self._engine is not None:
            return self._engine
        if self._failure is not None:
            raise self._failure
        with self._initialization_lock:
            if self._engine is not None:
                return self._engine
            if self._failure is not None:
                raise self._failure

            factory = self._engine_factory
            if factory is None:
                try:
                    from rapid_layout import RapidLayout
                except ImportError as exc:
                    self._failure = RapidLayoutBackendError(
                        "rapid-layout is not installed"
                    )
                    raise self._failure from exc
                factory = RapidLayout

            engine_cfg = {
                "use_dml": self.use_directml,
                "intra_op_num_threads": 1,
                "inter_op_num_threads": 1,
            }
            kwargs: dict[str, object] = {
                "model_type": self.model_type,
                "engine_cfg": engine_cfg,
                "conf_thresh": self.confidence_threshold,
            }
            if self.model_path:
                kwargs["model_dir_or_path"] = self.model_path
            try:
                engine = factory(**kwargs)
                providers = list(engine.session.session.get_providers())
            except Exception as exc:
                self._failure = RapidLayoutBackendError(
                    f"Document layout initialization failed: {exc}"
                )
                raise self._failure from exc

            if not providers:
                self._failure = RapidLayoutBackendError(
                    "Document layout did not expose an inference provider"
                )
                raise self._failure
            if self.require_accelerator:
                expected = "DmlExecutionProvider" if self.use_directml else None
                if expected is None or providers[0] != expected:
                    self._failure = RapidLayoutBackendError(
                        f"Document layout accelerator gate failed; expected "
                        f"{expected} first, got {providers}"
                    )
                    raise self._failure

            self._provider = providers[0]
            self._engine = engine
            return engine
