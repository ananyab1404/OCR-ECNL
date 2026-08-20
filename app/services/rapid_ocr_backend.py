from __future__ import annotations

import math
import multiprocessing
from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing.connection import Connection
from threading import Lock
from typing import Any, Self

from PIL import Image


class RapidOCRBackendError(RuntimeError):
    """Raised when the optional accelerated OCR backend cannot be used safely."""


@dataclass(frozen=True, slots=True)
class RapidOCRLine:
    text: str
    confidence: float
    quadrilateral: list[list[float]]


@dataclass(frozen=True, slots=True)
class RapidOCRBackendResult:
    lines: list[RapidOCRLine]
    provider: str
    engine_seconds: float | None = None


def _rapidocr_worker_main(
    connection: Connection,
    settings: dict[str, object],
) -> None:
    """Run RapidOCR in a spawn-safe process with no API-layer imports."""
    try:
        backend = RapidOCRBackend(
            use_directml=bool(settings["use_directml"]),
            require_accelerator=bool(settings["require_accelerator"]),
            max_side=int(settings["max_side"]),
            model_root=(
                str(settings["model_root"])
                if settings.get("model_root") is not None
                else None
            ),
            isolate_process=False,
        )
        backend.preload()
        connection.send(("ready", backend.provider))
    except Exception as exc:  # noqa: BLE001 - serialize failures across process boundary
        try:
            connection.send(
                (
                    "startup_error",
                    f"{type(exc).__name__}: {exc}",
                )
            )
        except (BrokenPipeError, EOFError, OSError):
            pass
        connection.close()
        return

    try:
        while True:
            try:
                message = connection.recv()
            except (EOFError, OSError):
                return
            if not isinstance(message, tuple) or not message:
                connection.send(("protocol_error", "Malformed worker request"))
                return
            if message[0] == "close":
                return
            if len(message) != 5 or message[0] != "recognize":
                connection.send(("protocol_error", "Unknown worker request"))
                return

            _, request_id, size, mode, pixels = message
            try:
                if (
                    not isinstance(request_id, int)
                    or not isinstance(size, tuple)
                    or len(size) != 2
                    or mode != "RGB"
                    or not isinstance(pixels, bytes)
                ):
                    raise ValueError("Invalid image payload")
                image = Image.frombytes("RGB", size, pixels)
                result = backend.recognize(image)
                connection.send(("result", request_id, result))
            except Exception as exc:  # noqa: BLE001 - serialize worker failures
                connection.send(
                    (
                        "error",
                        request_id,
                        f"{type(exc).__name__}: {exc}",
                    )
                )
    finally:
        connection.close()


class RapidOCRBackend:
    """Lazy, provider-verified wrapper around RapidOCR.

    DirectML sessions normally register CPU as a secondary provider. The safety
    gate checks that DirectML is first so a missing GPU runtime cannot silently
    turn a production deployment into the much slower CPU configuration.
    """

    def __init__(
        self,
        *,
        use_directml: bool,
        require_accelerator: bool,
        max_side: int = 2_048,
        model_root: str | None = None,
        engine_factory: Callable[..., object] | None = None,
        isolate_process: bool = False,
        inference_timeout_seconds: float = 15.0,
        recycle_after_calls: int = 100,
        _worker_target: Callable[[Connection, dict[str, object]], None] | None = None,
    ) -> None:
        if (
            not math.isfinite(inference_timeout_seconds)
            or inference_timeout_seconds <= 0
        ):
            raise ValueError("inference_timeout_seconds must be finite and positive")
        if recycle_after_calls <= 0:
            raise ValueError("recycle_after_calls must be positive")
        if isolate_process and engine_factory is not None:
            raise ValueError(
                "engine_factory cannot cross a spawned process; use local mode"
            )
        self.use_directml = use_directml
        self.require_accelerator = require_accelerator
        self.max_side = max_side
        self.model_root = model_root
        self.isolate_process = isolate_process
        self.inference_timeout_seconds = inference_timeout_seconds
        self.recycle_after_calls = recycle_after_calls
        self._engine_factory = engine_factory
        self._engine: object | None = None
        self._failure: RapidOCRBackendError | None = None
        self._provider = "uninitialized"
        self._initialization_lock = Lock()
        self._inference_lock = Lock()
        self._worker_target = _worker_target or _rapidocr_worker_main
        self._worker_process: multiprocessing.Process | None = None
        self._worker_connection: Connection | None = None
        self._worker_calls = 0
        self._request_id = 0

    @property
    def provider(self) -> str:
        return self._provider

    def preload(self) -> None:
        if not self.isolate_process:
            self._get_engine()
            return
        with self._inference_lock:
            self._start_worker_locked()

    def recognize(self, image: Image.Image) -> RapidOCRBackendResult:
        if self.isolate_process:
            return self._recognize_isolated(image)

        engine = self._get_engine()
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - RapidOCR depends on numpy
            raise RapidOCRBackendError("RapidOCR requires numpy") from exc

        rgb = image.convert("RGB")
        # RapidOCR treats numpy arrays as OpenCV/BGR input.
        bgr = np.asarray(rgb)[:, :, ::-1].copy()
        try:
            # Serialize a single DirectML session. This bounds the roughly 1 GB
            # observed peak RSS and avoids competing command queues.
            with self._inference_lock:
                output = engine(bgr)
        except Exception as exc:
            raise RapidOCRBackendError(f"RapidOCR inference failed: {exc}") from exc

        texts = list(getattr(output, "txts", None) or ())
        scores = list(getattr(output, "scores", None) or ())
        raw_boxes = getattr(output, "boxes", None)
        boxes = raw_boxes.tolist() if raw_boxes is not None else []
        lines = [
            RapidOCRLine(
                text=str(text).strip(),
                confidence=float(score) * 100.0,
                quadrilateral=[
                    [float(point[0]), float(point[1])]
                    for point in box
                ],
            )
            for text, score, box in zip(texts, scores, boxes, strict=False)
            if str(text).strip()
        ]
        engine_seconds = getattr(output, "elapse", None)
        return RapidOCRBackendResult(
            lines=lines,
            provider=self._provider,
            engine_seconds=(
                float(engine_seconds)
                if isinstance(engine_seconds, (int, float))
                else None
            ),
        )

    def close(self) -> None:
        """Stop the isolated worker, if any. Safe to call more than once."""
        if not self.isolate_process:
            return
        with self._inference_lock:
            self._stop_worker_locked(graceful=True)

    def __enter__(self) -> Self:
        self.preload()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _recognize_isolated(self, image: Image.Image) -> RapidOCRBackendResult:
        rgb = image.convert("RGB")
        with self._inference_lock:
            connection = self._start_worker_locked()
            self._request_id += 1
            request_id = self._request_id
            try:
                connection.send(
                    (
                        "recognize",
                        request_id,
                        rgb.size,
                        "RGB",
                        rgb.tobytes(),
                    )
                )
            except (BrokenPipeError, EOFError, OSError) as exc:
                self._stop_worker_locked(graceful=False)
                raise RapidOCRBackendError(
                    f"RapidOCR worker request failed: {exc}"
                ) from exc

            response = self._receive_worker_message_locked(
                phase="inference",
                timeout=self.inference_timeout_seconds,
            )
            if (
                not isinstance(response, tuple)
                or len(response) < 2
                or response[0] not in {"result", "error"}
                or response[1] != request_id
            ):
                self._stop_worker_locked(graceful=False)
                raise RapidOCRBackendError(
                    f"RapidOCR worker protocol failure: {response!r}"
                )

            self._worker_calls += 1
            should_recycle = self._worker_calls >= self.recycle_after_calls
            if response[0] == "error":
                detail = str(response[2]) if len(response) >= 3 else "unknown error"
                if should_recycle:
                    self._stop_worker_locked(graceful=True)
                raise RapidOCRBackendError(
                    f"RapidOCR worker inference failed: {detail}"
                )

            if len(response) != 3 or not isinstance(
                response[2], RapidOCRBackendResult
            ):
                self._stop_worker_locked(graceful=False)
                raise RapidOCRBackendError(
                    f"RapidOCR worker returned an invalid result: {response!r}"
                )
            result = response[2]
            self._provider = result.provider
            if should_recycle:
                self._stop_worker_locked(graceful=True)
            return result

    def _start_worker_locked(self) -> Connection:
        process = self._worker_process
        connection = self._worker_connection
        if process is not None and connection is not None:
            try:
                if process.is_alive():
                    return connection
            except ValueError:
                pass
            self._stop_worker_locked(graceful=False)

        context = multiprocessing.get_context("spawn")
        parent_connection, child_connection = context.Pipe(duplex=True)
        settings: dict[str, object] = {
            "use_directml": self.use_directml,
            "require_accelerator": self.require_accelerator,
            "max_side": self.max_side,
            "model_root": self.model_root,
        }
        process = context.Process(
            target=self._worker_target,
            args=(child_connection, settings),
            name="rapidocr-worker",
            daemon=True,
        )
        try:
            process.start()
        except Exception as exc:
            parent_connection.close()
            child_connection.close()
            raise RapidOCRBackendError(
                f"RapidOCR worker failed to start: {exc}"
            ) from exc
        child_connection.close()
        self._worker_process = process
        self._worker_connection = parent_connection
        self._worker_calls = 0

        response = self._receive_worker_message_locked(
            phase="startup",
            # Windows spawn imports the interpreter before the worker can send
            # readiness. Keep a small startup floor while preserving the
            # configured (often larger) production timeout.
            timeout=max(3.0, self.inference_timeout_seconds),
        )
        if (
            not isinstance(response, tuple)
            or len(response) != 2
            or response[0] != "ready"
            or not isinstance(response[1], str)
            or not response[1]
        ):
            self._stop_worker_locked(graceful=False)
            if (
                isinstance(response, tuple)
                and len(response) == 2
                and response[0] == "startup_error"
            ):
                raise RapidOCRBackendError(
                    f"RapidOCR worker initialization failed: {response[1]}"
                )
            raise RapidOCRBackendError(
                f"RapidOCR worker startup protocol failure: {response!r}"
            )
        self._provider = response[1]
        return parent_connection

    def _receive_worker_message_locked(
        self,
        *,
        phase: str,
        timeout: float,
    ) -> Any:
        connection = self._worker_connection
        if connection is None:
            raise RapidOCRBackendError("RapidOCR worker is not connected")
        try:
            if not connection.poll(timeout):
                self._stop_worker_locked(graceful=False)
                raise RapidOCRBackendError(
                    f"RapidOCR worker {phase} timed out after {timeout:.3f}s"
                )
            return connection.recv()
        except RapidOCRBackendError:
            raise
        except (BrokenPipeError, EOFError, OSError) as exc:
            self._stop_worker_locked(graceful=False)
            raise RapidOCRBackendError(
                f"RapidOCR worker {phase} failed: {exc}"
            ) from exc

    def _stop_worker_locked(self, *, graceful: bool) -> None:
        process = self._worker_process
        connection = self._worker_connection
        self._worker_process = None
        self._worker_connection = None
        self._worker_calls = 0

        if graceful and connection is not None:
            try:
                connection.send(("close",))
            except (BrokenPipeError, EOFError, OSError):
                pass
        if connection is not None:
            connection.close()
        if process is None:
            return

        try:
            if graceful:
                process.join(timeout=1.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
            if process.is_alive() and hasattr(process, "kill"):
                process.kill()
                process.join(timeout=2.0)
        finally:
            try:
                process.close()
            except ValueError:
                pass

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
                    from rapidocr import RapidOCR
                except ImportError as exc:
                    self._failure = RapidOCRBackendError(
                        "RapidOCR is not installed; install the accelerated dependencies"
                    )
                    raise self._failure from exc
                factory = RapidOCR

            params: dict[str, object] = {
                "Global.log_level": "warning",
                "EngineConfig.onnxruntime.intra_op_num_threads": 1,
                "EngineConfig.onnxruntime.inter_op_num_threads": 1,
                "EngineConfig.onnxruntime.use_dml": self.use_directml,
                "Det.limit_type": "max",
                "Det.limit_side_len": self.max_side,
            }
            if self.model_root:
                params["Global.model_root_dir"] = self.model_root

            try:
                engine = factory(params=params)
            except Exception as exc:
                self._failure = RapidOCRBackendError(
                    f"RapidOCR initialization failed: {exc}"
                )
                raise self._failure from exc

            providers = self._session_providers(engine)
            primary_providers = {
                name: values[0]
                for name, values in providers.items()
                if values
            }
            if len(primary_providers) != 3:
                self._failure = RapidOCRBackendError(
                    f"RapidOCR did not expose all inference sessions: {providers}"
                )
                raise self._failure

            if self.require_accelerator:
                expected = "DmlExecutionProvider" if self.use_directml else None
                if expected is None:
                    self._failure = RapidOCRBackendError(
                        "An accelerator is required but no supported provider was configured"
                    )
                    raise self._failure
                incorrect = {
                    name: values
                    for name, values in providers.items()
                    if not values or values[0] != expected
                }
                if incorrect:
                    self._failure = RapidOCRBackendError(
                        f"RapidOCR accelerator gate failed; expected {expected} first: "
                        f"{incorrect}"
                    )
                    raise self._failure

            unique_primary = sorted(set(primary_providers.values()))
            self._provider = ",".join(unique_primary)
            self._engine = engine
            return engine

    def _session_providers(self, engine: object) -> dict[str, list[str]]:
        providers: dict[str, list[str]] = {}
        for name, attribute in (
            ("det", "text_det"),
            ("cls", "text_cls"),
            ("rec", "text_rec"),
        ):
            try:
                session = getattr(engine, attribute).session.session
                providers[name] = list(session.get_providers())
            except (AttributeError, TypeError):
                providers[name] = []
        return providers
