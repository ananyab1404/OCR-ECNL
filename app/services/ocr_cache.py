from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
from typing import Protocol, TypeVar

T = TypeVar("T")


class OCRCache(Protocol[T]):
    def get(self, key: str) -> T | None: ...

    def set(self, key: str, value: T, estimated_bytes: int) -> None: ...


@dataclass(slots=True)
class _CacheEntry[T]:
    value: T
    expires_at: float
    estimated_bytes: int


class MemoryTTLCache[T]:
    def __init__(
        self,
        *,
        max_bytes: int,
        ttl_seconds: int,
        max_entries: int = 512,
    ) -> None:
        self.max_bytes = max(0, max_bytes)
        self.ttl_seconds = max(0, ttl_seconds)
        self.max_entries = max(1, max_entries)
        self._entries: OrderedDict[str, _CacheEntry[T]] = OrderedDict()
        self._total_bytes = 0
        self._lock = RLock()

    @property
    def enabled(self) -> bool:
        return self.max_bytes > 0 and self.ttl_seconds > 0

    def get(self, key: str) -> T | None:
        if not self.enabled:
            return None

        now = time.monotonic()
        with self._lock:
            self._remove_expired(now)
            entry = self._entries.get(key)
            if entry is None:
                return None
            self._entries.move_to_end(key)
            return entry.value

    def set(self, key: str, value: T, estimated_bytes: int) -> None:
        if not self.enabled:
            return

        entry_size = max(1, estimated_bytes)
        if entry_size > self.max_bytes:
            return

        with self._lock:
            existing = self._entries.pop(key, None)
            if existing is not None:
                self._total_bytes -= existing.estimated_bytes

            while self._entries and (
                self._total_bytes + entry_size > self.max_bytes
                or len(self._entries) >= self.max_entries
            ):
                _, evicted = self._entries.popitem(last=False)
                self._total_bytes -= evicted.estimated_bytes

            self._entries[key] = _CacheEntry(
                value=value,
                expires_at=time.monotonic() + self.ttl_seconds,
                estimated_bytes=entry_size,
            )
            self._total_bytes += entry_size

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._total_bytes = 0

    def _remove_expired(self, now: float) -> None:
        expired = [key for key, entry in self._entries.items() if entry.expires_at <= now]
        for key in expired:
            entry = self._entries.pop(key)
            self._total_bytes -= entry.estimated_bytes
