"""Tiny TTL cache + disk fallback so live feeds degrade gracefully offline."""
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Optional


class TTLCache:
    def __init__(self, ttl_seconds: float = 300, directory: Optional[Path] = None, clock=time.time):
        self.ttl, self._clock = ttl_seconds, clock
        self.dir = Path(directory) if directory else None
        self._mem: dict = {}
        if self.dir:
            self.dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(*parts: Any) -> str:
        return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:24]

    def get(self, k: str, allow_stale: bool = False) -> Optional[Any]:
        item = self._mem.get(k)
        if item is None and self.dir and (self.dir / f"{k}.json").exists():
            blob = json.loads((self.dir / f"{k}.json").read_text())
            item = (blob["t"], blob["v"])
            self._mem[k] = item
        if item is None:
            return None
        t, v = item
        return v if allow_stale or self._clock() - t <= self.ttl else None

    def set(self, k: str, v: Any) -> None:
        now = self._clock()
        self._mem[k] = (now, v)
        if self.dir:
            (self.dir / f"{k}.json").write_text(json.dumps({"t": now, "v": v}, default=str))
