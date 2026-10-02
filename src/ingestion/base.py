import time
from abc import ABC, abstractmethod
from typing import Iterable, Iterator, List

from pydantic import ValidationError

from src.schemas.risk_signal import RawArticle
from src.utils.logging import get_logger

log = get_logger("ingestion")


class RateLimiter:
    """Token-bucket limiter; clock/sleep injectable for deterministic tests."""

    def __init__(self, rate_per_sec: float, burst: int = 1, clock=time.monotonic, sleep=time.sleep):
        if rate_per_sec <= 0 or burst < 1:
            raise ValueError("rate_per_sec>0 and burst>=1 required")
        self.rate, self.capacity = rate_per_sec, float(burst)
        self._tokens, self._clock, self._sleep = float(burst), clock, sleep
        self._last = clock()

    def acquire(self) -> float:
        now = self._clock()
        self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate)
        self._last, waited = now, 0.0
        if self._tokens < 1.0:
            waited = (1.0 - self._tokens) / self.rate
            self._sleep(waited)
            self._last = self._clock()
            self._tokens = 1.0
        self._tokens -= 1.0
        return waited


def validate_records(records: Iterable[dict]) -> List[RawArticle]:
    """Schema-validate; invalid rows are dropped (logged), duplicates by id removed."""
    out, seen = [], set()
    for rec in records:
        try:
            art = RawArticle(**rec)
        except (ValidationError, TypeError) as e:
            log.warning("dropping invalid record: %s", str(e).splitlines()[0])
            continue
        if art.article_id in seen:
            continue
        seen.add(art.article_id)
        out.append(art)
    return sorted(out, key=lambda a: a.published_at)


class ArticleSource(ABC):
    @abstractmethod
    def fetch(self) -> List[RawArticle]: ...

    def stream(self) -> Iterator[RawArticle]:
        yield from self.fetch()
