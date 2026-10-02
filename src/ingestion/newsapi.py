import hashlib
import os
from typing import List, Optional

import requests

from src.ingestion.base import ArticleSource, RateLimiter, validate_records
from src.schemas.risk_signal import RawArticle
from src.utils.cache import TTLCache
from src.utils.errors import IngestionError
from src.utils.logging import get_logger

log = get_logger("newsapi")
URL = "https://newsapi.org/v2/everything"


class NewsAPISource(ArticleSource):
    def __init__(self, query: str, api_key: Optional[str] = None, page_size: int = 50,
                 limiter: Optional[RateLimiter] = None, cache: Optional[TTLCache] = None, session=None):
        self.query, self.page_size = query, page_size
        self.api_key = api_key or os.getenv("NEWSAPI_KEY")
        self.limiter = limiter or RateLimiter(rate_per_sec=1.0)
        self.cache = cache or TTLCache(300)
        self.session = session or requests

    @staticmethod
    def parse(payload: dict) -> List[dict]:
        recs = []
        for a in payload.get("articles", []):
            url = a.get("url") or ""
            basis = url or a.get("title") or ""
            recs.append({
                "article_id": hashlib.sha1(basis.encode()).hexdigest()[:16],
                "published_at": a.get("publishedAt"),
                "source": (a.get("source") or {}).get("name") or "newsapi",
                "title": a.get("title") or "", "body": a.get("description") or "", "url": url or None,
            })
        return recs

    def fetch(self) -> List[RawArticle]:
        if not self.api_key:
            raise IngestionError("NEWSAPI_KEY not set")
        k = TTLCache.key("newsapi", self.query, self.page_size)
        payload = self.cache.get(k)
        if payload is None:
            self.limiter.acquire()
            try:
                r = self.session.get(URL, params={"q": self.query, "pageSize": self.page_size, "language": "en",
                                                  "sortBy": "publishedAt", "apiKey": self.api_key}, timeout=10)
                r.raise_for_status()
                payload = r.json()
                self.cache.set(k, payload)
            except Exception as e:  # network down -> stale cache else fail loudly
                payload = self.cache.get(k, allow_stale=True)
                if payload is None:
                    raise IngestionError(f"NewsAPI failed: {e}") from e
                log.warning("NewsAPI unreachable; serving stale cache")
        return validate_records(self.parse(payload))
