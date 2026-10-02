import hashlib
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import List, Optional

import feedparser

from src.ingestion.base import ArticleSource, RateLimiter, validate_records
from src.schemas.risk_signal import RawArticle
from src.utils.errors import IngestionError
from src.utils.logging import get_logger

log = get_logger("gdelt")
DEFAULT_URL = "https://api.gdeltproject.org/api/v2/doc/doc?query={q}&mode=artlist&format=rss&maxrecords=50"


class GDELTRSSSource(ArticleSource):
    def __init__(self, query: str = "market OR economy", url_template: str = DEFAULT_URL,
                 limiter: Optional[RateLimiter] = None, parser=feedparser.parse):
        self.url = url_template.format(q=query.replace(" ", "%20"))
        self.limiter, self.parser = limiter or RateLimiter(0.2), parser

    @staticmethod
    def parse(feed) -> List[dict]:
        recs = []
        for e in getattr(feed, "entries", []):
            raw = e.get("published") or ""
            try:
                ts = parsedate_to_datetime(raw)
            except (TypeError, ValueError):
                ts = datetime.now(timezone.utc)
            link = e.get("link", "")
            recs.append({"article_id": hashlib.sha1((link or e.get("title", "")).encode()).hexdigest()[:16],
                         "published_at": ts, "source": "GDELT", "title": e.get("title", ""),
                         "body": e.get("summary", ""), "url": link or None})
        return recs

    def fetch(self) -> List[RawArticle]:
        self.limiter.acquire()
        feed = self.parser(self.url)
        if getattr(feed, "bozo", 0) and not getattr(feed, "entries", []):
            raise IngestionError("GDELT feed unavailable")
        return validate_records(self.parse(feed))
