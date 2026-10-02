from typing import List, Sequence

from src.ingestion.base import ArticleSource
from src.schemas.risk_signal import RawArticle
from src.utils.logging import get_logger

log = get_logger("ingestion.multi")


class MultiSource(ArticleSource):
    """Merge several sources (e.g. news + Twitter/X). One failing source never blocks the others."""

    def __init__(self, sources: Sequence[ArticleSource]):
        self.sources = list(sources)

    def fetch(self) -> List[RawArticle]:
        out, seen = [], set()
        for src in self.sources:
            try:
                for a in src.fetch():
                    if a.article_id not in seen:
                        seen.add(a.article_id); out.append(a)
            except Exception as e:
                log.warning("%s failed: %s", type(src).__name__, e)
        return sorted(out, key=lambda a: a.published_at)
