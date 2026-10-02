from pathlib import Path
from typing import List

import pandas as pd

from src.ingestion.base import ArticleSource, validate_records
from src.schemas.risk_signal import RawArticle
from src.utils.errors import IngestionError


def _split(v) -> list:
    return [] if pd.isna(v) or not str(v).strip() else [s.strip() for s in str(v).split("|")]


class CSVReplaySource(ArticleSource):
    """Offline synthetic replay. Columns: article_id,published_at,source,title,body,tickers,sectors."""

    def __init__(self, path):
        self.path = Path(path)

    def fetch(self) -> List[RawArticle]:
        if not self.path.exists():
            raise IngestionError(f"replay file not found: {self.path}")
        df = pd.read_csv(self.path)
        missing = {"article_id", "published_at", "source", "title"} - set(df.columns)
        if missing:
            raise IngestionError(f"missing columns: {sorted(missing)}")
        recs = []
        for r in df.to_dict("records"):
            ts = pd.to_datetime(r["published_at"], utc=True, errors="coerce")
            recs.append({
                "article_id": str(r["article_id"]),
                "published_at": ts.to_pydatetime() if pd.notna(ts) else None,
                "source": r["source"], "title": r["title"],
                "body": "" if pd.isna(r.get("body")) else r.get("body"),
                "tickers": _split(r.get("tickers")), "sectors": _split(r.get("sectors")),
            })
        return validate_records(recs)
