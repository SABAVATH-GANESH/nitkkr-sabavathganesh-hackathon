"""NLP risk engine (Step 3): batched sentiment + event classification + impact scoring.

Offline-first: FinBERT / zero-shot are loaded with local_files_only=True. If weights or the
`transformers` package are unavailable, deterministic lexicon / keyword fallbacks are used so
the pipeline never fails in a demo environment.
"""
from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Sequence, Tuple

from src.schemas.risk_signal import EventType, RawArticle, RiskSignal
from src.utils.logging import get_logger

log = get_logger("nlp_engine")

# ---- Impact formula parameters -------------------------------------------------------------
SEVERITY: Dict[EventType, float] = {
    EventType.GEOPOLITICAL: 0.90, EventType.CREDIT_EVENT: 0.85, EventType.CYBER_OPS: 0.80, EventType.MACRO: 0.80,
    EventType.REGULATORY: 0.75, EventType.LEGAL: 0.70, EventType.SUPPLY_CHAIN: 0.70,
    EventType.EARNINGS: 0.60, EventType.MA: 0.55, EventType.PRODUCT_LAUNCH: 0.25, EventType.NEUTRAL: 0.10,
}
SOURCE_CREDIBILITY: Dict[str, float] = {
    "reuters": 1.0, "bloomberg": 1.0, "economic times": 0.9, "mint": 0.9, "gdelt": 0.7,
    "twitter": 0.5, "x": 0.5, "reddit": 0.4,
}
DEFAULT_CREDIBILITY = 0.7


def compute_impact(sentiment: float, sentiment_conf: float, event: EventType,
                   event_conf: float, credibility: float = DEFAULT_CREDIBILITY) -> float:
    """Impact in [1, 10].

    raw    = 0.5*|s|*c_s + 0.5*severity(event)*c_e         (magnitude of tone + event gravity)
    impact = 1 + 9 * clip(raw * (0.6 + 0.4*credibility), 0, 1)
    """
    raw = 0.5 * abs(sentiment) * sentiment_conf + 0.5 * SEVERITY[event] * event_conf
    raw *= 0.6 + 0.4 * min(max(credibility, 0.0), 1.0)
    return 1.0 + 9.0 * min(max(raw, 0.0), 1.0)


def source_credibility(source: str) -> float:
    return SOURCE_CREDIBILITY.get(source.strip().lower(), DEFAULT_CREDIBILITY)


# ---- Sentiment -----------------------------------------------------------------------------
_POS = {"beat", "beats", "surge", "surges", "soar", "soars", "growth", "record", "profit", "upgrade",
        "upgraded", "rally", "strong", "gain", "gains", "approval", "approved", "expands", "wins",
        "robust", "rebound", "boost", "outperform", "raises", "easing", "launches", "unveils", "rolls"}
_NEG = {"miss", "misses", "plunge", "plunges", "slump", "slumps", "fraud", "probe", "downgrade",
        "downgraded", "crash", "loss", "losses", "default", "ban", "fine", "fined", "breach", "hack",
        "strike", "sanctions", "war", "recall", "lawsuit", "layoffs", "shutdown", "halt", "halts",
        "recession", "inflation", "cuts", "warning", "warns", "disruption", "attack", "tensions", "missed", "bankruptcy", "insolvency", "slumps", "escalates", "fears"}
_NEGATORS = {"no", "not", "never", "without", "fails", "denies"}
_TOKEN = re.compile(r"[a-z']+")


class LexiconSentiment:
    """Deterministic fallback. Returns (score in (-1,1), confidence in [0.3, 1))."""

    def score(self, text: str) -> Tuple[float, float]:
        toks = _TOKEN.findall(text.lower())
        pos = neg = 0.0
        for i, t in enumerate(toks):
            sign = -1.0 if _NEGATORS & set(toks[max(0, i - 2):i]) else 1.0
            if t in _POS:
                pos, neg = (pos + 1, neg) if sign > 0 else (pos, neg + 1)
            elif t in _NEG:
                pos, neg = (pos, neg + 1) if sign > 0 else (pos + 1, neg)
        n = pos + neg
        if n == 0:
            return 0.0, 0.3
        return (pos - neg) / (n + 1.0), 0.5 + 0.5 * n / (n + 1.0)

    def batch(self, texts: Sequence[str]) -> List[Tuple[float, float]]:
        return [self.score(t) for t in texts]


class FinBERTSentiment:
    """ProsusAI/finbert; score = P(pos) - P(neg), confidence = max prob. Offline weights only."""

    def __init__(self, model_name: str = "ProsusAI/finbert", device: int = -1):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline  # type: ignore
        tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        mdl = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=True)
        self._pipe = pipeline("text-classification", model=mdl, tokenizer=tok, top_k=None,
                              truncation=True, max_length=256, device=device)

    def batch(self, texts: Sequence[str]) -> List[Tuple[float, float]]:
        out = []
        for res in self._pipe(list(texts)):
            p = {r["label"].lower(): r["score"] for r in res}
            out.append((p.get("positive", 0.0) - p.get("negative", 0.0), max(p.values())))
        return out


# ---- Event classification ------------------------------------------------------------------
_KEYWORDS: Dict[EventType, Tuple[str, ...]] = {
    EventType.EARNINGS: ("earnings", "quarterly", "profit", "revenue", "guidance", "eps", "results"),
    EventType.REGULATORY: ("sebi", "rbi", "regulator", "regulation", "compliance", "ban", "fine", "probe", "license"),
    EventType.GEOPOLITICAL: ("war", "sanctions", "border", "conflict", "tensions", "missile", "embargo", "election"),
    EventType.MACRO: ("inflation", "gdp", "interest rate", "repo", "gdp", "recession", "currency", "rupee", "fiscal", "rate hike"),
    EventType.CYBER_OPS: ("cyber", "hack", "breach", "ransomware", "outage", "data leak", "shutdown"),
    EventType.CREDIT_EVENT: ("downgrade", "default", "bankruptcy", "insolvency", "debt restructuring", "missed payment", "credit rating", "npa"),
    EventType.PRODUCT_LAUNCH: ("launch", "unveil", "introduce", "new product", "roll out", "rolls out"),
    EventType.LEGAL: ("lawsuit", "court", "fraud", "litigation", "verdict", "settlement", "tribunal"),
    EventType.MA: ("acquisition", "acquire", "merger", "takeover", "stake", "buyout", "demerger"),
    EventType.SUPPLY_CHAIN: ("supply chain", "shipment", "logistics", "shortage", "port", "strike", "factory", "recall"),
}
_PATTERNS = {k: re.compile(r"\b" + re.escape(k) + r"(?:s|d|ed|es)?\b") for kws in _KEYWORDS.values() for k in kws}
_ZS_LABELS = {
    EventType.GEOPOLITICAL: "geopolitical conflict or sanctions", EventType.MACRO: "macroeconomic policy or data",
    EventType.CREDIT_EVENT: "credit event, default or rating downgrade", EventType.MA: "merger or acquisition",
    EventType.PRODUCT_LAUNCH: "product launch", EventType.REGULATORY: "regulatory action",
    EventType.CYBER_OPS: "cyber attack or operational outage", EventType.SUPPLY_CHAIN: "supply chain disruption",
    EventType.EARNINGS: "company earnings results", EventType.LEGAL: "lawsuit or legal ruling"}


class KeywordEventClassifier:
    def classify(self, text: str) -> Tuple[EventType, float]:
        low = text.lower()
        hits = {e: sum(1 for k in kws if _PATTERNS[k].search(low)) for e, kws in _KEYWORDS.items()}
        total = sum(hits.values())
        if total == 0:
            return EventType.NEUTRAL, 0.5
        best = max(sorted(hits, key=lambda e: e.value), key=lambda e: hits[e])  # deterministic tie-break
        h = hits[best]
        return best, (h / total) * (0.5 + 0.5 * h / (h + 1.0))

    def batch(self, texts: Sequence[str]) -> List[Tuple[EventType, float]]:
        return [self.classify(t) for t in texts]


class ZeroShotEventClassifier:
    def __init__(self, model_name: str = "facebook/bart-large-mnli", device: int = -1):
        from transformers import pipeline  # type: ignore
        self._pipe = pipeline("zero-shot-classification", model=model_name, device=device,
                              model_kwargs={"local_files_only": True})
        self._rev = {v: k for k, v in _ZS_LABELS.items()}

    def batch(self, texts: Sequence[str]) -> List[Tuple[EventType, float]]:
        res = self._pipe(list(texts), candidate_labels=list(self._rev))
        res = res if isinstance(res, list) else [res]
        return [(self._rev[r["labels"][0]], float(r["scores"][0])) for r in res]


# ---- Orchestrator --------------------------------------------------------------------------
def _try(factory, name: str):
    try:
        return factory()
    except Exception as e:  # missing package, missing offline weights, etc.
        log.warning("%s unavailable (%s); using deterministic fallback", name, type(e).__name__)
        return None


class RiskEngine:
    def __init__(self, use_transformers: Optional[bool] = None, batch_size: int = 16):
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.batch_size = batch_size
        want = use_transformers is not False
        self.sentiment = (_try(FinBERTSentiment, "FinBERT") if want else None) or LexiconSentiment()
        self.events = (_try(ZeroShotEventClassifier, "zero-shot") if want else None) or KeywordEventClassifier()
        self.backend = f"{type(self.sentiment).__name__}+{type(self.events).__name__}"

    def process(self, articles: Sequence[RawArticle]) -> List[RiskSignal]:
        signals: List[RiskSignal] = []
        for i in range(0, len(articles), self.batch_size):
            chunk = articles[i:i + self.batch_size]
            texts = [a.text for a in chunk]
            sent, evt = self.sentiment.batch(texts), self.events.batch(texts)
            for a, (s, sc), (e, ec) in zip(chunk, sent, evt):
                s, sc, ec = max(-1.0, min(1.0, s)), max(0.0, min(1.0, sc)), max(0.0, min(1.0, ec))
                signals.append(RiskSignal(
                    article_id=a.article_id, published_at=a.published_at, source=a.source,
                    tickers=a.tickers, sectors=a.sectors, sentiment=round(s, 4),
                    sentiment_confidence=round(sc, 4), event_type=e, event_confidence=round(ec, 4),
                    impact=round(compute_impact(s, sc, e, ec, source_credibility(a.source)), 3),
                    headline=a.title))
        return signals
