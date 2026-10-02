"""Strict inter-module contracts (Step 1). Every module boundary uses these models."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class EventType(str, Enum):
    GEOPOLITICAL = "geopolitical"
    MACRO = "macroeconomic"
    CREDIT_EVENT = "credit_event"
    MA = "merger_acquisition"
    PRODUCT_LAUNCH = "product_launch"
    REGULATORY = "regulatory"
    CYBER_OPS = "cyber_operational"
    SUPPLY_CHAIN = "supply_chain"
    EARNINGS = "earnings"
    LEGAL = "legal"
    NEUTRAL = "other"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


class RawArticle(_Strict):
    article_id: str = Field(min_length=1)
    published_at: datetime
    source: str = Field(min_length=1)
    title: str = Field(min_length=1)
    body: str = ""
    tickers: List[str] = Field(default_factory=list)
    sectors: List[str] = Field(default_factory=list)
    url: Optional[str] = None

    @field_validator("title", "body")
    @classmethod
    def _clean(cls, v: str) -> str:
        return " ".join(v.split())

    @field_validator("tickers")
    @classmethod
    def _upper(cls, v: List[str]) -> List[str]:
        return sorted({t.strip().upper() for t in v if t.strip()})

    @field_validator("published_at")
    @classmethod
    def _tz(cls, v: datetime) -> datetime:
        return _utc(v)

    @property
    def text(self) -> str:
        return f"{self.title}. {self.body}".strip()


class RiskSignal(_Strict):
    article_id: str
    published_at: datetime
    source: str
    tickers: List[str] = Field(default_factory=list)
    sectors: List[str] = Field(default_factory=list)
    sentiment: float = Field(ge=-1.0, le=1.0)
    sentiment_confidence: float = Field(ge=0.0, le=1.0)
    event_type: EventType
    event_confidence: float = Field(ge=0.0, le=1.0)
    impact: float = Field(ge=1.0, le=10.0)
    headline: str = ""

    @field_validator("published_at")
    @classmethod
    def _tz(cls, v: datetime) -> datetime:
        return _utc(v)


class AssetType(str, Enum):
    LOAN = "loan"
    BOND = "bond"
    DERIVATIVE = "derivative"
    EQUITY = "equity"


class Asset(_Strict):
    """One wholesale-banking exposure. Money in base currency (USD)."""
    asset_id: str = Field(min_length=1)
    asset_type: AssetType
    counterparty: str
    ticker: Optional[str] = None
    sector: str
    notional: float = Field(gt=0)
    market_value: float                       # derivatives may be negative MTM
    duration: float = Field(default=0.0, ge=0)
    convexity: float = Field(default=0.0, ge=0)
    pd: float = Field(default=0.0, ge=0, le=1)    # 1y probability of default
    lgd: float = Field(default=0.45, ge=0, le=1)
    delta: float = 0.0                        # equity delta (derivatives)
    dv01: float = 0.0                         # value change per +1bp rate move (derivatives)

    @field_validator("ticker")
    @classmethod
    def _upper(cls, v: Optional[str]) -> Optional[str]:
        return v.strip().upper() if v and v.strip() else None

    @model_validator(mode="after")
    def _check(self) -> "Asset":
        if self.asset_type in (AssetType.LOAN, AssetType.BOND) and self.market_value <= 0:
            raise ValueError("loans/bonds need positive market_value")
        if self.asset_type == AssetType.BOND and self.duration <= 0:
            raise ValueError("bond requires duration > 0")
        return self


class PortfolioState(_Strict):
    assets: List[Asset] = Field(min_length=1)
    base_currency: str = "USD"

    @model_validator(mode="after")
    def _uniq(self) -> "PortfolioState":
        if len({a.asset_id for a in self.assets}) != len(self.assets):
            raise ValueError("duplicate asset_id in portfolio")
        return self

    @property
    def total_value(self) -> float:
        return sum(a.market_value for a in self.assets)


class StressScenario(_Strict):
    """Shocks at the *event level*; per-asset sector beta / exposure is applied downstream."""
    event_type: EventType
    impact: float = Field(ge=1.0, le=10.0)
    equity_shock: float = Field(ge=-1.0, le=1.0)      # e.g. -0.10 = -10%
    rate_shock_bps: float                              # e.g. +200 = +2%
    spread_shock_bps: float = Field(ge=0)
    pd_multiplier: float = Field(ge=1.0)
    tickers: List[str] = Field(default_factory=list)
    sectors: List[str] = Field(default_factory=list)
    trigger_article_id: Optional[str] = None
    headline: str = ""
    triggered_at: Optional[datetime] = None


class AssetStress(_Strict):
    asset_id: str
    asset_type: AssetType
    counterparty: str
    value_before: float
    value_after: float
    pnl: float


class StressResult(_Strict):
    label: str
    scenario: Optional[StressScenario] = None
    trigger_ids: List[str] = Field(default_factory=list)
    value_before: float
    value_after: float
    pnl: float
    pnl_pct: float
    pnl_by_asset_type: Dict[str, float]
    by_asset: List[AssetStress]
