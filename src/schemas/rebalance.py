"""Strict contracts for Module A: Tactical Index Rebalancer."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


class StockConstituent(_Strict):
    """Constituent in the mock stock index."""
    ticker: str = Field(min_length=1)
    name: str = Field(min_length=1)
    sector: str = Field(min_length=1)
    base_weight: float = Field(gt=0.0, le=1.0)
    current_price: float = Field(default=100.0, gt=0.0)


IndexConstituent = StockConstituent


class RebalanceStep(_Strict):
    """Audit record for a single rebalancing event."""
    step_id: int
    timestamp: datetime
    trigger_signal_id: Optional[str] = None
    trigger_ticker: Optional[str] = None
    trigger_sentiment: float = 0.0
    trigger_impact: float = 1.0
    headline: str = ""
    weights: Dict[str, float]
    weight_changes: Dict[str, float]
    one_way_turnover: float = Field(ge=0.0)
    transaction_cost: float = Field(ge=0.0)
    index_nav: float = Field(gt=0.0)
    benchmark_nav: float = Field(gt=0.0)
    excess_return_bps: float

    @field_validator("timestamp")
    @classmethod
    def _tz(cls, v: datetime) -> datetime:
        return _utc(v)


class RebalanceSummary(_Strict):
    """Summary of dynamic rebalancing performance."""
    total_steps: int
    initial_nav: float = 1000.0
    final_index_nav: float
    final_benchmark_nav: float
    cumulative_return_pct: float
    benchmark_return_pct: float
    excess_return_pct: float
    annualized_sharpe: float
    max_drawdown_pct: float
    total_turnover: float
    total_cost_drag_bps: float
    current_weights: Dict[str, float]
