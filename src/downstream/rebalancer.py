"""Module A: Tactical, High-Frequency Stock Index Rebalancer.

Dynamically tilts mock stock index constituent weights based on real-time sentiment signals
from the unified AI/NLP Risk Engine.
- Positive Sentiment: increases constituent weight.
- Negative Sentiment: decreases constituent weight.
- Implements exponential sentiment decay, long-only boundaries (min/max weight constraints),
  turnover calculation, transaction cost drag, and benchmark comparison (Static Equal Weight).
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

from src.schemas.rebalance import RebalanceStep, RebalanceSummary, StockConstituent
from src.schemas.risk_signal import RiskSignal
from src.utils.logging import get_logger

log = get_logger("rebalancer")

# 15 Bellwether Stocks across Global & Regional Markets (Matching Portfolio & News Tickers)
DEFAULT_CONSTITUENTS: List[Dict[str, str]] = [
    {"ticker": "AAPL", "name": "Apple Inc.", "sector": "it"},
    {"ticker": "MSFT", "name": "Microsoft Corp.", "sector": "it"},
    {"ticker": "NVDA", "name": "Nvidia Corp.", "sector": "it"},
    {"ticker": "JPM", "name": "JPMorgan Chase & Co.", "sector": "banking"},
    {"ticker": "HDFCBANK", "name": "HDFC Bank Ltd.", "sector": "banking"},
    {"ticker": "XOM", "name": "Exxon Mobil Corp.", "sector": "energy"},
    {"ticker": "RELIANCE", "name": "Reliance Industries", "sector": "energy"},
    {"ticker": "BA", "name": "Boeing Co.", "sector": "industrials"},
    {"ticker": "TATAMOTORS", "name": "Tata Motors Ltd.", "sector": "auto"},
    {"ticker": "PFE", "name": "Pfizer Inc.", "sector": "pharma"},
    {"ticker": "WMT", "name": "Walmart Inc.", "sector": "retail"},
    {"ticker": "TCS", "name": "Tata Consultancy Services", "sector": "it"},
    {"ticker": "GOOGL", "name": "Alphabet Inc.", "sector": "it"},
    {"ticker": "AMZN", "name": "Amazon.com Inc.", "sector": "retail"},
    {"ticker": "TSLA", "name": "Tesla Inc.", "sector": "auto"},
]


class TacticalIndexRebalancer:
    """Tactical Index Rebalancing Engine subscribing to NLP sentiment signals."""

    def __init__(
        self,
        constituents: Optional[List[StockConstituent]] = None,
        tilt_sensitivity: float = 0.80,
        min_weight: float = 0.01,
        max_weight: float = 0.22,
        decay_factor: float = 0.65,
        fee_bps: float = 5.0,
        initial_nav: float = 1000.0,
    ):
        if not 0.0 < min_weight < max_weight <= 1.0:
            raise ValueError("min_weight must be strictly less than max_weight, bounded by (0, 1]")
        if tilt_sensitivity <= 0:
            raise ValueError("tilt_sensitivity must be > 0")

        self.tilt_sensitivity = tilt_sensitivity
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.decay_factor = decay_factor
        self.fee_bps = fee_bps
        self.initial_nav = initial_nav

        if constituents is None:
            n = len(DEFAULT_CONSTITUENTS)
            base_w = round(1.0 / n, 6)
            self.constituents: Dict[str, StockConstituent] = {
                c["ticker"]: StockConstituent(
                    ticker=c["ticker"],
                    name=c["name"],
                    sector=c["sector"],
                    base_weight=base_w,
                    current_price=100.0,
                )
                for c in DEFAULT_CONSTITUENTS
            }
        else:
            self.constituents = {c.ticker: c for c in constituents}

        # Initialize portfolio state
        n = len(self.constituents)
        equal_w = 1.0 / n
        self.current_weights: Dict[str, float] = {t: equal_w for t in self.constituents}
        self.base_weights: Dict[str, float] = {t: self.constituents[t].base_weight for t in self.constituents}
        self.rolling_sentiment: Dict[str, float] = {t: 0.0 for t in self.constituents}
        self.current_prices: Dict[str, float] = {t: self.constituents[t].current_price for t in self.constituents}

        self.index_nav = initial_nav
        self.benchmark_nav = initial_nav
        self.total_turnover = 0.0
        self.total_cost_drag = 0.0
        self.steps: List[RebalanceStep] = []
        self._returns_history: List[float] = []

    def reset(self) -> None:
        """Reset rebalancer to initial state."""
        n = len(self.constituents)
        equal_w = 1.0 / n
        self.current_weights = {t: equal_w for t in self.constituents}
        self.rolling_sentiment = {t: 0.0 for t in self.constituents}
        self.current_prices = {t: self.constituents[t].current_price for t in self.constituents}
        self.index_nav = self.initial_nav
        self.benchmark_nav = self.initial_nav
        self.total_turnover = 0.0
        self.total_cost_drag = 0.0
        self.steps = []
        self._returns_history = []

    def _normalize_weights(self, raw_weights: Dict[str, float]) -> Dict[str, float]:
        """Project weights into [min_weight, max_weight] and enforce sum = 1.0."""
        tickers = list(self.constituents.keys())
        w = {t: max(self.min_weight, min(self.max_weight, raw_weights[t])) for t in tickers}
        # Iterative proportional fitting for box constraints
        for _ in range(10):
            total = sum(w.values())
            if abs(total - 1.0) < 1e-6:
                break
            w = {t: w[t] / total for t in tickers}
            w = {t: max(self.min_weight, min(self.max_weight, w[t])) for t in tickers}

        total = sum(w.values())
        return {t: round(w[t] / total, 6) for t in tickers}

    def process_signal(self, signal: RiskSignal) -> Optional[RebalanceStep]:
        """Process an incoming RiskSignal, update sentiment state, and rebalance index weights."""
        relevant_tickers = [t for t in signal.tickers if t in self.constituents]
        if not relevant_tickers and signal.sectors:
            # Check sector spillover
            sec_lower = [s.lower() for s in signal.sectors]
            relevant_tickers = [
                t for t, c in self.constituents.items() if c.sector.lower() in sec_lower
            ]

        if not relevant_tickers:
            return None

        # Effective sentiment incorporating confidence and impact
        s_eff = signal.sentiment * (0.6 + 0.4 * signal.sentiment_confidence)
        # Decay previous sentiments slightly
        for t in self.constituents:
            self.rolling_sentiment[t] *= self.decay_factor

        # Apply new sentiment to impacted tickers
        primary_ticker = relevant_tickers[0]
        for t in relevant_tickers:
            self.rolling_sentiment[t] += s_eff

        # Compute new tilted weights
        raw_weights: Dict[str, float] = {}
        for t, base_w in self.base_weights.items():
            # Sentiment tilt formula: tilt sensitivity * tanh(rolling_sentiment)
            tilt = self.tilt_sensitivity * math.tanh(1.8 * self.rolling_sentiment[t])
            raw_weights[t] = base_w * (1.0 + tilt)

        new_weights = self._normalize_weights(raw_weights)

        # Calculate one-way turnover
        turnover = 0.5 * sum(abs(new_weights[t] - self.current_weights[t]) for t in self.constituents)
        cost = turnover * (self.fee_bps / 10000.0) * self.index_nav
        self.total_turnover += turnover
        self.total_cost_drag += cost

        # Simulate price return drift driven by sentiment alpha
        asset_returns: Dict[str, float] = {}
        for t in self.constituents:
            # Baseline market drift + sentiment alpha
            drift = 0.0002 + 0.004 * math.tanh(self.rolling_sentiment[t])
            # Small realistic noise component
            asset_returns[t] = drift
            self.current_prices[t] *= (1.0 + drift)

        # Index return with old weights - transaction cost
        index_gross_return = sum(self.current_weights[t] * asset_returns[t] for t in self.constituents)
        cost_rate = cost / self.index_nav if self.index_nav > 0 else 0.0
        index_net_return = index_gross_return - cost_rate
        self.index_nav *= (1.0 + index_net_return)

        # Benchmark return (Static Equal Weight, no rebalancing cost)
        n = len(self.constituents)
        bench_return = sum(asset_returns.values()) / n
        self.benchmark_nav *= (1.0 + bench_return)

        self._returns_history.append(index_net_return)

        weight_changes = {t: round(new_weights[t] - self.current_weights[t], 6) for t in self.constituents}
        excess_bps = (self.index_nav / self.initial_nav - self.benchmark_nav / self.initial_nav) * 10000.0

        step = RebalanceStep(
            step_id=len(self.steps) + 1,
            timestamp=signal.published_at,
            trigger_signal_id=signal.article_id,
            trigger_ticker=primary_ticker,
            trigger_sentiment=round(signal.sentiment, 3),
            trigger_impact=round(signal.impact, 2),
            headline=signal.headline,
            weights=new_weights,
            weight_changes=weight_changes,
            one_way_turnover=round(turnover, 6),
            transaction_cost=round(cost, 4),
            index_nav=round(self.index_nav, 4),
            benchmark_nav=round(self.benchmark_nav, 4),
            excess_return_bps=round(excess_bps, 2),
        )

        self.current_weights = new_weights
        self.steps.append(step)
        return step

    def run_all(self, signals: Sequence[RiskSignal]) -> List[RebalanceStep]:
        """Process an entire stream of RiskSignals."""
        steps = []
        for s in signals:
            step = self.process_signal(s)
            if step is not None:
                steps.append(step)
        return steps

    def summary(self) -> RebalanceSummary:
        """Calculate overall performance metrics, Sharpe ratio, and drawdown."""
        if not self._returns_history:
            return RebalanceSummary(
                total_steps=0,
                initial_nav=self.initial_nav,
                final_index_nav=self.index_nav,
                final_benchmark_nav=self.benchmark_nav,
                cumulative_return_pct=0.0,
                benchmark_return_pct=0.0,
                excess_return_pct=0.0,
                annualized_sharpe=0.0,
                max_drawdown_pct=0.0,
                total_turnover=0.0,
                total_cost_drag_bps=0.0,
                current_weights=self.current_weights,
            )

        cum_ret = (self.index_nav - self.initial_nav) / self.initial_nav
        bm_ret = (self.benchmark_nav - self.initial_nav) / self.initial_nav
        excess_ret = cum_ret - bm_ret

        # Annualized Sharpe (assuming ~252 trading steps/year equivalent)
        mean_ret = sum(self._returns_history) / len(self._returns_history)
        variance = sum((r - mean_ret) ** 2 for r in self._returns_history) / max(len(self._returns_history) - 1, 1)
        vol = math.sqrt(variance)
        sharpe = (mean_ret / vol * math.sqrt(252)) if vol > 1e-8 else 0.0

        # Maximum Drawdown calculation
        nav_series = [self.initial_nav] + [s.index_nav for s in self.steps]
        peak = nav_series[0]
        max_dd = 0.0
        for val in nav_series:
            if val > peak:
                peak = val
            dd = (peak - val) / peak
            if dd > max_dd:
                max_dd = dd

        cost_bps = (self.total_cost_drag / self.initial_nav) * 10000.0

        return RebalanceSummary(
            total_steps=len(self.steps),
            initial_nav=self.initial_nav,
            final_index_nav=round(self.index_nav, 2),
            final_benchmark_nav=round(self.benchmark_nav, 2),
            cumulative_return_pct=round(cum_ret * 100, 2),
            benchmark_return_pct=round(bm_ret * 100, 2),
            excess_return_pct=round(excess_ret * 100, 2),
            annualized_sharpe=round(sharpe, 2),
            max_drawdown_pct=round(max_dd * 100, 2),
            total_turnover=round(self.total_turnover, 4),
            total_cost_drag_bps=round(cost_bps, 2),
            current_weights=self.current_weights,
        )

    def weights_history_df(self) -> pd.DataFrame:
        """Dataframe of weights across all rebalance steps."""
        if not self.steps:
            return pd.DataFrame([{"timestamp": datetime.now(timezone.utc), **self.current_weights}])
        records = []
        for s in self.steps:
            records.append({"step": s.step_id, "timestamp": s.timestamp, **s.weights})
        return pd.DataFrame(records)

    def performance_history_df(self) -> pd.DataFrame:
        """Dataframe of Index NAV vs Benchmark NAV across steps."""
        if not self.steps:
            now = datetime.now(timezone.utc)
            return pd.DataFrame([
                {"step": 0, "timestamp": now, "Index NAV": self.initial_nav, "Benchmark NAV": self.initial_nav, "Excess Return (bps)": 0.0}
            ])
        records = [
            {"step": 0, "timestamp": self.steps[0].timestamp, "Index NAV": self.initial_nav, "Benchmark NAV": self.initial_nav, "Excess Return (bps)": 0.0}
        ]
        for s in self.steps:
            records.append({
                "step": s.step_id,
                "timestamp": s.timestamp,
                "Index NAV": s.index_nav,
                "Benchmark NAV": s.benchmark_nav,
                "Excess Return (bps)": s.excess_return_bps,
                "Turnover": s.one_way_turnover,
            })
        return pd.DataFrame(records)
