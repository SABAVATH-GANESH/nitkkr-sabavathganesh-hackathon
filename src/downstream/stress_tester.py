"""Module B: event-driven stress tester for a wholesale-banking portfolio (loans, bonds, derivatives, equity).

Trigger : impact > threshold (default 7) AND a scenario exists for the event type AND sentiment is negative.
Scenario: full-severity shocks per event type (below) scaled by impact/10.
Pricing (per asset, e = exposure in {1, .6, .3, 0}, b = sector beta):
  equity shock  eq = shock_eq * b * e          rate shock dy = bps/1e4 * e       spread shock ds = bps/1e4 * b * e
  PD stress     pd' = min(1, pd * (1 + (m-1) * b * e));  extra expected loss  EL = notional * LGD * (pd' - pd)
  LOAN        P&L = -MV * dur * ds - EL
  BOND        P&L = MV * (-dur*(dy+ds) + 0.5*convexity*(dy+ds)^2) - EL
  DERIVATIVE  P&L = delta * notional * eq + dv01 * bps * e
  EQUITY      P&L = MV * eq
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

from src.schemas.risk_signal import (Asset, AssetStress, AssetType, EventType, PortfolioState, RiskSignal,
                                     StressResult, StressScenario)
from src.utils.errors import StressTestError

# (equity shock, rate bps, spread bps, PD multiplier) at impact = 10
FULL_SEVERITY: Dict[EventType, tuple] = {
    EventType.GEOPOLITICAL: (-0.15, 75.0, 150.0, 1.6),
    EventType.MACRO: (-0.12, 200.0, 100.0, 1.5),
    EventType.CREDIT_EVENT: (-0.08, 0.0, 250.0, 2.5),
    EventType.REGULATORY: (-0.06, 25.0, 60.0, 1.2),
    EventType.CYBER_OPS: (-0.07, 0.0, 50.0, 1.15),
    EventType.SUPPLY_CHAIN: (-0.06, 25.0, 60.0, 1.25),
    EventType.EARNINGS: (-0.05, 0.0, 40.0, 1.1),
    EventType.LEGAL: (-0.05, 0.0, 60.0, 1.2),
    EventType.MA: (-0.03, 0.0, 20.0, 1.05),
}
SYSTEMIC = {EventType.GEOPOLITICAL, EventType.MACRO}
BETA: Dict[EventType, Dict[str, float]] = {
    EventType.GEOPOLITICAL: {"energy": 1.3, "banking": 1.0, "it": 0.6, "*": 0.9},
    EventType.MACRO: {"banking": 1.4, "realty": 1.5, "it": 0.7, "*": 1.0},
    EventType.CREDIT_EVENT: {"*": 1.0},
    EventType.REGULATORY: {"banking": 1.4, "pharma": 1.2, "*": 0.9},
    EventType.CYBER_OPS: {"it": 1.4, "banking": 1.3, "*": 0.6},
    EventType.SUPPLY_CHAIN: {"auto": 1.5, "industrials": 1.3, "*": 0.7},
}
SPILLOVER, MARKET_WIDE = 0.6, 0.3
DEFAULT_THRESHOLD, DEFAULT_MAX_SENTIMENT = 7.0, -0.1


def beta(event: EventType, sector: str) -> float:
    t = BETA.get(event, {"*": 1.0})
    return t.get(sector.lower(), t["*"])


def build_scenario(signal: RiskSignal, scale: float = 1.0) -> StressScenario:
    """Scenario for a signal; `scale` is a user severity dial (what-if)."""
    if signal.event_type not in FULL_SEVERITY:
        raise StressTestError(f"no stress scenario for event type {signal.event_type.value}")
    eq, rate, spr, pdm = FULL_SEVERITY[signal.event_type]
    k = signal.impact / 10.0 * scale
    return StressScenario(event_type=signal.event_type, impact=signal.impact,
                          equity_shock=max(eq * k, -1.0), rate_shock_bps=rate * k, spread_shock_bps=spr * k,
                          pd_multiplier=1.0 + (pdm - 1.0) * k, tickers=signal.tickers, sectors=signal.sectors,
                          trigger_article_id=signal.article_id, headline=signal.headline,
                          triggered_at=signal.published_at)


def exposure(scn: StressScenario, a: Asset) -> float:
    if scn.event_type in SYSTEMIC:
        return 1.0
    if a.ticker and a.ticker in scn.tickers:
        return 1.0
    if a.sector.lower() in {s.lower() for s in scn.sectors}:
        return SPILLOVER
    return MARKET_WIDE if not scn.tickers and not scn.sectors else 0.0


def asset_pnl(a: Asset, scn: StressScenario) -> float:
    e = exposure(scn, a)
    if e == 0.0:
        return 0.0
    b = beta(scn.event_type, a.sector)
    eq, dy, ds = scn.equity_shock * b * e, scn.rate_shock_bps / 1e4 * e, scn.spread_shock_bps / 1e4 * b * e
    pd_s = min(1.0, a.pd * (1.0 + (scn.pd_multiplier - 1.0) * b * e))
    extra_el = a.notional * a.lgd * (pd_s - a.pd)
    if a.asset_type == AssetType.LOAN:
        return -a.market_value * a.duration * ds - extra_el
    if a.asset_type == AssetType.BOND:
        y = dy + ds
        return a.market_value * (-a.duration * y + 0.5 * a.convexity * y * y) - extra_el
    if a.asset_type == AssetType.DERIVATIVE:
        return a.delta * a.notional * eq + a.dv01 * scn.rate_shock_bps * e
    return a.market_value * eq


def _result(pf: PortfolioState, pnls: Dict[str, float], label: str, scn=None, ids=()) -> StressResult:
    by_asset, by_type = [], {t.value: 0.0 for t in AssetType}
    for a in pf.assets:
        p = pnls[a.asset_id]
        by_asset.append(AssetStress(asset_id=a.asset_id, asset_type=a.asset_type, counterparty=a.counterparty,
                                    value_before=a.market_value, value_after=a.market_value + p, pnl=p))
        by_type[a.asset_type.value] += p
    before = pf.total_value
    pnl = sum(pnls.values())
    return StressResult(label=label, scenario=scn, trigger_ids=list(ids), value_before=before, value_after=before + pnl,
                        pnl=pnl, pnl_pct=pnl / before if before else 0.0, pnl_by_asset_type=by_type, by_asset=by_asset)


def apply_scenario(pf: PortfolioState, scn: StressScenario, label: Optional[str] = None) -> StressResult:
    return _result(pf, {a.asset_id: asset_pnl(a, scn) for a in pf.assets},
                   label or f"{scn.event_type.value} (impact {scn.impact:.1f})", scn, [scn.trigger_article_id or ""])


class PortfolioStressTester:
    def __init__(self, portfolio: PortfolioState, impact_threshold: float = DEFAULT_THRESHOLD,
                 max_sentiment: float = DEFAULT_MAX_SENTIMENT, severity_scale: float = 1.0):
        if not 1.0 <= impact_threshold <= 10.0:
            raise StressTestError("impact_threshold must be within [1, 10]")
        if severity_scale <= 0:
            raise StressTestError("severity_scale must be > 0")
        self.pf, self.threshold, self.max_sent, self.scale = portfolio, impact_threshold, max_sentiment, severity_scale

    def is_trigger(self, s: RiskSignal) -> bool:
        return s.impact > self.threshold and s.event_type in FULL_SEVERITY and s.sentiment <= self.max_sent

    def triggers(self, signals: Iterable[RiskSignal]) -> List[RiskSignal]:
        return [s for s in signals if self.is_trigger(s)]

    def stress(self, signal: RiskSignal) -> StressResult:
        return apply_scenario(self.pf, build_scenario(signal, self.scale))

    def stress_all(self, signals: Iterable[RiskSignal]) -> List[StressResult]:
        return [self.stress(s) for s in self.triggers(signals)]

    def combined(self, signals: Sequence[RiskSignal]) -> StressResult:
        """Worst-case per asset across all triggered scenarios (no double counting)."""
        trig = self.triggers(signals)
        scns = [build_scenario(s, self.scale) for s in trig]
        pnls = {a.asset_id: min([asset_pnl(a, c) for c in scns] + [0.0]) for a in self.pf.assets}
        return _result(self.pf, pnls, f"Combined worst-case ({len(trig)} triggers)", None, [s.article_id for s in trig])
