import pytest
from datetime import datetime, timezone
from src.downstream.rebalancer import TacticalIndexRebalancer, DEFAULT_CONSTITUENTS
from src.schemas.risk_signal import EventType, RiskSignal


def sig(ticker, sentiment, impact=5.0, confidence=0.9, source="Reuters"):
    return RiskSignal(
        article_id=f"art-{ticker}",
        published_at=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
        source=source,
        tickers=[ticker],
        sectors=[],
        sentiment=sentiment,
        sentiment_confidence=confidence,
        event_type=EventType.EARNINGS,
        event_confidence=0.9,
        impact=impact,
        headline=f"Test headline for {ticker}",
    )


def test_rebalancer_initialization_and_weights_sum_to_one():
    reb = TacticalIndexRebalancer()
    assert len(reb.constituents) == len(DEFAULT_CONSTITUENTS)
    total_w = sum(reb.current_weights.values())
    assert abs(total_w - 1.0) < 1e-4
    assert all(w > 0 for w in reb.current_weights.values())


def test_positive_sentiment_increases_weight():
    reb = TacticalIndexRebalancer()
    initial_aapl_w = reb.current_weights["AAPL"]
    step = reb.process_signal(sig("AAPL", sentiment=0.85, impact=7.0))
    assert step is not None
    assert reb.current_weights["AAPL"] > initial_aapl_w
    assert abs(sum(reb.current_weights.values()) - 1.0) < 1e-4


def test_negative_sentiment_decreases_weight():
    reb = TacticalIndexRebalancer()
    initial_aapl_w = reb.current_weights["AAPL"]
    step = reb.process_signal(sig("AAPL", sentiment=-0.85, impact=7.0))
    assert step is not None
    assert reb.current_weights["AAPL"] < initial_aapl_w
    assert abs(sum(reb.current_weights.values()) - 1.0) < 1e-4


def test_weight_boundaries_and_box_constraints():
    reb = TacticalIndexRebalancer(min_weight=0.02, max_weight=0.25)
    # Apply massive repeated negative shocks to AAPL
    for _ in range(10):
        reb.process_signal(sig("AAPL", sentiment=-1.0, impact=10.0))
    assert reb.current_weights["AAPL"] >= 0.02

    # Apply massive repeated positive shocks to MSFT
    for _ in range(10):
        reb.process_signal(sig("MSFT", sentiment=1.0, impact=10.0))
    assert reb.current_weights["MSFT"] <= 0.25
    assert abs(sum(reb.current_weights.values()) - 1.0) < 1e-4


def test_turnover_and_cost_accounting():
    reb = TacticalIndexRebalancer(fee_bps=10.0)
    step = reb.process_signal(sig("NVDA", sentiment=0.9))
    assert step is not None
    assert step.one_way_turnover > 0.0
    assert step.transaction_cost > 0.0
    assert reb.total_turnover > 0.0
    assert reb.total_cost_drag > 0.0


def test_sector_spillover_rebalancing():
    reb = TacticalIndexRebalancer()
    jpm_w = reb.current_weights["JPM"]
    hdfc_w = reb.current_weights["HDFCBANK"]
    sector_signal = RiskSignal(
        article_id="sec-01",
        published_at=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
        source="Bloomberg",
        tickers=[],
        sectors=["banking"],
        sentiment=-0.75,
        sentiment_confidence=0.85,
        event_type=EventType.REGULATORY,
        event_confidence=0.9,
        impact=7.5,
        headline="Regulator tightens reserve requirements on banks",
    )
    step = reb.process_signal(sector_signal)
    assert step is not None
    assert reb.current_weights["JPM"] < jpm_w
    assert reb.current_weights["HDFCBANK"] < hdfc_w


def test_run_all_and_summary_metrics():
    reb = TacticalIndexRebalancer()
    signals = [
        sig("AAPL", 0.7),
        sig("BA", -0.8),
        sig("XOM", 0.5),
        sig("JPM", -0.6),
        sig("WMT", 0.3),
    ]
    steps = reb.run_all(signals)
    assert len(steps) == len(signals)
    summary = reb.summary()
    assert summary.total_steps == 5
    assert summary.initial_nav == 1000.0
    assert summary.final_index_nav > 0.0
    assert summary.max_drawdown_pct >= 0.0
    assert summary.total_turnover > 0.0

    df_w = reb.weights_history_df()
    df_p = reb.performance_history_df()
    assert len(df_w) == 5
    assert len(df_p) == 6  # includes step 0
