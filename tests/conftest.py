from datetime import datetime, timezone

import pytest

from src.schemas.risk_signal import Asset, EventType, PortfolioState, RiskSignal

T0 = datetime(2026, 9, 30, 12, 0, tzinfo=timezone.utc)


def make_signal(sentiment=-0.8, impact=8.0, event=EventType.GEOPOLITICAL, tickers=("AAA",), sectors=(),
                published_at=T0, aid="s1"):
    return RiskSignal(article_id=aid, published_at=published_at, source="Reuters", tickers=list(tickers),
                      sectors=list(sectors), sentiment=sentiment, sentiment_confidence=0.9, event_type=event,
                      event_confidence=0.9, impact=impact, headline="h")


def asset(**kw):
    base = dict(asset_id="X", asset_type="equity", counterparty="Acme", ticker="AAA", sector="banking",
                notional=1_000_000, market_value=1_000_000)
    base.update(kw)
    return Asset(**base)


@pytest.fixture
def portfolio():
    return PortfolioState(assets=[
        asset(asset_id="EQ", asset_type="equity"),
        asset(asset_id="LN", asset_type="loan", notional=10e6, market_value=10e6, duration=2.0, pd=0.02, lgd=0.5),
        asset(asset_id="BD", asset_type="bond", notional=10e6, market_value=10e6, duration=5.0, convexity=30, pd=0.01),
        asset(asset_id="DV", asset_type="derivative", notional=20e6, market_value=0.5e6, delta=0.5, dv01=-5000),
        asset(asset_id="OTHER", asset_type="equity", ticker="ZZZ", sector="it")])
