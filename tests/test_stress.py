import pytest

from src.downstream import stress_tester as st
from src.downstream.stress_tester import PortfolioStressTester, apply_scenario, build_scenario
from src.schemas.risk_signal import EventType
from src.utils.errors import StressTestError
from tests.conftest import asset, make_signal


def pnl_of(res, aid):
    return next(a.pnl for a in res.by_asset if a.asset_id == aid)


def test_trigger_rules(portfolio):
    t = PortfolioStressTester(portfolio)
    assert t.is_trigger(make_signal(impact=7.5))
    assert not t.is_trigger(make_signal(impact=7.0))                        # strictly greater than 7
    assert not t.is_trigger(make_signal(impact=9, sentiment=0.6))           # good news never stresses
    assert not t.is_trigger(make_signal(impact=9, event=EventType.PRODUCT_LAUNCH))
    assert not t.is_trigger(make_signal(impact=9, event=EventType.NEUTRAL))
    assert len(t.triggers([make_signal(impact=8, aid="a"), make_signal(impact=2, aid="b")])) == 1
    with pytest.raises(StressTestError):
        PortfolioStressTester(portfolio, impact_threshold=11)
    with pytest.raises(StressTestError):
        PortfolioStressTester(portfolio, severity_scale=0)


def test_scenario_scales_with_impact_and_dial():
    s10 = build_scenario(make_signal(impact=10, event=EventType.MACRO))
    s5 = build_scenario(make_signal(impact=5, event=EventType.MACRO))
    assert s10.equity_shock == pytest.approx(-0.12) and s10.rate_shock_bps == pytest.approx(200)  # spec-style shocks
    assert s5.equity_shock == pytest.approx(s10.equity_shock / 2) and s5.pd_multiplier == pytest.approx(1.25)
    assert build_scenario(make_signal(impact=10), scale=2.0).equity_shock == pytest.approx(-0.30)
    with pytest.raises(StressTestError):
        build_scenario(make_signal(event=EventType.NEUTRAL))


def test_exposure_levels(portfolio):
    eq, other = portfolio.assets[0], portfolio.assets[-1]
    cyber = build_scenario(make_signal(event=EventType.CYBER_OPS, tickers=["AAA"]))
    assert st.exposure(cyber, eq) == 1.0 and st.exposure(cyber, other) == 0.0
    sect = build_scenario(make_signal(event=EventType.CYBER_OPS, tickers=[], sectors=["IT"]))
    assert st.exposure(sect, other) == st.SPILLOVER
    wide = build_scenario(make_signal(event=EventType.CYBER_OPS, tickers=[], sectors=[]))
    assert st.exposure(wide, eq) == st.MARKET_WIDE
    geo = build_scenario(make_signal(event=EventType.GEOPOLITICAL, tickers=["NOPE"]))
    assert st.exposure(geo, other) == 1.0                                    # systemic hits everyone


def test_equity_and_derivative_hand_calc(portfolio):
    scn = build_scenario(make_signal(impact=10, event=EventType.MACRO))      # eq -12%, +200bp, banking beta 1.4
    r = apply_scenario(portfolio, scn)
    assert pnl_of(r, "EQ") == pytest.approx(1e6 * -0.12 * 1.4)
    assert pnl_of(r, "DV") == pytest.approx(0.5 * 20e6 * -0.12 * 1.4 + -5000 * 200)


def test_loan_and_bond_hand_calc(portfolio):
    scn = build_scenario(make_signal(impact=10, event=EventType.CREDIT_EVENT))  # +250bp spread, PD x2.5, beta 1
    r = apply_scenario(portfolio, scn)
    ds = 0.025
    loan_el = 10e6 * 0.5 * (0.02 * 2.5 - 0.02)
    assert pnl_of(r, "LN") == pytest.approx(-10e6 * 2.0 * ds - loan_el)
    bond_el = 10e6 * 0.45 * (0.01 * 2.5 - 0.01)
    assert pnl_of(r, "BD") == pytest.approx(10e6 * (-5.0 * ds + 0.5 * 30 * ds**2) - bond_el)


def test_pd_capped_and_value_identity(portfolio):
    risky = asset(asset_id="R", asset_type="loan", notional=1e6, market_value=1e6, duration=1, pd=0.9, lgd=1.0)
    from src.schemas.risk_signal import PortfolioState
    r = apply_scenario(PortfolioState(assets=[risky]), build_scenario(make_signal(impact=10, event=EventType.CREDIT_EVENT)))
    assert pnl_of(r, "R") >= -1e6 * 1 * 0.025 - 1e6 * 1.0 * 0.1 - 1e-6        # PD capped at 1 -> extra EL <= 0.1*notional*LGD
    r2 = apply_scenario(portfolio, build_scenario(make_signal(impact=9, event=EventType.GEOPOLITICAL)))
    assert r2.value_before - r2.value_after == pytest.approx(-r2.pnl)
    assert sum(r2.pnl_by_asset_type.values()) == pytest.approx(r2.pnl)
    assert sum(a.pnl for a in r2.by_asset) == pytest.approx(r2.pnl) and r2.pnl < 0


def test_unexposed_asset_untouched_and_monotonic(portfolio):
    low = apply_scenario(portfolio, build_scenario(make_signal(impact=7.5, event=EventType.CYBER_OPS)))
    high = apply_scenario(portfolio, build_scenario(make_signal(impact=10, event=EventType.CYBER_OPS)))
    assert pnl_of(low, "OTHER") == 0.0
    assert high.pnl < low.pnl < 0


def test_combined_is_worst_case_not_sum(portfolio):
    sigs = [make_signal(impact=9, event=EventType.GEOPOLITICAL, aid="g"), make_signal(impact=8, event=EventType.CREDIT_EVENT, aid="c"),
            make_signal(impact=3, aid="ignored")]
    t = PortfolioStressTester(portfolio)
    each = t.stress_all(sigs)
    comb = t.combined(sigs)
    assert len(each) == 2 and comb.trigger_ids == ["g", "c"]
    assert comb.pnl <= min(r.pnl for r in each) + 1e-6
    assert comb.pnl >= sum(r.pnl for r in each)                               # no double counting
    assert t.combined([make_signal(impact=2)]).pnl == 0.0                     # nothing triggered -> no loss
