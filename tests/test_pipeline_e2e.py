from pathlib import Path

from src.downstream import load_portfolio


def test_end_to_end_offline_and_file_output(tmp_path):
    import json
    from src.nlp_engine import read_signals
    from src.pipeline import main, run_engine
    sigs = run_engine(use_transformers=False)
    assert len(sigs) == 125 and {"Twitter", "Reuters"} <= {s.source for s in sigs}
    assert all(-1 <= s.sentiment <= 1 and 1 <= s.impact <= 10 for s in sigs)
    main(["--out", str(tmp_path)])
    assert read_signals(tmp_path / "risk_signals.jsonl")[0].article_id
    out = json.loads((tmp_path / "stress_results.json").read_text())
    assert out["events"] and all(e["scenario"]["impact"] > 7 for e in out["events"])
    assert out["combined"]["pnl"] < 0


def test_portfolio_has_mixed_asset_types():
    pf = load_portfolio("data/portfolio.csv")
    assert {"loan", "bond", "derivative", "equity"} == {a.asset_type.value for a in pf.assets}


def test_dashboard_smoke():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "src/dashboard/app.py"), default_timeout=60).run()
    assert not at.exception, at.exception
    assert len(at.metric) >= 4


def test_dashboard_after_stream_shows_stress_results():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "src/dashboard/app.py"), default_timeout=60)
    at.session_state["cursor"] = 125
    at.run()
    assert not at.exception, at.exception
    vals = {m.label: m.value for m in at.metric}
    assert int(vals["Triggered events"]) > 0 and vals["Stress loss"] != "$0.0M"
