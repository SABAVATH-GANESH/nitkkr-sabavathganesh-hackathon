"""Unified Pipeline + CLI:
python main.py -> data/output/{risk_signals.jsonl, risk_signals.csv, stress_results.json, rebalance_steps.csv, rebalance_summary.json}
"""
import argparse
import json
from pathlib import Path
from typing import List, Optional

from src.downstream import PortfolioStressTester, TacticalIndexRebalancer, load_portfolio
from src.ingestion import CSVReplaySource, GDELTRSSSource, MultiSource, NewsAPISource
from src.nlp_engine import RiskEngine, write_signals
from src.schemas.risk_signal import RiskSignal

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def build_source(mode: str = "replay", query: str = "stock market OR sanctions OR bank"):
    news = CSVReplaySource(DATA / "synthetic_news.csv")
    social = CSVReplaySource(DATA / "synthetic_social.csv")
    if mode == "newsapi":
        return MultiSource([NewsAPISource(query), social])
    if mode in ("gdelt", "live"):
        return MultiSource([GDELTRSSSource(query), social])
    return MultiSource([news, social])


def run_engine(mode: str = "replay", use_transformers: Optional[bool] = None) -> List[RiskSignal]:
    return RiskEngine(use_transformers=use_transformers).process(build_source(mode).fetch())


def run_rebalancer(signals: List[RiskSignal], out_dir: Path) -> dict:
    """Run Module A: Tactical Index Rebalancer."""
    reb = TacticalIndexRebalancer()
    steps = reb.run_all(signals)
    summary = reb.summary()

    # Save rebalancing performance & weight history
    df_perf = reb.performance_history_df()
    df_perf.to_csv(out_dir / "rebalance_performance.csv", index=False, encoding="utf-8")
    df_weights = reb.weights_history_df()
    df_weights.to_csv(out_dir / "rebalance_weights_history.csv", index=False, encoding="utf-8")

    summary_dict = summary.model_dump(mode="json")
    (out_dir / "rebalance_summary.json").write_text(json.dumps(summary_dict, indent=2), encoding="utf-8")
    return summary_dict


def run_stress_tester(signals: List[RiskSignal], threshold: float, out_dir: Path) -> dict:
    """Run Module B: Wholesale Portfolio Stress Tester."""
    pf = load_portfolio(DATA / "portfolio.csv")
    tester = PortfolioStressTester(pf, impact_threshold=threshold)
    results = tester.stress_all(signals)
    comb = tester.combined(signals)

    stress_dict = {
        "threshold": threshold,
        "events": [r.model_dump(mode="json") for r in results],
        "combined": comb.model_dump(mode="json"),
    }
    (out_dir / "stress_results.json").write_text(json.dumps(stress_dict, indent=2), encoding="utf-8")
    return {"stress_count": len(results), "combined_pnl": comb.pnl, "combined_pnl_pct": comb.pnl_pct}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description="Unified AI/NLP Risk Engine + Tactical Index Rebalancer & Strategic Stress Tester"
    )
    ap.add_argument("--source", choices=["replay", "newsapi", "gdelt"], default="replay")
    ap.add_argument("--threshold", type=float, default=7.0)
    ap.add_argument("--module", choices=["all", "rebalance", "stress"], default="all")
    ap.add_argument("--out", default=str(DATA / "output"))
    a = ap.parse_args(argv)

    out_path = Path(a.out)
    out_path.mkdir(parents=True, exist_ok=True)

    signals = run_engine(a.source)
    write_signals(signals, str(out_path))

    print(f"=== Unified AI/NLP Risk Engine Processed {len(signals)} Signals ===")
    print(f"Signals written to: {out_path / 'risk_signals.jsonl'} and .csv")

    if a.module in ("all", "rebalance"):
        reb_sum = run_rebalancer(signals, out_path)
        print(f"[Module A - Tactical Rebalancer] {reb_sum['total_steps']} rebalances | "
              f"Index Return: {reb_sum['cumulative_return_pct']}% | "
              f"Benchmark: {reb_sum['benchmark_return_pct']}% | "
              f"Excess Alpha: {reb_sum['excess_return_pct']}% | "
              f"Sharpe: {reb_sum['annualized_sharpe']} | "
              f"Turnover: {reb_sum['total_turnover']:.2f}")

    if a.module in ("all", "stress"):
        st_res = run_stress_tester(signals, a.threshold, out_path)
        print(f"[Module B - Stress Tester] {st_res['stress_count']} triggers (impact > {a.threshold}) | "
              f"Combined Worst-Case P&L: ${st_res['combined_pnl']:,.0f} ({st_res['combined_pnl_pct']:.2%})")


if __name__ == "__main__":
    main()
