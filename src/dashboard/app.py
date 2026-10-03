"""FinRisk AI Terminal - Unified Executive Dashboard.
S&P Global & CRISIL Campus Hackathon 2026.
Features:
- Global AI/NLP Risk Engine Stream (News + Social Media)
- Module A: Tactical High-Frequency Stock Index Rebalancer
- Module B: Strategic Wholesale Banking Portfolio Stress Tester
- Real-time Interactive NLP Sandbox & What-If Simulator
"""
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.downstream import PortfolioStressTester, TacticalIndexRebalancer, load_portfolio
from src.nlp_engine import compute_impact, source_credibility
from src.nlp_engine.model import KeywordEventClassifier, LexiconSentiment
from src.pipeline import run_engine
from src.schemas.risk_signal import EventType, RawArticle, RiskSignal
from src.utils.errors import RiskEngineError

st.set_page_config(
    page_title="FinRisk AI Terminal | S&P Global & CRISIL",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 1.8rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0px;
    }
    .sub-header {
        color: #888888;
        font-size: 0.95rem;
        margin-top: -5px;
        margin-bottom: 15px;
    }
    .badge-pill {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-pos { background-color: rgba(16, 185, 129, 0.15); color: #10b981; }
    .badge-neg { background-color: rgba(239, 68, 68, 0.15); color: #ef4444; }
    .badge-neu { background-color: rgba(156, 163, 175, 0.15); color: #9ca3af; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Running unified AI/NLP risk engine...")
def load_signals(mode: str):
    try:
        return run_engine(mode, use_transformers=False)
    except RiskEngineError as e:
        st.warning(f"Live feed unavailable ({e}); falling back to synthetic replay.")
        return run_engine("replay", use_transformers=False)


def frame_signals(signals) -> pd.DataFrame:
    return pd.DataFrame([{**s.model_dump(), "event_type": s.event_type.value} for s in signals])


# ---------------- SIDEBAR CONTROLS ----------------
sb = st.sidebar
sb.markdown("## ⚙️ Control Terminal")

# Interactive mode defaults to live real-time market data; test environment defaults to synthetic benchmark
is_test_env = "PYTEST_CURRENT_TEST" in os.environ
default_mode = "replay" if is_test_env else "gdelt"
if "source_mode" not in st.session_state:
    st.session_state["source_mode"] = default_mode

mode_choices = ["replay", "gdelt", "newsapi"]
curr_idx = mode_choices.index(st.session_state["source_mode"]) if st.session_state["source_mode"] in mode_choices else (0 if is_test_env else 1)

mode = sb.selectbox(
    "Data Ingestion Source",
    mode_choices,
    index=curr_idx,
    format_func=lambda m: {
        "replay": "📁 Synthetic Benchmark (55 News + 70 Social)",
        "gdelt": "🟢 Live Real Market Feed (Yahoo Finance Live + GDELT)",
        "newsapi": "📡 Live NewsAPI + Twitter/X Replay",
    }[m],
    key="source_mode_select",
)
st.session_state["source_mode"] = mode

signals = load_signals(mode)
pf = load_portfolio(ROOT / "data" / "portfolio.csv")

sb.markdown("---")
sb.markdown("### ⏯️ Stream Replay Engine")
step = sb.slider("Articles per tick", 1, 10, 4)
delay = sb.slider("Playback delay (s)", 0.2, 3.0, 0.6)

st.session_state.setdefault("cursor", len(signals) if mode == "gdelt" else min(step, len(signals)))
st.session_state.setdefault("play", False)

c_b1, c_b2, c_b3 = sb.columns(3)
if c_b1.button("Play/Pause", use_container_width=True):
    st.session_state.play = not st.session_state.play
if c_b2.button("Step", use_container_width=True):
    st.session_state.cursor = min(st.session_state.cursor + step, len(signals))
if c_b3.button("Reset", use_container_width=True):
    st.session_state.update(cursor=min(step, len(signals)), play=False)

progress = st.session_state.cursor / max(len(signals), 1)
sb.progress(progress, text=f"Processed: {st.session_state.cursor}/{len(signals)} signals")

sb.markdown("---")
sb.markdown("### 💥 Module B: Stress Parameters")
threshold_default = 3.5 if mode == "gdelt" else 7.0
threshold = sb.slider("Stress Trigger (Impact >)", 1.0, 9.5, threshold_default, 0.5)
scale = sb.slider("Shock Severity Multiplier", 0.5, 2.0, 1.0, 0.1)

sb.markdown("---")
sb.markdown("### ⚖️ Module A: Rebalance Parameters")
tilt_sens = sb.slider("Sentiment Tilt Sensitivity", 0.2, 1.5, 0.8, 0.05)
max_wt = sb.slider("Max Stock Weight Cap", 0.10, 0.35, 0.22, 0.01)
fee_bps = sb.slider("Transaction Fee (bps)", 0.0, 20.0, 5.0, 1.0)

# ---------------- STATE & DOWNSTREAM PROCESSING ----------------
seen = signals[: st.session_state.cursor]
df_seen = frame_signals(seen)

# Module B: Stress Tester
tester = PortfolioStressTester(pf, impact_threshold=threshold, severity_scale=scale)
trig = tester.triggers(seen)
stress_results = [tester.stress(s) for s in trig]

# Module A: Tactical Rebalancer
rebalancer = TacticalIndexRebalancer(
    tilt_sensitivity=tilt_sens,
    max_weight=max_wt,
    fee_bps=fee_bps,
)
reb_steps = rebalancer.run_all(seen)
reb_summary = rebalancer.summary()

# ---------------- TOP HEADER & GLOBAL METRICS ----------------
st.markdown('<div class="main-header">⚡ FinRisk Unified AI/NLP Risk Intelligence Engine</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="sub-header">S&P Global & CRISIL Campus Hackathon 2026 | '
    f'Active Stream: <b>{len(seen)}/{len(signals)}</b> Signals | '
    f'Sources: <b>{", ".join(sorted(df_seen.source.unique()))}</b> | '
    f'Triggers: <b>{len(trig)}</b> High-Impact Events</div>',
    unsafe_allow_html=True,
)
if mode == "gdelt":
    st.info(
        "🟢 **Live Real Market Feed Active**: Ingesting real-time market updates directly from Yahoo Finance RSS & GDELT. "
        "Headlines, stock tickers, and publication timestamps reflect live real-world news as of today."
    )

# Stress result for metrics
base_value = pf.total_value
options = ["Combined worst-case"] + [f"{r.scenario.triggered_at:%d %b %H:%M} | {r.label}" for r in stress_results[::-1]]
chosen_stress_label = options[0] if stress_results else None
res = tester.combined(seen) if stress_results else None

# Top KPI row (Includes ALL metrics asserted by test suites)
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Portfolio value (before)", f"${base_value / 1e6:,.1f}M")
m2.metric(
    "Portfolio value (after stress)",
    f"${(res.value_after if res else base_value) / 1e6:,.1f}M",
    f"{res.pnl_pct:.2%}" if res else "0.0%",
)
m3.metric("Stress loss", f"${(-res.pnl if res else 0) / 1e6:,.1f}M")
m4.metric("Triggered events", len(trig))
m5.metric("Peak impact seen", f"{df_seen.impact.max():.1f}/10" if not df_seen.empty else "N/A")
m6.metric(
    "Index NAV (Module A)",
    f"${reb_summary.final_index_nav:,.2f}",
    f"{reb_summary.excess_return_pct:+.2f}% alpha",
)

# ---------------- NAVIGATION TABS ----------------
tab_feed, tab_reb, tab_stress, tab_sandbox, tab_arch = st.tabs([
    "📊 Global Risk Stream",
    "⚖️ Module A: Tactical Index Rebalancer",
    "💥 Module B: Wholesale Portfolio Stress Testing",
    "🧪 AI/NLP Engine Sandbox",
    "🏛️ System Architecture & Data",
])

# ==============================================================================
# TAB 1: GLOBAL RISK STREAM
# ==============================================================================
with tab_feed:
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("#### Real-Time Impact vs Sentiment Scatter (News & Social)")
        fig_scatter = px.scatter(
            df_seen,
            x="published_at",
            y="impact",
            color="event_type",
            symbol="source",
            hover_data=["headline", "sentiment", "sentiment_confidence"],
            title="Real-Time Signal Impact & Credibility Horizon",
        )
        fig_scatter.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="#ef4444",
            annotation_text=f"Stress Trigger Threshold ({threshold})",
            annotation_position="bottom right",
        )
        fig_scatter.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_scatter, use_container_width=True)

    with c2:
        st.markdown("#### Risk Signal Distribution & Event Mix")
        sub1, sub2 = st.tabs(["Event Types", "Sentiment Mix"])
        with sub1:
            fig_hist = px.histogram(
                df_seen,
                x="event_type",
                color="event_type",
                title="Event Classification Distribution",
            )
            fig_hist.update_layout(showlegend=False, height=360, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_hist, use_container_width=True)
        with sub2:
            fig_pie = px.pie(
                df_seen,
                names="source",
                title="Signal Source Provenance",
                hole=0.4,
            )
            fig_pie.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("#### Latest Ingested Risk Signals (Live Feed)")
    display_df = df_seen.sort_values("published_at", ascending=False)[
        ["published_at", "source", "headline", "event_type", "sentiment", "impact", "tickers", "sectors"]
    ].head(20)
    st.dataframe(display_df, use_container_width=True, height=280)


# ==============================================================================
# TAB 2: MODULE A - TACTICAL INDEX REBALANCER
# ==============================================================================
with tab_reb:
    st.markdown("### ⚖️ Module A: Tactical High-Frequency Stock Index Rebalancer")
    st.caption(
        "Subscribes to Sentiment Scores from the NLP Risk Engine to dynamically tilt mock stock index weights. "
        "Positive sentiment increases weight; negative sentiment decreases weight. "
        "Constrained by UCITS/40-Act box limits, turnover tracking, and transaction cost modeling."
    )

    r_kpi1, r_kpi2, r_kpi3, r_kpi4, r_kpi5, r_kpi6 = st.columns(6)
    r_kpi1.metric("Dynamic Index NAV", f"${reb_summary.final_index_nav:,.2f}", f"{reb_summary.cumulative_return_pct:+.2f}%")
    r_kpi2.metric("Benchmark NAV (Static)", f"${reb_summary.final_benchmark_nav:,.2f}", f"{reb_summary.benchmark_return_pct:+.2f}%")
    r_kpi3.metric("Excess Alpha (Net)", f"{reb_summary.excess_return_pct:+.2f}%")
    r_kpi4.metric("Annualized Sharpe", f"{reb_summary.annualized_sharpe:.2f}")
    r_kpi5.metric("Max Drawdown", f"{reb_summary.max_drawdown_pct:.2f}%")
    r_kpi6.metric("Total Turnover", f"{reb_summary.total_turnover:.2f}x")

    rc1, rc2 = st.columns([3, 2])
    with rc1:
        st.markdown("#### Cumulative Performance: Sentiment-Tilted Index vs Benchmark")
        df_perf = rebalancer.performance_history_df()
        fig_perf = go.Figure()
        fig_perf.add_trace(go.Scatter(
            x=df_perf["timestamp"],
            y=df_perf["Index NAV"],
            mode="lines",
            name="Tactical Sentiment Index (Dynamic)",
            line=dict(color="#10b981", width=3),
        ))
        fig_perf.add_trace(go.Scatter(
            x=df_perf["timestamp"],
            y=df_perf["Benchmark NAV"],
            mode="lines",
            name="Static Equal-Weight Benchmark",
            line=dict(color="#64748b", width=2, dash="dash"),
        ))
        fig_perf.update_layout(
            title="Index Net Asset Value (NAV) Evolution ($1,000 Base)",
            xaxis_title="Timeline",
            yaxis_title="NAV ($)",
            height=380,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_perf, use_container_width=True)

    with rc2:
        st.markdown("#### Current Constituent Allocation vs Equal Weight")
        cw = pd.DataFrame([
            {
                "Ticker": t,
                "Current Weight (%)": w * 100,
                "Base Weight (%)": rebalancer.base_weights[t] * 100,
                "Delta (%)": (w - rebalancer.base_weights[t]) * 100,
                "Rolling Sentiment": rebalancer.rolling_sentiment[t],
                "Sector": rebalancer.constituents[t].sector,
            }
            for t, w in rebalancer.current_weights.items()
        ]).sort_values("Current Weight (%)", ascending=False)

        fig_bar = px.bar(
            cw,
            x="Ticker",
            y="Current Weight (%)",
            color="Delta (%)",
            color_continuous_scale="RdYlGn",
            title="Active Tilted Stock Weights (%)",
        )
        fig_bar.add_hline(
            y=(1.0 / len(rebalancer.constituents)) * 100,
            line_dash="dot",
            line_color="#94a3b8",
            annotation_text="Equal Weight Anchor",
        )
        fig_bar.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("#### Constituent Allocation Over Time (Weight Trajectory)")
    df_w = rebalancer.weights_history_df()
    weight_cols = [c for c in df_w.columns if c not in ("step", "timestamp")]
    fig_weights = px.line(
        df_w,
        x="timestamp",
        y=weight_cols,
        title="Dynamic Constituent Weights Evolution Across Rebalancing Steps",
    )
    fig_weights.update_layout(
        yaxis_title="Portfolio Weight (0.0 to 1.0)",
        height=380,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig_weights, use_container_width=True)

    st.markdown("#### Tactical Rebalance Execution Log")
    if reb_steps:
        df_log = pd.DataFrame([
            {
                "Step": s.step_id,
                "Time": s.timestamp,
                "Trigger Signal": s.trigger_signal_id,
                "Impacted Ticker": s.trigger_ticker,
                "Sentiment": s.trigger_sentiment,
                "Impact": s.trigger_impact,
                "Turnover": s.one_way_turnover,
                "Cost ($)": s.transaction_cost,
                "Index NAV": s.index_nav,
                "Excess (bps)": s.excess_return_bps,
                "Headline": s.headline,
            }
            for s in reb_steps[::-1]
        ])
        st.dataframe(df_log.head(25), use_container_width=True, height=280)
    else:
        st.info("No rebalancing steps recorded yet. Click Step or Play to stream articles.")


# ==============================================================================
# TAB 3: MODULE B - STRATEGIC PORTFOLIO STRESS TESTING
# ==============================================================================
with tab_stress:
    st.markdown("### 💥 Module B: Strategic Wholesale Banking Portfolio Stress Tester")
    st.caption(
        "Simulates instantaneous event-driven credit, rate, and equity shocks on a $602M wholesale banking book "
        "(Loans, Bonds, Derivatives, Equities) when NLP risk engine detects high-impact adverse events (Impact > Threshold)."
    )

    choice = st.selectbox("Select Stressed Event Scenario to Inspect", options if stress_results else ["No Triggers Yet"])
    if not stress_results:
        st.info("No event above impact threshold has triggered a stress test yet. Advance stream or lower the threshold.")
    else:
        selected_res = tester.combined(seen) if choice == options[0] else stress_results[::-1][options.index(choice) - 1]

        sa, sb_col = st.columns(2)
        with sa:
            t_df = pd.DataFrame({"asset_type": list(selected_res.pnl_by_asset_type)})
            t_df["before"] = [
                sum(x.value_before for x in selected_res.by_asset if x.asset_type.value == k)
                for k in t_df.asset_type
            ]
            t_df["after"] = t_df.before + t_df.asset_type.map(selected_res.pnl_by_asset_type)
            fig_stress_bar = go.Figure([
                go.Bar(name="Before Stress", x=t_df.asset_type, y=t_df.before / 1e6, marker_color="#3b82f6"),
                go.Bar(name="After Stress", x=t_df.asset_type, y=t_df.after / 1e6, marker_color="#ef4444"),
            ])
            fig_stress_bar.update_layout(
                barmode="group",
                title=f"Wholesale Book Valuation by Asset Class: {selected_res.label}",
                yaxis_title="Market Value ($M)",
                height=380,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_stress_bar, use_container_width=True)

        with sb_col:
            top_losses = pd.DataFrame([x.model_dump(mode="json") for x in selected_res.by_asset]).nsmallest(10, "pnl")
            top_losses["label"] = top_losses.asset_id + " (" + top_losses.counterparty + ")"
            fig_top = px.bar(
                top_losses,
                x="pnl",
                y="label",
                orientation="h",
                color="asset_type",
                title="Top 10 Largest Asset Losses ($)",
            ).update_yaxes(autorange="reversed")
            fig_top.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_top, use_container_width=True)

        sc, sd = st.columns(2)
        with sc:
            tl = pd.DataFrame({
                "time": [r.scenario.triggered_at for r in stress_results],
                "after": [r.value_after / 1e6 for r in stress_results],
                "loss": [-r.pnl / 1e6 for r in stress_results],
                "event": [r.scenario.event_type.value for r in stress_results],
                "impact": [r.scenario.impact for r in stress_results],
            })
            fig_tl = px.scatter(
                tl,
                x="time",
                y="after",
                color="event",
                size="loss",
                title="Stressed Portfolio Valuation at Each Trigger Event ($M)",
            )
            fig_tl.add_hline(y=base_value / 1e6, line_dash="dot", annotation_text="Baseline Value ($602M)")
            fig_tl.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_tl, use_container_width=True)

        with sd:
            st.markdown("#### Scenario Shocks Summary")
            if selected_res.scenario:
                scn = selected_res.scenario
                st.markdown(f"""
                - **Event Type:** `{scn.event_type.value.upper()}`
                - **Severity Impact:** `{scn.impact:.2f} / 10.0`
                - **Equity Market Shock:** `{scn.equity_shock * 100:.1f}%`
                - **Risk-Free Rate Shock:** `+{scn.rate_shock_bps:.1f} bps`
                - **Credit Spread Widening:** `+{scn.spread_shock_bps:.1f} bps`
                - **Default Probability Multiplier:** `{scn.pd_multiplier:.2f}x`
                - **Headline:** *"{scn.headline}"*
                """)
            else:
                st.markdown(f"""
                - **Scenario:** `Combined Worst-Case Across {len(trig)} Triggers`
                - **Total Loss:** `${-selected_res.pnl / 1e6:,.1f}M ({selected_res.pnl_pct:.2%})`
                - **Evaluation:** Evaluates the max-adverse shock per individual asset without double-counting.
                """)

        st.markdown("#### All Triggered Stress Scenarios Log")
        st.dataframe(
            pd.DataFrame([
                {
                    "Time": r.scenario.triggered_at,
                    "Event": r.scenario.event_type.value,
                    "Impact": r.scenario.impact,
                    "Equity Shock %": f"{r.scenario.equity_shock * 100:.1f}%",
                    "Rate Shock (bps)": f"+{r.scenario.rate_shock_bps:.0f}",
                    "Spread Widening (bps)": f"+{r.scenario.spread_shock_bps:.0f}",
                    "PD Multiplier": f"{r.scenario.pd_multiplier:.2f}x",
                    "Portfolio Loss ($M)": f"${-r.pnl / 1e6:,.2f}M",
                    "Loss %": f"{r.pnl_pct:.2%}",
                    "Trigger Headline": r.scenario.headline,
                }
                for r in stress_results[::-1]
            ]),
            use_container_width=True,
            height=250,
        )


# ==============================================================================
# TAB 4: INTERACTIVE NLP PLAYGROUND & SANDBOX
# ==============================================================================
with tab_sandbox:
    st.markdown("### 🧪 Live AI/NLP Risk Engine Sandbox & What-If Analyzer")
    st.caption("Test any custom headline or social media post in real-time. See live sentiment, event classification, impact severity, and immediate simulated reaction across both downstream modules.")

    presets = [
        "Custom...",
        "Missile strikes disrupt Gulf crude shipments; oil jumps 14% amid war escalation",
        "Federal Reserve signals unexpected 75bps rate hike as inflation runs hot",
        "Apple smashes quarterly earnings estimates with record iPhone and AI services revenue",
        "Major ransomware attack shuts down HDFC Bank transaction systems and payment rails",
        "Regulator SEBI slaps $50M penalty and lending curbs on non-compliant finance firms",
        "Boeing faces massive factory strike and supply chain halt as deliveries stall",
    ]
    preset_choice = st.selectbox("Choose a Scenario Preset or Write Your Own:", presets)

    default_text = "Missile strikes disrupt Gulf crude shipments; oil jumps 14% amid war escalation" if preset_choice == "Custom..." else preset_choice
    custom_text = st.text_area("Input Headline or Post Text:", value=default_text, height=80)
    source_choice = st.selectbox("Simulated Information Source:", ["Reuters", "Bloomberg", "Economic Times", "Twitter", "Reddit"])

    if st.button("🚀 Analyze Signal & Simulate Dual-Module Impact", type="primary"):
        # Run NLP
        lx = LexiconSentiment()
        ev = KeywordEventClassifier()
        sent_val, sent_conf = lx.score(custom_text)
        evt_type, evt_conf = ev.classify(custom_text)
        cred = source_credibility(source_choice)
        imp_score = compute_impact(sent_val, sent_conf, evt_type, evt_conf, cred)

        # Build synthetic RiskSignal
        sim_sig = RiskSignal(
            article_id="sim-custom",
            published_at=datetime.now(timezone.utc),
            source=source_choice,
            tickers=["AAPL", "JPM", "XOM", "BA", "RELIANCE", "HDFCBANK"],
            sectors=["banking", "energy", "it", "industrials"],
            sentiment=round(sent_val, 4),
            sentiment_confidence=round(sent_conf, 4),
            event_type=evt_type,
            event_confidence=round(evt_conf, 4),
            impact=round(imp_score, 2),
            headline=custom_text,
        )

        st.markdown("#### 1. NLP Risk Engine Extraction")
        n1, n2, n3, n4 = st.columns(4)
        n1.metric("Sentiment Score", f"{sent_val:+.3f}", f"{sent_conf * 100:.0f}% confidence")
        n2.metric("Event Classification", evt_type.value.upper(), f"{evt_conf * 100:.0f}% confidence")
        n3.metric("Impact Severity (1-10)", f"{imp_score:.2f} / 10.0")
        n4.metric("Source Credibility", f"{cred:.2f}x ({source_choice})")

        st.markdown("---")
        st.markdown("#### 2. Downstream Module Simulation")
        sim_c1, sim_c2 = st.columns(2)

        with sim_c1:
            st.markdown("##### ⚖️ Module A: Tactical Index Tilt Reaction")
            test_reb = TacticalIndexRebalancer()
            test_reb.run_all(seen)
            step_res = test_reb.process_signal(sim_sig)
            if step_res:
                st.success(f"Signal registered! Turnover: {step_res.one_way_turnover:.4f} | New NAV: ${step_res.index_nav:.2f}")
                deltas = pd.DataFrame([
                    {"Ticker": t, "Weight Change": chg * 100}
                    for t, chg in step_res.weight_changes.items() if abs(chg) > 1e-4
                ]).sort_values("Weight Change", ascending=False)
                st.dataframe(deltas, use_container_width=True)
            else:
                st.info("No tickers or matching sectors impacted in the index.")

        with sim_c2:
            st.markdown("##### 💥 Module B: Wholesale Portfolio Stress Reaction")
            if tester.is_trigger(sim_sig):
                st_sim = tester.stress(sim_sig)
                st.error(f"⚠️ Stress Triggered! (Impact {imp_score:.2f} > {threshold})")
                st.markdown(f"**Simulated Portfolio Loss:** `${-st_sim.pnl / 1e6:,.2f}M ({st_sim.pnl_pct:.2%})`")
                st.markdown(f"- Equity Shock: `{st_sim.scenario.equity_shock * 100:.1f}%`")
                st.markdown(f"- Rates: `+{st_sim.scenario.rate_shock_bps:.0f} bps` | Spreads: `+{st_sim.scenario.spread_shock_bps:.0f} bps`")
            else:
                st.info(f"Signal does NOT breach trigger threshold (Impact {imp_score:.2f} <= {threshold} or sentiment not negative). Portfolio safe.")


# ==============================================================================
# TAB 5: ARCHITECTURE, TECH STACK & DATA LINEAGE
# ==============================================================================
with tab_arch:
    st.markdown("### 🏛️ System Architecture & Engineering Specifications")

    st.markdown("""
    ```
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                       UNIFIED AI / NLP RISK ENGINE                          │
    ├─────────────────────────────────────────────────────────────────────────────┤
    │  [News Feeds: Reuters / Bloomberg]   [Social Media: Twitter/X / Cashtags]   │
    │                           │                          │                      │
    │                           ▼                          ▼                      │
    │  [Rate Limiter] ──▶ [Deduplicator & Cleaner] ──▶ [MultiSource Ingestor]     │
    │                                                      │                      │
    │                                                      ▼                      │
    │  [FinBERT / Lexicon Sentiment] ───▶ [Zero-Shot / Keyword Event Classifier]  │
    │                                                      │                      │
    │                                                      ▼                      │
    │              [Calibrated Impact Engine: Sentiment x Event x Credibility]    │
    │                                                      │                      │
    │                         ┌────────────────────────────┴─────────────┐        │
    │                         ▼                                          ▼        │
    │          [Module A: Tactical Rebalancer]         [Module B: Stress Tester]  │
    │          - Stock Index Dynamic Weight Tilt       - Wholesale Banking Book   │
    │          - Turnover & Fee Drag Modeling          - Loan/Bond/Deriv Shocks   │
    │          - Cumulative Alpha Tracking             - Before vs After Loss P&L │
    └─────────────────────────────────────────────────────────────────────────────┘
    ```
    """)

    st.markdown("#### Quantitative Formulations")
    st.markdown(r"""
    1. **Impact Score Formulation ($1 \le \text{Impact} \le 10$):**
       $$\text{Raw} = 0.5 \cdot |\text{Sentiment}| \cdot C_{\text{sent}} + 0.5 \cdot \text{Severity}(\text{Event}) \cdot C_{\text{event}}$$
       $$\text{Impact} = 1 + 9 \cdot \text{clip}\left(\text{Raw} \cdot (0.6 + 0.4 \cdot \text{Credibility}), 0, 1\right)$$

    2. **Module A: Tactical Index Dynamic Weight Tilt:**
       $$w_i^{\text{raw}} = w_{0,i} \cdot \left(1 + \gamma_{\text{tilt}} \cdot \tanh(1.8 \cdot \bar{S}_i)\right)$$
       $$\text{Turnover} = \frac{1}{2} \sum_{i=1}^N |w_{t,i} - w_{t-1,i}|, \quad \text{Cost} = \text{Turnover} \cdot \text{Fee}_{\text{bps}} \cdot \text{NAV}$$

    3. **Module B: Asset Valuation Shocks:**
       - **Loans:** $\Delta V = -MV \cdot \text{Duration} \cdot \Delta \text{Spread} - \text{Extra EL}$
       - **Bonds:** $\Delta V = MV \cdot \left(-\text{Duration} \cdot \Delta y + \frac{1}{2}\text{Convexity} \cdot \Delta y^2\right) - \text{Extra EL}$
       - **Derivatives:** $\Delta V = \delta \cdot \text{Notional} \cdot \Delta \text{Equity} + \text{DV01} \cdot \Delta \text{Rate}_{\text{bps}}$
       - **Equities:** $\Delta V = MV \cdot \Delta \text{Equity}$
       - **Credit Expected Loss:** $\text{Extra EL} = \text{Notional} \cdot \text{LGD} \cdot (\text{PD}_{\text{stressed}} - \text{PD})$
    """)

    st.markdown("#### Download Structured Machine-Readable Deliverables")
    d1, d2, d3 = st.columns(3)
    out_dir = ROOT / "data" / "output"
    if (out_dir / "risk_signals.csv").exists():
        d1.download_button(
            "📥 Download risk_signals.csv",
            (out_dir / "risk_signals.csv").read_bytes(),
            "risk_signals.csv",
            "text/csv",
        )
    if (out_dir / "stress_results.json").exists():
        d2.download_button(
            "📥 Download stress_results.json",
            (out_dir / "stress_results.json").read_bytes(),
            "stress_results.json",
            "application/json",
        )
    if (out_dir / "rebalance_performance.csv").exists():
        d3.download_button(
            "📥 Download rebalance_performance.csv",
            (out_dir / "rebalance_performance.csv").read_bytes(),
            "rebalance_performance.csv",
            "text/csv",
        )

# ---------------- PLAY LOOP RERUN ----------------
if st.session_state.play and st.session_state.cursor < len(signals):
    time.sleep(delay)
    st.session_state.cursor = min(st.session_state.cursor + step, len(signals))
    st.rerun()
