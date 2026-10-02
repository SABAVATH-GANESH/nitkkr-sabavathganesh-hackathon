"""Seeded generator for ALL demo data (fully synthetic; no real or confidential data).
Writes data/portfolio.csv, data/synthetic_news.csv, data/synthetic_social.csv."""
import csv
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
CPTY = [("Reliance Industries", "RELIANCE", "energy"), ("HDFC Bank", "HDFCBANK", "banking"), ("Tata Consultancy", "TCS", "it"),
        ("Tata Motors", "TATAMOTORS", "auto"), ("JP Morgan", "JPM", "banking"), ("Exxon Mobil", "XOM", "energy"),
        ("Boeing", "BA", "industrials"), ("Pfizer", "PFE", "pharma"), ("Apple", "AAPL", "it"), ("Walmart", "WMT", "retail")]

NEWS = {  # kind -> (title, body)
 "geo": [("War escalates as sanctions and border tensions spark market crash fears", "Supply disruption and recession warning loom over {c}."),
         ("Missile conflict and new sanctions threaten oil supply; markets slump", "Border tensions raise default risk for exposed lenders including {c}.")],
 "macro": [("RBI rate hike and inflation fears hit {c} as GDP slumps", "Rupee weakens and recession risk grows, analysts warn.")],
 "credit": [("{c} credit rating downgraded after missed payment, default fears grow", "Debt restructuring talks begin as bondholders brace for losses."),
            ("{c} faces insolvency risk; lenders weigh bankruptcy filing", "Credit rating downgrade follows debt restructuring request.")],
 "cyber": [("{c} hit by cyber breach, systems shutdown for hours", "Ransomware attack disrupts operations; data leak feared.")],
 "reg": [("Regulator opens probe into {c}, fine possible", "Compliance failures raise ban risk.")],
 "supply": [("Port strike halts {c} shipments, supply chain disruption widens", "Factory shortage and logistics delays expected.")],
 "legal": [("Court fines {c} in fraud lawsuit", "Litigation risk rises after adverse verdict.")],
 "earn_neg": [("{c} misses quarterly earnings, issues profit warning", "Revenue and guidance disappoint; analysts downgrade stock.")],
 "earn_pos": [("{c} beats quarterly earnings, raises guidance on record profit", "Strong results; analysts upgrade.")],
 "launch": [("{c} launches new product line, shares rally", "The company unveils a platform expected to boost growth.")],
 "ma": [("{c} announces acquisition of rival; merger approved", "Deal expected to boost margins and growth.")],
 "neu": [("{c} to hold annual general meeting next month", "Shareholders to meet on routine agenda.")]}
TWEETS = {
 "geo": ["$%s 🚨 sanctions + border tensions = oil supply disruption, market crash incoming #war #markets", "war escalating, sanctions everywhere. recession fears are real, sell-off day $%s"],
 "macro": ["RBI rate hike + inflation fears again... GDP slumps, recession warning #macro $%s", "inflation fears and rate hike hit $%s hard, rupee weakens"],
 "credit": ["$%s credit rating downgraded!! default fears, debt restructuring talk #credit", "missed payment at $%s?? bankruptcy rumours, downgrade incoming"],
 "cyber": ["$%s hack confirmed, breach + shutdown. ransomware attack ugh #cyber"],
 "reg": ["regulator probe into $%s, fine and ban risk #regulation"],
 "supply": ["port strike halts $%s shipments, supply chain disruption again"],
 "earn_pos": ["$%s beats earnings, record profit, analysts upgrade 🚀 #earnings"],
 "launch": ["$%s launches new product today, shares rally 📈 #launch"],
 "ma": ["$%s acquisition approved, merger boosts growth"],
 "neu": ["$%s AGM next month, nothing new #investing"]}


def _times(rng, n, t0, step):
    return [t0 + timedelta(minutes=step * i + rng.randint(0, 5)) for i in range(n)]


def main(n_news: int = 55, n_social: int = 70, seed: int = 11):
    rng = random.Random(seed)
    DATA.mkdir(exist_ok=True)
    # --- portfolio: wholesale banking book
    rows, i = [], 0
    for name, tk, sec in CPTY:
        i += 1
        n = rng.choice([20, 35, 50, 80]) * 1e6
        pd_, lgd = rng.choice([.01, .02, .04, .08]), .45
        d = rng.uniform(1.5, 4)
        rows.append([f"LN{i:03d}", "loan", name, tk, sec, n, round(n * (1 - pd_ * lgd), 0), round(d, 2), round(d * d, 2), pd_, lgd, 0, 0])
    for name, tk, sec in CPTY[:8]:
        i += 1
        n = rng.choice([10, 25, 40]) * 1e6
        d = rng.uniform(3, 9); pd_ = rng.choice([.005, .01, .03])
        rows.append([f"BD{i:03d}", "bond", name, tk, sec, n, round(n * rng.uniform(.95, 1.02), 0), round(d, 2), round(d * d * 1.1, 2), pd_, .45, 0, 0])
    for name, tk, sec in rng.sample(CPTY, 6):
        i += 1
        n = rng.choice([50, 100, 150]) * 1e6
        rows.append([f"DV{i:03d}", "derivative", name, tk, sec, n, round(rng.uniform(-1, 1.5) * 1e6, 0), 0, 0, .01, .45,
                     round(rng.uniform(.1, .5), 2), round(-rng.uniform(2e3, 1.5e4), 0)])
    for name, tk, sec in rng.sample(CPTY, 4):
        i += 1
        mv = rng.choice([5, 10, 20]) * 1e6
        rows.append([f"EQ{i:03d}", "equity", name, tk, sec, mv, mv, 0, 0, 0, .45, 0, 0])
    with open(DATA / "portfolio.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["asset_id", "asset_type", "counterparty", "ticker", "sector", "notional", "market_value", "duration", "convexity", "pd", "lgd", "delta", "dv01"])
        w.writerows(rows)
    t0 = datetime(2026, 9, 28, 3, 30, tzinfo=timezone.utc)

    def kind(phase):  # a stress episode builds mid-stream
        hot = 0.3 < phase < 0.75
        w = {"geo": 1 + 4 * hot, "macro": 1 + 2 * hot, "credit": 1 + 3 * hot, "cyber": 1, "reg": 1, "supply": 1, "legal": 1,
             "earn_neg": 1, "earn_pos": 3, "launch": 2, "ma": 1.5, "neu": 3}
        return rng.choices(list(w), list(w.values()))[0]

    with open(DATA / "synthetic_news.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["article_id", "published_at", "source", "title", "body", "tickers", "sectors"])
        for j, ts in enumerate(_times(rng, n_news, t0, 40)):
            k = kind(j / n_news); name, tk, sec = rng.choice(CPTY)
            title, body = rng.choice(NEWS[k]); systemic = k in ("geo", "macro") and rng.random() < .6
            w.writerow([f"news-{j:04d}", ts.isoformat(), rng.choice(["Reuters", "Bloomberg", "Economic Times"]),
                        title.format(c=name), body.format(c=name), "" if systemic else tk, "" if systemic else sec])
    with open(DATA / "synthetic_social.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["article_id", "published_at", "source", "title", "body", "tickers", "sectors"])
        for j, ts in enumerate(_times(rng, n_social, t0, 32)):
            k = kind(j / n_social)
            if k not in TWEETS: k = "neu"
            name, tk, sec = rng.choice(CPTY)
            w.writerow([f"tw-{j:04d}", ts.isoformat(), rng.choice(["Twitter", "Twitter", "Reddit"]), rng.choice(TWEETS[k]) % tk, "", tk, sec])


if __name__ == "__main__":
    main()
