from pathlib import Path

import pandas as pd

from src.schemas.risk_signal import Asset, PortfolioState


def load_portfolio(path) -> PortfolioState:
    df = pd.read_csv(Path(path)).where(lambda d: d.notna(), None)
    return PortfolioState(assets=[Asset(**r) for r in df.to_dict("records")])
