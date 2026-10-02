"""Machine-readable output of the engine: JSON-Lines + CSV files (the 'simple file API')."""
import json
from pathlib import Path
from typing import List

import pandas as pd

from src.schemas.risk_signal import RiskSignal


def write_signals(signals: List[RiskSignal], out_dir) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jl = out / "risk_signals.jsonl"
    jl.write_text("".join(s.model_dump_json() + "\n" for s in signals), encoding="utf-8")
    pd.DataFrame([{**s.model_dump(mode="json")} for s in signals]).to_csv(
        out / "risk_signals.csv", index=False, encoding="utf-8"
    )
    return jl


def read_signals(path) -> List[RiskSignal]:
    return [RiskSignal(**json.loads(l)) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
