"""Small shared helpers (JSON, markdown tables, figure saving)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import config as C


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    return str(o)


def save_json(obj, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default, ensure_ascii=False)


def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def md_table(df: pd.DataFrame, index: bool = True, floatfmt: str = ".2f") -> str:
    """DataFrame -> markdown table (falls back to a code block without `tabulate`)."""
    try:
        return df.to_markdown(index=index, floatfmt=floatfmt)
    except ImportError:
        return "```\n" + df.to_string(index=index) + "\n```"


def savefig(fig, name: str) -> Path:
    import matplotlib.pyplot as plt

    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = C.FIGURES_DIR / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def rate_table(df: pd.DataFrame, col: str, target: str = C.TARGET_BIN) -> pd.DataFrame:
    """Placement (acceptance) rate in % and candidate count per group."""
    t = df.groupby(col, observed=True)[target].agg(rate="mean", n="size").reset_index()
    t["rate"] = t["rate"] * 100
    return t


def banner(msg: str) -> None:
    print(f"\n{'=' * 70}\n{msg}\n{'=' * 70}")
