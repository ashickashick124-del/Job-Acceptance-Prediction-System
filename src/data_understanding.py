"""Step 2: Data understanding - shape, dtypes, samples, null distribution, raw label problems."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src import config as C
from src.preprocessing import load_raw
from src.utils import banner, md_table


def main() -> None:
    banner("STEP 1 - Data understanding")
    C.ensure_dirs()
    df = load_raw()
    lines: list[str] = ["# Data Understanding (raw dataset)\n"]

    lines.append(f"**Source file:** `{C.RAW_CSV}`  \n**Shape:** {df.shape[0]:,} rows x {df.shape[1]} columns  \n"
                 f"**Memory:** {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB  \n"
                 f"**Exact duplicate rows:** {int(df.duplicated().sum()):,}\n")

    dtypes = pd.DataFrame({"dtype": df.dtypes.astype(str), "non_null": df.notna().sum(),
                           "nulls": df.isna().sum(), "null_%": (df.isna().mean() * 100).round(2),
                           "unique": df.nunique()})
    lines += ["## Data types & null distribution\n", md_table(dtypes), "\n"]

    lines += ["## Sample records\n", md_table(df.head(5), index=False), "\n"]

    num = df[C.NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")
    lines += ["## Numeric summary (raw)\n", md_table(num.describe().T), "\n"]

    lines.append("## Raw categorical values (repr shows hidden spaces / case variants)\n")
    for col in [*C.CATEGORICAL_COLUMNS, C.TARGET_COL]:
        vc = df[col].map(lambda v: repr(v) if pd.notna(v) else "<missing>").value_counts()
        lines.append(f"**{col}**: " + ", ".join(f"{k} x{v:,}" for k, v in vc.items()) + "\n")

    # Where does placement start as the technical score grows? (helps choose an eligibility threshold)
    tmp = pd.DataFrame({"tech": pd.to_numeric(df["technical_score"], errors="coerce"),
                        "placed": df[C.TARGET_COL].astype(str).str.strip().str.lower().eq("placed")})
    tmp = tmp.dropna(subset=["tech"])
    tmp["technical_band"] = pd.cut(tmp["tech"], [0, 40, 45, 50, 55, 60, 70, 80, 90, 100], include_lowest=True)
    band = tmp.groupby("technical_band", observed=True)["placed"].agg(placement_rate="mean", n="size")
    band["placement_rate"] = (band["placement_rate"] * 100).round(1)
    lines += ["## Placement rate by technical-score band (raw)\n", md_table(band), "\n"]

    C.DATA_UNDERSTANDING_MD.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:6]))
    print(f"\nFull report -> {C.DATA_UNDERSTANDING_MD.relative_to(C.ROOT)}")


if __name__ == "__main__":
    main()
