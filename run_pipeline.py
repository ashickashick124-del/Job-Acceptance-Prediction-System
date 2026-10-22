"""Run the whole project end to end:

    python run_pipeline.py                 # everything except SQL
    python run_pipeline.py --sql           # + store in MySQL (needs .env)
    python run_pipeline.py --sqlite        # + store in a local SQLite file instead
    python run_pipeline.py --fast          # quicker model training (for a first test)
"""
from __future__ import annotations

import argparse
import time

from src import data_understanding, eda, features, generate_report, modeling, preprocessing, sql_loader


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fast", action="store_true", help="3-fold CV, fewer trees and search iterations")
    ap.add_argument("--no-tune", action="store_true", help="skip hyper-parameter tuning")
    ap.add_argument("--sql", action="store_true", help="load the analytical dataset into MySQL")
    ap.add_argument("--sqlite", action="store_true", help="load the analytical dataset into a local SQLite file")
    a = ap.parse_args()

    t0 = time.time()
    data_understanding.main()
    preprocessing.main()
    features.main()
    modeling.main(fast=a.fast, tune=not a.no_tune)
    eda.main()
    if a.sql or a.sqlite:
        sql_loader.main(sqlite=a.sqlite and not a.sql)
    generate_report.main()
    print(f"\nDone in {time.time() - t0:.0f}s.  Next:  streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
