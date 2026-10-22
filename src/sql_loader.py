"""Step 6 (optional): store the analytical dataset in MySQL via SQLAlchemy.

    python -m src.sql_loader            -> MySQL (settings from .env)
    python -m src.sql_loader --sqlite   -> local SQLite file, no server needed (good for a quick test)
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL, Engine

from src import config as C
from src.utils import banner

load_dotenv(C.ROOT / ".env")


def mysql_settings() -> dict:
    return {
        "host": os.getenv("MYSQL_HOST", "localhost"),
        "port": int(os.getenv("MYSQL_PORT", "3306")),
        "user": os.getenv("MYSQL_USER", "root"),
        "password": os.getenv("MYSQL_PASSWORD", ""),
        "database": os.getenv("MYSQL_DATABASE", "hr_placement"),
    }


def mysql_engine(with_database: bool = True) -> Engine:
    s = mysql_settings()
    url = URL.create("mysql+pymysql", username=s["user"], password=s["password"], host=s["host"],
                     port=s["port"], database=s["database"] if with_database else None,
                     query={"charset": "utf8mb4"})
    return create_engine(url, pool_pre_ping=True)


def sqlite_engine() -> Engine:
    return create_engine(f"sqlite:///{C.SQLITE_DB.as_posix()}")


def _statements(path: Path) -> list[str]:
    sql = re.sub(r"--[^\n]*", "", path.read_text(encoding="utf-8"))
    return [s.strip() for s in sql.split(";") if s.strip()]


def load_mysql(df: pd.DataFrame) -> Engine:
    db = mysql_settings()["database"]
    if not re.fullmatch(r"[A-Za-z0-9_]+", db):
        raise ValueError("MYSQL_DATABASE may only contain letters, digits and underscores")
    with mysql_engine(with_database=False).begin() as conn:
        conn.execute(text(f"CREATE DATABASE IF NOT EXISTS `{db}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"))
    engine = mysql_engine()
    with engine.begin() as conn:
        for stmt in _statements(C.SQL_DIR / "schema.sql"):
            if stmt.upper().startswith(("CREATE DATABASE", "USE ")):
                continue  # database was created above
            conn.execute(text(stmt))
    df.to_sql(C.TABLE_NAME, engine, if_exists="append", index=False, chunksize=2000, method="multi")
    return engine


def load_sqlite(df: pd.DataFrame) -> Engine:
    engine = sqlite_engine()
    df.to_sql(C.TABLE_NAME, engine, if_exists="replace", index=False, chunksize=2000)
    return engine


def run_analysis_queries(engine: Engine, show: int = 8) -> None:
    text_ = (C.SQL_DIR / "analysis_queries.sql").read_text(encoding="utf-8")
    for block in re.split(r"(?m)^-- name:\s*", text_)[1:]:
        title, _, body = block.partition("\n")
        query = re.sub(r"--[^\n]*", "", body).strip().rstrip(";")
        if not query:
            continue
        res = pd.read_sql(text(query), engine)
        print(f"\n>> {title.strip()}\n{res.head(show).to_string(index=False)}")


def main(sqlite: bool = False, queries: bool = True) -> None:
    banner("STEP 6 - SQL storage (" + ("SQLite" if sqlite else "MySQL") + ")")
    if not C.ANALYTICAL_CSV.exists():
        raise FileNotFoundError("Run preprocessing / modelling first: python run_pipeline.py")
    df = pd.read_csv(C.ANALYTICAL_CSV)
    try:
        engine = load_sqlite(df) if sqlite else load_mysql(df)
    except Exception as exc:  # noqa: BLE001 - give beginners a readable hint
        if sqlite:
            raise
        s = mysql_settings()
        raise SystemExit(
            f"\nCould not load into MySQL at {s['host']}:{s['port']} as '{s['user']}'.\n"
            f"Reason: {type(exc).__name__}: {exc}\n\n"
            "Checklist: 1) is the MySQL server running?  2) did you copy .env.example to .env and set "
            "MYSQL_USER / MYSQL_PASSWORD?  3) does the user have CREATE privileges?\n"
            "To test the rest of the project without MySQL run:  python -m src.sql_loader --sqlite"
        ) from exc
    with engine.connect() as conn:
        n = conn.execute(text(f"SELECT COUNT(*) FROM {C.TABLE_NAME}")).scalar()
    where = C.SQLITE_DB.relative_to(C.ROOT) if sqlite else f"{mysql_settings()['database']}.{C.TABLE_NAME}"
    print(f"Loaded {n:,} rows into {where}")
    if queries:
        run_analysis_queries(engine)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sqlite", action="store_true", help="use a local SQLite file instead of MySQL")
    ap.add_argument("--no-queries", action="store_true", help="skip running the sample reporting queries")
    a = ap.parse_args()
    main(sqlite=a.sqlite, queries=not a.no_queries)
