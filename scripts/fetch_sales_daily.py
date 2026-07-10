"""
Fetch daily sales aggregation data through market_report's DB query API.

Smoke test:
    python scripts/fetch_sales_daily.py --smoke

Full extract:
    python scripts/fetch_sales_daily.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Query conditions: edit this block when the data scope changes.
# ---------------------------------------------------------------------------
PROJECT_CODE = "PR00759"
PROD_MDM_CODE = "11767002000200"
FROM_MDM_TYPE1 = "经销商"
TO_MDM_TYPE1 = "医疗机构"

TABLE_SCOPES = [
    {
        "table": "dm.dws_dg_ph_md_fact_sales_2022_2023",
        "bizym_start": "202201",
        "bizym_end": "202312",
    },
    {
        "table": "dm.dws_dg_ph_md_fact_sales",
        "bizym_start": "202401",
        "bizym_end": "202605",
    },
]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "sales_daily.csv"


def _quote_sql(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _build_scope_sql(table: str, bizym_start: str, bizym_end: str) -> str:
    return f"""
    SELECT
        ddpmfs.bizym,
        SUBSTRING(ddpmfs.transdate, 1, 10) AS transdate,
        COUNT(DISTINCT tomdphncode) AS num_hosp,
        SUM(ddpmfs.cnvrtdqty) AS qty,
        SUM(ddpmfs.taxamt) AS taxamt,
        AVG(ddpmfs.price) AS avg_price
    FROM {table} ddpmfs
    WHERE projectcode = {_quote_sql(PROJECT_CODE)}
      AND prodmdmcode = {_quote_sql(PROD_MDM_CODE)}
      AND bizym BETWEEN {_quote_sql(bizym_start)} AND {_quote_sql(bizym_end)}
      AND ddpmfs.frommdmtype1 = {_quote_sql(FROM_MDM_TYPE1)}
      AND ddpmfs.tomdmtype1 = {_quote_sql(TO_MDM_TYPE1)}
    GROUP BY bizym, transdate
    """.strip()


def build_sql(limit: int | None = None) -> str:
    scope_sql = "\n    UNION ALL\n    ".join(
        _build_scope_sql(
            table=scope["table"],
            bizym_start=scope["bizym_start"],
            bizym_end=scope["bizym_end"],
        )
        for scope in TABLE_SCOPES
    )
    sql = f"""
SELECT *
FROM (
    {scope_sql}
)
ORDER BY bizym, transdate
""".strip()
    if limit is not None:
        sql = f"{sql}\nLIMIT {limit}"
    return sql


def _load_market_report_query_api():
    project_root = str(PROJECT_ROOT)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from src.data.extract import get_df_by_sql

    return get_df_by_sql


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch daily sales aggregation data through the local DB query API."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"CSV output path. Defaults to {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional SQL row limit.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a smoke test with SQL LIMIT 1.",
    )
    parser.add_argument(
        "--print-sql",
        action="store_true",
        help="Print SQL before executing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    limit = 1 if args.smoke else args.limit
    sql = build_sql(limit=limit)

    if args.print_sql:
        print(sql)

    get_df_by_sql = _load_market_report_query_api()
    df = get_df_by_sql(sql=sql)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8-sig")

    print(f"Saved {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
