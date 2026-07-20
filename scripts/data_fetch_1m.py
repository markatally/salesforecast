"""Fetch hospital-level daily transactions for 1m-jenny.

The defaults follow the "医院采购需求预测（Dupixent）" SQL in the handover
document. All filtering dimensions can be changed with command-line arguments.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


DEFAULT_PROJECT_CODE = "PR00003"
DEFAULT_PRODUCT_CODES = ("162", "169", "193")
DEFAULT_FROM_TYPES = ("经销商", "零售")
DEFAULT_TERMINAL_TYPE = "医院"
TABLE_SCOPES = (
    ("dm.dws_dg_ph_md_fact_sales_2022_2023", "202201", "202312"),
    ("dm.dws_dg_ph_md_fact_sales", "202401", None),
)
SARX_TABLE = "sarx_ads.ads_sarx_sales"
SARX_START_BIZYM = "202401"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "sales_1m_hospital_daily.csv"


def _quote_sql(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _in_sql(values: tuple[str, ...]) -> str:
    if not values:
        raise ValueError("At least one value is required for an IN filter.")
    return ", ".join(_quote_sql(value) for value in values)


def _build_scope_sql(
    table: str,
    bizym_start: str,
    bizym_end: str | None,
    project_code: str,
    product_codes: tuple[str, ...],
    from_types: tuple[str, ...],
    terminal_type: str,
) -> str:
    date_expression = "SUBSTRING(ddpmfs.transdate, 1, 10)"
    conditions = [
        f"ddpmfs.projectcode = {_quote_sql(project_code)}",
        f"ddpmfs.prodmdmcode IN ({_in_sql(product_codes)})",
        f"ddpmfs.frommdmtype1 IN ({_in_sql(from_types)})",
        f"ddpmfs.tomdmtype2 = {_quote_sql(terminal_type)}",
        f"ddpmfs.bizym >= {_quote_sql(bizym_start)}",
    ]
    if bizym_end:
        conditions.append(f"ddpmfs.bizym <= {_quote_sql(bizym_end)}")
    where_sql = "\n      AND ".join(conditions)
    return f"""
    SELECT
        ddpmfs.bizym, {date_expression} AS transdate,
        ddpmfs.tomdmcode, ddpmfs.toedivndrnm, ddpmfs.tomdphncode, ddpmfs.tomdphnname,
        ddpmfs.tomdmtype1, ddpmfs.tomdmtype2, ddpmfs.tomdmtype3,
        ddpmfs.tomdmprovince, ddpmfs.tomdmcity, ddpmfs.tomdmcounty, ddpmfs.tohospitallevel,
        SUM(ddpmfs.cnvrtdqty) AS qty
    FROM {table} ddpmfs
    WHERE {where_sql}
    GROUP BY ddpmfs.bizym, transdate,
        ddpmfs.tomdmcode, ddpmfs.toedivndrnm, ddpmfs.tomdphncode, ddpmfs.tomdphnname,
        ddpmfs.tomdmtype1, ddpmfs.tomdmtype2, ddpmfs.tomdmtype3,
        ddpmfs.tomdmprovince, ddpmfs.tomdmcity, ddpmfs.tomdmcounty, ddpmfs.tohospitallevel
    """.strip()


def _build_sarx_sql(
    bizym_start: str,
    bizym_end: str | None,
    product_codes: tuple[str, ...],
    from_types: tuple[str, ...],
    terminal_type: str,
) -> str:
    conditions = [
        f"sarx.prodmdmcode IN ({_in_sql(product_codes)})",
        f"sarx.frommdmtype1 IN ({_in_sql(from_types)})",
        f"sarx.tomdmtype2 = {_quote_sql(terminal_type)}",
        f"sarx.bizym >= {_quote_sql(bizym_start)}",
    ]
    if bizym_end:
        conditions.append(f"sarx.bizym <= {_quote_sql(bizym_end)}")
    where_sql = "\n      AND ".join(conditions)
    return f"""
    SELECT
        sarx.bizym, toString(sarx.transdate) AS transdate,
        sarx.tomdmcode, sarx.toedivndrnm, sarx.tomdphncode, sarx.tomdphnname,
        sarx.tomdmtype1, sarx.tomdmtype2, sarx.tomdmtype3,
        sarx.tomdmprovince, sarx.tomdmcity, sarx.tomdmcounty, sarx.tohospitallevel,
        SUM(sarx.cnvrtdqty) AS qty
    FROM {SARX_TABLE} sarx
    WHERE {where_sql}
    GROUP BY sarx.bizym, transdate,
        sarx.tomdmcode, sarx.toedivndrnm, sarx.tomdphncode, sarx.tomdphnname,
        sarx.tomdmtype1, sarx.tomdmtype2, sarx.tomdmtype3,
        sarx.tomdmprovince, sarx.tomdmcity, sarx.tomdmcounty, sarx.tohospitallevel
    """.strip()


def build_sql(
    project_code: str,
    product_codes: tuple[str, ...],
    from_types: tuple[str, ...],
    terminal_type: str,
    bizym_start: str,
    bizym_end: str | None,
    include_sarx: bool = False,
    limit: int | None = None,
) -> str:
    scopes = []
    for table, scope_start, scope_end in TABLE_SCOPES:
        start = max(bizym_start, scope_start)
        end = min(bizym_end, scope_end) if bizym_end and scope_end else bizym_end or scope_end
        if end and start > end:
            continue
        scopes.append(_build_scope_sql(table, start, end, project_code, product_codes, from_types, terminal_type))
    if include_sarx:
        sarx_start = max(bizym_start, SARX_START_BIZYM)
        if not bizym_end or sarx_start <= bizym_end:
            scopes.append(_build_sarx_sql(sarx_start, bizym_end, product_codes, from_types, terminal_type))
    if not scopes:
        raise ValueError("No source-table scope overlaps the requested bizym range.")
    sql = "SELECT *\nFROM (\n    " + "\n    UNION ALL\n    ".join(scopes) + "\n)\nORDER BY bizym, transdate, tomdphncode"
    return f"{sql}\nLIMIT {limit}" if limit is not None else sql


def _load_query_api():
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.data.extract import get_df_by_sql
    return get_df_by_sql


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch hospital-level daily input data for 1m-jenny.")
    parser.add_argument("--project-code", default=DEFAULT_PROJECT_CODE)
    parser.add_argument("--product-codes", nargs="+", default=DEFAULT_PRODUCT_CODES)
    parser.add_argument("--from-types", nargs="+", default=DEFAULT_FROM_TYPES)
    parser.add_argument("--terminal-type", default=DEFAULT_TERMINAL_TYPE)
    parser.add_argument("--bizym-start", default="202401")
    parser.add_argument("--bizym-end", default=None)
    parser.add_argument(
        "--include-sarx",
        action="store_true",
        help="Include the WD6 supplementary source when the query account has access.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--print-sql", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sql = build_sql(
        project_code=args.project_code,
        product_codes=tuple(args.product_codes),
        from_types=tuple(args.from_types),
        terminal_type=args.terminal_type,
        bizym_start=args.bizym_start,
        bizym_end=args.bizym_end,
        include_sarx=args.include_sarx,
        limit=1 if args.smoke else args.limit,
    )
    if args.print_sql:
        print(sql)
    df = _load_query_api()(sql=sql)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Saved {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
