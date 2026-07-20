"""Check that the shared database query interface is reachable.

The check uses the same ``TS_FORECAST_DB_SERVICE_URL`` endpoint as
``fetch_sales_daily.py`` and executes a read-only ``SELECT 1`` query.  A
healthy interface must return HTTP 200.

Usage:
    python scripts/data_interface_health_check.py
    TS_FORECAST_DB_SERVICE_URL=http://host:8123/query \
        python scripts/data_interface_health_check.py
"""

from __future__ import annotations

import argparse
import os
import sys

import requests


DEFAULT_SERVICE_URL = "http://192.168.171.15:8123/query"
HEALTH_CHECK_SQL = "SELECT 1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check the shared database query interface with a read-only query."
    )
    parser.add_argument(
        "--url",
        default=os.getenv("TS_FORECAST_DB_SERVICE_URL", DEFAULT_SERVICE_URL),
        help="Query-service URL; defaults to TS_FORECAST_DB_SERVICE_URL.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10,
        help="Request timeout in seconds (default: 10).",
    )
    return parser.parse_args()


def check_health(url: str, timeout: float) -> int:
    """Return the HTTP status from the query service after a read-only probe."""
    response = requests.post(url, json={"sql": HEALTH_CHECK_SQL}, timeout=timeout)
    return response.status_code


def main() -> None:
    args = parse_args()
    try:
        status_code = check_health(args.url, args.timeout)
    except requests.RequestException as exc:
        print(f"Health check failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    if status_code != requests.codes.ok:
        print(
            f"Health check failed: expected HTTP 200, received HTTP {status_code}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    print(f"Health check passed: HTTP {status_code} ({args.url})")


if __name__ == "__main__":
    main()
