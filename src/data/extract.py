"""Database query helpers copied from market_report's data extract layer."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

DB_SERVICE_URL = os.getenv(
    "TS_FORECAST_DB_SERVICE_URL",
    "http://192.168.171.15:8123/query",
)


def get_df_by_sql(sql: str, limit: int | None = None) -> pd.DataFrame:
    """
    Execute a SQL query through the shared database query service.

    This mirrors market_report's `src.xinda.data.extract.get_df_by_sql` so
    ts-forecast can fetch data without importing the market_report repository.
    """
    logger.info("开始执行SQL查询，limit=%s", limit)
    logger.debug("SQL语句: %s", f"{sql[:200]}..." if len(sql) > 200 else sql)

    payload: dict[str, str | int] = {"sql": sql}
    if limit is not None:
        payload["limit"] = limit
        logger.info("已设置LIMIT限制: %s", limit)

    try:
        resp = requests.post(DB_SERVICE_URL, json=payload, timeout=60)
        resp.raise_for_status()

        try:
            result: dict[str, Any] = resp.json()
        except ValueError:
            logger.warning("标准JSON解析失败，尝试兼容解析包含NaN的数据库服务响应")
            result = json.loads(resp.text)

        df = pd.DataFrame(result["data"], columns=result["columns"])
        logger.info("SQL查询执行成功，返回%s行数据，列数: %s", len(df), len(df.columns))
        logger.debug("返回列名: %s", list(df.columns))
        return df
    except Exception as exc:
        logger.error("SQL查询执行失败: %s", exc, exc_info=True)
        raise
