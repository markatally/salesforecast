"""
30d Jenny anchor rhythm and statistical signal dashboard.

Run:
    streamlit run code/30d-jenny/streamlit/anchor_signal_dashboard.py
"""

from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats

try:
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
    from statsmodels.stats.multitest import multipletests
    from statsmodels.tsa.stattools import adfuller, kpss
except Exception:  # pragma: no cover - dashboard degrades gracefully.
    sm = None
    acorr_ljungbox = None
    het_arch = None
    multipletests = None
    adfuller = None
    kpss = None

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DAILY_PATH = REPO_ROOT / "data" / "sales_30d_daily.csv"
MONTH_ORDER = list(range(1, 13))
DEFAULT_ANCHORS = [10, 5]
DEFAULT_NATURAL_DAY_WINDOWS = [5, 10]
DEFAULT_ROLLING_LOOKBACKS = [3, 6]

COLOR_MAIN = "#2f6f73"
COLOR_WARN = "#d95f02"
COLOR_DANGER = "#c44e52"
COLOR_BLUE = "#4c72b0"
COLOR_PURPLE = "#8172b3"


@dataclass(frozen=True)
class WorkdayCalendar:
    source: str
    is_workday: Callable[[pd.Timestamp], bool]


def get_workday_calendar() -> WorkdayCalendar:
    try:
        from chinese_calendar import is_workday as cn_is_workday

        def is_business_workday(ts: pd.Timestamp) -> bool:
            return bool(cn_is_workday(pd.Timestamp(ts).date()))

        return WorkdayCalendar("chinese_calendar", is_business_workday)
    except Exception:

        def is_business_workday(ts: pd.Timestamp) -> bool:
            return pd.Timestamp(ts).weekday() < 5

        return WorkdayCalendar("weekday_fallback_Mon_Fri", is_business_workday)


def fmt_num(x: float | int | None) -> str:
    if pd.isna(x):
        return "-"
    return f"{x:,.0f}"


def fmt_pct(x: float | int | None) -> str:
    if pd.isna(x):
        return "-"
    return f"{x:.1%}"


def parse_yyyymm(value: str | int | None, name: str) -> tuple[int | None, pd.Timestamp | None]:
    if value is None or str(value).strip() == "":
        return None, None
    value = str(value).strip()
    if len(value) != 6 or not value.isdigit():
        raise ValueError(f"{name} must be blank or YYYYMM, got {value!r}")
    ts = pd.to_datetime(value + "01", format="%Y%m%d")
    return int(value), ts


def safe_divide(numerator: float, denominator: float) -> float:
    if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
        return np.nan
    return float(numerator) / float(denominator)


def safe_pct_change(curr: float, prev: float) -> float:
    ratio = safe_divide(curr, prev)
    return ratio - 1 if pd.notna(ratio) else np.nan


def linear_slope(y: pd.Series | np.ndarray | list[float]) -> float:
    y = pd.Series(y).dropna().astype(float)
    if len(y) < 2:
        return np.nan
    x = np.arange(1, len(y) + 1)
    return float(np.polyfit(x, y.values, 1)[0])


def robust_mad_zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    med = s.median()
    mad = (s - med).abs().median()
    if pd.isna(mad) or mad == 0:
        return pd.Series(np.nan, index=s.index)
    return 0.6745 * (s - med) / mad


def robust_zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    std = s.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.nan, index=s.index)
    return (s - s.mean()) / std


def download_button(df: pd.DataFrame, label: str, filename: str) -> None:
    st.download_button(
        label,
        df.to_csv(index=False).encode("utf-8-sig"),
        filename,
        mime="text/csv",
        use_container_width=True,
    )


def count_calendar_days(month_start: pd.Timestamp, is_workday: Callable[[pd.Timestamp], bool]) -> pd.Series:
    month_start = pd.Timestamp(month_start)
    dates = pd.Series(pd.date_range(month_start, month_start + pd.offsets.MonthEnd(0), freq="D"))
    flags = dates.map(is_workday)
    return pd.Series(
        {
            "weekend_days": int((dates.dt.dayofweek >= 5).sum()),
            "workdays": int(flags.sum()),
            "holiday_days": int((~flags & (dates.dt.dayofweek < 5)).sum()),
            "non_workdays": int((~flags).sum()),
        }
    )


def summarize_date_window(
    dates: pd.DatetimeIndex,
    prefix: str,
    is_workday: Callable[[pd.Timestamp], bool],
) -> dict[str, int]:
    date_series = pd.Series(pd.to_datetime(dates))
    flags = date_series.map(is_workday)
    weekday = date_series.dt.dayofweek
    result = {
        f"{prefix}_calendar_days": int(len(date_series)),
        f"{prefix}_workdays": int(flags.sum()),
        f"{prefix}_non_workdays": int((~flags).sum()),
        f"{prefix}_holiday_days": int((~flags & (weekday < 5)).sum()),
        f"{prefix}_weekend_days": int((weekday >= 5).sum()),
    }
    for weekday_idx, weekday_name in enumerate(["mon", "tue", "wed", "thu", "fri", "sat", "sun"]):
        result[f"{prefix}_{weekday_name}_days"] = int((weekday == weekday_idx).sum())
    return result


def get_mid_workday_window(g: pd.DataFrame, window: int) -> pd.DataFrame:
    if g.empty:
        return g
    max_workday_seq = int(g["max_workday_seq"].iloc[0])
    mid_start_seq = max(1, int(np.floor((max_workday_seq - window) / 2)) + 1)
    mid_end_seq = min(max_workday_seq, mid_start_seq + window - 1)
    return g[g["workday_seq"].between(mid_start_seq, mid_end_seq)].copy()


def metric_col_to_cn(col: str) -> str:
    fixed_cols = {
        "year_month": "年月",
        "bizym": "业务月份",
        "year": "年份",
        "month": "月份",
        "month_start": "月份开始",
        "month_total": "月销量",
        "month_total_mom_pct": "月销量环比",
        "yoy_pct": "月销量同比",
        "month_total_roll3": "月销量3月滚动均值",
        "month_total_roll6": "月销量6月滚动均值",
        "prev_month_weekend_days": "上月周末天数",
        "prev_month_workdays": "上月工作天数",
        "prev_month_holiday_days": "上月节假日天数",
        "prev_month_non_workdays": "上月非工作天数",
        "curr_month_weekend_days": "当月周末天数",
        "curr_month_workdays": "当月工作天数",
        "curr_month_holiday_days": "当月节假日天数",
        "curr_month_non_workdays": "当月非工作天数",
        "next_month_weekend_days": "下月周末天数",
        "next_month_workdays": "下月工作天数",
        "next_month_holiday_days": "下月节假日天数",
        "next_month_non_workdays": "下月非工作天数",
    }
    if col in fixed_cols:
        return fixed_cols[col]

    m = re.fullmatch(r"month_(start|mid|end)_(\d+)wd_(qty|avg_qty|std_qty|cv_qty|min_qty|max_qty|max_min_ratio)", col)
    if m:
        pos = {"start": "月初", "mid": "月中", "end": "月末"}[m.group(1)]
        name = {
            "qty": "销量",
            "avg_qty": "日均销量",
            "std_qty": "销量标准差",
            "cv_qty": "销量变异系数",
            "min_qty": "最低日销量",
            "max_qty": "最高日销量",
            "max_min_ratio": "最高/最低比",
        }[m.group(3)]
        return f"{pos}{m.group(2)}工作日{name}"

    m = re.fullmatch(r"month_(start|mid|end)_(\d+)wd_(qty_slope|qty_slope_per_avg|cumsum_qty_slope)", col)
    if m:
        pos = {"start": "月初", "mid": "月中", "end": "月末"}[m.group(1)]
        name = {
            "qty_slope": "日销量斜率",
            "qty_slope_per_avg": "日销量斜率/均值",
            "cumsum_qty_slope": "累计销量斜率",
        }[m.group(3)]
        return f"{pos}{m.group(2)}工作日{name}"

    m = re.fullmatch(r"month_(start|mid|end)_(\d+)wd_qty_contrib_pct", col)
    if m:
        pos = {"start": "月初", "mid": "月中", "end": "月末"}[m.group(1)]
        return f"{pos}{m.group(2)}工作日销量贡献占比"

    m = re.fullmatch(r"month_(end|mid)_minus_(start|mid)_(\d+)wd_qty_contrib_pct", col)
    if m:
        left = {"end": "月末", "mid": "月中"}[m.group(1)]
        right = {"start": "月初", "mid": "月中"}[m.group(2)]
        return f"{left}{m.group(3)}工作日贡献占比 - {right}{m.group(3)}工作日贡献占比"

    m = re.fullmatch(r"month_(start|end)_(\d+)wd_(first|last)_day_of_month", col)
    if m:
        pos = "月初" if m.group(1) == "start" else "月末"
        edge = "首个" if m.group(3) == "first" else "最后一个"
        return f"{pos}{m.group(2)}工作日{edge}工作日是几号"

    m = re.fullmatch(r"month_(start|mid|end)_(\d+)wd_(calendar_span|non_workday_gap)_days", col)
    if m:
        pos = {"start": "月初", "mid": "月中", "end": "月末"}[m.group(1)]
        name = {"calendar_span": "跨自然日", "non_workday_gap": "夹杂非工作日"}[m.group(3)]
        return f"{pos}{m.group(2)}工作日{name}天数"

    m = re.fullmatch(r"month_start_(\d+)wd_(first1|first2|last2)_share_in_window", col)
    if m:
        name = {"first1": "首1日", "first2": "前2日", "last2": "后2日"}[m.group(2)]
        return f"月初{m.group(1)}工作日窗口内{name}销量占比"

    m = re.fullmatch(r"month_(start|mid)_(\d+)wd_(.+)_(yoy_pct|vs_prev(\d+)m_avg_pct|mom_pct)", col)
    if m:
        base = metric_col_to_cn(f"month_{m.group(1)}_{m.group(2)}wd_{m.group(3)}")
        if m.group(4) == "yoy_pct":
            return f"{base}同比"
        if m.group(4) == "mom_pct":
            return f"{base}较上月同期环比"
        return f"{base}较前{m.group(5)}月均值偏离"

    m = re.fullmatch(r"month_mid_minus_start_(\d+)wd_qty_pct", col)
    if m:
        return f"月中{m.group(1)}工作日销量较月初{m.group(1)}工作日变化率"

    m = re.fullmatch(r"month_start_minus_prev_month_mid_(\d+)wd_qty_pct", col)
    if m:
        return f"月初{m.group(1)}工作日销量较上月中{m.group(1)}工作日变化率"

    m = re.fullmatch(r"(prev_month|prev_year_same_month)_(.+)", col)
    if m:
        prefix = "上月" if m.group(1) == "prev_month" else "去年同月"
        return f"{prefix}{metric_col_to_cn(m.group(2))}"

    m = re.fullmatch(r"(next|prev)_month_(start|end)_(\d+)natural_days_holiday_days", col)
    if m:
        month_pos = "下月" if m.group(1) == "next" else "上月"
        edge_pos = "初" if m.group(2) == "start" else "末"
        return f"{month_pos}{edge_pos}{m.group(3)}自然日内非工作日天数"

    m = re.fullmatch(r"curr_month_(start|end)_(\d+)natural_days_(calendar_days|workdays|non_workdays|holiday_days|weekend_days)", col)
    if m:
        edge = "初" if m.group(1) == "start" else "末"
        name = {
            "calendar_days": "自然日天数",
            "workdays": "工作日天数",
            "non_workdays": "非工作日天数",
            "holiday_days": "节假日天数",
            "weekend_days": "周末天数",
        }[m.group(3)]
        return f"当月{edge}{m.group(2)}自然日{name}"

    m = re.fullmatch(r"curr_month_(start|end)_(\d+)natural_days_(mon|tue|wed|thu|fri|sat|sun)_days", col)
    if m:
        edge = "初" if m.group(1) == "start" else "末"
        weekday_map = {"mon": "周一", "tue": "周二", "wed": "周三", "thu": "周四", "fri": "周五", "sat": "周六", "sun": "周日"}
        return f"当月{edge}{m.group(2)}自然日{weekday_map[m.group(3)]}天数"

    m = re.fullmatch(r"anchor_d(\d+)_(day_of_month|weekday|workday_seq|remaining_workdays_after_anchor|remaining_calendar_days_after_anchor|month_elapsed_workday_pct|month_elapsed_calendar_pct)", col)
    if m:
        name = {
            "day_of_month": "日期号",
            "weekday": "星期索引",
            "workday_seq": "当月第几个工作日",
            "remaining_workdays_after_anchor": "anchor后剩余工作日",
            "remaining_calendar_days_after_anchor": "anchor后剩余自然日",
            "month_elapsed_workday_pct": "工作日进度",
            "month_elapsed_calendar_pct": "自然日进度",
        }[m.group(2)]
        return f"D-{m.group(1)} anchor {name}"

    return col


def feature_group(col: str) -> str:
    patterns = [
        (
            "月初已发生销量形态",
            r"^month_start_\d+wd_(qty|avg_qty|std_qty|cv_qty|min_qty|max_qty|max_min_ratio|qty_slope|qty_slope_per_avg|cumsum_qty_slope|first\d_share_in_window|last\d_share_in_window|calendar_span_days|first_day_of_month|last_day_of_month|non_workday_gap_days|.*mom_pct|.*yoy_pct|.*vs_prev\d+m_avg_pct)$",
        ),
        ("月中销量形态（D-5优先）", r"^(month_mid_\d+wd_|month_mid_minus_start_\d+wd_qty_pct|month_start_minus_prev_month_mid_\d+wd_qty_pct).+"),
        ("当月/前后月日历结构", r"^(prev_month|curr_month|next_month|anchor_d\d+|prev_month_end_|next_month_start_).+"),
        ("历史已知贡献节奏", r"^(prev_month|prev_year_same_month)_month_(start|mid|end|end_minus_start|mid_minus_start|end_minus_mid).+"),
    ]
    for label, pattern in patterns:
        if re.search(pattern, col):
            return label
    return "其它前置特征"


@st.cache_data(show_spinner="加载并整理日销量数据...")
def load_and_prepare_data(path_str: str, start_yyyymm: str, end_yyyymm: str) -> dict[str, object]:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(path)

    calendar = get_workday_calendar()
    df_raw = pd.read_csv(path)
    required_cols = {"bizym", "transdate", "qty"}
    missing = required_cols - set(df_raw.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df_raw.copy()
    df["transdate"] = pd.to_datetime(df["transdate"])
    df["bizym"] = df["bizym"].astype(int)
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0.0)
    if "num_hosp" not in df.columns:
        df["num_hosp"] = np.nan

    start_bizym, start_month = parse_yyyymm(start_yyyymm, "START_YYYYMM")
    end_bizym, end_month = parse_yyyymm(end_yyyymm, "END_YYYYMM")
    if start_bizym is not None:
        df = df[df["bizym"] >= start_bizym]
    if end_bizym is not None:
        df = df[df["bizym"] <= end_bizym]
    if df.empty:
        raise ValueError("No rows remain after applying the YYYYMM range.")

    df["year"] = df["bizym"] // 100
    df["month"] = df["bizym"] % 100
    df["day_of_month"] = df["transdate"].dt.day
    df["month_start"] = df["transdate"].dt.to_period("M").dt.to_timestamp()
    df = df.sort_values(["transdate", "bizym"]).reset_index(drop=True)

    month_range_start = start_month if start_month is not None else df["month_start"].min()
    month_range_end = end_month if end_month is not None else df["month_start"].max()
    complete_months = pd.period_range(month_range_start, month_range_end, freq="M")
    complete_month_index = pd.DataFrame({"month_start": complete_months.to_timestamp()})
    complete_month_index["bizym"] = complete_month_index["month_start"].dt.strftime("%Y%m").astype(int)
    complete_month_index["year"] = complete_month_index["bizym"] // 100
    complete_month_index["month"] = complete_month_index["bizym"] % 100

    monthly_observed = (
        df.groupby(["bizym", "year", "month", "month_start"], as_index=False)
        .agg(
            date_min=("transdate", "min"),
            date_max=("transdate", "max"),
            observed_days=("transdate", "nunique"),
            month_total=("qty", "sum"),
            month_num_hosp_total=("num_hosp", "sum"),
            qty_missing=("qty", lambda s: int(s.isna().sum())),
            qty_negative=("qty", lambda s: int((s < 0).sum())),
        )
    )
    monthly = complete_month_index.merge(
        monthly_observed, on=["bizym", "year", "month", "month_start"], how="left"
    ).sort_values("month_start")
    monthly["observed_days"] = monthly["observed_days"].fillna(0).astype(int)
    monthly["calendar_days"] = monthly["month_start"].dt.days_in_month
    monthly["missing_calendar_days"] = monthly["calendar_days"] - monthly["observed_days"]
    monthly["is_complete_month"] = monthly["missing_calendar_days"].eq(0)
    monthly["month_label"] = monthly["month_start"].dt.strftime("%Y-%m")
    monthly["month_total"] = monthly["month_total"].fillna(0.0)
    monthly["prev_month_total"] = monthly["month_total"].shift(1)
    monthly["month_total_mom_pct"] = monthly["month_total"].pct_change()
    prev_year_lookup = monthly.set_index("bizym")["month_total"].to_dict()
    prev_year_bizym = (monthly["year"] - 1) * 100 + monthly["month"]
    monthly["prev_year_total"] = [prev_year_lookup.get(x, np.nan) for x in prev_year_bizym]
    monthly["yoy_pct"] = [
        safe_pct_change(curr, prev) for curr, prev in zip(monthly["month_total"], monthly["prev_year_total"])
    ]
    monthly["month_total_roll3"] = monthly["month_total"].rolling(3, min_periods=2).mean()
    monthly["month_total_roll6"] = monthly["month_total"].rolling(6, min_periods=3).mean()
    monthly["month_total_z"] = robust_zscore(monthly["month_total"])
    monthly["month_total_mad_z"] = robust_mad_zscore(monthly["month_total"])

    daily = df.copy()
    daily["month_label"] = daily["month_start"].dt.strftime("%Y-%m")
    daily["is_workday"] = daily["transdate"].map(calendar.is_workday)
    daily["workday_seq"] = daily.groupby("bizym")["is_workday"].cumsum().where(daily["is_workday"], np.nan)
    month_workdays = daily.groupby("bizym", as_index=False).agg(max_workday_seq=("workday_seq", "max"))
    daily = daily.merge(month_workdays, on="bizym", how="left")
    daily["workdays_to_month_end"] = daily["max_workday_seq"] - daily["workday_seq"]
    daily["mtd_qty"] = daily.groupby("bizym")["qty"].cumsum()
    daily = daily.merge(monthly[["bizym", "month_total"]], on="bizym", how="left")
    daily["mtd_pct"] = daily["mtd_qty"] / daily["month_total"].replace(0, np.nan)
    daily["days_to_month_end"] = daily["transdate"].dt.days_in_month - daily["day_of_month"]

    return {
        "raw": df_raw,
        "daily": daily,
        "monthly": monthly,
        "complete_month_index": complete_month_index,
        "workday_source": calendar.source,
    }


@st.cache_data(show_spinner="计算 anchor 节奏...")
def compute_anchor_tables(daily: pd.DataFrame, anchors: list[int], low_qty_filter: bool, low_qty_quantile: float) -> dict[str, pd.DataFrame]:
    workday_qty = daily[daily["is_workday"]].copy()
    positive_qty = workday_qty.loc[workday_qty["qty"] > 0, "qty"]
    low_threshold = float(positive_qty.quantile(low_qty_quantile)) if len(positive_qty) else 0.0
    workday_qty["is_low_qty_day"] = False
    if low_qty_filter:
        workday_qty["is_low_qty_day"] = workday_qty["qty"].le(low_threshold)
    workday_filtered = workday_qty[~workday_qty["is_low_qty_day"]].copy()

    anchor_rows = daily[
        daily["is_workday"] & daily["workdays_to_month_end"].isin([offset - 1 for offset in anchors])
    ].copy()
    anchor_rows["forecast_offset"] = anchor_rows["workdays_to_month_end"].astype(int) + 1
    anchor_rows = anchor_rows.sort_values(["forecast_offset", "month_start"])

    anchor_stability = (
        anchor_rows.groupby(["month", "forecast_offset"], as_index=False)
        .agg(
            anchor_mtd_pct_mean=("mtd_pct", "mean"),
            anchor_mtd_pct_std=("mtd_pct", "std"),
            anchor_months=("bizym", "nunique"),
            min_mtd_pct=("mtd_pct", "min"),
            max_mtd_pct=("mtd_pct", "max"),
        )
        .sort_values(["forecast_offset", "anchor_mtd_pct_std"], ascending=[True, False])
    )
    anchor_stability["range_mtd_pct"] = anchor_stability["max_mtd_pct"] - anchor_stability["min_mtd_pct"]
    anchor_stability["stability_flag"] = np.where(
        anchor_stability["anchor_mtd_pct_std"].fillna(0) >= 0.04,
        "高波动",
        np.where(anchor_stability["anchor_mtd_pct_std"].fillna(0) >= 0.025, "中波动", "较稳定"),
    )

    workday_filtered["workday_seq"] = workday_filtered["workday_seq"].astype(int)
    workday_filtered = workday_filtered.sort_values(["bizym", "workday_seq"]).copy()
    workday_filtered["workday_cumsum_qty"] = workday_filtered.groupby("bizym")["qty"].cumsum()
    workday_filtered["workday_final_qty"] = workday_filtered.groupby("bizym")["workday_cumsum_qty"].transform("max")
    workday_filtered["workday_cumsum_pct"] = workday_filtered["workday_cumsum_qty"] / workday_filtered["workday_final_qty"].replace(0, np.nan)

    return {
        "workday_qty": workday_qty,
        "workday_filtered": workday_filtered,
        "anchor_rows": anchor_rows,
        "anchor_stability": anchor_stability,
        "low_qty_threshold": pd.DataFrame({"low_qty_threshold": [low_threshold]}),
    }


@st.cache_data(show_spinner="构建 Block 8 泄露安全特征...")
def compute_feature_tables(
    daily: pd.DataFrame,
    monthly: pd.DataFrame,
    window: int,
    anchors: list[int],
    low_qty_filter: bool,
    low_qty_quantile: float,
) -> dict[str, pd.DataFrame | list[str]]:
    calendar = get_workday_calendar()
    anchor_tables = compute_anchor_tables(daily, anchors, low_qty_filter, low_qty_quantile)
    metric_base = anchor_tables["workday_filtered"].copy()
    metric_base = metric_base.sort_values(["bizym", "workday_seq"]).reset_index(drop=True)
    if metric_base.empty:
        return {
            "month_feature_detail": pd.DataFrame(),
            "anchor_safe_feature_cols": [],
            "anchor_d5_safe_extra_feature_cols": [],
            "leakage_feature_cols": [],
            "target_cols": [],
        }

    metric_base["workday_seq"] = metric_base["workday_seq"].astype(int)
    metric_base["reverse_workday_seq"] = metric_base["max_workday_seq"].astype(int) - metric_base["workday_seq"] + 1
    metric_base["month_workday_total_qty"] = metric_base.groupby("bizym")["qty"].transform("sum")
    metric_base["cumsum_qty"] = metric_base.groupby("bizym")["qty"].cumsum()

    month_start_window_qty_lookup = {}
    month_start_window_avg_lookup = {}
    month_end_window_qty_lookup = {}
    start_window = metric_base.sort_values(["bizym", "workday_seq"]).groupby("bizym").head(window)
    end_window = metric_base.sort_values(["bizym", "workday_seq"]).groupby("bizym").tail(window)
    month_start_window_qty_lookup[window] = start_window.groupby("bizym")["qty"].sum().to_dict()
    month_start_window_avg_lookup[window] = start_window.groupby("bizym")["qty"].mean().to_dict()
    month_end_window_qty_lookup[window] = end_window.groupby("bizym")["qty"].sum().to_dict()

    rows = []
    for bizym, g in metric_base.groupby("bizym"):
        g = g.sort_values("workday_seq").copy()
        row = {
            "bizym": int(bizym),
            "year": int(g["year"].iloc[0]),
            "month": int(g["month"].iloc[0]),
            "month_start": g["month_start"].iloc[0],
        }
        total_qty = float(g["month_workday_total_qty"].iloc[0])
        curr_month_start = pd.Timestamp(g["month_start"].iloc[0])
        prev_bizym = int((curr_month_start - pd.DateOffset(months=1)).strftime("%Y%m"))

        head = g.head(window)
        tail = g.tail(window)
        mid = get_mid_workday_window(g, window)
        head_qty = head["qty"].astype(float)
        tail_qty = tail["qty"].astype(float)
        mid_qty = mid["qty"].astype(float)

        curr_start_qty = head_qty.sum()
        curr_start_avg_qty = head_qty.mean()
        curr_start_std_qty = head_qty.std(ddof=0)
        curr_start_slope = linear_slope(head_qty)
        prev_start_qty = month_start_window_qty_lookup[window].get(prev_bizym, np.nan)
        prev_start_avg_qty = month_start_window_avg_lookup[window].get(prev_bizym, np.nan)

        row[f"month_start_{window}wd_qty"] = curr_start_qty
        row[f"month_start_{window}wd_avg_qty"] = curr_start_avg_qty
        row[f"month_start_{window}wd_std_qty"] = curr_start_std_qty
        row[f"month_start_{window}wd_cv_qty"] = safe_divide(curr_start_std_qty, curr_start_avg_qty)
        row[f"month_start_{window}wd_min_qty"] = head_qty.min()
        row[f"month_start_{window}wd_max_qty"] = head_qty.max()
        row[f"month_start_{window}wd_max_min_ratio"] = safe_divide(head_qty.max(), head_qty.min())
        row[f"month_start_{window}wd_qty_slope"] = curr_start_slope
        row[f"month_start_{window}wd_qty_slope_per_avg"] = safe_divide(curr_start_slope, curr_start_avg_qty)
        row[f"month_start_{window}wd_qty_mom_pct"] = safe_pct_change(curr_start_qty, prev_start_qty)
        row[f"month_start_{window}wd_avg_qty_mom_pct"] = safe_pct_change(curr_start_avg_qty, prev_start_avg_qty)
        row[f"month_start_{window}wd_qty_contrib_pct"] = safe_divide(curr_start_qty, total_qty)
        row[f"month_start_{window}wd_cumsum_qty_slope"] = linear_slope(head["cumsum_qty"])
        row[f"month_start_{window}wd_first1_share_in_window"] = safe_divide(head_qty.iloc[0], curr_start_qty) if len(head_qty) >= 1 else np.nan
        row[f"month_start_{window}wd_first2_share_in_window"] = safe_divide(head_qty.head(2).sum(), curr_start_qty) if len(head_qty) >= 2 else np.nan
        row[f"month_start_{window}wd_last2_share_in_window"] = safe_divide(head_qty.tail(2).sum(), curr_start_qty) if len(head_qty) >= 2 else np.nan
        row[f"month_start_{window}wd_calendar_span_days"] = int((head["transdate"].max() - head["transdate"].min()).days + 1) if len(head) else np.nan
        row[f"month_start_{window}wd_first_day_of_month"] = int(head["day_of_month"].iloc[0]) if len(head) else np.nan
        row[f"month_start_{window}wd_last_day_of_month"] = int(head["day_of_month"].iloc[-1]) if len(head) else np.nan
        row[f"month_start_{window}wd_non_workday_gap_days"] = row[f"month_start_{window}wd_calendar_span_days"] - len(head) if len(head) else np.nan

        curr_mid_qty = mid_qty.sum()
        curr_mid_avg_qty = mid_qty.mean()
        curr_mid_slope = linear_slope(mid_qty)
        row[f"month_mid_{window}wd_qty"] = curr_mid_qty
        row[f"month_mid_{window}wd_avg_qty"] = curr_mid_avg_qty
        row[f"month_mid_{window}wd_qty_slope"] = curr_mid_slope
        row[f"month_mid_{window}wd_qty_slope_per_avg"] = safe_divide(curr_mid_slope, curr_mid_avg_qty)
        row[f"month_mid_{window}wd_cumsum_qty_slope"] = linear_slope(mid["cumsum_qty"])
        row[f"month_mid_{window}wd_qty_contrib_pct"] = safe_divide(curr_mid_qty, total_qty)
        row[f"month_mid_{window}wd_start_workday_seq"] = int(mid["workday_seq"].iloc[0]) if len(mid) else np.nan
        row[f"month_mid_{window}wd_end_workday_seq"] = int(mid["workday_seq"].iloc[-1]) if len(mid) else np.nan
        row[f"month_mid_{window}wd_calendar_span_days"] = int((mid["transdate"].max() - mid["transdate"].min()).days + 1) if len(mid) else np.nan
        row[f"month_mid_{window}wd_first_day_of_month"] = int(mid["day_of_month"].iloc[0]) if len(mid) else np.nan
        row[f"month_mid_{window}wd_last_day_of_month"] = int(mid["day_of_month"].iloc[-1]) if len(mid) else np.nan
        row[f"month_mid_{window}wd_non_workday_gap_days"] = row[f"month_mid_{window}wd_calendar_span_days"] - len(mid) if len(mid) else np.nan
        row[f"month_mid_minus_start_{window}wd_qty_pct"] = safe_pct_change(curr_mid_qty, curr_start_qty)
        row[f"month_start_minus_prev_month_mid_{window}wd_qty_pct"] = np.nan

        curr_end_qty = tail_qty.sum()
        prev_end_qty = month_end_window_qty_lookup[window].get(prev_bizym, np.nan)
        row[f"month_end_{window}wd_qty"] = curr_end_qty
        row[f"month_end_{window}wd_avg_qty"] = tail_qty.mean()
        row[f"month_end_{window}wd_qty_slope"] = linear_slope(tail_qty)
        row[f"month_end_{window}wd_qty_mom_pct"] = safe_pct_change(curr_end_qty, prev_end_qty)
        row[f"month_end_{window}wd_qty_contrib_pct"] = safe_divide(curr_end_qty, total_qty)
        row[f"month_end_{window}wd_cumsum_qty_slope"] = linear_slope(tail["cumsum_qty"])
        row[f"month_end_{window}wd_calendar_span_days"] = int((tail["transdate"].max() - tail["transdate"].min()).days + 1) if len(tail) else np.nan
        row[f"month_end_{window}wd_first_day_of_month"] = int(tail["day_of_month"].iloc[0]) if len(tail) else np.nan
        row[f"month_end_{window}wd_last_day_of_month"] = int(tail["day_of_month"].iloc[-1]) if len(tail) else np.nan
        row[f"month_end_{window}wd_non_workday_gap_days"] = row[f"month_end_{window}wd_calendar_span_days"] - len(tail) if len(tail) else np.nan
        row[f"month_end_minus_mid_{window}wd_qty_pct"] = safe_pct_change(curr_end_qty, curr_mid_qty)
        rows.append(row)

    month_window_metrics = pd.DataFrame(rows).sort_values("month_start").reset_index(drop=True)
    prev_mid_qty_lookup = month_window_metrics.set_index("bizym")[f"month_mid_{window}wd_qty"].to_dict()
    prev_month_bizym = (month_window_metrics["month_start"] - pd.DateOffset(months=1)).dt.strftime("%Y%m").astype(int)
    month_window_metrics[f"month_start_minus_prev_month_mid_{window}wd_qty_pct"] = [
        safe_pct_change(curr_start_qty, prev_mid_qty_lookup.get(pm, np.nan))
        for curr_start_qty, pm in zip(month_window_metrics[f"month_start_{window}wd_qty"], prev_month_bizym)
    ]

    rolling_base_cols = [
        f"month_start_{window}wd_qty",
        f"month_start_{window}wd_avg_qty",
        f"month_start_{window}wd_qty_slope",
        f"month_start_{window}wd_qty_slope_per_avg",
        f"month_start_{window}wd_cv_qty",
        f"month_start_{window}wd_first1_share_in_window",
        f"month_start_{window}wd_first2_share_in_window",
        f"month_start_{window}wd_last2_share_in_window",
        f"month_mid_{window}wd_qty",
        f"month_mid_{window}wd_avg_qty",
        f"month_mid_{window}wd_qty_slope",
        f"month_mid_{window}wd_qty_slope_per_avg",
        f"month_mid_minus_start_{window}wd_qty_pct",
        f"month_start_minus_prev_month_mid_{window}wd_qty_pct",
    ]
    for base_col in rolling_base_cols:
        prev_year_lookup = month_window_metrics.set_index("bizym")[base_col].to_dict()
        prev_year_bizym = (month_window_metrics["year"] - 1) * 100 + month_window_metrics["month"]
        month_window_metrics[f"{base_col}_yoy_pct"] = [
            safe_pct_change(curr, prev_year_lookup.get(py, np.nan))
            for curr, py in zip(month_window_metrics[base_col], prev_year_bizym)
        ]
        for rolling_n in DEFAULT_ROLLING_LOOKBACKS:
            rolling_mean = month_window_metrics[base_col].shift(1).rolling(rolling_n, min_periods=2).mean()
            month_window_metrics[f"{base_col}_vs_prev{rolling_n}m_avg_pct"] = [
                safe_pct_change(curr, prev_avg)
                for curr, prev_avg in zip(month_window_metrics[base_col], rolling_mean)
            ]

    calendar_months = monthly[["bizym", "year", "month", "month_start"]].copy()
    calendar_rows = []
    for _, r in calendar_months.iterrows():
        row = r.to_dict()
        curr_start = pd.Timestamp(r["month_start"])
        curr_end = curr_start + pd.offsets.MonthEnd(0)
        for prefix, offset in [("prev_month", -1), ("curr_month", 0), ("next_month", 1)]:
            counts = count_calendar_days(curr_start + pd.DateOffset(months=offset), calendar.is_workday)
            for k, v in counts.items():
                row[f"{prefix}_{k}"] = v

        for w in DEFAULT_NATURAL_DAY_WINDOWS:
            row.update(summarize_date_window(pd.date_range(curr_start, periods=w, freq="D"), f"curr_month_start_{w}natural_days", calendar.is_workday))
            row.update(summarize_date_window(pd.date_range(curr_end - pd.Timedelta(days=w - 1), curr_end, freq="D"), f"curr_month_end_{w}natural_days", calendar.is_workday))

        month_workday_dates = pd.Series(pd.date_range(curr_start, curr_end, freq="D"))
        month_workday_dates = month_workday_dates[month_workday_dates.map(calendar.is_workday)].reset_index(drop=True)
        for offset in anchors:
            anchor_pos = len(month_workday_dates) - offset
            prefix = f"anchor_d{offset}"
            if 0 <= anchor_pos < len(month_workday_dates):
                anchor_date = pd.Timestamp(month_workday_dates.iloc[anchor_pos])
                row[f"{prefix}_day_of_month"] = int(anchor_date.day)
                row[f"{prefix}_weekday"] = int(anchor_date.dayofweek)
                row[f"{prefix}_workday_seq"] = int(anchor_pos + 1)
                row[f"{prefix}_remaining_workdays_after_anchor"] = int(offset - 1)
                row[f"{prefix}_remaining_calendar_days_after_anchor"] = int((curr_end - anchor_date).days)
                row[f"{prefix}_month_elapsed_workday_pct"] = safe_divide(anchor_pos + 1, len(month_workday_dates))
                row[f"{prefix}_month_elapsed_calendar_pct"] = safe_divide(anchor_date.day, curr_end.day)
            else:
                for suffix in [
                    "day_of_month",
                    "weekday",
                    "workday_seq",
                    "remaining_workdays_after_anchor",
                    "remaining_calendar_days_after_anchor",
                    "month_elapsed_workday_pct",
                    "month_elapsed_calendar_pct",
                ]:
                    row[f"{prefix}_{suffix}"] = np.nan
        calendar_rows.append(row)
    calendar_metrics = pd.DataFrame(calendar_rows)

    holiday_rows = []
    for _, r in calendar_months.iterrows():
        row = {"bizym": r["bizym"], "year": r["year"], "month": r["month"], "month_start": r["month_start"]}
        curr_start = pd.Timestamp(r["month_start"])
        next_start = curr_start + pd.DateOffset(months=1)
        prev_end = curr_start - pd.Timedelta(days=1)
        next_head_dates = pd.date_range(next_start, periods=window, freq="D")
        prev_tail_dates = pd.date_range(prev_end - pd.Timedelta(days=window - 1), prev_end, freq="D")
        row[f"next_month_start_{window}natural_days_holiday_days"] = int((~pd.Series(next_head_dates).map(calendar.is_workday)).sum())
        row[f"prev_month_end_{window}natural_days_holiday_days"] = int((~pd.Series(prev_tail_dates).map(calendar.is_workday)).sum())
        holiday_rows.append(row)
    holiday_window_metrics = pd.DataFrame(holiday_rows)

    month_feature_detail = (
        calendar_metrics.merge(
            month_window_metrics.drop(columns=["year", "month", "month_start"]),
            on="bizym",
            how="left",
        )
        .merge(
            holiday_window_metrics.drop(columns=["year", "month", "month_start"]),
            on="bizym",
            how="left",
        )
        .merge(monthly[["bizym", "month_total", "month_total_mom_pct", "yoy_pct", "month_total_roll3", "month_total_roll6"]], on="bizym", how="left")
        .sort_values("month_start")
        .reset_index(drop=True)
    )

    lag_source_cols = [
        f"month_start_{window}wd_qty_contrib_pct",
        f"month_start_{window}wd_cumsum_qty_slope",
        f"month_mid_{window}wd_qty_contrib_pct",
        f"month_mid_{window}wd_cumsum_qty_slope",
        f"month_mid_minus_start_{window}wd_qty_pct",
        f"month_start_minus_prev_month_mid_{window}wd_qty_pct",
        f"month_end_{window}wd_qty_contrib_pct",
        f"month_end_{window}wd_cumsum_qty_slope",
        f"month_end_{window}wd_qty_mom_pct",
        f"month_end_minus_mid_{window}wd_qty_pct",
    ]
    month_feature_detail[f"month_end_minus_start_{window}wd_qty_contrib_pct"] = (
        month_feature_detail[f"month_end_{window}wd_qty_contrib_pct"]
        - month_feature_detail[f"month_start_{window}wd_qty_contrib_pct"]
    )
    month_feature_detail[f"month_mid_minus_start_{window}wd_qty_contrib_pct"] = (
        month_feature_detail[f"month_mid_{window}wd_qty_contrib_pct"]
        - month_feature_detail[f"month_start_{window}wd_qty_contrib_pct"]
    )
    month_feature_detail[f"month_end_minus_mid_{window}wd_qty_contrib_pct"] = (
        month_feature_detail[f"month_end_{window}wd_qty_contrib_pct"]
        - month_feature_detail[f"month_mid_{window}wd_qty_contrib_pct"]
    )
    lag_source_cols.extend(
        [
            f"month_end_minus_start_{window}wd_qty_contrib_pct",
            f"month_mid_minus_start_{window}wd_qty_contrib_pct",
            f"month_end_minus_mid_{window}wd_qty_contrib_pct",
        ]
    )

    for col in lag_source_cols:
        if col not in month_feature_detail.columns:
            continue
        month_feature_detail[f"prev_month_{col}"] = month_feature_detail[col].shift(1)
        prev_year_lookup = month_feature_detail.set_index("bizym")[col].to_dict()
        prev_year_bizym = (month_feature_detail["year"] - 1) * 100 + month_feature_detail["month"]
        month_feature_detail[f"prev_year_same_month_{col}"] = [
            prev_year_lookup.get(py, np.nan) for py in prev_year_bizym
        ]

    leakage_feature_cols = {
        f"month_start_{window}wd_qty_contrib_pct",
        f"month_mid_{window}wd_qty_contrib_pct",
        f"month_mid_{window}wd_cumsum_qty_slope",
        f"month_mid_minus_start_{window}wd_qty_contrib_pct",
        f"month_end_{window}wd_qty_mom_pct",
        f"month_end_{window}wd_qty",
        f"month_end_{window}wd_avg_qty",
        f"month_end_{window}wd_qty_slope",
        f"month_end_{window}wd_qty_contrib_pct",
        f"month_end_{window}wd_cumsum_qty_slope",
        f"month_end_minus_start_{window}wd_qty_contrib_pct",
        f"month_end_minus_mid_{window}wd_qty_contrib_pct",
        f"month_end_minus_mid_{window}wd_qty_pct",
        "month_total",
    }

    anchor_d5_safe_extra_feature_cols = [
        c
        for c in month_feature_detail.select_dtypes(include=[np.number]).columns
        if c not in {"bizym", "year", "month"}
        and c not in leakage_feature_cols
        and (re.search(rf"^month_mid_{window}wd_", c) or re.search(rf"^month_mid_minus_start_{window}wd_qty_pct", c))
    ]

    anchor_safe_feature_cols = [
        c
        for c in month_feature_detail.select_dtypes(include=[np.number]).columns
        if c not in {"bizym", "year", "month"}
        and c not in leakage_feature_cols
        and c not in anchor_d5_safe_extra_feature_cols
    ]

    target_cols = [
        f"month_start_{window}wd_qty_contrib_pct",
        f"month_end_{window}wd_qty_contrib_pct",
        f"month_start_{window}wd_cumsum_qty_slope",
        f"month_end_{window}wd_cumsum_qty_slope",
        f"month_mid_{window}wd_qty_contrib_pct",
        f"month_end_minus_start_{window}wd_qty_contrib_pct",
        f"month_end_minus_mid_{window}wd_qty_contrib_pct",
    ]
    target_cols = [c for c in target_cols if c in month_feature_detail.columns]

    return {
        "month_feature_detail": month_feature_detail,
        "anchor_safe_feature_cols": anchor_safe_feature_cols,
        "anchor_d5_safe_extra_feature_cols": anchor_d5_safe_extra_feature_cols,
        "leakage_feature_cols": sorted(leakage_feature_cols),
        "target_cols": target_cols,
    }


@st.cache_data(show_spinner="筛选相关性与统计证据...")
def compute_correlations(
    feature_detail: pd.DataFrame,
    target_cols: list[str],
    anchor_safe_feature_cols: list[str],
    anchor_d5_safe_extra_feature_cols: list[str],
    leakage_feature_cols: list[str],
    include_d5_extra: bool,
    corr_method: str,
    corr_threshold: float,
    min_sample_size: int,
) -> pd.DataFrame:
    if feature_detail.empty:
        return pd.DataFrame()

    candidate_cols = list(dict.fromkeys(anchor_safe_feature_cols + (anchor_d5_safe_extra_feature_cols if include_d5_extra else [])))
    leakage_set = set(leakage_feature_cols)
    candidate_cols = [
        c
        for c in candidate_cols
        if c in feature_detail.columns
        and c not in {"bizym", "year", "month", "month_start"}
        and c not in target_cols
        and c not in leakage_set
    ]

    rows = []
    for target_col in target_cols:
        if target_col not in feature_detail.columns:
            continue
        for feature_col in candidate_cols:
            pair = feature_detail[[target_col, feature_col]].dropna()
            valid_n = len(pair)
            if valid_n < min_sample_size:
                continue
            x = pair[feature_col].astype(float)
            y = pair[target_col].astype(float)
            if x.nunique(dropna=True) < 2 or y.nunique(dropna=True) < 2:
                continue
            if corr_method == "spearman":
                corr_value, p_value = stats.spearmanr(x, y)
            else:
                corr_value, p_value = stats.pearsonr(x, y)
            if pd.isna(corr_value) or abs(corr_value) < corr_threshold:
                continue
            rows.append(
                {
                    "target": target_col,
                    "target_cn": metric_col_to_cn(target_col),
                    "feature": feature_col,
                    "feature_cn": metric_col_to_cn(feature_col),
                    "feature_group": feature_group(feature_col),
                    "availability": "D-5可用" if feature_col in anchor_d5_safe_extra_feature_cols else "D-10/D-5均可用",
                    "corr_method": corr_method,
                    "corr_coef": float(corr_value),
                    "abs_corr_coef": abs(float(corr_value)),
                    "p_value": float(p_value) if pd.notna(p_value) else np.nan,
                    "sample_size": int(valid_n),
                    "direction": "正相关" if corr_value > 0 else "负相关",
                }
            )
    corr_df = pd.DataFrame(rows)
    if corr_df.empty:
        return corr_df
    if multipletests is not None and corr_df["p_value"].notna().any():
        mask = corr_df["p_value"].notna()
        qvals = np.full(len(corr_df), np.nan)
        qvals[mask] = multipletests(corr_df.loc[mask, "p_value"], method="fdr_bh")[1]
        corr_df["fdr_q_value"] = qvals
    else:
        corr_df["fdr_q_value"] = np.nan
    corr_df["evidence_level"] = np.select(
        [
            corr_df["abs_corr_coef"].ge(0.65) & corr_df["fdr_q_value"].fillna(1).le(0.1),
            corr_df["abs_corr_coef"].ge(0.5),
            corr_df["abs_corr_coef"].ge(corr_threshold),
        ],
        ["强证据", "中等证据", "探索性证据"],
        default="弱",
    )
    return corr_df.sort_values(["target", "abs_corr_coef", "sample_size"], ascending=[True, False, False]).reset_index(drop=True)


def ewols(y: np.ndarray, lam: float = 0.9, warmup: int = 6) -> dict[str, np.ndarray]:
    y = np.asarray(y, dtype=float)
    n = len(y)
    s0 = s1 = s2 = t0 = t1 = 0.0
    alpha = np.full(n, np.nan)
    beta = np.full(n, np.nan)
    for t in range(n):
        s0 = lam * s0 + 1.0
        s1 = lam * s1 + t
        s2 = lam * s2 + t * t
        t0 = lam * t0 + y[t]
        t1 = lam * t1 + t * y[t]
        if t < warmup:
            continue
        det = s2 * s0 - s1 * s1
        if abs(det) < 1e-10:
            continue
        b = (t1 * s0 - t0 * s1) / det
        a = (t0 - b * s1) / s0
        alpha[t] = a
        beta[t] = b
    return {"alpha": alpha, "beta": beta}


def rls_ols(y: np.ndarray, warmup: int = 6, lam: float = 1.0) -> dict[str, np.ndarray] | None:
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < warmup + 2:
        return None
    tvar = np.arange(n, dtype=float)
    alpha = np.full(n, np.nan)
    beta = np.full(n, np.nan)
    idx0 = np.arange(0, warmup)
    x0 = np.column_stack([np.ones(len(idx0)), tvar[idx0]])
    y0 = y[idx0]
    w0 = np.array([lam ** (warmup - 1 - s) for s in idx0], dtype=float)
    q = x0.T @ (w0[:, None] * x0)
    p = np.linalg.pinv(q)
    theta = p @ (x0.T @ (w0 * y0))
    alpha[warmup - 1] = theta[0]
    beta[warmup - 1] = theta[1]
    for t in range(warmup, n):
        x = np.array([1.0, tvar[t]])
        px = p @ x
        denom = lam + float(x @ px)
        if abs(denom) < 1e-12:
            continue
        k = px / denom
        innov = y[t] - float(x @ theta)
        theta = theta + k * innov
        p = (p - np.outer(k, px)) / lam
        alpha[t] = theta[0]
        beta[t] = theta[1]
    return {"alpha": alpha, "beta": beta}


def abs_integral_linear(c0: float, c1: float, a: float, b: float) -> float:
    def primitive(s: float) -> float:
        return c0 * s + c1 * s**2 / 2.0

    if abs(c1) < 1e-15:
        return abs(c0) * (b - a)
    s_star = -c0 / c1
    if a < s_star < b:
        return abs(primitive(s_star) - primitive(a)) + abs(primitive(b) - primitive(s_star))
    return abs(primitive(b) - primitive(a))


def signed_area_curve(alpha: np.ndarray, beta: np.ndarray, win: int) -> np.ndarray:
    n = len(alpha)
    b_curve = np.full(n, np.nan)
    for t in range(n):
        if np.isnan(alpha[t]) or np.isnan(beta[t]):
            continue
        lo = float(max(0, t - win))
        hi = float(t)
        b_curve[t] = alpha[t] * (hi - lo) + beta[t] * (hi**2 - lo**2) / 2.0
    return b_curve


def area_change_ratio(res: dict[str, np.ndarray], win: int, mode: str) -> np.ndarray:
    alpha = res["alpha"]
    beta = res["beta"]
    n = len(alpha)
    out = np.full(n, np.nan)
    for t in range(1, n):
        if np.isnan(alpha[t]) or np.isnan(alpha[t - 1]):
            continue
        if mode == "fixed":
            lo_a, hi_a = float(t - win), float(t - 1)
            lo_b, hi_b = lo_a, hi_a
        else:
            lo_a, hi_a = float(t - win), float(t)
            lo_b, hi_b = float(t - win - 1), float(t - 1)
        a = abs_integral_linear(float(alpha[t]), float(beta[t]), lo_a, hi_a)
        b = abs_integral_linear(float(alpha[t - 1]), float(beta[t - 1]), lo_b, hi_b)
        if b > 1e-8:
            out[t] = (a - b) / b
    return out


def moving_average(y: np.ndarray, months: list[str], win: int) -> tuple[np.ndarray, list[str], np.ndarray]:
    y = np.asarray(y, dtype=float)
    if win <= 1:
        return y.copy(), months, np.arange(len(y))
    vals = pd.Series(y).rolling(win, min_periods=win).mean().dropna()
    idx = vals.index.to_numpy()
    return vals.to_numpy(dtype=float), [months[i] for i in idx], idx


def is_backtest_cancelled(r: np.ndarray, t: int, threshold: float, win: int = 3) -> bool:
    future = r[t + 1 : min(len(r), t + 1 + win)]
    future = future[~np.isnan(future)]
    if len(future) == 0:
        return False
    return bool(np.any(future <= threshold))


def chow_test(y: np.ndarray, break_idx: int, win: int, n_min: int = 3) -> dict[str, float] | None:
    y = np.asarray(y, dtype=float)
    lo = break_idx - win
    hi = break_idx + win
    if lo < 0 or hi > len(y):
        return None
    x1 = np.arange(lo, break_idx, dtype=float)
    y1 = y[lo:break_idx]
    x2 = np.arange(break_idx, hi, dtype=float)
    y2 = y[break_idx:hi]
    if len(y1) < n_min or len(y2) < n_min:
        return None

    def rss(x: np.ndarray, yy: np.ndarray) -> float:
        xmat = np.column_stack([np.ones(len(x)), x])
        coef, _, _, _ = np.linalg.lstsq(xmat, yy, rcond=None)
        resid = yy - xmat @ coef
        return float(resid @ resid)

    rss_pooled = rss(np.concatenate([x1, x2]), np.concatenate([y1, y2]))
    rss_split = rss(x1, y1) + rss(x2, y2)
    k = 2
    df_den = len(y1) + len(y2) - 2 * k
    if df_den <= 0 or rss_split <= 1e-12:
        return None
    f_stat = ((rss_pooled - rss_split) / k) / (rss_split / df_den)
    p_value = float(stats.f.sf(f_stat, k, df_den))
    return {"break_idx": float(break_idx), "F": float(f_stat), "p_value": p_value}


def merge_chow_breaks(chow_df: pd.DataFrame, gap: int = 3) -> pd.DataFrame:
    if chow_df.empty:
        return chow_df
    sig = chow_df.sort_values("break_idx").copy()
    groups = []
    current = [sig.iloc[0]]
    for _, row in sig.iloc[1:].iterrows():
        if int(row["break_idx"]) - int(current[-1]["break_idx"]) <= gap:
            current.append(row)
        else:
            groups.append(current)
            current = [row]
    groups.append(current)
    kept = [max(group, key=lambda r: r["F"]) for group in groups]
    return pd.DataFrame(kept)


def quadratic_area_anomaly(monthly_signal: pd.DataFrame, min_pts: int, pi_alpha: float, eps_ratio: float) -> pd.DataFrame:
    df = monthly_signal.dropna(subset=["B_t"]).copy()
    if len(df) < min_pts + 2:
        return pd.DataFrame()
    b = df["B_t"].to_numpy(dtype=float)
    t = np.arange(len(b), dtype=float)
    rows = []
    for i in range(min_pts, len(b)):
        t_fit = t[:i]
        b_fit = b[:i]
        x = np.column_stack([np.ones(len(t_fit)), t_fit, t_fit**2])
        coef, _, _, _ = np.linalg.lstsq(x, b_fit, rcond=None)
        resid = b_fit - x @ coef
        dof = len(t_fit) - x.shape[1]
        if dof <= 0:
            continue
        sigma2 = float((resid @ resid) / dof)
        xtx_inv = np.linalg.pinv(x.T @ x)
        x_new = np.array([1.0, t[i], t[i] ** 2])
        pred = float(x_new @ coef)
        se_pred = math.sqrt(max(0.0, sigma2 * (1.0 + x_new @ xtx_inv @ x_new)))
        tval = float(stats.t.ppf(1 - pi_alpha / 2, dof))
        lo = pred - tval * se_pred
        hi = pred + tval * se_pred
        delta_actual = b[i] - b[i - 1]
        trend_slope = float(coef[1] + 2 * coef[2] * t[i])
        reversal = np.sign(delta_actual) != np.sign(trend_slope) and abs(delta_actual) > np.nanmean(np.abs(b_fit)) * eps_ratio
        outside = b[i] < lo or b[i] > hi
        rows.append(
            {
                "month_label": df["month_label"].iloc[i],
                "B_t": b[i],
                "pred_B_t": pred,
                "pred_lo": lo,
                "pred_hi": hi,
                "trend_slope": trend_slope,
                "actual_delta": delta_actual,
                "outside_prediction_interval": bool(outside),
                "trend_reversal": bool(reversal),
                "area_anomaly": bool(outside and reversal),
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="运行 OLS/RLS/面积/Chow 高级统计流程...")
def compute_advanced_methods(
    monthly: pd.DataFrame,
    ma_window: int,
    warmup: int,
    lam: float,
    area_window: int,
    threshold: float,
    chow_alpha: float,
    chow_window: int,
    eps_ratio: float,
) -> dict[str, pd.DataFrame]:
    series_df = monthly[["month_label", "month_start", "month_total"]].copy()
    series_df = series_df[series_df["month_total"].notna()].reset_index(drop=True)
    y_full = series_df["month_total"].to_numpy(dtype=float)
    months_full = series_df["month_label"].tolist()
    y_ma, months_ma, raw_idx = moving_average(y_full, months_full, ma_window)
    if len(y_ma) < warmup + 2:
        return {"signal": pd.DataFrame(), "chow": pd.DataFrame(), "area_anomaly": pd.DataFrame()}

    res_rls = rls_ols(y_ma, warmup=warmup, lam=lam)
    res_ewols = ewols(y_ma, lam=lam, warmup=warmup)
    if res_rls is None:
        return {"signal": pd.DataFrame(), "chow": pd.DataFrame(), "area_anomaly": pd.DataFrame()}

    r_rolling = area_change_ratio(res_rls, win=area_window, mode="rolling")
    r_fixed = area_change_ratio(res_rls, win=area_window, mode="fixed")
    b_curve = signed_area_curve(res_rls["alpha"], res_rls["beta"], win=area_window)
    delta_beta = np.concatenate([[np.nan], np.diff(res_rls["beta"])])
    candidate = (np.nan_to_num(r_rolling) > threshold) & (np.nan_to_num(delta_beta) > 0)
    cancelled = np.array([is_backtest_cancelled(r_rolling, i, threshold) if candidate[i] else False for i in range(len(candidate))])
    confirmed = candidate & ~cancelled

    signal = pd.DataFrame(
        {
            "month_label": months_ma,
            "raw_month_idx": raw_idx,
            "qty_signal": y_ma,
            "alpha_rls": res_rls["alpha"],
            "beta_rls": res_rls["beta"],
            "alpha_ewols": res_ewols["alpha"],
            "beta_ewols": res_ewols["beta"],
            "delta_beta": delta_beta,
            "B_t": b_curve,
            "R_t_rolling": r_rolling,
            "R_t_fixed": r_fixed,
            "candidate_anomaly": candidate,
            "backtest_cancelled": cancelled,
            "confirmed_anomaly": confirmed,
        }
    )

    chow_rows = []
    for break_idx in range(chow_window, len(y_ma) - chow_window + 1):
        result = chow_test(y_ma, break_idx, chow_window)
        if result is None:
            continue
        result["month_label"] = months_ma[int(result["break_idx"])]
        result["is_significant"] = result["p_value"] < chow_alpha
        chow_rows.append(result)
    chow = pd.DataFrame(chow_rows)
    if not chow.empty:
        merged = merge_chow_breaks(chow[chow["is_significant"]].copy(), gap=max(2, chow_window // 2))
        chow["merged_significant"] = chow["break_idx"].isin(merged["break_idx"].tolist()) if not merged.empty else False

    area_anomaly = quadratic_area_anomaly(signal, min_pts=max(4, warmup), pi_alpha=0.05, eps_ratio=eps_ratio)
    return {"signal": signal, "chow": chow, "area_anomaly": area_anomaly}


@st.cache_data(show_spinner="运行残差和平稳性诊断...")
def compute_diagnostics(monthly: pd.DataFrame) -> pd.DataFrame:
    y = monthly["month_total"].dropna().astype(float).reset_index(drop=True)
    rows = []
    if len(y) < 8:
        return pd.DataFrame({"test": ["样本不足"], "statistic": [np.nan], "p_value": [np.nan], "interpretation": ["少于8个月，暂不运行高级诊断"]})

    if adfuller is not None:
        try:
            stat, pval, *_ = adfuller(y, autolag="AIC")
            rows.append({"test": "ADF 单位根检验", "statistic": stat, "p_value": pval, "interpretation": "p<0.05 支持平稳；p较大表示可能存在单位根"})
        except Exception as exc:
            rows.append({"test": "ADF 单位根检验", "statistic": np.nan, "p_value": np.nan, "interpretation": f"失败：{exc}"})

    if kpss is not None:
        try:
            stat, pval, *_ = kpss(y, regression="c", nlags="auto")
            rows.append({"test": "KPSS 平稳性检验", "statistic": stat, "p_value": pval, "interpretation": "p<0.05 反对平稳；与 ADF 互补判断"})
        except Exception as exc:
            rows.append({"test": "KPSS 平稳性检验", "statistic": np.nan, "p_value": np.nan, "interpretation": f"失败：{exc}"})

    if sm is not None and len(y) >= 10:
        x = sm.add_constant(np.arange(len(y), dtype=float))
        try:
            ols = sm.OLS(y, x).fit()
            resid = pd.Series(ols.resid)
            rows.append({"test": "OLS 趋势斜率 t 检验", "statistic": float(ols.tvalues[1]), "p_value": float(ols.pvalues[1]), "interpretation": "p<0.05 表示月销量存在显著线性趋势"})
            rows.append({"test": "Durbin-Watson 残差自相关", "statistic": float(sm.stats.durbin_watson(resid)), "p_value": np.nan, "interpretation": "接近2较好；明显低于2表示正自相关"})
            jb_stat, jb_p, _, _ = sm.stats.jarque_bera(resid)
            rows.append({"test": "Jarque-Bera 残差正态性", "statistic": float(jb_stat), "p_value": float(jb_p), "interpretation": "p<0.05 表示残差偏离正态"})
            if acorr_ljungbox is not None:
                lb = acorr_ljungbox(resid, lags=[min(6, len(resid) // 2)], return_df=True)
                rows.append({"test": "Ljung-Box 残差白噪声", "statistic": float(lb["lb_stat"].iloc[0]), "p_value": float(lb["lb_pvalue"].iloc[0]), "interpretation": "p<0.05 表示残差仍有自相关结构"})
            if het_arch is not None and len(resid) >= 12:
                arch_stat, arch_p, _, _ = het_arch(resid, nlags=min(4, len(resid) // 4))
                rows.append({"test": "ARCH 波动聚集检验", "statistic": float(arch_stat), "p_value": float(arch_p), "interpretation": "p<0.05 表示波动存在聚集，异常阈值应更稳健"})
        except Exception as exc:
            rows.append({"test": "OLS 残差诊断", "statistic": np.nan, "p_value": np.nan, "interpretation": f"失败：{exc}"})
    else:
        rows.append({"test": "statsmodels 诊断", "statistic": np.nan, "p_value": np.nan, "interpretation": "statsmodels 不可用或样本不足"})

    return pd.DataFrame(rows)


def plot_monthly_series(monthly: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=monthly["month_label"], y=monthly["month_total"], name="月销量", marker_color="#d0e9f7"))
    fig.add_trace(go.Scatter(x=monthly["month_label"], y=monthly["month_total_roll3"], name="3月滚动均值", line=dict(color=COLOR_BLUE, width=2)))
    fig.add_trace(go.Scatter(x=monthly["month_label"], y=monthly["month_total_roll6"], name="6月滚动均值", line=dict(color=COLOR_MAIN, width=2)))
    fig.update_layout(height=420, xaxis_title="月份", yaxis_title="qty", legend_orientation="h")
    return fig


def plot_anchor_trend(anchor_rows: pd.DataFrame, selected_anchor: int) -> go.Figure:
    df = anchor_rows[anchor_rows["forecast_offset"].eq(selected_anchor)].copy()
    fig = px.line(
        df,
        x="month_label",
        y="mtd_pct",
        color="year",
        markers=True,
        title=f"D-{selected_anchor} 工作日 MTD累计占比",
        labels={"month_label": "月份", "mtd_pct": "MTD累计占比", "year": "年份"},
    )
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray")
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(height=420, legend_orientation="h")
    return fig


def plot_workday_curves(workday_filtered: pd.DataFrame, selected_month: int, mode: str) -> go.Figure:
    df = workday_filtered[workday_filtered["month"].eq(selected_month)].copy()
    ycol = "workday_cumsum_pct" if mode == "累计占比" else "qty"
    title = f"{selected_month:02d}月 工作日对齐{'累计占比' if mode == '累计占比' else '日销量'}曲线"
    fig = px.line(
        df,
        x="workday_seq",
        y=ycol,
        color="year",
        markers=True,
        title=title,
        labels={"workday_seq": "月内第N个工作日", ycol: mode, "year": "年份"},
    )
    if ycol.endswith("pct"):
        fig.update_yaxes(tickformat=".0%")
    fig.update_layout(height=420, legend_orientation="h")
    return fig


def plot_corr_bar(corr_df: pd.DataFrame, target_cn: str) -> go.Figure:
    sub = corr_df[corr_df["target_cn"].eq(target_cn)].head(20).copy()
    sub = sub.sort_values("abs_corr_coef", ascending=True)
    colors = np.where(sub["corr_coef"] >= 0, COLOR_MAIN, COLOR_DANGER)
    fig = go.Figure(
        go.Bar(
            x=sub["corr_coef"],
            y=sub["feature_cn"],
            orientation="h",
            marker_color=colors,
            customdata=np.stack([sub["feature_group"], sub["sample_size"], sub["evidence_level"]], axis=-1) if len(sub) else None,
            hovertemplate="相关系数=%{x:.2f}<br>%{y}<br>类型=%{customdata[0]}<br>n=%{customdata[1]}<br>证据=%{customdata[2]}<extra></extra>",
        )
    )
    fig.update_layout(height=max(420, 24 * max(1, len(sub))), title=f"{target_cn}：Top 相关解释变量", xaxis_title="相关系数", yaxis_title="")
    return fig


def plot_advanced_signal(signal: pd.DataFrame, threshold: float) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if signal.empty:
        return go.Figure()
    colors = np.where(signal["confirmed_anomaly"], COLOR_DANGER, np.where(signal["candidate_anomaly"], COLOR_WARN, "#d0e9f7"))
    fig.add_trace(go.Bar(x=signal["month_label"], y=signal["qty_signal"], name="MA平滑月销量", marker_color=colors), secondary_y=False)
    fig.add_trace(go.Scatter(x=signal["month_label"], y=signal["beta_rls"], name="RLS斜率 beta", line=dict(color=COLOR_BLUE)), secondary_y=True)
    fig.add_trace(go.Scatter(x=signal["month_label"], y=signal["R_t_rolling"], name="滚动窗口 R_t", line=dict(color=COLOR_MAIN)), secondary_y=True)
    fig.add_trace(
        go.Scatter(
            x=signal["month_label"],
            y=np.repeat(threshold, len(signal)),
            name="R_t阈值",
            line=dict(color=COLOR_DANGER, dash="dash"),
        ),
        secondary_y=True,
    )
    fig.update_layout(height=480, title="RLS趋势、面积变化率与回测确认异常", legend_orientation="h")
    fig.update_yaxes(title_text="qty", secondary_y=False)
    fig.update_yaxes(title_text="beta / R_t", secondary_y=True)
    return fig


def plot_chow(chow: pd.DataFrame) -> go.Figure:
    if chow.empty:
        return go.Figure()
    fig = go.Figure()
    colors = np.where(chow.get("merged_significant", False), COLOR_DANGER, np.where(chow["is_significant"], COLOR_WARN, "#d0e9f7"))
    fig.add_trace(go.Bar(x=chow["month_label"], y=chow["F"], name="Chow F", marker_color=colors))
    fig.add_trace(go.Scatter(x=chow["month_label"], y=-np.log10(chow["p_value"].clip(lower=1e-12)), name="-log10(p)", yaxis="y2", line=dict(color=COLOR_PURPLE)))
    fig.update_layout(
        height=400,
        title="Chow Test 结构性断点证据",
        yaxis=dict(title="F statistic"),
        yaxis2=dict(title="-log10(p)", overlaying="y", side="right"),
        legend_orientation="h",
    )
    return fig


def business_recommendations(corr_df: pd.DataFrame, window: int) -> pd.DataFrame:
    if corr_df.empty:
        return pd.DataFrame()
    target_order = [
        f"月初{window}工作日销量贡献占比",
        f"月末{window}工作日销量贡献占比",
        f"月初{window}工作日累计销量斜率",
        f"月末{window}工作日累计销量斜率",
    ]
    rows = []
    for target in target_order:
        sub = corr_df[corr_df["target_cn"].eq(target)].sort_values("abs_corr_coef", ascending=False).head(5)
        if sub.empty:
            continue
        drivers = []
        for _, r in sub.iterrows():
            direction = "推高" if r["corr_coef"] > 0 else "压低"
            drivers.append(f"{r['feature_cn']}({direction}, r={r['corr_coef']:.2f})")
        rows.append(
            {
                "业务问题": f"{target}可能受哪些变量影响？",
                "主要候选解释": "；".join(drivers),
                "证据强度": sub["evidence_level"].iloc[0],
                "解释变量类型": " / ".join(sub["feature_group"].drop_duplicates().tolist()),
            }
        )
    return pd.DataFrame(rows)


st.set_page_config(layout="wide", page_title="30d Jenny Anchor Signal Dashboard")
st.title("30d Jenny 月内节奏与高级统计信号 Dashboard")
st.caption(
    "Block 7-8 复刻 + OLS/RLS/面积积分/Chow Test/残差诊断。"
    "目标是把严谨统计过程转成可解释的业务规律：月初/月末 X 工作日累计占比与累计斜率受什么影响。"
)

with st.sidebar:
    st.header("1. 数据与范围")
    source_name = st.radio(
        "数据源",
        ["sales_30d_daily.csv"],
        index=0,
        format_func=lambda _: "data/sales_30d_daily.csv",
    )
    data_path = DEFAULT_DAILY_PATH
    start_yyyymm = st.text_input("START_YYYYMM", value="202201")
    end_yyyymm = st.text_input("END_YYYYMM（空=到数据末尾）", value="")

    st.divider()
    st.header("2. Anchor 与窗口")
    selected_anchor = st.radio("预测锚点", options=DEFAULT_ANCHORS, index=0, horizontal=True, format_func=lambda x: f"D-{x}")
    analysis_window = st.slider("X 工作日窗口", 3, 10, 5)
    selected_month = st.slider("工作日曲线月份", 1, 12, 2)
    curve_mode = st.radio("曲线指标", ["累计占比", "日销量"], horizontal=True)

    st.divider()
    st.header("3. 特征与相关证据")
    corr_method = st.radio("相关方法", ["pearson", "spearman"], horizontal=True)
    corr_threshold = st.slider("|相关系数| 阈值", 0.10, 0.80, 0.25, step=0.05)
    min_sample_size = st.slider("最小样本数", 5, 24, 6)
    include_d5_extra = st.checkbox("纳入 D-5 月中特征", value=(selected_anchor == 5))

    st.divider()
    st.header("4. 高级统计参数")
    enable_low_qty_filter = st.checkbox("过滤极低量工作日", value=False)
    low_qty_quantile = st.slider("极低量分位数", 0.01, 0.20, 0.05, step=0.01)
    ma_window = st.radio("MA 平滑窗口", [1, 3, 6, 12], index=2, format_func=lambda x: "不平滑" if x == 1 else f"MA({x})", horizontal=True)
    rls_lambda = st.slider("RLS/EWOLS 遗忘因子 λ", 0.80, 1.00, 1.00, step=0.01)
    area_window = st.slider("面积积分窗口 W", 3, 10, 6)
    area_threshold = st.slider("R_t 异常阈值", 0.05, 0.80, 0.20, step=0.05)
    chow_alpha = st.slider("Chow alpha", 0.01, 0.20, 0.05, step=0.01)
    enable_advanced = st.checkbox("启用高级诊断", value=True)

try:
    prepared = load_and_prepare_data(str(data_path), start_yyyymm, end_yyyymm)
except Exception as exc:
    st.error(f"数据加载失败：{exc}")
    st.stop()

daily = prepared["daily"]
monthly = prepared["monthly"]
raw = prepared["raw"]
workday_source = prepared["workday_source"]
anchors = DEFAULT_ANCHORS

anchor_tables = compute_anchor_tables(daily, anchors, enable_low_qty_filter, low_qty_quantile)
features = compute_feature_tables(daily, monthly, analysis_window, anchors, enable_low_qty_filter, low_qty_quantile)
feature_detail = features["month_feature_detail"]
anchor_safe_cols = features["anchor_safe_feature_cols"]
d5_extra_cols = features["anchor_d5_safe_extra_feature_cols"]
leakage_cols = features["leakage_feature_cols"]
target_cols = features["target_cols"]
corr_df = compute_correlations(
    feature_detail,
    target_cols,
    anchor_safe_cols,
    d5_extra_cols,
    leakage_cols,
    include_d5_extra,
    corr_method,
    corr_threshold,
    min_sample_size,
)

adv = {"signal": pd.DataFrame(), "chow": pd.DataFrame(), "area_anomaly": pd.DataFrame()}
diagnostics = pd.DataFrame()
if enable_advanced:
    adv = compute_advanced_methods(
        monthly,
        ma_window=ma_window,
        warmup=6,
        lam=rls_lambda,
        area_window=area_window,
        threshold=area_threshold,
        chow_alpha=chow_alpha,
        chow_window=max(3, min(6, area_window)),
        eps_ratio=0.25,
    )
    diagnostics = compute_diagnostics(monthly)

complete_months = int(monthly["is_complete_month"].sum())
incomplete_months = int((~monthly["is_complete_month"]).sum())
top_unstable = anchor_tables["anchor_stability"].sort_values("anchor_mtd_pct_std", ascending=False).head(1)
top_unstable_label = "-"
if not top_unstable.empty:
    top_unstable_label = f"{int(top_unstable['month'].iloc[0]):02d}月 D-{int(top_unstable['forecast_offset'].iloc[0])}"

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("数据月份", f"{monthly['bizym'].min()}-{monthly['bizym'].max()}")
m2.metric("完整月份", f"{complete_months} / {len(monthly)}")
m3.metric("工作日口径", workday_source)
m4.metric("最不稳定锚点", top_unstable_label)
m5.metric("泄露安全特征", f"{len(anchor_safe_cols) + (len(d5_extra_cols) if include_d5_extra else 0)}")

tabs = st.tabs(["Scientific Process", "Anchor Rhythm", "Feature Evidence", "Diagnostics", "Business Readout", "Data Quality"])

with tabs[0]:
    st.subheader("统计分析过程：从月度聚合到面积异常证据")
    st.plotly_chart(plot_monthly_series(monthly), use_container_width=True)

    if not enable_advanced:
        st.info("左侧开启“高级诊断”后显示 OLS/RLS/面积积分/Chow Test。")
    elif adv["signal"].empty:
        st.info("当前样本长度不足，无法稳定运行 RLS/面积积分流程。")
    else:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.plotly_chart(plot_advanced_signal(adv["signal"], area_threshold), use_container_width=True)
        with c2:
            signal = adv["signal"]
            st.markdown("**方法证据摘要**")
            st.metric("候选异常点", int(signal["candidate_anomaly"].sum()))
            st.metric("回测确认异常", int(signal["confirmed_anomaly"].sum()))
            st.metric("回测取消", int(signal["backtest_cancelled"].sum()))
            confirmed = signal[signal["confirmed_anomaly"]]
            if not confirmed.empty:
                st.warning("确认异常月份：" + "、".join(confirmed["month_label"].tolist()))
            else:
                st.success("当前参数下未发现回测确认的面积异常。")

        st.markdown("**RLS / EWOLS / 面积信号明细**")
        signal_show = adv["signal"].copy()
        signal_show["R_t_rolling"] = signal_show["R_t_rolling"].map(lambda x: "" if pd.isna(x) else f"{x:.2%}")
        signal_show["R_t_fixed"] = signal_show["R_t_fixed"].map(lambda x: "" if pd.isna(x) else f"{x:.2%}")
        st.dataframe(signal_show, use_container_width=True, hide_index=True)

    st.markdown("**科学性说明**")
    st.markdown(
        "- OLS/EWOLS/RLS 用来估计销量基准水平和趋势斜率；RLS 适合逐月在线更新。\n"
        "- 面积积分把截距和斜率合成一个趋势体量指标，`R_t=(A-B)/B` 衡量趋势面积是否突然抬升。\n"
        "- Chow Test 用 RSS/F/p-value 验证某月前后是否更像两条不同趋势线，而不是同一条趋势线上的普通波动。\n"
        "- 回测取消用于过滤短期波动：若候选异常后续面积变化率回落，则不视为稳定结构变化。"
    )

with tabs[1]:
    st.subheader("Block 7：月内累计节奏与 Anchor 稳定性")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.plotly_chart(plot_anchor_trend(anchor_tables["anchor_rows"], selected_anchor), use_container_width=True)
    with c2:
        stability = anchor_tables["anchor_stability"].copy()
        st.markdown("**高波动 Anchor 月份**")
        unstable = stability.sort_values("anchor_mtd_pct_std", ascending=False).head(8).copy()
        for col in ["anchor_mtd_pct_mean", "anchor_mtd_pct_std", "range_mtd_pct"]:
            unstable[col] = unstable[col].map(fmt_pct)
        st.dataframe(unstable, use_container_width=True, hide_index=True)

    st.plotly_chart(plot_workday_curves(anchor_tables["workday_filtered"], selected_month, curve_mode), use_container_width=True)

    stability_full = anchor_tables["anchor_stability"].copy()
    stability_full["anchor_mtd_pct_mean_fmt"] = stability_full["anchor_mtd_pct_mean"].map(fmt_pct)
    stability_full["anchor_mtd_pct_std_fmt"] = stability_full["anchor_mtd_pct_std"].map(fmt_pct)
    st.dataframe(stability_full, use_container_width=True, hide_index=True)
    download_button(stability_full, "下载 Anchor 稳定性 CSV", "anchor_stability.csv")

with tabs[2]:
    st.subheader("Block 8：泄露安全特征与影响证据")
    c1, c2, c3 = st.columns(3)
    c1.metric("全 anchor 可用特征", len(anchor_safe_cols))
    c2.metric("D-5 月中特征", len(d5_extra_cols))
    c3.metric("排除泄露字段", len(leakage_cols))

    with st.expander("泄露控制规则", expanded=True):
        st.markdown(
            "D-10 只使用月初已发生销量形态、日历结构、历史已知贡献节奏；"
            "D-5 可额外使用已完整发生的月中特征。当前月全月分母、当前月月末窗口结果、当前月月末贡献等字段不作为解释变量。"
        )
        st.dataframe(pd.DataFrame({"excluded_leakage_col": leakage_cols, "中文含义": [metric_col_to_cn(c) for c in leakage_cols]}), use_container_width=True, hide_index=True)

    if corr_df.empty:
        st.info("当前阈值和样本数下没有命中的相关特征；可降低阈值或扩大月份范围。")
    else:
        target_options = corr_df["target_cn"].drop_duplicates().tolist()
        selected_target_cn = st.selectbox("选择业务目标指标", target_options)
        st.plotly_chart(plot_corr_bar(corr_df, selected_target_cn), use_container_width=True)

        corr_show = corr_df.copy()
        for col in ["corr_coef", "abs_corr_coef", "p_value", "fdr_q_value"]:
            corr_show[col] = corr_show[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
        st.dataframe(corr_show, use_container_width=True, hide_index=True)
        download_button(corr_df, "下载相关证据 CSV", "feature_correlation_evidence.csv")

        summary = (
            corr_df.groupby(["target_cn", "feature_group"], as_index=False)
            .agg(
                feature_count=("feature", "nunique"),
                max_abs_corr=("abs_corr_coef", "max"),
                median_abs_corr=("abs_corr_coef", "median"),
                min_fdr_q=("fdr_q_value", "min"),
            )
            .sort_values(["target_cn", "max_abs_corr"], ascending=[True, False])
        )
        st.markdown("**变量类型汇总**")
        st.dataframe(summary, use_container_width=True, hide_index=True)

    st.markdown("**特征明细样本**")
    if not feature_detail.empty:
        cols = ["bizym", "year", "month", "month_total"] + target_cols[:4]
        cols += [c for c in anchor_safe_cols[:8] if c not in cols]
        cols += [c for c in d5_extra_cols[:6] if c not in cols and include_d5_extra]
        show = feature_detail[[c for c in cols if c in feature_detail.columns]].copy()
        show = show.rename(columns={c: metric_col_to_cn(c) for c in show.columns})
        st.dataframe(show, use_container_width=True, hide_index=True)
        download_button(feature_detail, "下载完整特征表 CSV", "month_feature_detail.csv")

with tabs[3]:
    st.subheader("统计检验与模型诊断")
    if not enable_advanced:
        st.info("左侧开启“高级诊断”后显示残差和平稳性检验。")
    else:
        st.markdown("**残差/平稳性/波动诊断**")
        st.dataframe(diagnostics, use_container_width=True, hide_index=True)

        c1, c2 = st.columns(2)
        with c1:
            if adv["chow"].empty:
                st.info("Chow Test 样本不足或未产生有效检验点。")
            else:
                st.plotly_chart(plot_chow(adv["chow"]), use_container_width=True)
                chow_show = adv["chow"].copy()
                chow_show["p_value"] = chow_show["p_value"].map(lambda x: f"{x:.4f}")
                st.dataframe(chow_show, use_container_width=True, hide_index=True)
        with c2:
            area_anomaly = adv["area_anomaly"]
            if area_anomaly.empty:
                st.info("二次面积曲线预测区间样本不足。")
            else:
                st.markdown("**扩展窗口 OLS + 二次面积曲线 + t预测区间**")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=area_anomaly["month_label"], y=area_anomaly["B_t"], name="B_t", line=dict(color=COLOR_MAIN)))
                fig.add_trace(go.Scatter(x=area_anomaly["month_label"], y=area_anomaly["pred_B_t"], name="预测B_t", line=dict(color=COLOR_BLUE)))
                fig.add_trace(go.Scatter(x=area_anomaly["month_label"], y=area_anomaly["pred_hi"], name="上界", line=dict(color="gray", dash="dash")))
                fig.add_trace(go.Scatter(x=area_anomaly["month_label"], y=area_anomaly["pred_lo"], name="下界", line=dict(color="gray", dash="dash")))
                hits = area_anomaly[area_anomaly["area_anomaly"]]
                if not hits.empty:
                    fig.add_trace(go.Scatter(x=hits["month_label"], y=hits["B_t"], mode="markers", name="面积异常", marker=dict(color=COLOR_DANGER, size=11)))
                fig.update_layout(height=400, title="面积曲线预测区间与趋势反转证据", legend_orientation="h")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(area_anomaly, use_container_width=True, hide_index=True)

with tabs[4]:
    st.subheader("业务解释：月初/月末 X 工作日占比与斜率受什么影响？")
    readout = business_recommendations(corr_df, analysis_window)
    if readout.empty:
        st.info("当前参数下暂无足够证据形成业务解释；建议降低相关阈值或扩大时间范围。")
    else:
        st.dataframe(readout, use_container_width=True, hide_index=True)

    st.markdown("**业务读法**")
    st.markdown(
        f"- 月初{analysis_window}工作日累计占比：适合回答“早期节奏是否已经过快/过慢”。优先看月初销量形态、anchor日期位置和历史同月节奏。\n"
        f"- 月末{analysis_window}工作日累计占比：适合回答“最后冲量/回补是否明显”。优先看月末自然日节假日、当月工作日数量、前序斜率和历史月末贡献。\n"
        f"- 月初{analysis_window}工作日累计斜率：表示月初每天累积速度是否稳定抬升，若与月初日销量斜率、首两日占比高度相关，可作为 D-10 预警。\n"
        f"- 月末{analysis_window}工作日累计斜率：表示月末收口速度，若受节假日/非工作日夹杂影响大，预测时应单独做日历修正。"
    )

    if not anchor_tables["anchor_stability"].empty:
        unstable_months = (
            anchor_tables["anchor_stability"]
            .sort_values("anchor_mtd_pct_std", ascending=False)
            .head(5)
            .assign(label=lambda d: d["month"].map(lambda x: f"{int(x):02d}月") + " D-" + d["forecast_offset"].astype(int).astype(str))
        )
        st.warning("优先人工复核的高波动月份：" + "、".join(unstable_months["label"].tolist()))

with tabs[5]:
    st.subheader("数据质量与原始数据")
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("原始行数", fmt_num(len(raw)))
    q2.metric("分析后日行数", fmt_num(len(daily)))
    q3.metric("缺失自然日", fmt_num(monthly["missing_calendar_days"].sum()))
    q4.metric("负销量日", fmt_num((daily["qty"] < 0).sum()))

    st.plotly_chart(
        px.bar(
            monthly,
            x="month_label",
            y="missing_calendar_days",
            color="is_complete_month",
            title="月份完整性：缺失自然日数量",
            labels={"month_label": "月份", "missing_calendar_days": "缺失自然日"},
        ),
        use_container_width=True,
    )

    st.markdown("**月度质量表**")
    st.dataframe(monthly, use_container_width=True, hide_index=True)
    download_button(monthly, "下载月度质量 CSV", "monthly_quality.csv")

    st.markdown("**日明细预览**")
    st.dataframe(daily.head(500), use_container_width=True, hide_index=True)
