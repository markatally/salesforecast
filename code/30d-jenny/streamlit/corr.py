"""
Interactive statistical dashboard for pre-modeling correlation analysis.

Run:
    streamlit run code/30d-jenny/streamlit/corr.py
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

try:
    from statsmodels.tsa.stattools import adfuller, kpss
except Exception:  # pragma: no cover - app degrades gracefully if statsmodels is unavailable.
    adfuller = None
    kpss = None

try:
    from chinese_calendar import is_workday as cn_is_workday
except Exception:  # pragma: no cover - app works with a weekday fallback.
    cn_is_workday = None


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DAILY_PATH = REPO_ROOT / "data" / "sales_daily.csv"

GRAIN_OPTIONS = ["年", "月"]
VALUE_TYPE_OPTIONS = ["统计值", "环比值", "同比值"]
SMOOTHING_METHOD_OPTIONS = ["SMA", "EMA", "Median"]
STATIONARITY_METHOD_OPTIONS = ["KPSS", "ADF"]
SLOPE_METRIC_OPTIONS = {
    "月末 X 工作日累计销量斜率": "end",
    "月初 X 工作日累计销量斜率": "start",
}
CONTRIBUTION_SEGMENT_OPTIONS = {
    "期初25%": 0,
    "中前25%": 1,
    "中后25%": 2,
    "期末25%": 3,
}
DEFAULT_STATIONARITY_ALPHA = 0.05
RELATIVE_PERIOD_LABELS = {
    "prev": "上月",
    "curr": "当月",
    "next": "下月",
}
ANALYSIS_TARGET = "工作日平均销量"
CHART_HEIGHT = 360
INLINE_CONTROL_GAP = [0.22, 0.78]
TITLE_CONTROL_LAYOUT = [0.36, 0.64]
COMPACT_NUMBER_INPUT_WIDTH = 116
COMPACT_SELECT_WIDTH = 360
TABLE_VISIBLE_ROWS = 12
TABLE_ROW_HEIGHT = 35
TABLE_HEADER_HEIGHT = 38
MIN_SEGMENT_MONTHS = 8
MAX_CHANGEPOINTS = 3
MAX_STRUCTURAL_BREAKS = 2
EWOLS_HALFLIFE_MONTHS = 6.0
EWOLS_FORECAST_MIN_TRAIN_MONTHS = 10
CHANGEPOINT_PENALTY_SCALE = 2.0
CV_STABLE_MAX = 0.20
YOY_STABLE_MAX = 0.06
MAX_YOY_STABLE_MAX = 0.05
LOYO_WAPE_STABLE_MAX = 0.20
STABILITY_LABEL_STABLE = "稳定"
STABILITY_LABEL_WATCH = "需观察"
STABILITY_LABEL_UNSTABLE = "不稳定"
STABILITY_ORDER = {
    STABILITY_LABEL_STABLE: 0,
    STABILITY_LABEL_WATCH: 1,
    STABILITY_LABEL_UNSTABLE: 2,
}
COLOR_MAIN = "#2f6f73"
COLOR_ACCENT = "#d95f02"
COLOR_DANGER = "#c44e52"
COLOR_BLUE = "#4c72b0"
COLOR_MUTED = "#7a7f87"
CALENDAR_SUFFIXES = {
    "workdays",
    "non_workdays",
    "weekday_holidays",
    "weekend_rest_days",
    "calendar_days",
    "holiday_share",
    "workday_share",
}
HISTORICAL_SUFFIXES = {
    "avg_qty_per_workday",
    "avg_qty_per_calendar_day",
    "avg_num_hosp_per_workday",
}
NON_CALENDAR_BASE_SUFFIXES = {
    "avg_qty_per_workday",
    "avg_qty_per_calendar_day",
    "avg_num_hosp_per_workday",
}


@dataclass(frozen=True)
class SegmentFit:
    start: int
    end: int
    rss: float
    coef: np.ndarray
    nobs: int


def render_inline_label(label: str) -> None:
    st.markdown(
        (
            "<div style='white-space:nowrap;text-align:right;"
            "padding-right:1em;color:rgba(49, 51, 63, 0.70);"
            "font-size:0.875rem;line-height:2.4rem;'>"
            f"{label}</div>"
        ),
        unsafe_allow_html=True,
    )


def is_business_workday(ts: pd.Timestamp) -> bool:
    if cn_is_workday is not None:
        return bool(cn_is_workday(pd.Timestamp(ts).date()))
    return pd.Timestamp(ts).dayofweek < 5


def fmt_num(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{value:,.0f}"


def fmt_float(value: float | int | None, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):,.{digits}f}"


def fmt_pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.2f}%"


def fmt_pct_1(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.1f}%"


def fmt_corr(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):+.3f}"


def fmt_pvalue(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    value = float(value)
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def fmt_metric_value(value: float | int | None, value_type: str) -> str:
    if value is None or pd.isna(value):
        return "-"
    if value_type in {"环比值", "同比值"}:
        return f"{float(value) * 100:+.1f}%"
    return fmt_num(value)


def suffix_for_value_type(base_suffix: str, value_type: str) -> str:
    if value_type == "环比值":
        return f"{base_suffix}_mom_pct"
    if value_type == "同比值":
        return f"{base_suffix}_yoy_pct"
    return base_suffix


def target_col_for_value_type(value_type: str) -> str:
    return f"curr_{suffix_for_value_type('avg_qty_per_workday', value_type)}"


def target_label_for_value_type(value_type: str) -> str:
    if value_type == "环比值":
        return "当月平均每工作日销量环比"
    if value_type == "同比值":
        return "当月平均每工作日销量同比"
    return "当月平均每工作日销量"


def chart_number_format(value_type: str) -> str:
    return ".1%" if value_type in {"环比值", "同比值"} else ",.0f"


def safe_divide(numerator: float | int | None, denominator: float | int | None) -> float:
    if numerator is None or denominator is None or pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
        return np.nan
    return float(numerator) / float(denominator)


def linear_slope(values: pd.Series | np.ndarray) -> float:
    y = pd.Series(values).dropna().astype(float)
    if len(y) < 2:
        return np.nan
    x = np.arange(1, len(y) + 1, dtype=float)
    return float(np.polyfit(x, y.to_numpy(), 1)[0])


def dataframe_height(row_count: int) -> int:
    visible_rows = min(max(int(row_count), 1), TABLE_VISIBLE_ROWS)
    return TABLE_HEADER_HEIGHT + visible_rows * TABLE_ROW_HEIGHT


def show_dataframe_12_rows(df: pd.DataFrame, **kwargs) -> None:
    st.dataframe(df, height=dataframe_height(len(df)), **kwargs)


def group_columns(grains: list[str]) -> list[str]:
    cols: list[str] = []
    if "年" in grains:
        cols.append("year")
    if "月" in grains:
        cols.append("month")
    return cols


def make_group_label(row: pd.Series, grains: list[str]) -> str:
    if grains == ["年"]:
        return f"{int(row['year'])}年"
    if grains == ["月"]:
        return f"{int(row['month']):02d}月"
    if set(grains) == {"年", "月"}:
        return f"{int(row['year'])}-{int(row['month']):02d}"
    return "全部"


def effect_size_label(abs_corr: float) -> str:
    if pd.isna(abs_corr):
        return "样本不足"
    if abs_corr >= 0.50:
        return "强相关"
    if abs_corr >= 0.30:
        return "中等相关"
    if abs_corr >= 0.10:
        return "弱相关"
    return "很弱"


def relationship_direction(corr: float) -> str:
    if pd.isna(corr):
        return "样本不足"
    if corr > 0:
        return "正向"
    if corr < 0:
        return "负向"
    return "近似无方向"


@st.cache_data(show_spinner="加载日销量数据...")
def load_daily_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"bizym", "transdate", "qty"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df.copy()
    out["bizym"] = out["bizym"].astype(int)
    out["transdate"] = pd.to_datetime(out["transdate"])
    out["qty"] = pd.to_numeric(out["qty"], errors="coerce").fillna(0.0).clip(lower=0.0)
    if "num_hosp" in out.columns:
        out["num_hosp"] = pd.to_numeric(out["num_hosp"], errors="coerce").fillna(0.0).clip(lower=0.0)
    else:
        out["num_hosp"] = np.nan

    out["month_start"] = out["transdate"].dt.to_period("M").dt.to_timestamp()
    out["month_label"] = out["month_start"].dt.strftime("%Y-%m")
    out["year"] = out["bizym"] // 100
    out["month"] = out["bizym"] % 100
    out["day_of_week"] = out["transdate"].dt.dayofweek
    out["is_weekend"] = out["day_of_week"].ge(5)
    out["is_workday"] = out["transdate"].map(is_business_workday)
    out["is_weekday_holiday"] = (~out["is_workday"]) & (~out["is_weekend"])
    out["is_weekend_rest_day"] = (~out["is_workday"]) & out["is_weekend"]
    out = out.sort_values(["bizym", "transdate"]).reset_index(drop=True)
    out["workday_seq"] = out.groupby("bizym")["is_workday"].cumsum().where(out["is_workday"])
    month_workdays = (
        out[out["is_workday"]]
        .groupby("bizym", as_index=False)["workday_seq"]
        .max()
        .rename(columns={"workday_seq": "max_workday_seq"})
    )
    out = out.merge(month_workdays, on="bizym", how="left")
    out["mtd_qty"] = out.groupby("bizym")["qty"].cumsum()
    month_total = out.groupby("bizym", as_index=False)["qty"].sum().rename(columns={"qty": "actual_month_total"})
    out = out.merge(month_total, on="bizym", how="left")
    out["mtd_pct"] = out["mtd_qty"] / out["actual_month_total"].replace(0, np.nan)
    return out.sort_values(["bizym", "transdate"]).reset_index(drop=True)


def month_calendar_stats(month_start: pd.Timestamp) -> dict[str, int]:
    dates = pd.date_range(month_start, month_start + pd.offsets.MonthEnd(0), freq="D")
    cal = pd.DataFrame({"transdate": dates})
    cal["is_weekend"] = cal["transdate"].dt.dayofweek.ge(5)
    cal["is_workday"] = cal["transdate"].map(is_business_workday)
    cal["is_weekday_holiday"] = (~cal["is_workday"]) & (~cal["is_weekend"])
    cal["is_weekend_rest_day"] = (~cal["is_workday"]) & cal["is_weekend"]
    return {
        "calendar_days": int(len(cal)),
        "workdays": int(cal["is_workday"].sum()),
        "non_workdays": int((~cal["is_workday"]).sum()),
        "weekday_holidays": int(cal["is_weekday_holiday"].sum()),
        "weekend_rest_days": int(cal["is_weekend_rest_day"].sum()),
    }


@st.cache_data(show_spinner="构建月度统计口径...")
def build_monthly_stats(daily: pd.DataFrame) -> pd.DataFrame:
    observed = (
        daily.groupby(["bizym", "year", "month", "month_start"], as_index=False)
        .agg(
            month_total_qty=("qty", "sum"),
            month_total_num_hosp=("num_hosp", "sum"),
            observed_days=("transdate", "nunique"),
            observed_workdays=("is_workday", "sum"),
            observed_non_workdays=("is_workday", lambda s: int((~s).sum())),
            observed_weekday_holidays=("is_weekday_holiday", "sum"),
            observed_weekend_rest_days=("is_weekend_rest_day", "sum"),
        )
        .sort_values("month_start")
        .reset_index(drop=True)
    )

    calendar_rows = []
    for month_start in observed["month_start"].drop_duplicates().sort_values():
        row = {"month_start": pd.Timestamp(month_start)}
        row.update(month_calendar_stats(pd.Timestamp(month_start)))
        calendar_rows.append(row)

    out = observed.merge(pd.DataFrame(calendar_rows), on="month_start", how="left")
    out["avg_qty_per_workday"] = out["month_total_qty"] / out["workdays"].replace(0, np.nan)
    out["avg_qty_per_calendar_day"] = out["month_total_qty"] / out["calendar_days"].replace(0, np.nan)
    out["avg_qty_per_observed_workday"] = out["month_total_qty"] / out["observed_workdays"].replace(0, np.nan)
    out["avg_num_hosp_per_workday"] = out["month_total_num_hosp"] / out["workdays"].replace(0, np.nan)
    out["holiday_share"] = out["weekday_holidays"] / out["calendar_days"].replace(0, np.nan)
    out["workday_share"] = out["workdays"] / out["calendar_days"].replace(0, np.nan)
    out["is_complete_month"] = out["observed_days"].eq(out["calendar_days"])

    out = out.sort_values("month_start").reset_index(drop=True)
    for suffix in NON_CALENDAR_BASE_SUFFIXES:
        out[f"{suffix}_mom_pct"] = out[suffix] / out[suffix].shift(1).replace(0, np.nan) - 1
        out[f"{suffix}_yoy_pct"] = (
            out[suffix]
            / out.groupby("month")[suffix].shift(1).replace(0, np.nan)
            - 1
        )
    return out


def build_neighbor_panel(monthly: pd.DataFrame) -> pd.DataFrame:
    monthly = monthly.sort_values("month_start").copy()
    monthly["same_month_past_avg_qty_per_workday"] = (
        monthly.groupby("month")["avg_qty_per_workday"]
        .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    )
    monthly["m1_vs_m2_avg_qty_per_workday_change"] = (
        monthly["avg_qty_per_workday"].shift(1) / monthly["avg_qty_per_workday"].shift(2).replace(0, np.nan) - 1
    )

    base_cols = [
        "bizym",
        "month_start",
        "year",
        "month",
        "month_total_qty",
        "month_total_num_hosp",
        "avg_qty_per_workday",
        "avg_qty_per_workday_mom_pct",
        "avg_qty_per_workday_yoy_pct",
        "avg_qty_per_calendar_day",
        "avg_qty_per_calendar_day_mom_pct",
        "avg_qty_per_calendar_day_yoy_pct",
        "avg_num_hosp_per_workday",
        "avg_num_hosp_per_workday_mom_pct",
        "avg_num_hosp_per_workday_yoy_pct",
        "calendar_days",
        "workdays",
        "non_workdays",
        "weekday_holidays",
        "weekend_rest_days",
        "holiday_share",
        "workday_share",
        "is_complete_month",
        "same_month_past_avg_qty_per_workday",
        "m1_vs_m2_avg_qty_per_workday_change",
    ]
    base = monthly[base_cols].copy()
    parts = []
    for offset, prefix in [(-1, "prev"), (0, "curr"), (1, "next")]:
        part = base.copy()
        part["analysis_month_start"] = part["month_start"] - pd.DateOffset(months=offset)
        rename_cols = {
            col: f"{prefix}_{col}"
            for col in base_cols
            if col not in {"month_start", "year", "month"}
        }
        part = part.rename(columns=rename_cols)
        keep_cols = ["analysis_month_start"] + list(rename_cols.values())
        parts.append(part[keep_cols])

    panel = monthly[["bizym", "month_start", "year", "month"]].rename(
        columns={"month_start": "analysis_month_start"}
    )
    for part in parts:
        panel = panel.merge(part, on="analysis_month_start", how="left")

    panel = panel.rename(columns={"analysis_month_start": "month_start"})
    panel["month_label"] = panel["month_start"].dt.strftime("%Y-%m")
    return panel.sort_values("month_start").reset_index(drop=True)


def aggregate_panel(panel: pd.DataFrame, grains: list[str]) -> pd.DataFrame:
    group_cols = group_columns(grains)
    metric_cols = [
        c
        for c in panel.columns
        if any(c.startswith(f"{prefix}_") for prefix in RELATIVE_PERIOD_LABELS)
        and pd.api.types.is_numeric_dtype(panel[c])
    ]

    if group_cols:
        out = panel.groupby(group_cols, as_index=False)[metric_cols].mean(numeric_only=True)
        out["样本月份数"] = panel.groupby(group_cols).size().to_numpy()
    else:
        out = pd.DataFrame({col: [panel[col].mean()] for col in metric_cols})
        out["样本月份数"] = len(panel)

    out["分析粒度"] = out.apply(lambda row: make_group_label(row, grains), axis=1)
    out["散点标签"] = out["分析粒度"]
    return out


def candidate_metric_items(value_type: str) -> list[dict[str, str]]:
    metric_labels = {
        "workdays": "工作日天数",
        "avg_qty_per_workday": "平均每工作日销量",
        "avg_qty_per_workday_mom_pct": "平均每工作日销量环比",
        "avg_qty_per_workday_yoy_pct": "平均每工作日销量同比",
        "avg_qty_per_calendar_day": "平均自然日销量",
        "avg_qty_per_calendar_day_mom_pct": "平均自然日销量环比",
        "avg_qty_per_calendar_day_yoy_pct": "平均自然日销量同比",
        "avg_num_hosp_per_workday": "平均每工作日医院数",
        "avg_num_hosp_per_workday_mom_pct": "平均每工作日医院数环比",
        "avg_num_hosp_per_workday_yoy_pct": "平均每工作日医院数同比",
        "non_workdays": "非工作日天数",
        "weekday_holidays": "假期天数（非周末）",
        "weekend_rest_days": "周末休息天数（仅周末）",
        "calendar_days": "自然日天数",
        "holiday_share": "假期占比",
        "workday_share": "工作日占比",
    }

    items: list[dict[str, str]] = []
    for prefix, period_label in RELATIVE_PERIOD_LABELS.items():
        if prefix == "prev":
            allowed_suffixes = CALENDAR_SUFFIXES | {
                suffix_for_value_type(base_suffix, value_type)
                for base_suffix in HISTORICAL_SUFFIXES
            }
        elif prefix == "curr":
            allowed_suffixes = CALENDAR_SUFFIXES
        else:
            allowed_suffixes = CALENDAR_SUFFIXES

        for suffix in sorted(allowed_suffixes):
            items.append(
                {
                    "period": period_label,
                    "metric": metric_labels[suffix],
                    "field": f"{prefix}_{suffix}",
                }
            )

    items.extend(
        [
            {
                "period": "历史",
                "metric": "过去平均相同月份的平均工作日销量",
                "field": "curr_same_month_past_avg_qty_per_workday",
            },
            {
                "period": "历史",
                "metric": "M-1 相对 M-2 平均工作日销量变化",
                "field": "curr_m1_vs_m2_avg_qty_per_workday_change",
            },
        ]
    )
    return items


def corr_pvalue(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    pair = pd.concat([x, y], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    n = int(len(pair))
    if n < 3 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return np.nan, np.nan, n

    corr = float(pair.iloc[:, 0].corr(pair.iloc[:, 1], method="pearson"))
    try:
        from scipy import stats

        _, p_value = stats.pearsonr(pair.iloc[:, 0], pair.iloc[:, 1])
    except Exception:
        p_value = np.nan
    return corr, float(p_value) if pd.notna(p_value) else np.nan, n


def classify_binary_threshold(value: float | int | None, stable_max: float) -> str:
    if value is None or pd.isna(value):
        return STABILITY_LABEL_UNSTABLE
    return STABILITY_LABEL_STABLE if abs(float(value)) <= stable_max else STABILITY_LABEL_UNSTABLE


def worst_stability_label(labels: pd.Series | list[str]) -> str:
    valid = [label for label in list(labels) if label in STABILITY_ORDER]
    if not valid:
        return STABILITY_LABEL_WATCH
    return max(valid, key=lambda label: STABILITY_ORDER[label])


def add_binary_stability_columns(
    df: pd.DataFrame,
    value_col: str,
    stable_max: float,
    basis_template: str,
) -> pd.DataFrame:
    out = df.copy()
    if out.empty or value_col not in out.columns:
        out["稳定性"] = pd.Series(dtype="object")
        out["判定依据"] = pd.Series(dtype="object")
        return out
    out["稳定性"] = out[value_col].map(lambda value: classify_binary_threshold(value, stable_max))
    out["判定依据"] = out[value_col].map(
        lambda value: basis_template.format(
            value=fmt_pct(abs(value) * 100) if pd.notna(value) else "-",
            stable=fmt_pct(stable_max * 100),
        )
    )
    return out


def stability_summary(df: pd.DataFrame, value_col: str) -> tuple[str, str]:
    if df.empty or value_col not in df.columns or "稳定性" not in df.columns:
        return STABILITY_LABEL_WATCH, "有效样本不足，暂按需观察处理。"

    valid = df.dropna(subset=[value_col])
    if valid.empty:
        return STABILITY_LABEL_WATCH, "有效样本不足，暂按需观察处理。"

    label = worst_stability_label(valid["稳定性"])
    unstable_count = int(valid["稳定性"].eq(STABILITY_LABEL_UNSTABLE).sum())
    total_count = int(len(valid))
    if unstable_count:
        reason = f"{total_count} 行中有 {unstable_count} 行不稳定，整体判定为不稳定。"
    else:
        reason = f"{total_count} 行全部满足稳定阈值，整体判定为稳定。"
    return label, reason


def stability_summary_from_labels(labels: pd.Series | list[str]) -> tuple[str, str]:
    valid = [label for label in list(labels) if label in STABILITY_ORDER]
    if not valid:
        return STABILITY_LABEL_WATCH, "有效样本不足，暂按需观察处理。"

    label = worst_stability_label(valid)
    unstable_count = valid.count(STABILITY_LABEL_UNSTABLE)
    total_count = len(valid)
    if unstable_count:
        reason = f"{total_count} 行中有 {unstable_count} 行不稳定，整体判定为不稳定。"
    else:
        reason = f"{total_count} 行全部满足稳定阈值，整体判定为稳定。"
    return label, reason


def render_stability_basis(label: str, reason: str, threshold_text: str) -> None:
    st.info(
        f"**整体标签：{label}**。{reason} 判定阈值：{threshold_text}",
        icon=":material/rule:",
    )


@st.cache_data(show_spinner="构建月度斜率与 anchor 预测误差...")
def build_monthly_analysis(daily: pd.DataFrame, position: str, window: int) -> pd.DataFrame:
    workdays = daily[daily["is_workday"]].copy()
    if workdays.empty:
        return pd.DataFrame()
    workdays["workday_seq"] = workdays["workday_seq"].astype(int)
    workdays["max_workday_seq"] = workdays["max_workday_seq"].astype(int)

    rows = []
    for bizym, g in workdays.groupby("bizym"):
        g = g.sort_values("workday_seq").copy()
        if len(g) < window:
            continue
        if position == "end":
            metric_window = g.tail(window).copy()
            anchor = metric_window.iloc[0]
            anchor_rule = f"月末倒数第 {window} 个工作日"
        else:
            metric_window = g.head(window).copy()
            anchor = metric_window.iloc[-1]
            anchor_rule = f"月初第 {window} 个工作日"

        cumsum_slope = linear_slope(metric_window["mtd_qty"])
        avg_daily_qty = metric_window["qty"].mean()
        actual_total = float(g["actual_month_total"].iloc[0])
        anchor_mtd = float(anchor["mtd_qty"])

        rows.append(
            {
                "bizym": int(bizym),
                "month_start": g["month_start"].iloc[0],
                "month_label": g["month_label"].iloc[0],
                "month": int(g["month"].iloc[0]),
                "actual_month_total": actual_total,
                "max_workday_seq": int(g["max_workday_seq"].iloc[0]),
                "window": int(window),
                "anchor_date": pd.Timestamp(anchor["transdate"]),
                "anchor_rule": anchor_rule,
                "anchor_workday_seq": int(anchor["workday_seq"]),
                "remaining_workdays_after_anchor": int(g["max_workday_seq"].iloc[0] - anchor["workday_seq"]),
                "anchor_mtd_qty": anchor_mtd,
                "anchor_mtd_pct": safe_divide(anchor_mtd, actual_total),
                "selected_slope": cumsum_slope,
                "selected_slope_per_avg_daily_qty": safe_divide(cumsum_slope, avg_daily_qty),
                "window_qty": float(metric_window["qty"].sum()),
                "window_mtd_start": float(metric_window["mtd_qty"].iloc[0]),
                "window_mtd_end": float(metric_window["mtd_qty"].iloc[-1]),
            }
        )

    out = pd.DataFrame(rows).sort_values("month_start").reset_index(drop=True)
    if out.empty:
        return out

    out["prev_month_actual_total"] = out["actual_month_total"].shift(1)
    return add_anchor_ewols_predictions(out)


@st.cache_data(show_spinner="构建工作日贡献率稳定性指标...")
def build_contribution_detail(daily: pd.DataFrame, selected_segments: tuple[str, ...], aggregate_segments: bool) -> pd.DataFrame:
    workdays = daily[daily["is_workday"]].copy()
    if workdays.empty or not selected_segments:
        return pd.DataFrame()

    workdays["workday_seq"] = workdays["workday_seq"].astype(int)
    workdays["max_workday_seq"] = workdays["max_workday_seq"].astype(int)
    workdays["segment_idx"] = np.minimum(
        np.floor((workdays["workday_seq"] - 1) * 4 / workdays["max_workday_seq"]).astype(int),
        3,
    )

    segment_lookup = {v: k for k, v in CONTRIBUTION_SEGMENT_OPTIONS.items()}
    selected_idx = [CONTRIBUTION_SEGMENT_OPTIONS[name] for name in selected_segments]
    rows = []
    for bizym, g in workdays.groupby("bizym", sort=True):
        actual_total = float(g["actual_month_total"].iloc[0])
        common = {
            "bizym": int(bizym),
            "year": int(g["year"].iloc[0]),
            "month": int(g["month"].iloc[0]),
            "month_start": g["month_start"].iloc[0],
            "month_label": g["month_label"].iloc[0],
            "actual_month_total": actual_total,
            "max_workday_seq": int(g["max_workday_seq"].iloc[0]),
        }

        if aggregate_segments:
            selected = g[g["segment_idx"].isin(selected_idx)]
            rows.append(
                {
                    **common,
                    "窗口": "+".join(selected_segments),
                    "窗口销量": float(selected["qty"].sum()),
                    "贡献率": safe_divide(float(selected["qty"].sum()), actual_total),
                    "起始工作日": int(selected["workday_seq"].min()) if not selected.empty else np.nan,
                    "结束工作日": int(selected["workday_seq"].max()) if not selected.empty else np.nan,
                }
            )
        else:
            for idx in selected_idx:
                selected = g[g["segment_idx"].eq(idx)]
                rows.append(
                    {
                        **common,
                        "窗口": segment_lookup[idx],
                        "窗口销量": float(selected["qty"].sum()),
                        "贡献率": safe_divide(float(selected["qty"].sum()), actual_total),
                        "起始工作日": int(selected["workday_seq"].min()) if not selected.empty else np.nan,
                        "结束工作日": int(selected["workday_seq"].max()) if not selected.empty else np.nan,
                    }
                )

    return pd.DataFrame(rows).sort_values(["month_start", "窗口"]).reset_index(drop=True)


def summarize_contribution_stability(detail: pd.DataFrame, grains: list[str]) -> dict[str, pd.DataFrame]:
    if detail.empty:
        empty = pd.DataFrame()
        return {"cv": empty, "yoy": empty, "max_yoy": empty, "loyo": empty}

    group_cols = ["窗口"] + group_columns(grains)
    stats = (
        detail.groupby(group_cols, dropna=False)
        .agg(
            平均贡献率=("贡献率", "mean"),
            贡献率标准差=("贡献率", "std"),
            样本月数=("贡献率", "count"),
        )
        .reset_index()
    )
    stats["CV"] = stats["贡献率标准差"] / stats["平均贡献率"].abs().replace(0, np.nan)
    stats["分析粒度"] = stats.apply(lambda row: make_group_label(row, grains), axis=1)

    yoy_base = detail.sort_values(["窗口", "month", "year"]).copy()
    yoy_base["上年贡献率"] = yoy_base.groupby(["窗口", "month"])["贡献率"].shift(1)
    yoy_base["YoY变化"] = yoy_base["贡献率"] - yoy_base["上年贡献率"]
    yoy = (
        yoy_base.groupby(group_cols, dropna=False)
        .agg(
            平均YoY变化=("YoY变化", "mean"),
            最大正向YoY变化=("YoY变化", "max"),
            最大负向YoY变化=("YoY变化", "min"),
            有效同比样本=("YoY变化", "count"),
        )
        .reset_index()
    )
    yoy["分析粒度"] = yoy.apply(lambda row: make_group_label(row, grains), axis=1)

    max_yoy = (
        yoy_base.dropna(subset=["YoY变化"])
        .groupby(group_cols, dropna=False)
        .agg(最大YoY变化=("YoY变化", lambda s: float(np.nanmax(np.abs(s)))), 有效同比样本=("YoY变化", "count"))
        .reset_index()
    )
    max_yoy["分析粒度"] = max_yoy.apply(lambda row: make_group_label(row, grains), axis=1) if not max_yoy.empty else ""

    loyo_rows = []
    for (window_name, month), g in detail.groupby(["窗口", "month"], sort=True):
        valid = g.dropna(subset=["贡献率", "窗口销量", "actual_month_total"])
        for test_year, test in valid.groupby("year"):
            train = valid[valid["year"].ne(test_year)]
            baseline = train["贡献率"].median()
            if len(train) == 0 or pd.isna(baseline) or baseline <= 0:
                continue
            pred_total = test["窗口销量"] / baseline
            abs_error = (pred_total - test["actual_month_total"]).abs().sum()
            actual_sum = test["actual_month_total"].abs().sum()
            loyo_rows.append(
                {
                    "窗口": window_name,
                    "month": int(month),
                    "测试年": int(test_year),
                    "历史同月贡献率中位数": float(baseline),
                    "绝对误差": float(abs_error),
                    "实际销量": float(actual_sum),
                }
            )
    loyo_detail = pd.DataFrame(loyo_rows)
    if loyo_detail.empty:
        loyo = pd.DataFrame()
    else:
        loyo_detail["year"] = loyo_detail["测试年"]
        loyo_group_cols = ["窗口"] + group_columns(grains)
        loyo = (
            loyo_detail.groupby(loyo_group_cols, dropna=False)
            .agg(绝对误差=("绝对误差", "sum"), 实际销量=("实际销量", "sum"), 留一年样本=("测试年", "count"))
            .reset_index()
        )
        loyo["LOYO-WAPE"] = loyo["绝对误差"] / loyo["实际销量"].replace(0, np.nan)
        loyo["分析粒度"] = loyo.apply(lambda row: make_group_label(row, grains), axis=1)

    return {"cv": stats, "yoy": yoy, "max_yoy": max_yoy, "loyo": loyo}


def make_anchor_forecast_features(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)
    features["anchor_mtd_qty"] = df["anchor_mtd_qty"].astype(float)
    features["anchor_workday_seq"] = df["anchor_workday_seq"].astype(float)
    features["remaining_workdays_after_anchor"] = df["remaining_workdays_after_anchor"].astype(float)
    features["max_workday_seq"] = df["max_workday_seq"].astype(float)
    features["prev_month_actual_total"] = df["prev_month_actual_total"].astype(float)
    month_angle = 2 * np.pi * (df["month"].astype(float) - 1) / 12
    features["month_sin"] = np.sin(month_angle)
    features["month_cos"] = np.cos(month_angle)
    return features


def standardize_by_train(train_x: pd.DataFrame, current_x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    means = train_x.mean(axis=0)
    stds = train_x.std(axis=0, ddof=0).replace(0, 1.0)
    train_z = (train_x - means) / stds
    current_z = (current_x - means) / stds
    train_mat = np.column_stack([np.ones(len(train_z)), train_z.to_numpy(dtype=float)])
    current_mat = np.column_stack([np.ones(len(current_z)), current_z.to_numpy(dtype=float)])
    return train_mat, current_mat


def ewols_fit(y: np.ndarray, x: np.ndarray, halflife: float) -> tuple[np.ndarray, float, np.ndarray]:
    n = len(y)
    if n <= x.shape[1]:
        return np.full(x.shape[1], np.nan), np.inf, np.full(n, np.nan)
    age = np.arange(n - 1, -1, -1, dtype=float)
    weights = np.power(0.5, age / max(halflife, 1e-6))
    w_sqrt = np.sqrt(weights)
    xw = x * w_sqrt[:, None]
    yw = y * w_sqrt
    coef, *_ = np.linalg.lstsq(xw, yw, rcond=None)
    resid = y - x @ coef
    rss = float(np.sum(weights * resid**2))
    return coef, rss, resid


def add_anchor_ewols_predictions(monthly: pd.DataFrame) -> pd.DataFrame:
    out = monthly.copy()
    out["anchor_pred_month_total"] = np.nan
    out["anchor_pred_error"] = np.nan
    out["anchor_pred_error_pct"] = np.nan
    out["ewols_train_months"] = np.nan

    all_features = make_anchor_forecast_features(out)
    y_all = out["actual_month_total"].astype(float)

    for idx in range(len(out)):
        train_x = all_features.iloc[:idx].copy()
        train_y = y_all.iloc[:idx].copy()
        current_x = all_features.iloc[[idx]].copy()

        valid_train = train_x.notna().all(axis=1) & train_y.notna()
        if int(valid_train.sum()) < EWOLS_FORECAST_MIN_TRAIN_MONTHS or current_x.isna().any(axis=None):
            continue

        x_train_mat, x_current_mat = standardize_by_train(train_x.loc[valid_train], current_x)
        coef, _, _ = ewols_fit(train_y.loc[valid_train].to_numpy(dtype=float), x_train_mat, EWOLS_HALFLIFE_MONTHS)
        pred_total = float((x_current_mat @ coef)[0])
        pred_total = max(pred_total, float(out.loc[idx, "anchor_mtd_qty"]))

        out.loc[idx, "anchor_pred_month_total"] = pred_total
        out.loc[idx, "anchor_pred_error"] = pred_total - out.loc[idx, "actual_month_total"]
        out.loc[idx, "anchor_pred_error_pct"] = (
            (pred_total - out.loc[idx, "actual_month_total"]) / out.loc[idx, "actual_month_total"] * 100
        )
        out.loc[idx, "ewols_train_months"] = int(valid_train.sum())

    return out


def mean_segment_sse(y: np.ndarray, start: int, end: int) -> float:
    segment = y[start:end]
    if len(segment) == 0:
        return np.inf
    return float(np.sum((segment - np.mean(segment)) ** 2))


def dynamic_breaks_for_mean(y: np.ndarray, max_breaks: int, min_size: int) -> tuple[list[int], float]:
    n = len(y)
    if n < min_size * 2:
        return [], float(np.sum((y - np.mean(y)) ** 2))

    max_segments = min(max_breaks + 1, n // min_size)
    sse = np.full((max_segments + 1, n + 1), np.inf)
    prev = np.full((max_segments + 1, n + 1), -1, dtype=int)

    for end in range(min_size, n + 1):
        sse[1, end] = mean_segment_sse(y, 0, end)

    for seg in range(2, max_segments + 1):
        for end in range(seg * min_size, n + 1):
            candidates = [
                (sse[seg - 1, split] + mean_segment_sse(y, split, end), split)
                for split in range((seg - 1) * min_size, end - min_size + 1)
            ]
            if candidates:
                best_cost, best_split = min(candidates, key=lambda item: item[0])
                sse[seg, end] = best_cost
                prev[seg, end] = best_split

    penalty = CHANGEPOINT_PENALTY_SCALE * np.nanvar(y) * math.log(max(n, 2))
    best_seg = 1
    best_score = sse[1, n]
    for seg in range(2, max_segments + 1):
        score = sse[seg, n] + (seg - 1) * penalty
        if score < best_score:
            best_score = score
            best_seg = seg

    breaks = []
    end = n
    for seg in range(best_seg, 1, -1):
        split = int(prev[seg, end])
        if split <= 0:
            break
        breaks.append(split)
        end = split
    return sorted(breaks), float(sse[best_seg, n])


def summarize_mean_changepoints(df: pd.DataFrame, breaks: list[int]) -> pd.DataFrame:
    boundaries = [0] + breaks + [len(df)]
    rows = []
    for i in range(1, len(boundaries) - 1):
        split = boundaries[i]
        prev_segment = df.iloc[boundaries[i - 1] : split]
        next_segment = df.iloc[split : boundaries[i + 1]]
        prev_mean = prev_segment["selected_slope_per_avg_daily_qty"].mean()
        next_mean = next_segment["selected_slope_per_avg_daily_qty"].mean()
        rows.append(
            {
                "突变月份": df["month_label"].iloc[split],
                "突变前区间": f"{prev_segment['month_label'].iloc[0]} 至 {prev_segment['month_label'].iloc[-1]}",
                "突变后区间": f"{next_segment['month_label'].iloc[0]} 至 {next_segment['month_label'].iloc[-1]}",
                "突变前均值": prev_mean,
                "突变后均值": next_mean,
                "变化幅度": next_mean - prev_mean,
                "变化率": safe_divide(next_mean - prev_mean, abs(prev_mean)) * 100,
                "业务解读": "月内累计节奏变陡" if next_mean > prev_mean else "月内累计节奏变缓",
            }
        )
    return pd.DataFrame(rows)


def build_design_matrix(segment: pd.DataFrame) -> np.ndarray:
    t = np.arange(len(segment), dtype=float)
    t_scaled = (t - t.mean()) / (t.std(ddof=0) if t.std(ddof=0) > 0 else 1.0)
    slope = segment["selected_slope_per_avg_daily_qty"].astype(float).to_numpy()
    slope_scaled = (slope - np.nanmean(slope)) / (np.nanstd(slope) if np.nanstd(slope) > 0 else 1.0)
    return np.column_stack([np.ones(len(segment)), slope_scaled, t_scaled])


def fit_structural_segment(df: pd.DataFrame, start: int, end: int) -> SegmentFit:
    segment = df.iloc[start:end].copy()
    y = segment["anchor_pred_error_pct"].astype(float).to_numpy()
    x = build_design_matrix(segment)
    coef, rss, _ = ewols_fit(y, x, EWOLS_HALFLIFE_MONTHS)
    return SegmentFit(start=start, end=end, rss=rss, coef=coef, nobs=len(segment))


def dynamic_breaks_for_ewols(df: pd.DataFrame, max_breaks: int, min_size: int) -> tuple[list[int], list[SegmentFit]]:
    n = len(df)
    if n < min_size * 2:
        return [], [fit_structural_segment(df, 0, n)] if n >= 4 else []

    segment_cache: dict[tuple[int, int], SegmentFit] = {}

    def segment_fit(start: int, end: int) -> SegmentFit:
        key = (start, end)
        if key not in segment_cache:
            segment_cache[key] = fit_structural_segment(df, start, end)
        return segment_cache[key]

    max_segments = min(max_breaks + 1, n // min_size)
    cost = np.full((max_segments + 1, n + 1), np.inf)
    prev = np.full((max_segments + 1, n + 1), -1, dtype=int)

    for end in range(min_size, n + 1):
        cost[1, end] = segment_fit(0, end).rss

    for seg in range(2, max_segments + 1):
        for end in range(seg * min_size, n + 1):
            candidates = [
                (cost[seg - 1, split] + segment_fit(split, end).rss, split)
                for split in range((seg - 1) * min_size, end - min_size + 1)
            ]
            if candidates:
                best_cost, best_split = min(candidates, key=lambda item: item[0])
                cost[seg, end] = best_cost
                prev[seg, end] = best_split

    n_params_per_segment = 3
    best_seg = 1
    best_bic = n * math.log(max(cost[1, n] / max(n, 1), 1e-9)) + n_params_per_segment * math.log(n)
    for seg in range(2, max_segments + 1):
        bic = n * math.log(max(cost[seg, n] / max(n, 1), 1e-9)) + (seg * n_params_per_segment + seg - 1) * math.log(n)
        if bic < best_bic:
            best_bic = bic
            best_seg = seg

    breaks = []
    end = n
    for seg in range(best_seg, 1, -1):
        split = int(prev[seg, end])
        if split <= 0:
            break
        breaks.append(split)
        end = split
    breaks = sorted(breaks)
    boundaries = [0] + breaks + [n]
    fits = [segment_fit(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]
    return breaks, fits


def summarize_structural_breaks(df: pd.DataFrame, breaks: list[int], fits: list[SegmentFit]) -> pd.DataFrame:
    rows = []
    for fit in fits:
        segment = df.iloc[fit.start : fit.end]
        slope_coef = fit.coef[1] if len(fit.coef) > 1 else np.nan
        direction = "斜率越高，预测越偏高" if slope_coef > 0 else "斜率越高，预测越偏低"
        rows.append(
            {
                "时间段": f"{segment['month_label'].iloc[0]} 至 {segment['month_label'].iloc[-1]}",
                "样本月数": fit.nobs,
                "EWOLS 斜率系数": slope_coef,
                "平均预测误差%": segment["anchor_pred_error_pct"].mean(),
                "平均实际销量": segment["actual_month_total"].mean(),
                "关系解读": direction,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["结构变化月份"] = ""
        for idx, split in enumerate(breaks):
            if idx + 1 < len(out):
                out.loc[idx + 1, "结构变化月份"] = df["month_label"].iloc[split]
    return out


def target_series_frame(panel: pd.DataFrame, value_type: str) -> pd.DataFrame:
    target_col = target_col_for_value_type(value_type)
    out = panel[["month_start", "month_label", target_col]].copy()
    out = out.rename(columns={target_col: "value"})
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["value"])
    return out.sort_values("month_start").reset_index(drop=True)


def stationarity_test(series_df: pd.DataFrame, method: str) -> dict[str, object]:
    clean = series_df["value"].replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    clean = clean.reset_index(drop=True)
    n = int(len(clean))
    if n < 8:
        return {
            "method": method,
            "statistic": np.nan,
            "critical_values": {},
            "p_value": np.nan,
            "result": "样本不足",
            "detail": "少于 8 个有效月份，暂不运行平稳性检验。",
            "is_stationary": None,
            "nlags": None,
        }
    if clean.nunique() < 2:
        return {
            "method": method,
            "statistic": np.nan,
            "critical_values": {},
            "p_value": np.nan,
            "result": "平稳",
            "detail": "序列几乎为常数，统计检验不适用；业务上可视作无明显趋势波动。",
            "is_stationary": True,
            "nlags": None,
        }
    if method == "ADF":
        if adfuller is None:
            return {
                "method": method,
                "statistic": np.nan,
                "critical_values": {},
                "p_value": np.nan,
                "result": "不可用",
                "detail": "statsmodels 不可用，无法运行 ADF 单位根检验。",
                "is_stationary": None,
                "nlags": None,
            }
        try:
            statistic, p_value, _, _, critical_values, _ = adfuller(clean, autolag="AIC")
        except Exception as exc:
            return {
                "method": method,
                "statistic": np.nan,
                "critical_values": {},
                "p_value": np.nan,
                "result": "检验失败",
                "detail": f"ADF 检验失败：{exc}",
                "is_stationary": None,
                "nlags": None,
            }
        is_stationary = float(p_value) < DEFAULT_STATIONARITY_ALPHA
        return {
            "method": method,
            "statistic": float(statistic),
            "critical_values": {str(k): float(v) for k, v in critical_values.items()},
            "p_value": float(p_value),
            "result": "平稳" if is_stationary else "非平稳",
            "detail": "ADF 原假设为存在单位根；p<0.05 时更支持序列平稳。",
            "is_stationary": is_stationary,
            "nlags": None,
        }

    if kpss is None:
        return {
            "method": method,
            "statistic": np.nan,
            "critical_values": {},
            "p_value": np.nan,
            "result": "不可用",
            "detail": "statsmodels 不可用，无法运行 KPSS 平稳性检验。",
            "is_stationary": None,
            "nlags": None,
        }
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            statistic, p_value, nlags, critical_values = kpss(clean, regression="c", nlags="auto")
    except Exception as exc:
        return {
            "method": method,
            "statistic": np.nan,
            "critical_values": {},
            "p_value": np.nan,
            "result": "检验失败",
            "detail": f"KPSS 检验失败：{exc}",
            "is_stationary": None,
            "nlags": None,
        }
    is_stationary = float(p_value) >= DEFAULT_STATIONARITY_ALPHA
    return {
        "method": method,
        "statistic": float(statistic),
        "critical_values": {str(k): float(v) for k, v in critical_values.items()},
        "p_value": float(p_value),
        "result": "平稳" if is_stationary else "非平稳",
        "detail": "KPSS 原假设为趋势周围平稳；p<0.05 时更支持非平稳。",
        "is_stationary": is_stationary,
        "nlags": int(nlags),
    }


def stationarity_result_label(result: dict[str, object], alpha: float) -> str:
    p_value = result.get("p_value", np.nan)
    if pd.isna(p_value):
        return str(result["result"])
    if str(result["method"]) == "ADF":
        return "平稳" if float(p_value) < alpha else "非平稳"
    return "平稳" if float(p_value) >= alpha else "非平稳"


def stationarity_business_meaning(
    result: dict[str, object], value_type: str, target_label: str, alpha: float
) -> str:
    method = str(result["method"])
    p_value = result["p_value"]
    result_label = stationarity_result_label(result, alpha)
    nlag_hint = ""
    if method == "KPSS" and result.get("nlags") is not None:
        nlag_hint = f"，auto nlag 返回滞后期数 :blue[**{int(result['nlags'])}**]"
    if result_label == "平稳":
        return f"{method} p-value={fmt_pvalue(p_value)}，alpha={alpha:.2f}{nlag_hint}，当前{target_label}更像围绕稳定水平波动，业务上可优先关注短期扰动和季节结构。"
    if result_label == "非平稳":
        if value_type == "统计值":
            return f"{method} p-value={fmt_pvalue(p_value)}，alpha={alpha:.2f}{nlag_hint}，当前{target_label}存在趋势/结构变化迹象，建模前建议考虑差分、去趋势或分段口径。"
        return f"{method} p-value={fmt_pvalue(p_value)}，alpha={alpha:.2f}{nlag_hint}，当前{target_label}仍有持续性变化迹象，说明增长率口径也可能受结构性变化影响。"
    return f"{method} 当前结论为{result_label}，业务解释需谨慎；建议先补足有效月份或检查异常值后再判断。"


def stationarity_threshold_hint(result: dict[str, object], alpha: float) -> str:
    method = str(result["method"])
    if method == "ADF":
        return f"ADF 越低越支持平稳；在 alpha={alpha:.2f} 下，test statistic 低于 critical value 时拒绝单位根原假设。"
    return f"KPSS 越高越支持非平稳；在 alpha={alpha:.2f} 下，test statistic 高于 critical value 时拒绝平稳原假设。"


def parse_critical_alpha(label: str) -> float | None:
    try:
        return float(label.replace("%", "")) / 100
    except ValueError:
        return None


def critical_value_for_alpha(result: dict[str, object], alpha: float) -> tuple[float, bool]:
    critical_values = result.get("critical_values", {})
    if not isinstance(critical_values, dict):
        return np.nan, False

    points = []
    for label, value in critical_values.items():
        parsed_alpha = parse_critical_alpha(str(label))
        if parsed_alpha is not None and pd.notna(value):
            points.append((parsed_alpha, float(value)))
    if not points:
        return np.nan, False

    points = sorted(points)
    alphas = np.array([item[0] for item in points], dtype=float)
    values = np.array([item[1] for item in points], dtype=float)
    if len(points) == 1:
        return float(values[0]), not np.isclose(alpha, alphas[0])

    if alpha <= alphas[0]:
        x0, x1 = alphas[0], alphas[1]
        y0, y1 = values[0], values[1]
    elif alpha >= alphas[-1]:
        x0, x1 = alphas[-2], alphas[-1]
        y0, y1 = values[-2], values[-1]
    else:
        return float(np.interp(alpha, alphas, values)), not np.any(np.isclose(alpha, alphas))

    slope = (y1 - y0) / (x1 - x0)
    return float(y0 + slope * (alpha - x0)), not np.any(np.isclose(alpha, alphas))


def stationarity_critical_frame(result: dict[str, object], alpha: float) -> pd.DataFrame:
    critical_value, is_approximate = critical_value_for_alpha(result, alpha)
    rows = [
        {
            "label": "Test statistic",
            "value": result.get("statistic", np.nan),
            "kind": "检验统计量",
            "note": "",
            "order": 0,
        },
        {
            "label": "Critical value",
            "value": critical_value,
            "kind": "临界值",
            "note": f"alpha={alpha:.2f}" + (" 近似" if is_approximate else ""),
            "order": 1,
        },
    ]
    out = pd.DataFrame(rows)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return out.dropna(subset=["value"]).sort_values("order").reset_index(drop=True)


def stationarity_pvalue_frame(result: dict[str, object], alpha: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"label": "P-value", "value": result.get("p_value", np.nan), "kind": "P-value", "order": 0},
            {"label": f"alpha={alpha:.2f}", "value": alpha, "kind": "alpha", "order": 1},
        ]
    ).dropna(subset=["value"])


def lag_difference_frame(series_df: pd.DataFrame, lag: int, order: int) -> pd.DataFrame:
    out = series_df.copy()
    out["lag_month_label"] = out["month_label"].shift(lag)
    out["lag_value"] = out["value"].shift(lag)
    diff_series = out["value"].copy()
    for _ in range(order):
        diff_series = diff_series - diff_series.shift(lag)
    out["diff_value"] = diff_series
    out["diff_order"] = order
    return out.dropna(subset=["diff_value"]).reset_index(drop=True)


def add_smoothing(series_df: pd.DataFrame, window: int, method: str) -> pd.DataFrame:
    out = series_df.copy()
    if method == "EMA":
        out["smooth_value"] = out["value"].ewm(span=window, adjust=False, min_periods=1).mean()
    elif method == "Median":
        out["smooth_value"] = out["value"].rolling(window=window, min_periods=1).median()
    else:
        out["smooth_value"] = out["value"].rolling(window=window, min_periods=1).mean()
    return out


def stl_components(series_df: pd.DataFrame, period: int, robust: bool) -> pd.DataFrame:
    out = series_df.copy()
    out["trend"] = np.nan
    out["seasonal"] = np.nan
    out["residual"] = np.nan
    clean = out.dropna(subset=["value"]).copy()
    if len(clean) < max(period * 2, 8):
        return out

    try:
        from statsmodels.tsa.seasonal import STL

        result = STL(clean["value"], period=period, robust=robust).fit()
        out.loc[clean.index, "trend"] = result.trend
        out.loc[clean.index, "seasonal"] = result.seasonal
        out.loc[clean.index, "residual"] = result.resid
    except Exception:
        pass
    return out


def correlation_table(df: pd.DataFrame, value_type: str) -> pd.DataFrame:
    target_col = target_col_for_value_type(value_type)
    rows = []
    for item in candidate_metric_items(value_type):
        col = item["field"]
        if col == target_col or col not in df.columns or target_col not in df.columns:
            continue
        corr, p_value, n = corr_pvalue(df[col], df[target_col])
        rows.append(
            {
                "期间": item["period"],
                "指标": item["metric"],
                "字段": col,
                "样本数": n,
                "Pearson r": corr,
                "p-value": p_value,
                "|r|": abs(corr) if pd.notna(corr) else np.nan,
                "方向": relationship_direction(corr),
                "效应强度": effect_size_label(abs(corr) if pd.notna(corr) else np.nan),
            }
        )
    return pd.DataFrame(rows).sort_values(["|r|", "样本数"], ascending=[False, False]).reset_index(drop=True)


def describe_influences(corr_df: pd.DataFrame, complete_months: int, grain_label: str, value_type: str) -> list[str]:
    valid = corr_df.dropna(subset=["Pearson r"]).copy()
    if valid.empty:
        return [
            f"当前 {grain_label} 粒度 + {value_type} 口径下有效样本不足，暂不能稳定判断 {ANALYSIS_TARGET} 与日历/相邻月份因素的统计关系。",
            "建议扩大月份范围，或切换到更粗的统计粒度后再观察相关性。",
        ]

    strong = valid[valid["|r|"].ge(0.50)].head(5)
    medium = valid[(valid["|r|"].ge(0.30)) & (valid["|r|"].lt(0.50))].head(5)
    top = valid.iloc[0]

    lines = [
        f"当前口径覆盖 {complete_months} 个完整月份；在 {grain_label} 粒度 + {value_type} 口径下，{ANALYSIS_TARGET} 与若干日历结构、历史同月和上月销售强度指标存在统计相关。",
        f"相关性最高的是「{top['期间']} - {top['指标']}」，Pearson r={fmt_corr(top['Pearson r'])}，方向为{top['方向']}，效应强度为{top['效应强度']}。",
    ]

    if not strong.empty:
        items = "、".join(
            f"{row['期间']}{row['指标']}({fmt_corr(row['Pearson r'])})"
            for _, row in strong.iterrows()
        )
        lines.append(f"强相关因素包括：{items}。")
    elif not medium.empty:
        items = "、".join(
            f"{row['期间']}{row['指标']}({fmt_corr(row['Pearson r'])})"
            for _, row in medium.iterrows()
        )
        lines.append(f"中等相关因素包括：{items}。")
    else:
        lines.append("当前没有达到中等以上相关强度的因素，说明该粒度下关系较弱或样本不足。")

    calendar_hits = valid[
        valid["指标"].isin(["工作日天数", "非工作日天数", "假期天数（非周末）", "周末休息天数（仅周末）"])
    ].head(4)
    if not calendar_hits.empty:
        items = "、".join(
            f"{row['期间']}{row['指标']}({fmt_corr(row['Pearson r'])})"
            for _, row in calendar_hits.iterrows()
        )
        lines.append(f"纯日历因素中较明显的是：{items}。")

    sales_strength_hits = valid[
        valid["指标"].str.contains("平均每工作日销量", regex=False)
        | valid["指标"].str.contains("平均工作日销量", regex=False)
    ].head(3)
    if not sales_strength_hits.empty:
        items = "、".join(
            f"{row['期间']}{row['指标']}({fmt_corr(row['Pearson r'])})"
            for _, row in sales_strength_hits.iterrows()
        )
        lines.append(f"历史销售强度关系：{items}。")

    lines.append("注意：这里展示的是统计相关，不代表因果影响；p-value 仅作为探索性参考，结论应结合节假日错位和样本量判断。")
    return lines


def render_kpis(panel: pd.DataFrame, corr_df: pd.DataFrame, value_type: str) -> None:
    target_col = target_col_for_value_type(value_type)
    target_label = target_label_for_value_type(value_type)
    complete_months = int(panel["curr_is_complete_month"].fillna(False).sum())
    avg_target = panel[target_col].mean()
    top = corr_df.dropna(subset=["Pearson r"]).head(1)
    top_label = "-" if top.empty else f"{top['期间'].iloc[0]}{top['指标'].iloc[0]}"
    top_corr = np.nan if top.empty else top["Pearson r"].iloc[0]

    with st.container(horizontal=True):
        st.metric("完整月份", f"{complete_months}个", border=True)
        st.metric(target_label, fmt_metric_value(avg_target, value_type), border=True)
        st.metric("最强相关因素", top_label, fmt_corr(top_corr), border=True)
        st.metric("日历口径", "中国法定节假日" if cn_is_workday is not None else "周一至周五", border=True)


def render_correlation_bar(corr_df: pd.DataFrame) -> None:
    plot_df = corr_df.dropna(subset=["Pearson r"]).head(16).copy()
    if plot_df.empty:
        st.info("有效样本不足，无法绘制相关性图。")
        return
    plot_df["因素"] = plot_df["期间"] + " - " + plot_df["指标"]
    chart = (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            x=alt.X("Pearson r:Q", title="Pearson r", scale=alt.Scale(domain=[-1, 1])),
            y=alt.Y("因素:N", sort="-x", title=None),
            color=alt.condition(
                alt.datum["Pearson r"] >= 0,
                alt.value("#2f6f73"),
                alt.value("#c44e52"),
            ),
            tooltip=[
                alt.Tooltip("期间:N"),
                alt.Tooltip("指标:N"),
                alt.Tooltip("Pearson r:Q", format="+.3f"),
                alt.Tooltip("p-value:Q", format=".3f"),
                alt.Tooltip("样本数:Q", format=".0f"),
                alt.Tooltip("效应强度:N"),
            ],
        )
        .properties(height=CHART_HEIGHT)
    )
    st.altair_chart(chart)


def render_target_trend(panel: pd.DataFrame, value_type: str) -> None:
    target_col = target_col_for_value_type(value_type)
    target_label = target_label_for_value_type(value_type)
    plot_df = panel[["month_start", "month_label", target_col, "curr_workdays", "curr_weekday_holidays"]].dropna(
        subset=[target_col]
    )
    if plot_df.empty:
        st.info("没有可绘制的工作日平均销量趋势。")
        return
    chart = (
        alt.Chart(plot_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("month_start:T", title="月份"),
            y=alt.Y(f"{target_col}:Q", title=target_label),
            tooltip=[
                alt.Tooltip("month_label:N", title="月份"),
                alt.Tooltip(
                    f"{target_col}:Q",
                    title=target_label,
                    format=".1%" if value_type in {"环比值", "同比值"} else ",.0f",
                ),
                alt.Tooltip("curr_workdays:Q", title="当月工作日", format=".0f"),
                alt.Tooltip("curr_weekday_holidays:Q", title="当月假期天数", format=".0f"),
            ],
        )
        .properties(height=CHART_HEIGHT)
    )
    st.altair_chart(chart)


def render_stationarity_test(panel: pd.DataFrame, value_type: str) -> None:
    target_label = target_label_for_value_type(value_type)
    with st.container(border=True):
        title_col, controls_col = st.columns(TITLE_CONTROL_LAYOUT, vertical_alignment="center")
        with title_col:
            st.subheader("平稳性检验")
        with controls_col:
            with st.container(horizontal=True, horizontal_alignment="right", vertical_alignment="center", gap=None):
                render_inline_label("alpha")
                alpha = st.number_input(
                    "alpha",
                    min_value=0.01,
                    max_value=0.20,
                    value=DEFAULT_STATIONARITY_ALPHA,
                    step=0.01,
                    format="%.2f",
                    key="stationarity_alpha",
                    label_visibility="collapsed",
                    width=COMPACT_NUMBER_INPUT_WIDTH,
                )
                method = st.segmented_control(
                    "方法",
                    STATIONARITY_METHOD_OPTIONS,
                    default="KPSS",
                    key="stationarity_method",
                    label_visibility="collapsed",
                    width="content",
                )
        method = method or "KPSS"
        series_df = target_series_frame(panel, value_type)
        result = stationarity_test(series_df, method)
        st.caption(stationarity_business_meaning(result, value_type, target_label, float(alpha)))
        st.caption(stationarity_threshold_hint(result, float(alpha)))

        plot_df = stationarity_critical_frame(result, float(alpha))
        if plot_df.empty:
            st.info("有效样本不足或检验不可用，无法绘制 test statistic + critical value 图。")
            return

        domain_min = min(0.0, float(plot_df["value"].min()))
        domain_max = max(0.0, float(plot_df["value"].max()))
        padding = max((domain_max - domain_min) * 0.12, 0.1)
        zero_rule = alt.Chart(pd.DataFrame({"x": [0.0]})).mark_rule(color="#d0d4da").encode(x="x:Q")
        lollipop = (
            alt.Chart(plot_df)
            .mark_rule(strokeWidth=2)
            .encode(
                x=alt.X(
                    "value:Q",
                    title="统计量数值",
                    scale=alt.Scale(domain=[domain_min - padding, domain_max + padding]),
                ),
                x2=alt.value(0),
                y=alt.Y("label:N", sort=alt.SortField("order"), title=None),
                color=alt.Color(
                    "kind:N",
                    legend=None,
                    scale=alt.Scale(domain=["检验统计量", "临界值"], range=["#c44e52", "#4c72b0"]),
                ),
            )
        )
        points = (
            alt.Chart(plot_df)
            .mark_circle(size=110)
            .encode(
                x=alt.X("value:Q", title="统计量数值"),
                y=alt.Y("label:N", sort=alt.SortField("order"), title=None),
                color=alt.Color(
                    "kind:N",
                    legend=None,
                    scale=alt.Scale(domain=["检验统计量", "临界值"], range=["#c44e52", "#4c72b0"]),
                ),
                tooltip=[
                    alt.Tooltip("label:N", title="项目"),
                    alt.Tooltip("value:Q", title="数值", format=".3f"),
                    alt.Tooltip("kind:N", title="类型"),
                ],
            )
        )
        labels = (
            alt.Chart(plot_df)
            .mark_text(align="left", dx=8, fontSize=12)
            .encode(
                x=alt.X("value:Q", title="统计量数值"),
                y=alt.Y("label:N", sort=alt.SortField("order"), title=None),
                text=alt.Text("value:Q", format=".3f"),
            )
        )
        critical_chart = (zero_rule + lollipop + points + labels).properties(height=240)

        pvalue_df = stationarity_pvalue_frame(result, float(alpha))
        p_domain_max = max(0.20, float(pvalue_df["value"].max()) * 1.12)
        pvalue_label_df = pvalue_df[pvalue_df["kind"].eq("P-value")].copy()
        pvalue_label_df["display_label"] = pvalue_label_df["value"].map(lambda value: f"P-value={fmt_pvalue(value)}")
        p_zero = (
            alt.Chart(pd.DataFrame({"x": [0.0]}))
            .mark_rule(color="#d0d4da")
            .encode(x=alt.X("x:Q", scale=alt.Scale(domain=[0, p_domain_max])))
        )
        pvalue_rule = (
            alt.Chart(pvalue_df[pvalue_df["kind"].eq("P-value")])
            .mark_rule(strokeWidth=3, color="#2f6f73")
            .encode(
                x=alt.X("value:Q", title="P-value / alpha", scale=alt.Scale(domain=[0, p_domain_max])),
                tooltip=[
                    alt.Tooltip("label:N", title="项目"),
                    alt.Tooltip("value:Q", title="数值", format=".3f"),
                ],
            )
        )
        p_labels = (
            alt.Chart(pvalue_label_df)
            .mark_text(align="left", dx=8, fontSize=12)
            .encode(
                x=alt.X("value:Q", title="P-value / alpha"),
                y=alt.value(38),
                text=alt.Text("display_label:N"),
            )
        )
        alpha_rule = (
            alt.Chart(pd.DataFrame({"value": [float(alpha)], "label": [f"alpha={float(alpha):.2f}"]}))
            .mark_rule(strokeWidth=2, strokeDash=[6, 4], color="#d95f02")
            .encode(
                x=alt.X("value:Q", title="P-value / alpha", scale=alt.Scale(domain=[0, p_domain_max])),
                tooltip=[
                    alt.Tooltip("label:N", title="项目"),
                    alt.Tooltip("value:Q", title="数值", format=".3f"),
                ],
            )
        )
        alpha_label = (
            alt.Chart(pd.DataFrame({"value": [float(alpha)], "label": [f"alpha={float(alpha):.2f}"]}))
            .mark_text(align="left", dy=-20, dx=6, color="#d95f02", fontSize=12)
            .encode(x=alt.X("value:Q", title="P-value / alpha"), text="label:N")
        )
        pvalue_chart = (p_zero + pvalue_rule + p_labels + alpha_rule + alpha_label).properties(height=240)
        left_col, right_col = st.columns(2, gap="medium", border=True)
        with left_col:
            st.altair_chart(critical_chart)
        with right_col:
            st.altair_chart(pvalue_chart)


def render_lag_difference_chart(panel: pd.DataFrame, value_type: str) -> None:
    target_label = target_label_for_value_type(value_type)
    with st.container(border=True):
        title_col, controls_col = st.columns(TITLE_CONTROL_LAYOUT, vertical_alignment="center")
        with title_col:
            st.subheader("滞后差分")
        with controls_col:
            with st.container(horizontal=True, horizontal_alignment="right", vertical_alignment="center", gap=None):
                render_inline_label("差分阶数")
                diff_order = st.number_input(
                    "差分阶数",
                    min_value=1,
                    max_value=4,
                    value=1,
                    step=1,
                    key="lag_difference_order",
                    label_visibility="collapsed",
                    width=COMPACT_NUMBER_INPUT_WIDTH,
                )
                render_inline_label("滞后期")
                lag = st.number_input(
                    "滞后期",
                    min_value=1,
                    max_value=36,
                    value=12,
                    step=1,
                    key="lag_difference_period",
                    label_visibility="collapsed",
                    width=COMPACT_NUMBER_INPUT_WIDTH,
                )

        series_df = target_series_frame(panel, value_type)
        diff_df = lag_difference_frame(series_df, int(lag), int(diff_order))
        if diff_df.empty:
            st.info("有效样本不足，无法绘制滞后差分图。")
            return

        zero_rule = alt.Chart(pd.DataFrame({"y": [0.0]})).mark_rule(color="#d0d4da").encode(y="y:Q")
        chart = (
            alt.Chart(diff_df)
            .mark_line(point=True, color="#4c72b0")
            .encode(
                x=alt.X("month_start:T", title="月份"),
                y=alt.Y("diff_value:Q", title=f"{target_label} 差分"),
                tooltip=[
                    alt.Tooltip("month_label:N", title="月份"),
                    alt.Tooltip("diff_order:Q", title="差分阶数", format=".0f"),
                    alt.Tooltip("lag_month_label:N", title=f"滞后 {int(lag)} 期月份"),
                    alt.Tooltip("value:Q", title="当前值", format=chart_number_format(value_type)),
                    alt.Tooltip("lag_value:Q", title="滞后值", format=chart_number_format(value_type)),
                    alt.Tooltip("diff_value:Q", title="差分", format=chart_number_format(value_type)),
                ],
            )
            .properties(height=CHART_HEIGHT)
        )
        st.caption(
            f"计算方式：对指标连续做 {int(diff_order)} 阶滞后 {int(lag)} 期差分；"
            "每一阶都使用相同滞后期。"
        )
        st.altair_chart(zero_rule + chart)


def smoothing_config(default_window: int, key_prefix: str) -> tuple[int, str]:
    with st.container(horizontal=True, horizontal_alignment="right", vertical_alignment="center", gap=None):
        render_inline_label("时间窗口")
        window = st.number_input(
            "时间窗口",
            min_value=1,
            max_value=12,
            value=default_window,
            step=1,
            key=f"{key_prefix}_window",
            label_visibility="collapsed",
            width=COMPACT_NUMBER_INPUT_WIDTH,
        )
        render_inline_label("方法")
        method = st.segmented_control(
            "方法",
            SMOOTHING_METHOD_OPTIONS,
            default="SMA",
            key=f"{key_prefix}_method",
            label_visibility="collapsed",
            width="content",
        )
    return int(window), method or "SMA"


def render_smoothing_row(panel: pd.DataFrame, value_type: str) -> None:
    target_label = target_label_for_value_type(value_type)
    with st.container(border=True):
        title_col, controls_col = st.columns(TITLE_CONTROL_LAYOUT, vertical_alignment="center")
        with title_col:
            st.subheader("Rolling 平滑")
        with controls_col:
            left, right = st.columns(2, gap="small")
            with left:
                window_3m, method_3m = smoothing_config(3, "smooth_3m")
            with right:
                window_6m, method_6m = smoothing_config(6, "smooth_6m")

        series_df = target_series_frame(panel, value_type)
        if series_df.empty:
            st.info("没有足够数据绘制平滑趋势。")
            return

        smooth_3m = add_smoothing(series_df, window_3m, method_3m).assign(
            series=f"{window_3m}M {method_3m}",
            plot_value=lambda df: df["smooth_value"],
        )
        smooth_6m = add_smoothing(series_df, window_6m, method_6m).assign(
            series=f"{window_6m}M {method_6m}",
            plot_value=lambda df: df["smooth_value"],
        )
        raw = series_df.assign(series="原始序列", plot_value=series_df["value"])
        plot_df = pd.concat(
            [
                raw[["month_start", "month_label", "series", "plot_value"]],
                smooth_3m[["month_start", "month_label", "series", "plot_value"]],
                smooth_6m[["month_start", "month_label", "series", "plot_value"]],
            ],
            ignore_index=True,
        )
        legend_selection = alt.selection_point(fields=["series"], bind="legend")
        chart = (
            alt.Chart(plot_df)
            .mark_line(point=False)
            .encode(
                x=alt.X("month_start:T", title="月份"),
                y=alt.Y("plot_value:Q", title=target_label),
                color=alt.Color(
                    "series:N",
                    title=None,
                    scale=alt.Scale(range=["#8a9099", "#2f6f73", "#d95f02"]),
                ),
                strokeWidth=alt.condition(alt.datum.series == "原始序列", alt.value(1.4), alt.value(2.8)),
                opacity=alt.condition(legend_selection, alt.value(1.0), alt.value(0.12)),
                tooltip=[
                    alt.Tooltip("month_label:N", title="月份"),
                    alt.Tooltip("series:N", title="序列"),
                    alt.Tooltip("plot_value:Q", title=target_label, format=chart_number_format(value_type)),
                ],
            )
            .properties(height=CHART_HEIGHT)
            .add_params(legend_selection)
        )
        st.altair_chart(chart)


def render_stl_component_chart(
    stl_df: pd.DataFrame,
    component_col: str,
    title: str,
    value_type: str,
    with_left_divider: bool = False,
) -> None:
    label_map = {
        "trend": "Trend_t",
        "seasonal": "Seasonal_t",
        "residual": "Residual_t",
    }
    plot_df = stl_df[["month_start", "month_label", component_col]].dropna().copy()
    if with_left_divider:
        st.markdown(
            "<div style='border-left:1px solid rgba(49, 51, 63, 0.16); padding-left:1rem;'>",
            unsafe_allow_html=True,
        )
    st.subheader(title)
    if plot_df.empty:
        st.info("有效样本不足，无法稳定进行 STL 分解。")
        if with_left_divider:
            st.markdown("</div>", unsafe_allow_html=True)
        return
    chart = (
        alt.Chart(plot_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("month_start:T", title="月份"),
            y=alt.Y(f"{component_col}:Q", title=label_map.get(component_col, component_col)),
            tooltip=[
                alt.Tooltip("month_label:N", title="月份"),
                alt.Tooltip(
                    f"{component_col}:Q",
                    title=label_map.get(component_col, component_col),
                    format=chart_number_format(value_type),
                ),
            ],
        )
        .properties(height=CHART_HEIGHT)
    )
    st.altair_chart(chart)
    if with_left_divider:
        st.markdown("</div>", unsafe_allow_html=True)


def render_stl_section(panel: pd.DataFrame, value_type: str) -> None:
    with st.container(border=True):
        title_col, controls_col = st.columns(TITLE_CONTROL_LAYOUT, vertical_alignment="center")
        with title_col:
            st.subheader("Robust STL 分解参数")
        with controls_col:
            with st.container(horizontal=True, horizontal_alignment="right", vertical_alignment="center", gap=None):
                render_inline_label("周期 period")
                period = st.number_input(
                    "周期 period",
                    min_value=2,
                    max_value=24,
                    value=12,
                    step=1,
                    help="代表一个季节周期包含多少个观测点；月度数据默认 12。",
                    key="stl_period",
                    label_visibility="collapsed",
                    width=COMPACT_NUMBER_INPUT_WIDTH,
                )
                render_inline_label("是否鲁棒 robust")
                robust = st.toggle(
                    "是否鲁棒 robust",
                    value=True,
                    help="开启后，STL 会降低极端尖峰/深坑对 Trend 和 Seasonal 拟合的影响。",
                    key="stl_robust",
                    label_visibility="collapsed",
                    width="content",
                )

        series_df = target_series_frame(panel, value_type)
        stl_df = stl_components(series_df, period=int(period), robust=bool(robust))
        trend_col, seasonal_col, residual_col = st.columns(3, gap="medium")
        with trend_col:
            render_stl_component_chart(stl_df, "trend", "STL Trend_t", value_type)
        with seasonal_col:
            render_stl_component_chart(stl_df, "seasonal", "STL Seasonal_t", value_type, with_left_divider=True)
        with residual_col:
            render_stl_component_chart(stl_df, "residual", "STL Residual_t", value_type, with_left_divider=True)


def scatter_factor_options(corr_df: pd.DataFrame) -> dict[str, str]:
    valid_factors = corr_df.dropna(subset=["Pearson r"]).copy()
    if valid_factors.empty:
        return {}
    return {
        f"{row['期间']} - {row['指标']} ({fmt_corr(row['Pearson r'])})": row["字段"]
        for _, row in valid_factors.head(20).iterrows()
    }


def render_scatter(
    panel: pd.DataFrame,
    value_type: str,
    factor_options: dict[str, str],
    selected_label: str | None,
) -> None:
    target_col = target_col_for_value_type(value_type)
    target_label = target_label_for_value_type(value_type)
    if not factor_options or selected_label is None:
        st.info("没有可用于散点图的有效因素。")
        return
    selected_col = factor_options[selected_label]
    label_col = "散点标签" if "散点标签" in panel.columns else "month_label"
    plot_df = panel[[label_col, target_col, selected_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if plot_df.empty:
        st.info("当前选择没有足够样本绘制散点图。")
        return
    plot_df = plot_df.rename(columns={label_col: "散点标签"})
    chart = (
        alt.Chart(plot_df)
        .mark_circle(size=70, opacity=0.75)
        .encode(
            x=alt.X(f"{selected_col}:Q", title=selected_label),
            y=alt.Y(f"{target_col}:Q", title=target_label),
            tooltip=[
                alt.Tooltip("散点标签:N", title="分析粒度"),
                alt.Tooltip(f"{selected_col}:Q", title=selected_label, format=",.3f"),
                alt.Tooltip(
                    f"{target_col}:Q",
                    title=target_label,
                    format=".1%" if value_type in {"环比值", "同比值"} else ",.0f",
                ),
            ],
        )
        .properties(height=CHART_HEIGHT)
    )
    st.altair_chart(chart)


def grouped_bar_chart(df: pd.DataFrame, x_col: str, y_col: str, y_title: str, percent_axis: bool = True) -> alt.Chart:
    if df.empty or y_col not in df.columns or x_col not in df.columns or "窗口" not in df.columns:
        return alt.Chart(pd.DataFrame({"message": ["无可用样本"]})).mark_text(color=COLOR_MUTED).encode(text="message:N")

    plot_df = df.dropna(subset=[y_col]).copy()
    plot_df[x_col] = plot_df[x_col].astype(str)
    if plot_df.empty:
        return alt.Chart(pd.DataFrame({"message": ["无可用样本"]})).mark_text(color=COLOR_MUTED).encode(text="message:N")

    return (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            x=alt.X(f"{x_col}:N", title="分析粒度", axis=alt.Axis(labelAngle=-30)),
            xOffset=alt.XOffset("窗口:N"),
            y=alt.Y(f"{y_col}:Q", title=y_title, axis=alt.Axis(format=".0%" if percent_axis else ".3f")),
            color=alt.Color(
                "窗口:N",
                scale=alt.Scale(range=[COLOR_MAIN, COLOR_ACCENT, COLOR_BLUE, COLOR_DANGER]),
            ),
            tooltip=[
                alt.Tooltip("窗口:N", title="窗口"),
                alt.Tooltip(f"{x_col}:N", title="分析粒度"),
                alt.Tooltip(f"{y_col}:Q", title=y_title, format=".2%" if percent_axis else ".3f"),
            ],
        )
        .properties(height=260)
    )


def changepoint_altair_chart(df: pd.DataFrame, breaks: list[int]) -> alt.Chart:
    plot_df = df.copy()
    break_df = plot_df.iloc[breaks][["month_start", "month_label"]].copy() if breaks else pd.DataFrame()
    line = (
        alt.Chart(plot_df)
        .mark_line(point=True, color=COLOR_MAIN)
        .encode(
            x=alt.X("month_start:T", title="月份"),
            y=alt.Y("selected_slope_per_avg_daily_qty:Q", title="累计销量斜率 / 窗口日均销量"),
            tooltip=[
                alt.Tooltip("month_label:N", title="月份"),
                alt.Tooltip("selected_slope_per_avg_daily_qty:Q", title="标准化斜率", format=".3f"),
                alt.Tooltip("selected_slope:Q", title="累计销量斜率", format=",.1f"),
            ],
        )
    )
    if break_df.empty:
        return line.properties(height=CHART_HEIGHT)
    rules = (
        alt.Chart(break_df)
        .mark_rule(color=COLOR_DANGER, strokeDash=[6, 4], strokeWidth=2)
        .encode(
            x="month_start:T",
            tooltip=[alt.Tooltip("month_label:N", title="突变月份")],
        )
    )
    labels = (
        alt.Chart(break_df)
        .mark_text(align="left", dx=6, dy=-8, color=COLOR_DANGER, fontSize=12)
        .encode(x="month_start:T", y=alt.value(16), text="month_label:N")
    )
    return (line + rules + labels).properties(height=CHART_HEIGHT)


def structural_altair_chart(df: pd.DataFrame, breaks: list[int], fits: list[SegmentFit]) -> alt.Chart:
    plot_df = df.copy()
    boundaries = [0] + breaks + [len(plot_df)]
    plot_df["阶段"] = "阶段 1"
    for idx in range(len(boundaries) - 1):
        plot_df.loc[boundaries[idx] : boundaries[idx + 1] - 1, "阶段"] = f"阶段 {idx + 1}"

    scatter = (
        alt.Chart(plot_df)
        .mark_circle(size=70, opacity=0.82)
        .encode(
            x=alt.X("selected_slope_per_avg_daily_qty:Q", title="累计销量斜率 / 窗口日均销量"),
            y=alt.Y("anchor_pred_error_pct:Q", title="anchor预测误差%"),
            color=alt.Color("阶段:N", scale=alt.Scale(range=[COLOR_MAIN, COLOR_ACCENT, COLOR_BLUE, COLOR_DANGER])),
            tooltip=[
                alt.Tooltip("month_label:N", title="月份"),
                alt.Tooltip("阶段:N"),
                alt.Tooltip("selected_slope_per_avg_daily_qty:Q", title="标准化斜率", format=".3f"),
                alt.Tooltip("anchor_pred_error_pct:Q", title="anchor预测误差%", format=".1f"),
            ],
        )
    )
    lines = []
    for idx, fit in enumerate(fits):
        segment = plot_df.iloc[fit.start : fit.end].copy()
        if segment.empty or len(fit.coef) < 3:
            continue
        x_raw = segment["selected_slope_per_avg_daily_qty"].astype(float)
        x_grid = np.linspace(float(x_raw.min()), float(x_raw.max()), 20)
        slope_mean = float(x_raw.mean())
        slope_std = float(x_raw.std(ddof=0)) if float(x_raw.std(ddof=0)) > 0 else 1.0
        line_df = pd.DataFrame(
            {
                "selected_slope_per_avg_daily_qty": x_grid,
                "anchor_pred_error_pct": fit.coef[0] + fit.coef[1] * ((x_grid - slope_mean) / slope_std),
                "阶段": f"阶段 {idx + 1}",
            }
        )
        lines.append(line_df)
    line_layer = (
        alt.Chart(pd.concat(lines, ignore_index=True))
        .mark_line(strokeWidth=3)
        .encode(
            x="selected_slope_per_avg_daily_qty:Q",
            y="anchor_pred_error_pct:Q",
            color=alt.Color("阶段:N", scale=alt.Scale(range=[COLOR_MAIN, COLOR_ACCENT, COLOR_BLUE, COLOR_DANGER])),
        )
        if lines
        else alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_line()
    )
    zero_rule = alt.Chart(pd.DataFrame({"y": [0.0]})).mark_rule(color=COLOR_MUTED, strokeDash=[2, 3]).encode(y="y:Q")
    return (scatter + line_layer + zero_rule).properties(height=CHART_HEIGHT)


def render_contribution_stability_tab(daily: pd.DataFrame) -> None:
    title_col, controls_col = st.columns([0.38, 0.62], vertical_alignment="center")
    with title_col:
        st.subheader("贡献率稳定性")
    with controls_col:
        with st.container(horizontal=True, horizontal_alignment="right", vertical_alignment="center", gap=None):
            render_inline_label("工作日进度窗口")
            selected_segments = st.pills(
                "工作日进度窗口",
                list(CONTRIBUTION_SEGMENT_OPTIONS.keys()),
                default=list(CONTRIBUTION_SEGMENT_OPTIONS.keys()),
                selection_mode="multi",
                key="contribution_segments",
                label_visibility="collapsed",
                width="content",
            )
            render_inline_label("统计粒度")
            selected_grains = st.pills(
                "统计粒度",
                GRAIN_OPTIONS,
                default=["月"],
                selection_mode="multi",
                key="contribution_grains",
                label_visibility="collapsed",
                width="content",
            )
            render_inline_label("聚合")
            aggregate_segments = st.toggle(
                "聚合所选窗口",
                value=False,
                key="contribution_aggregate_segments",
                label_visibility="collapsed",
                width="content",
            )
    st.caption("将每个月按工作日进度切成四段，分析所选窗口销量对当月总量的贡献率稳定性。")

    if not selected_segments:
        st.warning("请至少选择一个工作日进度窗口。", icon=":material/warning:")
        return
    selected_grains = [g for g in GRAIN_OPTIONS if g in selected_grains]
    detail = build_contribution_detail(daily, tuple(selected_segments), bool(aggregate_segments))
    summaries = summarize_contribution_stability(detail, selected_grains)
    cv_df = add_binary_stability_columns(summaries["cv"], "CV", CV_STABLE_MAX, "CV={value}；≤{stable} 稳定，>{stable} 不稳定。")
    yoy_df = add_binary_stability_columns(
        summaries["yoy"], "平均YoY变化", YOY_STABLE_MAX, "平均YoY绝对变化={value}；≤{stable} 稳定，>{stable} 不稳定。"
    )
    max_yoy_df = add_binary_stability_columns(
        summaries["max_yoy"], "最大YoY变化", MAX_YOY_STABLE_MAX, "最大YoY绝对变化={value}；≤{stable} 稳定，>{stable} 不稳定。"
    )
    loyo_df = add_binary_stability_columns(
        summaries["loyo"], "LOYO-WAPE", LOYO_WAPE_STABLE_MAX, "LOYO-WAPE={value}；≤{stable} 稳定，>{stable} 不稳定。"
    )

    with st.container(border=True):
        st.subheader("CV 稳定性评估")
        cv_label, cv_reason = stability_summary(cv_df, "CV")
        render_stability_basis(cv_label, cv_reason, f"CV≤{fmt_pct(CV_STABLE_MAX * 100)} 为稳定，>{fmt_pct(CV_STABLE_MAX * 100)} 为不稳定。")
        st.caption("CV = 贡献率标准差 / 平均贡献率；越低表示该窗口对月总量的贡献越稳定。")
        st.altair_chart(grouped_bar_chart(cv_df, "分析粒度", "CV", "CV", percent_axis=False))
        show_dataframe_12_rows(
            cv_df[["窗口", "分析粒度", "稳定性", "平均贡献率", "贡献率标准差", "CV", "样本月数"]],
            hide_index=True,
            column_config={
                "平均贡献率": st.column_config.NumberColumn(format="percent"),
                "贡献率标准差": st.column_config.NumberColumn(format="percent"),
                "CV": st.column_config.NumberColumn(format="%.3f"),
            },
        )

    with st.container(border=True):
        st.subheader("YoY 及 max(YoY) 变化")
        yoy_display = yoy_df.rename(columns={"稳定性": "avgYoY 稳定性"}).copy()
        if not max_yoy_df.empty and {"窗口", "分析粒度", "最大YoY变化", "稳定性", "判定依据"}.issubset(max_yoy_df.columns):
            yoy_display = yoy_display.merge(
                max_yoy_df[["窗口", "分析粒度", "最大YoY变化", "稳定性", "判定依据"]].rename(
                    columns={"稳定性": "maxYoY稳定性", "判定依据": "最大YoY判定依据"}
                ),
                on=["窗口", "分析粒度"],
                how="left",
            )
        else:
            yoy_display["最大YoY变化"] = np.nan
            yoy_display["maxYoY稳定性"] = STABILITY_LABEL_UNSTABLE
        yoy_display["综合稳定性"] = yoy_display.apply(
            lambda row: worst_stability_label([row.get("avgYoY 稳定性"), row.get("maxYoY稳定性")]),
            axis=1,
        )
        yoy_label, yoy_reason = stability_summary_from_labels(yoy_display["综合稳定性"])
        render_stability_basis(
            yoy_label,
            yoy_reason,
            f"平均YoY绝对变化≤{fmt_pct_1(YOY_STABLE_MAX * 100)} 为稳定；最大YoY绝对变化≤{fmt_pct_1(MAX_YOY_STABLE_MAX * 100)} 为稳定。",
        )
        st.caption("YoY变化 = 当年贡献率 - 上年贡献率；最大 YoY 变化取当前统计粒度内绝对值最大的一次同比变化。")
        c1, c2 = st.columns(2, gap="medium", border=True)
        with c1:
            st.altair_chart(grouped_bar_chart(yoy_df, "分析粒度", "平均YoY变化", "平均 YoY 变化"))
        with c2:
            st.altair_chart(grouped_bar_chart(max_yoy_df, "分析粒度", "最大YoY变化", "最大 YoY 变化"))
        yoy_display_cols = [
            col
            for col in [
                "窗口",
                "分析粒度",
                "avgYoY 稳定性",
                "maxYoY稳定性",
                "综合稳定性",
                "平均YoY变化",
                "最大正向YoY变化",
                "最大负向YoY变化",
                "最大YoY变化",
                "有效同比样本",
            ]
            if col in yoy_display.columns
        ]
        show_dataframe_12_rows(
            yoy_display[yoy_display_cols],
            hide_index=True,
            column_config={
                "平均YoY变化": st.column_config.NumberColumn(format="percent"),
                "最大正向YoY变化": st.column_config.NumberColumn(format="percent"),
                "最大负向YoY变化": st.column_config.NumberColumn(format="percent"),
                "最大YoY变化": st.column_config.NumberColumn(format="percent"),
            },
        )

    with st.container(border=True):
        st.subheader("LOYO-WAPE")
        loyo_label, loyo_reason = stability_summary(loyo_df, "LOYO-WAPE")
        render_stability_basis(loyo_label, loyo_reason, f"LOYO-WAPE≤{fmt_pct(LOYO_WAPE_STABLE_MAX * 100)} 为稳定，>{fmt_pct(LOYO_WAPE_STABLE_MAX * 100)} 为不稳定。")
        st.caption("每次留出一个年份，用其他年份同月同窗口贡献率中位数反推测试年份月总量，再汇总 WAPE；越低表示贡献率基准越可复用。")
        if loyo_df.empty:
            st.write("历史年份不足，暂无法计算 LOYO-WAPE。")
        else:
            st.altair_chart(grouped_bar_chart(loyo_df, "分析粒度", "LOYO-WAPE", "LOYO-WAPE"))
            show_dataframe_12_rows(
                loyo_df[["窗口", "分析粒度", "稳定性", "判定依据", "LOYO-WAPE", "留一年样本", "绝对误差", "实际销量"]],
                hide_index=True,
                column_config={
                    "LOYO-WAPE": st.column_config.NumberColumn(format="percent"),
                    "绝对误差": st.column_config.NumberColumn(format="%.0f"),
                    "实际销量": st.column_config.NumberColumn(format="%.0f"),
                },
            )

    with st.expander("贡献率明细数据", expanded=False):
        show_dataframe_12_rows(
            detail,
            hide_index=True,
            column_config={
                "month_start": st.column_config.DateColumn("月份"),
                "actual_month_total": st.column_config.NumberColumn("实际月销量", format="%.0f"),
                "窗口销量": st.column_config.NumberColumn(format="%.0f"),
                "贡献率": st.column_config.NumberColumn(format="percent"),
            },
        )


def render_slope_break_tab(daily: pd.DataFrame) -> None:
    max_workdays = int(daily["max_workday_seq"].dropna().max()) if daily["max_workday_seq"].notna().any() else 1
    title_col, controls_col = st.columns([0.38, 0.62], vertical_alignment="center")
    with title_col:
        st.subheader("累计销量斜率")
    with controls_col:
        with st.container(horizontal=True, horizontal_alignment="right", vertical_alignment="center", gap=None):
            render_inline_label("指标")
            selected_metric = st.segmented_control(
                "指标",
                list(SLOPE_METRIC_OPTIONS.keys()),
                default="月末 X 工作日累计销量斜率",
                key="slope_metric",
                label_visibility="collapsed",
                width="content",
            )
            render_inline_label("窗口 X")
            selected_window = st.number_input(
                "窗口 X",
                min_value=1,
                max_value=max_workdays,
                value=min(5, max_workdays),
                step=1,
                key="slope_window",
                label_visibility="collapsed",
                width=COMPACT_NUMBER_INPUT_WIDTH,
            )
    selected_metric = selected_metric or "月末 X 工作日累计销量斜率"
    analysis_df = build_monthly_analysis(daily, SLOPE_METRIC_OPTIONS[selected_metric], int(selected_window))
    analysis_df = analysis_df.dropna(subset=["selected_slope_per_avg_daily_qty"]).reset_index(drop=True)

    if len(analysis_df) < MIN_SEGMENT_MONTHS * 2:
        st.warning("可用月份不足，暂无法进行稳定的多断点和结构变化分析。", icon=":material/warning:")
        show_dataframe_12_rows(analysis_df, hide_index=True)
        return

    slope_series = analysis_df["selected_slope_per_avg_daily_qty"].astype(float).to_numpy()
    cp_breaks, _ = dynamic_breaks_for_mean(slope_series, MAX_CHANGEPOINTS, MIN_SEGMENT_MONTHS)
    cp_summary_df = summarize_mean_changepoints(analysis_df, cp_breaks)

    structure_df = analysis_df.dropna(
        subset=["selected_slope_per_avg_daily_qty", "anchor_pred_error_pct", "anchor_pred_month_total"]
    ).reset_index(drop=True)
    if len(structure_df) >= MIN_SEGMENT_MONTHS * 2:
        structural_breaks, structural_fits = dynamic_breaks_for_ewols(structure_df, MAX_STRUCTURAL_BREAKS, MIN_SEGMENT_MONTHS)
        structure_summary_df = summarize_structural_breaks(structure_df, structural_breaks, structural_fits)
    else:
        structural_breaks, structural_fits, structure_summary_df = [], [], pd.DataFrame()

    if cp_summary_df.empty:
        cp_text = "未识别到稳定的历史突变点。当前窗口下，斜率节奏更像连续波动。"
    else:
        latest_cp = cp_summary_df.iloc[-1]
        cp_text = f"最近一次突变出现在 **{latest_cp['突变月份']}**，之后表现为 **{latest_cp['业务解读']}**，变化率约 **{latest_cp['变化率']:.1f}%**。"
    st.info(f"当前口径：**{selected_metric}**，窗口 **X={int(selected_window)}**。{cp_text}", icon=":material/insights:")

    left, right = st.columns(2, gap="medium")
    with left:
        with st.container(border=True):
            st.subheader("多变点检测")
            st.caption("识别所选累计销量斜率的历史突变月份。红色虚线为突变点，统计方法为带最小区间约束的动态规划多变点检测。")
            st.altair_chart(changepoint_altair_chart(analysis_df, cp_breaks))
            if cp_summary_df.empty:
                st.write("当前窗口未识别到稳定突变点。")
            else:
                show_dataframe_12_rows(
                    cp_summary_df,
                    hide_index=True,
                    column_config={
                        "突变前均值": st.column_config.NumberColumn(format="%.3f"),
                        "突变后均值": st.column_config.NumberColumn(format="%.3f"),
                        "变化幅度": st.column_config.NumberColumn(format="%.3f"),
                        "变化率": st.column_config.NumberColumn(format="%.1f%%"),
                    },
                )

    with right:
        with st.container(border=True):
            st.subheader("EWOLS Bai-Perron 结构变化")
            st.caption("纵轴为 anchor 预测误差%。分段线使用 EWOLS 拟合，BIC 选择 0-2 个结构断点，用于判断斜率和预测误差关系是否换挡。")
            if structure_df.empty or not structural_fits:
                st.write("anchor 预测误差样本不足，暂不能做结构变化分析。")
            else:
                st.altair_chart(structural_altair_chart(structure_df, structural_breaks, structural_fits))
                show_dataframe_12_rows(
                    structure_summary_df,
                    hide_index=True,
                    column_config={
                        "EWOLS 斜率系数": st.column_config.NumberColumn(format="%.3f"),
                        "平均预测误差%": st.column_config.NumberColumn(format="%.1f%%"),
                        "平均实际销量": st.column_config.NumberColumn(format="%.0f"),
                    },
                )

    with st.expander("方法说明与可复核数据", expanded=False):
        st.markdown(
            """
            - **累计销量斜率**：对窗口内的 MTD 累计销量做线性斜率，并除以窗口日均销量，减少规模变化带来的误读。
            - **anchor 预测当月销量**：在每个 anchor 月只使用此前历史月份训练 EWOLS，特征包括 anchor MTD、工作日位置、剩余工作日、当月工作日数、月份季节性、上月实际总量。
            - **多变点检测**：在斜率序列上寻找均值水平切换点；每段至少保留 8 个月，避免把短期噪声解释成结构变化。
            - **EWOLS Bai-Perron**：在每个候选时间段内拟合 `anchor预测误差% ~ 标准化斜率 + 时间趋势`，并用指数衰减权重强调近期月份。
            """
        )
        show_dataframe_12_rows(
            analysis_df[
                [
                    "month_label",
                    "anchor_date",
                    "anchor_rule",
                    "actual_month_total",
                    "anchor_pred_month_total",
                    "anchor_pred_error_pct",
                    "selected_slope",
                    "selected_slope_per_avg_daily_qty",
                    "anchor_mtd_qty",
                    "ewols_train_months",
                ]
            ],
            hide_index=True,
            column_config={
                "anchor_date": st.column_config.DateColumn("anchor日期"),
                "actual_month_total": st.column_config.NumberColumn("实际月销量", format="%.0f"),
                "anchor_pred_month_total": st.column_config.NumberColumn("anchor预测月销量", format="%.0f"),
                "anchor_pred_error_pct": st.column_config.NumberColumn("anchor预测误差%", format="%.1f%%"),
                "selected_slope": st.column_config.NumberColumn("累计销量斜率", format="%.1f"),
                "selected_slope_per_avg_daily_qty": st.column_config.NumberColumn("标准化斜率", format="%.3f"),
                "anchor_mtd_qty": st.column_config.NumberColumn("anchor MTD", format="%.0f"),
                "ewols_train_months": st.column_config.NumberColumn("EWOLS训练月数", format="%.0f"),
            },
        )


def render_workday_avg_sales_tab(panel: pd.DataFrame) -> None:
    title_col, controls_col = st.columns([0.38, 0.62], vertical_alignment="center")
    with title_col:
        st.subheader("工作日平均销量所受影响分析")
    with controls_col:
        with st.container(horizontal=True, horizontal_alignment="right", vertical_alignment="center", gap=None):
            render_inline_label("统计粒度")
            selected_grains = st.pills(
                "统计粒度",
                GRAIN_OPTIONS,
                default=["年", "月"],
                selection_mode="multi",
                help="选择“年+月”代表 YYYY-MM 粒度；只选“月”代表跨年份同自然月聚合；只选“年”代表年度聚合。",
                label_visibility="collapsed",
                width="content",
            )
            render_inline_label("指标类型")
            value_type = st.segmented_control(
                "指标类型",
                VALUE_TYPE_OPTIONS,
                default="统计值",
                help="非日历指标会跟随该口径切换；日历指标保持原始天数/占比。",
                label_visibility="collapsed",
                width="content",
            )
    st.caption("本页只做数据探索和统计相关分析，不生成建模特征，也不输出预测结论。")
    if not selected_grains:
        selected_grains = ["年", "月"]
    if value_type is None:
        value_type = "统计值"

    selected_grains = [g for g in GRAIN_OPTIONS if g in selected_grains]
    grain_label = " + ".join(selected_grains)
    aggregated = aggregate_panel(panel, selected_grains)
    corr_df = correlation_table(aggregated, value_type)
    target_col = target_col_for_value_type(value_type)
    target_label = target_label_for_value_type(value_type)

    with st.container(border=True):
        st.subheader("工作日平均销量趋势")
        render_target_trend(panel, value_type)

    render_stationarity_test(panel, value_type)
    render_lag_difference_chart(panel, value_type)
    render_smoothing_row(panel, value_type)
    render_stl_section(panel, value_type)

    with st.container(border=True):
        st.subheader("相关性排名")
        render_correlation_bar(corr_df)

    with st.container(border=True):
        title_col, controls_col = st.columns(TITLE_CONTROL_LAYOUT, vertical_alignment="center")
        factor_options = scatter_factor_options(corr_df)
        with title_col:
            st.subheader("因素散点关系")
        with controls_col:
            with st.container(horizontal=True, horizontal_alignment="right", vertical_alignment="center", gap=None):
                render_inline_label("因素")
                selected_factor_label = (
                    st.selectbox(
                        "因素",
                        list(factor_options.keys()),
                        label_visibility="collapsed",
                        width=COMPACT_SELECT_WIDTH,
                    )
                    if factor_options
                    else None
                )
        render_scatter(aggregated, value_type, factor_options, selected_factor_label)

    with st.container(border=True):
        st.subheader("统计明细")
        display_df = corr_df.copy()
        display_df["Pearson r"] = display_df["Pearson r"].map(fmt_corr)
        display_df["p-value"] = display_df["p-value"].map(fmt_pvalue)
        display_df["|r|"] = display_df["|r|"].map(lambda x: fmt_float(x, 3))
        st.dataframe(
            display_df[["期间", "指标", "样本数", "Pearson r", "p-value", "|r|", "方向", "效应强度"]],
            hide_index=True,
        )

    with st.expander("聚合后的分析数据", expanded=False):
        visible_cols = [
            "分析粒度",
            "样本月份数",
            "prev_workdays",
            "curr_workdays",
            "next_workdays",
            f"prev_{suffix_for_value_type('avg_qty_per_workday', value_type)}",
            target_col,
            f"next_{suffix_for_value_type('avg_qty_per_workday', value_type)}",
            f"prev_{suffix_for_value_type('avg_num_hosp_per_workday', value_type)}",
            "prev_non_workdays",
            "curr_non_workdays",
            "next_non_workdays",
            "prev_weekday_holidays",
            "curr_weekday_holidays",
            "next_weekday_holidays",
            "prev_weekend_rest_days",
            "curr_weekend_rest_days",
            "next_weekend_rest_days",
            "curr_same_month_past_avg_qty_per_workday",
            "curr_m1_vs_m2_avg_qty_per_workday_change",
        ]
        display_cols = list(dict.fromkeys(c for c in visible_cols if c in aggregated.columns))
        st.dataframe(
            aggregated[display_cols],
            hide_index=True,
            column_config={
                target_col: st.column_config.NumberColumn(
                    target_label,
                    format="percent" if value_type in {"环比值", "同比值"} else "%.0f",
                ),
                f"prev_{suffix_for_value_type('avg_qty_per_workday', value_type)}": st.column_config.NumberColumn(
                    f"上月平均每工作日销量{'' if value_type == '统计值' else value_type[:2]}",
                    format="percent" if value_type in {"环比值", "同比值"} else "%.0f",
                ),
                f"next_{suffix_for_value_type('avg_qty_per_workday', value_type)}": st.column_config.NumberColumn(
                    f"下月平均每工作日销量{'' if value_type == '统计值' else value_type[:2]}",
                    format="percent" if value_type in {"环比值", "同比值"} else "%.0f",
                ),
                f"prev_{suffix_for_value_type('avg_num_hosp_per_workday', value_type)}": st.column_config.NumberColumn(
                    f"上月平均每工作日医院数{'' if value_type == '统计值' else value_type[:2]}",
                    format="percent" if value_type in {"环比值", "同比值"} else "%.0f",
                ),
                "curr_same_month_past_avg_qty_per_workday": st.column_config.NumberColumn(
                    "过去平均相同月份的平均工作日销量",
                    format="%.0f",
                ),
                "curr_m1_vs_m2_avg_qty_per_workday_change": st.column_config.NumberColumn(
                    "M-1 相对 M-2 平均工作日销量变化",
                    format="percent",
                ),
            },
        )


st.set_page_config(
    page_title="统计相关分析",
    page_icon=":material/monitoring:",
    layout="wide",
)

st.title("预测任务统计相关分析")
st.caption("围绕 README 中的月总量预测契约，先观察销售强度与日历结构之间的统计关系。")

with st.sidebar:
    st.header("数据")
    data_path = st.text_input("日销量数据路径", value=str(DEFAULT_DAILY_PATH))
    st.caption("需要包含 bizym、transdate、qty；num_hosp 可选。")

daily_df = load_daily_data(data_path)
monthly_df = build_monthly_stats(daily_df)
neighbor_panel = build_neighbor_panel(monthly_df)

tab_workday_avg, tab_contribution, tab_slope = st.tabs(["工作日平均销量", "贡献率稳定性", "累计销量斜率"])
with tab_workday_avg:
    render_workday_avg_sales_tab(neighbor_panel)
with tab_contribution:
    render_contribution_stability_tab(daily_df)
with tab_slope:
    render_slope_break_tab(daily_df)
