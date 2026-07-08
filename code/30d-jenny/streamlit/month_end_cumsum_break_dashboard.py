"""
Month-end / month-start cumulative sales slope break dashboard.

Run:
    streamlit run code/30d-jenny/streamlit/month_end_cumsum_break_dashboard.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import streamlit as st

try:
    from chinese_calendar import is_workday as cn_is_workday
except Exception:  # pragma: no cover - app degrades gracefully when calendar is absent.
    cn_is_workday = None


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DAILY_PATH = REPO_ROOT / "data" / "sales_daily.csv"

METRIC_OPTIONS = {
    "月末 X 工作日累计销量斜率": "end",
    "月初 X 工作日累计销量斜率": "start",
}

CONTRIBUTION_SEGMENT_OPTIONS = {
    "期初25%": 0,
    "中前25%": 1,
    "中后25%": 2,
    "期末25%": 3,
}

GRAIN_OPTIONS = ["年", "月"]

STABILITY_LABEL_STABLE = "稳定"
STABILITY_LABEL_WATCH = "需观察"
STABILITY_LABEL_UNSTABLE = "不稳定"
STABILITY_ORDER = {
    STABILITY_LABEL_STABLE: 0,
    STABILITY_LABEL_WATCH: 1,
    STABILITY_LABEL_UNSTABLE: 2,
}
CV_STABLE_MAX = 0.20
YOY_STABLE_MAX = 0.06
MAX_YOY_STABLE_MAX = 0.05
LOYO_WAPE_STABLE_MAX = 0.20
TABLE_VISIBLE_ROWS = 12
TABLE_ROW_HEIGHT = 35
TABLE_HEADER_HEIGHT = 38
RETINA_DPI = 220

MIN_SEGMENT_MONTHS = 8
MAX_CHANGEPOINTS = 3
MAX_STRUCTURAL_BREAKS = 2
EWOLS_HALFLIFE_MONTHS = 6.0
EWOLS_FORECAST_MIN_TRAIN_MONTHS = 10
CHANGEPOINT_PENALTY_SCALE = 2.0

COLOR_MAIN = "#2f6f73"
COLOR_ACCENT = "#d95f02"
COLOR_DANGER = "#c44e52"
COLOR_MUTED = "#7a7f87"


def configure_matplotlib_fonts() -> None:
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in ["PingFang SC", "Arial Unicode MS", "Heiti TC", "Songti SC"]:
        if font_name in available_fonts:
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False


configure_matplotlib_fonts()


@dataclass(frozen=True)
class SegmentFit:
    start: int
    end: int
    rss: float
    coef: np.ndarray
    nobs: int


def fmt_num(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{value:,.0f}"


def fmt_pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{value:.2f}%"


def fmt_pct_1(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{value:.1f}%"


def is_business_workday(ts: pd.Timestamp) -> bool:
    if cn_is_workday is not None:
        return bool(cn_is_workday(pd.Timestamp(ts).date()))
    return pd.Timestamp(ts).weekday() < 5


def linear_slope(values: pd.Series | np.ndarray) -> float:
    y = pd.Series(values).dropna().astype(float)
    if len(y) < 2:
        return np.nan
    x = np.arange(1, len(y) + 1, dtype=float)
    return float(np.polyfit(x, y.to_numpy(), 1)[0])


def safe_divide(numerator: float, denominator: float) -> float:
    if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
        return np.nan
    return float(numerator) / float(denominator)


def group_columns(grains: list[str]) -> list[str]:
    columns = []
    if "年" in grains:
        columns.append("year")
    if "月" in grains:
        columns.append("month")
    return columns


def make_group_label(row: pd.Series, grains: list[str]) -> str:
    if not grains:
        return "全部"
    parts = []
    if "年" in grains:
        parts.append(f"{int(row['year'])}年")
    if "月" in grains:
        parts.append(f"{int(row['month']):02d}月")
    return "-".join(parts)


def classify_threshold(value: float | int | None, stable_max: float, watch_max: float) -> str:
    if value is None or pd.isna(value):
        return STABILITY_LABEL_WATCH
    value = abs(float(value))
    if value <= stable_max:
        return STABILITY_LABEL_STABLE
    if value <= watch_max:
        return STABILITY_LABEL_WATCH
    return STABILITY_LABEL_UNSTABLE


def classify_binary_threshold(value: float | int | None, stable_max: float) -> str:
    if value is None or pd.isna(value):
        return STABILITY_LABEL_UNSTABLE
    return STABILITY_LABEL_STABLE if abs(float(value)) <= stable_max else STABILITY_LABEL_UNSTABLE


def worst_stability_label(labels: pd.Series | list[str]) -> str:
    valid = [label for label in list(labels) if label in STABILITY_ORDER]
    if not valid:
        return STABILITY_LABEL_WATCH
    return max(valid, key=lambda label: STABILITY_ORDER[label])


def add_stability_columns(
    df: pd.DataFrame,
    value_col: str,
    stable_max: float,
    watch_max: float,
    basis_template: str,
) -> pd.DataFrame:
    out = df.copy()
    if out.empty or value_col not in out.columns:
        out["稳定性"] = pd.Series(dtype="object")
        out["判定依据"] = pd.Series(dtype="object")
        return out
    out["稳定性"] = out[value_col].map(lambda value: classify_threshold(value, stable_max, watch_max))
    out["判定依据"] = out[value_col].map(
        lambda value: basis_template.format(
            value=fmt_pct(abs(value) * 100) if pd.notna(value) else "-",
            stable=fmt_pct(stable_max * 100),
            watch=fmt_pct(watch_max * 100),
        )
    )
    return out


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
    watch_count = int(valid["稳定性"].eq(STABILITY_LABEL_WATCH).sum())
    total_count = int(len(valid))
    if unstable_count:
        reason = f"{total_count} 行中有 {unstable_count} 行不稳定，整体判定为不稳定。"
    elif watch_count:
        reason = f"{total_count} 行中有 {watch_count} 行需观察，整体判定为需观察。"
    else:
        reason = f"{total_count} 行全部满足稳定阈值，整体判定为稳定。"
    return label, reason


def stability_summary_from_labels(labels: pd.Series | list[str]) -> tuple[str, str]:
    valid = [label for label in list(labels) if label in STABILITY_ORDER]
    if not valid:
        return STABILITY_LABEL_WATCH, "有效样本不足，暂按需观察处理。"

    label = worst_stability_label(valid)
    unstable_count = valid.count(STABILITY_LABEL_UNSTABLE)
    watch_count = valid.count(STABILITY_LABEL_WATCH)
    total_count = len(valid)
    if unstable_count:
        reason = f"{total_count} 行中有 {unstable_count} 行不稳定，整体判定为不稳定。"
    elif watch_count:
        reason = f"{total_count} 行中有 {watch_count} 行需观察，整体判定为需观察。"
    else:
        reason = f"{total_count} 行全部满足稳定阈值，整体判定为稳定。"
    return label, reason


def render_stability_basis(title: str, label: str, reason: str, threshold_text: str) -> None:
    st.info(
        f"**整体标签：{label}**。{reason} 判定阈值：{threshold_text}",
        icon=":material/rule:",
    )


def dataframe_height(row_count: int) -> int:
    visible_rows = min(max(int(row_count), 1), TABLE_VISIBLE_ROWS)
    return TABLE_HEADER_HEIGHT + visible_rows * TABLE_ROW_HEIGHT


def show_dataframe_12_rows(df: pd.DataFrame, **kwargs) -> None:
    st.dataframe(df, height=dataframe_height(len(df)), **kwargs)


@st.cache_data(show_spinner="加载日销量数据...")
def load_daily_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"bizym", "transdate", "qty"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["transdate"] = pd.to_datetime(df["transdate"])
    df["bizym"] = df["bizym"].astype(int)
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0.0)
    df["month_start"] = df["transdate"].dt.to_period("M").dt.to_timestamp()
    df["month_label"] = df["month_start"].dt.strftime("%Y-%m")
    df["year"] = df["bizym"] // 100
    df["month"] = df["bizym"] % 100
    df["is_workday"] = df["transdate"].map(is_business_workday)
    df = df.sort_values(["bizym", "transdate"]).reset_index(drop=True)

    df["workday_seq"] = df.groupby("bizym")["is_workday"].cumsum().where(df["is_workday"])
    month_workdays = (
        df[df["is_workday"]]
        .groupby("bizym", as_index=False)["workday_seq"]
        .max()
        .rename(columns={"workday_seq": "max_workday_seq"})
    )
    df = df.merge(month_workdays, on="bizym", how="left")
    df["mtd_qty"] = df.groupby("bizym")["qty"].cumsum()
    month_total = df.groupby("bizym", as_index=False)["qty"].sum().rename(columns={"qty": "actual_month_total"})
    df = df.merge(month_total, on="bizym", how="left")
    df["mtd_pct"] = df["mtd_qty"] / df["actual_month_total"].replace(0, np.nan)
    return df


@st.cache_data(show_spinner="构建月度斜率与 anchor 预测误差...")
def build_monthly_analysis(daily: pd.DataFrame, position: str, window: int) -> pd.DataFrame:
    workdays = daily[daily["is_workday"]].copy()
    workdays["workday_seq"] = workdays["workday_seq"].astype(int)
    workdays["max_workday_seq"] = workdays["max_workday_seq"].astype(int)
    workdays["reverse_workday_seq"] = workdays["max_workday_seq"] - workdays["workday_seq"] + 1

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
        anchor_mtd_pct = safe_divide(anchor_mtd, actual_total)

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
                "anchor_mtd_pct": anchor_mtd_pct,
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
    if grains:
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
    else:
        yoy = (
            yoy_base.groupby("窗口", dropna=False)
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
    max_yoy["分析粒度"] = max_yoy.apply(lambda row: make_group_label(row, grains), axis=1)

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


def add_anchor_ewols_predictions(monthly: pd.DataFrame) -> pd.DataFrame:
    out = monthly.copy()
    out["anchor_pred_month_total"] = np.nan
    out["anchor_pred_error"] = np.nan
    out["anchor_pred_error_pct"] = np.nan
    out["ewols_train_months"] = np.nan

    all_features = make_anchor_forecast_features(out)
    y_all = out["actual_month_total"].astype(float)

    for idx in range(len(out)):
        train_idx = list(range(idx))
        train_x = all_features.iloc[train_idx].copy()
        train_y = y_all.iloc[train_idx].copy()
        current_x = all_features.iloc[[idx]].copy()

        valid_train = train_x.notna().all(axis=1) & train_y.notna()
        if int(valid_train.sum()) < EWOLS_FORECAST_MIN_TRAIN_MONTHS or current_x.isna().any(axis=None):
            continue

        train_x = train_x.loc[valid_train]
        train_y = train_y.loc[valid_train]
        x_train_mat, x_current_mat = standardize_by_train(train_x, current_x)
        coef, _, _ = ewols_fit(train_y.to_numpy(dtype=float), x_train_mat, EWOLS_HALFLIFE_MONTHS)
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
            candidates = []
            for split in range((seg - 1) * min_size, end - min_size + 1):
                candidates.append((sse[seg - 1, split] + mean_segment_sse(y, split, end), split))
            if candidates:
                best_cost, best_split = min(candidates, key=lambda x: x[0])
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
    if df.empty:
        return pd.DataFrame()
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
            candidates = []
            for split in range((seg - 1) * min_size, end - min_size + 1):
                candidates.append((cost[seg - 1, split] + segment_fit(split, end).rss, split))
            if candidates:
                best_cost, best_split = min(candidates, key=lambda x: x[0])
                cost[seg, end] = best_cost
                prev[seg, end] = best_split

    y = df["anchor_pred_error_pct"].astype(float).to_numpy()
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
    for i, fit in enumerate(fits):
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


def changepoint_chart(df: pd.DataFrame, breaks: list[int]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=RETINA_DPI)
    ax.plot(
        df["month_start"],
        df["selected_slope_per_avg_daily_qty"],
        color=COLOR_MAIN,
        marker="o",
        linewidth=2.5,
        markersize=4.5,
        label="标准化累计销量斜率",
    )
    for split in breaks:
        row = df.iloc[split]
        ax.axvline(row["month_start"], color=COLOR_DANGER, linestyle="--", linewidth=1.8)
        ax.annotate(
            str(row["month_label"]),
            xy=(row["month_start"], ax.get_ylim()[1]),
            xytext=(5, -8),
            textcoords="offset points",
            color=COLOR_DANGER,
            fontsize=9,
            ha="left",
            va="top",
        )
    ax.set_xlabel("月份")
    ax.set_ylabel("累计销量斜率 / 窗口日均销量")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, len(df) // 8)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(rotation=30, ha="right")
    fig.tight_layout()
    return fig


def structural_chart(df: pd.DataFrame, breaks: list[int], fits: list[SegmentFit]) -> plt.Figure:
    plot_df = df.copy()
    plot_df["segment"] = "整体"
    boundaries = [0] + breaks + [len(plot_df)]
    for i in range(len(boundaries) - 1):
        plot_df.loc[boundaries[i] : boundaries[i + 1] - 1, "segment"] = f"阶段 {i + 1}"

    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=RETINA_DPI)
    palette = [COLOR_MAIN, COLOR_ACCENT, "#4c72b0", COLOR_DANGER]
    for i, (segment_name, segment_df) in enumerate(plot_df.groupby("segment", sort=False)):
        ax.scatter(
            segment_df["selected_slope_per_avg_daily_qty"],
            segment_df["anchor_pred_error_pct"],
            s=45,
            color=palette[i % len(palette)],
            alpha=0.82,
            label=segment_name,
        )

    for i, fit in enumerate(fits):
        segment = plot_df.iloc[fit.start : fit.end].copy()
        if segment.empty or len(fit.coef) < 3:
            continue
        x_raw = segment["selected_slope_per_avg_daily_qty"].astype(float)
        x_grid = np.linspace(float(x_raw.min()), float(x_raw.max()), 20)
        slope_mean = float(x_raw.mean())
        slope_std = float(x_raw.std(ddof=0)) if float(x_raw.std(ddof=0)) > 0 else 1.0
        t_mid = 0.0
        y_grid = fit.coef[0] + fit.coef[1] * ((x_grid - slope_mean) / slope_std) + fit.coef[2] * t_mid
        ax.plot(
            x_grid,
            y_grid,
            color=palette[i % len(palette)],
            linewidth=2.5,
            label=f"阶段 {i + 1} EWOLS",
        )
    ax.axhline(0, color=COLOR_MUTED, linewidth=1, linestyle=":")
    ax.set_xlabel("累计销量斜率 / 窗口日均销量")
    ax.set_ylabel("anchor预测误差%")
    ax.legend(loc="best", frameon=False, fontsize=9)
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def bar_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    y_label: str,
    percent_axis: bool = True,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 3.8), dpi=RETINA_DPI)
    if df.empty or y_col not in df.columns or x_col not in df.columns or "窗口" not in df.columns:
        ax.text(0.5, 0.5, "无可用样本", ha="center", va="center", color=COLOR_MUTED)
        ax.axis("off")
        return fig

    plot_df = df.dropna(subset=[y_col]).copy()
    plot_df[x_col] = plot_df[x_col].astype(str)
    if plot_df.empty:
        ax.text(0.5, 0.5, "无可用样本", ha="center", va="center", color=COLOR_MUTED)
        ax.axis("off")
        return fig

    windows = plot_df["窗口"].drop_duplicates().tolist()
    x_labels = plot_df[x_col].drop_duplicates().tolist()
    x = np.arange(len(x_labels))
    width = min(0.75 / max(len(windows), 1), 0.28)
    palette = [COLOR_MAIN, COLOR_ACCENT, "#4c72b0", COLOR_DANGER]
    for i, window_name in enumerate(windows):
        values = (
            plot_df[plot_df["窗口"].eq(window_name)]
            .set_index(x_col)
            .reindex(x_labels)[y_col]
            .astype(float)
            .to_numpy()
        )
        ax.bar(x + (i - (len(windows) - 1) / 2) * width, values, width=width, label=window_name, color=palette[i % len(palette)])

    ax.set_title(title, loc="left", fontsize=12)
    ax.set_ylabel(y_label)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=30, ha="right")
    if percent_axis:
        ax.yaxis.set_major_formatter(lambda value, _: f"{value * 100:.0f}%")
    ax.axhline(0, color=COLOR_MUTED, linewidth=0.8, alpha=0.6)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best", frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def render_contribution_stability_tab(daily: pd.DataFrame) -> None:
    st.subheader("贡献率稳定性")
    st.caption("将每个月按工作日进度切成四段，分析所选窗口销量对当月总量的贡献率稳定性。")

    with st.container(border=True):
        c1, c2, c3 = st.columns([2.2, 1.4, 1.2])
        with c1:
            selected_segments = st.multiselect(
                "参数 1：工作日进度窗口",
                list(CONTRIBUTION_SEGMENT_OPTIONS.keys()),
                default=list(CONTRIBUTION_SEGMENT_OPTIONS.keys()),
                key="contribution_segments",
            )
        with c2:
            selected_grains = st.multiselect(
                "参数 2：统计粒度",
                GRAIN_OPTIONS,
                default=["月"],
                key="contribution_grains",
            )
        with c3:
            aggregate_segments = st.checkbox(
                "聚合所选窗口",
                value=False,
                help="开启后，参数 1 中勾选的多个 25% 段会合并成一个整体窗口。",
            )

    if not selected_segments:
        st.warning("请至少选择一个工作日进度窗口。", icon=":material/warning:")
        return

    detail = build_contribution_detail(daily, tuple(selected_segments), aggregate_segments)
    summaries = summarize_contribution_stability(detail, selected_grains)
    cv_df = summaries["cv"]
    yoy_df = summaries["yoy"]
    max_yoy_df = summaries["max_yoy"]
    loyo_df = summaries["loyo"]
    cv_df = add_binary_stability_columns(
        cv_df,
        "CV",
        CV_STABLE_MAX,
        "CV={value}；≤{stable} 稳定，>{stable} 不稳定。",
    )
    yoy_df = add_binary_stability_columns(
        yoy_df,
        "平均YoY变化",
        YOY_STABLE_MAX,
        "平均YoY绝对变化={value}；≤{stable} 稳定，>{stable} 不稳定。",
    )
    max_yoy_df = add_binary_stability_columns(
        max_yoy_df,
        "最大YoY变化",
        MAX_YOY_STABLE_MAX,
        "最大YoY绝对变化={value}；≤{stable} 稳定，>{stable} 不稳定。",
    )
    loyo_df = add_binary_stability_columns(
        loyo_df,
        "LOYO-WAPE",
        LOYO_WAPE_STABLE_MAX,
        "LOYO-WAPE={value}；≤{stable} 稳定，>{stable} 不稳定。",
    )

    with st.container(horizontal=True):
        st.metric("窗口指标数", f"{detail['窗口'].nunique()}个", border=True)
        st.metric("覆盖月份", f"{detail['bizym'].nunique()}个月", border=True)
        st.metric("统计粒度", " + ".join(selected_grains) if selected_grains else "全部聚合", border=True)
        best = (
            loyo_df.dropna(subset=["LOYO-WAPE"]).sort_values("LOYO-WAPE").head(1)
            if "LOYO-WAPE" in loyo_df.columns
            else pd.DataFrame()
        )
        st.metric("最低 LOYO-WAPE", fmt_pct(best["LOYO-WAPE"].iloc[0] * 100) if not best.empty else "-", border=True)

    with st.container(border=True):
        st.subheader("CV 稳定性评估")
        cv_label, cv_reason = stability_summary(cv_df, "CV")
        render_stability_basis(
            "CV",
            cv_label,
            cv_reason,
            f"CV≤{fmt_pct(CV_STABLE_MAX * 100)} 为稳定，>{fmt_pct(CV_STABLE_MAX * 100)} 为不稳定。",
        )
        st.caption("CV = 贡献率标准差 / 平均贡献率；越低表示该窗口对月总量的贡献越稳定。")
        st.pyplot(bar_chart(cv_df, "分析粒度", "CV", "贡献率 CV", "CV"))
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
            yoy_display["最大YoY判定依据"] = "有效同比样本不足，暂按需观察处理。"
        yoy_display["综合稳定性"] = yoy_display.apply(
            lambda row: worst_stability_label([row.get("avgYoY 稳定性"), row.get("maxYoY稳定性")]),
            axis=1,
        )
        yoy_label, yoy_reason = stability_summary_from_labels(yoy_display["综合稳定性"])
        render_stability_basis(
            "YoY变化",
            yoy_label,
            yoy_reason,
            (
                f"平均YoY绝对变化≤{fmt_pct_1(YOY_STABLE_MAX * 100)} 为稳定，>{fmt_pct_1(YOY_STABLE_MAX * 100)} 为不稳定；"
                f"最大YoY绝对变化≤{fmt_pct_1(MAX_YOY_STABLE_MAX * 100)} 为稳定，>{fmt_pct_1(MAX_YOY_STABLE_MAX * 100)} 为不稳定。"
            ),
        )
        st.caption("YoY变化 = 当年贡献率 - 上年贡献率；最大 YoY 变化取当前统计粒度内绝对值最大的一次同比变化。")
        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(bar_chart(yoy_df, "分析粒度", "平均YoY变化", "平均 YoY 变化", "贡献率差值"))
        with c2:
            st.pyplot(bar_chart(max_yoy_df, "分析粒度", "最大YoY变化", "最大 YoY 变化", "abs(贡献率差值)"))
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
        render_stability_basis(
            "LOYO-WAPE",
            loyo_label,
            loyo_reason,
            f"LOYO-WAPE≤{fmt_pct(LOYO_WAPE_STABLE_MAX * 100)} 为稳定，>{fmt_pct(LOYO_WAPE_STABLE_MAX * 100)} 为不稳定。",
        )
        st.caption(
            "每次留出一个年份，用其他年份同月同窗口贡献率中位数反推测试年份月总量，再汇总 WAPE；越低表示贡献率基准越可复用。"
        )
        if loyo_df.empty:
            st.write("历史年份不足，暂无法计算 LOYO-WAPE。")
        else:
            st.pyplot(bar_chart(loyo_df, "分析粒度", "LOYO-WAPE", "LOYO-WAPE", "WAPE"))
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


def display_business_summary(
    selected_label: str,
    window: int,
    analysis: pd.DataFrame,
    cp_summary: pd.DataFrame,
    structure_summary: pd.DataFrame,
) -> None:
    latest = analysis.dropna(subset=["selected_slope_per_avg_daily_qty"]).iloc[-1]
    with st.container(horizontal=True):
        st.metric("可分析月份", f"{analysis['bizym'].nunique()}个月", border=True)
        st.metric("最新月份", latest["month_label"], fmt_num(latest["actual_month_total"]), border=True)
        st.metric("最新标准化斜率", f"{latest['selected_slope_per_avg_daily_qty']:.3f}", border=True)
        st.metric("最新 anchor 误差", fmt_pct(latest["anchor_pred_error_pct"]), border=True)

    if cp_summary.empty:
        cp_text = "未识别到稳定的历史突变点。当前窗口下，斜率节奏更像连续波动，而不是明确换挡。"
    else:
        first = cp_summary.iloc[-1]
        cp_text = (
            f"最近一次突变出现在 **{first['突变月份']}**，"
            f"之后表现为 **{first['业务解读']}**，变化率约 **{first['变化率']:.1f}%**。"
        )

    if structure_summary.empty or structure_summary["结构变化月份"].mask(structure_summary["结构变化月份"].eq("")).dropna().empty:
        bp_text = "EWOLS 分段回归未发现足够稳定的结构断点，斜率与 anchor 预测误差的关系暂未显示阶段性换挡。"
    else:
        breaks = structure_summary["结构变化月份"].replace("", np.nan).dropna().tolist()
        bp_text = f"EWOLS 分段回归识别到关系变化月份：**{', '.join(breaks)}**。这些月份前后，斜率对预测误差的方向或强度发生了变化。"

    st.info(
        f"当前口径：**{selected_label}**，窗口 **X={window}**。{cp_text} {bp_text}",
        icon=":material/insights:",
    )


st.set_page_config(
    page_title="月末累计销量斜率突变分析",
    page_icon=":material/analytics:",
    layout="wide",
)

st.title("月内销量节奏稳定性分析")
st.caption(
    "面向业务汇报：先评估工作日窗口贡献率是否稳定，再回答累计销量斜率在哪里突变、与 anchor 预测误差的关系是否换挡。"
)

daily = load_daily_data(str(DEFAULT_DAILY_PATH))
max_workdays = int(daily["max_workday_seq"].dropna().max())

with st.sidebar:
    st.header("累计销量斜率参数")
    selected_metric = st.selectbox("指标选择", list(METRIC_OPTIONS.keys()))
    selected_window = st.number_input("窗口选择 X", min_value=1, max_value=max_workdays, value=min(5, max_workdays), step=1)
    st.caption(f"当前数据中每月最多 {max_workdays} 个工作日。")

contribution_tab, slope_tab = st.tabs(["贡献率稳定性", "累计销量斜率"])

with contribution_tab:
    render_contribution_stability_tab(daily)

with slope_tab:
    position = METRIC_OPTIONS[selected_metric]
    analysis_df = build_monthly_analysis(daily, position, int(selected_window))
    analysis_df = analysis_df.dropna(subset=["selected_slope_per_avg_daily_qty"]).reset_index(drop=True)

    if len(analysis_df) < MIN_SEGMENT_MONTHS * 2:
        st.warning("可用月份不足，暂无法进行稳定的多断点和结构变化分析。", icon=":material/warning:")
        show_dataframe_12_rows(analysis_df, hide_index=True)
    else:
        slope_series = analysis_df["selected_slope_per_avg_daily_qty"].astype(float).to_numpy()
        cp_breaks, _ = dynamic_breaks_for_mean(slope_series, MAX_CHANGEPOINTS, MIN_SEGMENT_MONTHS)
        cp_summary_df = summarize_mean_changepoints(analysis_df, cp_breaks)

        structure_df = analysis_df.dropna(
            subset=["selected_slope_per_avg_daily_qty", "anchor_pred_error_pct", "anchor_pred_month_total"]
        ).reset_index(drop=True)
        if len(structure_df) >= MIN_SEGMENT_MONTHS * 2:
            structural_breaks, structural_fits = dynamic_breaks_for_ewols(
                structure_df, MAX_STRUCTURAL_BREAKS, MIN_SEGMENT_MONTHS
            )
            structure_summary_df = summarize_structural_breaks(structure_df, structural_breaks, structural_fits)
        else:
            structural_breaks, structural_fits, structure_summary_df = [], [], pd.DataFrame()

        display_business_summary(selected_metric, int(selected_window), analysis_df, cp_summary_df, structure_summary_df)

        left, right = st.columns(2)
        with left:
            with st.container(border=True):
                st.subheader("内容 1：多变点检测")
                st.caption("识别所选累计销量斜率的历史突变月份。红色虚线为突变点，统计方法为带最小区间约束的动态规划多变点检测。")
                st.pyplot(changepoint_chart(analysis_df, cp_breaks))
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
                st.subheader("内容 2：EWOLS Bai-Perron 结构变化")
                st.caption(
                    "纵轴为 anchor 预测误差%。分段线使用 EWOLS 拟合，BIC 选择 0-2 个结构断点，用于判断斜率和预测误差关系是否换挡。"
                )
                if structure_df.empty or not structural_fits:
                    st.write("anchor 预测误差样本不足，暂不能做结构变化分析。")
                else:
                    st.pyplot(structural_chart(structure_df, structural_breaks, structural_fits))
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
