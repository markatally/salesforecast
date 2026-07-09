"""
Interactive statistical dashboard for pre-modeling correlation analysis.

Run:
    streamlit run code/30d-jenny/streamlit/corr.py
"""

from __future__ import annotations

import warnings
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
    out["year"] = out["bizym"] // 100
    out["month"] = out["bizym"] % 100
    out["day_of_week"] = out["transdate"].dt.dayofweek
    out["is_weekend"] = out["day_of_week"].ge(5)
    out["is_workday"] = out["transdate"].map(is_business_workday)
    out["is_weekday_holiday"] = (~out["is_workday"]) & (~out["is_weekend"])
    out["is_weekend_rest_day"] = (~out["is_workday"]) & out["is_weekend"]
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
        }
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            statistic, p_value, _, critical_values = kpss(clean, regression="c", nlags="auto")
    except Exception as exc:
        return {
            "method": method,
            "statistic": np.nan,
            "critical_values": {},
            "p_value": np.nan,
            "result": "检验失败",
            "detail": f"KPSS 检验失败：{exc}",
            "is_stationary": None,
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
    if result_label == "平稳":
        return f"{method} p-value={fmt_pvalue(p_value)}，alpha={alpha:.2f}，当前{target_label}更像围绕稳定水平波动，业务上可优先关注短期扰动和季节结构。"
    if result_label == "非平稳":
        if value_type == "统计值":
            return f"{method} p-value={fmt_pvalue(p_value)}，alpha={alpha:.2f}，当前{target_label}存在趋势/结构变化迹象，建模前建议考虑差分、去趋势或分段口径。"
        return f"{method} p-value={fmt_pvalue(p_value)}，alpha={alpha:.2f}，当前{target_label}仍有持续性变化迹象，说明增长率口径也可能受结构性变化影响。"
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
                    width=360,
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
                    width=340,
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
                    width=500,
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
            width=130,
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
                    width=440,
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
                        width=760,
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

tab_workday_avg = st.tabs(["工作日平均销量"])[0]
with tab_workday_avg:
    render_workday_avg_sales_tab(neighbor_panel)
