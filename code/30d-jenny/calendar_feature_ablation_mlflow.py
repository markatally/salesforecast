from __future__ import annotations

import argparse
from dataclasses import replace
import json
import sys
import tempfile
import types
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance


NOTEBOOK_PATH = Path("code/30d-jenny/03-experiment-20260701.ipynb")
EXPERIMENT_NAME = "direct_monthly_calendar_feature_ablation"

CALENDAR_FEATURES = [
    "target_month_workdays",
    "target_month_nonworkdays",
    "target_month_weekend_days",
    "target_month_lunar_holiday_days",
    "target_month_spring_festival_core_days",
    "target_month_spring_festival_days",
    "target_month_spring_festival_spans_adjacent_month",
]

MODEL_NAME = "hist_gradient_boosting"
MAX_PARAM_CANDIDATES = 3


def _noop_display(*_: Any, **__: Any) -> None:
    return None


def load_notebook_namespace(notebook_path: Path) -> dict[str, Any]:
    """Execute the feature-engineering cells from the experiment notebook."""
    nb = json.loads(notebook_path.read_text())
    module_name = "__calendar_ablation__"
    module = types.ModuleType(module_name)
    sys.modules[module_name] = module
    ns = module.__dict__
    ns.update({"display": _noop_display, "__name__": module_name})

    for idx in range(5):
        exec("".join(nb["cells"][idx]["source"]), ns)

    direct_monthly_src = "".join(nb["cells"][5]["source"])
    prefix = direct_monthly_src.split("direct_monthly_panel = build_direct_monthly_panel", 1)[0]
    exec(prefix, ns)
    return ns


def add_target_month_calendar_features(direct_panel: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.sort_values(["bizym", "transdate"]).copy()
    d["target_month_weekend_days_flag"] = d["day_of_week"].isin([5, 6]).astype(int)
    d["target_month_nonworkdays_flag"] = (~d["is_workday"].astype(bool)).astype(int)
    d["target_month_spring_festival_days_flag"] = d["lunar_holiday"].eq("春节").astype(int)
    d["target_month_spring_festival_core_days_flag"] = d["lunar_holiday"].isin(["除夕", "春节"]).astype(int)

    summary = (
        d.groupby("bizym", as_index=False)
        .agg(
            target_month_workdays=("is_workday", "sum"),
            target_month_nonworkdays=("target_month_nonworkdays_flag", "sum"),
            target_month_weekend_days=("target_month_weekend_days_flag", "sum"),
            target_month_lunar_holiday_days=("is_lunar_holiday", "sum"),
            target_month_spring_festival_core_days=("target_month_spring_festival_core_days_flag", "sum"),
            target_month_spring_festival_days=("target_month_spring_festival_days_flag", "sum"),
        )
        .sort_values("bizym")
        .reset_index(drop=True)
    )

    summary["target_month_spring_festival_spans_adjacent_month"] = 0
    spring_months = summary["target_month_spring_festival_core_days"].gt(0)
    prev_has_spring = summary["target_month_spring_festival_core_days"].shift(1, fill_value=0).gt(0)
    next_has_spring = summary["target_month_spring_festival_core_days"].shift(-1, fill_value=0).gt(0)
    adjacent_month = (
        summary["bizym"].diff().fillna(999).isin([1, 89])
        | summary["bizym"].diff(-1).abs().fillna(999).isin([1, 89])
    )
    summary.loc[spring_months & (prev_has_spring | next_has_spring) & adjacent_month, "target_month_spring_festival_spans_adjacent_month"] = 1

    return direct_panel.merge(summary, on="bizym", how="left")


def build_linear_feature_candidates(base_features: list[str]) -> list[str]:
    linear_candidates = [
        "month_num",
        "day_of_month",
        "days_in_month",
        "days_to_month_end",
        "workdays_to_month_end",
        "completed_workday_seq",
        "forecast_workday_seq",
        "mtd_qty",
        "mtd_num_hosp",
        "anchor_day_qty",
        "anchor_day_num_hosp",
        "qty_xmonth_lag_1d",
        "qty_xmonth_lag_7d",
        "qty_xmonth_lag_14d",
        "num_hosp_xmonth_lag_1d",
        "num_hosp_xmonth_lag_7d",
        "qty_xmonth_roll_7d_mean_through_anchor",
        "qty_xmonth_roll_14d_mean_through_anchor",
        "qty_xmonth_roll_30d_mean_through_anchor",
        "qty_xmonth_roll_30d_sum_through_anchor",
        "num_hosp_xmonth_roll_7d_mean_through_anchor",
        "num_hosp_xmonth_roll_14d_mean_through_anchor",
        "num_hosp_xmonth_roll_30d_mean_through_anchor",
        "qty_per_hosp_xmonth_roll_14d_mean_through_anchor",
        "qty_per_hosp_xmonth_roll_30d_mean_through_anchor",
        "anchor_month_progress",
        "anchor_workday_progress",
        "anchor_mtd_qty_per_workday",
        "anchor_mtd_num_hosp_per_workday",
        "anchor_mtd_qty_per_hosp",
        "anchor_mtd_vs_same_month_mean",
        "anchor_mtd_hosp_vs_same_month_mean",
        "same_month_total_qty_mean",
        "same_month_total_qty_std",
        "same_month_qty_cv",
        "prev_month_total_qty",
        "prev_month_mom_pct",
        "prev_month_yoy_pct",
        "month_total_roll_3m_lag1",
        "hist_anchor_mtd_share_expanding_mean",
        "hist_anchor_mtd_share_expanding_std",
        "hist_anchor_backtest_ape_mean",
        "expected_month_total_from_hist_mtd_share",
        "expected_remaining_qty_from_hist_mtd_share",
        "expected_month_total_blend_same_month",
        "log1p_mtd_qty",
        "log1p_same_month_total_qty_mean",
        "log1p_prev_month_total_qty",
        "log1p_month_total_roll_3m_lag1",
        "log1p_expected_month_total_from_hist_mtd_share",
        "log1p_expected_remaining_qty_from_hist_mtd_share",
        "log1p_expected_month_total_blend_same_month",
    ]
    return [col for col in linear_candidates if col in base_features]


def mape_pct(y_true: pd.Series, y_pred: pd.Series) -> float:
    mask = y_true.notna() & y_pred.notna() & y_true.gt(0)
    if not mask.any():
        return float("nan")
    return float(((y_pred[mask] - y_true[mask]).abs() / y_true[mask]).mean() * 100)


def fit_predict_variant(
    ns: dict[str, Any],
    model_name: str,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[dict[str, Any], pd.DataFrame]:
    spec = ns["MODEL_SPECS"][model_name]
    record = ns["fit_direct_monthly_model_with_search"](
        model_name=model_name,
        model_spec=spec,
        train_data=train_df,
        feature_columns=feature_columns,
        config=ns["CONFIG"],
    )
    pred = ns["predict_direct_month_total"](
        model=record["model"],
        df=eval_df,
        feature_columns=feature_columns,
    )
    pred["model_name"] = model_name
    return record, pred


def summarize_predictions(pred: pd.DataFrame, calendar_summary: pd.DataFrame) -> pd.DataFrame:
    out = pred.merge(calendar_summary[["bizym"] + CALENDAR_FEATURES], on="bizym", how="left")
    out["is_jan_feb"] = out["bizym"].astype(str).str[-2:].isin(["01", "02"])
    out["is_spring_festival_month"] = out["target_month_spring_festival_core_days"].gt(0)
    out["is_high_nonworkday_month"] = out["target_month_nonworkdays"].ge(11)
    out["is_early_wd1_5"] = out["forecast_workday_seq"].between(1, 5)
    out["is_early_wd1_10"] = out["forecast_workday_seq"].between(1, 10)
    return out


def metric_rows(variant: str, pred: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    slice_masks = {
        "overall": pd.Series(True, index=pred.index),
        "jan_feb": pred["is_jan_feb"],
        "spring_festival_month": pred["is_spring_festival_month"],
        "high_nonworkday_month": pred["is_high_nonworkday_month"],
        "wd1_5": pred["is_early_wd1_5"],
        "wd1_10": pred["is_early_wd1_10"],
        "jan_feb_wd1_10": pred["is_jan_feb"] & pred["is_early_wd1_10"],
        "spring_wd1_10": pred["is_spring_festival_month"] & pred["is_early_wd1_10"],
    }
    for split in ["valid", "test"]:
        split_mask = pred["split"].eq(split)
        for slice_name, slice_mask in slice_masks.items():
            mask = split_mask & slice_mask
            if not mask.any():
                continue
            rows.append(
                {
                    "variant": variant,
                    "split": split,
                    "slice": slice_name,
                    "n_predictions": int(mask.sum()),
                    "n_months": int(pred.loc[mask, "bizym"].nunique()),
                    "mean_mape_pct": float(pred.loc[mask, "month_total_mape_pct"].mean()),
                    "median_mape_pct": float(pred.loc[mask, "month_total_mape_pct"].median()),
                }
            )
    return rows


def permutation_importance_rows(
    model: Any,
    eval_df: pd.DataFrame,
    feature_columns: list[str],
    random_seed: int,
) -> pd.DataFrame:
    x = eval_df[feature_columns].replace([np.inf, -np.inf], np.nan).copy()
    y = pd.to_numeric(eval_df["actual_month_total"], errors="coerce")
    mask = y.notna() & y.gt(0)
    x = x.loc[mask]
    y = y.loc[mask]

    def neg_mape(estimator: Any, x_eval: pd.DataFrame, y_eval: pd.Series) -> float:
        raw = estimator.predict(x_eval)
        pred = np.maximum(raw, 0)
        pred = np.maximum(pred, eval_df.loc[x_eval.index, "mtd_qty"].to_numpy(dtype=float))
        return -mape_pct(y_eval, pd.Series(pred, index=x_eval.index))

    result = permutation_importance(
        model,
        x,
        y,
        scoring=neg_mape,
        n_repeats=20,
        random_state=random_seed,
    )
    rows = pd.DataFrame(
        {
            "feature": feature_columns,
            "mape_increase_pct_mean": result.importances_mean,
            "mape_increase_pct_std": result.importances_std,
        }
    )
    return rows[rows["feature"].isin(CALENDAR_FEATURES)].sort_values(
        "mape_increase_pct_mean",
        ascending=False,
    )


def log_table_artifact(df: pd.DataFrame, artifact_name: str) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / artifact_name
        df.to_csv(path, index=False)
        mlflow.log_artifact(str(path), artifact_path="tables")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri", default="http://127.0.0.1:5058/")
    parser.add_argument("--experiment-name", default=EXPERIMENT_NAME)
    parser.add_argument("--model-name", default=MODEL_NAME)
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    ns = load_notebook_namespace(NOTEBOOK_PATH)
    ns["CONFIG"] = replace(ns["CONFIG"], max_param_candidates_per_model=MAX_PARAM_CANDIDATES)
    direct_panel = ns["build_direct_monthly_panel"](ns["all_fe"], ns["daily_df"])
    enhanced_panel = add_target_month_calendar_features(direct_panel, ns["daily_df"])
    calendar_summary = enhanced_panel[["bizym"] + CALENDAR_FEATURES].drop_duplicates("bizym")

    base_features = [
        col for col in direct_panel.columns if ns["is_direct_monthly_feature"](col, direct_panel)
    ]
    all_features = base_features + CALENDAR_FEATURES
    variants = {"baseline": base_features, "calendar_all": all_features}
    variants.update({f"single__{feature}": base_features + [feature] for feature in CALENDAR_FEATURES})
    variants.update(
        {
            f"drop_one__{feature}": [col for col in all_features if col != feature]
            for feature in CALENDAR_FEATURES
        }
    )

    train_df = enhanced_panel[enhanced_panel["split"].eq("train")].copy()
    eval_df = enhanced_panel[enhanced_panel["split"].isin(["valid", "test"])].copy()

    metric_frames = []
    prediction_frames = []
    records: dict[str, dict[str, Any]] = {}

    with mlflow.start_run(run_name=f"{args.model_name}_calendar_feature_ablation") as run:
        mlflow.log_params(
            {
                "model_name": args.model_name,
                "train_ym_range": str(ns["CONFIG"].train_ym_range),
                "valid_ym_range": str(ns["CONFIG"].valid_ym_range),
                "test_ym_range": str(ns["CONFIG"].test_ym_range),
                "n_calendar_features": len(CALENDAR_FEATURES),
                "n_variants": len(variants),
                "max_param_candidates_per_model": ns["CONFIG"].max_param_candidates_per_model,
            }
        )

        for idx, (variant, feature_columns) in enumerate(variants.items(), start=1):
            print(f"[{idx}/{len(variants)}] fitting variant={variant} feature_count={len(feature_columns)}", flush=True)
            record, pred = fit_predict_variant(ns, args.model_name, train_df, eval_df, feature_columns)
            pred = summarize_predictions(pred, calendar_summary)
            pred["variant"] = variant
            records[variant] = record
            prediction_frames.append(pred)
            metric_frames.extend(metric_rows(variant, pred))
            mlflow.log_metric(f"{variant}.cv_mape_pct", float(record["cv_month_total_mape"]))
            mlflow.log_metric(f"{variant}.feature_count", len(feature_columns))

        metrics = pd.DataFrame(metric_frames)
        predictions = pd.concat(prediction_frames, ignore_index=True)

        baseline_metrics = metrics[metrics["variant"].eq("baseline")][
            ["split", "slice", "mean_mape_pct"]
        ].rename(columns={"mean_mape_pct": "baseline_mean_mape_pct"})
        contribution = metrics.merge(baseline_metrics, on=["split", "slice"], how="left")
        contribution["mape_improvement_pct_point"] = (
            contribution["baseline_mean_mape_pct"] - contribution["mean_mape_pct"]
        )

        all_record = records["calendar_all"]
        importance = permutation_importance_rows(
            model=all_record["model"],
            eval_df=eval_df,
            feature_columns=all_features,
            random_seed=ns["CONFIG"].random_seed,
        )

        for _, row in contribution[
            contribution["variant"].isin(["baseline", "calendar_all"])
            & contribution["slice"].isin(["overall", "jan_feb", "spring_festival_month", "wd1_10", "spring_wd1_10"])
        ].iterrows():
            metric_name = f"{row['variant']}.{row['split']}.{row['slice']}.mean_mape_pct"
            mlflow.log_metric(metric_name, float(row["mean_mape_pct"]))
            if row["variant"] == "calendar_all":
                mlflow.log_metric(
                    f"calendar_all.{row['split']}.{row['slice']}.mape_improvement_pct_point",
                    float(row["mape_improvement_pct_point"]),
                )

        log_table_artifact(metrics, "metrics_by_variant_slice.csv")
        log_table_artifact(contribution, "contribution_vs_baseline.csv")
        log_table_artifact(predictions, "predictions_by_variant.csv")
        log_table_artifact(calendar_summary, "target_month_calendar_summary.csv")
        log_table_artifact(importance, "calendar_permutation_importance.csv")

        print(f"MLflow run_id: {run.info.run_id}")
        print(f"MLflow experiment: {args.experiment_name}")
        print("\nCalendar all vs baseline:")
        cols = [
            "variant",
            "split",
            "slice",
            "n_predictions",
            "n_months",
            "mean_mape_pct",
            "baseline_mean_mape_pct",
            "mape_improvement_pct_point",
        ]
        print(
            contribution[
                contribution["variant"].eq("calendar_all")
                & contribution["slice"].isin(["overall", "jan_feb", "spring_festival_month", "wd1_10", "spring_wd1_10"])
            ][cols].to_string(index=False)
        )
        print("\nCalendar feature permutation importance:")
        print(importance.to_string(index=False))


if __name__ == "__main__":
    main()
