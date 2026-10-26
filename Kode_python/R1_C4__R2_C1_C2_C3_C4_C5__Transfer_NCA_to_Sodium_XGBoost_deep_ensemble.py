"""Leakage-safe five-member deep ensemble for NCA -> Wenzhou sodium.

This new experiment preserves earlier code/results and uses cycles 1--800 from
each of seven sodium cells. Ranking, clipping, fitting, and uncertainty
calibration exclude the held-out target cell. Gated transfer is absent.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import R2_C1_C2_C3_C4_C5__Transfer_NCA_to_NMC_XGBoost_ranked_groups_deep_ensemble as deep
import R1_C4__R2_C1_C2_C3_C4_C5__Dependency_Sodium_XGBoost_Common_Utilities as sodium


SEED = 42
N_ENSEMBLE = 5
FEATURE_COUNTS = [4, 5, 6, 7]
ROUNDS = 80
SOURCE_ROUNDS = 40
TARGET_ROUNDS = 40
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
SOURCE_CSV = ROOT / "NCA_Integral_XGBoost_38.csv"
TARGET_CSV = (
    PROJECT
    / "kode_feature_extraction_sodium_wenzhou"
    / "Wenzhou_Sodium_7_features.csv"
)
OUTPUT_DIR = ROOT / "R1_C4__R2_C1_C2_C3_C4_C5__results_NCA_to_Sodium_800cycles"
FEATURES = sodium.FEATURES.copy()
SETUPS = ["Sodium_target_only", "NCA_source_only_matched",
          "NCA_Sodium_naive_pooled", "NCA_to_Sodium_raw_continuation"]
SOURCE_NAME = "NCA"
DIRECTION_NAME = "NCA_to_Wenzhou_sodium"


def train_members(xs, xf, xc, xt, xc_source, xt_source, source_y,
                  fit_y, pooled_x, pooled_y, seed_base):
    cal = {name: [] for name in SETUPS}
    test = {name: [] for name in SETUPS}
    for member in range(N_ENSEMBLE):
        seed = seed_base + member
        target_model = sodium.make_model(seed, ROUNDS).fit(xf, fit_y)
        source_model = sodium.make_model(seed + 1000, ROUNDS).fit(xs, source_y)
        pooled_model = sodium.make_model(seed + 2000, ROUNDS).fit(pooled_x, pooled_y)
        pretrained = sodium.make_model(seed + 3000, SOURCE_ROUNDS).fit(xs, source_y)
        continued = sodium.make_model(seed + 4000, TARGET_ROUNDS).fit(
            xf, fit_y, xgb_model=pretrained.get_booster())
        models = [target_model, source_model, pooled_model, continued]
        cal_x = [xc, xc_source, xc, xc]
        test_x = [xt, xt_source, xt, xt]
        for setup, fitted, cx, tx in zip(SETUPS, models, cal_x, test_x):
            cal[setup].append(fitted.predict(cx))
            test[setup].append(fitted.predict(tx))
    return ({k: np.column_stack(v) for k, v in cal.items()},
            {k: np.column_stack(v) for k, v in test.items()})


def configure_shared_helpers() -> None:
    deep.SETUPS = SETUPS
    deep.FEATURE_COUNTS = FEATURE_COUNTS
    deep.OUTPUT_DIR = OUTPUT_DIR
    deep.DIRECTION_TITLE = f"{SOURCE_NAME} to sodium (cycles 1--800)"


def main() -> None:
    configure_shared_helpers()
    started = time.time(); OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source_raw = pd.read_csv(SOURCE_CSV).rename(columns={"soh": "SOH"})
    target_raw = pd.read_csv(TARGET_CSV)
    target_raw = target_raw[target_raw.Cycle.between(1, 800)].copy()
    counts = target_raw.groupby("file_number").size()
    if len(counts) != 7 or not (counts == 800).all():
        raise ValueError(f"Expected seven cells with exactly 800 rows: {counts.to_dict()}")
    source_raw["file_number"] = SOURCE_NAME + "_" + source_raw.file_number.astype(str)
    source = sodium.prepare(source_raw).rename(columns={"SOH": "soh"})
    target = sodium.prepare(target_raw).rename(columns={"SOH": "soh"})

    fold_rows, prediction_rows, rank_frames, timing_rows = [], [], [], []
    cells = sorted(target.file_number.unique())
    for fold_number, test_cell in enumerate(cells, 1):
        fold_started = time.time()
        remaining = [cell for cell in cells if cell != test_cell]
        fit_cells, calibration_cells = sodium.split_fit_calibration(
            remaining, sodium.stable_seed(test_cell))
        fit = target[target.file_number.isin(fit_cells)].copy()
        calibration = target[target.file_number.isin(calibration_cells)].copy()
        test = target[target.file_number == test_cell].copy()
        ranking = sodium.fusion_ranking(fit, sodium.stable_seed(test_cell, 1000))
        ranking.insert(0, "test_cell", test_cell); rank_frames.append(ranking)
        ordered = ranking.feature.tolist()

        for k in FEATURE_COUNTS:
            group_started = time.time(); selected = ordered[:k]
            target_lo, target_hi = sodium.limits(fit, selected)
            source_lo, source_hi = sodium.limits(source, selected)
            xf, xc, xt = (sodium.clipped(frame, selected, target_lo, target_hi)
                          for frame in (fit, calibration, test))
            xs = sodium.clipped(source, selected, source_lo, source_hi)
            xc_source, xt_source = (
                sodium.clipped(frame, selected, source_lo, source_hi)
                for frame in (calibration, test))
            pooled_x = pd.concat([xs, xf], ignore_index=True)
            pooled_y = pd.concat([source.soh, fit.soh], ignore_index=True)
            cal_members, test_members = train_members(
                xs, xf, xc, xt, xc_source, xt_source, source.soh, fit.soh,
                pooled_x, pooled_y, sodium.stable_seed(test_cell, k * 100))

            for setup in SETUPS:
                stats, arrays = deep.ensemble_statistics(
                    test.soh.to_numpy(), test_members[setup],
                    calibration.soh.to_numpy(), cal_members[setup])
                fold_rows.append({
                    "test_cell": test_cell, "n_features": k, "setup": setup,
                    "features": "|".join(selected), "n_fit_cells": 4,
                    "n_calibration_cells": 2, "n_ensemble": N_ENSEMBLE, **stats})
                for index, cycle, actual, predicted, epi, total, low, high, covered, nll in zip(
                        test.index, test.Cycle, test.soh, arrays["predicted_soh"],
                        arrays["epistemic_variance"], arrays["total_predictive_variance"],
                        arrays["lower_95"], arrays["upper_95"],
                        arrays["covered_95"], arrays["gaussian_nll"]):
                    prediction_rows.append({
                        "row_index": int(index), "test_cell": test_cell,
                        "Cycle": int(cycle), "n_features": k, "setup": setup,
                        "actual_soh": actual, "predicted_soh": predicted,
                        "epistemic_variance": epi,
                        "residual_noise_variance": stats["residual_noise_variance"],
                        "total_predictive_variance": total,
                        "lower_95": low, "upper_95": high,
                        "covered_95": covered, "gaussian_nll": nll})
            timing_rows.append({"test_cell": test_cell, "n_features": k,
                                "seconds_all_methods": time.time() - group_started})
        timing_rows.append({"test_cell": test_cell, "n_features": "ALL",
                            "seconds_all_methods": time.time() - fold_started})
        print(f"{fold_number}/7 held-out sodium cell: {test_cell}", flush=True)

    folds = pd.DataFrame(fold_rows)
    predictions = pd.DataFrame(prediction_rows)
    rankings = pd.concat(rank_frames, ignore_index=True)
    metrics = ["rmse", "mae", "r2", "mean_gaussian_nll",
               "mean_epistemic_variance", "calibration_mse",
               "residual_noise_variance", "mean_total_predictive_variance",
               "picp_95", "coverage_error_95", "mpiw_95"]
    summary = folds.groupby(["n_features", "setup"])[metrics].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    pooled = []
    for (k, setup), block in predictions.groupby(["n_features", "setup"]):
        pooled.append({"n_features": k, "setup": setup,
                       "oof_rmse": mean_squared_error(
                           block.actual_soh, block.predicted_soh) ** .5,
                       "oof_mae": mean_absolute_error(
                           block.actual_soh, block.predicted_soh),
                       "oof_r2": r2_score(block.actual_soh, block.predicted_soh),
                       "oof_mean_gaussian_nll": block.gaussian_nll.mean(),
                       "oof_picp_95": block.covered_95.mean(),
                       "oof_mpiw_95": (block.upper_95 - block.lower_95).mean()})
    pooled = pd.DataFrame(pooled)

    folds.to_csv(OUTPUT_DIR / "fold_metrics_all_feature_groups.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "out_of_fold_predictions_with_intervals.csv", index=False)
    rankings.to_csv(OUTPUT_DIR / "nested_feature_rankings.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "summary_metrics_by_feature_count.csv", index=False)
    pooled.to_csv(OUTPUT_DIR / "pooled_oof_metrics_by_feature_count.csv", index=False)
    deep.paired_tests(folds).to_csv(
        OUTPUT_DIR / "paired_transfer_comparison_by_feature_count.csv", index=False)
    pd.DataFrame(timing_rows).to_csv(OUTPUT_DIR / "training_times.csv", index=False)
    deep.plot_results(summary, rankings)
    target_raw.groupby("file_number").agg(
        first_cycle=("Cycle", "min"), last_cycle=("Cycle", "max"),
        observations=("Cycle", "size"), soh_min=("SOH", "min"),
        soh_max=("SOH", "max")).to_csv(OUTPUT_DIR / "target_dataset_audit.csv")

    metadata = {
        "direction": DIRECTION_NAME, "source_chemistry": SOURCE_NAME,
        "experiment": "five-member deep ensemble",
        "target_cells": 7, "target_cycles_per_cell": 800,
        "target_observations": 5600, "feature_counts": FEATURE_COUNTS,
        "candidate_features": FEATURES,
        "excluded_model_inputs": ["Cycle", "Charge_Capacity_mAh",
            "Discharge_Capacity_mAh", "SOH", "file_number", "source_file"],
        "outer_split": "7-fold sodium target-cell LOCO",
        "target_partition_per_fold": "4 fitting cells, 2 calibration cells, 1 test cell",
        "ranking": "fold-local target-fitting-cell correlation--SHAP fusion rank",
        "hard_correlation_filter_threshold": None,
        "source_only_label_boundary": f"No sodium labels fit {SOURCE_NAME} source-model parameters; sodium fitting labels select the matched feature subset shared by all methods.",
        "preprocessing": "cell-relative median of first five unlabelled observations; fold-training 0.1%/99.9% clipping",
        "ensemble_members": N_ENSEMBLE,
        "uncertainty": {
            "epistemic": "sample variance across five members",
            "residual_noise": "max(calibration MSE - mean calibration epistemic variance, 1e-10)",
            "total": "epistemic + calibration residual-noise variance",
            "test_labels": "evaluation only: RMSE, MAE, R2, Gaussian mean NLL, PICP, MPIW"},
        "matched_methods": SETUPS, "gated_transfer_used": False,
        "tree_budget": {"target_only": ROUNDS, "source_only": ROUNDS,
            "naive_pooled": ROUNDS, "source_pretraining": SOURCE_ROUNDS,
            "target_continuation": TARGET_ROUNDS,
            "raw_continuation_total": SOURCE_ROUNDS + TARGET_ROUNDS},
        "xgboost": {"learning_rate": .08, "max_depth": 4,
            "min_child_weight": 3.0, "reg_lambda": 5.0, "reg_alpha": 0.0,
            "subsample": .9, "colsample_bytree": .9,
            "objective": "reg:squarederror", "tree_method": "hist"},
        "random_seed_base": SEED,
        "comment_4_scope": "No mechanistic complexity claim; sodium temperature is unavailable and chemistry, cell design, and cycling protocols are not jointly controlled.",
        "comment_3_literature": "Shang et al. is discussed in Reviewer_2_Response_Revised_v4.md; numerical comparison is not matched.",
        "elapsed_seconds": time.time() - started}
    (OUTPUT_DIR / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary.to_string(index=False)); print(f"Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
