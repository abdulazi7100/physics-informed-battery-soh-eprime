"""Reviewer-safe five-member deep ensemble for NMC -> NCA.

This new experiment preserves all previous scripts and result folders. It uses
the same audited deep-ensemble implementation as NCA -> NMC, while reversing
source and target chemistry. Gated transfer is deliberately absent.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import R2_C1_C2_C3_C4_C5__Transfer_NCA_to_NMC_XGBoost_ranked_groups_deep_ensemble as deep
import R2_C1_C2_C3_C4_C5__Dependency_Transfer_NMC_to_NCA_XGBoost_ranked_groups as direction
import R2_C1_C2_C3_C4_C5__Dependency_XGBoost_Common_Utilities as core


SEED = deep.SEED
N_ENSEMBLE = deep.N_ENSEMBLE
FEATURE_COUNTS = deep.FEATURE_COUNTS
ROUNDS = deep.ROUNDS
SOURCE_ROUNDS = deep.SOURCE_ROUNDS
TARGET_ROUNDS = deep.TARGET_ROUNDS
ROOT = Path(__file__).resolve().parent
SOURCE_CSV = ROOT / "NMC_Integral_XGBoost.csv"
TARGET_CSV = ROOT / "NCA_Integral_XGBoost_38.csv"
OUTPUT_DIR = ROOT / "R2_C1_C2_C3_C4_C5__results_NMC_to_NCA_deep_ensemble"
FEATURES = direction.FEATURES.copy()
SETUPS = ["NCA_target_only", "NMC_source_only_matched",
          "NMC_NCA_naive_pooled", "NMC_to_NCA_raw_continuation"]


def configure_shared_implementation() -> None:
    """Set labels/output used by direction-independent ensemble helpers."""
    deep.SETUPS = SETUPS
    deep.OUTPUT_DIR = OUTPUT_DIR
    deep.DIRECTION_TITLE = "NMC to NCA"


def main() -> None:
    configure_shared_implementation()
    started = time.time(); OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source = direction.prepare(pd.read_csv(SOURCE_CSV))
    target = direction.prepare(pd.read_csv(TARGET_CSV))
    fold_rows, prediction_rows, rank_frames, times = [], [], [], []
    cells = sorted(target.file_number.unique())

    for fold_number, test_cell in enumerate(cells, 1):
        fold_started = time.time()
        outer_train = target[target.file_number != test_cell].copy()
        test = target[target.file_number == test_cell].copy()
        fit_cells, calibration_cells = core.split_fit_calibration(
            outer_train, SEED + int(test_cell))
        fit = outer_train[outer_train.file_number.isin(fit_cells)].copy()
        calibration = outer_train[
            outer_train.file_number.isin(calibration_cells)].copy()
        ranking = direction.fusion_ranking(fit, SEED + int(test_cell))
        ranking.insert(0, "test_cell", test_cell); rank_frames.append(ranking)
        ordered = ranking.feature.tolist()

        for k in FEATURE_COUNTS:
            group_started = time.time(); selected = ordered[:k]
            target_lo, target_hi = core.limits(fit, selected)
            source_lo, source_hi = core.limits(source, selected)
            xf, xc, xt = (core.clipped(frame, selected, target_lo, target_hi)
                          for frame in (fit, calibration, test))
            xs = core.clipped(source, selected, source_lo, source_hi)
            xc_source, xt_source = (
                core.clipped(frame, selected, source_lo, source_hi)
                for frame in (calibration, test))
            pooled_x = pd.concat([xs, xf], ignore_index=True)
            pooled_y = pd.concat([source.soh, fit.soh], ignore_index=True)
            cal_members, test_members = deep.train_members(
                xs, xf, xc, xt, xc_source, xt_source, source.soh, fit.soh,
                pooled_x, pooled_y,
                SEED + int(test_cell) * 10_000 + k * 100)

            for setup in SETUPS:
                stats, arrays = deep.ensemble_statistics(
                    test.soh.to_numpy(), test_members[setup],
                    calibration.soh.to_numpy(), cal_members[setup])
                fold_rows.append({
                    "test_cell": test_cell, "n_features": k, "setup": setup,
                    "features": "|".join(selected),
                    "n_fit_cells": len(fit_cells),
                    "n_calibration_cells": len(calibration_cells),
                    "n_ensemble": N_ENSEMBLE, **stats})
                for index, cycle, actual, predicted, epi, total, low, high, covered, nll in zip(
                        test.index, test.Cycle, test.soh, arrays["predicted_soh"],
                        arrays["epistemic_variance"], arrays["total_predictive_variance"],
                        arrays["lower_95"], arrays["upper_95"],
                        arrays["covered_95"], arrays["gaussian_nll"]):
                    prediction_rows.append({
                        "row_index": int(index), "test_cell": test_cell,
                        "Cycle": cycle, "n_features": k, "setup": setup,
                        "actual_soh": actual, "predicted_soh": predicted,
                        "epistemic_variance": epi,
                        "residual_noise_variance": stats["residual_noise_variance"],
                        "total_predictive_variance": total,
                        "lower_95": low, "upper_95": high,
                        "covered_95": covered, "gaussian_nll": nll})
            times.append({"test_cell": test_cell, "n_features": k,
                          "seconds_all_methods": time.time() - group_started})
        times.append({"test_cell": test_cell, "n_features": "ALL",
                      "seconds_all_methods": time.time() - fold_started})
        print(f"{fold_number:02d}/{len(cells)} held-out NCA cell={int(test_cell):02d}", flush=True)

    folds = pd.DataFrame(fold_rows)
    predictions = pd.DataFrame(prediction_rows)
    rankings = pd.concat(rank_frames, ignore_index=True)
    metric_columns = ["rmse", "mae", "r2", "mean_gaussian_nll",
                      "mean_epistemic_variance", "calibration_mse",
                      "residual_noise_variance", "mean_total_predictive_variance",
                      "picp_95", "coverage_error_95", "mpiw_95"]
    summary = folds.groupby(["n_features", "setup"])[metric_columns].agg(["mean", "std"])
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
    pd.DataFrame(times).to_csv(OUTPUT_DIR / "training_times.csv", index=False)
    deep.plot_results(summary, rankings)
    direction.matched_condition_sensitivity(source, target).to_csv(
        OUTPUT_DIR / "matched_condition_complexity.csv", index=False)

    metadata = {
        "direction": "NMC_to_NCA", "experiment": "five-member deep ensemble",
        "feature_counts": FEATURE_COUNTS, "candidate_features": FEATURES,
        "excluded_model_inputs": ["Cycle", "Capacity", "soh", "file_number"],
        "outer_split": "65-fold NCA target-cell LOCO",
        "ranking": "recomputed per outer fold on target fitting cells only; mean of absolute target-correlation and mean-|SHAP| ranks",
        "hard_correlation_filter_threshold": None,
        "hard_correlation_filter_note": "No hard removal; correlation is one explicit component of the fusion rank.",
        "target_partition": "complete fitting cells, complete calibration cells, one held-out test cell",
        "source_only_label_boundary": "No target SOH labels fit source-model parameters; fold-specific target-fitting labels select the matched feature subset shared by every method.",
        "preprocessing": "non-temperature indicators divided by per-cell median of first five unlabelled observations; clipping at 0.1%/99.9% training-domain quantiles",
        "ensemble_members": N_ENSEMBLE,
        "uncertainty": {
            "epistemic": "sample variance across five independently seeded predictions",
            "residual_noise": "max(calibration MSE - mean calibration epistemic variance, 1e-10)",
            "total": "epistemic variance + calibration residual-noise variance",
            "test_labels": "used only after prediction for RMSE, MAE, R2, Gaussian mean NLL, PICP, and MPIW",
            "interval": "mean prediction +/- 1.959963984540054 * sqrt(total variance)"},
        "matched_methods": SETUPS, "gated_transfer_used": False,
        "tree_budget": {"target_only": ROUNDS, "source_only": ROUNDS,
                        "naive_pooled": ROUNDS,
                        "raw_continuation_source": SOURCE_ROUNDS,
                        "raw_continuation_target": TARGET_ROUNDS,
                        "raw_continuation_total": SOURCE_ROUNDS + TARGET_ROUNDS},
        "xgboost": {"learning_rate": .10, "max_depth": 4,
                    "min_child_weight": 3.0, "reg_lambda": 5.0,
                    "reg_alpha": 0.0, "subsample": .9,
                    "colsample_bytree": .9, "objective": "reg:squarederror",
                    "eval_metric": "rmse", "tree_method": "hist"},
        "random_seed_base": SEED,
        "reviewer_comment_3_literature_discussion": "Addressed separately in Reviewer_2_Response_Revised_v3.md and manuscript text.",
        "comment_4_scope": "Matched-condition sensitivity controls temperature, cell count, and rows per cell; cycling rate and cell design remain limitations.",
        "elapsed_seconds": time.time() - started}
    (OUTPUT_DIR / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
