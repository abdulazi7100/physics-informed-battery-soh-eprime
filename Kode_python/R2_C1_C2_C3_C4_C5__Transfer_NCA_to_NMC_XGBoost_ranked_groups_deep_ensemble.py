"""Reviewer-safe five-member deep ensemble for NCA -> NMC.

This is a new experiment; it does not alter the earlier ranked-groups script or
results. Correlation--SHAP ranking, clipping, model fitting, and uncertainty
calibration are repeated without the held-out target cell in every outer LOCO
fold. Gated transfer is deliberately absent.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import R2_C1_C2_C3_C4_C5__Dependency_Transfer_NCA_to_NMC_XGBoost_ranked_groups as legacy
import R2_C1_C2_C3_C4_C5__Dependency_XGBoost_Common_Utilities as core


SEED = 42
N_ENSEMBLE = 5
FEATURE_COUNTS = list(range(4, 16))
ROUNDS = 40
SOURCE_ROUNDS = 20
TARGET_ROUNDS = 20
Z95 = 1.959963984540054
VARIANCE_FLOOR = 1e-10
ROOT = Path(__file__).resolve().parent
SOURCE_CSV = ROOT / "NCA_Integral_XGBoost_38.csv"
TARGET_CSV = ROOT / "NMC_Integral_XGBoost.csv"
OUTPUT_DIR = ROOT / "R2_C1_C2_C3_C4_C5__results_NCA_to_NMC_deep_ensemble"
FEATURES = legacy.FEATURES.copy()
SETUPS = ["NMC_target_only", "NCA_source_only_matched",
          "NCA_NMC_naive_pooled", "NCA_to_NMC_raw_continuation"]
DIRECTION_TITLE = "NCA to NMC"


def ensemble_statistics(y: np.ndarray, test_members: np.ndarray,
                        cal_y: np.ndarray, cal_members: np.ndarray):
    """Calibrate predictive variance without consulting held-out test labels."""
    mean = test_members.mean(axis=1)
    cal_mean = cal_members.mean(axis=1)
    epistemic = test_members.var(axis=1, ddof=1)
    cal_epistemic = cal_members.var(axis=1, ddof=1)
    calibration_mse = float(np.mean((cal_y - cal_mean) ** 2))
    # Residual MSE contains model disagreement; subtract its calibration average
    # to avoid counting that component twice in total predictive variance.
    residual_noise = float(max(
        calibration_mse - float(cal_epistemic.mean()), VARIANCE_FLOOR))
    total_variance = np.maximum(epistemic + residual_noise, VARIANCE_FLOOR)
    sd = np.sqrt(total_variance)
    low, high = mean - Z95 * sd, mean + Z95 * sd
    nll = .5 * (np.log(2 * np.pi * total_variance)
                + (y - mean) ** 2 / total_variance)
    covered = (y >= low) & (y <= high)
    metrics = {
        "rmse": float(mean_squared_error(y, mean) ** .5),
        "mae": float(mean_absolute_error(y, mean)),
        "r2": float(r2_score(y, mean)),
        "mean_gaussian_nll": float(nll.mean()),
        "mean_epistemic_variance": float(epistemic.mean()),
        "calibration_mse": calibration_mse,
        "residual_noise_variance": residual_noise,
        "mean_total_predictive_variance": float(total_variance.mean()),
        "picp_95": float(covered.mean()),
        "coverage_error_95": float(abs(covered.mean() - .95)),
        "mpiw_95": float((high - low).mean()),
    }
    arrays = {"predicted_soh": mean, "epistemic_variance": epistemic,
              "total_predictive_variance": total_variance,
              "lower_95": low, "upper_95": high,
              "covered_95": covered.astype(int), "gaussian_nll": nll}
    return metrics, arrays


def train_members(xs, xf, xc, xt, xc_source, xt_source, source_y,
                  fit_y, pooled_x, pooled_y, seed_base):
    cal = {name: [] for name in SETUPS}
    test = {name: [] for name in SETUPS}
    for member in range(N_ENSEMBLE):
        seed = seed_base + member
        target_model = legacy.model(seed, ROUNDS).fit(xf, fit_y)
        source_model = legacy.model(seed + 1000, ROUNDS).fit(xs, source_y)
        pooled_model = legacy.model(seed + 2000, ROUNDS).fit(pooled_x, pooled_y)
        pretrained = legacy.model(seed + 3000, SOURCE_ROUNDS).fit(xs, source_y)
        continued = legacy.model(seed + 4000, TARGET_ROUNDS).fit(
            xf, fit_y, xgb_model=pretrained.get_booster())
        models = [target_model, source_model, pooled_model, continued]
        cal_x = [xc, xc_source, xc, xc]
        test_x = [xt, xt_source, xt, xt]
        for setup, fitted, cx, tx in zip(SETUPS, models, cal_x, test_x):
            cal[setup].append(fitted.predict(cx))
            test[setup].append(fitted.predict(tx))
    return ({k: np.column_stack(v) for k, v in cal.items()},
            {k: np.column_stack(v) for k, v in test.items()})


def paired_tests(folds: pd.DataFrame) -> pd.DataFrame:
    output = []
    for k in FEATURE_COUNTS:
        wide = folds[folds.n_features == k].pivot(
            index="test_cell", columns="setup", values="rmse")
        transfer = wide[SETUPS[3]].to_numpy()
        for j, comparator in enumerate([SETUPS[0], SETUPS[2]]):
            baseline = wide[comparator].to_numpy()
            difference = transfer - baseline
            rng = np.random.default_rng(SEED + 100 * j + k)
            bootstrap = difference[
                rng.integers(0, len(difference), (5000, len(difference)))].mean(1)
            try:
                test = wilcoxon(transfer, baseline)
                statistic, p_value = float(test.statistic), float(test.pvalue)
            except ValueError:
                statistic, p_value = np.nan, np.nan
            output.append({
                "n_features": k, "transfer_setup": SETUPS[3],
                "comparator": comparator,
                "transfer_rmse": transfer.mean(),
                "comparator_rmse": baseline.mean(),
                "paired_rmse_difference": difference.mean(),
                "bootstrap_ci95_low": np.quantile(bootstrap, .025),
                "bootstrap_ci95_high": np.quantile(bootstrap, .975),
                "cells_transfer_improved": int((transfer < baseline).sum()),
                "n_test_cells": len(transfer),
                "wilcoxon_statistic": statistic, "wilcoxon_p": p_value})
    return pd.DataFrame(output)


def plot_results(summary: pd.DataFrame, rankings: pd.DataFrame) -> None:
    labels = ["Target-only", "Source-only matched", "Naive pooled", "Raw continuation"]
    colors = ["#2878B5", "#777777", "#F39C34", "#D9534F"]
    specs = [("r2_mean", r"Mean cell-wise $R^2$", "Figure_R2"),
             ("rmse_mean", "Mean cell-wise RMSE", "Figure_RMSE"),
             ("picp_95_mean", "Mean 95% PICP", "Figure_PICP"),
             ("mean_gaussian_nll_mean", "Mean Gaussian NLL", "Figure_NLL")]
    for metric, ylabel, stem in specs:
        fig, ax = legacy.plt.subplots(figsize=(10.5, 6))
        for setup, label, color in zip(SETUPS, labels, colors):
            z = summary[summary.setup == setup].sort_values("n_features")
            ax.plot(z.n_features, z[metric], marker="o", label=label, color=color)
        if metric == "r2_mean":
            ax.axhline(.9, color="black", linestyle="--", linewidth=1,
                       label=r"Reference $R^2=0.90$")
        if metric == "picp_95_mean":
            ax.axhline(.95, color="black", linestyle="--", linewidth=1,
                       label="Nominal coverage = 0.95")
        ax.set(xlabel="Number of fold-ranked features", ylabel=ylabel,
               xticks=FEATURE_COUNTS,
               title=f"{DIRECTION_TITLE}: leakage-safe five-member deep ensemble")
        ax.grid(alpha=.25); ax.legend(frameon=False, ncol=2); fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
        fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight")
        legacy.plt.close(fig)

    consensus = (rankings.groupby("feature")
                  .agg(mean_fusion_rank=("fusion_rank", "mean"),
                       std_fusion_rank=("fusion_rank", "std"),
                       mean_abs_target_correlation=("abs_target_correlation", "mean"),
                       mean_abs_shap=("mean_abs_shap", "mean"))
                  .sort_values("mean_fusion_rank").reset_index())
    consensus.to_csv(OUTPUT_DIR / "consensus_feature_ranking.csv", index=False)
    fig, ax = legacy.plt.subplots(figsize=(9, 6))
    positions = np.arange(len(consensus))
    ax.barh(positions, consensus.mean_fusion_rank,
            xerr=consensus.std_fusion_rank.fillna(0), color="#4C78A8", capsize=3)
    ax.set_yticks(positions, consensus.feature); ax.invert_yaxis()
    ax.set(xlabel="Mean fusion rank across LOCO folds (lower is better)",
           title="Fold-local feature-ranking stability")
    ax.grid(axis="x", alpha=.25); fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "Figure_feature_ranking.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "Figure_feature_ranking.pdf", bbox_inches="tight")
    legacy.plt.close(fig)


def main() -> None:
    started = time.time(); OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source = legacy.prepare(pd.read_csv(SOURCE_CSV))
    target = legacy.prepare(pd.read_csv(TARGET_CSV))
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
        ranking = legacy.fusion_ranking(fit, SEED + int(test_cell))
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
            cal_members, test_members = train_members(
                xs, xf, xc, xt, xc_source, xt_source, source.soh, fit.soh,
                pooled_x, pooled_y,
                SEED + int(test_cell) * 10_000 + k * 100)

            for setup in SETUPS:
                stats, arrays = ensemble_statistics(
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
        print(f"{fold_number:02d}/{len(cells)} held-out NMC cell={int(test_cell):02d}", flush=True)

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
    paired_tests(folds).to_csv(
        OUTPUT_DIR / "paired_transfer_comparison_by_feature_count.csv", index=False)
    pd.DataFrame(times).to_csv(OUTPUT_DIR / "training_times.csv", index=False)
    plot_results(summary, rankings)
    legacy.matched_condition_sensitivity(source, target).to_csv(
        OUTPUT_DIR / "matched_condition_complexity.csv", index=False)

    metadata = {
        "direction": "NCA_to_NMC", "experiment": "five-member deep ensemble",
        "feature_counts": FEATURE_COUNTS, "candidate_features": FEATURES,
        "excluded_model_inputs": ["Cycle", "Capacity", "soh", "file_number"],
        "outer_split": "55-fold target-cell LOCO",
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
        "reviewer_comment_3_literature_discussion": "Must be addressed in manuscript text; it is not an executable-code operation.",
        "comment_4_scope": "Matched-condition sensitivity controls temperature, cell count, and rows per cell; cycling rate and cell design remain limitations.",
        "elapsed_seconds": time.time() - started}
    (OUTPUT_DIR / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
