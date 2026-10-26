"""Leakage-safe five-member deep-ensemble NCA -> Wenzhou Multiorbits.

Primary target: regime_normalised_SOH. Sensitivity target: raw_SOH.
Nested target-cell LOCO recomputes correlation--SHAP ranking on fitting cells.
Calibration cells are disjoint from fitting and held-out test cells. Gated
transfer is deliberately absent.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import wilcoxon
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SEED = 42
N_ENSEMBLE = 5
FEATURE_COUNTS = list(range(4, 15))
ROUNDS = 40
SOURCE_ROUNDS = 20
TARGET_ROUNDS = 20
Z95 = 1.959963984540054
VARIANCE_FLOOR = 1e-10
SOURCE_CSV = ROOT / "NCA_Integral_XGBoost_38.csv"
TARGET_CSV = (PROJECT_ROOT / "kode_multiorbits_wenzhou" / "extracted_features_14" /
              "Wenzhou_Multiorbits_14_features.csv")
OUTPUT_DIR = ROOT / "R1_C2__R2_C1_C2_C3_C4_C5__results_NCA_to_Multiorbits"
SOURCE_NAME = "NCA"
DIRECTION_NAME = "NCA_to_Wenzhou_Multiorbits"
DIRECTION_TITLE = "NCA to Multiorbits"
FEATURES = [
    "CVCT", "CCCT", "TCT", "AVCC", "ACCV", "TES", "IVCC", "IVCCCV",
    "RMSVCC", "SkewVCC", "KurtVCC", "RMSCCV", "SkewCCV", "KurtCCV",
]
SETUPS = [
    "Multiorbits_target_only", "NCA_source_only_matched",
    "NCA_Multiorbits_naive_pooled", "NCA_to_Multiorbits_raw_continuation",
]
SENSITIVITY_SETUPS = ["Multiorbits_target_only", "NCA_to_Multiorbits_raw_continuation"]
PARAMS = dict(max_depth=4, min_child_weight=3.0, reg_lambda=5.0, reg_alpha=0.0)


def prepare(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    """Normalize indicators to five early unlabelled records of each cell."""
    out = frame.sort_values(["file_number", "Cycle"]).copy()
    out["soh"] = out[target].astype(float)
    reference = (out.groupby("file_number", sort=False).head(5)
                 .groupby("file_number")[FEATURES].median())
    for feature in FEATURES:
        ref = out.file_number.map(reference[feature]).astype(float)
        value = out[feature].astype(float)
        out[feature] = np.where(ref.abs() > 1e-12, value / ref, value - ref)
    return out.reset_index(drop=True)


def split_fit_calibration(outer_train: pd.DataFrame, seed: int):
    """Select one calibration cell per protocol group where possible."""
    meta = outer_train.groupby("file_number", as_index=False).protocol_group.first()
    rng = np.random.default_rng(seed)
    calibration = []
    for _, block in meta.groupby("protocol_group", sort=True):
        candidates = block.file_number.to_numpy(copy=True)
        rng.shuffle(candidates)
        calibration.append(candidates[0])
    calibration = sorted(set(calibration))
    fit = sorted(set(meta.file_number) - set(calibration))
    return fit, calibration


def limits(frame, features):
    return frame[features].quantile(.001), frame[features].quantile(.999)


def clipped(frame, features, low, high):
    return frame[features].clip(lower=low, upper=high, axis=1)


def model(seed, rounds):
    return XGBRegressor(
        n_estimators=rounds, learning_rate=.10, subsample=.90,
        colsample_bytree=.90, objective="reg:squarederror", eval_metric="rmse",
        tree_method="hist", n_jobs=-1, random_state=seed, verbosity=0, **PARAMS,
    )


def fusion_ranking(fit, seed):
    """Compute complete feature ranking using target fitting cells only."""
    low, high = limits(fit, FEATURES)
    x = clipped(fit, FEATURES, low, high)
    with np.errstate(invalid="ignore", divide="ignore"):
        correlation = x.corrwith(fit.soh).abs().fillna(0.0)
    selector = model(seed, 40).fit(x, fit.soh)
    sample = x.sample(min(5000, len(x)), random_state=seed)
    contributions = selector.get_booster().predict(
        xgb.DMatrix(sample, feature_names=FEATURES), pred_contribs=True)
    shap = pd.Series(np.abs(contributions[:, :-1]).mean(0), index=FEATURES)
    result = pd.DataFrame({
        "feature": FEATURES,
        "abs_target_correlation": correlation[FEATURES].to_numpy(),
        "mean_abs_shap": shap[FEATURES].to_numpy(),
    })
    result["correlation_rank"] = result.abs_target_correlation.rank(
        ascending=False, method="average")
    result["shap_rank"] = result.mean_abs_shap.rank(ascending=False, method="average")
    result["fusion_rank"] = (result.correlation_rank + result.shap_rank) / 2
    return result.sort_values(
        ["fusion_rank", "shap_rank", "correlation_rank", "feature"]
    ).reset_index(drop=True)


def ensemble_statistics(y, test_members, cal_y, cal_members):
    mean = test_members.mean(axis=1)
    cal_mean = cal_members.mean(axis=1)
    epistemic = test_members.var(axis=1, ddof=1)
    cal_epistemic = cal_members.var(axis=1, ddof=1)
    calibration_mse = float(np.mean((cal_y - cal_mean) ** 2))
    residual_noise = float(max(
        calibration_mse - float(cal_epistemic.mean()), VARIANCE_FLOOR))
    total = np.maximum(epistemic + residual_noise, VARIANCE_FLOOR)
    low = mean - Z95 * np.sqrt(total)
    high = mean + Z95 * np.sqrt(total)
    nll = .5 * (np.log(2 * np.pi * total) + (y - mean) ** 2 / total)
    covered = (y >= low) & (y <= high)
    metrics = {
        "rmse": float(mean_squared_error(y, mean) ** .5),
        "mae": float(mean_absolute_error(y, mean)),
        "r2": float(r2_score(y, mean)),
        "mean_gaussian_nll": float(nll.mean()),
        "mean_epistemic_variance": float(epistemic.mean()),
        "calibration_mse": calibration_mse,
        "residual_noise_variance": residual_noise,
        "mean_total_predictive_variance": float(total.mean()),
        "picp_95": float(covered.mean()),
        "coverage_error_95": float(abs(covered.mean() - .95)),
        "mpiw_95": float((high - low).mean()),
    }
    arrays = dict(predicted_soh=mean, epistemic_variance=epistemic,
                  total_predictive_variance=total, lower_95=low, upper_95=high,
                  covered_95=covered.astype(int), gaussian_nll=nll)
    return metrics, arrays


def train_primary_members(xs, xf, xc, xt, xc_source, xt_source,
                          source_y, fit_y, pooled_x, pooled_y, seed_base):
    cal = {name: [] for name in SETUPS}
    test = {name: [] for name in SETUPS}
    for member in range(N_ENSEMBLE):
        seed = seed_base + member
        fitted = [
            model(seed, ROUNDS).fit(xf, fit_y),
            model(seed + 1000, ROUNDS).fit(xs, source_y),
            model(seed + 2000, ROUNDS).fit(pooled_x, pooled_y),
        ]
        pretrained = model(seed + 3000, SOURCE_ROUNDS).fit(xs, source_y)
        fitted.append(model(seed + 4000, TARGET_ROUNDS).fit(
            xf, fit_y, xgb_model=pretrained.get_booster()))
        for setup, estimator, cal_x, test_x in zip(
                SETUPS, fitted, [xc, xc_source, xc, xc],
                [xt, xt_source, xt, xt]):
            cal[setup].append(estimator.predict(cal_x))
            test[setup].append(estimator.predict(test_x))
    return ({key: np.column_stack(value) for key, value in cal.items()},
            {key: np.column_stack(value) for key, value in test.items()})


def train_sensitivity_members(xs, xf, xc, xt, source_y, fit_y, seed_base):
    cal = {name: [] for name in SENSITIVITY_SETUPS}
    test = {name: [] for name in SENSITIVITY_SETUPS}
    for member in range(N_ENSEMBLE):
        seed = seed_base + member
        target_model = model(seed, ROUNDS).fit(xf, fit_y)
        pretrained = model(seed + 3000, SOURCE_ROUNDS).fit(xs, source_y)
        continued = model(seed + 4000, TARGET_ROUNDS).fit(
            xf, fit_y, xgb_model=pretrained.get_booster())
        for setup, estimator in zip(SENSITIVITY_SETUPS, [target_model, continued]):
            cal[setup].append(estimator.predict(xc))
            test[setup].append(estimator.predict(xt))
    return ({key: np.column_stack(value) for key, value in cal.items()},
            {key: np.column_stack(value) for key, value in test.items()})


def append_predictions(rows, test, k, setup, stats, arrays, target_name):
    for values in zip(
            test.index, test.Cycle, test.soh, arrays["predicted_soh"],
            arrays["epistemic_variance"], arrays["total_predictive_variance"],
            arrays["lower_95"], arrays["upper_95"], arrays["covered_95"],
            arrays["gaussian_nll"], test.protocol_group,
            test.discharge_regime_A, test.final_discharge_current_A):
        (index, cycle, actual, predicted, epistemic, total, low, high,
         covered, nll, protocol, regime, current) = values
        rows.append({
            "row_index": int(index), "test_cell": test.file_number.iloc[0],
            "Cycle": cycle, "target": target_name, "n_features": k,
            "setup": setup, "actual_soh": actual, "predicted_soh": predicted,
            "epistemic_variance": epistemic,
            "residual_noise_variance": stats["residual_noise_variance"],
            "total_predictive_variance": total, "lower_95": low,
            "upper_95": high, "covered_95": covered, "gaussian_nll": nll,
            "protocol_group": protocol, "discharge_regime_A": regime,
            "final_discharge_current_A": current,
        })


def aggregate_metrics(predictions, groups):
    output = []
    for keys, block in predictions.groupby(groups, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(groups, keys))
        row.update({
            "observations": len(block), "test_cells": block.test_cell.nunique(),
            "rmse": mean_squared_error(block.actual_soh, block.predicted_soh) ** .5,
            "mae": mean_absolute_error(block.actual_soh, block.predicted_soh),
            "r2": r2_score(block.actual_soh, block.predicted_soh),
            "mean_gaussian_nll": block.gaussian_nll.mean(),
            "picp_95": block.covered_95.mean(),
            "coverage_error_95": abs(block.covered_95.mean() - .95),
            "mpiw_95": (block.upper_95 - block.lower_95).mean(),
        })
        output.append(row)
    return pd.DataFrame(output)


def paired_tests(folds):
    output = []
    for k in FEATURE_COUNTS:
        wide = folds[folds.n_features == k].pivot(
            index="test_cell", columns="setup", values="rmse")
        transfer = wide[SETUPS[3]].to_numpy()
        for number, comparator in enumerate([SETUPS[0], SETUPS[2]]):
            baseline = wide[comparator].to_numpy()
            difference = transfer - baseline
            rng = np.random.default_rng(SEED + 100 * number + k)
            bootstrap = difference[rng.integers(0, len(difference), (5000, len(difference)))].mean(1)
            try:
                result = wilcoxon(transfer, baseline)
                statistic, p_value = float(result.statistic), float(result.pvalue)
            except ValueError:
                statistic, p_value = np.nan, np.nan
            output.append({
                "n_features": k, "transfer_setup": SETUPS[3], "comparator": comparator,
                "transfer_rmse": transfer.mean(), "comparator_rmse": baseline.mean(),
                "paired_rmse_difference": difference.mean(),
                "bootstrap_ci95_low": np.quantile(bootstrap, .025),
                "bootstrap_ci95_high": np.quantile(bootstrap, .975),
                "cells_transfer_improved": int((transfer < baseline).sum()),
                "n_test_cells": len(transfer), "wilcoxon_statistic": statistic,
                "wilcoxon_p": p_value,
            })
    return pd.DataFrame(output)


def plot_results(summary, rankings):
    labels = ["Target-only", "Source-only matched", "Naive pooled", "Raw continuation"]
    colors = ["#2878B5", "#777777", "#F39C34", "#D9534F"]
    for metric, ylabel, stem in [
        ("r2_mean", r"Mean cell-wise $R^2$", "Figure_R2"),
        ("rmse_mean", "Mean cell-wise RMSE", "Figure_RMSE"),
        ("picp_95_mean", "Mean 95% PICP", "Figure_PICP"),
        ("mean_gaussian_nll_mean", "Mean Gaussian NLL", "Figure_NLL"),
    ]:
        fig, ax = plt.subplots(figsize=(10.5, 6))
        for setup, label, color in zip(SETUPS, labels, colors):
            block = summary[summary.setup == setup].sort_values("n_features")
            ax.plot(block.n_features, block[metric], marker="o", label=label, color=color)
        if metric == "r2_mean":
            ax.axhline(.9, color="black", linestyle="--", label=r"Reference $R^2=0.90$")
        if metric == "picp_95_mean":
            ax.axhline(.95, color="black", linestyle="--", label="Nominal coverage = 0.95")
        ax.set(xlabel="Number of fold-ranked features", ylabel=ylabel,
               xticks=FEATURE_COUNTS,
               title=f"{DIRECTION_TITLE}: leakage-safe five-member deep ensemble")
        ax.grid(alpha=.25); ax.legend(frameon=False, ncol=2); fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
        fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)

    consensus = (rankings.groupby("feature")
                 .agg(mean_fusion_rank=("fusion_rank", "mean"),
                      std_fusion_rank=("fusion_rank", "std"),
                      mean_abs_target_correlation=("abs_target_correlation", "mean"),
                      mean_abs_shap=("mean_abs_shap", "mean"))
                 .sort_values("mean_fusion_rank").reset_index())
    consensus.to_csv(OUTPUT_DIR / "consensus_feature_ranking.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 7))
    y = np.arange(len(consensus))
    ax.barh(y, consensus.mean_fusion_rank,
            xerr=consensus.std_fusion_rank.fillna(0), color="#4C78A8", capsize=3)
    ax.set_yticks(y, consensus.feature); ax.invert_yaxis()
    ax.set(xlabel="Mean fold-local fusion rank (lower is better)",
           title=f"{DIRECTION_TITLE} feature-ranking stability")
    ax.grid(axis="x", alpha=.25); fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "Figure_feature_ranking.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "Figure_feature_ranking.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    started = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source_raw = pd.read_csv(SOURCE_CSV)
    source_raw = source_raw.rename(columns={"soh": "source_soh"})
    source_raw["file_number"] = SOURCE_NAME + "_" + source_raw.file_number.astype(str)
    source_raw["regime_normalised_SOH"] = source_raw.source_soh
    source_raw["raw_SOH"] = source_raw.source_soh
    target_raw = pd.read_csv(TARGET_CSV)

    source = prepare(source_raw, "regime_normalised_SOH")
    target_primary = prepare(target_raw, "regime_normalised_SOH")
    target_sensitivity = prepare(target_raw, "raw_SOH")
    folds, predictions, rankings, sensitivity_folds, sensitivity_predictions, timings = [], [], [], [], [], []
    cells = sorted(target_primary.file_number.unique())

    for fold_number, test_cell in enumerate(cells, start=1):
        fold_started = time.time()
        outer_primary = target_primary[target_primary.file_number != test_cell]
        test_primary = target_primary[target_primary.file_number == test_cell]
        fit_cells, cal_cells = split_fit_calibration(outer_primary, SEED + fold_number)
        fit_primary = outer_primary[outer_primary.file_number.isin(fit_cells)]
        cal_primary = outer_primary[outer_primary.file_number.isin(cal_cells)]
        ranking = fusion_ranking(fit_primary, SEED + fold_number)
        ranking.insert(0, "test_cell", test_cell)
        rankings.append(ranking)
        ordered = ranking.feature.tolist()

        outer_raw = target_sensitivity[target_sensitivity.file_number != test_cell]
        fit_raw = outer_raw[outer_raw.file_number.isin(fit_cells)]
        cal_raw = outer_raw[outer_raw.file_number.isin(cal_cells)]
        test_raw = target_sensitivity[target_sensitivity.file_number == test_cell]

        for k in FEATURE_COUNTS:
            group_started = time.time()
            selected = ordered[:k]
            target_low, target_high = limits(fit_primary, selected)
            source_low, source_high = limits(source, selected)
            xf, xc, xt = [clipped(frame, selected, target_low, target_high)
                          for frame in (fit_primary, cal_primary, test_primary)]
            xs = clipped(source, selected, source_low, source_high)
            xc_source, xt_source = [clipped(frame, selected, source_low, source_high)
                                    for frame in (cal_primary, test_primary)]
            pooled_x = pd.concat([xs, xf], ignore_index=True)
            pooled_y = pd.concat([source.soh, fit_primary.soh], ignore_index=True)
            cal_members, test_members = train_primary_members(
                xs, xf, xc, xt, xc_source, xt_source, source.soh, fit_primary.soh,
                pooled_x, pooled_y, SEED + fold_number * 10000 + k * 100)
            for setup in SETUPS:
                stats, arrays = ensemble_statistics(
                    test_primary.soh.to_numpy(), test_members[setup],
                    cal_primary.soh.to_numpy(), cal_members[setup])
                folds.append({
                    "test_cell": test_cell, "test_protocol": test_primary.protocol_group.iloc[0],
                    "n_features": k, "setup": setup, "features": "|".join(selected),
                    "n_fit_cells": len(fit_cells), "n_calibration_cells": len(cal_cells),
                    "fit_cells": "|".join(fit_cells), "calibration_cells": "|".join(cal_cells),
                    "n_ensemble": N_ENSEMBLE, **stats,
                })
                append_predictions(predictions, test_primary, k, setup, stats, arrays,
                                   "regime_normalised_SOH")

            # raw-SOH sensitivity uses the same cells, ranking, features, clipping,
            # seeds, and source pretraining; only the target label definition changes.
            xf_raw, xc_raw, xt_raw = [clipped(frame, selected, target_low, target_high)
                                      for frame in (fit_raw, cal_raw, test_raw)]
            cal_sens, test_sens = train_sensitivity_members(
                xs, xf_raw, xc_raw, xt_raw, source.soh, fit_raw.soh,
                SEED + 500000 + fold_number * 10000 + k * 100)
            for setup in SENSITIVITY_SETUPS:
                stats, arrays = ensemble_statistics(
                    test_raw.soh.to_numpy(), test_sens[setup],
                    cal_raw.soh.to_numpy(), cal_sens[setup])
                sensitivity_folds.append({
                    "test_cell": test_cell, "test_protocol": test_raw.protocol_group.iloc[0],
                    "n_features": k, "setup": setup, "features": "|".join(selected),
                    "n_fit_cells": len(fit_cells), "n_calibration_cells": len(cal_cells),
                    "n_ensemble": N_ENSEMBLE, **stats,
                })
                append_predictions(sensitivity_predictions, test_raw, k, setup, stats, arrays,
                                   "raw_SOH")
            timings.append({"test_cell": test_cell, "n_features": k,
                            "seconds": time.time() - group_started})
        timings.append({"test_cell": test_cell, "n_features": "ALL",
                        "seconds": time.time() - fold_started})
        print(f"{fold_number:02d}/{len(cells)} held-out cell={test_cell}", flush=True)

    folds = pd.DataFrame(folds)
    predictions = pd.DataFrame(predictions)
    rankings = pd.concat(rankings, ignore_index=True)
    sensitivity_folds = pd.DataFrame(sensitivity_folds)
    sensitivity_predictions = pd.DataFrame(sensitivity_predictions)
    metric_columns = ["rmse", "mae", "r2", "mean_gaussian_nll",
                      "mean_epistemic_variance", "calibration_mse",
                      "residual_noise_variance", "mean_total_predictive_variance",
                      "picp_95", "coverage_error_95", "mpiw_95"]
    summary = folds.groupby(["n_features", "setup"])[metric_columns].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    sensitivity_summary = sensitivity_folds.groupby(
        ["n_features", "setup"])[metric_columns].agg(["mean", "std"])
    sensitivity_summary.columns = [f"{metric}_{stat}" for metric, stat in sensitivity_summary.columns]
    sensitivity_summary = sensitivity_summary.reset_index()

    pooled = aggregate_metrics(predictions, ["n_features", "setup"])
    sensitivity_pooled = aggregate_metrics(sensitivity_predictions, ["n_features", "setup"])
    protocol_metrics = aggregate_metrics(
        predictions, ["n_features", "setup", "protocol_group"])
    regime_metrics = aggregate_metrics(
        predictions, ["n_features", "setup", "discharge_regime_A"])

    folds.to_csv(OUTPUT_DIR / "fold_metrics_all_feature_groups.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "out_of_fold_predictions_with_intervals.csv", index=False)
    rankings.to_csv(OUTPUT_DIR / "nested_feature_rankings.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "summary_metrics_by_feature_count.csv", index=False)
    pooled.to_csv(OUTPUT_DIR / "pooled_oof_metrics_by_feature_count.csv", index=False)
    paired_tests(folds).to_csv(OUTPUT_DIR / "paired_transfer_comparison_by_feature_count.csv", index=False)
    sensitivity_folds.to_csv(OUTPUT_DIR / "raw_soh_sensitivity_fold_metrics.csv", index=False)
    sensitivity_predictions.to_csv(OUTPUT_DIR / "raw_soh_sensitivity_oof_predictions.csv", index=False)
    sensitivity_summary.to_csv(OUTPUT_DIR / "raw_soh_sensitivity_summary.csv", index=False)
    sensitivity_pooled.to_csv(OUTPUT_DIR / "raw_soh_sensitivity_pooled_oof.csv", index=False)
    protocol_metrics.to_csv(OUTPUT_DIR / "metrics_by_protocol_group.csv", index=False)
    regime_metrics.to_csv(OUTPUT_DIR / "metrics_by_discharge_regime.csv", index=False)
    pd.DataFrame(timings).to_csv(OUTPUT_DIR / "training_times.csv", index=False)
    plot_results(summary, rankings)

    audit = target_raw.groupby("file_number").agg(
        observations=("Cycle", "size"), first_cycle=("Cycle", "min"),
        last_cycle=("Cycle", "max"), protocol_group=("protocol_group", "first"),
        regimes=("discharge_regime_A", "nunique"),
    )
    audit.to_csv(OUTPUT_DIR / "target_dataset_audit.csv")
    metadata = {
        "direction": DIRECTION_NAME, "source_chemistry": SOURCE_NAME,
        "primary_target": "regime_normalised_SOH",
        "sensitivity_target": "raw_SOH", "target_cells": len(cells),
        "target_observations": len(target_raw), "candidate_features": FEATURES,
        "feature_counts": FEATURE_COUNTS, "excluded_feature": "Temperature unavailable in Multiorbits",
        "excluded_model_inputs": ["Cycle", "cell identity", "capacity", "SOH targets",
                                  "discharge regime", "protocol group", "throughput", "elapsed time"],
        "outer_split": "12-fold target-cell LOCO",
        "target_partition_per_fold": "8 fitting, 3 protocol-stratified calibration, 1 test cell",
        "ranking": "fold-local target-fitting-cell correlation--SHAP fusion ranking",
        "hard_correlation_filter_threshold": None,
        "preprocessing": "cell-relative median of first five unlabelled feature observations; fold-domain 0.1%/99.9% clipping",
        "regime_target_reference": "median first five capacity observations within each cell and final-current regime",
        "target_reference_disclosure": "early-life target-side capacity initialization; no future labels or test residuals",
        "ensemble_members": N_ENSEMBLE,
        "uncertainty": {"epistemic": "sample variance across five members",
                        "residual_noise": "max(calibration MSE - mean calibration epistemic variance, 1e-10)",
                        "test_labels": "evaluation only after target definition"},
        "matched_methods": SETUPS, "gated_transfer_used": False,
        "tree_budget": {"target_only": ROUNDS, "source_only": ROUNDS,
                        "naive_pooled": ROUNDS, "source_pretraining": SOURCE_ROUNDS,
                        "target_continuation": TARGET_ROUNDS, "raw_total": ROUNDS},
        "xgboost": {"learning_rate": .10, "max_depth": 4, "min_child_weight": 3.0,
                    "reg_lambda": 5.0, "reg_alpha": 0.0, "subsample": .9,
                    "colsample_bytree": .9, "objective": "reg:squarederror",
                    "tree_method": "hist"},
        "random_seed_base": SEED,
        "reviewer_1_scope": "controlled individual-cell variable-current regimes; not pack or fully random field validation",
        "reviewer_2_comment_4": "no intrinsic chemistry-complexity claim",
        "elapsed_seconds": time.time() - started,
    }
    (OUTPUT_DIR / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
