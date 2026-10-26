"""Leakage-safe ranked-feature study for NCA -> NMC (top 4 through top 15).

Each outer NMC LOCO fold recomputes correlation + SHAP fusion ranking using
outer-fit NMC cells only. Identical fold-specific top-k features, target
supervision, and evaluation rows are used by all matched methods. Cycle,
Capacity, SOH, and cell identity are never model inputs.
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
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib_cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import R2_C1_C2_C3_C4_C5__Dependency_XGBoost_Common_Utilities as core
except ModuleNotFoundError:  # Support importing as Kode_python.<module> from repo root.
    from Kode_python import R2_C1_C2_C3_C4_C5__Dependency_XGBoost_Common_Utilities as core


SEED = 42
FEATURE_COUNTS = list(range(4, 16))
ROUNDS = 40
SOURCE_ROUNDS = 20
TARGET_ROUNDS = ROUNDS - SOURCE_ROUNDS
MIN_GATE_GAIN = 0.03
FEATURES = core.HEALTH_FEATURES.copy()
ROOT = Path(__file__).resolve().parent
SOURCE_CSV = ROOT / "NCA_Integral_XGBoost_38.csv"
TARGET_CSV = ROOT / "NMC_Integral_XGBoost.csv"
OUTPUT_DIR = ROOT / "R2_C1_C2_C3_C4_C5__dependency_results_NCA_to_NMC_ranked_groups"
PARAMS = {"max_depth": 4, "min_child_weight": 3.0,
          "reg_lambda": 5.0, "reg_alpha": 0.0}


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep 15 columns; normalize 14 to five unlabelled early-life records."""
    out = frame.copy()
    ref = (out.groupby("file_number", sort=False).head(5)
             .groupby("file_number")[FEATURES].median())
    for feature in FEATURES:
        if feature == "Temperature":
            continue
        r = out.file_number.map(ref[feature]).astype(float)
        v = out[feature].astype(float)
        out[feature] = np.where(r.abs() > 1e-12, v / r, v - r)
    return out


def model(seed: int, rounds: int) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=rounds, learning_rate=.10, subsample=.9,
        colsample_bytree=.9, objective="reg:squarederror", eval_metric="rmse",
        tree_method="hist", n_jobs=-1, random_state=seed, verbosity=0, **PARAMS)


def fusion_ranking(fit: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Rank all 15 features; nothing from the held-out cell is consulted."""
    lo, hi = core.limits(fit, FEATURES)
    x = core.clipped(fit, FEATURES, lo, hi)
    # Temperature is constant inside some temperature-matched sensitivity
    # folds. Its undefined correlation is deliberately ranked as zero without
    # emitting divide-by-zero warnings.
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = x.corrwith(fit.soh).abs().fillna(0.0)
    selector = model(seed, 40).fit(x, fit.soh)
    sample = x.sample(min(5000, len(x)), random_state=seed)
    values = selector.get_booster().predict(
        xgb.DMatrix(sample, feature_names=FEATURES), pred_contribs=True)
    shap = pd.Series(np.abs(values[:, :-1]).mean(0), index=FEATURES)
    table = pd.DataFrame({"feature": FEATURES,
                          "abs_target_correlation": corr[FEATURES].values,
                          "mean_abs_shap": shap[FEATURES].values})
    table["correlation_rank"] = table.abs_target_correlation.rank(
        ascending=False, method="average")
    table["shap_rank"] = table.mean_abs_shap.rank(
        ascending=False, method="average")
    table["fusion_rank"] = (table.correlation_rank + table.shap_rank) / 2
    return table.sort_values(
        ["fusion_rank", "shap_rank", "correlation_rank", "feature"]
    ).reset_index(drop=True)


def metrics(y: np.ndarray, pred: np.ndarray, cal_y: np.ndarray,
            cal_pred: np.ndarray) -> dict[str, float]:
    # Variance and Gaussian interval are estimated exclusively on calibration cells.
    variance = float(max(np.mean((cal_y - cal_pred) ** 2), 1e-8))
    std = variance ** .5
    low, high = pred - 1.959963984540054 * std, pred + 1.959963984540054 * std
    nll = .5 * (np.log(2 * np.pi * variance) + (y - pred) ** 2 / variance)
    return {"rmse": float(mean_squared_error(y, pred) ** .5),
            "mae": float(mean_absolute_error(y, pred)),
            "r2": float(r2_score(y, pred)), "calibrated_variance": variance,
            "mean_nll": float(nll.mean()),
            "picp_95": float(((y >= low) & (y <= high)).mean()),
            "mpiw_95": float((high - low).mean())}


def plot_outputs(summary: pd.DataFrame, rankings: pd.DataFrame) -> None:
    order = ["NMC_target_only", "NCA_source_only_zero_shot", "NCA_NMC_naive_pooled",
             "NCA_to_NMC_raw_continuation", "NCA_to_NMC_calibration_gated"]
    labels = ["Target-only", "Source-only", "Naive pooled", "Continuation", "Gated"]
    colors = ["#2878B5", "#999999", "#F39C34", "#D9534F", "#2CA25F"]
    fig, ax = plt.subplots(figsize=(11, 6))
    for setup, label, color in zip(order, labels, colors):
        z = summary[summary.setup == setup].sort_values("n_features")
        ax.plot(z.n_features, z.r2_mean, marker="o", label=label, color=color)
    ax.axhline(.9, color="black", linestyle="--", label=r"Requested mean $R^2=0.90$")
    ax.set(xlabel="Number of top-ranked features", ylabel=r"Mean cell-wise $R^2$",
           xticks=FEATURE_COUNTS, title="NCA to NMC: nested LOCO feature-group comparison")
    ax.grid(alpha=.25); ax.legend(ncol=2, frameon=False); fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "Figure_R2_by_feature_count.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "Figure_R2_by_feature_count.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    for setup, label, color in zip(order, labels, colors):
        z = summary[summary.setup == setup].sort_values("n_features")
        ax.plot(z.n_features, z.rmse_mean, marker="o", label=label, color=color)
    ax.set(xlabel="Number of top-ranked features", ylabel="Macro RMSE",
           xticks=FEATURE_COUNTS, title="Matched-baseline error by feature-group size")
    ax.grid(alpha=.25); ax.legend(ncol=2, frameon=False); fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "Figure_RMSE_by_feature_count.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "Figure_RMSE_by_feature_count.pdf", bbox_inches="tight")
    plt.close(fig)

    consensus = (rankings.groupby("feature")
                  .agg(mean_fusion_rank=("fusion_rank", "mean"),
                       std_fusion_rank=("fusion_rank", "std"),
                       mean_abs_shap=("mean_abs_shap", "mean"))
                  .sort_values("mean_fusion_rank").reset_index())
    fig, ax = plt.subplots(figsize=(9, 6))
    y = np.arange(len(consensus))
    ax.barh(y, consensus.mean_fusion_rank, xerr=consensus.std_fusion_rank,
            color="#4C78A8", alpha=.9, capsize=3)
    ax.set_yticks(y, consensus.feature); ax.invert_yaxis()
    ax.set(xlabel="Mean fusion rank across outer folds (lower is better)",
           title="Stability of leakage-free feature ranking")
    ax.grid(axis="x", alpha=.25); fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "Figure_feature_ranking.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "Figure_feature_ranking.pdf", bbox_inches="tight")
    plt.close(fig)
    consensus.to_csv(OUTPUT_DIR / "consensus_feature_ranking.csv", index=False)


def matched_condition_sensitivity(source: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    """Control temperature, cell count, and row count for the complexity claim."""
    rng = np.random.default_rng(SEED)
    output = []
    common = sorted(set(source.Temperature.unique()) & set(target.Temperature.unique()))
    for temp in common:
        blocks = {"NCA": source[source.Temperature == temp],
                  "NMC": target[target.Temperature == temp]}
        n_cells = min(z.file_number.nunique() for z in blocks.values())
        rows_per_cell = int(min(z.groupby("file_number").size().min() for z in blocks.values()))
        if n_cells < 4:
            continue
        for chemistry, frame in blocks.items():
            chosen = rng.choice(frame.file_number.unique(), n_cells, replace=False)
            balanced = frame[frame.file_number.isin(chosen)].groupby(
                "file_number", group_keys=False).sample(rows_per_cell, random_state=SEED)
            cv = GroupKFold(n_splits=min(3, n_cells)); fold_rows = []
            for inner, (tr, va) in enumerate(cv.split(
                    balanced, balanced.soh, balanced.file_number)):
                train, valid = balanced.iloc[tr], balanced.iloc[va]
                ordered = fusion_ranking(train, SEED + inner).feature.tolist()
                for k in [4, 8, 12, 15]:
                    selected = ordered[:k]; lo, hi = core.limits(train, selected)
                    fitted = model(SEED + inner, ROUNDS).fit(
                        core.clipped(train, selected, lo, hi), train.soh)
                    pred = fitted.predict(core.clipped(valid, selected, lo, hi))
                    fold_rows.append({"n_features": k,
                                      "rmse": mean_squared_error(valid.soh, pred) ** .5,
                                      "r2": r2_score(valid.soh, pred)})
            z = pd.DataFrame(fold_rows).groupby("n_features")[["rmse", "r2"]].mean().reset_index()
            z.insert(0, "rows_per_cell", rows_per_cell)
            z.insert(0, "balanced_cells", n_cells)
            z.insert(0, "chemistry", chemistry)
            z.insert(0, "temperature", temp)
            output.append(z)
    return pd.concat(output, ignore_index=True)


def main() -> None:
    started = time.time(); OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source, target = prepare(pd.read_csv(SOURCE_CSV)), prepare(pd.read_csv(TARGET_CSV))
    rows, rank_frames = [], []
    cells = sorted(target.file_number.unique())
    for fold_no, test_cell in enumerate(cells, 1):
        train = target[target.file_number != test_cell].copy()
        test = target[target.file_number == test_cell].copy()
        fit_cells, cal_cells = core.split_fit_calibration(train, SEED + int(test_cell))
        fit = train[train.file_number.isin(fit_cells)].copy()
        cal = train[train.file_number.isin(cal_cells)].copy()
        ranking = fusion_ranking(fit, SEED + int(test_cell))
        ranking.insert(0, "test_cell", test_cell); rank_frames.append(ranking)
        ordered = ranking.feature.tolist()
        for k in FEATURE_COUNTS:
            features = ordered[:k]
            tlo, thi = core.limits(fit, features)
            slo, shi = core.limits(source, features)
            xf, xc, xt = (core.clipped(z, features, tlo, thi) for z in (fit, cal, test))
            xs = core.clipped(source, features, slo, shi)
            xc_s, xt_s = (core.clipped(z, features, slo, shi) for z in (cal, test))

            base = model(SEED + int(test_cell), ROUNDS).fit(xf, fit.soh)
            src = model(SEED + 1000 + int(test_cell), ROUNDS).fit(xs, source.soh)
            pre = model(SEED + 2000 + int(test_cell), SOURCE_ROUNDS).fit(xs, source.soh)
            cont = model(SEED + 3000 + int(test_cell), TARGET_ROUNDS).fit(
                xf, fit.soh, xgb_model=pre.get_booster())
            pool = model(SEED + 4000 + int(test_cell), ROUNDS).fit(
                pd.concat([xs, xf], ignore_index=True),
                pd.concat([source.soh, fit.soh], ignore_index=True))

            cal_pred = {"NMC_target_only": base.predict(xc),
                        "NCA_source_only_zero_shot": src.predict(xc_s),
                        "NCA_NMC_naive_pooled": pool.predict(xc),
                        "NCA_to_NMC_raw_continuation": cont.predict(xc)}
            test_pred = {"NMC_target_only": base.predict(xt),
                         "NCA_source_only_zero_shot": src.predict(xt_s),
                         "NCA_NMC_naive_pooled": pool.predict(xt),
                         "NCA_to_NMC_raw_continuation": cont.predict(xt)}
            base_mse = mean_squared_error(cal.soh, cal_pred["NMC_target_only"])
            choices = ["NCA_NMC_naive_pooled", "NCA_to_NMC_raw_continuation"]
            gains = {n: 1 - mean_squared_error(cal.soh, cal_pred[n]) / base_mse for n in choices}
            chosen = max(gains, key=gains.get)
            if gains[chosen] < MIN_GATE_GAIN:
                alpha = 0.0
            else:
                direction = cal_pred[chosen] - cal_pred["NMC_target_only"]
                den = float(direction @ direction)
                alpha = 0.0 if den < 1e-15 else float(np.clip(
                    ((cal.soh.to_numpy() - cal_pred["NMC_target_only"]) @ direction) / den, 0, 1))
            cal_pred["NCA_to_NMC_calibration_gated"] = cal_pred["NMC_target_only"] + alpha * (
                cal_pred[chosen] - cal_pred["NMC_target_only"])
            test_pred["NCA_to_NMC_calibration_gated"] = test_pred["NMC_target_only"] + alpha * (
                test_pred[chosen] - test_pred["NMC_target_only"])
            for setup, pred in test_pred.items():
                rows.append({"test_cell": test_cell, "n_features": k, "setup": setup,
                             "features": "|".join(features), "n_fit_cells": len(fit_cells),
                             "n_calibration_cells": len(cal_cells), "gate_choice": chosen,
                             "gate_weight": alpha, **metrics(test.soh.to_numpy(), pred,
                             cal.soh.to_numpy(), cal_pred[setup])})
        print(f"{fold_no:02d}/{len(cells)} cell={int(test_cell):02d}", flush=True)

    folds = pd.DataFrame(rows); rankings = pd.concat(rank_frames, ignore_index=True)
    metric_cols = ["rmse", "mae", "r2", "mean_nll", "picp_95", "mpiw_95"]
    summary = folds.groupby(["n_features", "setup"])[metric_cols].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    paired = []
    for k in FEATURE_COUNTS:
        z = folds[folds.n_features == k].pivot(index="test_cell", columns="setup", values="rmse")
        b, g = z.NMC_target_only.to_numpy(), z.NCA_to_NMC_calibration_gated.to_numpy()
        rng = np.random.default_rng(SEED + k); diff = g - b
        boot = diff[rng.integers(0, len(diff), (5000, len(diff)))].mean(1)
        try: w = wilcoxon(g, b); stat, p = float(w.statistic), float(w.pvalue)
        except ValueError: stat, p = np.nan, np.nan
        paired.append({"n_features": k, "baseline_rmse": b.mean(), "gated_rmse": g.mean(),
                       "difference": diff.mean(), "ci95_low": np.quantile(boot, .025),
                       "ci95_high": np.quantile(boot, .975), "cells_improved": int((g < b).sum()),
                       "wilcoxon_statistic": stat, "wilcoxon_p": p})
    folds.to_csv(OUTPUT_DIR / "fold_metrics_all_feature_groups.csv", index=False)
    rankings.to_csv(OUTPUT_DIR / "nested_feature_rankings.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "summary_metrics_by_feature_count.csv", index=False)
    pd.DataFrame(paired).to_csv(OUTPUT_DIR / "paired_gated_comparison_by_feature_count.csv", index=False)
    plot_outputs(summary, rankings)
    matched_condition_sensitivity(source, target).to_csv(
        OUTPUT_DIR / "matched_condition_complexity.csv", index=False)
    metadata = {"direction": "NCA_to_NMC", "feature_counts": FEATURE_COUNTS,
                "candidate_features": FEATURES, "cycle_used": False,
                "capacity_used": False, "outer_split": "55-fold NMC LOCO",
                "ranking": "fold-local mean of correlation rank and SHAP rank",
                "calibration": "complete cell-disjoint NMC calibration cells",
                "rounds": ROUNDS, "source_rounds": SOURCE_ROUNDS,
                "target_continuation_rounds": TARGET_ROUNDS, "random_seed": SEED,
                "minimum_gate_gain": MIN_GATE_GAIN,
                "early_reference": "median of first five unlabelled records per cell",
                "elapsed_seconds": time.time() - started}
    (OUTPUT_DIR / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary.to_string(index=False)); print(f"Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
