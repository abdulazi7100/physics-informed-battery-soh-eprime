"""Leakage-audited Wenzhou individual-cell to parallel-pack transfer.

Primary target: discharge-regime-normalised SOH.
Sensitivity target: nominal-capacity SOH.
Outer validation: leave one complete pack out (six folds).
Gated transfer is deliberately absent.
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
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))
import matplotlib.pyplot as plt


SEED = 42
N_ENSEMBLE = 5
FEATURE_COUNTS = list(range(4, 15))
ROUNDS = 40
SOURCE_ROUNDS = 20
TARGET_ROUNDS = 20
Z95 = 1.959963984540054
VARIANCE_FLOOR = 1e-10
INPUT = (PROJECT_ROOT / "kode_pack_wenzhou" /
         "extracted_features_14_cell_to_pack" /
         "Wenzhou_Cell_Pack_14_features.csv")
OUTPUT = ROOT / "R1_C1__R2_C1_C2_C3_C4_C5__results_Wenzhou_Cell_to_Pack"
FEATURES = [
    "CVCT", "CCCT", "TCT", "AVCC", "ACCV", "TES", "IVCC", "IVCCCV",
    "RMSVCC", "SkewVCC", "KurtVCC", "RMSCCV", "SkewCCV", "KurtCCV",
]
SETUPS = [
    "Pack_target_only", "Cell_source_only_matched",
    "Cell_Pack_naive_pooled", "Cell_to_Pack_raw_continuation",
]
SENSITIVITY_SETUPS = ["Pack_target_only", "Cell_to_Pack_raw_continuation"]
CALIBRATION_MAP = {
    "Wenzhou_Pack_#09": "Wenzhou_Pack_#12",
    "Wenzhou_Pack_#10": "Wenzhou_Pack_#11",
    "Wenzhou_Pack_#11": "Wenzhou_Pack_#13",
    "Wenzhou_Pack_#12": "Wenzhou_Pack_#09",
    "Wenzhou_Pack_#13": "Wenzhou_Pack_#10",
    # No second complex pack exists; use a nominal three-cell pack.
    "Wenzhou_Pack_#14": "Wenzhou_Pack_#12",
}
PARAMS = dict(max_depth=4, min_child_weight=3.0, reg_lambda=5.0, reg_alpha=0.0)


def prepare(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    """Use five early unlabelled feature records as a unit-specific reference."""
    out = frame.sort_values(["file_number", "Cycle"]).copy()
    out["soh"] = out[target].astype(float)
    reference = (out.groupby("file_number", sort=False).head(5)
                 .groupby("file_number")[FEATURES].median())
    for feature in FEATURES:
        ref = out.file_number.map(reference[feature]).astype(float)
        value = out[feature].astype(float)
        out[feature] = np.where(ref.abs() > 1e-12, value / ref, value - ref)
    finite = np.isfinite(out[FEATURES + ["soh"]]).all(axis=1)
    return out.loc[finite].reset_index(drop=True)


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
    low, high = limits(fit, FEATURES)
    xx = clipped(fit, FEATURES, low, high)
    correlation = xx.corrwith(fit.soh).abs().fillna(0.0)
    selector = model(seed, ROUNDS).fit(xx, fit.soh)
    sample = xx.sample(min(5000, len(xx)), random_state=seed)
    contribution = selector.get_booster().predict(
        xgb.DMatrix(sample, feature_names=FEATURES), pred_contribs=True)
    shap = pd.Series(np.abs(contribution[:, :-1]).mean(0), index=FEATURES)
    ranking = pd.DataFrame({
        "feature": FEATURES,
        "abs_target_correlation": correlation[FEATURES].to_numpy(),
        "mean_abs_shap": shap[FEATURES].to_numpy(),
    })
    ranking["correlation_rank"] = ranking.abs_target_correlation.rank(
        ascending=False, method="average")
    ranking["shap_rank"] = ranking.mean_abs_shap.rank(
        ascending=False, method="average")
    ranking["fusion_rank"] = (
        ranking.correlation_rank + ranking.shap_rank) / 2
    return ranking.sort_values(
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
    sd = np.sqrt(total)
    low, high = mean - Z95 * sd, mean + Z95 * sd
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
    arrays = dict(
        predicted_soh=mean, epistemic_variance=epistemic,
        total_predictive_variance=total, lower_95=low, upper_95=high,
        covered_95=covered.astype(int), gaussian_nll=nll,
    )
    return metrics, arrays


def train_primary(xs, xf, xc, xt, xc_source, xt_source,
                  source_y, fit_y, pooled_x, pooled_y, seed_base):
    cal = {name: [] for name in SETUPS}
    test = {name: [] for name in SETUPS}
    for member in range(N_ENSEMBLE):
        seed = seed_base + member
        estimators = [
            model(seed, ROUNDS).fit(xf, fit_y),
            model(seed + 1000, ROUNDS).fit(xs, source_y),
            model(seed + 2000, ROUNDS).fit(pooled_x, pooled_y),
        ]
        pretrained = model(seed + 3000, SOURCE_ROUNDS).fit(xs, source_y)
        estimators.append(model(seed + 4000, TARGET_ROUNDS).fit(
            xf, fit_y, xgb_model=pretrained.get_booster()))
        for setup, estimator, cx, tx in zip(
                SETUPS, estimators, [xc, xc_source, xc, xc],
                [xt, xt_source, xt, xt]):
            cal[setup].append(estimator.predict(cx))
            test[setup].append(estimator.predict(tx))
    return ({k: np.column_stack(v) for k, v in cal.items()},
            {k: np.column_stack(v) for k, v in test.items()})


def train_sensitivity(xs, xf, xc, xt, source_y, fit_y, seed_base):
    cal = {name: [] for name in SENSITIVITY_SETUPS}
    test = {name: [] for name in SENSITIVITY_SETUPS}
    for member in range(N_ENSEMBLE):
        seed = seed_base + member
        target_model = model(seed, ROUNDS).fit(xf, fit_y)
        pretrained = model(seed + 3000, SOURCE_ROUNDS).fit(xs, source_y)
        continuation = model(seed + 4000, TARGET_ROUNDS).fit(
            xf, fit_y, xgb_model=pretrained.get_booster())
        for setup, estimator in zip(
                SENSITIVITY_SETUPS, [target_model, continuation]):
            cal[setup].append(estimator.predict(xc))
            test[setup].append(estimator.predict(xt))
    return ({k: np.column_stack(v) for k, v in cal.items()},
            {k: np.column_stack(v) for k, v in test.items()})


def append_predictions(rows, test, k, setup, stats, arrays, target_name):
    for position, (_, item) in enumerate(test.iterrows()):
        rows.append({
            "row_index": int(item.name), "test_pack": item.file_number,
            "Cycle": item.Cycle, "target": target_name,
            "n_features": k, "setup": setup,
            "actual_soh": item.soh,
            "predicted_soh": arrays["predicted_soh"][position],
            "epistemic_variance": arrays["epistemic_variance"][position],
            "residual_noise_variance": stats["residual_noise_variance"],
            "total_predictive_variance": arrays["total_predictive_variance"][position],
            "lower_95": arrays["lower_95"][position],
            "upper_95": arrays["upper_95"][position],
            "covered_95": arrays["covered_95"][position],
            "gaussian_nll": arrays["gaussian_nll"][position],
            "group_size": item.group_size,
            "operating_mode": item.operating_mode,
            "final_discharge_C_rate": item.final_discharge_C_rate,
            "life_stage": item.life_stage,
        })


def aggregate_metrics(predictions, groups):
    output = []
    for keys, block in predictions.groupby(groups, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(groups, keys))
        y, p = block.actual_soh, block.predicted_soh
        row.update({
            "observations": len(block),
            "test_packs": block.test_pack.nunique(),
            "rmse": mean_squared_error(y, p) ** .5,
            "mae": mean_absolute_error(y, p),
            "r2": r2_score(y, p) if len(block) > 1 and y.var() > 0 else np.nan,
            "mean_gaussian_nll": block.gaussian_nll.mean(),
            "picp_95": block.covered_95.mean(),
            "coverage_error_95": abs(block.covered_95.mean() - .95),
            "mpiw_95": (block.upper_95 - block.lower_95).mean(),
        })
        output.append(row)
    return pd.DataFrame(output)


def paired_tests(folds):
    rows = []
    for k in FEATURE_COUNTS:
        wide = folds[folds.n_features == k].pivot(
            index="test_pack", columns="setup", values="rmse")
        transfer = wide[SETUPS[3]].to_numpy()
        for j, comparator in enumerate([SETUPS[0], SETUPS[2]]):
            baseline = wide[comparator].to_numpy()
            difference = transfer - baseline
            rng = np.random.default_rng(SEED + 100 * j + k)
            bootstrap = difference[
                rng.integers(0, len(difference), (5000, len(difference)))].mean(1)
            try:
                result = wilcoxon(transfer, baseline)
                statistic, p_value = float(result.statistic), float(result.pvalue)
            except ValueError:
                statistic, p_value = np.nan, np.nan
            rows.append({
                "n_features": k, "transfer_setup": SETUPS[3],
                "comparator": comparator,
                "transfer_rmse": transfer.mean(),
                "comparator_rmse": baseline.mean(),
                "paired_rmse_difference": difference.mean(),
                "bootstrap_ci95_low": np.quantile(bootstrap, .025),
                "bootstrap_ci95_high": np.quantile(bootstrap, .975),
                "packs_transfer_improved": int((transfer < baseline).sum()),
                "n_test_packs": len(transfer),
                "wilcoxon_statistic": statistic, "wilcoxon_p": p_value,
            })
    return pd.DataFrame(rows)


def plot_results(summary, rankings, predictions):
    labels = ["Pack target-only", "Cell source-only", "Naive pooled", "Raw continuation"]
    colors = ["#2878B5", "#777777", "#F39C34", "#D9534F"]
    specs = [
        ("r2_mean", r"Mean pack-wise $R^2$", "Figure_R2"),
        ("rmse_mean", "Mean pack-wise RMSE", "Figure_RMSE"),
        ("picp_95_mean", "Mean 95% PICP", "Figure_PICP"),
        ("mean_gaussian_nll_mean", "Mean Gaussian NLL", "Figure_NLL"),
        ("mpiw_95_mean", "Mean 95% MPIW", "Figure_MPIW"),
    ]
    for metric, ylabel, stem in specs:
        fig, ax = plt.subplots(figsize=(10.5, 6))
        for setup, label, color in zip(SETUPS, labels, colors):
            block = summary[summary.setup == setup].sort_values("n_features")
            ax.plot(block.n_features, block[metric], marker="o", label=label, color=color)
        if metric == "r2_mean":
            ax.axhline(.9, color="black", linestyle="--", linewidth=1)
        if metric == "picp_95_mean":
            ax.axhline(.95, color="black", linestyle="--", linewidth=1)
        ax.set(xlabel="Number of fold-ranked features", ylabel=ylabel,
               xticks=FEATURE_COUNTS,
               title="Wenzhou individual-cell to pack deep-ensemble transfer")
        ax.grid(alpha=.25); ax.legend(frameon=False, ncol=2); fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(OUTPUT / f"{stem}.{suffix}", dpi=300, bbox_inches="tight")
        plt.close(fig)

    consensus = (rankings.groupby("feature")
                 .agg(mean_fusion_rank=("fusion_rank", "mean"),
                      std_fusion_rank=("fusion_rank", "std"),
                      mean_abs_target_correlation=("abs_target_correlation", "mean"),
                      mean_abs_shap=("mean_abs_shap", "mean"))
                 .sort_values("mean_fusion_rank").reset_index())
    consensus.to_csv(OUTPUT / "consensus_feature_ranking.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 6))
    pos = np.arange(len(consensus))
    ax.barh(pos, consensus.mean_fusion_rank,
            xerr=consensus.std_fusion_rank.fillna(0), color="#4C78A8", capsize=3)
    ax.set_yticks(pos, consensus.feature); ax.invert_yaxis()
    ax.set(xlabel="Mean fusion rank across leave-one-pack-out folds",
           title="Fold-local feature-ranking stability")
    ax.grid(axis="x", alpha=.25); fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT / f"Figure_feature_ranking.{suffix}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)

    # Descriptive trajectories at the matched k with lowest continuation mean RMSE.
    best_k = int(summary[summary.setup == SETUPS[3]].sort_values("rmse_mean").iloc[0].n_features)
    subset = predictions[predictions.n_features == best_k]
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharey=True)
    for ax, (pack, block) in zip(axes.ravel(), subset.groupby("test_pack", sort=True)):
        actual = block[block.setup == SETUPS[0]].sort_values("Cycle")
        ax.plot(actual.Cycle, actual.actual_soh, color="black", lw=1.8, label="Actual")
        for setup, label, color in zip(SETUPS, labels, colors):
            curve = block[block.setup == setup].sort_values("Cycle")
            ax.plot(curve.Cycle, curve.predicted_soh, color=color, lw=1.0,
                    alpha=.9, label=label)
        ax.set_title(pack.replace("Wenzhou_", "")); ax.grid(alpha=.2)
        ax.set_xlabel("Cycle"); ax.set_ylabel("Regime-normalised SOH")
    handles, legend_labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=5, frameon=False)
    fig.suptitle(f"Held-out pack OOF trajectories ({best_k} fold-ranked features)")
    fig.tight_layout(rect=(0, .05, 1, .97))
    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT / f"Figure_pack_OOF_trajectories.{suffix}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)

    # Joint sharpness/coverage view: desirable points are near PICP=.95 and left.
    fig, ax = plt.subplots(figsize=(9, 6))
    for setup, label, color in zip(SETUPS, labels, colors):
        block = summary[summary.setup == setup].sort_values("n_features")
        ax.plot(block.mpiw_95_mean, block.picp_95_mean, marker="o",
                color=color, label=label)
        for row in block.itertuples():
            ax.annotate(str(int(row.n_features)),
                        (row.mpiw_95_mean, row.picp_95_mean), fontsize=7,
                        xytext=(3, 2), textcoords="offset points")
    ax.axhline(.95, color="black", linestyle="--", linewidth=1)
    ax.set(xlabel="Mean 95% interval width (lower is sharper)",
           ylabel="Mean pack-wise 95% coverage",
           title="Prediction-interval calibration and sharpness")
    ax.grid(alpha=.25); ax.legend(frameon=False); fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT / f"Figure_uncertainty_calibration.{suffix}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)

    pack_metrics = aggregate_metrics(
        subset, ["setup", "test_pack"]
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    pack_names = sorted(pack_metrics.test_pack.unique())
    positions = np.arange(len(pack_names)); width = .19
    for j, (setup, label, color) in enumerate(zip(SETUPS, labels, colors)):
        block = (pack_metrics[pack_metrics.setup == setup]
                 .set_index("test_pack").reindex(pack_names))
        ax.bar(positions + (j - 1.5) * width, block.rmse, width,
               color=color, label=label)
    ax.set_xticks(positions, [x.replace("Wenzhou_", "") for x in pack_names])
    ax.set(ylabel="OOF RMSE", title=f"Held-out pack errors at {best_k} features")
    ax.grid(axis="y", alpha=.25); ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT / f"Figure_RMSE_by_pack.{suffix}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)


def main():
    started = time.time()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(INPUT)
    raw = raw[raw.analysis_eligible_primary.astype(bool)].copy()
    primary = prepare(raw, "regime_normalised_SOH")
    sensitivity = prepare(raw, "nominal_capacity_SOH")
    source = primary[primary.unit_type == "cell"].copy()
    target = primary[primary.unit_type == "pack"].copy()
    source_s = sensitivity[sensitivity.unit_type == "cell"].copy()
    target_s = sensitivity[sensitivity.unit_type == "pack"].copy()
    packs = sorted(target.file_number.unique())

    fold_rows, prediction_rows, ranks, timing = [], [], [], []
    sensitivity_folds, sensitivity_predictions = [], []
    split_rows = []
    for fold_no, test_pack in enumerate(packs, 1):
        fold_start = time.time()
        calibration_pack = CALIBRATION_MAP[test_pack]
        fit_packs = sorted(set(packs) - {test_pack, calibration_pack})
        fit = target[target.file_number.isin(fit_packs)].copy()
        calibration = target[target.file_number == calibration_pack].copy()
        test = target[target.file_number == test_pack].copy()
        fit_s = target_s[target_s.file_number.isin(fit_packs)].copy()
        calibration_s = target_s[target_s.file_number == calibration_pack].copy()
        test_s = target_s[target_s.file_number == test_pack].copy()
        ranking = fusion_ranking(fit, SEED + fold_no)
        ranking.insert(0, "test_pack", test_pack); ranks.append(ranking)
        ordered = ranking.feature.tolist()
        split_rows.append({
            "test_pack": test_pack, "calibration_pack": calibration_pack,
            "fitting_packs": "|".join(fit_packs),
            "n_fit_packs": len(fit_packs),
            "test_group_size": int(test.group_size.iloc[0]),
            "test_operating_mode": test.operating_mode.iloc[0],
        })

        for k in FEATURE_COUNTS:
            group_start = time.time(); selected = ordered[:k]
            target_low, target_high = limits(fit, selected)
            source_low, source_high = limits(source, selected)
            xf, xc, xt = (clipped(z, selected, target_low, target_high)
                          for z in (fit, calibration, test))
            xs = clipped(source, selected, source_low, source_high)
            xc_source, xt_source = (
                clipped(z, selected, source_low, source_high)
                for z in (calibration, test))
            pooled_x = pd.concat([xs, xf], ignore_index=True)
            pooled_y = pd.concat([source.soh, fit.soh], ignore_index=True)
            cal_members, test_members = train_primary(
                xs, xf, xc, xt, xc_source, xt_source,
                source.soh, fit.soh, pooled_x, pooled_y,
                SEED + fold_no * 10_000 + k * 100,
            )
            for setup in SETUPS:
                stats, arrays = ensemble_statistics(
                    test.soh.to_numpy(), test_members[setup],
                    calibration.soh.to_numpy(), cal_members[setup])
                fold_rows.append({
                    "test_pack": test_pack, "calibration_pack": calibration_pack,
                    "n_features": k, "setup": setup,
                    "features": "|".join(selected), "n_fit_packs": len(fit_packs),
                    "n_calibration_packs": 1, "n_ensemble": N_ENSEMBLE,
                    "test_group_size": int(test.group_size.iloc[0]),
                    "test_operating_mode": test.operating_mode.iloc[0], **stats,
                })
                append_predictions(
                    prediction_rows, test, k, setup, stats, arrays,
                    "regime_normalised_SOH")

            # Same fold ranking and preprocessing; replace only the supervised target.
            xs_s = source_s.loc[source.index, selected]
            xf_s = fit_s.loc[fit.index, selected]
            xc_s = calibration_s.loc[calibration.index, selected]
            xt_s = test_s.loc[test.index, selected]
            xs_s = xs_s.clip(lower=source_low, upper=source_high, axis=1)
            xf_s = xf_s.clip(lower=target_low, upper=target_high, axis=1)
            xc_s = xc_s.clip(lower=target_low, upper=target_high, axis=1)
            xt_s = xt_s.clip(lower=target_low, upper=target_high, axis=1)
            scal, stest = train_sensitivity(
                xs_s, xf_s, xc_s, xt_s, source_s.soh, fit_s.soh,
                SEED + 500_000 + fold_no * 10_000 + k * 100)
            for setup in SENSITIVITY_SETUPS:
                stats, arrays = ensemble_statistics(
                    test_s.soh.to_numpy(), stest[setup],
                    calibration_s.soh.to_numpy(), scal[setup])
                sensitivity_folds.append({
                    "test_pack": test_pack, "calibration_pack": calibration_pack,
                    "n_features": k, "setup": setup,
                    "features": "|".join(selected), **stats,
                })
                append_predictions(
                    sensitivity_predictions, test_s, k, setup, stats, arrays,
                    "nominal_capacity_SOH")
            timing.append({"test_pack": test_pack, "n_features": k,
                           "seconds_all_methods": time.time() - group_start})
        timing.append({"test_pack": test_pack, "n_features": "ALL",
                       "seconds_all_methods": time.time() - fold_start})
        print(f"[{fold_no}/6] held-out {test_pack} complete", flush=True)

    folds = pd.DataFrame(fold_rows)
    predictions = pd.DataFrame(prediction_rows)
    rankings = pd.concat(ranks, ignore_index=True)
    s_folds = pd.DataFrame(sensitivity_folds)
    s_predictions = pd.DataFrame(sensitivity_predictions)
    metric_columns = [
        "rmse", "mae", "r2", "mean_gaussian_nll", "mean_epistemic_variance",
        "calibration_mse", "residual_noise_variance",
        "mean_total_predictive_variance", "picp_95", "coverage_error_95", "mpiw_95",
    ]
    summary = folds.groupby(["n_features", "setup"])[metric_columns].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    pooled = aggregate_metrics(predictions, ["n_features", "setup"])
    conditional = aggregate_metrics(
        predictions,
        ["n_features", "setup", "group_size", "operating_mode", "life_stage"],
    )
    by_pack = aggregate_metrics(predictions, ["n_features", "setup", "test_pack"])
    by_regime = aggregate_metrics(
        predictions, ["n_features", "setup", "final_discharge_C_rate"])
    s_summary = s_folds.groupby(["n_features", "setup"])[metric_columns].agg(["mean", "std"])
    s_summary.columns = [f"{a}_{b}" for a, b in s_summary.columns]
    s_summary = s_summary.reset_index()
    s_pooled = aggregate_metrics(s_predictions, ["n_features", "setup"])

    outputs = {
        "fold_metrics_all_feature_groups.csv": folds,
        "out_of_fold_predictions_with_intervals.csv": predictions,
        "nested_feature_rankings.csv": rankings,
        "summary_metrics_by_feature_count.csv": summary,
        "pooled_oof_metrics_by_feature_count.csv": pooled,
        "metrics_by_pack.csv": by_pack,
        "metrics_by_pack_size_mode_life_stage.csv": conditional,
        "metrics_by_discharge_regime.csv": by_regime,
        "paired_transfer_comparison_by_feature_count.csv": paired_tests(folds),
        "nominal_soh_sensitivity_fold_metrics.csv": s_folds,
        "nominal_soh_sensitivity_oof_predictions.csv": s_predictions,
        "nominal_soh_sensitivity_summary.csv": s_summary,
        "nominal_soh_sensitivity_pooled_oof.csv": s_pooled,
        "leave_one_pack_out_splits.csv": pd.DataFrame(split_rows),
        "training_times.csv": pd.DataFrame(timing),
    }
    for filename, frame in outputs.items():
        frame.to_csv(OUTPUT / filename, index=False)

    metadata = {
        "direction": "Wenzhou individual cells to Wenzhou parallel packs",
        "source_units": sorted(source.file_number.unique().tolist()),
        "target_packs": packs,
        "outer_validation": "leave-one-complete-pack-out",
        "calibration_map": CALIBRATION_MAP,
        "features": FEATURES,
        "feature_counts": FEATURE_COUNTS,
        "primary_target": "regime_normalised_SOH",
        "primary_target_boundary": "median of first five capacity labels within each unit and discharge C-rate regime; disclosed early-life target initialization",
        "sensitivity_target": "nominal_capacity_SOH",
        "sensitivity_target_definition": "capacity / known nominal capacity; no test-derived reference",
        "setups": SETUPS,
        "gated_transfer": False,
        "ensemble_members": N_ENSEMBLE,
        "total_trees": ROUNDS,
        "continuation_trees": {"source": SOURCE_ROUNDS, "target": TARGET_ROUNDS},
        "xgboost": {"learning_rate": .10, "max_depth": 4,
                     "min_child_weight": 3.0, "subsample": .90,
                     "colsample_bytree": .90, "reg_lambda": 5.0,
                     "reg_alpha": 0.0, "tree_method": "hist"},
        "base_seed": SEED,
        "clipping": "0.1 and 99.9 percentiles fitted independently on source or target fitting domain",
        "test_label_boundary": "Test labels evaluate metrics only; they do not rank features, fit models, estimate residual variance, or construct intervals. Primary target normalization requires disclosed early-life same-regime capacity initialization.",
        "runtime_seconds": time.time() - started,
    }
    (OUTPUT / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    plot_results(summary, rankings, predictions)
    print(f"Completed in {(time.time() - started)/60:.1f} min: {OUTPUT}")


if __name__ == "__main__":
    main()
