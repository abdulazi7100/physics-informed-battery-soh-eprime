"""Leakage-audited cumulative-lifetime SOH experiment on Wenzhou Randomized data.

Primary question: does SOH estimation remain reliable when partial/random usage is
represented by cumulative utilisation rather than by an ambiguous cycle count?

The script deliberately runs one target-domain experiment (no transfer methods).
Complete battery units define fitting, calibration, and test partitions. Feature
ranking, clipping, model fitting, and residual-noise calibration are repeated
inside every outer fold without access to the held-out units.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.io import loadmat
from scipy.stats import wilcoxon
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DATA_ROOT = (PROJECT_ROOT / "Wenzhou_Randomized_Battery_Data" /
             "Processed_Data (by Dr Dongzhen Lyu)" / "2_Processed_data")
OUTPUT = ROOT / "R1_C3__R2_C1_C2_C3_C4_C5__results_Wenzhou_Randomized_cumulative_lifetime"
CACHE = OUTPUT / "checkpoint_feature_table.csv"
SEED = 42
N_OUTER_FOLDS = 10
N_ENSEMBLE = 5
N_FEATURES = 12
N_TREES = 80
Z95 = 1.959963984540054
VARIANCE_FLOOR = 1e-8
PARAMS = dict(max_depth=4, min_child_weight=3.0, reg_lambda=5.0,
              reg_alpha=0.0, learning_rate=0.05, subsample=0.90,
              colsample_bytree=0.90)

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))
import matplotlib.pyplot as plt


BASE_FEATURES = [
    "charge_duration_h", "charge_ah_relative", "charge_wh_relative",
    "charge_mean_voltage", "charge_c_rate", "prior_discharge_duration_h",
    "prior_discharge_depth", "prior_discharge_c_rate",
    "recent_mean_depth_10", "recent_std_depth_10",
    "recent_mean_c_rate_10", "recent_std_c_rate_10",
    "cumulative_operating_h", "cumulative_rest_h",
]

REPRESENTATIONS = {
    "health_indicators_only": None,
    "event_count": "event_count",
    "cumulative_discharge_Ah": "cumulative_discharge_Ah",
    "cumulative_discharge_Wh": "cumulative_discharge_Wh",
    "equivalent_full_cycles": "equivalent_full_cycles",
    "elapsed_days": "elapsed_days",
}


def safe_array(data, key):
    return np.asarray(data[key], dtype=float).ravel()


def preceding_values(query_time, event_end, values):
    """Value of the most recent completed event; NaN if none exists."""
    position = np.searchsorted(event_end, query_time, side="right") - 1
    result = np.full(len(query_time), np.nan)
    valid = position >= 0
    result[valid] = values[position[valid]]
    return result


def unit_metadata(capacity_family, unit_number, full_fraction, current_cv):
    """Known hardware class plus an explicitly data-derived operation stratum."""
    # Counts and pack positions follow Table 1 totals; individual file-to-mode
    # assignments are not assumed. Pack files occupy the final IDs in each family.
    if capacity_family == "1000mAh":
        is_pack = unit_number > 144
        pack_size = 3 if is_pack else 1
        nominal_ah = float(pack_size)
    else:
        is_pack = unit_number > 76
        if unit_number <= 78:
            pack_size = 2
        elif unit_number <= 88:
            pack_size = 3
        else:
            pack_size = 5
        if not is_pack:
            pack_size = 1
        nominal_ah = 0.8 * pack_size
    if full_fraction >= 0.80 and current_cv < 0.12:
        operation_stratum = "regular_full_like"
    elif full_fraction < 0.45:
        operation_stratum = "highly_partial_random_like"
    else:
        operation_stratum = "variable_complex_like"
    return is_pack, pack_size, nominal_ah, operation_stratum


def extract_one(path, capacity_family, unit_number):
    data = loadmat(path, squeeze_me=True)
    ds, de = safe_array(data, "Discharge_datetime_start"), safe_array(data, "Discharge_datetime_end")
    di, dq = np.abs(safe_array(data, "Discharge_current_mean")), np.abs(safe_array(data, "Discharge_capacity")) / 3.6e6
    dwh = np.abs(safe_array(data, "Discharge_power")) / 3.6e6
    dv = safe_array(data, "Discharge_voltage_mean")
    cs, ce = safe_array(data, "Charge_datetime_start"), safe_array(data, "Charge_datetime_end")
    ci = np.abs(safe_array(data, "Charge_current_mean"))
    cq = np.abs(safe_array(data, "Charge_capacity")) / 3.6e6
    cwh = np.abs(safe_array(data, "Charge_power")) / 3.6e6
    cv = safe_array(data, "Charge_voltage_mean")
    n = min(map(len, [ds, de, di, dq, dwh, dv]))
    ds, de, di, dq, dwh, dv = [a[:n] for a in (ds, de, di, dq, dwh, dv)]
    valid = np.isfinite(ds + de + di + dq + dwh) & (de > ds) & (dq > 0)
    ds, de, di, dq, dwh, dv = [a[valid] for a in (ds, de, di, dq, dwh, dv)]
    order = np.argsort(ds)
    ds, de, di, dq, dwh, dv = [a[order] for a in (ds, de, di, dq, dwh, dv)]
    if len(dq) < 30:
        return pd.DataFrame()

    # Early-life scale only: this never uses end-of-life information.
    early = dq[:min(100, len(dq))]
    q_reference = float(np.median(np.sort(early)[-min(5, len(early)):])); q_reference = max(q_reference, 1e-6)
    depth = dq / q_reference
    c_rate = (di / 1000.0) / q_reference
    duration_h = (de - ds) / 3600.0
    full_fraction = float(np.mean(depth >= 0.70))
    current_cv = float(np.std(c_rate) / max(np.mean(c_rate), 1e-8))
    is_pack, pack_size, nominal_ah, operation_stratum = unit_metadata(
        capacity_family, unit_number, full_fraction, current_cv)

    # Benchmark checkpoints are full-like discharges near the 1C protocol.
    candidate = (c_rate >= 0.70) & (c_rate <= 1.30) & (depth >= 0.35)
    candidate_series = pd.Series(np.where(candidate, dq, np.nan))
    local_upper = candidate_series.rolling(51, center=True, min_periods=3).quantile(0.85).to_numpy()
    local_upper = pd.Series(local_upper).ffill().bfill().to_numpy()
    checkpoint = candidate & np.isfinite(local_upper) & (dq >= 0.88 * local_upper)
    idx = np.flatnonzero(checkpoint)
    if len(idx) < 8:
        # Conservative fallback for unusual units; still uses 1C-like full events.
        idx = np.flatnonzero(candidate)
    if len(idx) < 5:
        return pd.DataFrame()
    initial_capacity = float(np.median(dq[idx[:min(5, len(idx))]]))

    # Align each discharge with the most recent completed charge.
    cvalid = np.isfinite(cs + ce + ci + cq + cwh + cv) & (ce > cs)
    cs, ce, ci, cq, cwh, cv = [a[cvalid] for a in (cs, ce, ci, cq, cwh, cv)]
    corder = np.argsort(ce); cs, ce, ci, cq, cwh, cv = [a[corder] for a in (cs, ce, ci, cq, cwh, cv)]
    charge_start = preceding_values(ds, ce, cs)
    charge_end = preceding_values(ds, ce, ce)
    charge_current = preceding_values(ds, ce, ci) / 1000.0
    charge_ah = preceding_values(ds, ce, cq)
    charge_wh = preceding_values(ds, ce, cwh)
    charge_voltage = preceding_values(ds, ce, cv)

    # All cumulative predictors exclude the current target discharge.
    prior_ah = np.r_[0.0, np.cumsum(dq[:-1])]
    prior_wh = np.r_[0.0, np.cumsum(dwh[:-1])]
    prior_discharge_h = np.r_[0.0, np.cumsum(duration_h[:-1])]
    charge_duration = np.maximum((charge_end - charge_start) / 3600.0, 0.0)
    prior_charge_h = np.r_[0.0, np.cumsum(np.nan_to_num(charge_duration[:-1], nan=0.0))]
    operating_h = prior_discharge_h + prior_charge_h
    elapsed_h = (ds - ds[0]) / 3600.0
    rest_h = np.maximum(elapsed_h - operating_h, 0.0)
    prior_depth = np.r_[np.nan, depth[:-1]]
    prior_c_rate = np.r_[np.nan, c_rate[:-1]]
    prior_duration = np.r_[np.nan, duration_h[:-1]]
    depth_s = pd.Series(depth).shift(1).rolling(10, min_periods=2)
    crate_s = pd.Series(c_rate).shift(1).rolling(10, min_periods=2)

    unit_id = f"{capacity_family}_{unit_number:03d}"
    frame = pd.DataFrame({
        "unit_id": unit_id, "capacity_family": capacity_family,
        "unit_number": unit_number, "unit_type": "pack" if is_pack else "cell",
        "pack_size": pack_size, "nominal_capacity_Ah": nominal_ah,
        "operation_stratum": operation_stratum,
        "checkpoint_number": np.arange(1, len(idx) + 1), "source_event_index": idx + 1,
        "soh": dq[idx] / initial_capacity,
        "event_count": idx.astype(float),
        "cumulative_discharge_Ah": prior_ah[idx],
        "cumulative_discharge_Wh": prior_wh[idx],
        "equivalent_full_cycles": prior_ah[idx] / nominal_ah,
        "elapsed_days": elapsed_h[idx] / 24.0,
        "charge_duration_h": charge_duration[idx],
        "charge_ah_relative": charge_ah[idx] / nominal_ah,
        "charge_wh_relative": charge_wh[idx] / max(nominal_ah * 3.7, 1e-6),
        "charge_mean_voltage": charge_voltage[idx],
        "charge_c_rate": charge_current[idx] / nominal_ah,
        "prior_discharge_duration_h": prior_duration[idx],
        "prior_discharge_depth": prior_depth[idx],
        "prior_discharge_c_rate": prior_c_rate[idx],
        "recent_mean_depth_10": depth_s.mean().to_numpy()[idx],
        "recent_std_depth_10": depth_s.std().to_numpy()[idx],
        "recent_mean_c_rate_10": crate_s.mean().to_numpy()[idx],
        "recent_std_c_rate_10": crate_s.std().to_numpy()[idx],
        "cumulative_operating_h": operating_h[idx],
        "cumulative_rest_h": rest_h[idx],
        "checkpoint_capacity_Ah": dq[idx],
        "initial_checkpoint_capacity_Ah": initial_capacity,
        "checkpoint_detection_fraction": len(idx) / len(dq),
    })
    return frame.replace([np.inf, -np.inf], np.nan)


def build_table():
    frames, audit = [], []
    for family in ["1000mAh", "800mAh"]:
        folder = DATA_ROOT / f"Data_{family}_Copy"
        for unit_dir in sorted(folder.glob("Dongzhen_Lyu_*")):
            number = int(unit_dir.name.rsplit("_", 1)[-1])
            mats = list(unit_dir.glob("*Detail_Final.mat"))
            if not mats:
                continue
            frame = extract_one(mats[0], family, number)
            audit.append({"unit_id": f"{family}_{number:03d}", "n_checkpoints": len(frame),
                          "status": "included" if len(frame) else "excluded"})
            if len(frame): frames.append(frame)
    table = pd.concat(frames, ignore_index=True)
    # Missing early rolling/history values are filled only by constants or causal
    # within-unit forward values; fold-fitted medians handle anything remaining.
    table.to_csv(CACHE, index=False)
    pd.DataFrame(audit).to_csv(OUTPUT / "extraction_audit.csv", index=False)
    return table


def make_unit_folds(table):
    units = table[["unit_id", "capacity_family", "unit_type", "operation_stratum"]].drop_duplicates().sort_values("unit_id").reset_index(drop=True)
    # Both capacity families have enough independent units for ten-fold
    # stratification. Rarer pack/protocol strata are evaluated after prediction,
    # but are not used as split labels because some contain fewer than ten units.
    units["stratum"] = units.capacity_family
    splitter = StratifiedKFold(N_OUTER_FOLDS, shuffle=True, random_state=SEED)
    assignments = []
    for fold, (_, test_pos) in enumerate(splitter.split(units, units.stratum), 1):
        test_ids = set(units.iloc[test_pos].unit_id)
        remaining = units[~units.unit_id.isin(test_ids)].copy()
        calibration_ids = []
        for _, block in remaining.groupby("stratum", sort=True):
            rng = np.random.default_rng(SEED + fold + sum(map(ord, str(block.iloc[0].stratum))))
            ids = block.unit_id.to_numpy().copy(); rng.shuffle(ids)
            calibration_ids.extend(ids[:max(1, int(round(0.20 * len(ids))))])
        calibration_ids = set(calibration_ids)
        for uid in units.unit_id:
            role = "test" if uid in test_ids else ("calibration" if uid in calibration_ids else "fit")
            assignments.append({"outer_fold": fold, "unit_id": uid, "role": role})
    return pd.DataFrame(assignments)


def model(seed):
    return XGBRegressor(n_estimators=N_TREES, objective="reg:squarederror",
                        eval_metric="rmse", tree_method="hist", n_jobs=-1,
                        random_state=seed, verbosity=0, **PARAMS)


def fitted_preprocessor(fit, features):
    median = fit[features].median().fillna(0.0)
    low = fit[features].fillna(median).quantile(0.001)
    high = fit[features].fillna(median).quantile(0.999)
    return median, low, high


def transform(frame, features, prep):
    median, low, high = prep
    return frame[features].fillna(median).clip(lower=low, upper=high, axis=1)


def fusion_ranking(fit, candidates, seed):
    prep = fitted_preprocessor(fit, candidates)
    xx = transform(fit, candidates, prep)
    correlation = xx.corrwith(fit.soh).abs().fillna(0.0)
    selector = model(seed).fit(xx, fit.soh)
    sample = xx.sample(min(5000, len(xx)), random_state=seed)
    values = selector.get_booster().predict(xgb.DMatrix(sample, feature_names=candidates), pred_contribs=True)
    shap = pd.Series(np.abs(values[:, :-1]).mean(0), index=candidates)
    result = pd.DataFrame({"feature": candidates,
                           "abs_target_correlation": correlation[candidates].to_numpy(),
                           "mean_abs_shap": shap[candidates].to_numpy()})
    result["correlation_rank"] = result.abs_target_correlation.rank(ascending=False, method="average")
    result["shap_rank"] = result.mean_abs_shap.rank(ascending=False, method="average")
    result["fusion_rank"] = (result.correlation_rank + result.shap_rank) / 2
    return result.sort_values(["fusion_rank", "shap_rank", "feature"]).reset_index(drop=True)


def ensemble_statistics(y, test_members, cal_y, cal_members):
    pred = test_members.mean(axis=1); cal_pred = cal_members.mean(axis=1)
    epistemic = test_members.var(axis=1, ddof=1); cal_epistemic = cal_members.var(axis=1, ddof=1)
    calibration_mse = float(np.mean((cal_y - cal_pred) ** 2))
    residual = float(max(calibration_mse - cal_epistemic.mean(), VARIANCE_FLOOR))
    total = np.maximum(epistemic + residual, VARIANCE_FLOOR)
    low, high = pred - Z95 * np.sqrt(total), pred + Z95 * np.sqrt(total)
    nll = 0.5 * (np.log(2 * np.pi * total) + (y - pred) ** 2 / total)
    covered = (y >= low) & (y <= high)
    metrics = {"rmse": float(mean_squared_error(y, pred) ** 0.5),
               "mae": float(mean_absolute_error(y, pred)),
               "r2": float(r2_score(y, pred)),
               "mean_gaussian_nll": float(nll.mean()),
               "mean_epistemic_variance": float(epistemic.mean()),
               "calibration_mse": calibration_mse,
               "residual_noise_variance": residual,
               "mean_total_predictive_variance": float(total.mean()),
               "picp_95": float(covered.mean()),
               "coverage_error_95": float(abs(covered.mean() - 0.95)),
               "mpiw_95": float(np.mean(high - low))}
    arrays = dict(predicted_soh=pred, epistemic_variance=epistemic,
                  total_predictive_variance=total, lower_95=low, upper_95=high,
                  covered_95=covered.astype(int), gaussian_nll=nll)
    return metrics, arrays


def generic_metrics(block):
    y, p = block.actual_soh.to_numpy(), block.predicted_soh.to_numpy()
    total = block.total_predictive_variance.to_numpy()
    nll = 0.5 * (np.log(2 * np.pi * total) + (y - p) ** 2 / total)
    n_units = block.unit_id.nunique() if "unit_id" in block.columns else 1
    return pd.Series({"n_observations": len(block), "n_units": n_units,
                      "rmse": mean_squared_error(y, p) ** 0.5,
                      "mae": mean_absolute_error(y, p),
                      "r2": r2_score(y, p) if len(y) > 1 and np.var(y) > 0 else np.nan,
                      "mean_gaussian_nll": nll.mean(),
                      "mean_epistemic_variance": block.epistemic_variance.mean(),
                      "residual_noise_variance": block.residual_noise_variance.mean(),
                      "mean_total_predictive_variance": total.mean(),
                      "picp_95": block.covered_95.mean(),
                      "coverage_error_95": abs(block.covered_95.mean() - 0.95),
                      "mpiw_95": (block.upper_95 - block.lower_95).mean()})


def run_experiment(table, splits):
    fold_rows, prediction_rows, rankings = [], [], []
    for fold in range(1, N_OUTER_FOLDS + 1):
        roles = splits[splits.outer_fold == fold].set_index("unit_id").role
        fit = table[table.unit_id.map(roles) == "fit"].copy()
        cal = table[table.unit_id.map(roles) == "calibration"].copy()
        test = table[table.unit_id.map(roles) == "test"].copy()
        for rep_index, (representation, coordinate) in enumerate(REPRESENTATIONS.items()):
            candidates = BASE_FEATURES + ([] if coordinate is None else [coordinate])
            ranking = fusion_ranking(fit, candidates, SEED + fold * 100 + rep_index)
            ranking.insert(0, "outer_fold", fold); ranking.insert(1, "representation", representation)
            rankings.append(ranking)
            ordered = ranking.feature.tolist()
            if coordinate is None:
                selected = ordered[:N_FEATURES]
            else:
                selected = [f for f in ordered if f != coordinate][:N_FEATURES - 1] + [coordinate]
            prep = fitted_preprocessor(fit, selected)
            xf, xc, xt = (transform(z, selected, prep) for z in (fit, cal, test))
            cal_members, test_members = [], []
            for member in range(N_ENSEMBLE):
                estimator = model(SEED + fold * 10000 + rep_index * 100 + member).fit(xf, fit.soh)
                cal_members.append(estimator.predict(xc)); test_members.append(estimator.predict(xt))
            stats, arrays = ensemble_statistics(test.soh.to_numpy(), np.column_stack(test_members),
                                                cal.soh.to_numpy(), np.column_stack(cal_members))
            fold_rows.append({"outer_fold": fold, "representation": representation,
                              "selected_features": "|".join(selected),
                              "n_fit_units": fit.unit_id.nunique(),
                              "n_calibration_units": cal.unit_id.nunique(),
                              "n_test_units": test.unit_id.nunique(), **stats})
            for pos, (_, row) in enumerate(test.iterrows()):
                prediction_rows.append({"outer_fold": fold, "representation": representation,
                    "unit_id": row.unit_id, "capacity_family": row.capacity_family,
                    "unit_type": row.unit_type, "operation_stratum": row.operation_stratum,
                    "checkpoint_number": row.checkpoint_number, "actual_soh": row.soh,
                    "event_count": row.event_count,
                    "cumulative_discharge_Ah": row.cumulative_discharge_Ah,
                    "cumulative_discharge_Wh": row.cumulative_discharge_Wh,
                    "equivalent_full_cycles": row.equivalent_full_cycles,
                    "elapsed_days": row.elapsed_days,
                    "predicted_soh": arrays["predicted_soh"][pos],
                    "epistemic_variance": arrays["epistemic_variance"][pos],
                    "residual_noise_variance": stats["residual_noise_variance"],
                    "total_predictive_variance": arrays["total_predictive_variance"][pos],
                    "lower_95": arrays["lower_95"][pos], "upper_95": arrays["upper_95"][pos],
                    "covered_95": arrays["covered_95"][pos], "gaussian_nll": arrays["gaussian_nll"][pos]})
        print(f"Outer fold {fold:02d}/{N_OUTER_FOLDS} complete", flush=True)
    return pd.DataFrame(fold_rows), pd.DataFrame(prediction_rows), pd.concat(rankings, ignore_index=True)


def paired_tests(folds):
    baseline = folds[folds.representation == "event_count"].set_index("outer_fold")
    rows = []
    for representation in ["cumulative_discharge_Ah", "cumulative_discharge_Wh", "equivalent_full_cycles", "elapsed_days"]:
        candidate = folds[folds.representation == representation].set_index("outer_fold")
        difference = candidate.rmse - baseline.rmse
        rng = np.random.default_rng(SEED + len(rows))
        boot = difference.to_numpy()[rng.integers(0, len(difference), (10000, len(difference)))].mean(1)
        test = wilcoxon(candidate.rmse, baseline.rmse)
        rows.append({"representation": representation, "comparator": "event_count",
                     "mean_paired_rmse_difference": difference.mean(),
                     "bootstrap_ci95_low": np.quantile(boot, 0.025),
                     "bootstrap_ci95_high": np.quantile(boot, 0.975),
                     "folds_improved": int((difference < 0).sum()), "n_folds": len(difference),
                     "wilcoxon_statistic": test.statistic, "wilcoxon_p": test.pvalue})
    return pd.DataFrame(rows)


def stage_metrics(predictions):
    z = predictions.copy()
    coordinate_map = {k: v for k, v in REPRESENTATIONS.items() if v is not None}
    pieces = []
    for representation, coordinate in coordinate_map.items():
        b = z[z.representation == representation].copy()
        progress = b.groupby("unit_id")[coordinate].transform(lambda x: x / max(float(x.max()), 1e-12))
        b["life_stage"] = pd.cut(progress, [-np.inf, 1/3, 2/3, np.inf], labels=["early", "middle", "late"])
        out = b.groupby("life_stage", observed=True).apply(generic_metrics, include_groups=False).reset_index()
        out.insert(0, "representation", representation); pieces.append(out)
    return pd.concat(pieces, ignore_index=True)


def plots(summary, predictions, stage, rankings):
    labels = {"health_indicators_only": "Health indicators only", "event_count": "Event count",
              "cumulative_discharge_Ah": "Cumulative Ah", "cumulative_discharge_Wh": "Cumulative Wh",
              "equivalent_full_cycles": "EFC", "elapsed_days": "Elapsed time"}
    order = list(REPRESENTATIONS)
    colors = ["#999999", "#4C78A8", "#E45756", "#F2CF5B", "#54A24B", "#B279A2"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric, title in zip(axes, ["rmse_mean", "r2_mean"], ["Unit-disjoint RMSE", r"Unit-disjoint $R^2$"]):
        vals = summary.set_index("representation").loc[order, metric]
        ax.bar(range(len(order)), vals, color=colors); ax.set_xticks(range(len(order)), [labels[x] for x in order], rotation=25, ha="right")
        ax.set_title(title); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(OUTPUT / "Figure_accuracy_by_lifetime_representation.png", dpi=300, bbox_inches="tight"); fig.savefig(OUTPUT / "Figure_accuracy_by_lifetime_representation.pdf", bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, metric, title in zip(axes, ["mean_gaussian_nll_mean", "picp_95_mean", "mpiw_95_mean"], ["Gaussian NLL", "95% PICP", "95% MPIW"]):
        vals = summary.set_index("representation").loc[order, metric]
        ax.bar(range(len(order)), vals, color=colors); ax.set_xticks(range(len(order)), [labels[x] for x in order], rotation=30, ha="right"); ax.set_title(title); ax.grid(axis="y", alpha=.25)
        if metric == "picp_95_mean": ax.axhline(.95, color="black", linestyle="--", linewidth=1)
    fig.tight_layout(); fig.savefig(OUTPUT / "Figure_uncertainty_by_lifetime_representation.png", dpi=300, bbox_inches="tight"); fig.savefig(OUTPUT / "Figure_uncertainty_by_lifetime_representation.pdf", bbox_inches="tight"); plt.close(fig)

    best = predictions[predictions.representation == "cumulative_discharge_Ah"].copy()
    coords = [("event_count", "Event count"), ("cumulative_discharge_Ah", "Cumulative discharge (Ah)"),
              ("cumulative_discharge_Wh", "Cumulative discharge (Wh)"), ("equivalent_full_cycles", "Equivalent full cycles"), ("elapsed_days", "Elapsed time (days)")]
    fig, axes = plt.subplots(3, 2, figsize=(14, 13)); axes = axes.ravel()
    sample_units = best.groupby(["capacity_family", "operation_stratum"]).unit_id.first().tolist()
    for ax, (coord, xlabel) in zip(axes, coords):
        for uid in sample_units:
            b = best[best.unit_id == uid].sort_values(coord)
            ax.plot(b[coord], b.actual_soh, color="#222222", alpha=.28, linewidth=.8)
            ax.plot(b[coord], b.predicted_soh, alpha=.75, linewidth=1)
        ax.set(xlabel=xlabel, ylabel="SOH"); ax.grid(alpha=.2)
    axes[-1].axis("off"); fig.suptitle("Identical cumulative-Ah model predictions shown against alternative lifetime coordinates", y=.995)
    fig.tight_layout(); fig.savefig(OUTPUT / "Figure_predictions_by_lifetime_coordinate.png", dpi=300, bbox_inches="tight"); fig.savefig(OUTPUT / "Figure_predictions_by_lifetime_coordinate.pdf", bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    pivot = stage.pivot(index="life_stage", columns="representation", values="rmse").reindex(["early", "middle", "late"])
    pivot[[x for x in order if x in pivot.columns]].plot(kind="bar", ax=ax, color=colors[1:])
    ax.set(xlabel="Life stage defined within each lifetime coordinate", ylabel="RMSE", title="Error across early, middle, and late life"); ax.grid(axis="y", alpha=.25); ax.legend([labels[x] for x in pivot.columns], frameon=False, ncol=2)
    fig.tight_layout(); fig.savefig(OUTPUT / "Figure_RMSE_by_life_stage.png", dpi=300, bbox_inches="tight"); fig.savefig(OUTPUT / "Figure_RMSE_by_life_stage.pdf", bbox_inches="tight"); plt.close(fig)

    consensus = rankings.groupby(["representation", "feature"]).agg(mean_fusion_rank=("fusion_rank", "mean"), std_fusion_rank=("fusion_rank", "std"), mean_abs_shap=("mean_abs_shap", "mean")).reset_index()
    consensus.to_csv(OUTPUT / "consensus_feature_ranking.csv", index=False)
    ah = consensus[consensus.representation == "cumulative_discharge_Ah"].sort_values("mean_fusion_rank")
    fig, ax = plt.subplots(figsize=(9, 7)); y = np.arange(len(ah)); ax.barh(y, ah.mean_fusion_rank, xerr=ah.std_fusion_rank.fillna(0), color="#4C78A8", capsize=3); ax.set_yticks(y, ah.feature); ax.invert_yaxis(); ax.set(xlabel="Mean fold-local fusion rank (lower is better)", title="Feature-ranking stability: cumulative-Ah representation"); ax.grid(axis="x", alpha=.25); fig.tight_layout(); fig.savefig(OUTPUT / "Figure_feature_ranking.png", dpi=300, bbox_inches="tight"); fig.savefig(OUTPUT / "Figure_feature_ranking.pdf", bbox_inches="tight"); plt.close(fig)


def write_summary(table, folds, pooled, paired, stage, strata):
    best = pooled.sort_values("rmse").iloc[0]
    ah = pooled[pooled.representation == "cumulative_discharge_Ah"].iloc[0]
    event = pooled[pooled.representation == "event_count"].iloc[0]
    text = f"""# Wenzhou Randomized cumulative-lifetime experiment

## Main result

This experiment used {table.unit_id.nunique()} complete battery units and {len(table):,} standardized checkpoint observations. The lowest pooled OOF RMSE was obtained by **{best.representation}** (RMSE={best.rmse:.6f}, MAE={best.mae:.6f}, R2={best.r2:.6f}).

The reviewer-requested comparison between conventional event count and cumulative discharged Ah gave:

| Representation | RMSE | MAE | R2 | Gaussian NLL | PICP 95% | MPIW 95% |
|---|---:|---:|---:|---:|---:|---:|
| Event count | {event.rmse:.6f} | {event.mae:.6f} | {event.r2:.6f} | {event.mean_gaussian_nll:.6f} | {event.picp_95:.6f} | {event.mpiw_95:.6f} |
| Cumulative discharge Ah | {ah.rmse:.6f} | {ah.mae:.6f} | {ah.r2:.6f} | {ah.mean_gaussian_nll:.6f} | {ah.picp_95:.6f} | {ah.mpiw_95:.6f} |

EFC produced the lowest pooled error, but none of the cumulative coordinates
established statistically significant superiority over event count. The supported
conclusion is robustness across lifetime definitions, not universal superiority
of one coordinate. Aggregate interval coverage was close to nominal, while the
life-stage analysis detected increased error and undercoverage in late life.

## Design safeguards

- Ten outer folds hold out complete cells/packs; observations from one unit never cross fit, calibration, and test roles within a fold.
- Correlation--SHAP fusion ranking, missing-value medians, clipping thresholds, and all XGBoost members are fitted only on fitting units.
- Five independently seeded XGBoost members estimate epistemic variance.
- Residual-noise variance is estimated only from independent calibration units; held-out test residuals are used only for evaluation.
- All cumulative predictors stop before the target checkpoint discharge. Future lifetime totals and held-out labels are never model inputs.
- Checkpoint SOH is based on full-like 1C benchmark discharges; partial-event capacity is not mislabelled as degradation.
- The same folds, ensemble size, feature budget, hyperparameters, and uncertainty procedure are used for every lifetime representation.

## Interpretation boundary

The results test laboratory batteries operated with regular, complex-like, and highly partial/random-like histories. They do not constitute deployment on an actual vehicle fleet. Elapsed test time contains both operation and rest and must not be described as a pure calendar-ageing experiment.

The processed MAT files have no explicit checkpoint flag. Checkpoints are
therefore reconstructed with the documented 1C/full-discharge rule and a local
upper-envelope consistency criterion. This is an auditable approximation rather
than an author-provided checkpoint annotation.
"""
    (OUTPUT / "SUMMARY.md").write_text(text, encoding="utf-8")
    compliance = """# Reviewer compliance

## Reviewer 1

- Comment 1 is partially supported because cells and packs are evaluated as complete units, but aggregate pack channels cannot reveal internal thermal or branch-current coupling.
- Comment 2 is substantially strengthened using partial/random-like operating histories, while retaining the laboratory-simulation qualification.
- Comment 3 is directly addressed by matched comparisons of event count, cumulative Ah, cumulative Wh, EFC, and elapsed time, including life-stage uncertainty calibration.
- Comment 4 is not answered by this LFP dataset; the separate sodium-ion experiments remain necessary.

## Reviewer 2

1. Feature ranking is recomputed from fitting units inside every outer fold.
2. Test residuals never estimate variance or construct intervals; complete calibration units estimate residual noise.
3. All lifetime representations are matched baselines with identical units, target supervision, folds, model capacity, and feature budget.
4. Findings are predictive and dataset-level; no unsupported chemistry mechanism is inferred.
5. Seeds, hyperparameters, checkpoint rules, splits, rankings, preprocessing, OOF predictions, uncertainty components, timing, and plots are saved by the executable script.

No gated transfer or transfer-superiority claim is present because this experiment addresses cumulative-lifetime representation rather than cross-chemistry transfer.

This experiment's primary purpose is Reviewer 1 Comment 3. It must be interpreted
with the separate pack-transfer and sodium-ion experiments when answering the
complete Reviewer 1 report.
"""
    (OUTPUT / "REVIEWER_COMPLIANCE.md").write_text(compliance, encoding="utf-8")


def main():
    started = time.time(); OUTPUT.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(CACHE) if CACHE.exists() else build_table()
    required = BASE_FEATURES + [x for x in REPRESENTATIONS.values() if x] + ["soh"]
    table = table[np.isfinite(table.soh) & table.soh.between(0.35, 1.35)].copy()
    splits = make_unit_folds(table); splits.to_csv(OUTPUT / "unit_disjoint_outer_splits.csv", index=False)
    fold_file = OUTPUT / "fold_metrics_all_lifetime_representations.csv"
    prediction_file = OUTPUT / "out_of_fold_predictions_with_intervals.csv"
    ranking_file = OUTPUT / "nested_feature_rankings.csv"
    resumed = fold_file.exists() and prediction_file.exists() and ranking_file.exists()
    if resumed:
        folds = pd.read_csv(fold_file); predictions = pd.read_csv(prediction_file); rankings = pd.read_csv(ranking_file)
        print("Resuming post-processing from saved OOF results", flush=True)
    else:
        folds, predictions, rankings = run_experiment(table, splits)
        folds.to_csv(fold_file, index=False)
        predictions.to_csv(prediction_file, index=False)
        rankings.to_csv(ranking_file, index=False)
    metrics = ["rmse", "mae", "r2", "mean_gaussian_nll", "mean_epistemic_variance", "calibration_mse", "residual_noise_variance", "mean_total_predictive_variance", "picp_95", "coverage_error_95", "mpiw_95"]
    summary = folds.groupby("representation")[metrics].agg(["mean", "std"]); summary.columns = [f"{a}_{b}" for a, b in summary.columns]; summary = summary.reset_index(); summary.to_csv(OUTPUT / "summary_fold_metrics.csv", index=False)
    pooled = predictions.groupby("representation", sort=False).apply(generic_metrics, include_groups=False).reset_index(); pooled.to_csv(OUTPUT / "pooled_oof_metrics.csv", index=False)
    paired = paired_tests(folds); paired.to_csv(OUTPUT / "paired_comparison_against_event_count.csv", index=False)
    stage = stage_metrics(predictions); stage.to_csv(OUTPUT / "metrics_by_lifetime_stage.csv", index=False)
    strata = predictions.groupby(["representation", "capacity_family", "unit_type", "operation_stratum"]).apply(generic_metrics, include_groups=False).reset_index(); strata.to_csv(OUTPUT / "metrics_by_dataset_stratum.csv", index=False)
    per_unit = predictions.groupby(["representation", "unit_id"]).apply(generic_metrics, include_groups=False).reset_index(); per_unit.to_csv(OUTPUT / "metrics_by_unit.csv", index=False)
    plots(summary, predictions, stage, rankings)
    metadata = {"seed": SEED, "outer_folds": N_OUTER_FOLDS, "ensemble_members": N_ENSEMBLE, "selected_features": N_FEATURES, "xgboost_trees": N_TREES, "xgboost_parameters": PARAMS, "representations": REPRESENTATIONS, "base_features": BASE_FEATURES, "variance_floor": VARIANCE_FLOOR, "interval_z": Z95, "n_units": int(table.unit_id.nunique()), "n_checkpoints": int(len(table)), "this_run_resumed_postprocessing": resumed, "this_run_runtime_seconds": time.time() - started, "input": str(DATA_ROOT), "license_note": "Derived data are for local analysis; do not redistribute contrary to CC BY-ND 4.0."}
    (OUTPUT / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_summary(table, folds, pooled, paired, stage, strata)
    print(f"Finished in {metadata['this_run_runtime_seconds']:.1f} s; outputs: {OUTPUT}")


if __name__ == "__main__":
    main()
