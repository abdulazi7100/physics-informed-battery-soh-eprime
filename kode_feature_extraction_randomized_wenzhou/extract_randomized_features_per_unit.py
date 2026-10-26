"""Extract Wenzhou Randomized checkpoint features into one CSV per unit.

This script is intentionally separate from the modelling pipeline. It rereads the
processed MAT files, applies the same documented checkpoint reconstruction used by
the reviewer experiment, writes per-unit and combined tables, and creates dataset
visualisations. Derived files are for local analysis and must not be redistributed
contrary to the source dataset's CC BY-ND 4.0 terms.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
KODE_PYTHON = PROJECT / "Kode_python"
sys.path.insert(0, str(KODE_PYTHON))

from R1_C3__R2_C1_C2_C3_C4_C5__Wenzhou_Randomized_cumulative_lifetime_XGBoost_deep_ensemble import (  # noqa: E402
    BASE_FEATURES,
    DATA_ROOT,
    extract_one,
)

OUTPUT = ROOT / "extracted_features_checkpoint_per_unit"
PER_UNIT = OUTPUT / "per_unit_csv"
FIGURES = OUTPUT / "figures"
SOH_MIN, SOH_MAX = 0.35, 1.35

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))
import matplotlib.pyplot as plt  # noqa: E402


def save_figure(fig, stem):
    fig.tight_layout()
    fig.savefig(FIGURES / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def extract_all():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    PER_UNIT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    frames, audit = [], []
    for family in ["1000mAh", "800mAh"]:
        folder = DATA_ROOT / f"Data_{family}_Copy"
        for unit_dir in sorted(folder.glob("Dongzhen_Lyu_*")):
            number = int(unit_dir.name.rsplit("_", 1)[-1])
            unit_id = f"{family}_{number:03d}"
            mats = list(unit_dir.glob("*Detail_Final.mat"))
            if not mats:
                audit.append({"unit_id": unit_id, "capacity_family": family,
                              "status": "excluded", "reason": "MAT file unavailable",
                              "n_rows_before_soh_filter": 0, "n_rows_saved": 0})
                continue
            frame = extract_one(mats[0], family, number)
            before = len(frame)
            if before:
                frame = frame[np.isfinite(frame.soh) & frame.soh.between(SOH_MIN, SOH_MAX)].copy()
            if len(frame):
                frame.to_csv(PER_UNIT / f"{unit_id}_checkpoint_features.csv", index=False)
                frames.append(frame)
                audit.append({"unit_id": unit_id, "capacity_family": family,
                              "status": "included", "reason": "",
                              "n_rows_before_soh_filter": before,
                              "n_rows_saved": len(frame)})
            else:
                audit.append({"unit_id": unit_id, "capacity_family": family,
                              "status": "excluded",
                              "reason": "fewer than five reliable full-like 1C checkpoints",
                              "n_rows_before_soh_filter": before, "n_rows_saved": 0})
    combined = pd.concat(frames, ignore_index=True)
    audit = pd.DataFrame(audit)
    combined.to_csv(OUTPUT / "Wenzhou_Randomized_checkpoint_features_combined.csv", index=False)
    audit.to_csv(OUTPUT / "extraction_audit.csv", index=False)
    return combined, audit


def dataset_summary(data, audit):
    unit = (data.groupby("unit_id", as_index=False)
            .agg(capacity_family=("capacity_family", "first"),
                 unit_type=("unit_type", "first"),
                 pack_size=("pack_size", "first"),
                 operation_stratum=("operation_stratum", "first"),
                 n_checkpoints=("soh", "size"),
                 initial_capacity_Ah=("initial_checkpoint_capacity_Ah", "first"),
                 final_observed_soh=("soh", "last"),
                 max_event_count=("event_count", "max"),
                 cumulative_discharge_Ah=("cumulative_discharge_Ah", "max"),
                 cumulative_discharge_Wh=("cumulative_discharge_Wh", "max"),
                 equivalent_full_cycles=("equivalent_full_cycles", "max"),
                 elapsed_days=("elapsed_days", "max"),
                 checkpoint_fraction=("checkpoint_detection_fraction", "first")))
    unit.to_csv(OUTPUT / "unit_level_summary.csv", index=False)
    feature_summary = data[BASE_FEATURES + [
        "event_count", "cumulative_discharge_Ah", "cumulative_discharge_Wh",
        "equivalent_full_cycles", "elapsed_days", "soh"]].describe().T
    feature_summary.to_csv(OUTPUT / "feature_descriptive_statistics.csv")
    metadata = {
        "source": "Wenzhou Randomized Battery Data",
        "source_processed_mat_root": str(DATA_ROOT),
        "total_units_scanned": int(len(audit)),
        "included_units": int((audit.status == "included").sum()),
        "excluded_units": int((audit.status == "excluded").sum()),
        "checkpoint_rows": int(len(data)),
        "per_unit_csv_files": int(len(list(PER_UNIT.glob("*.csv")))),
        "soh_filter": [SOH_MIN, SOH_MAX],
        "base_features": BASE_FEATURES,
        "lifetime_coordinates": ["event_count", "cumulative_discharge_Ah",
                                 "cumulative_discharge_Wh",
                                 "equivalent_full_cycles", "elapsed_days"],
        "license_note": "Local derived analysis only; do not redistribute contrary to CC BY-ND 4.0.",
    }
    (OUTPUT / "extraction_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    return unit, metadata


def make_plots(data, unit):
    # 1. Composition of retained units.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, column, title in zip(
            axes, ["capacity_family", "unit_type", "operation_stratum"],
            ["Capacity family", "Unit type", "Data-derived operating heterogeneity"]):
        counts = unit[column].value_counts()
        ax.bar(counts.index.astype(str), counts.values, color="#4C78A8")
        ax.set_title(title); ax.set_ylabel("Number of units")
        ax.tick_params(axis="x", rotation=25); ax.grid(axis="y", alpha=.25)
    save_figure(fig, "Figure_01_dataset_composition")

    # 2. Number of reconstructed checkpoints and test duration.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for stratum, block in unit.groupby("operation_stratum"):
        axes[0].scatter(block.elapsed_days, block.n_checkpoints, s=22, alpha=.70, label=stratum)
        axes[1].scatter(block.max_event_count, block.checkpoint_fraction, s=22, alpha=.70, label=stratum)
    axes[0].set(xlabel="Elapsed test time (days)", ylabel="Reconstructed checkpoints",
                title="Checkpoint availability by test duration")
    axes[1].set(xlabel="Last event index", ylabel="Checkpoint fraction",
                title="Fraction of events identified as checkpoints")
    for ax in axes: ax.grid(alpha=.25)
    axes[0].legend(frameon=False, fontsize=8)
    save_figure(fig, "Figure_02_checkpoint_availability")

    # 3. SOH trajectories for representative units, identical observations shown
    # against all lifetime coordinates.
    sample = (unit.sort_values("n_checkpoints", ascending=False)
              .groupby(["capacity_family", "operation_stratum"], group_keys=False)
              .head(2).unit_id.tolist())
    coords = [("event_count", "Event count"),
              ("cumulative_discharge_Ah", "Cumulative discharge (Ah)"),
              ("cumulative_discharge_Wh", "Cumulative discharge (Wh)"),
              ("equivalent_full_cycles", "Equivalent full cycles"),
              ("elapsed_days", "Elapsed time (days)")]
    fig, axes = plt.subplots(3, 2, figsize=(14, 13)); axes = axes.ravel()
    for ax, (column, label) in zip(axes, coords):
        for uid in sample:
            block = data[data.unit_id == uid].sort_values(column)
            ax.plot(block[column], block.soh, linewidth=.9, alpha=.75, label=uid)
        ax.set(xlabel=label, ylabel="Checkpoint SOH"); ax.grid(alpha=.22)
    axes[-1].axis("off")
    axes[0].legend(frameon=False, fontsize=6, ncol=2)
    fig.suptitle("The same SOH observations under alternative lifetime coordinates", y=.995)
    save_figure(fig, "Figure_03_SOH_by_lifetime_coordinate")

    # 4. Operating-history descriptors retained at checkpoints.
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    plots = [("prior_discharge_depth", "Prior discharge depth relative to early-life reference", False),
             ("prior_discharge_c_rate", "Prior discharge C-rate", False),
             ("charge_c_rate", "Preceding charge C-rate", False),
             ("cumulative_rest_h", "log(1 + cumulative rest time [h])", True)]
    colors = {"regular_full_like": "#54A24B", "variable_complex_like": "#F2CF5B",
              "highly_partial_random_like": "#E45756"}
    for ax, (column, label, log_transform) in zip(axes.ravel(), plots):
        for stratum, block in data.groupby("operation_stratum"):
            values = block[column].replace([np.inf, -np.inf], np.nan).dropna()
            if len(values) > 10000: values = values.sample(10000, random_state=42)
            if log_transform: values = np.log1p(values.clip(lower=0))
            ax.hist(values, bins=60, density=True, alpha=.38,
                    color=colors.get(stratum), label=stratum)
        ax.set(xlabel=label, ylabel="Density"); ax.grid(alpha=.2)
    axes[0, 0].legend(frameon=False, fontsize=8)
    save_figure(fig, "Figure_04_operating_history_distributions")

    # 5. Feature correlation map for a bounded reproducible sample.
    columns = BASE_FEATURES + ["event_count", "cumulative_discharge_Ah",
                               "cumulative_discharge_Wh", "equivalent_full_cycles",
                               "elapsed_days", "soh"]
    sample_data = data[columns].sample(min(30000, len(data)), random_state=42)
    corr = sample_data.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(13, 11))
    image = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
    ax.set_xticks(range(len(columns)), columns, rotation=75, ha="right", fontsize=8)
    ax.set_yticks(range(len(columns)), columns, fontsize=8)
    ax.set_title("Spearman correlation of extracted checkpoint variables")
    fig.colorbar(image, ax=ax, shrink=.8, label="Spearman correlation")
    save_figure(fig, "Figure_05_feature_correlation")


def write_readme(metadata):
    text = f"""# Wenzhou Randomized per-unit feature extraction

This folder contains locally derived checkpoint-level features for
{metadata['included_units']} of {metadata['total_units_scanned']} scanned battery
units ({metadata['checkpoint_rows']:,} rows). One CSV is stored for each included
unit in `per_unit_csv/`; the same rows are also available in the combined table.

## Main files

- `per_unit_csv/`: {metadata['per_unit_csv_files']} separate unit CSV files.
- `Wenzhou_Randomized_checkpoint_features_combined.csv`: combined local table.
- `extraction_audit.csv`: included and excluded units with reasons.
- `unit_level_summary.csv`: one-row-per-unit summary.
- `feature_descriptive_statistics.csv`: feature distributions.
- `extraction_metadata.json`: extraction settings and feature list.
- `figures/`: PNG and PDF exploratory figures.

The target `soh` is derived only at reconstructed full-like 1C benchmark
checkpoints. Partial-event capacity is not interpreted directly as SOH. The
processed source MAT files do not contain an explicit checkpoint flag, so the rule
is a documented reconstruction rather than an author-supplied annotation.

The source dataset uses CC BY-ND 4.0 terms. These derived CSV files are intended
for local analysis and should not be redistributed without confirming permission
from the dataset owner.
"""
    (OUTPUT / "README.md").write_text(text, encoding="utf-8")


def main():
    data, audit = extract_all()
    unit, metadata = dataset_summary(data, audit)
    make_plots(data, unit)
    write_readme(metadata)
    print(f"Saved {metadata['per_unit_csv_files']} per-unit files and "
          f"{metadata['checkpoint_rows']:,} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
