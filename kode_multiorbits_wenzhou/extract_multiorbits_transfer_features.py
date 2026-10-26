"""Extract leakage-auditable charging indicators from Wenzhou Multiorbits.

Fourteen candidate model inputs are extracted from the charge signal. The only
NCA/NMC indicator omitted is Temperature, because Multiorbits has no measured
temperature channel.

Capacity, SOH variants, discharge regime, time, protocol, and throughput are
saved as target/context metadata.  They are not candidate model inputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import kurtosis, skew


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Wenzhou_Multiorbits_dataset"
OUTPUT_DIR = ROOT / "kode_multiorbits_wenzhou" / "extracted_features_14"
OUTPUT_CSV = OUTPUT_DIR / "Wenzhou_Multiorbits_14_features.csv"
PER_CELL_DIR = OUTPUT_DIR / "per_cell"
EARLY_REFERENCE_COUNT = 5
FEATURES = [
    "CVCT", "CCCT", "TCT", "AVCC", "ACCV", "TES", "IVCC", "IVCCCV",
    "RMSVCC", "SkewVCC", "KurtVCC", "RMSCCV", "SkewCCV", "KurtCCV",
]
POST_VARIABLES = [
    "Charging_voltage", "Charging_current", "Charging_datetime",
    "Discharging_voltage", "Discharging_current", "Discharging_datetime",
]
FINAL_VARIABLES = ["Discharge_capacity", "Discharge_datetime_end"]


def as_cycles(value) -> list[np.ndarray]:
    arrays = []
    for item in np.atleast_1d(value).flat:
        array = np.asarray(item, dtype=float).ravel()
        arrays.append(array)
    return arrays


def clean_triplet(time, voltage, current):
    n = min(len(time), len(voltage), len(current))
    time = np.asarray(time[:n], dtype=float)
    voltage = np.asarray(voltage[:n], dtype=float)
    current = np.asarray(current[:n], dtype=float)
    keep = np.isfinite(time) & np.isfinite(voltage) & np.isfinite(current)
    time, voltage, current = time[keep], voltage[keep], current[keep]
    if time.size:
        order = np.argsort(time, kind="stable")
        time, voltage, current = time[order], voltage[order], current[order]
        unique = np.r_[True, np.diff(time) > 0]
        time, voltage, current = time[unique], voltage[unique], current[unique]
    return time, voltage, current


def trapz(y, x):
    return float(np.trapezoid(y, x)) if hasattr(np, "trapezoid") else float(np.trapz(y, x))


def charge_features(time, voltage, current_mA):
    time, voltage, current_mA = clean_triplet(time, voltage, current_mA)
    active = current_mA > 50.0
    if active.sum() < 10:
        return None
    time, voltage, current_mA = time[active], voltage[active], current_mA[active]
    time = time - time[0]

    # CC ends close to the upper voltage plateau. Requiring both high voltage
    # and a current below 98% of the robust CC level avoids classifying the
    # initial transient or a single noisy sample as the CV phase.
    upper_voltage = float(np.nanpercentile(voltage, 99.5))
    cc_level = float(np.nanpercentile(current_mA, 90.0))
    cv_candidates = np.flatnonzero(
        (voltage >= upper_voltage - 0.010) & (current_mA < 0.98 * cc_level)
    )
    if cv_candidates.size:
        cv_start = int(cv_candidates[0])
    else:
        voltage_candidates = np.flatnonzero(voltage >= upper_voltage - 0.005)
        cv_start = int(voltage_candidates[0]) if voltage_candidates.size else len(time) - 1
    cv_start = max(2, min(cv_start, len(time) - 2))

    cc_time = time[: cv_start + 1]
    cc_voltage = voltage[: cv_start + 1]
    cc_current_a = current_mA[: cv_start + 1] / 1000.0
    cv_time = time[cv_start:]
    cv_voltage = voltage[cv_start:]
    cv_current_mA = current_mA[cv_start:]
    ccct = float(cc_time[-1] - cc_time[0])
    cvct = float(time[-1] - time[cv_start])
    if ccct <= 0 or cvct < 0 or len(cc_voltage) < 3:
        return None

    ivcc = trapz(cc_voltage, cc_time)
    ivcccv = trapz(voltage, time)
    return {
        "CCCT": ccct,
        "CVCT": cvct,
        "TCT": ccct + cvct,
        "AVCC": ivcc / ccct,
        "ACCV": trapz(cv_current_mA, cv_time) / cvct if cvct > 0 else float(np.mean(cv_current_mA)),
        "TES": trapz(cc_voltage * cc_current_a, cc_time) / 3600.0,
        "IVCC": ivcc,
        "IVCCCV": ivcccv,
        "RMSVCC": float(np.sqrt(np.mean(cc_voltage**2))),
        "SkewVCC": float(skew(cc_voltage, bias=True)),
        "KurtVCC": float(kurtosis(cc_voltage, fisher=True, bias=True)),
        "RMSCCV": float(np.sqrt(np.mean(cv_current_mA**2))),
        "SkewCCV": float(skew(cv_current_mA, bias=True)),
        "KurtCCV": float(kurtosis(cv_current_mA, fisher=True, bias=True)),
        "charge_throughput_Ah": trapz(current_mA / 1000.0, time) / 3600.0,
        "charge_duration_s": float(time[-1]),
        "cv_start_voltage_V": float(voltage[cv_start]),
        "cc_current_A": cc_level / 1000.0,
    }


def discharge_context(time, voltage, current_mA):
    time, voltage, current_mA = clean_triplet(time, voltage, current_mA)
    active = current_mA < -50.0
    if active.sum() < 5:
        return None
    t, v, i = time[active], voltage[active], current_mA[active]
    t = t - t[0]
    rounded = np.round(i / 50.0) * 50.0
    # Run-length sequence preserves the multi-stage ordering while suppressing
    # repeated samples within the same stage.
    stage_values = rounded[np.r_[True, np.diff(rounded) != 0]]
    stage_values = stage_values[np.abs(stage_values) >= 50.0]
    tail = rounded[-max(10, int(0.10 * len(rounded))):]
    final_current = abs(float(np.nanmedian(tail))) / 1000.0
    energy_wh = abs(trapz(v * i / 1000.0, t)) / 3600.0
    capacity_ah = abs(trapz(i / 1000.0, t)) / 3600.0
    sequence = "-".join(f"{abs(x) / 1000.0:g}" for x in stage_values)
    return {
        "final_discharge_current_A": final_current,
        "discharge_current_sequence_A": sequence,
        "n_discharge_stages": int(len(stage_values)),
        "discharge_duration_s": float(t[-1]),
        "discharge_throughput_Ah_integrated": capacity_ah,
        "discharge_energy_Wh": energy_wh,
    }


def protocol_group(cell_number):
    if cell_number in {1, 5, 6, 7, 11}:
        return "ACC"
    if cell_number in {2, 3, 4, 8, 9}:
        return "AMC"
    return "ACC_AMC_alternating"


def load_cell(folder):
    post_file = next(folder.glob("*Detail-Postprocess.mat"))
    final_file = next(folder.glob("*Detail-Final.mat"))
    post = loadmat(
        post_file, squeeze_me=True, struct_as_record=False,
        variable_names=POST_VARIABLES,
    )
    final = loadmat(
        final_file, squeeze_me=True, struct_as_record=False,
        variable_names=FINAL_VARIABLES,
    )
    post = {key: as_cycles(post[key]) for key in POST_VARIABLES}
    capacity_mAh = -np.asarray(final["Discharge_capacity"], dtype=float).ravel() / 3600.0
    end_time_h = np.asarray(final["Discharge_datetime_end"], dtype=float).ravel() / 3600.0
    n = min(len(capacity_mAh), *(len(post[key]) for key in POST_VARIABLES))
    cell_number = int(folder.name.rsplit("_", 1)[-1])
    rows = []
    for index in range(n):
        charge = charge_features(
            post["Charging_datetime"][index], post["Charging_voltage"][index],
            post["Charging_current"][index],
        )
        discharge = discharge_context(
            post["Discharging_datetime"][index], post["Discharging_voltage"][index],
            post["Discharging_current"][index],
        )
        if charge is None or discharge is None or not np.isfinite(capacity_mAh[index]):
            continue
        rows.append({
            "file_number": folder.name,
            "cell_number": cell_number,
            "Cycle": index + 1,
            "protocol_group": protocol_group(cell_number),
            "Capacity_mAh": float(capacity_mAh[index]),
            "elapsed_experimental_time_h": float(end_time_h[index] - end_time_h[0]),
            **charge,
            **discharge,
        })
    return pd.DataFrame(rows)


def add_targets_and_cumulative_context(data):
    frames = []
    for _, cell in data.groupby("file_number", sort=False):
        cell = cell.sort_values("Cycle").copy()
        initial = cell.iloc[:EARLY_REFERENCE_COUNT]["Capacity_mAh"].median()
        cell["raw_SOH"] = cell["Capacity_mAh"] / initial
        cell["regime_reference_capacity_mAh"] = np.nan
        cell["regime_normalised_SOH"] = np.nan
        # Final-current regimes are known test-bench settings rounded to 0.05 A.
        cell["discharge_regime_A"] = (
            cell["final_discharge_current_A"] / 0.05
        ).round() * 0.05
        for regime, indices in cell.groupby("discharge_regime_A", sort=True).groups.items():
            chronological = cell.loc[indices].sort_values("Cycle")
            reference = chronological.iloc[:EARLY_REFERENCE_COUNT]["Capacity_mAh"].median()
            cell.loc[indices, "regime_reference_capacity_mAh"] = reference
            cell.loc[indices, "regime_normalised_SOH"] = cell.loc[indices, "Capacity_mAh"] / reference
        cell["cumulative_charge_throughput_Ah"] = cell["charge_throughput_Ah"].cumsum()
        cell["cumulative_discharge_throughput_Ah"] = cell["discharge_throughput_Ah_integrated"].cumsum()
        cell["cumulative_discharge_energy_Wh"] = cell["discharge_energy_Wh"].cumsum()
        frames.append(cell)
    return pd.concat(frames, ignore_index=True)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PER_CELL_DIR.mkdir(parents=True, exist_ok=True)
    folders = sorted(path for path in DATA_DIR.glob("Wenzhou_Multiorbits_*") if path.is_dir())
    frames = []
    audit = []
    for index, folder in enumerate(folders, start=1):
        print(f"[{index}/{len(folders)}] {folder.name}", flush=True)
        frame = load_cell(folder)
        frames.append(frame)
        audit.append({
            "file_number": folder.name,
            "extracted_observations": len(frame),
            "first_cycle": int(frame.Cycle.min()) if len(frame) else None,
            "last_cycle": int(frame.Cycle.max()) if len(frame) else None,
            "protocol_group": protocol_group(int(folder.name.rsplit("_", 1)[-1])),
        })
    data = add_targets_and_cumulative_context(pd.concat(frames, ignore_index=True))
    data.to_csv(OUTPUT_CSV, index=False)
    for cell_name, cell in data.groupby("file_number", sort=True):
        cell.to_csv(PER_CELL_DIR / f"{cell_name}_14_features.csv", index=False)
    pd.DataFrame(audit).to_csv(OUTPUT_DIR / "extraction_audit.csv", index=False)

    quality = data.groupby("file_number").agg(
        observations=("Cycle", "size"),
        regimes=("discharge_regime_A", "nunique"),
        capacity_min_mAh=("Capacity_mAh", "min"),
        capacity_max_mAh=("Capacity_mAh", "max"),
        soh_min=("regime_normalised_SOH", "min"),
        soh_max=("regime_normalised_SOH", "max"),
        elapsed_h=("elapsed_experimental_time_h", "max"),
    )
    quality.to_csv(OUTPUT_DIR / "feature_quality_by_cell.csv")

    metadata = {
        "dataset": "Wenzhou Orbital Degradation Data",
        "candidate_model_features": FEATURES,
        "n_candidate_features": len(FEATURES),
        "target_recommended": "regime_normalised_SOH",
        "alternative_target": "raw_SOH",
        "reference_rule": f"median of first {EARLY_REFERENCE_COUNT} observations within each cell and final-current regime",
        "reference_boundary_note": "This target definition uses early-life capacity labels from the same cell. It must be disclosed as target-side initialization and kept fixed before model evaluation.",
        "excluded_from_model_inputs": [
            "Cycle", "file_number", "cell_number", "protocol_group", "Capacity_mAh",
            "raw_SOH", "regime_normalised_SOH", "regime_reference_capacity_mAh",
            "final_discharge_current_A", "discharge_regime_A",
            "discharge_current_sequence_A", "n_discharge_stages",
            "elapsed_experimental_time_h", "charge_throughput_Ah",
            "discharge_throughput_Ah_integrated", "discharge_energy_Wh",
            "cumulative_charge_throughput_Ah", "cumulative_discharge_throughput_Ah",
            "cumulative_discharge_energy_Wh", "charge_duration_s", "discharge_duration_s",
            "cv_start_voltage_V", "cc_current_A",
        ],
        "feature_units": {
            "CCCT": "s", "CVCT": "s", "TCT": "s", "AVCC": "V",
            "ACCV": "mA", "TES": "Wh", "IVCC": "V s", "IVCCCV": "V s",
            "RMSVCC": "V", "SkewVCC": "dimensionless", "KurtVCC": "dimensionless",
            "RMSCCV": "mA", "SkewCCV": "dimensionless", "KurtCCV": "dimensionless",
        },
        "segmentation": "CC/CV boundary uses the first high-voltage sample with current below 98% of the robust CC level.",
        "raw_files_unchanged": True,
    }
    (OUTPUT_DIR / "extraction_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Saved {len(data):,} rows to {OUTPUT_CSV}")
    print(quality.to_string())


if __name__ == "__main__":
    main()
