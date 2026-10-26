"""Extract seven physically available CC-charge features from Wenzhou SIB data.

The raw NEWARE files contain Rest, CC_Chg, and CC_DChg records, but no CV_Chg
or usable temperature channel. Consequently, this script extracts only the
seven indicators that retain distinct physical meanings under this protocol.
No CV-derived feature is imputed or duplicated.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import NewareNDA
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "Wenzhou_Sodium_ion_dataset"
OUTPUT_DIR = Path(__file__).resolve().parent
RAW_PATTERNS = ("*.nda", "*.ndax")
REFERENCE_CYCLES = 5


def cell_name(path: Path) -> str:
    """Recover the short experimental cell name from the NEWARE filename."""
    name = re.split(r"_127\.0\.0\.1|-h_127\.0\.0\.1", path.stem)[0]
    return name


def trapz(y: np.ndarray, x: np.ndarray) -> float:
    """NumPy-version-independent trapezoidal integration."""
    if len(y) < 2:
        return float("nan")
    return float(np.trapezoid(y, x))


def extract_cycle(cycle: int, frame: pd.DataFrame) -> dict[str, float] | None:
    charge = frame[frame["Status"].astype(str).eq("CC_Chg")].copy()
    discharge = frame[frame["Status"].astype(str).eq("CC_DChg")].copy()
    if len(charge) < 3:
        return None

    # NEWARE's per-step elapsed time retains sub-second precision that can be
    # rounded away in NDA timestamps. Each supplied cycle has one CC_Chg step.
    charge = charge.sort_values(["Timestamp", "Index"])
    time_s = charge["Time"].to_numpy(float)
    if np.any(np.diff(time_s) < 0):
        # Defensive fallback for a future file containing multiple CC steps.
        time_s = (charge["Timestamp"] - charge["Timestamp"].iloc[0]).dt.total_seconds().to_numpy(float)
    voltage = charge["Voltage"].to_numpy(float)
    current_a = charge["Current(mA)"].to_numpy(float) / 1000.0
    finite = np.isfinite(time_s) & np.isfinite(voltage) & np.isfinite(current_a)
    time_s, voltage, current_a = time_s[finite], voltage[finite], current_a[finite]
    if len(time_s) < 3 or time_s[-1] <= 0:
        return None

    # Drop repeated time stamps so integration does not double-count a point.
    keep = np.r_[True, np.diff(time_s) > 0]
    time_s, voltage, current_a = time_s[keep], voltage[keep], current_a[keep]
    duration_s = float(time_s[-1] - time_s[0])
    if duration_s <= 0:
        return None

    mean_v = trapz(voltage, time_s) / duration_s
    centered = voltage - voltage.mean()
    std = float(np.sqrt(np.mean(centered**2)))
    skew = 0.0 if std <= 1e-15 else float(np.mean((centered / std) ** 3))
    kurt = 0.0 if std <= 1e-15 else float(np.mean((centered / std) ** 4) - 3.0)

    q_discharge = float(discharge["Discharge_Capacity(mAh)"].max()) if len(discharge) else np.nan
    q_charge = float(charge["Charge_Capacity(mAh)"].max())
    median_dt = float(np.median(np.diff(time_s))) if len(time_s) > 1 else np.nan

    return {
        "Cycle": int(cycle),
        "CCCT": duration_s,
        "AVCC": mean_v,
        # Total electrical energy during CC charging, reported in Wh.
        "TES": trapz(voltage * current_a, time_s) / 3600.0,
        # Integral of the CC voltage trajectory with respect to time (V s).
        "IVCC": trapz(voltage, time_s),
        "RMSVCC": float(np.sqrt(np.mean(voltage**2))),
        "SkewVCC": skew,
        "KurtVCC": kurt,
        "Charge_Capacity_mAh": q_charge,
        "Discharge_Capacity_mAh": q_discharge,
        "n_CC_points": int(len(time_s)),
        "median_sampling_interval_s": median_dt,
    }


def extract_file(path: Path) -> pd.DataFrame:
    raw = NewareNDA.read(str(path))
    required = {
        "Cycle", "Status", "Time", "Timestamp", "Index", "Voltage", "Current(mA)",
        "Charge_Capacity(mAh)", "Discharge_Capacity(mAh)",
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"{path.name}: missing required columns {missing}")

    rows = []
    for cycle, frame in raw.groupby("Cycle", sort=True, observed=True):
        row = extract_cycle(int(cycle), frame)
        if row is not None:
            rows.append(row)
    result = pd.DataFrame(rows)
    result.insert(0, "file_number", cell_name(path))
    result.insert(1, "source_file", path.name)

    valid = result.loc[result["Discharge_Capacity_mAh"].gt(0), "Discharge_Capacity_mAh"]
    if valid.empty:
        raise ValueError(f"{path.name}: no valid discharge capacity was found")
    # A robust early-life capacity reference; labels are retained explicitly so
    # readers can reproduce or replace this SOH definition.
    reference = float(valid.head(REFERENCE_CYCLES).median())
    result["SOH"] = result["Discharge_Capacity_mAh"] / reference
    result["reference_discharge_capacity_mAh"] = reference
    return result


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted({p for pattern in RAW_PATTERNS for p in INPUT_DIR.glob(pattern)})
    if not files:
        raise FileNotFoundError(f"No NDA/NDAX files found in {INPUT_DIR}")

    frames = []
    for number, path in enumerate(files, 1):
        print(f"[{number}/{len(files)}] Reading {path.name}", flush=True)
        extracted = extract_file(path)
        frames.append(extracted)
        extracted.to_csv(OUTPUT_DIR / f"features_{cell_name(path)}.csv", index=False)
        print(f"    extracted {len(extracted)} charge cycles", flush=True)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["file_number", "Cycle"]).reset_index(drop=True)
    combined.to_csv(OUTPUT_DIR / "Wenzhou_Sodium_7_features.csv", index=False)

    summary = (combined.groupby("file_number", as_index=False)
               .agg(n_cycles=("Cycle", "size"),
                    first_cycle=("Cycle", "min"), last_cycle=("Cycle", "max"),
                    reference_capacity_mAh=("reference_discharge_capacity_mAh", "first"),
                    min_soh=("SOH", "min"), max_soh=("SOH", "max"),
                    median_cc_points=("n_CC_points", "median"),
                    median_sampling_s=("median_sampling_interval_s", "median")))
    summary.to_csv(OUTPUT_DIR / "extraction_summary.csv", index=False)

    metadata = {
        "input_directory": str(INPUT_DIR),
        "n_cells": int(combined.file_number.nunique()),
        "n_cycle_rows": int(len(combined)),
        "features": ["CCCT", "AVCC", "TES", "IVCC", "RMSVCC", "SkewVCC", "KurtVCC"],
        "target": "SOH = discharge capacity / median of first five valid discharge capacities per cell",
        "units": {"CCCT": "s", "AVCC": "V", "TES": "Wh", "IVCC": "V s", "RMSVCC": "V"},
        "phase": "CC_Chg only",
        "excluded_unavailable_features": ["CVCT", "Temperature"],
        "excluded_redundant_features": ["TCT", "ACCV", "IVCCCV", "RMSCCV", "SkewCCV", "KurtCCV"],
        "notes": "No SOH smoothing, interpolation, outlier deletion, or cycle truncation is applied.",
    }
    (OUTPUT_DIR / "extraction_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(f"Saved {len(combined)} rows to {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
