"""Extract scale-matched charging indicators from Wenzhou cell/pack signals.

The source dataset is never modified. Derived tables are local analysis artefacts
and must not be redistributed because the dataset licence is CC BY-ND 4.0.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import kurtosis, skew


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Wenzhou_Pack_Degradation_Data" / "Wenzhou_Pack_Degradation"
OUTPUT = ROOT / "kode_pack_wenzhou" / "extracted_features_14_cell_to_pack"
COMBINED = OUTPUT / "Wenzhou_Cell_Pack_14_features.csv"
PER_UNIT = OUTPUT / "per_unit"

FEATURES = [
    "CVCT", "CCCT", "TCT", "AVCC", "ACCV", "TES", "IVCC", "IVCCCV",
    "RMSVCC", "SkewVCC", "KurtVCC", "RMSCCV", "SkewCCV", "KurtCCV",
]
POST_VARIABLES = [
    "Charging_voltage", "Charging_current", "Charging_datetime",
    "Discharging_voltage", "Discharging_current", "Discharging_datetime",
]
FINAL_VARIABLES = ["Discharge_capacity", "Discharge_datetime_end"]
PACK_META = {
    9: (3, "nominal"), 10: (5, "nominal"), 11: (5, "nominal"),
    12: (3, "nominal"), 13: (5, "nominal"), 14: (3, "complex"),
}


def cycles(value) -> list[np.ndarray]:
    return [np.asarray(item, dtype=float).ravel()
            for item in np.atleast_1d(value).flat]


def clean(time, voltage, current):
    n = min(len(time), len(voltage), len(current))
    time = np.asarray(time[:n], float)
    voltage = np.asarray(voltage[:n], float)
    current = np.asarray(current[:n], float)
    keep = np.isfinite(time) & np.isfinite(voltage) & np.isfinite(current)
    time, voltage, current = time[keep], voltage[keep], current[keep]
    if len(time):
        order = np.argsort(time, kind="stable")
        time, voltage, current = time[order], voltage[order], current[order]
        unique = np.r_[True, np.diff(time) > 0]
        time, voltage, current = time[unique], voltage[unique], current[unique]
    return time, voltage, current


def integrate(y, x):
    return float(np.trapezoid(y, x))


def charge_features(time, voltage, current_mA, nominal_capacity_Ah):
    time, voltage, current_mA = clean(time, voltage, current_mA)
    # Neware exports can contain transition/rest samples in a charge segment.
    active = current_mA > max(50.0, 0.05 * nominal_capacity_Ah * 1000.0)
    if active.sum() < 10:
        return None
    time, voltage, current_mA = time[active], voltage[active], current_mA[active]
    time = time - time[0]
    current_rate_mA_per_Ah = current_mA / nominal_capacity_Ah

    upper_voltage = float(np.nanpercentile(voltage, 99.5))
    cc_level = float(np.nanpercentile(current_rate_mA_per_Ah, 90))
    cv_candidates = np.flatnonzero(
        (voltage >= upper_voltage - 0.010)
        & (current_rate_mA_per_Ah < 0.98 * cc_level)
    )
    if len(cv_candidates):
        cv_start = int(cv_candidates[0])
    else:
        candidates = np.flatnonzero(voltage >= upper_voltage - 0.005)
        cv_start = int(candidates[0]) if len(candidates) else len(time) - 1
    cv_start = max(2, min(cv_start, len(time) - 2))

    tcc, vcc = time[:cv_start + 1], voltage[:cv_start + 1]
    icc_rate = current_rate_mA_per_Ah[:cv_start + 1] / 1000.0
    tcv = time[cv_start:]
    icv_rate = current_rate_mA_per_Ah[cv_start:]
    ccct = float(tcc[-1] - tcc[0])
    cvct = float(time[-1] - time[cv_start])
    if ccct <= 0 or cvct < 0 or len(vcc) < 3:
        return None
    ivcc = integrate(vcc, tcc)
    return {
        "CCCT": ccct,
        "CVCT": cvct,
        "TCT": ccct + cvct,
        "AVCC": ivcc / ccct,
        "ACCV": (integrate(icv_rate, tcv) / cvct
                 if cvct > 0 else float(np.mean(icv_rate))),
        # Scale-free energy per nominal Ah because icc_rate is A/Ah.
        "TES": integrate(vcc * icc_rate, tcc) / 3600.0,
        "IVCC": ivcc,
        "IVCCCV": integrate(voltage, time),
        "RMSVCC": float(np.sqrt(np.mean(vcc ** 2))),
        "SkewVCC": float(skew(vcc, bias=True)),
        "KurtVCC": float(kurtosis(vcc, fisher=True, bias=True)),
        "RMSCCV": float(np.sqrt(np.mean(icv_rate ** 2))),
        "SkewCCV": float(skew(icv_rate, bias=True)),
        "KurtCCV": float(kurtosis(icv_rate, fisher=True, bias=True)),
        "charge_C_rate": cc_level / 1000.0,
        "charge_throughput_EFC": integrate(
            current_rate_mA_per_Ah / 1000.0, time) / 3600.0,
    }


def discharge_context(time, voltage, current_mA, nominal_capacity_Ah):
    time, voltage, current_mA = clean(time, voltage, current_mA)
    active = current_mA < -max(50.0, 0.05 * nominal_capacity_Ah * 1000.0)
    if active.sum() < 5:
        return None
    t, v, i = time[active], voltage[active], current_mA[active]
    t = t - t[0]
    c_rate = np.abs(i) / 1000.0 / nominal_capacity_Ah
    rounded = np.round(c_rate / 0.1) * 0.1
    stages = rounded[np.r_[True, np.diff(rounded) != 0]]
    tail = rounded[-max(10, int(0.10 * len(rounded))):]
    final_rate = float(np.nanmedian(tail))
    return {
        "final_discharge_C_rate": final_rate,
        "discharge_C_rate_sequence": "-".join(f"{x:g}" for x in stages),
        "n_discharge_stages": int(len(stages)),
        "discharge_duration_s": float(t[-1]),
        "discharge_throughput_EFC_integrated": integrate(c_rate, t) / 3600.0,
        "discharge_energy_Wh_per_nominal_Ah": abs(
            integrate(v * i / 1000.0, t)) / 3600.0 / nominal_capacity_Ah,
    }


def unit_metadata(folder: Path):
    number = int(folder.name.rsplit("#", 1)[-1])
    if "Cell" in folder.name:
        return number, "cell", 1, "nominal", 0.8
    group_size, mode = PACK_META[number]
    return number, "pack", group_size, mode, 0.8 * group_size


def extract_unit(folder: Path):
    number, unit_type, group_size, mode, nominal_capacity = unit_metadata(folder)
    post_file = next(folder.glob("*Postprocess.mat"))
    final_file = next(folder.glob("*_Final.mat"))
    post_raw = loadmat(
        post_file, squeeze_me=True, struct_as_record=False,
        variable_names=POST_VARIABLES,
    )
    final = loadmat(
        final_file, squeeze_me=True, struct_as_record=False,
        variable_names=FINAL_VARIABLES,
    )
    post = {key: cycles(post_raw[key]) for key in POST_VARIABLES}
    # Stored integrated capacity is mA*s with negative discharge sign.
    capacity_Ah = -np.asarray(final["Discharge_capacity"], float).ravel() / 3.6e6
    end_h = np.asarray(final["Discharge_datetime_end"], float).ravel() / 3600.0
    n = min(len(capacity_Ah), *(len(post[key]) for key in POST_VARIABLES))
    rows = []
    for j in range(n):
        charge = charge_features(
            post["Charging_datetime"][j], post["Charging_voltage"][j],
            post["Charging_current"][j], nominal_capacity,
        )
        discharge = discharge_context(
            post["Discharging_datetime"][j], post["Discharging_voltage"][j],
            post["Discharging_current"][j], nominal_capacity,
        )
        cap = capacity_Ah[j]
        if charge is None or discharge is None or not np.isfinite(cap) or cap <= 0:
            continue
        rows.append({
            "file_number": folder.name,
            "unit_number": number,
            "unit_type": unit_type,
            "group_size": group_size,
            "operating_mode": mode,
            "Cycle": j + 1,
            "nominal_capacity_Ah": nominal_capacity,
            "discharge_capacity_Ah": float(cap),
            "nominal_capacity_SOH": float(cap / nominal_capacity),
            "elapsed_experimental_time_h": float(end_h[j] - end_h[0]),
            **charge,
            **discharge,
        })
    del post_raw, post, final
    gc.collect()
    frame = pd.DataFrame(rows).sort_values("Cycle").reset_index(drop=True)
    # Regime target is sensitivity/context only; primary modelling uses nominal SOH.
    frame["regime_normalised_SOH"] = np.nan
    frame["regime_reference_capacity_Ah"] = np.nan
    for _, idx in frame.groupby("final_discharge_C_rate").groups.items():
        chronological = frame.loc[idx].sort_values("Cycle")
        reference = chronological.head(5).discharge_capacity_Ah.median()
        frame.loc[idx, "regime_reference_capacity_Ah"] = reference
        frame.loc[idx, "regime_normalised_SOH"] = (
            frame.loc[idx, "discharge_capacity_Ah"] / reference
        )
    dominant_regime = float(frame.final_discharge_C_rate.value_counts().idxmax())
    frame["dominant_benchmark_discharge_C_rate"] = dominant_regime
    # Nominal units contain sparse periodic checkpoint cycles. The primary
    # benchmark uses their dominant repeated regime; the deliberately complex
    # Pack #14 retains all of its designed discharge-current regimes.
    frame["analysis_eligible_primary"] = (
        True if mode == "complex"
        else np.isclose(frame.final_discharge_C_rate, dominant_regime)
    )
    frame["lifetime_progress"] = (
        (frame.Cycle - frame.Cycle.min()) /
        max(1, frame.Cycle.max() - frame.Cycle.min())
    )
    frame["life_stage"] = pd.cut(
        frame.lifetime_progress, [-np.inf, 1/3, 2/3, np.inf],
        labels=["early", "middle", "late"],
    ).astype(str)
    return frame


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    PER_UNIT.mkdir(parents=True, exist_ok=True)
    folders = sorted(
        [p for p in DATA.iterdir() if p.is_dir() and
         (p.name.startswith("Wenzhou_Cell_") or p.name.startswith("Wenzhou_Pack_"))],
        key=lambda p: int(p.name.rsplit("#", 1)[-1]),
    )
    frames, audit = [], []
    for i, folder in enumerate(folders, 1):
        print(f"[{i:02d}/{len(folders)}] extracting {folder.name}", flush=True)
        frame = extract_unit(folder)
        frames.append(frame)
        frame.to_csv(PER_UNIT / f"{folder.name}_14_features.csv", index=False)
        audit.append({
            "file_number": folder.name,
            "unit_type": frame.unit_type.iloc[0],
            "group_size": int(frame.group_size.iloc[0]),
            "operating_mode": frame.operating_mode.iloc[0],
            "observations": len(frame),
            "first_cycle": int(frame.Cycle.min()),
            "last_cycle": int(frame.Cycle.max()),
            "discharge_regimes": frame.final_discharge_C_rate.nunique(),
            "dominant_benchmark_C_rate": frame.dominant_benchmark_discharge_C_rate.iloc[0],
            "primary_observations": int(frame.analysis_eligible_primary.sum()),
            "soh_min": frame.nominal_capacity_SOH.min(),
            "soh_max": frame.nominal_capacity_SOH.max(),
        })
    data = pd.concat(frames, ignore_index=True)
    if data[FEATURES].isna().any().any() or not np.isfinite(data[FEATURES]).all().all():
        raise ValueError("Non-finite model feature detected after extraction.")
    data.to_csv(COMBINED, index=False)
    pd.DataFrame(audit).to_csv(OUTPUT / "extraction_audit.csv", index=False)
    metadata = {
        "dataset_name": "Wenzhou Pack Degradation Data",
        "source_units": "Wenzhou_Cell_#01 through #08",
        "target_units": "Wenzhou_Pack_#09 through #14",
        "candidate_features": FEATURES,
        "primary_target": "nominal_capacity_SOH",
        "primary_target_definition": "discharge capacity Ah / known nominal unit capacity Ah",
        "alternative_target": "regime_normalised_SOH",
        "primary_observation_rule": "dominant discharge regime for nominal units; all designed regimes for complex Pack #14",
        "current_scaling": "current divided by known nominal unit capacity; current-derived features use C-rate scale",
        "raw_data_modified": False,
        "redistribution_warning": "Derived tables are local analysis artefacts and must not be redistributed under the source CC BY-ND licence without permission.",
    }
    (OUTPUT / "extraction_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Saved {len(data):,} observations to {COMBINED}")
    print(pd.DataFrame(audit).to_string(index=False))


if __name__ == "__main__":
    main()
