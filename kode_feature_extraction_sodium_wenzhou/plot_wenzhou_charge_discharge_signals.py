"""Plot charge/discharge voltage and current for every Wenzhou sodium cell.

Every cycle is drawn with its own elapsed-time coordinate starting at zero.
All cycles are retained; long trajectories are evenly downsampled only while
constructing the figures so that vector and raster output remain manageable.
"""

from __future__ import annotations

import re
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".matplotlib_cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import NewareNDA
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "Wenzhou_Sodium_ion_dataset"
OUTPUT_DIR = SCRIPT_DIR / "plots_charge_discharge"
MAX_POINTS_PER_CURVE = 300
DPI = 250


def cell_name(path: Path) -> str:
    return re.split(r"_127\.0\.0\.1|-h_127\.0\.0\.1", path.stem)[0]


def elapsed_time_hours(frame: pd.DataFrame) -> np.ndarray:
    """Return phase-local elapsed time, beginning at zero for every cycle."""
    values = frame["Time"].to_numpy(float)
    if len(values) and np.all(np.diff(values) >= 0):
        return (values - values[0]) / 3600.0
    stamps = pd.to_datetime(frame["Timestamp"])
    return (stamps - stamps.iloc[0]).dt.total_seconds().to_numpy(float) / 3600.0


def downsample(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if len(x) <= MAX_POINTS_PER_CURVE:
        return x, y
    keep = np.unique(np.linspace(0, len(x) - 1, MAX_POINTS_PER_CURVE).astype(int))
    return x[keep], y[keep]


def phase_segments(raw: pd.DataFrame, status: str, column: str):
    segments, cycles = [], []
    phase = raw[raw["Status"].astype(str).eq(status)]
    for cycle, frame in phase.groupby("Cycle", sort=True, observed=True):
        frame = frame.sort_values(["Timestamp", "Index"])
        time_h = elapsed_time_hours(frame)
        values = frame[column].to_numpy(float)
        time_h, values = downsample(time_h, values)
        if len(time_h) >= 2:
            segments.append(np.column_stack([time_h, values]))
            cycles.append(float(cycle))
    return segments, np.asarray(cycles, dtype=float)


def add_collection(ax, raw: pd.DataFrame, status: str, column: str,
                   norm: Normalize, ylabel: str, title: str):
    segments, cycles = phase_segments(raw, status, column)
    collection = LineCollection(
        segments, cmap="viridis", norm=norm, linewidths=0.38,
        alpha=0.42, rasterized=True,
    )
    collection.set_array(cycles)
    ax.add_collection(collection)
    ax.autoscale_view()
    ax.set_xlabel("Phase-local time (h; reset to 0 for each cycle)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(alpha=0.20)
    return collection, len(segments)


def plot_cell(path: Path) -> None:
    name = cell_name(path)
    print(f"Reading {path.name}", flush=True)
    raw = NewareNDA.read(str(path))
    cycle_min, cycle_max = float(raw.Cycle.min()), float(raw.Cycle.max())
    norm = Normalize(vmin=cycle_min, vmax=cycle_max)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    specs = [
        (axes[0, 0], "CC_Chg", "Voltage", "Voltage (V)", "(a) Charge voltage"),
        (axes[1, 0], "CC_Chg", "Current(mA)", "Current (mA)", "(c) Charge current"),
        (axes[0, 1], "CC_DChg", "Voltage", "Voltage (V)", "(b) Discharge voltage"),
        (axes[1, 1], "CC_DChg", "Current(mA)", "Current (mA)", "(d) Discharge current"),
    ]
    counts = []
    last_collection = None
    for ax, status, column, ylabel, title in specs:
        last_collection, count = add_collection(
            ax, raw, status, column, norm, ylabel, title)
        counts.append(count)

    colorbar = fig.colorbar(last_collection, ax=axes, fraction=0.025, pad=0.02)
    colorbar.set_label("Cycle number")
    fig.suptitle(
        f"Wenzhou sodium-ion cell {name}: charge/discharge trajectories\n"
        f"All cycles shown; each phase starts at t = 0",
        fontsize=15, fontweight="bold",
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{name}_charge_discharge_signals.png", dpi=DPI,
                bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{name}_charge_discharge_signals.pdf",
                bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}: {counts[0]} charge and {counts[2]} discharge cycles", flush=True)


def main() -> None:
    files = sorted(list(INPUT_DIR.glob("*.nda")) + list(INPUT_DIR.glob("*.ndax")))
    if not files:
        raise FileNotFoundError(f"No NDA/NDAX files found in {INPUT_DIR}")
    for path in files:
        plot_cell(path)
    print(f"Saved {len(files)} cell figures to {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
