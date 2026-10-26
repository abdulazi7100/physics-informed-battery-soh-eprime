"""Plot cycle-reset charge/discharge signals and orbital diagnostics.

The processed MATLAB files already separate each charge and discharge event.
For every trajectory, elapsed time is calculated as t - t[0], so time returns
to zero after every reconstructed cycle.  Raw Cycle.xlsx indices are not used.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Wenzhou_Multiorbits_dataset"
OUTPUT_DIR = ROOT / "kode_multiorbits_wenzhou" / "plots"
MAX_POINTS_PER_TRACE = 350

SIGNAL_VARIABLES = [
    "Charging_voltage", "Charging_current", "Charging_datetime",
    "Discharging_voltage", "Discharging_current", "Discharging_datetime",
]
FINAL_VARIABLES = ["Discharge_capacity", "Discharge_datetime_end"]


def cycles(value):
    """Return a MATLAB cell array as a list of finite one-dimensional arrays."""
    values = np.atleast_1d(value)
    answer = []
    for item in values.flat:
        array = np.asarray(item, dtype=float).ravel()
        answer.append(array[np.isfinite(array)])
    return answer


def downsample(*arrays, maximum=MAX_POINTS_PER_TRACE):
    n = min(map(len, arrays))
    if n == 0:
        return [np.array([]) for _ in arrays]
    index = np.unique(np.linspace(0, n - 1, min(n, maximum)).astype(int))
    return [np.asarray(array[:n])[index] for array in arrays]


def final_discharge_current(current):
    """Estimate the final active discharge-current stage in amperes."""
    current = np.asarray(current, dtype=float)
    active = current[current < -50.0]
    if active.size == 0:
        return np.nan
    tail = active[-max(10, int(0.10 * active.size)):]
    # The test bench uses nominal 50-mA increments.
    return abs(float(np.nanmedian(np.round(tail / 50.0) * 50.0))) / 1000.0


def load_cell(folder):
    post = next(folder.glob("*Detail-Postprocess.mat"))
    final = next(folder.glob("*Detail-Final.mat"))
    signals = loadmat(
        post, squeeze_me=True, struct_as_record=False,
        variable_names=SIGNAL_VARIABLES,
    )
    targets = loadmat(
        final, squeeze_me=True, struct_as_record=False,
        variable_names=FINAL_VARIABLES,
    )
    result = {name: cycles(signals[name]) for name in SIGNAL_VARIABLES}
    # Integrated current is stored as mA.s; convert to positive mAh.
    result["capacity_mAh"] = -np.asarray(
        targets["Discharge_capacity"], dtype=float
    ).ravel() / 3600.0
    result["end_time_h"] = np.asarray(
        targets["Discharge_datetime_end"], dtype=float
    ).ravel() / 3600.0
    result["final_current_A"] = np.asarray([
        final_discharge_current(x) for x in result["Discharging_current"]
    ])
    return result


def plot_signal_trajectories(cell_name, data, output):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    panels = [
        (axes[0, 0], "Charging_datetime", "Charging_voltage", "Charge voltage", "Voltage (V)", 1.0),
        (axes[1, 0], "Charging_datetime", "Charging_current", "Charge current", "Current (A)", 1e-3),
        (axes[0, 1], "Discharging_datetime", "Discharging_voltage", "Discharge voltage", "Voltage (V)", 1.0),
        (axes[1, 1], "Discharging_datetime", "Discharging_current", "Discharge current", "Current (A)", 1e-3),
    ]
    n_cycles = min(len(data[name]) for name in SIGNAL_VARIABLES)
    cmap = plt.get_cmap("viridis")
    norm = Normalize(1, max(n_cycles, 2))

    for ax, time_key, signal_key, title, ylabel, scale in panels:
        for index, (time, signal) in enumerate(zip(data[time_key], data[signal_key]), start=1):
            time, signal = downsample(time, signal)
            if time.size < 2:
                continue
            elapsed_minutes = (time - time[0]) / 60.0
            ax.plot(
                elapsed_minutes, signal * scale, color=cmap(norm(index)),
                linewidth=0.35, alpha=0.22, rasterized=True,
            )
        ax.set(title=title, xlabel="Elapsed time within cycle (min)", ylabel=ylabel)
        ax.grid(alpha=0.2)

    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes.ravel().tolist(),
        shrink=0.80, pad=0.02,
    )
    colorbar.set_label("Reconstructed cycle / discharge-event index")
    fig.suptitle(
        f"{cell_name}: cycle-reset charge and discharge signals\n"
        "Every trajectory starts at elapsed time = 0",
        fontsize=15,
    )
    fig.savefig(output / f"{cell_name}_cycle_reset_signals.png", dpi=220)
    fig.savefig(output / f"{cell_name}_cycle_reset_signals.pdf", dpi=220)
    plt.close(fig)


def plot_orbital_diagnostics(cell_name, data, output):
    n = min(len(data["capacity_mAh"]), len(data["final_current_A"]))
    cycle = np.arange(1, n + 1)
    capacity = data["capacity_mAh"][:n]
    current = data["final_current_A"][:n]
    elapsed = data["end_time_h"][:n] - data["end_time_h"][0]
    valid = np.isfinite(capacity) & np.isfinite(current)

    fig, axes = plt.subplots(3, 1, figsize=(12, 11), constrained_layout=True)
    axes[0].plot(cycle, capacity, color="0.75", linewidth=0.6, zorder=1)
    scatter = axes[0].scatter(
        cycle[valid], capacity[valid], c=current[valid], cmap="turbo",
        s=12, edgecolors="none", zorder=2,
    )
    fig.colorbar(scatter, ax=axes[0], label="Final discharge current (A)")
    axes[0].set(ylabel="Measured discharge capacity (mAh)", title="Capacity orbits by current regime")

    axes[1].step(cycle, current, where="mid", linewidth=0.8, color="#7b3294")
    axes[1].set(ylabel="Final discharge current (A)", title="Operating-regime sequence")

    axes[2].scatter(elapsed[valid], capacity[valid], c=current[valid], cmap="turbo", s=12, edgecolors="none")
    axes[2].set(xlabel="Elapsed experimental time (h)", ylabel="Measured discharge capacity (mAh)",
                title="Capacity against calendar/experimental time")
    for ax in axes:
        ax.grid(alpha=0.2)
    axes[1].set_xlabel("Reconstructed cycle / discharge-event index")
    axes[0].set_xlabel("Reconstructed cycle / discharge-event index")
    fig.suptitle(f"{cell_name}: orbital degradation diagnostics", fontsize=15)
    fig.savefig(output / f"{cell_name}_orbital_diagnostics.png", dpi=220)
    fig.savefig(output / f"{cell_name}_orbital_diagnostics.pdf", dpi=220)
    plt.close(fig)


def plot_all_cells(all_cells, output):
    fig, axes = plt.subplots(4, 3, figsize=(16, 15), sharex=False, sharey=True, constrained_layout=True)
    for ax, (cell_name, data) in zip(axes.flat, all_cells.items()):
        n = min(len(data["capacity_mAh"]), len(data["final_current_A"]))
        cycle = np.arange(1, n + 1)
        capacity = data["capacity_mAh"][:n]
        current = data["final_current_A"][:n]
        valid = np.isfinite(capacity) & np.isfinite(current)
        ax.plot(cycle, capacity, color="0.8", linewidth=0.5)
        ax.scatter(cycle[valid], capacity[valid], c=current[valid], cmap="turbo", vmin=0.5, vmax=2.5,
                   s=5, edgecolors="none", rasterized=True)
        ax.set_title(cell_name.replace("Wenzhou_Multiorbits_", "Orbital-Cell "))
        ax.grid(alpha=0.15)
    fig.supxlabel("Reconstructed cycle / discharge-event index")
    fig.supylabel("Measured discharge capacity (mAh)")
    fig.suptitle("Wenzhou Multiorbits: capacity trajectories coloured by final discharge current", fontsize=16)
    scalar = plt.cm.ScalarMappable(norm=Normalize(0.5, 2.5), cmap="turbo")
    fig.colorbar(scalar, ax=axes.ravel().tolist(), label="Final discharge current (A)", shrink=0.7)
    fig.savefig(output / "All_cells_capacity_orbits.png", dpi=220)
    fig.savefig(output / "All_cells_capacity_orbits.pdf", dpi=220)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    folders = sorted(path for path in DATA_DIR.glob("Wenzhou_Multiorbits_*" ) if path.is_dir())
    all_cells = {}
    for index, folder in enumerate(folders, start=1):
        print(f"[{index}/{len(folders)}] {folder.name}", flush=True)
        data = load_cell(folder)
        all_cells[folder.name] = data
        plot_signal_trajectories(folder.name, data, OUTPUT_DIR)
        plot_orbital_diagnostics(folder.name, data, OUTPUT_DIR)
    plot_all_cells(all_cells, OUTPUT_DIR)
    print(f"Saved plots to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
