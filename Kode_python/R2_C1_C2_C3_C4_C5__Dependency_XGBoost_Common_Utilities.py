"""Shared leakage-safe utilities for the NCA<->NMC XGBoost experiments.

This module is direction-neutral.  It contains only preprocessing and
cell-disjoint calibration helpers shared by the NCA-to-NMC and NMC-to-NCA
pipelines; running this file does not create experimental results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


SEED = 42
CALIBRATION_FRACTION = 0.20

HEALTH_FEATURES = [
    "CVCT", "CCCT", "TCT", "AVCC", "ACCV", "TES", "IVCC", "IVCCCV",
    "RMSVCC", "SkewVCC", "KurtVCC", "RMSCCV", "SkewCCV", "KurtCCV",
    "Temperature",
]


def split_fit_calibration(
    df: pd.DataFrame,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Split target training data by whole cells, stratified by temperature.

    The held-out LOCO test cell must be removed before this function is called.
    Consequently, neither fitting nor calibration can contain observations
    from the held-out cell.
    """
    meta = df.groupby("file_number", as_index=False)["Temperature"].first()
    rng = np.random.default_rng(seed)
    calibration_cells: list[int] = []

    for _, block in meta.groupby("Temperature"):
        cells = block.file_number.astype(int).to_numpy(copy=True)
        rng.shuffle(cells)
        n_calibration = min(
            max(1, round(CALIBRATION_FRACTION * len(cells))),
            len(cells) - 1,
        )
        calibration_cells.extend(cells[:n_calibration].tolist())

    fitting_cells = sorted(
        set(meta.file_number.astype(int)) - set(calibration_cells)
    )
    return fitting_cells, sorted(calibration_cells)


def limits(
    df: pd.DataFrame,
    features: list[str],
) -> tuple[pd.Series, pd.Series]:
    """Estimate clipping limits exclusively from the supplied training data."""
    return df[features].quantile(0.001), df[features].quantile(0.999)


def clipped(
    df: pd.DataFrame,
    features: list[str],
    lower: pd.Series,
    upper: pd.Series,
) -> pd.DataFrame:
    """Apply already-fitted clipping limits without inspecting labels."""
    return df[features].clip(lower=lower, upper=upper, axis=1)
