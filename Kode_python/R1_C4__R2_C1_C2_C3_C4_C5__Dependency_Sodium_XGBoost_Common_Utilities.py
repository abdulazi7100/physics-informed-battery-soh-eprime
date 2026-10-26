"""Shared leakage-safe utilities for NCA/NMC-to-Sodium experiments.

The helpers in this module are independent of the source chemistry.  Feature
preprocessing, target-cell splitting, clipping, and fold-specific fusion
ranking are shared unchanged by the NCA-to-Sodium and NMC-to-Sodium pipelines.
This module is not an independently executable experiment and creates no
result directory.
"""

from __future__ import annotations

import zlib

import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor


SEED = 42
FEATURES = ["CCCT", "AVCC", "TES", "IVCC", "RMSVCC", "SkewVCC", "KurtVCC"]
FEATURE_COUNTS = [4, 5, 6, 7]
ROUNDS = 80
SOURCE_ROUNDS = 40
TARGET_ROUNDS = ROUNDS - SOURCE_ROUNDS
PARAMS = dict(max_depth=4, min_child_weight=3.0, reg_lambda=5.0, reg_alpha=0.0)


def stable_seed(cell: str, offset: int = 0) -> int:
    """Return a reproducible seed for a target cell and experiment stage."""
    return SEED + offset + zlib.crc32(str(cell).encode("utf-8")) % 100_000


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply cell-relative early-life scaling without using SOH labels."""
    out = frame.sort_values(["file_number", "Cycle"]).copy()
    references = (
        out.groupby("file_number", sort=False)
        .head(5)
        .groupby("file_number")[FEATURES]
        .median()
    )
    for feature in FEATURES:
        reference = out.file_number.map(references[feature]).astype(float)
        values = out[feature].astype(float)
        out[feature] = np.where(
            reference.abs() > 1e-12,
            values / reference,
            values - reference,
        )
    target_name = "SOH" if "SOH" in out else "soh"
    return out.replace([np.inf, -np.inf], np.nan).dropna(
        subset=FEATURES + [target_name]
    )


def make_model(seed: int, rounds: int) -> XGBRegressor:
    """Build the fixed-capacity XGBoost regressor used in both directions."""
    return XGBRegressor(
        n_estimators=rounds,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        n_jobs=-1,
        random_state=seed,
        verbosity=0,
        **PARAMS,
    )


def limits(
    frame: pd.DataFrame,
    features: list[str],
) -> tuple[pd.Series, pd.Series]:
    """Fit clipping limits only on the relevant training partition."""
    return frame[features].quantile(0.001), frame[features].quantile(0.999)


def clipped(
    frame: pd.DataFrame,
    features: list[str],
    lower: pd.Series,
    upper: pd.Series,
) -> pd.DataFrame:
    """Apply pre-fitted clipping limits unchanged to another partition."""
    return frame[features].clip(lower=lower, upper=upper, axis=1)


def split_fit_calibration(
    train_cells: list[str],
    seed: int,
) -> tuple[list[str], list[str]]:
    """Reserve two complete target cells for uncertainty calibration."""
    cells = np.asarray(sorted(train_cells), dtype=object)
    if len(cells) < 3:
        raise ValueError("At least three non-test target cells are required.")
    rng = np.random.default_rng(seed)
    rng.shuffle(cells)
    return sorted(cells[2:].tolist()), sorted(cells[:2].tolist())


def fusion_ranking(fit: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Fit correlation-SHAP fusion ranking only on outer-fold fitting cells."""
    lower, upper = limits(fit, FEATURES)
    x = clipped(fit, FEATURES, lower, upper)
    correlation = x.corrwith(fit.soh).abs().fillna(0.0)
    selector = make_model(seed, ROUNDS).fit(x, fit.soh)
    sample = x.sample(min(4000, len(x)), random_state=seed)
    contributions = selector.get_booster().predict(
        xgb.DMatrix(sample, feature_names=FEATURES),
        pred_contribs=True,
    )
    shap_importance = pd.Series(
        np.abs(contributions[:, :-1]).mean(axis=0),
        index=FEATURES,
    )
    table = pd.DataFrame(
        {
            "feature": FEATURES,
            "abs_target_correlation": correlation[FEATURES].values,
            "mean_abs_shap": shap_importance[FEATURES].values,
        }
    )
    table["correlation_rank"] = table.abs_target_correlation.rank(
        ascending=False,
        method="average",
    )
    table["shap_rank"] = table.mean_abs_shap.rank(
        ascending=False,
        method="average",
    )
    table["fusion_rank"] = (table.correlation_rank + table.shap_rank) / 2
    return table.sort_values(
        ["fusion_rank", "shap_rank", "correlation_rank", "feature"]
    ).reset_index(drop=True)
