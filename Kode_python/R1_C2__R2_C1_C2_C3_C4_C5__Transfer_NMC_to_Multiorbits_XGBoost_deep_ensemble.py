"""Leakage-safe deep-ensemble NMC -> Wenzhou Multiorbits experiment.

This new entry point deliberately reuses the audited NCA-to-Multiorbits engine
so that cells, fold-local features, target supervision, calibration, ensemble
size, tree budget, metrics, and sensitivity analysis remain exactly matched.
Gated transfer is absent.
"""

from pathlib import Path

import R1_C2__R2_C1_C2_C3_C4_C5__Transfer_NCA_to_Multiorbits_XGBoost_deep_ensemble as experiment


ROOT = Path(__file__).resolve().parent
experiment.SOURCE_CSV = ROOT / "NMC_Integral_XGBoost.csv"
experiment.OUTPUT_DIR = ROOT / "R1_C2__R2_C1_C2_C3_C4_C5__results_NMC_to_Multiorbits"
experiment.SOURCE_NAME = "NMC"
experiment.DIRECTION_NAME = "NMC_to_Wenzhou_Multiorbits"
experiment.DIRECTION_TITLE = "NMC to Multiorbits"
experiment.SETUPS = [
    "Multiorbits_target_only",
    "NMC_source_only_matched",
    "NMC_Multiorbits_naive_pooled",
    "NMC_to_Multiorbits_raw_continuation",
]
experiment.SENSITIVITY_SETUPS = [
    "Multiorbits_target_only",
    "NMC_to_Multiorbits_raw_continuation",
]


if __name__ == "__main__":
    experiment.main()
