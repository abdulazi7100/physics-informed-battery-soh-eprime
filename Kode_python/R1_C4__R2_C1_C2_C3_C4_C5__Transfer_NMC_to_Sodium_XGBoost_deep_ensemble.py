"""Leakage-safe five-member deep ensemble for NMC -> Wenzhou sodium.

The audited NCA-to-sodium implementation is reused with only the declared
source chemistry, source file, method labels, direction, and output folder
changed. This guarantees identical target cells, folds, feature groups,
uncertainty procedure, metrics, model capacity, and seeds. Gated transfer is
absent, and previous code/results are preserved.
"""

from pathlib import Path

import R1_C4__R2_C1_C2_C3_C4_C5__Transfer_NCA_to_Sodium_XGBoost_deep_ensemble as experiment


ROOT = Path(__file__).resolve().parent
experiment.SOURCE_CSV = ROOT / "NMC_Integral_XGBoost.csv"
experiment.OUTPUT_DIR = ROOT / "R1_C4__R2_C1_C2_C3_C4_C5__results_NMC_to_Sodium_800cycles"
experiment.SOURCE_NAME = "NMC"
experiment.DIRECTION_NAME = "NMC_to_Wenzhou_sodium"
experiment.SETUPS = ["Sodium_target_only", "NMC_source_only_matched",
                     "NMC_Sodium_naive_pooled",
                     "NMC_to_Sodium_raw_continuation"]


if __name__ == "__main__":
    experiment.main()
