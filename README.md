# Physics-Informed Uncertainty-Aware Battery SOH Estimation

This repository contains the reproducibility code and aggregate outputs for the
manuscript:

> **Physics-Informed Uncertainty-Aware Learning for Cross-Cell State-of-Health
> Estimation of Lithium-Ion Batteries**  
> Manuscript ID: EPRIME-D-26-00192R1

The repository reproduces the leakage-safe feature-selection, matched-baseline,
deep-ensemble XGBoost, uncertainty-calibration, cell-to-pack, operational-regime,
cumulative-lifetime, and lithium-to-sodium experiments reported in the paper.

## Important data notice

No raw battery data are redistributed here. All datasets are third-party public
resources and must be downloaded from their original repositories. The Wenzhou
datasets are distributed under CC BY-ND 4.0; modified or restructured copies
must not be republished. See [DATA_SOURCES.md](DATA_SOURCES.md).

## Repository structure

```text
Kode_python/                              Main modelling code and dependencies
kode_pack_wenzhou/                       Cell/pack feature extraction
kode_multiorbits_wenzhou/                Multiorbits feature extraction
kode_feature_extraction_sodium_wenzhou/  Sodium-ion feature extraction
kode_feature_extraction_randomized_wenzhou/ Randomized-data extraction
results/                                  Aggregate outputs and figures
data/                                     Instructions only; no data included
```

## Main experiments

| Experiment | Entry-point script |
|---|---|
| NCA to NMC | `R2_C1_C2_C3_C4_C5__Transfer_NCA_to_NMC_XGBoost_ranked_groups_deep_ensemble.py` |
| NMC to NCA | `R2_C1_C2_C3_C4_C5__Transfer_NMC_to_NCA_XGBoost_ranked_groups_deep_ensemble.py` |
| Cell to pack | `R1_C1__R2_C1_C2_C3_C4_C5__Transfer_Wenzhou_Cell_to_Pack_XGBoost_deep_ensemble.py` |
| NCA to Multiorbits | `R1_C2__R2_C1_C2_C3_C4_C5__Transfer_NCA_to_Multiorbits_XGBoost_deep_ensemble.py` |
| NMC to Multiorbits | `R1_C2__R2_C1_C2_C3_C4_C5__Transfer_NMC_to_Multiorbits_XGBoost_deep_ensemble.py` |
| Randomized cumulative lifetime | `R1_C3__R2_C1_C2_C3_C4_C5__Wenzhou_Randomized_cumulative_lifetime_XGBoost_deep_ensemble.py` |
| NCA to sodium-ion | `R1_C4__R2_C1_C2_C3_C4_C5__Transfer_NCA_to_Sodium_XGBoost_deep_ensemble.py` |
| NMC to sodium-ion | `R1_C4__R2_C1_C2_C3_C4_C5__Transfer_NMC_to_Sodium_XGBoost_deep_ensemble.py` |

The four files containing `Dependency` in their names are required modules and
must remain in `Kode_python/` beside the entry-point scripts.

## Installation

Python 3.11 was used for the archived run.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

## Data preparation

1. Download each dataset from the original source listed in
   [DATA_SOURCES.md](DATA_SOURCES.md).
2. Preserve the directory names described in [data/README.md](data/README.md).
3. Run the relevant feature-extraction script.
4. Run the required modelling entry point from `Kode_python/`.

Example:

```bash
python kode_multiorbits_wenzhou/extract_multiorbits_transfer_features.py
python Kode_python/R1_C2__R2_C1_C2_C3_C4_C5__Transfer_NCA_to_Multiorbits_XGBoost_deep_ensemble.py
```

The complete deep-ensemble LOCO experiments can take substantial time. Random
seeds, model parameters, feature counts, calibration fractions, and output paths
are recorded in the scripts and the corresponding `experiment_metadata.json`
files.

## Leakage-control design

- complete cells, packs, or units define the outer evaluation partitions;
- feature ranking is recomputed from fitting units within every outer fold;
- clipping thresholds are fitted only on the relevant training partition;
- calibration units are disjoint from fitting and held-out test units;
- test labels are used only after predictions and intervals have been fixed;
- target-only, source-only matched, naive pooled, and raw continuation use
  identical folds, selected features, evaluation rows, and model capacity.

## Archived outputs

The `results/` directory contains aggregate metrics, fold-level metrics,
feature rankings, statistical comparisons, metadata, summaries, and figures.
Row-level out-of-fold prediction files and reconstructed Wenzhou feature tables
are intentionally excluded to avoid redistributing modified third-party data.

## Validation before release

Run:

```bash
python check_repository.py
```

This checks Python syntax, required scripts, accidental data files, and the
presence of key documentation. It does not rerun the computational experiments.

## Citation

Please cite the associated E-Prime article after its final bibliographic details
and DOI become available. A machine-readable template is provided in
[`CITATION.cff`](CITATION.cff).

## Licence

The authors' source code in this repository is released under the MIT Licence.
This licence does not apply to any third-party battery dataset. Dataset users
must comply with the terms stated by each original provider.
