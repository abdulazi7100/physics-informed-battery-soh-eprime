# Data sources and access conditions

Raw and processed battery observations are not included in this repository.

## Derived TJU feature tables

The following processed modelling inputs are included:

- `Kode_python/NCA_Integral_XGBoost_38.csv`
- `Kode_python/NMC_Integral_XGBoost.csv`

These tables were derived from the publicly available TJU NMC/NCA
battery ageing dataset:

J. Zhu et al., Data-driven capacity estimation of commercial
lithium-ion batteries from voltage relaxation,
https://doi.org/10.5281/zenodo.6405084.

The original dataset is distributed under the Creative Commons
Attribution 4.0 International licence. The included tables are not
original raw measurements. They contain health indicators extracted
and transformed according to the methods described in the associated
article. Changes include feature extraction, cycle-level aggregation,
SOH construction, and preparation for machine-learning analysis.

## Wenzhou Pack Degradation Data

- Official repository:
  https://github.com/lvdongzhen/Wenzhou-Pack-Degradation-Data
- Feature extractor:
  `kode_pack_wenzhou/extract_wenzhou_cell_pack_features.py`

## Wenzhou Orbital Degradation Data

- Official repository:
  https://github.com/lvdongzhen/Wenzhou-Orbital-Degradation-Data
- Feature extractor:
  `kode_multiorbits_wenzhou/extract_multiorbits_transfer_features.py`

## Wenzhou Randomized Battery Data

- Official repository:
  https://github.com/lvdongzhen/Wenzhou-Randomized-Battery-Data
- Extraction script:
  `kode_feature_extraction_randomized_wenzhou/extract_randomized_features_per_unit.py`

## Wenzhou Sodium-ion Battery Degradation Data

- Official repository:
  https://github.com/lvdongzhen/Wenzhou-Sodium-ion-Battery-Degradation-Data
- Feature extractor:
  `kode_feature_extraction_sodium_wenzhou/extract_wenzhou_sodium_features.py`

## Wenzhou licence restriction

The official repositories state that the Wenzhou data are available under the
Creative Commons Attribution-NoDerivatives 4.0 International licence. Users
should download the original material directly, preserve attribution, and avoid
redistributing modified, processed, restructured, or repackaged copies. The code
licence in this repository does not override the dataset licence.
