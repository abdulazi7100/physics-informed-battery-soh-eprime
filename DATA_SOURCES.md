# Data sources and access conditions

Raw and processed battery observations are not included in this repository.

## TJU NMC/NCA ageing data

- Dataset: *Data-driven capacity estimation of commercial lithium-ion batteries
  from voltage relaxation*
- Repository: https://doi.org/10.5281/zenodo.6405084
- Required local model inputs:
  `Kode_python/NCA_Integral_XGBoost_38.csv` and
  `Kode_python/NMC_Integral_XGBoost.csv`
- These feature tables are not included. They must be reconstructed from the
  source data using the indicator definitions reported in the paper.

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
