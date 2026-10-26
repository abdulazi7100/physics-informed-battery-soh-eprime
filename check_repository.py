"""Static pre-release checks; this script does not run model training."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REQUIRED = [
    "README.md",
    "DATA_SOURCES.md",
    "requirements.txt",
    "LICENSE",
    "CITATION.cff",
    "Kode_python/R2_C1_C2_C3_C4_C5__Transfer_NCA_to_NMC_XGBoost_ranked_groups_deep_ensemble.py",
    "Kode_python/R2_C1_C2_C3_C4_C5__Transfer_NMC_to_NCA_XGBoost_ranked_groups_deep_ensemble.py",
    "Kode_python/R1_C1__R2_C1_C2_C3_C4_C5__Transfer_Wenzhou_Cell_to_Pack_XGBoost_deep_ensemble.py",
    "Kode_python/R1_C2__R2_C1_C2_C3_C4_C5__Transfer_NCA_to_Multiorbits_XGBoost_deep_ensemble.py",
    "Kode_python/R1_C2__R2_C1_C2_C3_C4_C5__Transfer_NMC_to_Multiorbits_XGBoost_deep_ensemble.py",
    "Kode_python/R1_C3__R2_C1_C2_C3_C4_C5__Wenzhou_Randomized_cumulative_lifetime_XGBoost_deep_ensemble.py",
    "Kode_python/R1_C4__R2_C1_C2_C3_C4_C5__Transfer_NCA_to_Sodium_XGBoost_deep_ensemble.py",
    "Kode_python/R1_C4__R2_C1_C2_C3_C4_C5__Transfer_NMC_to_Sodium_XGBoost_deep_ensemble.py",
]


def main() -> None:
    missing = [name for name in REQUIRED if not (ROOT / name).is_file()]
    if missing:
        raise SystemExit("Missing required files:\n- " + "\n- ".join(missing))

    syntax_errors: list[str] = []
    for script in ROOT.rglob("*.py"):
        try:
            ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
        except (SyntaxError, UnicodeDecodeError) as exc:
            syntax_errors.append(f"{script.relative_to(ROOT)}: {exc}")
    if syntax_errors:
        raise SystemExit("Python syntax errors:\n- " + "\n- ".join(syntax_errors))

    forbidden = []
    for pattern in ("*.mat", "*.h5", "*.hdf5", "*.xlsx", "*.xls", "*.rar"):
        forbidden.extend(path.relative_to(ROOT) for path in ROOT.rglob(pattern))
    non_result_csv = [
        path.relative_to(ROOT)
        for path in ROOT.rglob("*.csv")
        if "results" not in path.relative_to(ROOT).parts
    ]
    forbidden.extend(non_result_csv)
    if forbidden:
        raise SystemExit(
            "Possible research data found outside aggregate results:\n- "
            + "\n- ".join(map(str, forbidden))
        )

    scripts = sum(1 for _ in ROOT.rglob("*.py"))
    result_files = sum(1 for path in (ROOT / "results").rglob("*") if path.is_file())
    print(f"Repository check passed: {scripts} Python files, {result_files} result files.")
    print("No raw-data file type was detected.")


if __name__ == "__main__":
    main()
