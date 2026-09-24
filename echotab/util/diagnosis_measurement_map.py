from typing import Dict, List

import pandas as pd


def load_diagnosis_measurement_map(csv_path: str) -> Dict[str, List[str]]:
    """Load the diagnosis -> measurement-name mapping.

    The CSV has columns "diagnosis", "measurement": one row per diagnosis, with
    "measurement" holding a comma-separated list of measurement names in the
    phenotype vocabulary of the pretraining phenotype list.

    Returns: {diagnosis: [measurement_name, ...]}.
    """
    df = pd.read_csv(csv_path)
    for col in ("diagnosis", "measurement"):
        if col not in df.columns:
            raise ValueError(f"{csv_path} is missing required column {col!r}")

    mapping: Dict[str, List[str]] = {}
    for _, row in df.iterrows():
        diagnosis = row["diagnosis"]
        cell = row["measurement"]
        if pd.isna(cell) or not str(cell).strip():
            raise ValueError(
                f"{csv_path}: empty measurement list for diagnosis {diagnosis!r}"
            )
        names = [n.strip() for n in str(cell).split(",")]
        if any(not n for n in names):
            raise ValueError(
                f"{csv_path}: malformed measurement list for diagnosis {diagnosis!r}: {cell!r}"
            )
        mapping[diagnosis] = names

    return mapping
