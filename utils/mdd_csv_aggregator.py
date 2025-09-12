"""Standalone script to aggregate all CSV files in Source_MDDs into a single
CSV file inside Output_ref_mdd.

Simply run:

    python mdd_csv_aggregator.py

from the `Check_logic_collated_MDD` directory (or provide the full path). The
script detects the input/output folders relative to its own location so you
don't need to pass any arguments.

Generated file: `Output_ref_mdd/aggregated_checks.csv`
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List
logger = logging.getLogger(__name__)
import pandas as pd
# ---------------------------------------------------------------------------
# Field alias mapping --------------------------------------------------------
# Each canonical key maps to all possible column name variations seen in source
# MDD CSV files. Extend this list as needed.
FIELD_ALIASES: Dict[str, List[str]] = {
    # Sponsor / study identifiers
    "sponsor_name": [
        "Sponsor", "Sponsor Name", "sponsor_name", "sponsorname",
    ],
    "study_id": [
        "Study", "Study ID", "Source Study ID", "study_id", "studyid","Study"
    ],
    # Data-quality identifiers
    "dq_name": [
        "DQ Name", "DQ name", "Check Name", "Plan Name", "name", "dq_name","Check name"
    ],
    # Pseudo / check logic text
    "pseudo_code": [
        "Pseudo Code", "pseudo code", "Check logic", "Check Logic",
        "logic", "pseudo_code","DQ Logic"
    ],
}

_INVALID_STRS = {"", "n/a", "na", "nan", "none"}

# Canonical output columns (order preserved)
OUTPUT_COLUMNS = [
    "File_Name",
    "sponsor_name",
    "study_id",
    "dq_name",
    "check_logic",  # unified column name for pseudo / check logic text
]


# ---------------------------------------------------------------------------
# Utility helpers ------------------------------------------------------------

def _norm(val: Any) -> str:
    """Normalise *val* to lowercase string for comparison."""
    return str(val).strip().lower()


def get_field_value(row: Dict[str, Any], canonical_key: str) -> str:
    """Return the first non-empty value in *row* matching *canonical_key* aliases."""
    aliases = FIELD_ALIASES.get(canonical_key, [canonical_key])
    for alias in aliases:
        value = row.get(alias, "")
        if value and _norm(value) not in _INVALID_STRS:
            return str(value).strip()
    return ""


# ---------------------------------------------------------------------------
# CSV processing -------------------------------------------------------------

def list_csv_files(input_root: Path) -> List[Path]:
    """Recursively collect all `.csv` files under *input_root*."""
    csv_paths: List[Path] = []
    for dirpath, _dirnames, filenames in os.walk(input_root):
        for fname in filenames:
            if fname.lower().endswith(".csv"):
                csv_paths.append(Path(dirpath) / fname)
    return csv_paths


def extract_row_values(row: Dict[str, str], folder_name: str, file_stem: str) -> Dict[str, str]:
    """Return dict for one output row with fallbacks for missing values."""
    file_name = file_stem
    sponsor = get_field_value(row, "sponsor_name") or folder_name
    study = get_field_value(row, "study_id") or file_stem
    dq_name = get_field_value(row, "dq_name") or "UNKNOWN"
    logic_txt = get_field_value(row, "pseudo_code") or "N/A"

    return {
        "File_Name": file_name,
        "sponsor_name": sponsor,
        "study_id": study,
        "dq_name": dq_name,
        "check_logic": logic_txt,
    }


def process_csv(csv_path: Path) -> List[Dict[str, str]]:
    """Read *csv_path* and convert its rows to the unified format."""
    folder_name = csv_path.parent.name
    file_stem = csv_path.stem

    try:
        # Peek first two rows to decide header row.
        # logger.info(f"Rsline110: folder_name: {folder_name}, file_stem: {file_stem}")
        peek = pd.read_csv(csv_path, dtype=str, keep_default_na=False, header=None, nrows=2)
        first_row_vals = [str(v).strip().lower() for v in peek.iloc[0].tolist()]
        # If first row does NOT contain any header-like keywords, use second row as header
        keywords = ("description", "query")  # covers 'DQ Description' and similar
        has_header_keywords = any(val and any(k in val for k in keywords) for val in first_row_vals)
        header_row = 0 if has_header_keywords else 1
        # logger.info(f"Rsline114: header_row: {header_row}")
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, header=header_row)
        # logger.info(f"Rsline116: df: {df}")
        # Normalize column names for robust alias matching
        df.columns = [str(c).strip() for c in df.columns]
        # logger.info(f"Rsline118: df.columns: {df.columns}")
    except Exception as exc:
        logger.error("Failed to read %s: %s", csv_path, exc)
        return []

    rows: List[Dict[str, str]] = []
    for _, row in df.iterrows():
        rows.append(extract_row_values(row.to_dict(), folder_name, file_stem))
    return rows


def aggregate_csvs(input_root: Path) -> pd.DataFrame:
    """Aggregate all source CSVs into a single DataFrame."""
    csv_files = list_csv_files(input_root)
    logging.info("Found %d CSV files", len(csv_files))

    all_rows: List[Dict[str, str]] = []
    for csv_path in csv_files:
        logging.debug("Processing %s", csv_path)
        all_rows.extend(process_csv(csv_path))

    df_out = pd.DataFrame(all_rows)
    # Guarantee all expected columns exist
    for col in OUTPUT_COLUMNS:
        if col not in df_out.columns:
            df_out[col] = "N/A"
    df_out = df_out[OUTPUT_COLUMNS]
    df_out.drop_duplicates(subset=OUTPUT_COLUMNS, inplace=True)
    return df_out


# ---------------------------------------------------------------------------
# Entry point ----------------------------------------------------------------

# def main() -> None:
#     logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

#     root = Path(__file__).resolve().parent
#     input_root = root / "Source_MDDs"
#     output_dir = root / "Output_ref_mdd"
#     output_dir.mkdir(parents=True, exist_ok=True)

#     df = aggregate_csvs(input_root)

#     output_path = output_dir / "aggregated_checks.csv"
#     df.to_csv(output_path, index=False)
#     logging.info("Aggregated CSV written to %s (%d rows)", output_path, len(df))


# if __name__ == "__main__":  # pragma: no cover
#     main()
