"""Data tools for MOD550 Assignment 1 Task 12.

The checks are general and receive a DataFrame as input. The download function
keeps a local cached copy so repeated runs do not download the same data again.
"""

from datetime import datetime
from pathlib import Path
import json
import urllib.request
import zipfile

import numpy as np
import pandas as pd


FAOSTAT_BULK_URL = (
    "https://bulks-faostat.fao.org/production/"
    "Food_Security_Data_E_All_Data_(Normalized).zip"
)

REQUIRED_RAW_COLUMNS = [
    "Area", "Year", "Unit", "Value", "Item", "Flag", "Flag Description"
]


def download_faostat_data(
    output_file="data/FAOSTAT_food_security_raw.csv",
    metadata_file="data/faostat_download_metadata.json",
    source_file=None,
    url=FAOSTAT_BULK_URL,
):
    """Load FAOSTAT from a local cache, a supplied CSV, or the official bulk URL.

    A small JSON file is saved next to the data with the source and time at
    which this project recorded the local copy.
    """
    project_directory = Path(__file__).resolve().parent
    output_path = project_directory / output_file
    metadata_path = project_directory / metadata_file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        print("Using cached FAOSTAT data. No download was needed.")
        return pd.read_csv(output_path, encoding="utf-8-sig")

    if source_file is not None:
        source_path = Path(source_file).expanduser().resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"Source file was not found: {source_path}")
        data = pd.read_csv(source_path, encoding="utf-8-sig")
        data.to_csv(output_path, index=False)
        download_type = "student-provided downloaded CSV"
    else:
        print("Downloading FAOSTAT data...")
        zip_path = output_path.with_suffix(".zip")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as archive:
            csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if not csv_names:
                raise ValueError("The FAOSTAT archive did not contain a CSV file.")
            with archive.open(csv_names[0]) as file:
                data = pd.read_csv(file, encoding="latin-1")
        data.to_csv(output_path, index=False)
        zip_path.unlink(missing_ok=True)
        download_type = "official FAOSTAT bulk download"

    metadata = {
        "source": "FAOSTAT Suite of Food Security Indicators",
        "source_url": url,
        "download_type": download_type,
        "recorded_at": datetime.now().astimezone().isoformat(),
        "rows": int(len(data)),
        "columns": int(len(data.columns)),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved raw data to: {output_path}")
    print(f"Saved metadata to: {metadata_path}")
    return data


def check_data_integrity(
    data,
    expected_columns=None,
    expected_shape=None,
    required_non_missing=None,
):
    """Check structural properties of a DataFrame.

    The function checks required columns, optional shape, duplicates and
    missing values in required identifier fields. It does not decide whether
    numerical values are scientifically meaningful or whether a model is good.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    if expected_columns is None:
        expected_columns = list(data.columns)

    missing_columns = [column for column in expected_columns if column not in data.columns]
    duplicate_rows = int(data.duplicated().sum())
    missing_values = int(data.isna().sum().sum())

    shape_valid = expected_shape is None or data.shape == expected_shape

    missing_required = {}
    if required_non_missing is not None:
        for column in required_non_missing:
            if column in data.columns:
                missing_required[column] = int(data[column].isna().sum())
            else:
                missing_required[column] = None

    required_valid = all(value == 0 for value in missing_required.values())
    passed = len(missing_columns) == 0 and shape_valid and duplicate_rows == 0 and required_valid

    report = {
        "valid": passed,
        "missing_columns": missing_columns,
        "actual_shape": data.shape,
        "expected_shape": expected_shape,
        "shape_valid": shape_valid,
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
        "missing_required_fields": missing_required,
    }

    if passed:
        print("Integrity check passed.")
    else:
        print("Integrity check failed.")
        print(report)

    return report


def prepare_food_security_data(data, selected_items):
    """Turn the selected long-format FAOSTAT items into one row per Area-period.

    Numeric values are converted from the raw ``Value`` column. Values written
    as ``<2.5`` are represented by the lower bound 2.5 for this analysis and
    this choice is reported to the user. Observations still missing a selected
    item are removed from the final complete-case analysis table.
    """
    data = data.copy()
    data["Value_num"] = pd.to_numeric(data["Value"], errors="coerce")

    lower_bound_mask = data["Value"].astype(str).str.strip().str.match(r"^<\s*[0-9.]+$")
    data.loc[lower_bound_mask, "Value_num"] = pd.to_numeric(
        data.loc[lower_bound_mask, "Value"].str.extract(r"([0-9.]+)")[0],
        errors="coerce",
    )

    data["period_end"] = (
        data["Year"].astype(str).str.extract(r"(\d{4})$")[0].astype(int)
    )

    selected = data[data["Item"].isin(selected_items)].copy()

    wide = selected.pivot_table(
        index=["Area", "period_end"],
        columns="Item",
        values="Value_num",
        aggfunc="first",
    ).reset_index()

    missing_cells = int(wide[selected_items].isna().sum().sum())
    complete = wide.dropna(subset=selected_items).copy()

    report = {
        "raw_rows": int(len(data)),
        "selected_rows": int(len(selected)),
        "lower_bound_values": int(lower_bound_mask.sum()),
        "missing_cells_after_pivot": missing_cells,
        "complete_observations": int(len(complete)),
        "areas": int(complete["Area"].nunique()),
        "periods": int(complete["period_end"].nunique()),
    }

    return complete, report
