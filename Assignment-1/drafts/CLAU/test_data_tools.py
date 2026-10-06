"""Create valid and corrupted datasets and test the integrity checker.

The synthetic dataset is generated first and saved normally. A deep copy is
then deliberately corrupted in several ways and saved separately. Loading both
files before checking them verifies the actual datasets stored on disk.
"""

from copy import deepcopy
from pathlib import Path
from runpy import run_path

from data_tools import (
    check_data_integrity,
    generate_maize_dataset,
    save_maize_dataset,
)


def create_corrupted_dataset(valid_dataset):
    """Return a copy containing the corruptions required by Task 3.

    The corruptions are three NaN values, one duplicated observation, one
    impossible percentage and four observations with a required column removed.
    Samples used for the duplicate are not modified by the other corruptions.
    """
    corrupted_dataset = deepcopy(valid_dataset)

    for sample_id in ("sample_001", "sample_200", "sample_400"):
        corrupted_dataset[sample_id]["plant_height_cm"] = float("nan")

    corrupted_dataset["sample_006"] = deepcopy(corrupted_dataset["sample_005"])
    corrupted_dataset["sample_003"]["leaf_damage_percent"] = 150

    for sample_id in ("sample_004", "sample_150", "sample_300", "sample_450"):
        del corrupted_dataset[sample_id]["ear_length_cm"]

    return corrupted_dataset


def test_valid_and_corrupted_datasets():
    """Confirm that valid saved data pass and all planned corruptions fail."""
    valid_dataset = generate_maize_dataset()
    corrupted_dataset = create_corrupted_dataset(valid_dataset)

    save_maize_dataset(
        corrupted_dataset,
        "data_tools_output_corrupted.py",
        "Synthetic maize harvest data deliberately corrupted for integrity tests.",
    )

    folder = Path(__file__).resolve().parent
    corrupted_from_file = run_path(str(folder / "data_tools_output_corrupted.py"))[
        "maize_harvest_data"
    ]
    valid_from_file = run_path(str(folder / "data_tools_output_synthetic.py"))[
        "maize_harvest_data"
    ]

    corrupted_report = check_data_integrity(
        corrupted_from_file,
        "data_tools_output_corrupted.py",
    )
    valid_report = check_data_integrity(
        valid_from_file,
        "data_tools_output_synthetic.py",
    )

    assert corrupted_report["valid"] is False
    assert len(corrupted_report["errors"]) == 9
    assert valid_report["valid"] is True
    assert valid_report["errors"] == []

    nan_ids = {
        error.get("sample_id")
        for error in corrupted_report["errors"]
        if error["code"] == "non_finite_value"
    }
    assert nan_ids == {"sample_001", "sample_200", "sample_400"}

    assert any(
        error["code"] == "duplicate_observation"
        and error.get("sample_id") == "sample_006"
        for error in corrupted_report["errors"]
    )
    assert any(
        error["code"] == "out_of_range"
        and error.get("sample_id") == "sample_003"
        and error.get("field") == "leaf_damage_percent"
        for error in corrupted_report["errors"]
    )

    missing_column_ids = {
        error.get("sample_id")
        for error in corrupted_report["errors"]
        if error["code"] == "missing_column" and error.get("field") == "ear_length_cm"
    }
    assert missing_column_ids == {"sample_004", "sample_150", "sample_300", "sample_450"}


if __name__ == "__main__":
    test_valid_and_corrupted_datasets()
