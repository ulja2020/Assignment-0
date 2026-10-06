"""Independently check the saved synthetic and corrupted maize datasets.

This file runs independently. Its inputs are the two existing dataset files,
and it uses ``check_data_integrity`` from ``data_tools.py`` to confirm that the
synthetic data pass and the corrupted data fail with the nine expected problems.
"""

from pathlib import Path
from runpy import run_path

from data_tools import check_data_integrity


def test_complete_workflow():
    """Load both saved datasets and confirm their expected integrity results."""
    folder = Path(__file__).resolve().parent
    synthetic_path = folder / "data_tools_output_synthetic.py"
    corrupted_path = folder / "data_tools_output_corrupted.py"

    synthetic_dataset = run_path(str(synthetic_path))["maize_harvest_data"]
    corrupted_dataset = run_path(str(corrupted_path))["maize_harvest_data"]

    synthetic_report = check_data_integrity(
        synthetic_dataset,
        "data_tools_output_synthetic.py",
    )
    corrupted_report = check_data_integrity(
        corrupted_dataset,
        "data_tools_output_corrupted.py",
    )

    assert isinstance(synthetic_dataset, dict)
    assert isinstance(corrupted_dataset, dict)
    assert len(synthetic_dataset) == 600
    assert len(corrupted_dataset) == 600
    assert synthetic_report["valid"] is True
    assert synthetic_report["errors"] == []
    assert corrupted_report["valid"] is False
    assert len(corrupted_report["errors"]) == 9


if __name__ == "__main__":
    test_complete_workflow()
