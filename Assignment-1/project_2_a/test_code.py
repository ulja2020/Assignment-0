"""Task 10 complete-workflow and regression-leakage validation tests."""

from contextlib import contextmanager, redirect_stdout
from copy import deepcopy
from io import StringIO
from pathlib import Path
from runpy import run_path
import shutil
import unittest

import numpy as np

from data_tools import generate_maize_dataset
from model_tools import split_regression_data


MODULE_DIRECTORY = Path(__file__).resolve().parent
PROJECT_CODE = run_path(str(MODULE_DIRECTORY / "code.py"))


@contextmanager
def temporary_output_directory(name):
    """Create and remove one known test-output folder beside this file."""
    root = MODULE_DIRECTORY / "_task10_test_outputs"
    directory = root / name
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    try:
        yield directory
    finally:
        shutil.rmtree(directory, ignore_errors=True)
        if root.exists() and not any(root.iterdir()):
            root.rmdir()


class TestCompleteWorkflow(unittest.TestCase):
    """Verify that project components connect and keep test rows isolated."""

    def test_complete_workflow(self):
        """Task 10: run every task on a small reproducible dataset."""
        with temporary_output_directory("complete") as directory:
            with redirect_stdout(StringIO()):
                result = PROJECT_CODE["main"](
                    number_of_observations=30,
                    seed=42,
                    output_directory=directory,
                    save_dataset=False,
                )

            self.assertTrue(result["integrity_report"]["valid"])
            self.assertEqual(len(result["dataset"]), 30)
            self.assertEqual(
                set(result["statistics"]),
                set(PROJECT_CODE["NUMERICAL_FEATURES"]),
            )
            self.assertEqual(len(result["figure_paths"]), 16)
            for path in result["figure_paths"]:
                self.assertTrue(Path(path).exists())

            for method in ("kmeans", "gmm"):
                labels = result["task8"][method]["labels"]
                sizes = result["task8"][method]["evaluation"]["cluster_sizes"]
                self.assertEqual(len(labels), 30)
                self.assertEqual(sum(sizes.values()), 30)

            task9 = result["task9"]
            self.assertEqual(sum(task9["split_sizes"].values()), 30)
            self.assertEqual(
                set(task9["test_evaluations"]),
                {"Baseline", "Sensible model", "Wrong ear-length-only model"},
            )
            for path in task9["figure_paths"]:
                self.assertTrue(Path(path).exists())

    def test_regression_test_targets_do_not_leak_into_fitting(self):
        """Task 10: changing only test targets must not change fitted models."""
        dataset = generate_maize_dataset(
            number_of_observations=120,
            seed=42,
            output_filename=None,
        )
        split = split_regression_data(120, random_seed=42)
        test_sample_ids = [list(dataset)[index] for index in split["test"]]
        changed = deepcopy(dataset)
        for sample_id in test_sample_ids:
            changed[sample_id]["grain_mass_per_ear_g"] += 50.0

        with temporary_output_directory("leakage") as directory:
            with redirect_stdout(StringIO()):
                original = PROJECT_CODE["run_task9"](
                    dataset,
                    output_directory=directory,
                )
                altered = PROJECT_CODE["run_task9"](
                    changed,
                    output_directory=directory,
                )

        for model_name in ("sensible", "wrong"):
            first = original["models"][model_name]
            second = altered["models"][model_name]
            self.assertTrue(
                np.allclose(first.coef_, second.coef_, atol=1e-12)
            )
            self.assertTrue(
                np.isclose(first.intercept_, second.intercept_, atol=1e-12)
            )
        self.assertTrue(
            np.isclose(
                original["models"]["baseline"].constant_,
                altered["models"]["baseline"].constant_,
                atol=1e-12,
            )
        )
        self.assertFalse(
            np.isclose(
                original["test_evaluations"]["Sensible model"]["rmse"],
                altered["test_evaluations"]["Sensible model"]["rmse"],
                atol=1e-12,
            )
        )

    def test_task7_reference_data_shows_triangle_failure(self):
        """Retain the known Task 7 counterexample in the complete workflow."""
        with temporary_output_directory("task7") as directory:
            with redirect_stdout(StringIO()):
                result = PROJECT_CODE["main"](
                    number_of_observations=600,
                    seed=42,
                    output_directory=directory,
                    save_dataset=False,
                )

        self.assertFalse(result["task7"]["triangle_holds"])
        self.assertGreater(
            result["task7"]["triangle_right"],
            result["task7"]["triangle_left"],
        )


if __name__ == "__main__":
    unittest.main()
