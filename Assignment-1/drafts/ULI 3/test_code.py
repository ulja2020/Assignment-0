from pathlib import Path

from runpy import run_path

import numpy as np

from model_tools import fit_mean_baseline, split_regression_data


def test_complete_workflow(tmp_path):
    """Run the workflow on a small input without opening figures or saving data."""
    module_path = Path(__file__).resolve().parent / "code.py"
    project_code = run_path(str(module_path))

    result = project_code["main"](
        number_of_observations=30,
        seed=42,
        output_directory=tmp_path,
        save_dataset=False,
    )

    assert result["integrity_report"]["valid"] is True
    assert len(result["dataset"]) == 30
    assert set(result["statistics"]) == set(project_code["NUMERICAL_FEATURES"])
    assert len(result["figure_paths"]) == 19

    for figure_path in result["figure_paths"]:
        assert Path(figure_path).exists()

    assert isinstance(result["task7"]["triangle_holds"], bool)


def test_task7_reference_data_shows_triangle_failure(tmp_path):
    """Check that the fixed 600-observation dataset gives a triangle counterexample."""
    module_path = Path(__file__).resolve().parent / "code.py"
    project_code = run_path(str(module_path))
    result = project_code["main"](
        number_of_observations=600,
        seed=42,
        output_directory=tmp_path,
        save_dataset=False,
    )

    assert result["task7"]["triangle_holds"] is False
    assert result["task7"]["triangle_right"] > result["task7"]["triangle_left"]


def test_regression_split_has_no_data_overlap():
    """Check that training, validation and final test observations are disjoint."""
    X = np.arange(100).reshape(50, 2)
    y = np.arange(50, dtype=float)

    split = split_regression_data(
        X,
        y,
        test_size=0.20,
        validation_size=0.20,
        random_seed=42,
    )

    train = set(split["train_indices"])
    validation = set(split["validation_indices"])
    test = set(split["test_indices"])

    assert train.isdisjoint(validation)
    assert train.isdisjoint(test)
    assert validation.isdisjoint(test)
    assert train | validation | test == set(range(50))


def test_regression_baseline_does_not_use_validation_or_test_targets():
    """Check that the baseline predicts the training target mean only."""
    y_train = np.array([10.0, 20.0, 30.0, 40.0])
    y_test = np.array([100.0, 110.0])
    baseline = fit_mean_baseline(y_train)

    predictions = baseline.predict(np.zeros((len(y_test), 1)))

    assert np.allclose(predictions, np.mean(y_train))
    assert not np.isclose(np.mean(y_train), np.mean(np.concatenate([y_train, y_test])))


def test_task9_outputs_use_same_final_test_set(tmp_path):
    """Check that Task 9 evaluates all three models on the same held-out test target."""
    module_path = Path(__file__).resolve().parent / "code.py"
    project_code = run_path(str(module_path))
    result = project_code["main"](
        number_of_observations=30,
        seed=42,
        output_directory=tmp_path,
        save_dataset=False,
    )

    test_size = len(result["task9"]["splits"]["y_test"])
    for model_name in ("baseline", "sensible", "wrong"):
        assert len(result["task9"]["predictions"]["test"][model_name]) == test_size
        assert np.all(np.isfinite(result["task9"]["predictions"]["test"][model_name]))
