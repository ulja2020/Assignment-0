from pathlib import Path

from runpy import run_path


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
    assert len(result["figure_paths"]) == 16

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
