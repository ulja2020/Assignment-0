from copy import deepcopy

from data_tools import check_data_integrity, generate_maize_dataset


NUMERICAL_COLUMNS = (
    "plant_height_cm",
    "leaf_damage_percent",
    "moisture_sensor_percent",
    "ear_length_cm",
    "grain_mass_per_ear_g",
)
EXPECTED_COLUMNS = NUMERICAL_COLUMNS + ("field_zone",)
EXPECTED_DTYPES = {
    "plant_height_cm": "numeric",
    "leaf_damage_percent": "numeric",
    "moisture_sensor_percent": "numeric",
    "ear_length_cm": "numeric",
    "grain_mass_per_ear_g": "numeric",
    "field_zone": "categorical",
}
EXPECTED_RANGES = {
    "plant_height_cm": (170, 310),
    "leaf_damage_percent": (0, 100),
    "moisture_sensor_percent": (0, 100),
    "ear_length_cm": (10, 28),
    "grain_mass_per_ear_g": (40, 300),
}
EXPECTED_CATEGORIES = {
    "field_zone": (
        "Northwest",
        "Northeast",
        "Centre",
        "Southwest",
        "Southeast",
    ),
}
EXPECTED_SHAPE = (600, len(EXPECTED_COLUMNS))


def check(dataset):
    """Run the integrity checker with the project expectations."""
    return check_data_integrity(
        dataset,
        expected_columns=EXPECTED_COLUMNS,
        expected_shape=EXPECTED_SHAPE,
        expected_dtypes=EXPECTED_DTYPES,
        expected_ranges=EXPECTED_RANGES,
        expected_categories=EXPECTED_CATEGORIES,
        expected_sample_count=600,
        data_name="test dataset",
    )


def create_corrupted_dataset(valid_dataset):
    """Return a deep copy containing the four required Task 3 corruptions."""
    corrupted_dataset = deepcopy(valid_dataset)

    corrupted_dataset["sample_001"]["plant_height_cm"] = float("nan")
    corrupted_dataset["sample_200"]["plant_height_cm"] = float("nan")
    corrupted_dataset["sample_400"]["plant_height_cm"] = float("nan")

    corrupted_dataset["sample_006"] = deepcopy(corrupted_dataset["sample_005"])
    corrupted_dataset["sample_003"]["leaf_damage_percent"] = 150

    for sample_id in ("sample_004", "sample_150", "sample_300", "sample_450"):
        del corrupted_dataset[sample_id]["ear_length_cm"]

    return corrupted_dataset


def test_valid_data_pass():
    """Check that correctly generated data pass all supplied integrity rules."""
    data = generate_maize_dataset(output_filename=None)
    report = check(data)

    assert report["valid"] is True
    assert report["errors"] == []


def test_nan_detection():
    """Check that a deliberately inserted NaN is detected."""
    data = generate_maize_dataset(output_filename=None)
    data["sample_001"]["plant_height_cm"] = float("nan")

    report = check(data)

    assert report["valid"] is False
    assert any(
        error["code"] == "non_finite_value"
        and error["field"] == "plant_height_cm"
        for error in report["errors"]
    )


def test_duplicate_detection():
    """Check that an identical observation is detected as a duplicate."""
    data = generate_maize_dataset(output_filename=None)
    data["sample_006"] = deepcopy(data["sample_005"])

    report = check(data)

    assert report["valid"] is False
    assert any(
        error["code"] == "duplicate_observation"
        and error.get("sample_id") == "sample_006"
        for error in report["errors"]
    )


def test_impossible_value_detection():
    """Check that an impossible percentage value is detected."""
    data = generate_maize_dataset(output_filename=None)
    data["sample_003"]["leaf_damage_percent"] = 150

    report = check(data)

    assert report["valid"] is False
    assert any(
        error["code"] == "out_of_range"
        and error.get("sample_id") == "sample_003"
        and error.get("field") == "leaf_damage_percent"
        for error in report["errors"]
    )


def test_missing_column_detection():
    """Check that a removed required column is detected."""
    data = generate_maize_dataset(output_filename=None)
    del data["sample_004"]["ear_length_cm"]

    report = check(data)

    assert report["valid"] is False
    assert any(
        error["code"] == "missing_column"
        and error.get("sample_id") == "sample_004"
        and error.get("field") == "ear_length_cm"
        for error in report["errors"]
    )


def test_all_required_corruptions_together():
    """Check that all planned corruptions are detected in one dataset."""
    data = generate_maize_dataset(output_filename=None)
    corrupted = create_corrupted_dataset(data)

    report = check(corrupted)

    assert report["valid"] is False
    assert len(report["errors"]) == 9
