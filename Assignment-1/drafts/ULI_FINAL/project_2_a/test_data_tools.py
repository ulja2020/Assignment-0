"""Task 10 generic validation tests for dataset integrity."""

from copy import deepcopy
import unittest

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
    **{column: "numeric" for column in NUMERICAL_COLUMNS},
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
        "Northwest", "Northeast", "Centre", "Southwest", "Southeast"
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
    """Create the NaNs, duplicate, impossible value and missing columns."""
    corrupted = deepcopy(valid_dataset)
    for sample_id in ("sample_001", "sample_200", "sample_400"):
        corrupted[sample_id]["plant_height_cm"] = float("nan")
    corrupted["sample_006"] = deepcopy(corrupted["sample_005"])
    corrupted["sample_003"]["leaf_damage_percent"] = 150
    for sample_id in ("sample_004", "sample_150", "sample_300", "sample_450"):
        del corrupted[sample_id]["ear_length_cm"]
    return corrupted


class TestDataIntegrity(unittest.TestCase):
    """Verify valid data pass and known corruptions are rejected."""

    def test_valid_data_pass(self):
        """Check correctly generated data against every integrity rule."""
        report = check(generate_maize_dataset(output_filename=None))
        self.assertTrue(report["valid"])
        self.assertEqual(report["errors"], [])

    def test_all_required_corruptions_together(self):
        """Task 10: demonstrate detection of deliberately broken input."""
        valid = generate_maize_dataset(output_filename=None)
        report = check(create_corrupted_dataset(valid))
        error_codes = {error["code"] for error in report["errors"]}

        self.assertFalse(report["valid"])
        self.assertEqual(len(report["errors"]), 9)
        self.assertTrue(
            {
                "non_finite_value",
                "duplicate_observation",
                "out_of_range",
                "missing_column",
            }.issubset(error_codes)
        )


if __name__ == "__main__":
    unittest.main()