from pathlib import Path
from pprint import pformat
import math

import numpy as np


FIELD_ZONE_ORDER = (
    "Northwest",
    "Northeast",
    "Centre",
    "Southwest",
    "Southeast",
)


# These are used by the maize generator itself.  The integrity checker does
# not depend on these values; code.py passes its expectations explicitly.
ZONE_HEIGHT_EFFECT = {
    "Northwest": 3.0,
    "Northeast": 0.0,
    "Centre": 5.0,
    "Southwest": -2.0,
    "Southeast": 2.0,
}

ZONE_DAMAGE_EFFECT = {
    "Northwest": -0.4,
    "Northeast": 0.2,
    "Centre": -0.6,
    "Southwest": 0.7,
    "Southeast": 0.0,
}


def generate_maize_dataset(
    number_of_observations=600,
    seed=42,
    output_filename="data_tools_output_synthetic.py",
):
    """
    Generate a reproducible plant-level dataset for a synthetic maize field.

    Parameters
    ----------
    number_of_observations : int, default=600
        Number of plant observations to generate.

    seed : int, default=42
        Seed used by NumPy's random number generator so that the generated
        dataset is reproducible.

    output_filename : str or None, default="data_tools_output_synthetic.py"
        Name of the optional Python data file written beside this module.
        If ``None``, no file is written.

    Returns
    -------
    dict
        A dictionary whose keys are sample IDs and whose values contain five
        numerical variables and one categorical variable.

    Dataset design
    --------------
    The dataset intentionally contains:

    - plant_height_cm: approximately symmetric numerical feature;
    - leaf_damage_percent: clearly right-skewed numerical feature;
    - moisture_sensor_percent: almost constant sensor feature;
    - ear_length_cm: approximately symmetric feature correlated with the target;
    - grain_mass_per_ear_g: numerical target depending on several other features;
    - field_zone: meaningful categorical variable describing five field areas.

    The target depends on ear length, plant height, leaf damage, moisture
    reading and random noise.  Field-zone effects alter some observations but
    the category is kept separate so it can be excluded from later numerical
    clustering inputs.

    Notes
    -----
    ``numpy.random.default_rng`` is used so the project has one reproducible
    random-number generator.  The exact seed and observation count therefore
    reproduce the same synthetic dataset.
    """
    if number_of_observations < 1:
        raise ValueError("number_of_observations must be positive.")

    rng = np.random.default_rng(seed)

    field_zone = rng.choice(
        FIELD_ZONE_ORDER,
        size=number_of_observations,
        p=[0.18, 0.20, 0.24, 0.18, 0.20],
    )

    height_effect = np.array(
        [ZONE_HEIGHT_EFFECT[zone] for zone in field_zone]
    )
    damage_effect = np.array(
        [ZONE_DAMAGE_EFFECT[zone] for zone in field_zone]
    )

    # Approximately symmetric numerical feature.
    plant_height_cm = (
        rng.normal(240, 18, number_of_observations)
        + height_effect
    )
    plant_height_cm = np.clip(plant_height_cm, 170, 310)

    # Clearly right-skewed numerical feature.
    leaf_damage_percent = (
        rng.lognormal(0.7, 0.9, number_of_observations)
        + damage_effect
    )
    leaf_damage_percent = np.clip(leaf_damage_percent, 0, 100)

    # Almost constant sensor.  The standard deviation is intentionally tiny.
    moisture_sensor_percent = (
        18
        + rng.normal(0, 0.005, number_of_observations)
    )

    # Approximately symmetric numerical feature.
    ear_length_cm = rng.normal(
        19,
        2,
        number_of_observations,
    )
    ear_length_cm = np.clip(ear_length_cm, 10, 28)

    # Numerical target with a known generating relationship.
    noise = rng.normal(0, 10, number_of_observations)
    grain_mass_per_ear_g = (
        8.5 * ear_length_cm
        + 0.12 * plant_height_cm
        - 0.9 * leaf_damage_percent
        + 0.8 * moisture_sensor_percent
        + noise
    )
    grain_mass_per_ear_g = np.clip(grain_mass_per_ear_g, 40, 300)

    dataset = {}
    for observation_number in range(number_of_observations):
        sample_id = f"sample_{observation_number + 1:03d}"
        dataset[sample_id] = {
            "plant_height_cm": round(
                float(plant_height_cm[observation_number]),
                2,
            ),
            "leaf_damage_percent": round(
                float(leaf_damage_percent[observation_number]),
                2,
            ),
            "moisture_sensor_percent": round(
                float(moisture_sensor_percent[observation_number]),
                3,
            ),
            "ear_length_cm": round(
                float(ear_length_cm[observation_number]),
                2,
            ),
            "grain_mass_per_ear_g": round(
                float(grain_mass_per_ear_g[observation_number]),
                2,
            ),
            "field_zone": str(field_zone[observation_number]),
        }

    if output_filename is not None:
        save_maize_dataset(
            dataset,
            output_filename,
            "Synthetic maize harvest data generated by data_tools.py.",
        )

    return dataset


def save_maize_dataset(dataset, output_filename, description):
    """
    Save a maize dictionary in an importable Python file.

    Parameters
    ----------
    dataset : dict
        Dictionary-of-dictionaries containing the generated observations.

    output_filename : str or Path
        Name of the Python file written beside this module.

    description : str
        Short module description placed at the top of the generated file.

    Notes
    -----
    The generated file defines the dataset as ``maize_harvest_data``.  NaN
    values are written as ``nan`` so deliberately corrupted test data can be
    loaded again while preserving the corruption.
    """
    output_path = Path(__file__).resolve().parent / output_filename
    output_text = f'"""{description}"""\n\n'
    output_text += "from math import nan\n\n"
    output_text += (
        "maize_harvest_data = "
        + pformat(dataset, sort_dicts=False, width=100)
        + "\n"
    )
    output_path.write_text(output_text, encoding="utf-8")


def check_data_integrity(
    data,
    expected_columns=None,
    expected_shape=None,
    expected_dtypes=None,
    expected_ranges=None,
    expected_categories=None,
    expected_sample_count=None,
    data_name="dataset",
):
    """
    Check general structural and validity properties of a dataset.

    Parameters
    ----------
    data : dict
        Dataset represented as a dictionary of observations. Each observation
        must itself be a dictionary.

    expected_columns : iterable, optional
        Columns that must be present in every observation. Unexpected columns
        are also reported.

    expected_shape : tuple, optional
        Expected ``(number_of_observations, number_of_columns)``. This is useful
        when the caller knows both dimensions in advance.

    expected_dtypes : dict, optional
        Expected broad data types. Values can be ``"numeric"``, ``"integer"``,
        ``"categorical"`` or ``"string"``.

    expected_ranges : dict, optional
        Mapping from numerical column names to ``(minimum, maximum)`` allowed
        values. Use ``None`` for an open lower or upper bound.

    expected_categories : dict, optional
        Mapping from categorical columns to the allowed category values.

    expected_sample_count : int, optional
        Expected number of observations. If supplied, this is checked
        separately from ``expected_shape``.

    data_name : str, default="dataset"
        Short label used in the printed summary.

    Returns
    -------
    dict
        A report containing the validity result, observation count and a
        structured list of detected errors.

    Assumptions
    -----------
    This function checks structural integrity and simple validity constraints
    supplied by the caller. It deliberately does not determine whether:

    - relationships between variables are scientifically correct;
    - distributions are appropriate;
    - a regression model is suitable; or
    - clusters are meaningful.

    Those questions belong to later analysis in the assignment.
    """
    report = {
        "valid": True,
        "data_name": data_name,
        "observation_count": 0,
        "errors": [],
    }

    def add_error(code, message, sample_id=None, field=None):
        """Add one structured integrity error to the report."""
        error = {"code": code, "message": message}
        if sample_id is not None:
            error["sample_id"] = sample_id
        if field is not None:
            error["field"] = field
        report["errors"].append(error)

    if not isinstance(data, dict):
        add_error(
            "invalid_dataset_type",
            "The dataset must be a dictionary.",
        )
        report["valid"] = False
        print(f"Data integrity check failed for {data_name}.")
        return report

    if expected_columns is not None:
        expected_columns = tuple(expected_columns)
    else:
        expected_columns = tuple(
            sorted(
                {
                    field
                    for observation in data.values()
                    if isinstance(observation, dict)
                    for field in observation
                }
            )
        )

    if expected_sample_count is not None and len(data) != expected_sample_count:
        add_error(
            "unexpected_observation_count",
            f"Expected {expected_sample_count} observations, found {len(data)}.",
        )

    if expected_shape is not None:
        actual_shape = (
            len(data),
            len(expected_columns),
        )
        if actual_shape != expected_shape:
            add_error(
                "unexpected_shape",
                f"Expected shape {expected_shape}, found {actual_shape}.",
            )

    seen_observations = {}

    for sample_id, observation in data.items():
        report["observation_count"] += 1

        if not isinstance(sample_id, str):
            add_error(
                "invalid_sample_id",
                "The sample ID must be a string.",
                sample_id=sample_id,
            )

        if not isinstance(observation, dict):
            add_error(
                "invalid_observation_type",
                "The observation must be a dictionary.",
                sample_id=sample_id,
            )
            continue

        missing_columns = set(expected_columns) - set(observation)
        unexpected_columns = set(observation) - set(expected_columns)

        for field in sorted(missing_columns):
            add_error(
                "missing_column",
                f"Missing required column: {field}.",
                sample_id=sample_id,
                field=field,
            )

        for field in sorted(unexpected_columns):
            add_error(
                "unexpected_column",
                f"Unexpected column: {field}.",
                sample_id=sample_id,
                field=field,
            )

        if expected_dtypes is not None:
            for field, expected_type in expected_dtypes.items():
                if field not in observation:
                    continue

                value = observation[field]
                if expected_type == "numeric":
                    valid_type = (
                        isinstance(value, (int, float, np.number))
                        and not isinstance(value, (bool, np.bool_))
                    )
                elif expected_type == "integer":
                    valid_type = (
                        isinstance(value, (int, np.integer))
                        and not isinstance(value, (bool, np.bool_))
                    )
                elif expected_type in ("categorical", "string"):
                    valid_type = isinstance(value, str)
                else:
                    raise ValueError(
                        f"Unknown expected type: {expected_type}"
                    )

                if not valid_type:
                    add_error(
                        "invalid_type",
                        f"Expected type {expected_type}.",
                        sample_id=sample_id,
                        field=field,
                    )

        if expected_ranges is not None:
            for field, (minimum, maximum) in expected_ranges.items():
                if field not in observation:
                    continue

                value = observation[field]
                is_number = (
                    isinstance(value, (int, float, np.number))
                    and not isinstance(value, (bool, np.bool_))
                )
                if not is_number:
                    continue
                if not math.isfinite(float(value)):
                    add_error(
                        "non_finite_value",
                        "The value must be finite.",
                        sample_id=sample_id,
                        field=field,
                    )
                    continue
                if minimum is not None and value < minimum:
                    add_error(
                        "out_of_range",
                        f"The value must be at least {minimum}.",
                        sample_id=sample_id,
                        field=field,
                    )
                elif maximum is not None and value > maximum:
                    add_error(
                        "out_of_range",
                        f"The value must be at most {maximum}.",
                        sample_id=sample_id,
                        field=field,
                    )

        if expected_categories is not None:
            for field, allowed_values in expected_categories.items():
                if field not in observation:
                    continue
                value = observation[field]
                if value not in set(allowed_values):
                    add_error(
                        "invalid_category",
                        f"Unexpected category: {value}.",
                        sample_id=sample_id,
                        field=field,
                    )

        # Detect duplicated observations independently of sample ID.
        signature = tuple(
            (field, repr(observation.get(field)))
            for field in sorted(expected_columns)
        )
        if signature in seen_observations:
            add_error(
                "duplicate_observation",
                f"This observation duplicates {seen_observations[signature]}.",
                sample_id=sample_id,
            )
        else:
            seen_observations[signature] = sample_id

    report["valid"] = len(report["errors"]) == 0

    if report["valid"]:
        print("Integrity check passed.")
    else:
        problem_count = len(report["errors"])
        noun = "problem" if problem_count == 1 else "problems"
        print(
            f"Data integrity check failed: {problem_count} {noun} found. "
            f"File checked: {data_name}"
        )

    return report


if __name__ == "__main__":
    generate_maize_dataset()
