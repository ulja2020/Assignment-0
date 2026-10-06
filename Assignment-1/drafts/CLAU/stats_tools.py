"""Describe distributions in the validated synthetic maize dataset.

The functions in this module assume that Task 3 has already confirmed data
integrity. Inputs therefore contain the required columns and valid, finite
numerical values within their expected ranges. These functions focus only on
the statistical calculations and do not repeat the integrity checks.
"""

import math

import numpy as np


DEFAULT_BINS = 25
NUMERICAL_FEATURES = (
    "plant_height_cm",
    "leaf_damage_percent",
    "moisture_sensor_percent",
    "ear_length_cm",
    "grain_mass_per_ear_g",
)


def calculate_pdf(values, bins=DEFAULT_BINS):
    """Return bin centres, density values and edges for a histogram PDF.

    The histogram is normalized by the observation count and each bin width,
    so the sum of ``density * bin_width`` is one. The density values themselves
    are not required to sum to one.
    """
    values = np.asarray(values, dtype=float)
    counts, bin_edges = np.histogram(values, bins=bins)
    bin_widths = np.diff(bin_edges)
    densities = counts / (len(values) * bin_widths)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centres, densities, bin_edges


def calculate_cdf(values, bins=DEFAULT_BINS):
    """Return right-hand bin edges and a histogram-based cumulative distribution."""
    values = np.asarray(values, dtype=float)
    counts, bin_edges = np.histogram(values, bins=bins)
    probabilities = counts / len(values)

    cumulative_probabilities = []
    running_probability = 0.0
    for probability in probabilities:
        running_probability = min(1.0, running_probability + float(probability))
        cumulative_probabilities.append(running_probability)

    return bin_edges[1:], np.asarray(cumulative_probabilities)


def calculate_average(values):
    """Return the arithmetic average of the numerical values."""
    values = np.asarray(values, dtype=float)
    return float(np.sum(values) / len(values))


def calculate_standard_deviation(values):
    """Return the population standard deviation of the numerical values.

    The denominator is the number of observations, equivalent to ``ddof=0``.
    """
    values = np.asarray(values, dtype=float)
    average = calculate_average(values)
    squared_deviations = (values - average) ** 2
    variance = float(np.sum(squared_deviations) / len(values))
    return math.sqrt(variance)


def calculate_skewness(values):
    """Return moment skewness calculated from the second and third moments.

    This is the biased central-moment coefficient ``m3 / m2**1.5``. A constant
    feature has zero variance, so its skewness is undefined and the function
    returns NaN.
    """
    values = np.asarray(values, dtype=float)
    average = calculate_average(values)
    deviations = values - average
    second_moment = float(np.sum(deviations**2) / len(values))

    if second_moment == 0:
        return float("nan")

    third_moment = float(np.sum(deviations**3) / len(values))
    return third_moment / (second_moment**1.5)


def calculate_entropy(values, bins=DEFAULT_BINS):
    """Return histogram-based Shannon entropy in bits.

    Logarithm base 2 is used. A bin for which ``p_i = 0`` contributes zero to
    entropy and is omitted before taking the logarithm; this applies the
    convention ``0 * log2(0) = 0``.
    """
    values = np.asarray(values, dtype=float)
    counts, _ = np.histogram(values, bins=bins)
    probabilities = counts / len(values)

    entropy = 0.0
    for probability in probabilities:
        if probability > 0:
            entropy -= float(probability) * math.log2(float(probability))

    return entropy


def describe_numerical_features(dataset, bins=DEFAULT_BINS):
    """Compute every required distribution statistic for five numerical features.

    Task 3 is assumed to have already verified the dataset structure, column
    names, types, ranges, missing values and duplicates. The categorical
    ``field_zone`` column is deliberately excluded because Task 4 requests
    statistics only for numerical features.
    """
    descriptions = {}

    for feature in NUMERICAL_FEATURES:
        values = [observation[feature] for observation in dataset.values()]
        bin_centres, densities, pdf_edges = calculate_pdf(values, bins)
        cdf_positions, cumulative_probabilities = calculate_cdf(values, bins)

        descriptions[feature] = {
            "pdf": {
                "bin_centres": bin_centres,
                "densities": densities,
                "bin_edges": pdf_edges,
            },
            "cdf": {
                "bin_right_edges": cdf_positions,
                "cumulative_probabilities": cumulative_probabilities,
            },
            "average": calculate_average(values),
            "standard_deviation": calculate_standard_deviation(values),
            "skewness": calculate_skewness(values),
            "entropy_bits": calculate_entropy(values, bins),
        }

    return descriptions
