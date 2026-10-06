"""Verify the Task 4 statistics with small hand-checkable datasets.

These checks form part of the validation suite requested for Task 10. They use
known numerical results and mathematical properties rather than ready-made
skewness or entropy functions.
"""

import math

import numpy as np

from stats_tools import (
    calculate_average,
    calculate_cdf,
    calculate_entropy,
    calculate_pdf,
    calculate_skewness,
    calculate_standard_deviation,
)


def test_pdf():
    """Confirm the density values and unit area of a two-bin histogram."""
    _, densities, bin_edges = calculate_pdf([0, 0, 1, 1], bins=2)
    assert np.allclose(densities, [1.0, 1.0])
    assert np.isclose(np.sum(densities * np.diff(bin_edges)), 1.0)


def test_cdf():
    """Confirm the cumulative probabilities of two equally populated bins."""
    _, cumulative_probabilities = calculate_cdf([0, 0, 1, 1], bins=2)
    assert np.allclose(cumulative_probabilities, [0.5, 1.0])


def test_average_and_standard_deviation():
    """Check the average and population deviation of four simple values."""
    values = [1, 2, 3, 4]
    assert np.isclose(calculate_average(values), 2.5)
    assert np.isclose(calculate_standard_deviation(values), math.sqrt(1.25))


def test_skewness():
    """Check symmetric, right-skewed and constant examples."""
    assert np.isclose(calculate_skewness([-2, -1, 0, 1, 2]), 0.0)
    assert np.isclose(calculate_skewness([1, 2, 3, 4, 10]), 1.1384199576606167)
    assert np.isnan(calculate_skewness([3, 3, 3, 3]))


def test_entropy():
    """Check one-bit balance, zero entropy and empty-bin handling."""
    assert np.isclose(calculate_entropy([0, 0, 1, 1], bins=2), 1.0)
    assert np.isclose(calculate_entropy([5, 5, 5, 5], bins=4), 0.0)


def run_all_tests():
    """Run every hand-checkable Task 4 statistics test."""
    test_pdf()
    test_cdf()
    test_average_and_standard_deviation()
    test_skewness()
    test_entropy()


if __name__ == "__main__":
    run_all_tests()
    print("All stats_tools tests passed.")
