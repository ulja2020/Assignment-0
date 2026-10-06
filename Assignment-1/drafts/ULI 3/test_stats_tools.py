import math

import numpy as np

from stats_tools import (
    compute_cdf,
    compute_entropy,
    compute_kl_divergence,
    compute_mean_std,
    compute_pdf,
    compute_skewness,
)


def test_pdf_area_and_density():
    """Check density values and confirm that the histogram PDF has unit area."""
    _, density, bin_widths = compute_pdf([0, 0, 1, 1], bins=2)

    assert np.allclose(density, [1.0, 1.0])
    assert np.isclose(np.sum(density * bin_widths), 1.0)


def test_cdf():
    """Check the cumulative probabilities at the right-hand histogram edges."""
    edges, cdf = compute_cdf([0, 0, 1, 1], bins=2)

    assert np.allclose(edges, [0.5, 1.0])
    assert np.allclose(cdf, [0.5, 1.0])
    assert np.all(np.diff(cdf) >= 0)
    assert np.isclose(cdf[-1], 1.0)


def test_mean_and_population_standard_deviation():
    """Check the average and ddof=0 standard deviation of four values."""
    mean, std = compute_mean_std([1, 2, 3, 4])

    assert np.isclose(mean, 2.5)
    assert np.isclose(std, math.sqrt(1.25))


def test_skewness():
    """Check symmetric, right-skewed and constant examples."""
    assert np.isclose(
        compute_skewness([-2, -1, 0, 1, 2]),
        0.0,
    )
    assert np.isclose(
        compute_skewness([1, 2, 3, 4, 10]),
        1.1384199576606167,
    )
    assert np.isnan(compute_skewness([3, 3, 3, 3]))


def test_entropy():
    """Check one-bit balance, zero entropy and empty-bin handling."""
    assert np.isclose(
        compute_entropy([0, 0, 1, 1], bins=2),
        1.0,
    )
    assert np.isclose(
        compute_entropy([5, 5, 5, 5], bins=4),
        0.0,
    )


def test_kl_known_distributions():
    """Check KL against a small distribution with a hand-computable answer."""
    p = np.array([0.5, 0.5])
    q = np.array([0.75, 0.25])

    expected = 0.5 * math.log2(0.5 / 0.75) + 0.5 * math.log2(0.5 / 0.25)
    assert np.isclose(compute_kl_divergence(p, q), expected)
    assert not np.isclose(
        compute_kl_divergence(p, q),
        compute_kl_divergence(q, p),
    )


def test_kl_zero_handling():
    """Check the exact mathematical behavior when Q assigns zero mass."""
    p = np.array([1.0, 0.0])
    q = np.array([0.0, 1.0])

    assert np.isinf(compute_kl_divergence(p, q))


def test_pdf_area_catches_wrong_normalization():
    """Demonstrate that a count-normalized PDF would fail the area test."""
    values = np.array([0.0, 0.0, 1.0, 1.0])
    counts, edges = np.histogram(values, bins=2)
    wrong_density = counts / np.sum(counts)
    wrong_area = np.sum(wrong_density * np.diff(edges))

    assert not np.isclose(wrong_area, 1.0)
