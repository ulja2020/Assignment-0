"""Task 10 generic validation tests for the statistical functions."""

import math
import unittest

import numpy as np

from stats_tools import (
    compute_cdf,
    compute_entropy,
    compute_kl_divergence,
    compute_mean_std,
    compute_pdf,
    compute_skewness,
)


class TestStatisticalTools(unittest.TestCase):
    """Verify mathematical properties independently of the maize dataset."""

    def test_pdf_area_and_density(self):
        """Task 10: a histogram PDF must have unit area using bin widths."""
        _, density, bin_widths = compute_pdf([0, 0, 1, 1], bins=2)
        self.assertTrue(np.allclose(density, [1.0, 1.0], atol=1e-12))
        self.assertTrue(
            np.isclose(np.sum(density * bin_widths), 1.0, atol=1e-12)
        )

    def test_cdf(self):
        """Task 10: a CDF must be normalized, bounded and non-decreasing."""
        edges, cdf = compute_cdf([0, 0, 1, 1], bins=2)
        self.assertTrue(np.allclose(edges, [0.5, 1.0], atol=1e-12))
        self.assertTrue(np.allclose(cdf, [0.5, 1.0], atol=1e-12))
        self.assertTrue(np.all((cdf >= 0) & (cdf <= 1)))
        self.assertTrue(np.all(np.diff(cdf) >= 0))
        self.assertTrue(np.isclose(cdf[-1], 1.0, atol=1e-12))

    def test_mean_and_population_standard_deviation(self):
        """Check the average and population deviation of known values."""
        mean, std = compute_mean_std([1, 2, 3, 4])
        self.assertTrue(np.isclose(mean, 2.5, atol=1e-12))
        self.assertTrue(np.isclose(std, math.sqrt(1.25), atol=1e-12))

    def test_skewness(self):
        """Check symmetric, right-skewed and undefined constant examples."""
        self.assertTrue(
            np.isclose(compute_skewness([-2, -1, 0, 1, 2]), 0.0, atol=1e-12)
        )
        self.assertTrue(
            np.isclose(
                compute_skewness([1, 2, 3, 4, 10]),
                1.1384199576606167,
                atol=1e-12,
            )
        )
        self.assertTrue(np.isnan(compute_skewness([3, 3, 3, 3])))

    def test_entropy(self):
        """Check one-bit balance, zero entropy and empty-bin handling."""
        self.assertTrue(
            np.isclose(compute_entropy([0, 0, 1, 1], bins=2), 1.0, atol=1e-12)
        )
        self.assertTrue(
            np.isclose(compute_entropy([5, 5, 5, 5], bins=4), 0.0, atol=1e-12)
        )

    def test_kl_known_distributions(self):
        """Check KL against a hand-computable asymmetric example."""
        p = np.array([0.5, 0.5])
        q = np.array([0.75, 0.25])
        expected = (
            0.5 * math.log2(0.5 / 0.75)
            + 0.5 * math.log2(0.5 / 0.25)
        )
        self.assertTrue(
            np.isclose(compute_kl_divergence(p, q), expected, atol=1e-12)
        )
        self.assertFalse(
            np.isclose(
                compute_kl_divergence(p, q),
                compute_kl_divergence(q, p),
                atol=1e-12,
            )
        )

    def test_kl_zero_handling(self):
        """Check infinite KL when Q gives zero mass where P is positive."""
        p = np.array([1.0, 0.0])
        q = np.array([0.0, 1.0])
        self.assertTrue(np.isinf(compute_kl_divergence(p, q)))


if __name__ == "__main__":
    unittest.main()