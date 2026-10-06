"""Task 10 generic validation test for the Task 9 regression functions."""

import unittest

import numpy as np

from model_tools import fit_linear_regression, predict_regression


class TestRegressionTools(unittest.TestCase):
    """Check regression mathematics independently of the maize dataset."""

    def test_exact_linear_relationship(self):
        """Task 10: OLS must recover the exact line y = 2 + 3x."""
        values = np.array([[0.0], [1.0], [2.0], [3.0]])
        targets = 2.0 + 3.0 * values[:, 0]

        model = fit_linear_regression(values, targets)
        predictions = predict_regression(model, values)

        self.assertTrue(np.isclose(model.intercept_, 2.0, atol=1e-12))
        self.assertTrue(np.allclose(model.coef_, [3.0], atol=1e-12))
        self.assertTrue(np.allclose(predictions, targets, atol=1e-12))


if __name__ == "__main__":
    unittest.main()