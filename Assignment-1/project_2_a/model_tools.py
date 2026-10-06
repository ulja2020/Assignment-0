"""Clustering tools for Task 8.

The public fitting functions receive prepared numerical data and settings and
return a fitted model, one label per observation, and an evaluation report.
Only NumPy is required.  This keeps the clustering calculations reproducible
and makes every assumption used by the two methods visible in this project.
"""

from dataclasses import dataclass

import numpy as np


TASK8_FEATURES = (
    "ear_length_cm",
    "grain_mass_per_ear_g",
    "leaf_damage_percent",
)


@dataclass
class Standardizer:
    """Store training means and deviations and apply the same scaling later."""

    mean_: np.ndarray
    scale_: np.ndarray

    def transform(self, values):
        """Standardize values using the fitted feature means and deviations."""
        return (np.asarray(values, dtype=float) - self.mean_) / self.scale_


@dataclass
class KMeansModel:
    """Store the fitted K-means centres and convergence information."""

    cluster_centers_: np.ndarray
    inertia_: float
    n_iter_: int

    def predict(self, values):
        """Assign each observation to its nearest fitted centre."""
        values = np.asarray(values, dtype=float)
        distances = np.sum(
            (values[:, None, :] - self.cluster_centers_[None, :, :]) ** 2,
            axis=2,
        )
        return np.argmin(distances, axis=1)


@dataclass
class GaussianMixtureModel:
    """Store a fitted full-covariance Gaussian mixture model."""

    weights_: np.ndarray
    means_: np.ndarray
    covariances_: np.ndarray
    lower_bound_: float
    n_iter_: int

    def predict_proba(self, values):
        """Return the posterior probability of every mixture component."""
        log_weighted = _log_weighted_gaussians(
            np.asarray(values, dtype=float),
            self.weights_,
            self.means_,
            self.covariances_,
        )
        log_total = _logsumexp(log_weighted, axis=1)
        return np.exp(log_weighted - log_total[:, None])

    def predict(self, values):
        """Assign each observation to its most probable component."""
        return np.argmax(self.predict_proba(values), axis=1)


def prepare_clustering_data(dataset, features=TASK8_FEATURES):
    """Extract selected features and standardize them for clustering.

    Standardization is fitted on the supplied observations because Euclidean
    distances and Gaussian covariance estimates otherwise depend on feature
    units.  No distribution transformation is applied, so all three selected
    features retain their original interpretation.

    Returns a dictionary containing the standardized matrix, original matrix,
    ordered sample IDs, feature names and fitted ``Standardizer``.
    """
    sample_ids = list(dataset)
    original = np.asarray(
        [[dataset[sample_id][feature] for feature in features]
         for sample_id in sample_ids],
        dtype=float,
    )
    if original.ndim != 2 or original.shape[0] < 2:
        raise ValueError("Clustering requires at least two observations.")
    if not np.all(np.isfinite(original)):
        raise ValueError("Clustering features must contain only finite values.")

    means = np.mean(original, axis=0)
    scales = np.std(original, axis=0, ddof=0)
    if np.any(scales == 0):
        raise ValueError("Selected clustering features must not be constant.")

    scaler = Standardizer(means, scales)
    return {
        "values": scaler.transform(original),
        "original_values": original,
        "sample_ids": sample_ids,
        "features": tuple(features),
        "scaler": scaler,
    }


def _validate_fit_input(values, n_clusters):
    """Validate a numerical clustering matrix and requested cluster count."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("values must be a 2D matrix with at least two rows.")
    if not np.all(np.isfinite(values)):
        raise ValueError("values must contain only finite numbers.")
    if not isinstance(n_clusters, (int, np.integer)):
        raise TypeError("n_clusters must be an integer.")
    if not 1 <= n_clusters < len(values):
        raise ValueError("n_clusters must be between 1 and N - 1.")
    return values


def _initial_centres(values, n_clusters, rng):
    """Choose reproducible K-means++ starting centres."""
    centres = [values[rng.integers(len(values))]]
    while len(centres) < n_clusters:
        squared = np.min(
            np.sum(
                (values[:, None, :] - np.asarray(centres)[None, :, :]) ** 2,
                axis=2,
            ),
            axis=1,
        )
        total = np.sum(squared)
        if total == 0:
            remaining = [
                index for index in range(len(values))
                if not any(np.array_equal(values[index], centre) for centre in centres)
            ]
            centres.append(values[rng.choice(remaining)])
        else:
            centres.append(values[rng.choice(len(values), p=squared / total)])
    return np.asarray(centres, dtype=float)


def _run_kmeans_once(values, n_clusters, random_seed, max_iterations, tolerance):
    """Run one K-means optimization from a K-means++ initialization."""
    rng = np.random.default_rng(random_seed)
    centres = _initial_centres(values, n_clusters, rng)

    for iteration in range(1, max_iterations + 1):
        distances = np.sum(
            (values[:, None, :] - centres[None, :, :]) ** 2,
            axis=2,
        )
        labels = np.argmin(distances, axis=1)
        new_centres = centres.copy()
        for cluster in range(n_clusters):
            members = values[labels == cluster]
            if len(members):
                new_centres[cluster] = np.mean(members, axis=0)
            else:
                new_centres[cluster] = values[np.argmax(np.min(distances, axis=1))]
        shift = float(np.max(np.linalg.norm(new_centres - centres, axis=1)))
        centres = new_centres
        if shift <= tolerance:
            break

    final_distances = np.sum(
        (values[:, None, :] - centres[None, :, :]) ** 2,
        axis=2,
    )
    labels = np.argmin(final_distances, axis=1)
    inertia = float(np.sum(final_distances[np.arange(len(values)), labels]))
    return centres, labels, inertia, iteration


def fit_kmeans_clustering(
    values,
    n_clusters=3,
    random_seed=42,
    n_initializations=10,
    max_iterations=300,
    tolerance=1e-4,
):
    """Fit K-means and return ``(model, labels, evaluation)``.

    Multiple reproducible K-means++ starts are compared, and the solution with
    the lowest within-cluster sum of squared distances is retained.
    """
    values = _validate_fit_input(values, n_clusters)
    if n_initializations < 1:
        raise ValueError("n_initializations must be positive.")

    best = None
    for start in range(n_initializations):
        result = _run_kmeans_once(
            values,
            n_clusters,
            random_seed + start,
            max_iterations,
            tolerance,
        )
        if best is None or result[2] < best[2]:
            best = result

    centres, labels, inertia, iterations = best
    model = KMeansModel(centres, inertia, iterations)
    evaluation = evaluate_clustering(values, labels)
    evaluation["inertia"] = inertia
    return model, labels, evaluation


def _logsumexp(values, axis):
    """Compute log(sum(exp(values))) without numerical overflow."""
    maximum = np.max(values, axis=axis, keepdims=True)
    result = maximum + np.log(np.sum(np.exp(values - maximum), axis=axis, keepdims=True))
    return np.squeeze(result, axis=axis)


def _log_weighted_gaussians(values, weights, means, covariances):
    """Return weighted log densities for full-covariance Gaussians."""
    n_features = values.shape[1]
    result = np.empty((len(values), len(weights)))
    constant = n_features * np.log(2 * np.pi)
    for cluster in range(len(weights)):
        sign, log_determinant = np.linalg.slogdet(covariances[cluster])
        if sign <= 0:
            raise ValueError("A Gaussian covariance matrix is not positive definite.")
        difference = values - means[cluster]
        solved = np.linalg.solve(covariances[cluster], difference.T).T
        quadratic = np.sum(difference * solved, axis=1)
        result[:, cluster] = (
            np.log(weights[cluster])
            - 0.5 * (constant + log_determinant + quadratic)
        )
    return result


def fit_gmm_clustering(
    values,
    n_components=3,
    random_seed=42,
    max_iterations=300,
    tolerance=1e-4,
    regularization=1e-6,
):
    """Fit a full-covariance GMM and return ``(model, labels, evaluation)``.

    The expectation-maximization algorithm starts from a reproducible K-means
    solution.  ``regularization`` is added to covariance diagonals to keep the
    matrices invertible.
    """
    values = _validate_fit_input(values, n_components)
    _, labels, _ = fit_kmeans_clustering(
        values,
        n_clusters=n_components,
        random_seed=random_seed,
        n_initializations=5,
    )
    n_observations, n_features = values.shape
    weights = np.bincount(labels, minlength=n_components) / n_observations
    means = np.asarray([np.mean(values[labels == k], axis=0) for k in range(n_components)])
    covariances = np.empty((n_components, n_features, n_features))
    for cluster in range(n_components):
        difference = values[labels == cluster] - means[cluster]
        covariances[cluster] = difference.T @ difference / len(difference)
        covariances[cluster].flat[:: n_features + 1] += regularization

    previous = -np.inf
    for iteration in range(1, max_iterations + 1):
        log_weighted = _log_weighted_gaussians(values, weights, means, covariances)
        log_total = _logsumexp(log_weighted, axis=1)
        lower_bound = float(np.mean(log_total))
        responsibilities = np.exp(log_weighted - log_total[:, None])
        effective_counts = np.sum(responsibilities, axis=0)

        weights = effective_counts / n_observations
        means = responsibilities.T @ values / effective_counts[:, None]
        for cluster in range(n_components):
            difference = values - means[cluster]
            weighted = responsibilities[:, cluster, None] * difference
            covariances[cluster] = weighted.T @ difference / effective_counts[cluster]
            covariances[cluster].flat[:: n_features + 1] += regularization

        if abs(lower_bound - previous) <= tolerance:
            break
        previous = lower_bound

    model = GaussianMixtureModel(weights, means, covariances, lower_bound, iteration)
    labels = model.predict(values)
    evaluation = evaluate_clustering(values, labels)
    evaluation["average_log_likelihood"] = lower_bound
    evaluation["mean_membership_confidence"] = float(
        np.mean(np.max(model.predict_proba(values), axis=1))
    )
    return model, labels, evaluation


def evaluate_clustering(values, labels):
    """Return cluster sizes, noise count and mean silhouette score.

    Labels equal to ``-1`` are treated as noise.  The silhouette score is
    undefined when fewer than two non-noise clusters remain, when fewer than
    two non-noise observations remain, or when every non-noise observation is
    its own cluster.  In those cases the report contains ``NaN`` and explains
    why instead of raising an error.
    """
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels)
    if len(values) != len(labels):
        raise ValueError("values and labels must contain the same number of rows.")

    cluster_labels = sorted(label for label in np.unique(labels) if label != -1)
    sizes = {int(label): int(np.sum(labels == label)) for label in cluster_labels}
    noise_count = int(np.sum(labels == -1))
    mask = labels != -1
    filtered_values = values[mask]
    filtered_labels = labels[mask]

    reason = None
    if len(cluster_labels) < 2:
        reason = "Silhouette is undefined because fewer than two clusters are present."
    elif len(filtered_values) <= len(cluster_labels):
        reason = "Silhouette is undefined when every observation forms its own cluster."

    if reason is not None:
        silhouette = float("nan")
    else:
        differences = filtered_values[:, None, :] - filtered_values[None, :, :]
        distances = np.sqrt(np.sum(differences ** 2, axis=2))
        sample_scores = np.empty(len(filtered_values))
        for index, label in enumerate(filtered_labels):
            same = filtered_labels == label
            same[index] = False
            if not np.any(same):
                sample_scores[index] = 0.0
                continue
            within = float(np.mean(distances[index, same]))
            nearest = min(
                float(np.mean(distances[index, filtered_labels == other]))
                for other in cluster_labels if other != label
            )
            denominator = max(within, nearest)
            sample_scores[index] = 0.0 if denominator == 0 else (nearest - within) / denominator
        silhouette = float(np.mean(sample_scores))

    return {
        "cluster_sizes": sizes,
        "noise_points": noise_count,
        "number_of_clusters": len(cluster_labels),
        "silhouette_score": silhouette,
        "silhouette_defined": reason is None,
        "silhouette_note": "Defined for this partition." if reason is None else reason,
    }


def project_to_two_dimensions(values):
    """Project standardized observations onto their first two principal components."""
    values = np.asarray(values, dtype=float)
    centred = values - np.mean(values, axis=0)
    _, _, right_vectors = np.linalg.svd(centred, full_matrices=False)
    return centred @ right_vectors[:2].T


def cluster_profiles(original_values, labels, features=TASK8_FEATURES):
    """Return cluster sizes and feature averages in their original units."""
    original_values = np.asarray(original_values, dtype=float)
    labels = np.asarray(labels)
    profiles = {}
    for label in sorted(value for value in np.unique(labels) if value != -1):
        members = original_values[labels == label]
        profiles[int(label)] = {
            "size": int(len(members)),
            "averages": {
                feature: float(average)
                for feature, average in zip(features, np.mean(members, axis=0))
            },
        }
    return profiles


# ---------------------------------------------------------------------------
# Task 9: reusable linear-regression tools
# ---------------------------------------------------------------------------


@dataclass
class LinearRegressionModel:
    """Task 9 fitted ordinary least-squares model with an intercept."""

    coef_: np.ndarray
    intercept_: float

    def predict(self, values):
        """Predict the numerical target for a two-dimensional feature matrix."""
        values = np.asarray(values, dtype=float)
        return values @ self.coef_ + self.intercept_


@dataclass
class MeanBaselineModel:
    """Task 9 baseline that always predicts the fitted target mean."""

    constant_: float

    def predict(self, values):
        """Return the training-target mean once for every supplied row."""
        return np.full(len(values), self.constant_, dtype=float)


def prepare_regression_data(dataset, predictors, target):
    """Task 9: extract ordered predictors, target values and sample IDs."""
    sample_ids = list(dataset)
    values = np.asarray(
        [[dataset[sample_id][feature] for feature in predictors]
         for sample_id in sample_ids],
        dtype=float,
    )
    targets = np.asarray(
        [dataset[sample_id][target] for sample_id in sample_ids],
        dtype=float,
    )
    if values.ndim != 2 or len(values) < 5:
        raise ValueError("Regression requires a 2D matrix with at least five rows.")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(targets)):
        raise ValueError("Regression data must contain only finite numbers.")
    return {
        "values": values,
        "targets": targets,
        "sample_ids": sample_ids,
        "predictors": tuple(predictors),
        "target": target,
    }


def split_regression_data(number_of_observations, train_fraction=0.60,
                          validation_fraction=0.20, random_seed=42):
    """Task 9: create train, validation and reserved-test row indices.

    Splitting happens before any model is fitted.  The returned indices can be
    applied to every candidate predictor matrix, ensuring identical splits.
    """
    if number_of_observations < 5:
        raise ValueError("At least five observations are required for splitting.")
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("Split fractions must be between zero and one.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("The split fractions must leave observations for testing.")

    rng = np.random.default_rng(random_seed)
    indices = rng.permutation(number_of_observations)
    train_end = int(number_of_observations * train_fraction)
    validation_end = train_end + int(
        number_of_observations * validation_fraction
    )
    return {
        "train": indices[:train_end],
        "validation": indices[train_end:validation_end],
        "test": indices[validation_end:],
    }


def fit_linear_regression(values, targets):
    """Task 9: fit ordinary least squares with an intercept using NumPy."""
    values = np.asarray(values, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if values.ndim != 2 or len(values) != len(targets):
        raise ValueError("Predictors must be 2D and aligned with the target.")
    design = np.column_stack((np.ones(len(values)), values))
    parameters, _, _, _ = np.linalg.lstsq(design, targets, rcond=None)
    return LinearRegressionModel(
        coef_=parameters[1:],
        intercept_=float(parameters[0]),
    )


def fit_mean_baseline(targets):
    """Task 9: fit a baseline that ignores predictors and predicts their mean."""
    targets = np.asarray(targets, dtype=float)
    if targets.size == 0 or not np.all(np.isfinite(targets)):
        raise ValueError("Baseline targets must be non-empty and finite.")
    return MeanBaselineModel(float(np.mean(targets)))


def predict_regression(model, values):
    """Task 9: obtain predictions from a fitted regression or baseline model."""
    predictions = np.asarray(model.predict(values), dtype=float)
    if predictions.ndim != 1:
        raise ValueError("Regression predictions must be one-dimensional.")
    return predictions


def evaluate_regression(targets, predictions):
    """Task 9: return MAE, RMSE and R-squared with units preserved.

    R-squared is undefined when all evaluated target values are identical.  In
    that case the function returns NaN and a clear explanation.
    """
    targets = np.asarray(targets, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    if targets.shape != predictions.shape or targets.ndim != 1:
        raise ValueError("Targets and predictions must be aligned 1D arrays.")
    residuals = targets - predictions
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    total = float(np.sum((targets - np.mean(targets)) ** 2))
    if total == 0:
        r_squared = float("nan")
        note = "R-squared is undefined because the evaluated target is constant."
    else:
        r_squared = float(1 - np.sum(residuals ** 2) / total)
        note = "Defined for this target sample."
    return {
        "mae": mae,
        "rmse": rmse,
        "r_squared": r_squared,
        "r_squared_defined": total != 0,
        "r_squared_note": note,
    }
