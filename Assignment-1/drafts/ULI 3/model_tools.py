"""Clustering tools for MOD550 Assignment 1 Task 8.

The public fitting functions receive prepared numerical data and settings and
return a fitted model, one label per observation, and an evaluation report.
Only NumPy is required.  This keeps the clustering calculations reproducible
and makes every assumption used by the two methods visible in this project.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


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

REGRESSION_FEATURES = (
    "plant_height_cm",
    "leaf_damage_percent",
    "ear_length_cm",
)

WRONG_REGRESSION_FEATURES = (
    "ear_length_cm",
)


@dataclass
class MeanBaselineModel:
    """Store the training-target mean used by the regression baseline."""

    target_mean_: float

    def predict(self, values):
        """Predict the training-target mean for every supplied observation."""
        values = np.asarray(values)
        if values.ndim == 0:
            raise ValueError("values must contain observations.")
        number_of_observations = len(values)
        return np.full(number_of_observations, self.target_mean_)


def _validate_regression_input(X, y):
    """Validate a regression matrix and target vector before splitting."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    if X.ndim != 2 or X.shape[0] < 3 or X.shape[1] < 1:
        raise ValueError("X must be a 2D matrix with at least three observations.")
    if y.ndim != 1 or len(y) != len(X):
        raise ValueError("y must be one-dimensional and match X row count.")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError("X and y must contain only finite values.")
    return X, y


def split_regression_data(
    X,
    y,
    test_size=0.20,
    validation_size=0.20,
    random_seed=42,
):
    """Split regression data into disjoint training, validation and test sets.

    The final test set is separated first and remains untouched while the
    training and validation sets are created from the remaining observations.
    Returned indices make the separation explicit and are useful for workflow
    validation tests.
    """
    X, y = _validate_regression_input(X, y)
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1.")
    if test_size + validation_size >= 1:
        raise ValueError("test_size + validation_size must be less than 1.")

    indices = np.arange(len(X))
    train_validation_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_seed,
    )

    validation_fraction = validation_size / (1 - test_size)
    train_indices, validation_indices = train_test_split(
        train_validation_indices,
        test_size=validation_fraction,
        random_state=random_seed,
    )

    return {
        "X_train": X[train_indices],
        "X_validation": X[validation_indices],
        "X_test": X[test_indices],
        "y_train": y[train_indices],
        "y_validation": y[validation_indices],
        "y_test": y[test_indices],
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "test_indices": test_indices,
    }


def fit_linear_regression(X_train, y_train):
    """Fit ordinary least-squares linear regression with an intercept."""
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    if X_train.ndim != 2 or y_train.ndim != 1 or len(X_train) != len(y_train):
        raise ValueError("X_train and y_train have incompatible shapes.")
    if not np.all(np.isfinite(X_train)) or not np.all(np.isfinite(y_train)):
        raise ValueError("Training data must contain only finite values.")

    model = LinearRegression(fit_intercept=True)
    model.fit(X_train, y_train)
    return model


def fit_mean_baseline(y_train):
    """Fit a baseline that always predicts the mean of the training target."""
    y_train = np.asarray(y_train, dtype=float)
    if y_train.ndim != 1 or y_train.size == 0:
        raise ValueError("y_train must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(y_train)):
        raise ValueError("y_train must contain only finite values.")
    return MeanBaselineModel(float(np.mean(y_train)))


def predict_regression(model, X):
    """Return predictions from a fitted regression model or baseline."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2 or not np.all(np.isfinite(X)):
        raise ValueError("X must be a finite two-dimensional matrix.")
    return np.asarray(model.predict(X), dtype=float)


def evaluate_regression(y_true, y_pred):
    """Return MAE, RMSE and R-squared for held-out regression predictions."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.ndim != 1 or y_pred.ndim != 1 or len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be one-dimensional and aligned.")
    if not np.all(np.isfinite(y_true)) or not np.all(np.isfinite(y_pred)):
        raise ValueError("y_true and y_pred must contain only finite values.")

    if len(y_true) < 2 or np.isclose(np.var(y_true), 0.0):
        r_squared = float("nan")
        r_squared_defined = False
        r_squared_note = "R-squared is undefined for a constant or single-value target."
    else:
        r_squared = float(r2_score(y_true, y_pred))
        r_squared_defined = True
        r_squared_note = "R-squared is defined for this evaluation set."

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r_squared": r_squared,
        "r_squared_defined": r_squared_defined,
        "r_squared_note": r_squared_note,
    }


def regression_coefficients(model, feature_names):
    """Return OLS coefficients and intercept paired with feature names."""
    if not hasattr(model, "coef_") or not hasattr(model, "intercept_"):
        raise TypeError("The supplied model does not expose linear coefficients.")
    if len(feature_names) != len(model.coef_):
        raise ValueError("feature_names must match the number of coefficients.")

    return {
        feature: float(coefficient)
        for feature, coefficient in zip(feature_names, model.coef_)
    }, float(model.intercept_)
