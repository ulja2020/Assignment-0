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


def _as_finite_array(values):
    """Convert values to a one-dimensional finite floating-point array."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("values must be one-dimensional.")
    if array.size == 0:
        raise ValueError("values must not be empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError("values must contain only finite numbers.")
    return array


def compute_pdf(values, bins=DEFAULT_BINS):
    """
    Estimate a probability density function using a histogram.

    Parameters
    ----------
    values : array-like
        Numerical observations.

    bins : int, default=25
        Number of histogram bins.

    Returns
    -------
    tuple of numpy.ndarray
        ``bin_centres``, ``density`` and ``bin_widths``.

    Notes
    -----
    The density is normalized so that the AREA under the histogram equals 1.
    It is therefore incorrect to simply sum the density values and expect 1.
    The density is computed as count divided by observation count and bin
    width, i.e. ``count / (N * bin_width)``.
    """
    values = _as_finite_array(values)
    if not isinstance(bins, (int, np.integer)) or bins < 1:
        raise ValueError("bins must be a positive integer.")

    counts, bin_edges = np.histogram(values, bins=int(bins))
    bin_widths = np.diff(bin_edges)
    density = counts / (len(values) * bin_widths)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

    return bin_centres, density, bin_widths


def compute_cdf(values, bins=DEFAULT_BINS):
    """
    Compute a histogram-based cumulative distribution function.

    Parameters
    ----------
    values : array-like
        Numerical observations.

    bins : int, default=25
        Number of histogram bins.

    Returns
    -------
    tuple of numpy.ndarray
        Right-hand bin edges and cumulative probabilities.  The cumulative
        probability at a bin edge is the probability of observations falling
        at or below that edge.

    Notes
    -----
    The CDF uses the same histogram probabilities as the PDF and entropy
    calculations, so its resolution depends on the chosen number of bins.
    """
    values = _as_finite_array(values)
    if not isinstance(bins, (int, np.integer)) or bins < 1:
        raise ValueError("bins must be a positive integer.")

    counts, bin_edges = np.histogram(values, bins=int(bins))
    probabilities = counts / np.sum(counts)
    cdf = np.cumsum(probabilities)
    cdf[-1] = 1.0

    return bin_edges[1:], cdf


def compute_mean_std(values):
    """
    Compute the arithmetic mean and population standard deviation.

    The standard deviation uses the population convention ``ddof=0``.  This
    is the convention used consistently in this assignment and in the
    standardisation steps used for Task 6 and Task 7.
    """
    values = _as_finite_array(values)
    mean = np.mean(values)
    std = np.sqrt(np.mean((values - mean) ** 2))
    return float(mean), float(std)


def compute_skewness(values):
    """
    Compute moment skewness using the standardized third central moment.

    The definition used is ``m3 / m2**1.5``.  Positive values indicate a
    longer or heavier right tail, negative values indicate a longer left tail,
    and values near zero indicate little asymmetry.  For a constant feature
    the variance is zero, so skewness is undefined and the function returns
    NaN rather than assigning an artificial value.
    """
    values = _as_finite_array(values)
    mean, std = compute_mean_std(values)
    if std == 0:
        return float("nan")

    skewness = np.mean(((values - mean) / std) ** 3)
    return float(skewness)


def compute_entropy(values, bins=DEFAULT_BINS):
    """
    Compute Shannon entropy from histogram probabilities.

    Parameters
    ----------
    values : array-like
        Numerical observations.

    bins : int, default=25
        Number of histogram bins.

    Returns
    -------
    float
        Shannon entropy in bits.

    Notes
    -----
    Base-2 logarithms are used, so entropy is measured in bits. Terms with
    ``p_i = 0`` are excluded because their contribution is defined as zero
    (the usual convention ``0 * log2(0) = 0``). Histogram entropy depends on
    the selected number of bins.
    """
    values = _as_finite_array(values)
    if not isinstance(bins, (int, np.integer)) or bins < 1:
        raise ValueError("bins must be a positive integer.")

    counts, _ = np.histogram(values, bins=int(bins))
    probabilities = counts / np.sum(counts)
    positive_probabilities = probabilities[probabilities > 0]
    entropy = -np.sum(
        positive_probabilities * np.log2(positive_probabilities)
    )
    return float(entropy)


def compute_kl_divergence(probabilities_p, probabilities_q):
    """
    Compute the Kullback-Leibler divergence ``D_KL(P || Q)`` in bits.

    Parameters
    ----------
    probabilities_p : array-like
        Probability distribution P.

    probabilities_q : array-like
        Probability distribution Q defined on the same bins as P.

    Returns
    -------
    float
        ``D_KL(P || Q)`` measured with base-2 logarithms.

    Notes
    -----
    Both distributions must have the same one-dimensional shape, contain no
    negative probabilities and sum to one. Terms where ``p_i = 0`` contribute
    zero. If ``p_i > 0`` while ``q_i = 0``, the mathematical KL divergence is
    infinite, so this function returns ``np.inf`` rather than dropping the bin
    or adding epsilon smoothing. This preserves the exact definition, but it
    can make numerical comparisons less informative when sparse histograms
    contain zero-probability bins. KL is directional; the reverse calculation
    must be performed separately.
    """
    p = np.asarray(probabilities_p, dtype=float)
    q = np.asarray(probabilities_q, dtype=float)

    if p.ndim != 1 or q.ndim != 1:
        raise ValueError("Both probability distributions must be one-dimensional.")
    if len(p) != len(q):
        raise ValueError("Probability distributions must have the same length.")
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(q)):
        raise ValueError("Probability distributions must contain finite values.")
    if np.any(p < 0) or np.any(q < 0):
        raise ValueError("Probabilities cannot be negative.")
    if not np.isclose(np.sum(p), 1.0):
        raise ValueError("P must sum to 1.")
    if not np.isclose(np.sum(q), 1.0):
        raise ValueError("Q must sum to 1.")

    positive_p = p > 0
    if np.any(q[positive_p] == 0):
        return float("inf")

    return float(
        np.sum(
            p[positive_p]
            * np.log2(p[positive_p] / q[positive_p])
        )
    )


def describe_numerical_features(dataset, bins=DEFAULT_BINS):
    """
    Compute every required distribution statistic for the five numerical fields.

    ``dataset`` is a dictionary whose values are observation dictionaries.
    Task 3 is assumed to have already verified that the requested numerical
    fields are present and finite. The categorical field is intentionally
    excluded because Task 4 requires a different frequency-based treatment.
    """
    descriptions = {}

    for feature in NUMERICAL_FEATURES:
        values = [observation[feature] for observation in dataset.values()]
        bin_centres, densities, bin_widths = compute_pdf(values, bins)
        cdf_positions, cumulative_probabilities = compute_cdf(values, bins)
        mean, std = compute_mean_std(values)

        descriptions[feature] = {
            "pdf": {
                "bin_centres": bin_centres,
                "densities": densities,
                "bin_widths": bin_widths,
            },
            "cdf": {
                "bin_right_edges": cdf_positions,
                "cumulative_probabilities": cumulative_probabilities,
            },
            "average": mean,
            "standard_deviation": std,
            "skewness": compute_skewness(values),
            "entropy_bits": compute_entropy(values, bins),
        }

    return descriptions


