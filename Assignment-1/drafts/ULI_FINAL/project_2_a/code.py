from pathlib import Path
import math

import numpy as np

from data_tools import FIELD_ZONE_ORDER, check_data_integrity, generate_maize_dataset
from model_tools import (
    TASK8_FEATURES,
    cluster_profiles,
    evaluate_clustering,
    evaluate_regression,
    fit_gmm_clustering,
    fit_kmeans_clustering,
    fit_linear_regression,
    fit_mean_baseline,
    prepare_clustering_data,
    prepare_regression_data,
    predict_regression,
    project_to_two_dimensions,
    split_regression_data,
)
from plot_tools import (
    FEATURE_DETAILS,
    plot_category_frequencies,
    plot_clustering_comparison,
    plot_nonsense_clusters,
    plot_entropy_vs_bins,
    plot_histogram,
    plot_pdf_and_cdf,
    plot_regression_predictions,
    plot_regression_residuals,
    plot_statistics_table,
    plot_two_features,
)
from stats_tools import (
    DEFAULT_BINS,
    NUMERICAL_FEATURES,
    compute_entropy,
    compute_kl_divergence,
    compute_mean_std,
    compute_skewness,
    describe_numerical_features,
)


NUMBER_OF_OBSERVATIONS = 600
RANDOM_SEED = 42
NUMBER_OF_BINS = DEFAULT_BINS


EXPECTED_COLUMNS = (*NUMERICAL_FEATURES, "field_zone")

EXPECTED_DTYPES = {
    "plant_height_cm": "numeric",
    "leaf_damage_percent": "numeric",
    "moisture_sensor_percent": "numeric",
    "ear_length_cm": "numeric",
    "grain_mass_per_ear_g": "numeric",
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
    "field_zone": FIELD_ZONE_ORDER,
}



def standardize(values):
    """Return values standardized to average zero and population deviation one."""
    values = np.asarray(values, dtype=float)
    mean = np.mean(values)
    std = np.std(values, ddof=0)
    if std == 0:
        raise ValueError("Cannot standardize a constant feature.")
    return (values - mean) / std


def common_histogram_probabilities(first, second, bins):
    """Return two probability histograms using identical numeric bin edges."""
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    lower = min(np.min(first), np.min(second))
    upper = max(np.max(first), np.max(second))
    edges = np.linspace(lower, upper, bins + 1)

    first_counts, _ = np.histogram(first, bins=edges)
    second_counts, _ = np.histogram(second, bins=edges)

    first_probabilities = first_counts / np.sum(first_counts)
    second_probabilities = second_counts / np.sum(second_counts)
    return first_probabilities, second_probabilities, edges


def gaussian_bin_probabilities(bin_edges, mean, std):
    """Return Gaussian probability mass inside each supplied histogram bin."""
    if std <= 0:
        raise ValueError("Gaussian standard deviation must be positive.")

    upper = 0.5 * (
        1
        + np.vectorize(math.erf)(
            (bin_edges[1:] - mean) / (std * np.sqrt(2))
        )
    )
    lower = 0.5 * (
        1
        + np.vectorize(math.erf)(
            (bin_edges[:-1] - mean) / (std * np.sqrt(2))
        )
    )
    probabilities = upper - lower
    probabilities = probabilities / np.sum(probabilities)
    return probabilities


def print_section(title):
    """Print a terminal section heading in the project report style."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_first_five_observations(dataset):
    """Print the first five observations in a compact table."""
    print("\nFirst five observations:")

    columns = [*NUMERICAL_FEATURES, "field_zone"]
    header = (
        f"{'':>3} "
        f"{'plant_height_cm':>16} "
        f"{'leaf_damage_percent':>21} "
        f"{'moisture_sensor_percent':>25} "
        f"{'ear_length_cm':>15} "
        f"{'grain_mass_per_ear_g':>23} "
        f"{'field_zone':>12}"
    )
    print(header)

    for index, (sample_id, observation) in enumerate(
        list(dataset.items())[:5]
    ):
        print(
            f"{index:>3} "
            f"{observation['plant_height_cm']:>16.2f} "
            f"{observation['leaf_damage_percent']:>21.2f} "
            f"{observation['moisture_sensor_percent']:>25.3f} "
            f"{observation['ear_length_cm']:>15.2f} "
            f"{observation['grain_mass_per_ear_g']:>23.2f} "
            f"{observation['field_zone']:>12}"
        )


def print_numerical_table(statistics):
    """Print average, standard deviation, skewness and entropy per feature."""
    print("\nSummary statistics:")

    header = (
        f"{'feature':>28} "
        f"{'mean':>9} "
        f"{'std':>9} "
        f"{'skewness':>11} "
        f"{'entropy_bits':>14}"
    )
    print(header)

    for feature, result in statistics.items():
        print(
            f"{feature:>28} "
            f"{result['average']:>9.4f} "
            f"{result['standard_deviation']:>9.4f} "
            f"{result['skewness']:>11.4f} "
            f"{result['entropy_bits']:>14.4f}"
        )


def print_categorical_frequencies(dataset):
    """Print category frequencies because field_zone is categorical."""
    frequencies = {
        zone: 0
        for zone in FIELD_ZONE_ORDER
    }
    for observation in dataset.values():
        frequencies[observation["field_zone"]] += 1

    print("\nField zone frequencies")
    print(f"{'Field zone':<16} {'Plants':>8}")
    print("-" * 25)
    for zone in FIELD_ZONE_ORDER:
        print(f"{zone:<16} {frequencies[zone]:>8}")


def create_all_figures(dataset, statistics, output_directory="figures"):
    """Create every Task 4 figure and return the paths of the saved PNG files."""
    figure_paths = []

    for feature in NUMERICAL_FEATURES:
        values = [row[feature] for row in dataset.values()]
        figure_paths.append(
            plot_histogram(
                values,
                feature,
                bins=NUMBER_OF_BINS,
                output_directory=output_directory,
            )
        )
        figure_paths.append(
            plot_pdf_and_cdf(
                values,
                feature,
                bins=NUMBER_OF_BINS,
                output_directory=output_directory,
            )
        )

    field_zones = [row["field_zone"] for row in dataset.values()]
    figure_paths.append(
        plot_category_frequencies(field_zones, output_directory)
    )
    figure_paths.append(
        plot_two_features(dataset, output_directory=output_directory)
    )

    # Keep the partner's requested entropy-vs-bins experiment:
    # use the clearly skewed leaf-damage feature.
    leaf_damage = [row["leaf_damage_percent"] for row in dataset.values()]
    figure_paths.append(
        plot_entropy_vs_bins(
            leaf_damage,
            feature="leaf_damage_percent",
            output_directory=output_directory,
        )
    )
    figure_paths.append(
        plot_statistics_table(statistics, output_directory)
    )

    return figure_paths


def run_task5(dataset):
    """Run the Task 5 rescaling, shifting, entropy and correlation calculations."""
    feature = "ear_length_cm"
    original_values = np.asarray(
        [row[feature] for row in dataset.values()],
        dtype=float,
    )

    scaled_values = original_values * 1000
    shifted_values = original_values + 50

    _, original_std = compute_mean_std(original_values)
    _, scaled_std = compute_mean_std(scaled_values)
    _, shifted_std = compute_mean_std(shifted_values)

    original_entropy = compute_entropy(original_values, NUMBER_OF_BINS)
    scaled_entropy = compute_entropy(scaled_values, NUMBER_OF_BINS)
    shifted_entropy = compute_entropy(shifted_values, NUMBER_OF_BINS)

    other_values = np.asarray(
        [row["grain_mass_per_ear_g"] for row in dataset.values()],
        dtype=float,
    )
    correlation = float(np.corrcoef(original_values, other_values)[0, 1])

    return {
        "feature": feature,
        "original_std": original_std,
        "scaled_std": scaled_std,
        "shifted_std": shifted_std,
        "original_entropy": original_entropy,
        "scaled_entropy": scaled_entropy,
        "shifted_entropy": shifted_entropy,
        "correlation_ear_vs_grain": correlation,
    }


def run_task6(dataset):
    """Run both required KL experiments using common histogram bins."""
    feature_a = standardize(
        [row["ear_length_cm"] for row in dataset.values()]
    )
    feature_b = standardize(
        [row["grain_mass_per_ear_g"] for row in dataset.values()]
    )

    p, q, common_edges = common_histogram_probabilities(
        feature_a,
        feature_b,
        NUMBER_OF_BINS,
    )
    feature_kl = {
        "D_KL(ear_length || grain_mass)": compute_kl_divergence(p, q),
        "D_KL(grain_mass || ear_length)": compute_kl_divergence(q, p),
        "bin_edges": common_edges,
    }

    gaussian_results = []
    for feature in NUMERICAL_FEATURES:
        values = np.asarray(
            [row[feature] for row in dataset.values()],
            dtype=float,
        )
        mean, std = compute_mean_std(values)
        counts, edges = np.histogram(values, bins=NUMBER_OF_BINS)
        p_real = counts / np.sum(counts)
        q_gaussian = gaussian_bin_probabilities(edges, mean, std)
        gaussian_results.append(
            {
                "feature": feature,
                "KL_real_to_gaussian": compute_kl_divergence(
                    p_real,
                    q_gaussian,
                ),
                "KL_gaussian_to_real": compute_kl_divergence(
                    q_gaussian,
                    p_real,
                ),
            }
        )

    return {
        "feature_comparison": feature_kl,
        "gaussian_comparison": gaussian_results,
    }


def run_task7(dataset):
    """Construct three distributions from the data and test KL metric properties."""
    # The three distributions are standardized before using the same five bins.
    # This particular triple gives a clear numerical counterexample to the
    # triangle inequality while still coming directly from our own dataset.
    values = {
        "P_ear_length": standardize(
            [row["ear_length_cm"] for row in dataset.values()]
        ),
        "Q_moisture": standardize(
            [row["moisture_sensor_percent"] for row in dataset.values()]
        ),
        "R_leaf_damage": standardize(
            [row["leaf_damage_percent"] for row in dataset.values()]
        ),
    }

    all_values = np.concatenate(list(values.values()))
    edges = np.linspace(
        np.min(all_values),
        np.max(all_values),
        6,
    )

    distributions = {}
    for name, array in values.items():
        counts, _ = np.histogram(array, bins=edges)
        distributions[name] = counts / np.sum(counts)

    p = distributions["P_ear_length"]
    q = distributions["Q_moisture"]
    r = distributions["R_leaf_damage"]

    p_to_q = compute_kl_divergence(p, q)
    q_to_r = compute_kl_divergence(q, r)
    p_to_r = compute_kl_divergence(p, r)

    # A second example uses the real leaf-damage histogram and the matching
    # Gaussian from Task 6. This shows why direction matters for a distance.
    damage = np.asarray(
        [row["leaf_damage_percent"] for row in dataset.values()],
        dtype=float,
    )
    mean, std = compute_mean_std(damage)
    counts, damage_edges = np.histogram(damage, bins=NUMBER_OF_BINS)
    damage_p = counts / np.sum(counts)
    damage_q = gaussian_bin_probabilities(damage_edges, mean, std)

    return {
        "P_name": "standardised ear_length_cm",
        "Q_name": "standardised moisture_sensor_percent",
        "R_name": "standardised leaf_damage_percent",
        "P_to_Q": p_to_q,
        "Q_to_R": q_to_r,
        "P_to_R": p_to_r,
        "triangle_left": p_to_q + q_to_r,
        "triangle_right": p_to_r,
        "triangle_holds": p_to_r <= p_to_q + q_to_r + 1e-12,
        "damage_to_gaussian": compute_kl_divergence(damage_p, damage_q),
        "gaussian_to_damage": compute_kl_divergence(damage_q, damage_p),
        "edges": edges,
    }


def print_task6_results(results):
    """Print the Task 6 KL results in compact table form."""
    print("\nKL divergence between standardized features:")
    print(
        f"{'feature pair':<42} "
        f"{'KL(P || Q)':>14} "
        f"{'KL(Q || P)':>14}"
    )
    print(
        f"{'ear_length_cm vs grain_mass_per_ear_g':<42} "
        f"{results['feature_comparison']['D_KL(ear_length || grain_mass)']:>14.6f} "
        f"{results['feature_comparison']['D_KL(grain_mass || ear_length)']:>14.6f}"
    )

    print("\nKL divergence to a Gaussian with the same mean and standard deviation:")
    print(
        f"{'feature':>28} "
        f"{'KL_P_to_Q_bits':>18} "
        f"{'KL_Q_to_P_bits':>18}"
    )
    for result in results["gaussian_comparison"]:
        reverse = result["KL_gaussian_to_real"]
        reverse_text = "inf" if np.isinf(reverse) else f"{reverse:.6f}"
        print(
            f"{result['feature']:>28} "
            f"{result['KL_real_to_gaussian']:>18.6f} "
            f"{reverse_text:>18}"
        )


def run_task8_nonsense_clustering(dataset, purpose_data,
                                  output_directory="figures", n_clusters=3):
    """Create deliberately meaningless clusters from an irrelevant sensor.

    K-means is valid, but its only input is the almost constant moisture sensor.
    The labels are evaluated both in that sensor space and in the meaningful
    productivity-feature space to show why the result fails the stated purpose.
    """
    sensor_data = prepare_clustering_data(
        dataset,
        features=("moisture_sensor_percent",),
    )
    model, labels, sensor_evaluation = fit_kmeans_clustering(
        sensor_data["values"],
        n_clusters=n_clusters,
        random_seed=RANDOM_SEED,
    )
    purpose_evaluation = evaluate_clustering(purpose_data["values"], labels)
    profiles = cluster_profiles(
        purpose_data["original_values"],
        labels,
        TASK8_FEATURES,
    )
    figure_path = plot_nonsense_clusters(
        sensor_data["original_values"][:, 0],
        purpose_data["original_values"][:, 1],
        labels,
        output_directory=output_directory,
    )
    return {
        "model": model,
        "labels": labels,
        "sensor_evaluation": sensor_evaluation,
        "purpose_evaluation": purpose_evaluation,
        "profiles": profiles,
        "figure_path": figure_path,
    }


def run_task8(dataset, output_directory="figures", n_clusters=3):
    """Fit and compare K-means and a full-covariance Gaussian mixture.

    The analysis asks whether plants form distinct productivity groups based on
    ear development, grain production and leaf damage.  The three selected
    features are standardized because they use different units.  Both methods
    receive exactly the same observations and prepared values.
    """
    prepared = prepare_clustering_data(dataset, TASK8_FEATURES)
    values = prepared["values"]

    kmeans_model, kmeans_labels, kmeans_evaluation = fit_kmeans_clustering(
        values,
        n_clusters=n_clusters,
        random_seed=RANDOM_SEED,
    )
    gmm_model, gmm_labels, gmm_evaluation = fit_gmm_clustering(
        values,
        n_components=n_clusters,
        random_seed=RANDOM_SEED,
    )

    projection = project_to_two_dimensions(values)
    figure_path = plot_clustering_comparison(
        projection,
        kmeans_labels,
        gmm_labels,
        output_directory=output_directory,
    )
    nonsense = run_task8_nonsense_clustering(
        dataset,
        prepared,
        output_directory=output_directory,
        n_clusters=n_clusters,
    )

    return {
        "features": TASK8_FEATURES,
        "prepared_data": prepared,
        "kmeans": {
            "model": kmeans_model,
            "labels": kmeans_labels,
            "evaluation": kmeans_evaluation,
            "profiles": cluster_profiles(
                prepared["original_values"], kmeans_labels, TASK8_FEATURES
            ),
        },
        "gmm": {
            "model": gmm_model,
            "labels": gmm_labels,
            "evaluation": gmm_evaluation,
            "profiles": cluster_profiles(
                prepared["original_values"], gmm_labels, TASK8_FEATURES
            ),
        },
        "figure_path": figure_path,
        "nonsense": nonsense,
    }


def print_task8_results(results):
    """Print comparable Task 8 scores, sizes and original-unit profiles."""
    print("\nQuestion: do plants form distinct productivity groups based on ear "
          "development, grain production and leaf damage?")
    print("\nSelected features:")
    for feature in results["features"]:
        print(f"  - {feature}")

    print("\nMethod comparison:")
    print(f"{'method':<28} {'silhouette':>12} {'noise':>8} {'cluster sizes':>25}")
    for method_key, method_name in (("kmeans", "K-means"), ("gmm", "GMM")):
        evaluation = results[method_key]["evaluation"]
        score = evaluation["silhouette_score"]
        score_text = f"{score:.4f}" if evaluation["silhouette_defined"] else "undefined"
        sizes = ", ".join(
            f"{label}: {size}"
            for label, size in evaluation["cluster_sizes"].items()
        )
        print(
            f"{method_name:<28} {score_text:>12} "
            f"{evaluation['noise_points']:>8} {sizes:>25}"
        )
        if not evaluation["silhouette_defined"]:
            print(f"  {evaluation['silhouette_note']}")

    print("\nCluster profiles (averages in original units):")
    for method_key, method_name in (("kmeans", "K-means"), ("gmm", "GMM")):
        print(f"\n{method_name}")
        for label, profile in results[method_key]["profiles"].items():
            averages = profile["averages"]
            print(
                f"  Cluster {label} (n={profile['size']}): "
                f"ear length={averages['ear_length_cm']:.2f} cm, "
                f"grain mass={averages['grain_mass_per_ear_g']:.2f} g, "
                f"leaf damage={averages['leaf_damage_percent']:.2f}%"
            )

    nonsense = results["nonsense"]
    sensor_score = nonsense["sensor_evaluation"]["silhouette_score"]
    purpose_score = nonsense["purpose_evaluation"]["silhouette_score"]
    print("\nNonsense experiment: K-means using only moisture-sensor noise")
    print(f"  Silhouette in sensor space:       {sensor_score:.4f}")
    print(f"  Silhouette in productivity space: {purpose_score:.4f}")
    print("  Nonsense cluster profiles:")
    for label, profile in nonsense["profiles"].items():
        averages = profile["averages"]
        print(
            f"    Cluster {label} (n={profile['size']}): "
            f"ear length={averages['ear_length_cm']:.2f} cm, "
            f"grain mass={averages['grain_mass_per_ear_g']:.2f} g, "
            f"leaf damage={averages['leaf_damage_percent']:.2f}%"
        )


# ---------------------------------------------------------------------------
# Task 9: linear regression — useful even when wrong?
# ---------------------------------------------------------------------------


def run_task9(dataset, output_directory="figures"):
    """Run the complete Task 9 regression workflow.

    The prediction question is whether grain mass per ear, in grams, can be
    predicted from plant measurements.  Rows are split before fitting.  Model
    choices and failure diagnosis use validation data; the final test rows are
    used only after the configurations have been fixed and refitted on the
    combined training and validation rows.
    """
    sensible_predictors = (
        "ear_length_cm",
        "plant_height_cm",
        "leaf_damage_percent",
    )
    target = "grain_mass_per_ear_g"
    sensible_data = prepare_regression_data(
        dataset,
        sensible_predictors,
        target,
    )
    wrong_data = prepare_regression_data(
        dataset,
        ("ear_length_cm",),
        target,
    )
    split = split_regression_data(
        len(dataset),
        train_fraction=0.60,
        validation_fraction=0.20,
        random_seed=RANDOM_SEED,
    )
    train = split["train"]
    validation = split["validation"]
    test = split["test"]
    targets = sensible_data["targets"]

    # Validation models are fitted only on training rows.  Their residuals
    # diagnose the deliberately omitted predictors before the test is opened.
    validation_sensible_model = fit_linear_regression(
        sensible_data["values"][train], targets[train]
    )
    validation_wrong_model = fit_linear_regression(
        wrong_data["values"][train], targets[train]
    )
    validation_baseline = fit_mean_baseline(targets[train])
    validation_predictions = {
        "baseline": predict_regression(
            validation_baseline, sensible_data["values"][validation]
        ),
        "sensible": predict_regression(
            validation_sensible_model, sensible_data["values"][validation]
        ),
        "wrong": predict_regression(
            validation_wrong_model, wrong_data["values"][validation]
        ),
    }
    validation_evaluations = {
        name: evaluate_regression(targets[validation], predictions)
        for name, predictions in validation_predictions.items()
    }
    validation_damage = sensible_data["values"][validation, 2]
    failure_diagnosis = {
        "sensible_residual_damage_correlation": float(
            np.corrcoef(
                validation_damage,
                targets[validation] - validation_predictions["sensible"],
            )[0, 1]
        ),
        "wrong_residual_damage_correlation": float(
            np.corrcoef(
                validation_damage,
                targets[validation] - validation_predictions["wrong"],
            )[0, 1]
        ),
    }
    residual_figure = plot_regression_residuals(
        targets[validation],
        validation_predictions["sensible"],
        validation_predictions["wrong"],
        sensible_data["values"][validation, 2],
        output_directory=output_directory,
    )

    # Final models use all development rows.  The reserved test is evaluated
    # once, using the same rows for the baseline, sensible and wrong models.
    development = np.concatenate((train, validation))
    sensible_model = fit_linear_regression(
        sensible_data["values"][development], targets[development]
    )
    wrong_model = fit_linear_regression(
        wrong_data["values"][development], targets[development]
    )
    baseline_model = fit_mean_baseline(targets[development])
    test_predictions = {
        "Baseline": predict_regression(
            baseline_model, sensible_data["values"][test]
        ),
        "Sensible model": predict_regression(
            sensible_model, sensible_data["values"][test]
        ),
        "Wrong ear-length-only model": predict_regression(
            wrong_model, wrong_data["values"][test]
        ),
    }
    test_evaluations = {
        name: evaluate_regression(targets[test], predictions)
        for name, predictions in test_predictions.items()
    }
    prediction_figure = plot_regression_predictions(
        targets[test],
        test_predictions,
        output_directory=output_directory,
    )

    wrong_predictions = test_predictions["Wrong ear-length-only model"]
    ranking_correlation = float(np.corrcoef(wrong_predictions, targets[test])[0, 1])
    top_quarter_cutoff = float(np.quantile(wrong_predictions, 0.75))
    top_quarter = wrong_predictions >= top_quarter_cutoff
    usefulness = {
        "prediction_observation_correlation": ranking_correlation,
        "overall_actual_mean": float(np.mean(targets[test])),
        "top_quarter_actual_mean": float(np.mean(targets[test][top_quarter])),
        "top_quarter_size": int(np.sum(top_quarter)),
    }

    return {
        "question": (
            "Can grain mass per ear (g) be predicted from plant measurements?"
        ),
        "target": target,
        "sensible_predictors": sensible_predictors,
        "wrong_predictors": ("ear_length_cm",),
        "split_sizes": {name: len(indices) for name, indices in split.items()},
        "validation_evaluations": validation_evaluations,
        "failure_diagnosis": failure_diagnosis,
        "models": {
            "baseline": baseline_model,
            "sensible": sensible_model,
            "wrong": wrong_model,
        },
        "test_evaluations": test_evaluations,
        "test_predictions": test_predictions,
        "usefulness": usefulness,
        "figure_paths": [prediction_figure, residual_figure],
    }


def print_task9_results(results):
    """Print Task 9 split sizes, test metrics, coefficients and useful scope."""
    print(f"\nPrediction question: {results['question']}")
    sizes = results["split_sizes"]
    print(
        f"Split sizes — train: {sizes['train']}, validation: "
        f"{sizes['validation']}, test: {sizes['test']}"
    )
    print("\nFinal reserved-test performance:")
    print(f"{'model':<32} {'MAE (g)':>10} {'RMSE (g)':>11} {'R-squared':>12}")
    for name, evaluation in results["test_evaluations"].items():
        r_squared = evaluation["r_squared"]
        r_text = f"{r_squared:.4f}" if evaluation["r_squared_defined"] else "undefined"
        print(
            f"{name:<32} {evaluation['mae']:>10.4f} "
            f"{evaluation['rmse']:>11.4f} {r_text:>12}"
        )

    model = results["models"]["sensible"]
    print("\nSensible-model coefficients:")
    print(f"  Intercept: {model.intercept_:.4f} g")
    for feature, coefficient in zip(results["sensible_predictors"], model.coef_):
        print(f"  {feature}: {coefficient:.4f}")

    usefulness = results["usefulness"]
    diagnosis = results["failure_diagnosis"]
    print("\nValidation residual correlation with omitted leaf damage:")
    print(
        f"  Sensible model: {diagnosis['sensible_residual_damage_correlation']:.4f}; "
        f"wrong model: {diagnosis['wrong_residual_damage_correlation']:.4f}"
    )
    print("\nWrong-model limited use as a ranking tool:")
    print(
        "  Prediction-observation correlation: "
        f"{usefulness['prediction_observation_correlation']:.4f}"
    )
    print(
        f"  Overall actual mean: {usefulness['overall_actual_mean']:.2f} g; "
        f"top predicted quarter: {usefulness['top_quarter_actual_mean']:.2f} g"
    )
    print("  Saved 2 Task 9 figures in the selected figures directory.")


def main(
    number_of_observations=NUMBER_OF_OBSERVATIONS,
    seed=RANDOM_SEED,
    output_directory="figures",
    save_dataset=True,
    output_filename="data_tools_output_synthetic.py",
):
    """Run Tasks 2-7 and return the generated data and analysis results."""
    print_section("TASK 2 — GENERATING SYNTHETIC MAIZE DATA")

    output_filename = output_filename if save_dataset else None
    dataset = generate_maize_dataset(
        number_of_observations=number_of_observations,
        seed=seed,
        output_filename=output_filename,
    )

    print(
        f"Generated dataset with {len(dataset)} observations and "
        f"{len(EXPECTED_COLUMNS)} columns."
    )
    print_first_five_observations(dataset)

    print_section("TASK 3 — DATA INTEGRITY")
    report = check_data_integrity(
        dataset,
        expected_columns=EXPECTED_COLUMNS,
        expected_shape=(number_of_observations, len(EXPECTED_COLUMNS)),
        expected_dtypes=EXPECTED_DTYPES,
        expected_ranges=EXPECTED_RANGES,
        expected_categories=EXPECTED_CATEGORIES,
        expected_sample_count=number_of_observations,
        data_name="synthetic maize dataset",
    )
    if not report["valid"]:
        raise ValueError("The generated dataset failed the integrity check.")

    print_section("TASK 4 — NUMERICAL DISTRIBUTIONS")
    statistics = describe_numerical_features(dataset, bins=NUMBER_OF_BINS)
    print_numerical_table(statistics)

    print("\nCreating distribution plots...")
    figure_paths = create_all_figures(
        dataset,
        statistics,
        output_directory=output_directory,
    )
    print("\nAll Task 4 figures were saved in the figures/ directory.")

    print_section("TASK 5 — RESCALING AND SHIFTING")
    task5 = run_task5(dataset)
    print(f"\nFeature: ear_length_cm")
    print(f"Number of bins: {NUMBER_OF_BINS}")

    print("\nOriginal:")
    print(f"  Standard deviation: {task5['original_std']:.4f}")
    print(f"  Entropy:            {task5['original_entropy']:.4f} bits")

    print("\nAfter multiplying by 1000:")
    print(f"  Standard deviation: {task5['scaled_std']:.4f}")
    print(f"  Entropy:            {task5['scaled_entropy']:.4f} bits")

    print("\nAfter adding 50:")
    print(f"  Standard deviation: {task5['shifted_std']:.4f}")
    print(f"  Entropy:            {task5['shifted_entropy']:.4f} bits")

    print_section("TASK 5 — FEATURE SELECTION")
    correlation_matrix = np.corrcoef(
        np.asarray(
            [
                [row[feature] for feature in NUMERICAL_FEATURES]
                for row in dataset.values()
            ],
            dtype=float,
        ),
        rowvar=False,
    )
    print("\nCorrelation matrix:")
    print(
        f"{'':>28} " + " ".join(
            f"{feature:>24}" for feature in NUMERICAL_FEATURES
        )
    )
    for i, feature in enumerate(NUMERICAL_FEATURES):
        print(
            f"{feature:>28} "
            + " ".join(f"{value:>24.3f}" for value in correlation_matrix[i])
        )

    print("\nEntropy of each feature:")
    for feature in NUMERICAL_FEATURES:
        print(
            f"  {feature}: "
            f"{statistics[feature]['entropy_bits']:.4f} bits"
        )

    print_section("TASK 6 — KL DIVERGENCE")
    task6 = run_task6(dataset)
    print_task6_results(task6)

    task7 = run_task7(dataset)
    print_section("TASK 7 — IS KL A DISTANCE?")
    print(f"  D(P || Q) = {task7['P_to_Q']:.6f} bits")
    print(f"  D(Q || R) = {task7['Q_to_R']:.6f} bits")
    print(f"  D(P || R) = {task7['P_to_R']:.6f} bits")
    print(
        f"  Left side D(P||Q)+D(Q||R) = {task7['triangle_left']:.6f}"
    )
    print(f"  Right side D(P||R)         = {task7['triangle_right']:.6f}")
    print(f"  Triangle inequality holds? {task7['triangle_holds']}")
    print(
        "  Leaf-damage||Gaussian = "
        f"{task7['damage_to_gaussian']:.6f}; "
        "Gaussian||leaf-damage = "
        f"{task7['gaussian_to_damage']}"
    )

    print_section("TASK 8 — CLUSTERING: CHOOSE, BREAK, EXPLAIN")
    task8 = run_task8(dataset, output_directory=output_directory)
    print_task8_results(task8)
    figure_paths.append(task8["figure_path"])
    figure_paths.append(task8["nonsense"]["figure_path"])

    print_section("TASK 9 — LINEAR REGRESSION: USEFUL EVEN WHEN WRONG?")
    task9 = run_task9(dataset, output_directory=output_directory)
    print_task9_results(task9)

    print_section("DATASET INFORMATION")

    print("\nNumerical columns:")
    for feature in NUMERICAL_FEATURES:
        print(f"  - {feature}")

    print("\nCategorical column:")
    print("  - field_zone")

    print("\nNumber of observations:")
    print(len(dataset))

    print("\nNumber of numerical features:")
    print(len(NUMERICAL_FEATURES))

    print("\nNumber of categorical features:")
    print(1)

    print("\nChosen histogram bin count:")
    print(NUMBER_OF_BINS)

    print(
        "\nCorrelation between ear length and grain mass: "
        f"{task5['correlation_ear_vs_grain']:.3f}"
    )

    print(f"\nSaved {len(figure_paths)} Tasks 4 and 8 figures in:")
    print(Path(figure_paths[0]).parent)

    return {
        "dataset": dataset,
        "integrity_report": report,
        "statistics": statistics,
        "figure_paths": figure_paths,
        "task5": task5,
        "task6": task6,
        "task7": task7,
        "task8": task8,
        "task9": task9,
    }


if __name__ == "__main__":
    main()