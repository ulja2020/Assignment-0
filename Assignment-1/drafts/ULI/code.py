from pathlib import Path
import math

import numpy as np

from data_tools import FIELD_ZONE_ORDER, check_data_integrity, generate_maize_dataset
from plot_tools import (
    FEATURE_DETAILS,
    plot_category_frequencies,
    plot_entropy_vs_bins,
    plot_histogram,
    plot_pdf_and_cdf,
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

    print(f"\nSaved {len(figure_paths)} figures in:")
    print(Path(figure_paths[0]).parent)

    return {
        "dataset": dataset,
        "integrity_report": report,
        "statistics": statistics,
        "figure_paths": figure_paths,
        "task5": task5,
        "task6": task6,
        "task7": task7,
    }


if __name__ == "__main__":
    main()
