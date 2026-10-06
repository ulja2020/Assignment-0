"""Run the Task 4 description and plotting workflow.

Running this file directly loads the valid synthetic dataset, prints numerical
statistics and categorical frequencies, saves every required figure under
``figures/`` and then opens each PNG through Windows Explorer. Importing this
module does not run the workflow or open any files.
"""

from collections import Counter
from pathlib import Path
from runpy import run_path
import subprocess

from plot_tools import (
    FIELD_ZONE_ORDER,
    FEATURE_DETAILS,
    plot_category_frequencies,
    plot_entropy_vs_bins,
    plot_histogram,
    plot_pdf_and_cdf,
    plot_statistics_table,
    plot_two_features,
)
from stats_tools import NUMERICAL_FEATURES, describe_numerical_features


def load_synthetic_dataset():
    """Load and return the validated synthetic dataset created in Task 3."""
    dataset_path = Path(__file__).resolve().parent / "data_tools_output_synthetic.py"
    return run_path(str(dataset_path))["maize_harvest_data"]


def print_numerical_table(statistics):
    """Print average, deviation, skewness and entropy for each numerical feature."""
    header = (
        f"{'Feature':<26} {'Average':>11} {'Std. dev.':>11} "
        f"{'Skewness':>11} {'Entropy':>11}"
    )
    print("\nNumerical feature statistics")
    print(header)
    print("-" * len(header))

    for feature, result in statistics.items():
        label = FEATURE_DETAILS[feature][0]
        print(
            f"{label:<26} "
            f"{result['average']:>11.3f} "
            f"{result['standard_deviation']:>11.3f} "
            f"{result['skewness']:>11.3f} "
            f"{result['entropy_bits']:>11.3f}"
        )


def print_categorical_frequencies(dataset):
    """Print field-zone counts because categories require frequency summaries."""
    frequencies = Counter(row["field_zone"] for row in dataset.values())
    print("\nField zone frequencies")
    print(f"{'Field zone':<16} {'Plants':>8}")
    print("-" * 25)
    for zone in FIELD_ZONE_ORDER:
        print(f"{zone:<16} {frequencies[zone]:>8}")


def create_all_figures(dataset, statistics):
    """Create every Task 4 PNG and return their paths without opening them.

    Five histograms, five PDF/CDF figures, one category chart, one scatter plot,
    one entropy plot and one PNG statistics table are created under ``figures/``.
    """
    figure_paths = []

    for feature in NUMERICAL_FEATURES:
        values = [row[feature] for row in dataset.values()]
        figure_paths.append(plot_histogram(values, feature))
        figure_paths.append(plot_pdf_and_cdf(values, feature))

    field_zones = [row["field_zone"] for row in dataset.values()]
    figure_paths.append(plot_category_frequencies(field_zones))
    figure_paths.append(plot_two_features(dataset))

    leaf_damage = [row["leaf_damage_percent"] for row in dataset.values()]
    figure_paths.append(plot_entropy_vs_bins(leaf_damage))
    figure_paths.append(plot_statistics_table(statistics))
    return figure_paths


def open_figures(figure_paths):
    """Ask Windows Explorer to open every saved PNG with its default viewer."""
    for figure_path in figure_paths:
        subprocess.Popen(["explorer.exe", str(figure_path)])


def main(open_saved_figures=True):
    """Print both summaries, save all figures and optionally open every PNG.

    ``open_saved_figures`` defaults to true for normal desktop use. Setting it
    to false permits automated verification without opening fourteen windows.
    """
    dataset = load_synthetic_dataset()
    statistics = describe_numerical_features(dataset)

    print_numerical_table(statistics)
    print_categorical_frequencies(dataset)
    figure_paths = create_all_figures(dataset, statistics)

    print(f"\nSaved {len(figure_paths)} figures in:")
    print(figure_paths[0].parent)

    if open_saved_figures:
        open_figures(figure_paths)

    return figure_paths


if __name__ == "__main__":
    main()
