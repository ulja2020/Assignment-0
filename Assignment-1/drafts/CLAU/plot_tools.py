"""Create and save the figures required for Assignment 1 Task 4.

The functions save PNG files under ``figures/`` and return their paths. They do
not open the files. ``code.py`` coordinates figure creation and opens every PNG
after all figures have been written successfully.
"""

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

from stats_tools import DEFAULT_BINS, calculate_cdf, calculate_entropy, calculate_pdf


FEATURE_DETAILS = {
    "plant_height_cm": ("Plant height", "cm", "plant_height"),
    "leaf_damage_percent": ("Leaf damage", "%", "leaf_damage"),
    "moisture_sensor_percent": ("Moisture sensor reading", "%", "moisture_sensor"),
    "ear_length_cm": ("Ear length", "cm", "ear_length"),
    "grain_mass_per_ear_g": ("Grain mass per ear", "g", "grain_mass_per_ear"),
}
FIELD_ZONE_ORDER = ["Northwest", "Northeast", "Centre", "Southwest", "Southeast"]


def _figures_directory(output_directory="figures"):
    """Return the output directory, creating it beside this module if needed."""
    directory = Path(output_directory)
    if not directory.is_absolute():
        directory = Path(__file__).resolve().parent / directory
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _axis_label(feature):
    """Return a readable numerical-feature label that includes its unit."""
    label, unit, _ = FEATURE_DETAILS[feature]
    return f"{label} ({unit})"


def plot_histogram(values, feature, bins=DEFAULT_BINS, output_directory="figures"):
    """Save one frequency histogram for a numerical feature and return its path."""
    label, _, slug = FEATURE_DETAILS[feature]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(values, bins=bins, color="#236F9E", edgecolor="white")
    axis.set_title(f"Histogram of {label.lower()}")
    axis.set_xlabel(_axis_label(feature))
    axis.set_ylabel("Frequency (number of plants)")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()

    output_path = _figures_directory(output_directory) / f"{slug}_histogram.png"
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def plot_pdf_and_cdf(values, feature, bins=DEFAULT_BINS, output_directory="figures"):
    """Save one two-panel PDF and CDF figure for a feature and return its path."""
    label, _, slug = FEATURE_DETAILS[feature]
    bin_centres, densities, _ = calculate_pdf(values, bins)
    cdf_positions, cumulative_probabilities = calculate_cdf(values, bins)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(bin_centres, densities, color="#236F9E", marker="o", markersize=3)
    axes[0].set_title(f"PDF of {label.lower()}")
    axes[0].set_xlabel(_axis_label(feature))
    axes[0].set_ylabel("Probability density")
    axes[0].grid(True, alpha=0.25)

    axes[1].step(cdf_positions, cumulative_probabilities, where="post", color="#228B45")
    axes[1].set_title(f"CDF of {label.lower()}")
    axes[1].set_xlabel(_axis_label(feature))
    axes[1].set_ylabel("Cumulative probability")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(True, alpha=0.25)
    figure.tight_layout()

    output_path = _figures_directory(output_directory) / f"{slug}_pdf_cdf.png"
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def plot_category_frequencies(values, output_directory="figures"):
    """Save the field-zone frequency bar chart and return its output path.

    ``field_zone`` is categorical, so counts are meaningful while averages,
    deviations, numerical PDFs and numerical CDFs are not.
    """
    frequencies = Counter(values)
    counts = [frequencies[zone] for zone in FIELD_ZONE_ORDER]

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(FIELD_ZONE_ORDER, counts, color="#236F9E")
    axis.set_title("Frequency of sampled plants by field zone")
    axis.set_xlabel("Field zone")
    axis.set_ylabel("Frequency (number of plants)")
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()

    output_path = _figures_directory(output_directory) / "field_zone_frequency.png"
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def plot_two_features(
    dataset,
    x_feature="ear_length_cm",
    y_feature="grain_mass_per_ear_g",
    output_directory="figures",
):
    """Save a scatter plot of two numerical features and return its path."""
    x_values = [row[x_feature] for row in dataset.values()]
    y_values = [row[y_feature] for row in dataset.values()]

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.scatter(x_values, y_values, alpha=0.55, color="#236F9E", edgecolors="none")
    axis.set_title("Ear length and grain mass per ear")
    axis.set_xlabel(_axis_label(x_feature))
    axis.set_ylabel(_axis_label(y_feature))
    axis.grid(True, alpha=0.25)
    figure.tight_layout()

    output_path = _figures_directory(output_directory) / "ear_length_vs_grain_mass.png"
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def plot_entropy_vs_bins(
    values,
    feature="leaf_damage_percent",
    bin_counts=range(5, 101, 5),
    output_directory="figures",
):
    """Save entropy versus bin count using the Task 4 entropy function."""
    label, _, slug = FEATURE_DETAILS[feature]
    bin_counts = list(bin_counts)
    entropies = [calculate_entropy(values, bins) for bins in bin_counts]

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(bin_counts, entropies, marker="o", color="#236F9E")
    axis.set_title(f"Entropy of {label.lower()} as a function of histogram bins")
    axis.set_xlabel("Number of bins")
    axis.set_ylabel("Shannon entropy (bits)")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()

    output_path = _figures_directory(output_directory) / f"entropy_{slug}_vs_bins.png"
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def plot_statistics_table(statistics, output_directory="figures"):
    """Render the numerical statistics as a PNG table and return its path."""
    rows = []
    for feature, result in statistics.items():
        label = FEATURE_DETAILS[feature][0]
        rows.append(
            [
                label,
                f"{result['average']:.3f}",
                f"{result['standard_deviation']:.3f}",
                f"{result['skewness']:.3f}",
                f"{result['entropy_bits']:.3f}",
            ]
        )

    columns = ["Feature", "Average", "Std. deviation", "Skewness", "Entropy (bits)"]
    figure, axis = plt.subplots(figsize=(10, 3.2))
    axis.axis("off")
    table = axis.table(cellText=rows, colLabels=columns, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

    for column in range(len(columns)):
        table[(0, column)].set_facecolor("#17324D")
        table[(0, column)].set_text_props(color="white", weight="bold")
    for row in range(1, len(rows) + 1):
        background = "#EDF5FA" if row % 2 else "#FFFFFF"
        for column in range(len(columns)):
            table[(row, column)].set_facecolor(background)

    axis.set_title("Numerical feature statistics", fontsize=15, pad=14)
    figure.tight_layout()
    output_path = _figures_directory(output_directory) / "numerical_statistics_table.png"
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path
