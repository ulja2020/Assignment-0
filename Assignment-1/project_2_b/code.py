
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_tools import (
    REQUIRED_RAW_COLUMNS,
    download_faostat_data,
    check_data_integrity,
    prepare_food_security_data,
)

PROJECT_2_A = Path(__file__).resolve().parent.parent / "project_2_a"
sys.path.append(str(PROJECT_2_A))

from stats_tools import (
    DEFAULT_BINS,
    compute_pdf,
    compute_cdf,
    compute_mean_std,
    compute_skewness,
    compute_entropy,
    compute_kl_divergence,
)

import plot_tools
from model_tools import (
    prepare_clustering_data,
    fit_kmeans_clustering,
    fit_gmm_clustering,
    project_to_two_dimensions,
    split_regression_data,
    fit_linear_regression,
    fit_mean_baseline,
    predict_regression,
    evaluate_regression,
)


NUMBER_OF_BINS = DEFAULT_BINS
RANDOM_SEED = 42
FIGURES_DIRECTORY = Path(__file__).resolve().parent / "figures"

TARGET = "undernourishment_percent"

PREDICTORS = (
    "energy_adequacy_percent",
    "protein_supply_g_cap_day",
    "animal_protein_g_cap_day",
    "cereal_share_percent",
    "food_supply_variability_kcal_cap_day",
    "energy_requirement_kcal_cap_day",
)

WRONG_PREDICTORS = ("energy_adequacy_percent",)

ITEMS = {
    "energy_adequacy_percent": (
        "Average dietary energy supply adequacy (percent) (3-year average)",
        "%",
        "Energy supply adequacy (%)",
        "energy_adequacy_percent",
    ),
    "cereal_share_percent": (
        "Share of dietary energy supply derived from cereals, roots and tubers (percent) (3-year average)",
        "%",
        "Cereal/root/tuber share (%)",
        "cereal_share_percent",
    ),
    "protein_supply_g_cap_day": (
        "Average protein supply (g/cap/day) (3-year average)",
        "g/cap/day",
        "Protein supply (g/cap/day)",
        "protein_supply_g_cap_day",
    ),
    "animal_protein_g_cap_day": (
        "Average supply of protein of animal origin (g/cap/day) (3-year average)",
        "g/cap/day",
        "Animal protein supply (g/cap/day)",
        "animal_protein_g_cap_day",
    ),
    "food_supply_variability_kcal_cap_day": (
        "Per capita food supply variability (kcal/cap/day)",
        "kcal/cap/day",
        "Food supply variability (kcal/cap/day)",
        "food_supply_variability_kcal_cap_day",
    ),
    "energy_requirement_kcal_cap_day": (
        "Average dietary energy requirement (kcal/cap/day)",
        "kcal/cap/day",
        "Dietary energy requirement (kcal/cap/day)",
        "energy_requirement_kcal_cap_day",
    ),
    TARGET: (
        "Prevalence of undernourishment (percent) (3-year average)",
        "%",
        "Undernourishment (%)",
        "undernourishment_percent",
    ),
}

plot_tools.FEATURE_DETAILS = {
    key: (details[2], details[1], details[3])
    for key, details in ITEMS.items()
}


def _make_cluster_dataset(data):
    """Convert the analysis table to the row format expected by clustering tools."""
    return {
        str(index): {feature: float(row[feature]) for feature in PREDICTORS}
        for index, row in data.iterrows()
    }


def _probabilities(values, bins=NUMBER_OF_BINS):
    """Return histogram probabilities and bin edges."""
    counts, edges = np.histogram(values, bins=bins)
    return counts / np.sum(counts), edges


def _gaussian_probabilities(values, bins=NUMBER_OF_BINS):
    """Return Gaussian probabilities on the same histogram bins as the data."""
    _, edges = np.histogram(values, bins=bins)
    mean, std = compute_mean_std(values)
    if std == 0:
        probabilities = np.zeros(bins)
        probabilities[0] = 1.0
        return probabilities

    upper = 0.5 * (
        1 + np.vectorize(math.erf)((edges[1:] - mean) / (std * np.sqrt(2)))
    )
    lower = 0.5 * (
        1 + np.vectorize(math.erf)((edges[:-1] - mean) / (std * np.sqrt(2)))
    )
    probabilities = upper - lower
    return probabilities / np.sum(probabilities)


def _standardized_kl(data, feature_a, feature_b):
    """Compare two standardised features with KL divergence in both directions."""
    values_a = data[feature_a].to_numpy(dtype=float)
    values_b = data[feature_b].to_numpy(dtype=float)
    mean_a, std_a = compute_mean_std(values_a)
    mean_b, std_b = compute_mean_std(values_b)

    values_a = (values_a - mean_a) / std_a
    values_b = (values_b - mean_b) / std_b

    low = min(values_a.min(), values_b.min())
    high = max(values_a.max(), values_b.max())
    counts_a, _ = np.histogram(values_a, bins=NUMBER_OF_BINS, range=(low, high))
    counts_b, _ = np.histogram(values_b, bins=NUMBER_OF_BINS, range=(low, high))

    p = counts_a / np.sum(counts_a)
    q = counts_b / np.sum(counts_b)
    return compute_kl_divergence(p, q), compute_kl_divergence(q, p)


def _print_statistics_table(statistics):
    """Print mean, standard deviation, skewness and entropy."""
    rows = []
    for feature, result in statistics.items():
        rows.append({
            "feature": feature,
            "mean": result["mean"],
            "std": result["std"],
            "skewness": result["skewness"],
            "entropy_bits": result["entropy_bits"],
        })
    table = pd.DataFrame(rows)
    print(table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def _print_regression_table(evaluations):
    """Print regression evaluation metrics."""
    rows = []
    for model, result in evaluations.items():
        rows.append({
            "model": model,
            "MAE": result["mae"],
            "RMSE": result["rmse"],
            "R2": result["r_squared"],
        })
    table = pd.DataFrame(rows)
    print(table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))



def _save(figure, name):
    """Save a figure in the figures directory and return its path."""
    FIGURES_DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIRECTORY / name
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return path


def _plot_category_frequencies(values, category_name, top_n=15):
    """Bar chart of the most frequent categories."""
    counts = pd.Series(values).value_counts().head(top_n)
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(counts.index.astype(str), counts.values)
    axis.set_title(f"Frequency of observations by {category_name}")
    axis.set_xlabel(category_name)
    axis.set_ylabel("Frequency (number of observations)")
    axis.tick_params(axis="x", rotation=60)
    axis.grid(axis="y", alpha=0.25)
    return _save(figure, f"{category_name.lower().replace(' ', '_')}_frequency.png")


def _plot_two_features(data, x_feature, y_feature):
    """Scatter plot of two numerical features."""
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.scatter(data[x_feature], data[y_feature], alpha=0.35, s=14, edgecolors="none")
    axis.set_xlabel(ITEMS[x_feature][2])
    axis.set_ylabel(ITEMS[y_feature][2])
    axis.set_title(f"{ITEMS[x_feature][2]} versus {ITEMS[y_feature][2]}")
    axis.grid(True, alpha=0.25)
    return _save(figure, f"{x_feature}_vs_{y_feature}.png")


def _plot_clustering_comparison(projection, kmeans_labels, gmm_labels, title):
    """K-means and GMM labels on the same PCA axes."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    vmax = int(max(np.max(kmeans_labels), np.max(gmm_labels)))
    for axis, (name, labels) in zip(axes, (("K-means", kmeans_labels), ("Gaussian mixture model", gmm_labels))):
        axis.scatter(projection[:, 0], projection[:, 1], c=labels, cmap="viridis",
                     vmin=0, vmax=vmax, s=14, alpha=0.7, edgecolors="none")
        axis.set_title(name)
        axis.set_xlabel("Principal component 1 (standardized features)")
        axis.grid(True, alpha=0.22)
    axes[0].set_ylabel("Principal component 2 (standardized features)")
    figure.suptitle(title)
    figure.tight_layout()
    return _save(figure, "task12_clustering_comparison.png")


def _plot_nonsense_clusters(x_values, y_values, labels, x_label, y_label, title):
    """Scatter plot of the deliberately bad clustering."""
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.scatter(x_values, y_values, c=labels, cmap="viridis", s=14, alpha=0.7, edgecolors="none")
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.set_title(title)
    axis.grid(True, alpha=0.22)
    return _save(figure, "task12_nonsense_clusters.png")


def _plot_observed_vs_predicted(targets, predictions, target_label):
    """Observed versus predicted target for every model on the same axes."""
    values = np.concatenate([targets, *predictions.values()])
    low, high = float(values.min()), float(values.max())
    margin = 0.04 * (high - low)
    limits = (low - margin, high + margin)
    figure, axes = plt.subplots(1, len(predictions), figsize=(14, 4.5), sharex=True, sharey=True)
    for axis, (name, predicted) in zip(axes, predictions.items()):
        axis.scatter(targets, predicted, alpha=0.5, s=14, edgecolors="none")
        axis.plot(limits, limits, linestyle="--", color="black", linewidth=1)
        axis.set_title(name)
        axis.set_xlabel(f"Observed {target_label}")
        axis.set_xlim(limits)
        axis.set_ylim(limits)
        axis.grid(True, alpha=0.22)
    axes[0].set_ylabel(f"Predicted {target_label}")
    figure.suptitle("Task 12: Observed versus predicted (held-out test set)")
    figure.tight_layout()
    return _save(figure, "task12_observed_vs_predicted.png")


def _plot_residuals(targets, predictions, target_label):
    """Residuals versus predictions for every model."""
    figure, axes = plt.subplots(1, len(predictions), figsize=(14, 4.5), sharey=True)
    for axis, (name, predicted) in zip(axes, predictions.items()):
        axis.scatter(predicted, targets - predicted, alpha=0.5, s=14, edgecolors="none")
        axis.axhline(0, linestyle="--", color="black", linewidth=1)
        axis.set_title(name)
        axis.set_xlabel(f"Predicted {target_label}")
        axis.grid(True, alpha=0.22)
    axes[0].set_ylabel(f"Residual: observed minus predicted {target_label}")
    figure.suptitle("Task 12: Residuals (held-out test set)")
    figure.tight_layout()
    return _save(figure, "task12_residuals.png")


def _plot_wrong_residuals(feature_values, residuals, feature_label, target_label):
    """Residuals of the wrong model against the feature it omitted."""
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.scatter(feature_values, residuals, alpha=0.5, s=14, edgecolors="none")
    axis.axhline(0, linestyle="--", color="black", linewidth=1)
    axis.set_xlabel(feature_label)
    axis.set_ylabel(f"Wrong-model residual ({target_label})")
    axis.set_title("Wrong-model residuals versus an omitted feature")
    axis.grid(True, alpha=0.22)
    return _save(figure, "task12_wrong_residuals_vs_omitted_feature.png")


def _rank_regression_models(evaluations, metric="rmse"):
    """Rank models by an error metric, lowest first."""
    ordered = sorted(evaluations.items(), key=lambda item: item[1][metric])
    return [{"rank": rank, "model": name, metric: result[metric]}
            for rank, (name, result) in enumerate(ordered, start=1)]


def _regression_coefficients(model, feature_names):
    """Return {feature: coefficient} and the intercept of a fitted model."""
    return dict(zip(feature_names, map(float, model.coef_))), float(model.intercept_)


def _split_rows(X, y):
    """60/20/20 train/validation/test split using project_2_a's index-based splitter."""
    index = split_regression_data(
        len(y), train_fraction=0.60, validation_fraction=0.20, random_seed=RANDOM_SEED
    )
    split = {"test_indices": index["test"]}
    for part in ("train", "validation", "test"):
        split[f"X_{part}"] = X[index[part]]
        split[f"y_{part}"] = y[index[part]]
    return split


def run_task12(data):
    """Run the full real-data analysis and return the main results."""
    
    print("\n" + "=" * 70)
    print("REAL-DATA WORKFLOW")
    print("=" * 70)

    raw_report = check_data_integrity(
        data,
        expected_columns=REQUIRED_RAW_COLUMNS,
        required_non_missing=["Area", "Year", "Item", "Unit"],
    )

    analysis_data, cleaning = prepare_food_security_data(
        data,
        [details[0] for details in ITEMS.values()],
    )
    rename = {details[0]: key for key, details in ITEMS.items()}
    analysis_data = analysis_data.rename(columns=rename)
    analysis_data = analysis_data.sort_values(["Area", "period_end"]).reset_index(drop=True)

    print("\nCleaning summary:")
    print(f"  '<2.5' lower-bound values: {cleaning['lower_bound_values']}")

    print(f"\nAnalysis dataset shape: {analysis_data.shape}")
    print("First five observations:")
    print(analysis_data[["Area", "period_end", *PREDICTORS, TARGET]].head().to_string(index=False))

    print("\n" + "=" * 70)
    print("NUMERICAL DISTRIBUTIONS")
    print("=" * 70)

    statistics = {}
    for feature in (*PREDICTORS, TARGET):
        values = analysis_data[feature].to_numpy(dtype=float)
        mean, std = compute_mean_std(values)
        statistics[feature] = {
            "mean": mean,
            "std": std,
            "average": mean,
            "standard_deviation": std,
            "skewness": compute_skewness(values),
            "entropy_bits": compute_entropy(values, bins=NUMBER_OF_BINS),
        }
        plot_tools.plot_histogram(values, feature, NUMBER_OF_BINS, str(FIGURES_DIRECTORY))
        plot_tools.plot_pdf_and_cdf(values, feature, NUMBER_OF_BINS, str(FIGURES_DIRECTORY))

    _print_statistics_table(statistics)
    plot_tools.plot_entropy_vs_bins(
        analysis_data["food_supply_variability_kcal_cap_day"].to_numpy(dtype=float),
        feature="food_supply_variability_kcal_cap_day",
        output_directory=str(FIGURES_DIRECTORY),
    )
    category_plot = _plot_category_frequencies(
        analysis_data["Area"], category_name="Area (top 15)", top_n=15
    )
    _plot_two_features(analysis_data, "energy_adequacy_percent", TARGET)
    plot_tools.plot_statistics_table(statistics, str(FIGURES_DIRECTORY))
    print(f"\nCategorical frequency figure: {category_plot.name}")
    print("All Task 12 distribution figures were saved.")

    print("\n" + "=" * 70)
    print("KL DIVERGENCE")
    print("=" * 70)

    kl_rows = []
    for feature in (*PREDICTORS, TARGET):
        values = analysis_data[feature].to_numpy(dtype=float)
        p, _ = _probabilities(values)
        q = _gaussian_probabilities(values)
        kl_rows.append({
            "feature": feature,
            "KL_P_to_Gaussian": compute_kl_divergence(p, q),
            "KL_Gaussian_to_P": compute_kl_divergence(q, p),
        })
    kl_table = pd.DataFrame(kl_rows)
    print("\nFeature versus Gaussian:")
    print(kl_table.to_string(index=False))

    pair_a = "energy_adequacy_percent"
    pair_b = "protein_supply_g_cap_day"
    pair_kl = _standardized_kl(analysis_data, pair_a, pair_b)
    print("\nStandardised feature comparison:")
    print(f"  {pair_a} || {pair_b} = {pair_kl[0]:.6f} bits")
    print(f"  {pair_b} || {pair_a} = {pair_kl[1]:.6f} bits")

    print("\n" + "=" * 70)
    print("CLUSTERING")
    print("=" * 70)

    cluster_dataset = _make_cluster_dataset(analysis_data)
    prepared = prepare_clustering_data(cluster_dataset, features=PREDICTORS)
    _, kmeans_labels, kmeans_evaluation = fit_kmeans_clustering(
        prepared["values"], n_clusters=3, random_seed=RANDOM_SEED, n_initializations=10
    )
    _, gmm_labels, gmm_evaluation = fit_gmm_clustering(
        prepared["values"], n_components=3, random_seed=RANDOM_SEED
    )

    print("\nK-means evaluation:")
    print(kmeans_evaluation)
    print("\nGMM evaluation:")
    print(gmm_evaluation)

    projection = project_to_two_dimensions(prepared["values"])
    _plot_clustering_comparison(
        projection, kmeans_labels, gmm_labels,
        title="Task 12 clustering comparison: FAOSTAT observations",
    )

    for name, labels in (("K-means", kmeans_labels), ("GMM", gmm_labels)):
        target_means = []
        for cluster in sorted(np.unique(labels)):
            target_values = analysis_data.loc[labels == cluster, TARGET]
            target_means.append((int(cluster), len(target_values), float(target_values.mean())))
        print(f"\n{name} target means (cluster, n, mean %):")
        print(target_means)

    area_codes = (
        data[["Area", "Area Code (M49)"]]
        .drop_duplicates("Area")
        .set_index("Area")
    )
    bad_codes = analysis_data["Area"].map(area_codes["Area Code (M49)"]).to_numpy(dtype=float)
    bad_matrix = ((bad_codes - bad_codes.mean()) / bad_codes.std()).reshape(-1, 1)
    _, bad_labels, bad_evaluation = fit_kmeans_clustering(
        bad_matrix, n_clusters=3, random_seed=RANDOM_SEED, n_initializations=10
    )
    print("\nDeliberately bad clustering using only Area Code:")
    print(bad_evaluation)
    _plot_nonsense_clusters(
        bad_codes,
        analysis_data[TARGET],
        bad_labels,
        x_label="FAOSTAT Area Code (M49) — identifier only",
        y_label="Undernourishment (%)",
        title="Deliberately bad clustering using an identifier",
    )

    print("\n" + "=" * 70)
    print("LINEAR REGRESSION")
    print("=" * 70)

    X = analysis_data[list(PREDICTORS)].to_numpy(dtype=float)
    y = analysis_data[TARGET].to_numpy(dtype=float)
    split = _split_rows(X, y)

    sensible = fit_linear_regression(split["X_train"], split["y_train"])
    wrong = fit_linear_regression(split["X_train"][:, [0]], split["y_train"])
    baseline = fit_mean_baseline(split["y_train"])

    validation_predictions = {
        "baseline": predict_regression(baseline, split["X_validation"]),
        "sensible": predict_regression(sensible, split["X_validation"]),
        "wrong": predict_regression(wrong, split["X_validation"][:, [0]]),
    }
    validation_evaluations = {
        name: evaluate_regression(split["y_validation"], prediction)
        for name, prediction in validation_predictions.items()
    }
    print("\nValidation performance:")
    _print_regression_table(validation_evaluations)

    ranking = _rank_regression_models(validation_evaluations, metric="rmse")
    print("\nValidation ranking:")
    for item in ranking:
        print(f"  {item['rank']}. {item['model']} — RMSE = {item['rmse']:.4f}")

    test_predictions = {
        "baseline": predict_regression(baseline, split["X_test"]),
        "sensible": predict_regression(sensible, split["X_test"]),
        "wrong": predict_regression(wrong, split["X_test"][:, [0]]),
    }
    test_evaluations = {
        name: evaluate_regression(split["y_test"], prediction)
        for name, prediction in test_predictions.items()
    }
    print("\nFinal test performance:")
    _print_regression_table(test_evaluations)

    coefficients, intercept = _regression_coefficients(sensible, PREDICTORS)
    print("\nSensible OLS coefficients:")
    for feature, coefficient in coefficients.items():
        print(f"  {feature}: {coefficient:.6f}")
    print(f"  intercept: {intercept:.6f}")

    _plot_observed_vs_predicted(split["y_test"], test_predictions, "undernourishment (%)")
    _plot_residuals(split["y_test"], test_predictions, "undernourishment (%)")

    wrong_residuals = split["y_test"] - test_predictions["wrong"]
    omitted_values = analysis_data.iloc[split["test_indices"]]["energy_requirement_kcal_cap_day"].to_numpy()
    residual_correlation = float(np.corrcoef(omitted_values, wrong_residuals)[0, 1])
    _plot_wrong_residuals(
        omitted_values,
        wrong_residuals,
        feature_label="Omitted energy requirement (kcal/cap/day)",
        target_label="undernourishment (%)",
    )
    print(f"\nWrong-model residual correlation with omitted energy requirement: {residual_correlation:.4f}")
    print("The wrong model is useful for simple screening when only energy-supply adequacy is available,")
    print("but the omitted-variable residual pattern shows that important information is missing.")

    return {
        "raw_integrity": raw_report,
        "cleaning": cleaning,
        "analysis_data": analysis_data,
        "statistics": statistics,
        "kl": kl_table,
        "clustering": {
            "kmeans": kmeans_evaluation,
            "gmm": gmm_evaluation,
            "bad": bad_evaluation,
        },
        "regression": {
            "validation": validation_evaluations,
            "ranking": ranking,
            "test": test_evaluations,
        },
    }


def main():
    """Run using the cached FAOSTAT CSV."""
    data = download_faostat_data()
    result = run_task12(data)
    print("All real-data figures were saved in project_2_b/figures/.")
    return result


if __name__ == "__main__":
    main()

