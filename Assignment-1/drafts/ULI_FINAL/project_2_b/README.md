# MOD550 Assignment 1 — Task 11 and Task 12

## Task 11 — Find the data repository

### Repository

**FAOSTAT — Suite of Food Security Indicators**

Name: Food and Agriculture Organization of the United Nations (FAO).

Dataset page: https://www.fao.org/faostat/en/#data/FS

License: All datasets disseminated through FAO corporate statistical databases are licensed under the Creative Commons Attribution-4.0 International licence (CC BY 4.0).

API endpoint: https://faostatservices.fao.org/api/v1/ping

The downloaded extract contains:

28928 observations and 6 numerical predictors with 1 numerical target:

Target:

`undernourishment_percent` — Prevalence of undernourishment (%), 3-year average.

Predictors:

1. `energy_adequacy_percent` — Average dietary energy supply adequacy (%)
2. `protein_supply_g_cap_day` — Average protein supply (g/cap/day)
3. `animal_protein_g_cap_day` — Average supply of protein of animal origin (g/cap/day)
4. `cereal_share_percent` — Share of dietary energy supply derived from cereals, roots and tubers (%)
5. `food_supply_variability_kcal_cap_day` — Per capita food supply variability (kcal/cap/day)
6. `energy_requirement_kcal_cap_day` — Average dietary energy requirement (kcal/cap/day)


### Three representative features

| Feature | Unit | Type used in analysis | Observed range in complete analysis data | Meaning |
|---|---|---|---:|---|
| Average dietary energy supply adequacy | % | numerical | 73–167 | Adequacy of available dietary energy supply relative to requirement |
| Average protein supply | g/cap/day | numerical | 28.3–152.4 | Average daily protein supply per person |
| Prevalence of undernourishment | % | numerical target | 2.5–67.8 | Estimated proportion of the population whose habitual food consumption is insufficient |

### Missing and invalid values

The raw FAOSTAT data use missing-value flags. In the supplied extract, missing observations are represented by blank/NaN values in `Value` and are associated with FAOSTAT missing-value flags such as `O` (Missing value). The raw data also contain values written as `<2.5` for the undernourishment indicator.

For the analysis, `<2.5` is represented by the lower bound **2.5**. This keeps more country-period observations, but it is only a lower-bound representation and not an exact measured value.

### Metadata quality assessment

The metadata quality is good for this project: the source, indicator names, units, dimensions, licensing information and data flags are documented.

We consider the data suitable for the requested statistical, clustering and regression exercises, with the main limitations being missing values, lower-bound values and the fact that the observations are observational rather than a controlled experiment.

---

## Task 12 — Download it and re-run everything

### Integrity check

The raw-data integrity check verifies the expected columns, required identifier fields, duplicates and missing values. The check is deliberately limited to structural integrity: it does not claim that the scientific relationships or the eventual models are correct.

The raw extract contains missing observations that the synthetic maize data did not contain. In particular, FAOSTAT missing-value flags are present and the `Value` field contains blank observations. The extract also contains `<2.5` lower-bound values, which were not present in the synthetic data.

Cleaning gives:

| Quantity | Result |
|---|---:|
| `<2.5` lower-bound values | 1,220 |

### Distribution analysis

The same `stats_tools.py` functions from `project_2_a` are reused without copying them into `project_2_b`.

| Feature | Mean | Std | Skewness | Entropy (bits) |
|---|---:|---:|---:|---:|
| Energy supply adequacy | 120.4289 | 14.6930 | -0.0296 | 3.9982 |
| Protein supply | 82.8010 | 22.4898 | 0.1424 | 4.1504 |
| Animal protein supply | 38.1348 | 21.7942 | 0.4141 | 4.1861 |
| Cereal/root/tuber share | 46.8224 | 14.3092 | 0.3183 | 4.4155 |
| Food supply variability | 36.8977 | 26.0408 | 2.7414 | 2.8237 |
| Energy requirement | 2359.4490 | 129.9207 | -0.1406 | 4.3809 |
| Undernourishment | 10.5131 | 10.1941 | 1.5904 | 2.9515 |

The `food_supply_variability_kcal_cap_day` feature is strongly right-skewed, with skewness about 2.74. The target `undernourishment_percent` is also right-skewed. Energy-supply adequacy is close to symmetric in this representation, with skewness near zero.

The entropy values depend on the chosen histogram bin count, so entropy is used as a description of the histogram distribution rather than as a standalone measure of predictor importance.

### What surprised us compared with the synthetic data

The synthetic maize dataset was deliberately clean: its distributions, correlations, and target relationship were constructed by design. The FAOSTAT data are observational and contain missing values and lower-bound values.

The real data also show that a bad model or bad clustering choice can sometimes obtain a respectable numerical score. The strongest example is the identifier-based clustering: its silhouette score (0.6005) is higher than the sensible K-means score (0.3047), even though `Area Code` has no scientific meaning as a measurement. This is a strong example of why model usefulness cannot be judged by a single internal score.

### What cannot be checked on real data

For the synthetic data, the generating relationship and the intended groups were known because we created them. For FAOSTAT, the true generating relationship is unknown and there are no known “correct” clustering labels for the stated question.

Therefore, the real-data workflow can assess numerical stability, predictive performance on held-out observations, residual structure and internal clustering scores, but it cannot prove a causal relationship or prove that a discovered cluster corresponds to a true real-world group.
