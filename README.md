# Project 2 — Data Generation and Distribution Analysis 

## Project overview

This project develops a reusable Python workflow for generating, checking, analysing, visualising, and later modelling numerical data.

For Part A, a synthetic agriculture dataset is created representing plant-level observations from a French maize field. The dataset was designed deliberately so that the statistical properties and relationships between variables are known before analysis. This makes it possible to test whether the implemented statistical tools correctly identify properties that were intentionally built into the data.

The project is divided into separate modules:

* `data_tools.py` — generates the dataset and checks its integrity.
* `stats_tools.py` — contains implementations of the statistical calculations.
* `plot_tools.py` — creates and saves visualisations.
* `code.py` — runs the complete workflow for the selected dataset.
* `test_data_tools.py` — tests the data generation and integrity checks.
* `test_stats_tools.py` — tests the statistical implementations.

The tools are designed to receive data as input rather than depending on the maize dataset directly. The dataset-specific decisions are kept in `code.py`.

---

## Task 2 — Synthetic dataset

### Dataset description

The synthetic dataset contains 600 plant-level observations and six columns:

| Feature                   | Type        | Description                                  |
| ------------------------- | ----------- | -------------------------------------------- |
| `plant_height_cm`         | Numerical   | Height of the maize plant in centimetres     |
| `leaf_damage_percent`     | Numerical   | Percentage of leaf damage                    |
| `moisture_sensor_percent` | Numerical   | Soil/field moisture sensor measurement       |
| `ear_length_cm`           | Numerical   | Length of the maize ear in centimetres       |
| `grain_mass_per_ear_g`    | Numerical   | Grain mass per maize ear in grams            |
| `field_zone`              | Categorical | Location of the observation within the field |

The dataset contains five numerical features and one categorical feature, satisfying the requirement for at least five numerical variables and a meaningful categorical variable.

A fixed NumPy random seed (`42`) is used so that the same dataset can be reproduced.

### Deliberately generated properties

The data was generated so that several properties are known in advance.

`plant_height_cm` and `ear_length_cm` are approximately symmetric variables. For example, the measured skewness of `ear_length_cm` is approximately `-0.0215`, which is very close to zero.

`leaf_damage_percent` is deliberately right-skewed using a log-normal distribution. Its measured skewness is approximately `3.1657`, showing a strong positive tail.

`moisture_sensor_percent` is designed to be almost constant around 18%. Its standard deviation is approximately `0.0050`, which is very small relative to its mean of approximately 18%.

The dataset also contains a deliberate relationship between `ear_length_cm` and `grain_mass_per_ear_g`. Grain mass is generated partly from ear length, so longer ears tend to have greater grain mass. The measured Pearson correlation between these variables is approximately `0.857`.

The categorical variable `field_zone` represents different areas of the maize field. Different zones are given small effects on some numerical measurements, making the category meaningful rather than an arbitrary label.

### Reproducibility

The dataset is generated using NumPy's random number generator with a fixed seed:

```python
rng = np.random.default_rng(42)
```

This means that the generated observations are reproducible.

---

## Task 3 — Check the data integrity

A general `check_data_integrity()` function was implemented in `data_tools.py`.

The function receives a pandas DataFrame and optional expectations such as:

* expected columns
* expected shape
* expected data types
* expected numerical ranges

It checks for:

* missing columns
* unexpected shape
* missing values
* duplicate rows
* incorrect data types
* values outside specified ranges

The function returns a report dictionary and prints a clear pass/fail message.

The integrity function is intentionally general. It does not contain maize-specific assumptions unless those assumptions are supplied by the calling code.

### Deliberate failure tests

The test suite creates deliberately corrupted versions of the generated dataset.

The following cases are tested:

1. A valid dataset should pass.
2. A NaN value should be detected.
3. A duplicated row should be detected.
4. An impossible numerical value should be detected.
5. A required column that has been removed should be detected.

This verifies that the integrity checker can identify common data-quality problems rather than only confirming that the original dataset is valid.

---

## Task 4 — Describe the distributions

The statistical calculations are implemented in `stats_tools.py` without using `scipy.stats.skew` or `scipy.stats.entropy` in the actual implementations.

The following calculations are performed for every numerical feature:

* histogram-based probability density function (PDF)
* cumulative distribution function (CDF)
* mean
* population standard deviation
* skewness
* Shannon entropy

### Histogram bin choice

A fixed value of **25 bins** is used for the histogram-based PDF, CDF, and entropy calculations.

Using the same number of bins makes the distributions comparable between features and also provides a consistent basis for later entropy and KL-divergence calculations.

The bin count is explicitly defined in `code.py` as:

```python
NUMBER_OF_BINS = 25
```

### Probability density function

The PDF is estimated from a histogram. The histogram counts are converted into density values by accounting for both the number of observations and the width of each bin.

The important property is that the **area under the PDF is approximately 1**, rather than simply requiring the sum of the histogram heights to equal 1.

### Cumulative distribution function

The CDF is calculated from the histogram probabilities by taking their cumulative sum. The resulting values are non-decreasing and finish at approximately 1.

### Mean and standard deviation

The mean describes the central value of each numerical feature. The population standard deviation measures the typical spread around that mean.

For example, the moisture sensor feature has a very small standard deviation, demonstrating that its observations are concentrated very closely around 18%.

### Skewness

Skewness measures asymmetry in a distribution.

The main examples in this dataset are:

* `ear_length_cm`: approximately symmetric, with skewness close to zero.
* `leaf_damage_percent`: strongly right-skewed, with skewness approximately `3.17`.

### Shannon entropy

Shannon entropy is calculated from the histogram probabilities using base-2 logarithms, so the result is expressed in **bits**.

The entropy calculation depends on the chosen number of histogram bins. Therefore, the same binning approach is used consistently when comparing features.

### Visualisations

The following figures are generated and saved in the `figures/` directory:

* histogram for each numerical feature
* PDF for each numerical feature
* CDF for each numerical feature
* frequency bar chart for `field_zone`
* 2D scatter plot of `ear_length_cm` versus `grain_mass_per_ear_g`

The ear-length/grain-mass scatter plot was selected because these variables were deliberately generated with a positive relationship. The measured correlation is approximately `0.857`, making the relationship visible in a 2D plot.

All numerical axes include appropriate units where applicable.








## Task 5 — What is entropy good for?

1) In summary statistics, we got following entropy_bits and standard deviation:

| feature                   | std         | entropy bits  |
| ------------------------- | ----------- | ------------- |
| `plant_height_cm`         | 18.3428     | 3.8728        |
| `leaf_damage_percent`     | 3.2966      | 2.7251        |
| `moisture_sensor_percent` | 0.0050      | 4.0754        |
| `ear_length_cm`           | 2.0191      | 4.1169        |
| `grain_mass_per_ear_g`    | 20.1209     | 4.1015        |


If we compare the almost constant feature (`moisture_sensor_percent`) and the widest feature (`grain_mass_per_ear_g`) we see that despite std of `moisture_sensor_percent` (0.0050) is so different compared to `grain_mass_per_ear_g` (20.1209), their entropy bits are very close (4.0754 and 4.1015). This happens because entropy calculation is based on the probabilities of observations falling into histogram bins, so entropy and std measure different things:

- Entropy measures uncertainty in which bin an observation will fall into, given the chosen binning.

- Standard deviation measures physical numerical spread. 

If our goal is to learn something that varies from plant to plant, then the wide measurement is better than almost constant measurement -> Grain Mass measurement is more informative. The interesting thing is that entropy alone doesn't clearly distinguish them, since the uncertainty is very similar for both features.

2) What happens if a measurement has entropy = 0? It basically means that there is no uncertainty in which bin an observation will fall into, so every measurement will fall into the same bin.
Therefore, before measuring, you already know exactly where the next measurement will fall. So, no additional measurements are required, since you already know the result.

3) Suppose that we have a histogram with 8 bins, and the next observation could be in any of them with equal probability. 

The Shannon entropy is calculated as:

$$
H = -\sum_{i=1}^{n} p_i \log_2(p_i)
$$

So,

$$
\log_2(8) = 3
$$

Gives us 3 yes/no questions, and it means the uncertainty is roughly equivalent to needing 3 binary decisions on average to identify the next bin, depending on the probability distribution.

For 25 bins the theoretical maximum becomes,

$$
\log_2(25) = 4.64
$$

And if we compare the theoretical maximum entropy with one of our features (f.ex. `ear_length_cm` = 4.1169), we see that 4.1169 < 4.64, so that means our distribution has less uncertainty than the maximum possible uncertainty for 25 bins (the maximum entropy/uncertainty can be achieved only when all 25 bins are equally likely). So, the number we computed makes sense.

4) I we run "code.py", one of the parts we get in terminal is following:

Feature: ear_length_cm
Number of bins: 25

| Transformation | Standard deviation | Entropy |
|---|---:|---:|
| **Original** | 2.0191 | 4.1169 bits |
| **After multiplying by 1000** | 2019.1411 | 4.1169 bits |
| **After adding 50** | 2.0191 | 4.1169 bits |  

Thus, we observe that std is sensitive to scale (it changes dramatically from 2.0191 to 2019.1411), but if we simply move everything (x -> x + 50), then std doesn't change.
So standard deviation tells us about physical/numerical spread.

While entropy, on the other hand, doesn't care about the absolute measurement scale in the same way. It also doesn't care about simply moving the whole distribution. That is why entropy remains the same in all experiments. 

5) To find out which 2 features are reasonable to keep, we can create a correlation matrix in code.py:

| Variable                    | plant_height_cm | leaf_damage_percent | moisture_sensor_percent | ear_length_cm | grain_mass_per_ear_g |
| --------------------------- | --------------: | ------------------: | ----------------------: | ------------: | -------------------: |
| **plant_height_cm**         |           1.000 |              -0.019 |                   0.027 |         0.064 |                0.154 |
| **leaf_damage_percent**     |          -0.019 |               1.000 |                  -0.054 |        -0.020 |               -0.180 |
| **moisture_sensor_percent** |           0.027 |              -0.054 |                   1.000 |        -0.036 |               -0.016 |
| **ear_length_cm**           |           0.064 |              -0.020 |                  -0.036 |         1.000 |                0.857 |
| **grain_mass_per_ear_g**    |           0.154 |              -0.180 |                  -0.016 |         0.857 |                1.000 |

**Entropy of each feature:**

plant_height_cm: 3.8728 bits  
leaf_damage_percent: 2.7251 bits  
moisture_sensor_percent: 4.0754 bits  
ear_length_cm: 4.1169 bits  
grain_mass_per_ear_g: 4.1015 bits

We would keep `ear_length_cm` and `leaf_damage_percent`. Ear length has high entropy and a strong relationship with grain mass, while leaf damage provides different information about plant condition. I would not choose the moisture sensor because its standard deviation is extremely small, indicating that it is almost constant. I also would not automatically choose grain mass together with ear length, because their correlation is high (r ≈ 0.857), meaning that the two features contain overlapping information. Therefore, entropy alone is not sufficient for feature selection; correlation and the meaning of the features also need to be considered.

## Task 6 Kullback-Leibler divergence

If your feature looks very Gaussian, its KL divergence from the matching Gaussian should be relatively small. If it looks very different from a Gaussian, the KL divergence should be larger. 

If we run updated code.py, we get this table:

KL divergence:
               | Feature | KL(P‖Q) (bits) | KL(Q‖P) (bits) |
|---|---:|---:|
| **plant_height_cm** | 0.034685 | ∞ |
| **leaf_damage_percent** | 0.551120 | ∞ |
| **moisture_sensor_percent** | 0.078830 | 0.072575 |
| **ear_length_cm** | 0.020503 | 0.021184 |
| **grain_mass_per_ear_g** | 0.053493 | ∞ |

Therefore, your smallest values are:
- `ear_length_cm`: 0.020503
- `plant_height_cm`: 0.034685
- `grain_mass_per_ear_g`: 0.053493

These distributions are quite similar to their matching Gaussian distributions. This makes sense because their skewness values are -0.0215, -0.1163, and -0.1277. These values are close to zero, which means that the three features are fairly symmetric. However, being symmetric does not automatically mean that a distribution is Gaussian. It only tells us that the distribution is not strongly skewed.

Now, let's look at `leaf_damage_percent`. This feature has a much larger KL divergence than the other finite values. We also know that its skewness is 3.1657, which means that the distribution is strongly skewed and not very symmetric. This suggests that it does not have the typical bell shape of a Gaussian distribution. Therefore, the large KL divergence makes sense because the `leaf_damage_percent` distribution is quite different from a Gaussian distribution.

The interesting thing happens when we handle zeros q_i = 0 and p_i > 0. We have chosen to use Kullback–Leibler (KL) divergence formula:

$$
D_{KL}(P\parallel Q) = \sum_i p_i \log_2\left(\frac{p_i}{q_i}\right)
$$

So the KL divergence is measured in bits.
When $p_i = 0$, that term contributes 0. If $p_i > 0$ and $q_i = 0$,
the KL divergence is infinite.

## Task 7 Is KL a distance?

KL divergence satisfies following properties:

- Non-negative (A distance should always be zero or positive)

- Identity (Identical things have distance zero)

KL divergence doesn't satisfy following properties:

- Symmetry (Going in either direction should give the same distance)

- Triangle inequality (Going through an intermediate point should not be shorter)


So, KL is a divergence, as it is assymetric, and KL is not a metric.

KL is not the right choice if our goal is to find a distance between two distributions, but if we just want to know how different is our observed distribution \(P\) from a Gaussian reference distribution \(Q\).


To find a genuine distance between distributions we can use following distance/proximity techniques:

- Euclidean distance

$$
d_{ij}^{(E)}=\left[
\sum_{k=1}^{N}
(x_{ik}-x_{jk})^2
\right]^{\frac{1}{2}}
$$

- Manhattan distance

- Minkowski distance

- Mahalanobis distance
