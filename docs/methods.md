# Methods and limitations

## Win rates and profiles

The raw win rate gives every recorded player appearance equal weight. The player-balanced rate first calculates each person's win rate within the selected body/hitbox and filters, then averages those rates equally across people. A player with 100 appearances has the same total weight as a player with one. The people contributing to each body can still differ in skill and playstyle.

For outcome `y_i`, let `n_p` be the number of appearances by player `p` in the selected group. Each appearance has weight `w_i = 1 / n_p`. The balanced estimate is `mu = sum(w_i * y_i) / sum(w_i)`, equivalent to the mean of individual player win rates. Its influence contribution is `u_i = w_i * (y_i - mu) / sum(w_i)`.

Hitbox rates pool all bodies in that hitbox before weighting each player; they are not an average of body averages. Style means weight measured appearances equally. Missing values are omitted separately for each statistic, and each statistic reports its measured support.

Exploratory profiles require 100 appearances, 30 players and 100 matches. Large-sample flags require 500 appearances, 100 players and 200 matches. The flag describes sample counts; it does not establish representative sampling or adequate power for every question.

The `mean_summary()` and `cluster_variance()` functions use a two-way cluster sandwich estimate: aggregate influence by player, plus by match, minus the player/match intersection. Finite-cluster corrections are included. This relies on unique player/match rows. Individual mean ranges use 1.96 standard errors; measured-statistic ranges require 100 observations, 30 players and 30 matches. Nonpositive or nonfinite variance estimates are withheld. These are normal approximations, not bounded binomial intervals, and can extend beyond a statistic's physical range.

With `P` player clusters, `M` match clusters and `N` appearances, the implemented variance is `P/(P-1) * sum_p(sum_i_in_p u_i)^2 + M/(M-1) * sum_m(sum_i_in_m u_i)^2 - N/(N-1) * sum_i u_i^2`. For an unweighted mean, `u_i = (y_i - mu)/N`. Clustered uncertainty does not cover arbitrary dependence across otherwise unrelated players/matches, upload selection or unmeasured confounding.

## Same players on different hitboxes

`paired_rows()` keeps player/quarter cells with at least three appearances on each of the two hitboxes. Quarters are fixed UTC calendar quarters: January–March, April–June, July–September and October–December. They are not rolling 90-day windows. A quarter can be partially represented in the collected sample.

By default, appearances can cross modes and queues. The same-queue variant also matches on the exact playlist. Each person has equal total weight on either side, divided equally among their qualifying cells and then among appearances within each side of a cell. Style comparisons repeat the eligibility check on measured values.

For a paired sample of `P` people, `C_p` qualifying cells for person `p`, and `n_pch` appearances in cell `c` on hitbox `h`, an appearance's weight is `1 / (P * C_p * n_pch)`. The paired difference is the weighted rate on hitbox A minus the weighted rate on hitbox B. Positive A-side and negative B-side influence contributions are combined before the two-way cluster calculation.

Differences are **first hitbox minus second hitbox**, in percentage points for win rate. Reported ranges account for shared players and matches and use a Bonferroni correction for the 15 possible pairs. Supported ranges require 30 players, 100 matches and 100 appearances on each side, with an estimable variance. Large samples require 100 players, 200 matches and 500 appearances on each side. The correction applies within a view, not across every overlapping dashboard filter.

Matching helps reduce stable differences between people. It does not hold opponents, parties, improving skill, mode mix, match state or reasons for switching constant. Different pairs can contain different people; their differences cannot be chained into a universal ranking. Familiarity is earlier collected use, not lifetime use.

## Win classifiers

`match_table()` constructs one row per match. Each input is the blue team's average minus the orange team's average for that statistic, and the target is whether blue won. Opposing sides never become separate observations split between training and evaluation.

`temporal_split()` sorts distinct timestamps: the earliest approximately 60% train, the next 20% validate, and the last 20% test. This partitions timestamps rather than guaranteeing those proportions of matches. Equal timestamps remain together. Every period needs at least 200 matches and both outcomes. Repeated players can occur across periods; a separate test score is calculated for matches with no training players when at least 200 such matches and both outcomes exist.

Usable statistics need at least 95% measured training appearances and nonzero training variation. Missing-value medians and logistic-regression scaling are fitted only on training rows. Variable 1v1, 3v3 and casual indicators provide context; they are excluded from the ranking of gameplay-statistic importance.

| Model | Configuration |
| --- | --- |
| Logistic regression | Median imputation, standardization, `C=0.3`, `max_iter=500` |
| Random forest | Median imputation, 100 trees, depth 10, minimum leaf size 30 |
| Histogram gradient boosting | Median imputation, 120 iterations, 15 leaves, L2 penalty 3, early stopping disabled |

The seed is `20260908`. The model with the highest validation AUC is selected, while all three test results are retained. Models are fitted separately for each requested mode/queue combination. A second set of fits excludes shots and saves.

Score, goals, assists, shooting percentage and direct outcome fields are excluded from predictors. However, the retained statistics are still measured during the same completed match. Some reflect consequences of winning, score state or correlated scoring information. This is retrospective classification, not a model for predicting the winner before kickoff or estimating causal improvement.

## Reading model results

- **AUC:** how well wins rank above losses. A value of 0.5 is chance ranking; this is not accuracy.
- **Accuracy:** fraction classified correctly using a probability cutoff of 0.5.
- **Brier score:** mean squared probability error; lower is better.
- **Calibration:** predicted probabilities compared with observed win fractions in up to eight quantile bins.
- **Permutation importance:** test AUC decrease after shuffling one feature, using up to 3,000 test matches and three repeats. Standard deviations describe variation across shuffles, not confidence intervals. Correlated inputs can mask or divide importance. Values can be negative.
- **Logistic odds ratios:** exponentiated coefficients for standardized inputs, conditional on the other included features; these are not percentage-point changes in win rate.
- **Statistic curves:** descriptive win rates in up to eight bins of held-out team-stat differences. They are not adjusted intervention effects.

A constant training-outcome baseline is reported. Test performance can still depend on recurring players, uploaded-match selection and differences between time periods. Trying many analyses on the same test period can also turn it into an exploratory resource rather than a fresh confirmation set.

## Findings and scope

Relative differences use 100 × (body mean − group mean) / |group mean|. The reference group includes the body, and a small baseline can yield a large percentage. The backend findings function selects unusual descriptive patterns using sample thresholds and caps on repeated bodies/statistics; the website applies additional presentation rules. These highlights are exploratory and are not independent significance tests.

All-mode and narrower views overlap and must not be added together. Rank is unavailable, uploads are not random, and neither player balancing nor a large dataset fixes that selection. Model evaluation and uncertainty checks do not establish that a car or behavior causes more wins.
