# Rocket Data Science — statistical methods

Code and explanations behind [Rocket Data Science](https://prejump.net/), a personal project by InsomniArch exploring car bodies, hitboxes and playing styles in Rocket League.

This repository is for reviewing the analysis: how rates are weighted, how comparisons are constructed, how the models are evaluated, and where the conclusions are limited. It contains the statistical calculations used by **version 1.3**, whose reference snapshot includes **2,195,201 matches**, captured on **September 26, 2026 (UTC)**.

## Start here

Read [Methods and limitations](docs/methods.md) for the reasoning, [Data assumptions](docs/data-assumptions.md) for the expected inputs, and the [published model evaluation](reference/model-evaluation.md) for the observed diagnostics.

| Question | Code to inspect |
| --- | --- |
| How does each player get equal weight? | [`balanced_result()`](rocket_data_science/research.py) |
| How are win rates and playing styles summarized? | [`profiles()`](rocket_data_science/profiles.py) and [`mean_summary()`](rocket_data_science/snapshot.py) |
| How are repeated players and shared matches reflected in uncertainty? | [`cluster_variance()`](rocket_data_science/research.py) and [`mean_summary()`](rocket_data_science/snapshot.py) |
| What happens when the same person uses different hitboxes? | [`paired_rows()` and `pair_result()`](rocket_data_science/matched_hitboxes.py) |
| What do the models see, and how are they tested? | [`match_table()`, `temporal_split()`, `training_features()` and `fit_models()`](rocket_data_science/research.py) |
| Which matches belong to each view? | [`group_modes()`](rocket_data_science/cohorts.py) and [`choose()`](rocket_data_science/profiles.py) |

The classifiers are logistic regression, random forest and gradient boosting. They use statistics from completed matches: they are **not pre-match forecasts**, and feature importance is not a measurement of causal benefit.

## What this release lets you check

The statistical function bodies were extracted from the v1.3 implementation without changing their calculations. Imports and file organization were adapted so the calculations can be inspected separately. `provenance.json` records syntax fingerprints for 26 extracted functions/constants, and a test checks them. These fingerprints record this release; they are not an independent attestation of correctness.

Tests cover equal-player weighting, chronological boundaries, training-only feature selection, same-player quarter matching, and agreement of uncertainty calculations with independent statsmodels calculations. Fabricated test records also exercise all three classifiers. These tests support review of the implementation; they do not establish that the observations are unbiased or that the conclusions are causal.

`car_rankings()` and `playing_styles()` in `research.py` are earlier supporting analyses retained with their tests. The v1.3 dashboard uses `profiles()` for its body and hitbox summaries.

## What is not included

The replay database, player records, fitted model files, collection system and website deployment are not included. This is a methods reference, not a complete application. **The code alone cannot reproduce the exact published numbers without the same input data**, or establish that every input record was accurate. Dataset redistribution permission is separate from access to this code.

Rank is unavailable. Ballchasing uploads are a convenience sample, and upload habits, skill, parties, opponents and reasons for switching cars can differ across groups. Player balancing does not adjust for skill. Larger samples reduce some sampling uncertainty but do not remove those biases.

## Optional calculation checks

Reading the code does not require installing anything. Reviewers who want to run the tests can use an isolated Python environment:

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The reference environment uses Python 3.14. `requirements-lock.txt` records the scientific dependency versions used for that environment. Test data is fabricated in memory and contains no real player records. It is not evidence about Rocket League.

Questions about the calculations and reproducible bug reports are welcome through GitHub issues. Please use a small fabricated example instead of posting player records or private data.

## Reuse

This code is public for inspection. No reuse license is provided at this time.
