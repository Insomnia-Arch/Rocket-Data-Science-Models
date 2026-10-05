# Data assumptions

The statistical functions work on pandas tables prepared from normalized replay statistics. The operational data pipeline is not part of this repository. These assumptions matter when assessing whether the calculations apply to another dataset.

## Appearance table

Each row represents one complete player appearance in a deduplicated match. Both complete teams are present, outcomes are untied and consistent within teams, and each player/match pair is unique. Partial appearances, bots, missing identities and replays under 60 seconds are excluded upstream. Different uploads of the same match must be deduplicated by match identity, not merely by upload ID.

| Column | Meaning |
| --- | --- |
| `replay_id` | Stable match identity |
| `player_id` | Stable player identity across matches; a consistent pseudonym suffices |
| `team` | `blue` or `orange` |
| `car`, `hitbox` | Resolved equipment names; known hitboxes are Octane, Dominus, Breakout, Hybrid, Merc and Plank |
| `playlist` | Original ranked or casual playlist identifier |
| `date` | UTC match timestamp normalized to second resolution |
| `month` | UTC month in `YYYY-MM` format, derived from `date` |
| `duration` | Match duration in seconds |
| `won` | 1 if the player's team won, otherwise 0 |
| `win_rate` | 100 × `won`, for appearance-weighted summaries |
| `analysis_group` | Derived by `group_modes()`: `duels` or `team` |
| `prior_hitbox` | Derived by `recorded_familiarity()`: earlier collected appearances on that hitbox and original playlist |

`PLAYLIST_GROUPS` lists eligible 1v1, 2v2 and 3v3 playlists, including the historical ranked solo-standard playlist. Other game modes are excluded. Ranked/casual filters retain exact original playlist identities. Unknown hitboxes are excluded from body/hitbox profiles and matched hitbox comparisons. The group-average style baseline uses all eligible appearances, including unresolved equipment.

`recorded_familiarity()` runs on the full captured frame before dashboard filtering. Appearances with identical timestamps do not count one another as earlier experience. Same-player comparisons compute fixed UTC calendar quarters from `month`.

## Measured statistics

Missing measured values remain missing rather than becoming zero. The frame supplies all profile columns, with missing values where unavailable. Model feature selection checks availability using training data only.

- `score`, `goals`, `assists`, `shots`, `saves`, `demos_inflicted` and `demos_taken` are per-appearance totals.
- `average_speed` is recorded in Unreal units per second. Distances retain replay units.
- `boost_per_minute` is reported boost consumption per minute; `average_boost` is mean boost held.
- Fields ending in `_percent`, and `shooting_percentage`, use the 0–100 scale. Shooting percentage is a supplied per-appearance measurement, not a pooled ratio reconstructed across players.
- `score_share` is 100 × player score / team total score; it is missing if any teammate's score is missing or the total is zero.
- `shots_per5`, `saves_per5`, `demos_per5`, `boost_stolen_per5`, `boost_overfill_per5`, `boost_supersonic_per5` and `powerslides_per5` are the respective counts/amounts × 300 / match duration.

The source count/amount fields for those rates are `shots`, `saves`, `demos_inflicted`, `boost_stolen`, `boost_overfill`, `boost_supersonic` and `powerslides`. Rates are calculated per appearance before averaging.

The full classifier feature list is the `FEATURES` dictionary in `research.py`. It includes positioning, speed, boost, demos, shots and saves. The playing-style variant additionally excludes shots and saves. Car body and hitbox are not classifier inputs; the classifiers study match-statistic associations separately from the body/hitbox comparisons.

## Review boundary

The included tests check calculations on fabricated records. They cannot check the completeness of the privately held dataset, the correctness of every equipment mapping, historical rank, or representativeness of uploaded replays. Those are distinct data-quality and study-design questions.
