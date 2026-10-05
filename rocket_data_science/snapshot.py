"""Appearance-weighted summaries with player and match clustered uncertainty."""
import math
import numpy as np
import pandas as pd

METADATA = {
    'score': ('Score', 'points / appearance'),
    'average_speed': ('Average speed', 'uu/s'),
    'shooting_percentage': ('Shooting percentage', '%'),
    'demos_inflicted': ('Demos inflicted', 'per appearance'),
    'demos_taken': ('Demos taken', 'per appearance'),
    'goals': ('Goals', 'per appearance'),
    'assists': ('Assists', 'per appearance'),
    'saves': ('Saves', 'per appearance'),
    'shots': ('Shots', 'per appearance'),
    'boost_per_minute': ('Boost used', 'boost / minute'),
    'supersonic_percent': ('Time at supersonic speed', '% of time'),
    'air_percent': ('Time in low air', '% of time'),
    'closest_to_ball_percent': ('Time closest to ball', '% of time'),
    'offensive_half_percent': ('Time in offensive half', '% of time'),
    'score_share': ('Share of team score', '%'),
    'win_rate': ('Win rate', '%'),
}


def mean_summary(rows, metric):
    valid = rows.loc[np.isfinite(rows[metric]),['player_id','replay_id',metric]]
    n = len(valid)
    players = valid.player_id.nunique()
    matches = valid.replay_id.nunique()
    output = dict(mean=None,low=None,high=None,n=n,players=int(players),matches=int(matches),
                  coverage=round(100*n/len(rows),1),interval_reason='Too few distinct players or matches')
    if not n: return output
    mean = float(valid[metric].mean())
    output['mean'] = mean
    if min(players,matches)<30 or n<100: return output
    # Intercept-only two-way cluster sandwich: player + match - intersection.
    # (player, replay) is unique in normalized data, so its correction is n/(n-1).
    # This matches statsmodels cov_cluster_2groups for an intercept-only OLS fit.
    influence = (valid[metric]-mean)/n
    by_player = influence.groupby(valid.player_id).sum()
    by_match = influence.groupby(valid.replay_id).sum()
    variance = (players/(players-1)*float(by_player.dot(by_player))
                +matches/(matches-1)*float(by_match.dot(by_match))
                -n/(n-1)*float(influence.dot(influence)))
    if not math.isfinite(variance) or variance<=0:
        output['interval_reason'] = 'Clustered uncertainty could not be estimated reliably'
        return output
    width = 1.96*math.sqrt(variance)
    output.update(low=mean-width,high=mean+width,interval_reason='')
    return output
