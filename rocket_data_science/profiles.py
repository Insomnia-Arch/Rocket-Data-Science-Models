"""Body and hitbox profiles used by the version 1.3 dashboard."""
import numpy as np
import pandas as pd
from .snapshot import METADATA, mean_summary
from .research import FEATURES, balanced_result
from .matched_hitboxes import HITBOXES

MEASURES = {**METADATA, **{f: (label, '% of time' if f.endswith('_percent') else 'per 5 minutes' if f.endswith('_per5') else 'replay units') for f, label in FEATURES.items()}}
MEASURES.pop('win_rate', None)

MODES=['all','duels','team']


QUEUES=['all','ranked','casual']


def recorded_familiarity(frame):
    prior=frame.groupby(['player_id','playlist','hitbox','date']).size().rename('at_time').reset_index().sort_values('date',kind='stable')
    prior['prior_hitbox']=prior.groupby(['player_id','playlist','hitbox'])['at_time'].cumsum()-prior['at_time']
    return frame.merge(prior.drop(columns='at_time'),on=['player_id','playlist','hitbox','date'],validate='many_to_one')


def choose(frame,mode,queue):
    if mode not in MODES or queue not in QUEUES:raise ValueError('Unknown view')
    rows=frame if mode=='all' else frame[frame.analysis_group==mode]
    if queue!='all':rows=rows[rows.playlist.str.startswith('ranked-' if queue=='ranked' else 'unranked-')]
    return rows


def profiles(frame,group):
    known=frame[frame.hitbox.isin(HITBOXES)]
    centers=frame[list(MEASURES)].mean();scales=frame[list(MEASURES)].std()
    output=[]
    for name,rows in known.groupby(group):
        count=len(rows);players=int(rows.player_id.nunique());matches=int(rows.replay_id.nunique())
        raw=mean_summary(rows,'win_rate');balanced,_,se=balanced_result(rows)
        qualified=count>=100 and players>=30 and matches>=100
        # Keep all identities, but only compute detailed statistics at the
        # exploratory threshold. Sparse options remain explicitly flagged.
        stats={}
        if qualified:
            for feature in MEASURES:
                m=mean_summary(rows,feature)
                center=centers[feature];scale=scales[feature]
                m.update(group_mean=float(center) if np.isfinite(center) else None,
                    relative_percent=100*(m['mean']-center)/abs(center) if m['mean'] is not None and np.isfinite(center) and abs(center)>1e-9 else None,
                    z=float((m['mean']-center)/scale) if m['mean'] is not None and np.isfinite(scale) and scale>0 else None)
                stats[feature]=m
        output.append(dict(name=name,kind=group,hitbox=name if group=='hitbox' else str(rows.hitbox.mode().iloc[0]),
            appearances=count,players=players,matches=matches,bodies=int(rows.car.nunique()),usage=100*count/len(frame),
            raw_win_rate=raw['mean'],raw_low=raw['low'],raw_high=raw['high'],win_rate=100*balanced,
            low=100*(balanced-1.96*se) if se is not None and qualified else None,
            high=100*(balanced+1.96*se) if se is not None and qualified else None,
            qualified=qualified,strong_sample=count>=500 and players>=100 and matches>=200,metrics=stats))
    return sorted(output,key=lambda p:p['raw_win_rate'],reverse=True)


def findings(bodies):
    candidates=[]
    for body in bodies:
        if not body['strong_sample']:continue
        for feature,metric in body['metrics'].items():
            if feature not in FEATURES or metric['z'] is None or metric['relative_percent'] is None:continue
            if metric['n']<500 or metric['players']<100 or metric['matches']<200:continue
            candidates.append(dict(body=body['name'],feature=feature,label=MEASURES[feature][0],**metric))
    candidates.sort(key=lambda r:abs(r['z']),reverse=True)
    selected=[];body_counts={};feature_counts={}
    for row in candidates:
        if body_counts.get(row['body'],0)>=3 or feature_counts.get(row['feature'],0)>=2:continue
        selected.append(row);body_counts[row['body']]=body_counts.get(row['body'],0)+1;feature_counts[row['feature']]=feature_counts.get(row['feature'],0)+1
        if len(selected)>=18:break
    gaps=sorted([dict(name=b['name'],hitbox=b['hitbox'],raw=b['raw_win_rate'],balanced=b['win_rate'],gap=b['raw_win_rate']-b['win_rate'],
                      players=b['players'],matches=b['matches'],appearances=b['appearances']) for b in bodies if b['strong_sample']],key=lambda b:abs(b['gap']),reverse=True)[:8]
    return dict(styles=selected,gaps=gaps)
