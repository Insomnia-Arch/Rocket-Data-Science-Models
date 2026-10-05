"""Equal-player paired comparisons within calendar quarter, optionally same queue.

Descriptive matching, not a causal estimate or an estimate of official skill.
No post-match performance variables are used as adjustment covariates.
"""
from itertools import combinations
import math
import numpy as np
import pandas as pd
from scipy.stats import norm
from rocket_data_science.research import cluster_variance,STYLE,FEATURES

HITBOXES=['Octane','Dominus','Breakout','Hybrid','Merc','Plank']
KEYS=['player_id','playlist','quarter']
MIN_EACH=3
PERIOD_MONTHS=3
PERIOD_KIND='calendar_quarter'


def paired_rows(frame,a,b,minimum=MIN_EACH,same_queue=False):
    keys=KEYS if same_queue else ['player_id','quarter']
    rows=frame[frame.hitbox.isin([a,b])].copy()
    if rows.empty:return rows
    # The frozen frame's month is UTC. Map the small set of month labels to
    # fixed, non-overlapping quarters, including the year in every key.
    # Recompute here so restored frames and measured-stat subsets use the
    # same rule without depending on a previously cached matching column.
    quarters={month:str(pd.Period(month,freq='M').asfreq('Q-DEC')) for month in rows.month.dropna().unique()}
    rows['quarter']=rows.month.map(quarters)
    counts=rows.groupby(keys+['hitbox']).size().unstack(fill_value=0)
    if a not in counts or b not in counts:return rows.iloc[:0]
    supported=counts[(counts[a]>=minimum)&(counts[b]>=minimum)].index
    rows=rows[pd.MultiIndex.from_frame(rows[keys]).isin(supported)].copy()
    if rows.empty:return rows
    rows['cell']=pd.factorize(pd.MultiIndex.from_frame(rows[keys]))[0]
    cells_per_player=rows[['player_id','cell']].drop_duplicates().groupby('player_id').size()
    # Each side sums to one. Each person has equal total weight on each side,
    # divided equally over their matched periods and appearances.
    rows['weight']=1/(rows.player_id.nunique()*rows.player_id.map(cells_per_player)*
                       rows.groupby(['cell','hitbox']).hitbox.transform('size'))
    return rows


def pair_result(frame,a,b,same_queue=False):
    rows=paired_rows(frame,a,b,same_queue=same_queue)
    empty=dict(a=a,b=b,players=0,matches=0,appearances=0,cells=0,a_n=0,b_n=0,
        a_rate=None,b_rate=None,difference=None,raw_difference=None,low=None,high=None,
        supported=False,strong_sample=False,status='Not enough matched players',styles=[],distribution=[],familiarity=[])
    if rows.empty:return empty
    left=rows.hitbox.eq(a);right=~left
    a_rate=float(np.dot(rows.loc[left,'weight'],rows.loc[left,'won']))
    b_rate=float(np.dot(rows.loc[right,'weight'],rows.loc[right,'won']))
    delta=a_rate-b_rate
    influence=rows.weight.to_numpy()*np.where(left,rows.won-a_rate,-(rows.won-b_rate))
    variance=cluster_variance(influence,rows.player_id,rows.replay_id)
    players=int(rows.player_id.nunique());matches=int(rows.replay_id.nunique())
    supported=players>=30 and matches>=100 and min(int(left.sum()),int(right.sum()))>=100 and variance is not None
    # All 15 possible hitbox pairs, including sparse pairs, form the family.
    width=float(norm.ppf(1-.05/(2*15)))*math.sqrt(variance) if supported else None
    means=rows.groupby(['player_id','cell','hitbox']).won.mean().unstack()
    per_player=(means[a]-means[b]).groupby('player_id').mean()*100
    edges=[-100.0001,-25,-10,-2,2,10,25,100.0001]
    labels=['−100 to −25','−25 to −10','−10 to −2','−2 to +2','+2 to +10','+10 to +25','+25 to +100']
    hist=pd.cut(per_player,edges,labels=labels).value_counts(sort=False)
    styles=[]
    for feature in STYLE:
        if feature not in rows:continue
        # Require measured appearances on both sides for each matched cell.
        measured=paired_rows(rows[np.isfinite(rows[feature])],a,b,same_queue=same_queue)
        if measured.empty:continue
        l=measured.hitbox.eq(a)
        av=float(np.dot(measured.loc[l,'weight'],measured.loc[l,feature]));bv=float(np.dot(measured.loc[~l,'weight'],measured.loc[~l,feature]))
        styles.append(dict(feature=feature,label=FEATURES[feature],a_mean=av,b_mean=bv,difference=av-bv,
                           relative_percent=100*(av-bv)/abs(bv) if abs(bv)>1e-9 else None,
                           players=int(measured.player_id.nunique()),n=len(measured)))
    familiarity=[]
    if 'prior_hitbox' in rows:
        for name,subset in rows.groupby('hitbox'):
            for label,lo,hi in [('0–4 earlier',0,4),('5–19 earlier',5,19),('20+ earlier',20,np.inf)]:
                r=subset[subset.prior_hitbox.between(lo,hi)]
                familiarity.append(dict(name=name,bucket=label,appearances=len(r),players=int(r.player_id.nunique()),
                    matches=int(r.replay_id.nunique()),rate=100*float(r.won.mean()) if len(r) else None))
    return dict(empty,players=players,matches=matches,appearances=len(rows),cells=int(rows.cell.nunique()),
        a_n=int(left.sum()),b_n=int(right.sum()),a_rate=100*a_rate,b_rate=100*b_rate,difference=100*delta,
        raw_difference=100*(float(frame.loc[frame.hitbox.eq(a),'won'].mean())-float(frame.loc[frame.hitbox.eq(b),'won'].mean())),
        low=100*(delta-width) if width is not None else None,high=100*(delta+width) if width is not None else None,
        supported=supported,strong_sample=supported and players>=100 and matches>=200 and min(int(left.sum()),int(right.sum()))>=500,
        status=('Difference remains' if abs(delta)>width else 'No clear difference') if supported else 'Not enough matched players',
        styles=styles,distribution=[dict(label=label,players=int(hist[label])) for label in labels],familiarity=familiarity)


def compare_switchers(frame,same_queue=False):
    pairs=[pair_result(frame,a,b,same_queue=same_queue) for a,b in combinations(HITBOXES,2)]
    return dict(pairs=pairs,minimum_each=MIN_EACH,same_queue=same_queue,period_months=PERIOD_MONTHS,period_kind=PERIOD_KIND,
                method='Equal-player paired rates within the same UTC calendar quarter (Jan–Mar, Apr–Jun, Jul–Sep, Oct–Dec)'+(' and original queue' if same_queue else ', allowing hitbox use across modes and queues')+'; at least three appearances on each hitbox in every retained cell.',
                uncertainty='Ranges cluster by player and replay and correct for all 15 hitbox pairs within the selected view. Filters overlap; results across views are exploratory.',
                limitations='Matches the same players within non-overlapping three-month calendar periods, not rolling 90-day windows. A quarter may be only partly represented in the collected data. Mode mix, opponents, parties, improvement, match state and hitbox familiarity can still differ; a wider period allows more time for those factors to change. Recorded familiarity is earlier collected appearances, not lifetime experience.')
