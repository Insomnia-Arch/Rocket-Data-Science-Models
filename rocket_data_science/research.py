"""Player-balanced comparisons and held-out win classifiers."""
import math
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss, roc_curve
from sklearn.inspection import permutation_importance
from sklearn.calibration import calibration_curve

FEATURES={
 'shots_per5':'Shots / 5 min','saves_per5':'Saves / 5 min','demos_per5':'Demos / 5 min',
 'average_speed':'Average speed','supersonic_percent':'Supersonic time',
 'high_air_percent':'High-air time','offensive_half_percent':'Offensive-half time',
 'behind_ball_percent':'Time behind ball','distance_to_ball':'Distance to ball',
 'distance_to_teammates':'Distance to teammates','average_boost':'Average boost held',
 'zero_boost_percent':'Time without boost','boost_per_minute':'Boost consumed / min',
 'boost_stolen_per5':'Boost stolen / 5 min','boost_overfill_per5':'Boost overfill / 5 min',
 'boost_supersonic_per5':'Boost used at supersonic / 5 min','powerslides_per5':'Powerslides / 5 min'}


CONTEXT_LABELS={'is_duels':'1v1 context','is_3v3':'3v3 context','is_casual':'Casual context'}


STYLE=['average_speed','supersonic_percent','high_air_percent','demos_per5','shots_per5','saves_per5','average_boost','zero_boost_percent','behind_ball_percent','distance_to_ball']


SEED=20260908


def cluster_variance(influence,players,matches):
    n=len(influence);p=pd.Series(influence).groupby(np.asarray(players)).sum();m=pd.Series(influence).groupby(np.asarray(matches)).sum()
    if min(len(p),len(m),n)<2:return None
    value=len(p)/(len(p)-1)*float(p.dot(p))+len(m)/(len(m)-1)*float(m.dot(m))-n/(n-1)*float(np.dot(influence,influence))
    return value if math.isfinite(value) and value>0 else None


def balanced_result(rows):
    weights=1/rows.groupby('player_id').player_id.transform('size').to_numpy()
    mean=float(np.average(rows.won,weights=weights))
    influence=weights*(rows.won.to_numpy()-mean)/weights.sum()
    variance=cluster_variance(influence,rows.player_id,rows.replay_id)
    se=math.sqrt(variance) if variance is not None else None
    return mean,influence,se


def car_rankings(df):
    ranked=[];members={};influences={}
    for car,rows in df[df.hitbox!='Unknown'].groupby('car'):
        mean,influence,se=balanced_result(rows)
        raw=float(rows.won.mean())
        raw_influence=(rows.won.to_numpy()-raw)/len(rows)
        raw_variance=cluster_variance(raw_influence,rows.player_id,rows.replay_id)
        raw_se=math.sqrt(raw_variance) if raw_variance is not None else None
        row=dict(name=car,hitbox=rows.hitbox.mode().iloc[0],appearances=len(rows),players=int(rows.player_id.nunique()),
                 matches=int(rows.replay_id.nunique()),raw_win_rate=100*float(rows.won.mean()),win_rate=100*mean,
                 low=100*(mean-1.96*se) if se else None,high=100*(mean+1.96*se) if se else None,
                 raw_low=100*(raw-1.96*raw_se) if raw_se else None,raw_high=100*(raw+1.96*raw_se) if raw_se else None,
                 usage=100*len(rows)/len(df))
        row['qualified']=len(rows)>=100 and row['players']>=30 and row['matches']>=100 and se is not None
        row['strong_sample']=len(rows)>=500 and row['players']>=100 and row['matches']>=200
        ranked.append(row);members[car]=rows;influences[car]=raw_influence
    winners=[]
    for hitbox in ('Octane','Dominus','Breakout','Hybrid','Merc','Plank'):
        candidates=sorted([r for r in ranked if r['hitbox']==hitbox and r['qualified']],key=lambda r:r['raw_win_rate'],reverse=True)
        leader=candidates[0] if candidates else None
        comparisons=[]
        if leader:
            # Family adjustment covers every possible within-hitbox pair, not
            # just comparisons chosen after observing the highest mean.
            pairs=max(1,len(candidates)*(len(candidates)-1)//2)
            critical=float(norm.ppf(1-.05/(2*pairs)))
            a=members[leader['name']]
            for other in candidates[1:]:
                b=members[other['name']]
                influence=np.concatenate([influences[leader['name']],-influences[other['name']]])
                variance=cluster_variance(influence,pd.concat([a.player_id,b.player_id]),pd.concat([a.replay_id,b.replay_id]))
                delta=leader['raw_win_rate']-other['raw_win_rate'];width=100*critical*math.sqrt(variance) if variance else None
                comparisons.append(dict(name=other['name'],difference=delta,low=delta-width if width else None,high=delta+width if width else None))
        separated=bool(comparisons) and all(c['low'] is not None and c['low']>0 for c in comparisons)
        winners.append(dict(hitbox=hitbox,leader=leader,candidates=len(candidates),comparisons=comparisons,rate='raw_win_rate',
                            verdict='Clear observed separation' if separated else 'No clear winner' if len(candidates)>1 else 'Only one qualified body' if leader else 'More data needed'))
    return sorted(ranked,key=lambda r:r['appearances'],reverse=True),winners


def match_table(df,playlist):
    # Construct candidates without consulting later-period distributions.
    # Availability and variation are checked using training appearances below.
    features=[f for f in FEATURES if f in df]
    numeric=df[features].replace([np.inf,-np.inf],np.nan)
    teams=numeric.groupby([df.replay_id,df.team]).mean()
    blue=teams.xs('blue',level='team');orange=teams.xs('orange',level='team')
    # One observation per match avoids placing opposing sides in different sets.
    x=blue.subtract(orange)
    info=df.groupby('replay_id').agg(date=('date','first'),roster=('player_id',list),playlist=('playlist','first'))
    y=df[df.team=='blue'].groupby('replay_id').won.first()
    data=x.join(info).join(y).dropna(subset=['won']).sort_values(['date'],kind='stable')
    for key,values in {'is_duels':data.playlist.str.endswith('duels'),'is_3v3':data.playlist.str.endswith('standard'),'is_casual':data.playlist.str.startswith('unranked-')}.items():
        if values.nunique()>1:
            data[key]=values.astype(int);features.append(key)
    return data,features


def temporal_split(data):
    # Equal timestamps always stay in the same partition.
    dates=sorted(data.date.unique())
    if len(dates)<5:return data.iloc[:0],data.iloc[:0],data
    a=dates[int(len(dates)*.6)];b=dates[int(len(dates)*.8)]
    return data[data.date<a],data[(data.date>=a)&(data.date<b)],data[data.date>=b]


def fit_models(df,playlist,without_shots_saves=False):
    table,features=match_table(df,playlist)
    train,validation,test=temporal_split(table)
    if min(len(train),len(validation),len(test))<200:return {'available':False,'reason':'Too few temporally separated matches'}
    features=training_features(df,train,features,without_shots_saves)
    stat_features=[f for f in features if f in FEATURES]
    if not stat_features:return {'available':False,'reason':'No statistics have sufficient usable training data'}
    if any(part.won.nunique()<2 for part in (train,validation,test)):
        return {'available':False,'reason':'Each time period needs both blue wins and orange wins'}
    fitted=[];models=[]
    algorithms=[
      ('Logistic regression',make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(C=.3,max_iter=500,random_state=SEED))),
      ('Random forest',make_pipeline(SimpleImputer(strategy='median'),RandomForestClassifier(n_estimators=100,max_depth=10,min_samples_leaf=30,n_jobs=1,random_state=SEED))),
      ('Gradient boosting',make_pipeline(SimpleImputer(strategy='median'),HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=15,l2_regularization=3,early_stopping=False,random_state=SEED)))]
    train_players=set(p for roster in train.roster for p in roster)
    novel=test.roster.map(lambda roster:all(p not in train_players for p in roster))
    importance_test=test.sample(min(3000,len(test)),random_state=SEED)
    for name,model in algorithms:
        print(f'{playlist}: fitting {name} ({len(train):,} train matches)',flush=True)
        model.fit(train[features],train.won)
        validation_probability=model.predict_proba(validation[features])[:,1]
        probability=model.predict_proba(test[features])[:,1]
        importance=permutation_importance(model,importance_test[features],importance_test.won,n_repeats=3,scoring='roc_auc',n_jobs=1,random_state=SEED)
        fpr,tpr,_=roc_curve(test.won,probability)
        indices=np.unique(np.linspace(0,len(fpr)-1,min(80,len(fpr))).astype(int))
        actual,predicted=calibration_curve(test.won,probability,n_bins=8,strategy='quantile')
        item=dict(name=name,validation_auc=float(roc_auc_score(validation.won,validation_probability)),
            test_auc=float(roc_auc_score(test.won,probability)),accuracy=float(accuracy_score(test.won,probability>=.5)),
            brier=float(brier_score_loss(test.won,probability)),
            importance=[dict(feature=f,label=FEATURES[f],mean=float(importance.importances_mean[i]),std=float(importance.importances_std[i])) for i,f in enumerate(features) if f in FEATURES],
            roc=[dict(false_positive=float(fpr[i]),true_positive=float(tpr[i])) for i in indices],
            calibration=[dict(predicted=float(p),actual=float(a)) for p,a in zip(predicted,actual)],
            new_player_matches=int(novel.sum()),new_player_auc=float(roc_auc_score(test.loc[novel,'won'],probability[novel])) if novel.sum()>=200 and test.loc[novel,'won'].nunique()==2 else None)
        item['test_by_playlist']=[dict(playlist=p,matches=len(g),auc=float(roc_auc_score(g.won,probability[test.index.get_indexer(g.index)])) if len(g)>=200 and g.won.nunique()==2 else None) for p,g in test.groupby('playlist')]
        if name=='Logistic regression':item['coefficients']=[dict(feature=f,label={**FEATURES,**CONTEXT_LABELS}[f],odds_ratio=float(np.exp(c))) for f,c in zip(features,model[-1].coef_[0])]
        models.append(item);fitted.append(model)
    selected=max(models,key=lambda m:m['validation_auc'])['name']
    consensus=sorted([dict(feature=f,label=FEATURES[f],importance=float(np.mean([m['importance'][i]['mean'] for m in models]))) for i,f in enumerate(features) if f in FEATURES],key=lambda x:x['importance'],reverse=True)
    curves={}
    # Held-out descriptive response curves; neither interventions nor adjusted effects.
    for feature in stat_features:
        valid=test[[feature,'won']].dropna()
        valid['bin']=pd.qcut(valid[feature],8,duplicates='drop')
        curves[feature]=[dict(x=float(group[feature].mean()),win_rate=100*float(group.won.mean()),matches=len(group)) for _,group in valid.groupby('bin',observed=True)]
    baseline=float(train.won.mean())
    return dict(available=True,features=stat_features,context_features=[f for f in features if f in CONTEXT_LABELS],
                training_playlists=train.playlist.value_counts().astype(int).to_dict(),
                unseen_test_playlists=sorted(set(test.playlist)-set(train.playlist)),
                train_matches=len(train),validation_matches=len(validation),test_matches=len(test),
                train_end=str(train.date.max()),validation_start=str(validation.date.min()),test_start=str(test.date.min()),test_end=str(test.date.max()),
                baseline_auc=.5,baseline_accuracy=float(np.mean(test.won==(baseline>=.5))),baseline_brier=float(brier_score_loss(test.won,np.full(len(test),baseline))),
                selected_model=selected,models=models,consensus=consensus,curves=curves,
                repeated_player_test_matches=int((~novel).sum()),seed=SEED,importance_sample_matches=len(importance_test))


def training_features(df,train,candidates,without_shots_saves=False):
    """Decide usable inputs from training rows only, never validation/test."""
    appearances=df[df.replay_id.isin(train.index)]
    chosen=[]
    for feature in candidates:
        if feature in CONTEXT_LABELS:
            if train[feature].nunique()>1:chosen.append(feature)
            continue
        if without_shots_saves and feature in ('shots_per5','saves_per5'):continue
        measured=appearances[feature].replace([np.inf,-np.inf],np.nan)
        if measured.notna().mean()>=.95 and measured.std()>1e-9 and train[feature].notna().any():
            chosen.append(feature)
    return chosen


def playing_styles(df,rankings):
    bodies=[r['name'] for r in rankings if r['qualified']][:12]
    means=df.groupby('car')[STYLE].mean();center=df[STYLE].mean();scale=df[STYLE].std().replace(0,np.nan)
    cells=[]
    for body in bodies:
        for feature in STYLE:
            value=means.loc[body,feature];z=(value-center[feature])/scale[feature]
            if np.isfinite(z):cells.append(dict(body=body,feature=feature,label=FEATURES[feature],mean=float(value),z=float(z)))
    # Descriptive highlights, not significance tests. Require measured support,
    # diversify statistics, and avoid letting one body dominate the selection.
    supported=[]
    for cell in cells:
        measured=df.loc[(df.car==cell['body']) & df[cell['feature']].notna()]
        n=len(measured);players=int(measured.player_id.nunique());matches=int(measured.replay_id.nunique())
        if n>=500 and players>=100 and matches>=200:
            supported.append(dict(**cell,n=n,players=players,matches=matches,group_mean=float(center[cell['feature']])))
    interesting=sorted(supported,key=lambda c:(-abs(c['z']),c['feature'],c['body']))
    findings=[];used=set();used_bodies={}
    for cell in interesting:
        if cell['feature'] in used or used_bodies.get(cell['body'],0)>=2:continue
        used.add(cell['feature'])
        used_bodies[cell['body']]=used_bodies.get(cell['body'],0)+1
        findings.append(dict(**cell,text=f"{cell['body']} users average {cell['mean']:.2f} for {cell['label'].lower()}, compared with {cell['group_mean']:.2f} across this group. Based on {cell['n']:,} measured appearances."))
        if len(findings)==9:break
    hitboxes=['Octane','Dominus','Breakout','Hybrid','Merc','Plank']
    hitbox_cells=[];samples=[]
    for hitbox in hitboxes:
        rows=df[df.hitbox==hitbox]
        samples.append(dict(name=hitbox,appearances=len(rows),players=int(rows.player_id.nunique()),
                            matches=int(rows.replay_id.nunique()),bodies=int(rows.car.nunique())))
        for feature in STYLE:
            measured=rows[feature].dropna();value=measured.mean()
            if not np.isfinite(value):continue
            z=(value-center[feature])/scale[feature]
            hitbox_cells.append(dict(body=hitbox,feature=feature,label=FEATURES[feature],mean=float(value),
                                     z=float(z) if np.isfinite(z) else None,n=len(measured)))
    return dict(bodies=bodies,features=[dict(key=f,label=FEATURES[f]) for f in STYLE],cells=cells,findings=findings,
                hitboxes=dict(names=hitboxes,cells=hitbox_cells,samples=samples))
