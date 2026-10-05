import unittest
import numpy as np
import pandas as pd
from rocket_data_science.research import balanced_result,temporal_split,FEATURES,car_rankings,playing_styles,STYLE,match_table,training_features,fit_models

class ResearchTests(unittest.TestCase):
    def test_observed_leader_uses_raw_rate_with_balanced_rate_preserved(self):
        rows=[]
        for i in range(500):
            rows.append(dict(player_id='regular' if i<400 else str(i),replay_id='a'+str(i),car='A',hitbox='Octane',won=int(i<400)))
        for i in range(200):
            rows.append(dict(player_id='b'+str(i//2),replay_id='b'+str(i),car='B',hitbox='Octane',won=i%2))
        ranked,winners=car_rankings(pd.DataFrame(rows))
        a=next(r for r in ranked if r['name']=='A')
        self.assertAlmostEqual(a['raw_win_rate'],80)
        self.assertLess(a['win_rate'],1)
        self.assertIsNotNone(a['raw_low'])
        self.assertEqual(winners[0]['leader']['name'],'A')
        self.assertEqual(winners[0]['rate'],'raw_win_rate')

    def test_weighted_uncertainty_matches_independent_statsmodels_calculation(self):
        import statsmodels.api as sm
        from statsmodels.stats.sandwich_covariance import cov_cluster_2groups
        rng=np.random.default_rng(31)
        frame=pd.DataFrame({'player_id':rng.integers(0,60,600),'replay_id':np.repeat(np.arange(100),6)})
        frame=frame.drop_duplicates(['player_id','replay_id'])
        frame['won']=rng.integers(0,2,len(frame))
        weights=1/frame.groupby('player_id').player_id.transform('size')
        fit=sm.WLS(frame.won,np.ones((len(frame),1)),weights=weights).fit()
        expected=cov_cluster_2groups(fit,frame.player_id.to_numpy(),frame.replay_id.to_numpy())[0][0,0]
        mean,_,se=balanced_result(frame)
        self.assertAlmostEqual(mean,frame.groupby('player_id').won.mean().mean())
        self.assertAlmostEqual(se**2,expected)

    def test_features_cannot_be_selected_by_future_data(self):
        rows=[]
        for i in range(20):
            for team,sign in [('blue',1),('orange',-1)]:
                rows.append(dict(replay_id=str(i),player_id=f'{team}-{i}',team=team,date=f'2026-01-{i+1:02d}',
                    playlist='ranked-doubles' if i<12 else 'unranked-duels',won=int((i%2)==(team=='blue')),
                    average_speed=1000+sign*i,behind_ball_percent=np.nan if i<12 else 50+sign*i,
                    high_air_percent=5 if i<12 else i))
        frame=pd.DataFrame(rows)
        table,candidates=match_table(frame,'all');train,_,_=temporal_split(table)
        chosen=training_features(frame,train,candidates)
        self.assertEqual(chosen,['average_speed'])
        self.assertEqual(len(table),20)
        self.assertEqual(table.loc['2','average_speed'],4)
        changed=frame.copy()
        changed.loc[~changed.replay_id.isin(train.index),'average_speed']=np.nan
        self.assertEqual(training_features(changed,train,candidates),chosen)

    def test_single_outcome_period_is_withheld_instead_of_reporting_invalid_auc(self):
        rows=[]
        for i in range(1000):
            for team in ['blue','orange']:
                rows.append(dict(replay_id=str(i),player_id=f'{team}-{i}',team=team,date=f'{i:04d}',
                    playlist='ranked-duels',won=int(team=='blue'),average_speed=1000+i))
        result=fit_models(pd.DataFrame(rows),'duels')
        self.assertFalse(result['available'])
        self.assertIn('both blue wins',result['reason'])

    def test_findings_require_measured_support_and_offer_distinct_statistics(self):
        rows=[]
        for body,value in [('A',10),('B',30),('C',50),('D',70),('E',90),('Sparse',1000)]:
            for i in range(500):
                rows.append(dict(car=body,hitbox='Octane',player_id=f'{body}-{i}',replay_id=f'{body}-{i}',
                                 **{feature:(value if body!='Sparse' or i<10 else np.nan) for feature in STYLE}))
        frame=pd.DataFrame(rows)
        result=playing_styles(frame,[dict(name=body,qualified=True) for body in frame.car.unique()])
        findings=result['findings']
        self.assertEqual(len(findings),9)
        self.assertEqual(len({f['feature'] for f in findings}),9)
        self.assertNotIn('Sparse',{f['body'] for f in findings})
        self.assertTrue(all(sum(f['body']==body for f in findings)<=2 for body in frame.car.unique()))
        for finding in findings:
            self.assertEqual(finding['n'],500)
            self.assertAlmostEqual(finding['group_mean'],frame[finding['feature']].mean())

    def test_hitbox_styles_weight_appearances_and_include_small_bodies(self):
        rows=[]
        for i,(body,hitbox,value) in enumerate([('Fennec','Octane',0)]*3+[('Rare','Octane',100),('Dominus','Dominus',50)]):
            rows.append(dict(car=body,hitbox=hitbox,player_id=str(i),replay_id=str(i),**{f:value for f in STYLE}))
        frame=pd.DataFrame(rows)
        frame.loc[frame.hitbox=='Dominus','average_boost']=np.nan
        result=playing_styles(frame,[dict(name='Fennec',qualified=True)])['hitboxes']
        cell=next(c for c in result['cells'] if c['body']=='Octane' and c['feature']=='average_speed')
        self.assertEqual(cell['mean'],25)  # Four appearances, not the mean of two body means (50).
        self.assertEqual(cell['n'],4)
        self.assertAlmostEqual(cell['z'],(25-frame.average_speed.mean())/frame.average_speed.std())
        self.assertEqual(next(s for s in result['samples'] if s['name']=='Octane')['bodies'],2)
        self.assertFalse(any(c['body']=='Dominus' and c['feature']=='average_boost' for c in result['cells']))
        self.assertEqual(next(s for s in result['samples'] if s['name']=='Merc')['appearances'],0)

    def test_repeat_player_does_not_dominate_point_estimate(self):
        frame=pd.DataFrame({'player_id':['a']*100+['b'],'replay_id':range(101),'won':[1]*100+[0]})
        mean,_,_=balanced_result(frame)
        self.assertAlmostEqual(mean,.5)

    def test_temporal_splits_do_not_share_timestamps_or_matches(self):
        frame=pd.DataFrame({'date':np.repeat([f'2026-01-{i:02d}' for i in range(1,21)],3),'replay_id':range(60)})
        train,validation,test=temporal_split(frame)
        self.assertLess(train.date.max(),validation.date.min())
        self.assertLess(validation.date.max(),test.date.min())
        self.assertEqual(len(train)+len(validation)+len(test),len(frame))
        self.assertFalse(set(train.replay_id)&set(test.replay_id))

    def test_no_direct_score_or_goal_features(self):
        self.assertTrue(set(FEATURES).isdisjoint({'goals','score','goals_against','shooting_percentage','assists','won','win_rate'}))

    def test_rare_body_not_declared_winner(self):
        rows=[]
        for i in range(150):
            rows.append(dict(player_id=str(i),replay_id=str(i),car='Octane',hitbox='Octane',won=i%2))
        rows.append(dict(player_id='lucky',replay_id='lucky',car='Rare',hitbox='Octane',won=1))
        ranked,winners=car_rankings(pd.DataFrame(rows))
        self.assertFalse(next(r for r in ranked if r['name']=='Rare')['qualified'])
        self.assertEqual(winners[0]['leader']['name'],'Octane')

if __name__=='__main__':unittest.main()
