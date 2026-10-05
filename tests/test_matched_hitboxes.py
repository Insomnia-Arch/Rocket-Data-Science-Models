import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from rocket_data_science.profiles import choose, profiles, recorded_familiarity, MEASURES
from rocket_data_science.matched_hitboxes import paired_rows, pair_result


class Version13Tests(unittest.TestCase):
    def test_recorded_familiarity_never_counts_simultaneous_or_future_games(self):
        frame=pd.DataFrame([dict(player_id='p',playlist='ranked-duels',hitbox='Octane',date=d) for d in ['2026-03-01','2026-01-01','2026-01-01','2026-02-01']])
        actual=recorded_familiarity(frame)
        self.assertEqual(actual.prior_hitbox.tolist(),[3,0,0,2])
        future=pd.concat([frame,pd.DataFrame([dict(player_id='p',playlist='unranked-duels',hitbox='Octane',date='2026-01-01')])])
        self.assertEqual(recorded_familiarity(future).prior_hitbox.tolist(),[3,0,0,2,0])

    def test_hitbox_balance_is_calculated_across_bodies_for_each_person(self):
        rows=[]
        for person,body,n,won in [('p','Octane',9,1),('p','Fennec',1,0),('q','Fennec',2,0)]:
            for i in range(n):
                row=dict(player_id=person,car=body,hitbox='Octane',replay_id=f'{person}{body}{i}',won=won,win_rate=won*100)
                row.update({key:np.nan for key in MEASURES});rows.append(row)
        result=profiles(pd.DataFrame(rows),'hitbox')[0]
        self.assertEqual(result['players'],2)
        self.assertAlmostEqual(result['raw_win_rate'],75)
        self.assertAlmostEqual(result['win_rate'],45)

    def test_ranked_filter_does_not_include_unranked(self):
        frame=pd.DataFrame(dict(playlist=['ranked-duels','unranked-duels','ranked-doubles'],analysis_group=['duels','duels','team']))
        self.assertEqual(list(choose(frame,'all','ranked').index),[0,2])
        self.assertEqual(list(choose(frame,'duels','casual').index),[1])
        with self.assertRaises(ValueError):choose(frame,'bad','all')

    def test_matching_pools_months_and_modes_but_respects_quarter_and_year_boundaries(self):
        rows=[]
        for month,hitbox in [('2026-01','Octane'),('2026-02','Dominus')]:
            rows.extend(dict(player_id='p',playlist='ranked-duels',month=month,hitbox=hitbox,won=1,replay_id=f'{hitbox}{i}') for i in range(3))
        frame=pd.DataFrame(rows)
        self.assertEqual(len(paired_rows(frame,'Octane','Dominus')),6)
        for outside in ['2026-04','2025-12','2025-01']:
            frame.loc[frame.hitbox=='Dominus','month']=outside
            self.assertTrue(paired_rows(frame,'Octane','Dominus').empty)
        frame.loc[frame.hitbox=='Dominus','month']='2026-03'
        self.assertEqual(len(paired_rows(frame,'Octane','Dominus')),6)
        frame.loc[frame.hitbox=='Dominus','playlist']='unranked-standard'
        self.assertTrue(paired_rows(frame,'Octane','Dominus',same_queue=True).empty)
        self.assertEqual(len(paired_rows(frame,'Octane','Dominus')),6)

    def test_three_appearances_can_accumulate_across_months_but_not_quarters(self):
        rows=[dict(player_id='p',playlist='ranked-duels',month=month,hitbox=hitbox,won=1,replay_id=f'{hitbox}{month}')
              for hitbox in ['Octane','Dominus'] for month in ['2026-01','2026-02','2026-03']]
        frame=pd.DataFrame(rows)
        self.assertEqual(len(paired_rows(frame,'Octane','Dominus')),6)
        frame.loc[(frame.hitbox=='Dominus')&(frame.month=='2026-03'),'month']='2026-04'
        self.assertTrue(paired_rows(frame,'Octane','Dominus').empty)

    def test_each_player_has_equal_weight_across_unequal_numbers_of_quarters(self):
        rows=[]
        for player,month,n,a_win,b_win in [('p','2026-01',30,1,0),('p','2026-04',3,0,1),('q','2026-02',3,0,1)]:
            for hitbox,won in [('Octane',a_win),('Dominus',b_win)]:
                rows.extend(dict(player_id=player,playlist='ranked-duels',month=month,hitbox=hitbox,won=won,
                                 replay_id=f'{player}{month}{hitbox}{i}') for i in range(n))
        frame=pd.DataFrame(rows)
        weighted=paired_rows(frame,'Octane','Dominus')
        self.assertTrue(np.allclose(weighted.groupby(['player_id','hitbox']).weight.sum(),.5))
        result=pair_result(frame,'Octane','Dominus')
        self.assertAlmostEqual(result['a_rate'],25)
        self.assertAlmostEqual(result['b_rate'],75)
        self.assertAlmostEqual(result['difference'],-50)

    def test_style_measurements_use_the_same_quarter_and_require_three_on_each_side(self):
        rows=[dict(player_id='p',playlist='ranked-duels',month=month,hitbox=hitbox,won=1,
                   replay_id=f'{hitbox}{month}',average_speed=100 if hitbox=='Octane' else 200)
              for hitbox in ['Octane','Dominus'] for month in ['2026-01','2026-02','2026-03']]
        frame=pd.DataFrame(rows)
        with patch('rocket_data_science.matched_hitboxes.STYLE',['average_speed']),patch('rocket_data_science.matched_hitboxes.FEATURES',{'average_speed':'Average speed'}):
            result=pair_result(frame,'Octane','Dominus')
            self.assertEqual(result['styles'][0]['relative_percent'],-50)
            frame.loc[0,'average_speed']=np.nan
            self.assertEqual(pair_result(frame,'Octane','Dominus')['styles'],[])

    def test_pair_gives_people_equal_weight_despite_unequal_upload_counts(self):
        rows=[]
        for player,n,outcomes in [('frequent',60,{'Octane':1,'Dominus':0}),('occasional',3,{'Octane':0,'Dominus':1})]:
            for hitbox,won in outcomes.items():
                rows.extend(dict(player_id=player,playlist='ranked-duels',month='2026-01',hitbox=hitbox,won=won,replay_id=f'{player}{hitbox}{i}') for i in range(n))
        frame=pd.DataFrame(rows)
        result=pair_result(frame,'Octane','Dominus')
        self.assertAlmostEqual(result['a_rate'],50)
        self.assertAlmostEqual(result['b_rate'],50)
        self.assertAlmostEqual(result['difference'],0)
        self.assertFalse(result['supported'])
        self.assertIsNone(result['low'])

    def test_identical_people_with_opposite_preferences_have_paired_uncertainty(self):
        rng=np.random.default_rng(4);rows=[]
        for player in range(110):
            for hitbox in ['Octane','Dominus']:
                for i in range(8):
                    rows.append(dict(player_id=str(player),playlist='ranked-duels',month='2026-01',hitbox=hitbox,
                                     won=int(rng.random()<(.8 if hitbox=='Octane' else .2)),replay_id=f'{player}{hitbox}{i}'))
        frame=pd.DataFrame(rows);result=pair_result(frame,'Octane','Dominus');reverse=pair_result(frame,'Dominus','Octane')
        self.assertTrue(result['strong_sample']);self.assertGreater(result['low'],0)
        self.assertAlmostEqual(result['difference'],-reverse['difference'])
        self.assertAlmostEqual(result['low'],-reverse['high'])
        self.assertEqual(sum(row['players'] for row in result['distribution']),110)
