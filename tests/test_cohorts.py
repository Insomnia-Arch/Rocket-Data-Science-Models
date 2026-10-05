import unittest
import pandas as pd
from rocket_data_science.cohorts import group_modes,composition,analysis_cohorts
from rocket_data_science.research import car_rankings,match_table


class CohortTests(unittest.TestCase):
    def test_all_view_preserves_unique_rows_and_player_weights(self):
        frame=group_modes(pd.DataFrame([
            dict(playlist='ranked-duels',replay_id='a',player_id='same',won=1),
            dict(playlist='unranked-doubles',replay_id='b',player_id='same',won=0),
            dict(playlist='ranked-standard',replay_id='c',player_id='other',won=0)]))
        views=dict(analysis_cohorts(frame))
        self.assertEqual(list(views),['all','duels','team'])
        self.assertEqual(len(views['all']),3)
        self.assertEqual(views['all'].replay_id.nunique(),3)
        self.assertEqual(views['all'].player_id.nunique(),2)
        from rocket_data_science.research import balanced_result
        self.assertAlmostEqual(balanced_result(views['all'])[0],.25)

    def test_only_standard_soccar_modes_are_pooled_and_originals_preserved(self):
        playlists=['ranked-duels','unranked-duels','ranked-doubles','unranked-doubles',
                   'ranked-standard','unranked-standard','ranked-solo-standard','unranked-chaos','private','ranked-hoops']
        source=pd.DataFrame({'playlist':playlists})
        pooled=group_modes(source)
        self.assertEqual(pooled.analysis_group.tolist(),['duels']*2+['team']*5)
        self.assertEqual(pooled.playlist.tolist(),playlists[:7])
        self.assertNotIn('analysis_group',source)

    def test_pool_increases_body_support_without_double_counting_players(self):
        frame=pd.DataFrame([dict(playlist='ranked-doubles' if i<60 else 'unranked-standard',
            replay_id=str(i),player_id=str(i%40),car='Octane',hitbox='Octane',won=(i%40)%2) for i in range(120)])
        for _,part in frame.groupby('playlist'):
            self.assertFalse(car_rankings(part)[0][0]['qualified'])
        pooled=group_modes(frame)
        ranking=car_rankings(pooled)[0][0]
        self.assertTrue(ranking['qualified'])
        self.assertEqual(ranking['matches'],120)
        self.assertEqual(ranking['players'],40)
        self.assertEqual(sum(r['matches'] for r in composition(pooled)),120)

    def test_models_use_team_means_and_keep_playlist_context(self):
        rows=[]
        for i,(playlist,size) in enumerate([('ranked-doubles',2),('unranked-standard',3)]):
            for team in ('blue','orange'):
                for player in range(size):
                    rows.append(dict(replay_id=str(i),player_id=f'{i}-{team}-{player}',date=f'2026-01-0{i+1}',
                        playlist=playlist,team=team,won=int(team=='blue'),average_speed=1000 if team=='blue' else 900))
        table,features=match_table(pd.DataFrame(rows),'team')
        self.assertEqual(len(table),2)
        self.assertEqual(table.average_speed.tolist(),[100,100])
        self.assertEqual(table.is_3v3.tolist(),[0,1])
        self.assertEqual(table.is_casual.tolist(),[0,1])
        self.assertIn('is_3v3',features)


if __name__=='__main__':unittest.main()
