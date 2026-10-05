import unittest
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_cluster_2groups
from rocket_data_science.snapshot import mean_summary


class SnapshotTests(unittest.TestCase):
    def test_clustered_mean_interval_matches_statsmodels(self):
        rng=np.random.default_rng(27)
        player=np.tile(np.arange(60),10)
        match=np.repeat(np.arange(100),6)
        values=rng.normal(size=60)[player]+rng.normal(size=100)[match]+rng.normal(size=600)
        frame=pd.DataFrame({'player_id':player,'replay_id':match,'score':values})
        result=mean_summary(frame,'score')
        fit=sm.OLS(values,np.ones((600,1))).fit()
        variance=cov_cluster_2groups(fit,player,match)[0][0,0]
        self.assertAlmostEqual(result['mean'],values.mean())
        self.assertAlmostEqual(result['high'],values.mean()+1.96*np.sqrt(variance))
        self.assertAlmostEqual(result['low'],values.mean()-1.96*np.sqrt(variance))
