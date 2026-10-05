"""Calculation fingerprints and end-to-end classifiers on fabricated records."""
import ast
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from rocket_data_science.research import fit_models

ROOT = Path(__file__).resolve().parents[1]


class ReferenceTests(unittest.TestCase):
    def test_extracted_calculation_fingerprints(self):
        provenance = json.loads((ROOT / 'provenance.json').read_text())
        for entry in provenance['calculations']:
            tree = ast.parse((ROOT / entry['module']).read_text(encoding='utf-8'))
            def name(node):
                if isinstance(node, ast.Assign) and len(node.targets) == 1:
                    return getattr(node.targets[0], 'id', None)
                return getattr(node, 'name', None)
            node = next(n for n in tree.body if name(n) == entry['symbol'])
            actual = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
            self.assertEqual(actual, entry['syntax_sha256'], entry['symbol'])

    def test_classifiers_and_shots_saves_exclusion(self):
        rng = np.random.default_rng(21)
        rows = []
        for match in range(1200):
            signal = rng.normal()
            winner = int(rng.random() < 1 / (1 + np.exp(-signal)))
            for side, team in enumerate(['blue', 'orange']):
                rows.append(dict(replay_id=f'match-{match:06d}',
                    player_id=f'person-{(match + 200 * side) % 400}', team=team,
                    playlist='ranked-duels',
                    date=(pd.Timestamp('2026-01-01', tz='UTC') + pd.Timedelta(minutes=match)).isoformat(),
                    won=winner if side == 0 else 1-winner,
                    average_speed=1400 + 70 * signal * (1 if side == 0 else -1) + rng.normal(0, 30),
                    behind_ball_percent=float(rng.uniform(35, 75)),
                    shots_per5=int(rng.integers(1, 10)), saves_per5=int(rng.integers(0, 6))))
        frame = pd.DataFrame(rows)
        with threadpool_limits(limits=2):
            full = fit_models(frame, 'fabricated-test')
            style = fit_models(frame, 'fabricated-test', without_shots_saves=True)
        for result in [full, style]:
            self.assertTrue(result['available'])
            self.assertEqual(len(result['models']), 3)
            self.assertEqual(result['train_matches'], 720)
            self.assertEqual(result['validation_matches'], 240)
            self.assertEqual(result['test_matches'], 240)
            for model in result['models']:
                for metric in ['test_auc', 'validation_auc', 'accuracy', 'brier']:
                    self.assertTrue(0 <= model[metric] <= 1)
            json.dumps(result, allow_nan=False)
        self.assertIn('shots_per5', full['features'])
        self.assertNotIn('shots_per5', style['features'])
        self.assertNotIn('saves_per5', style['features'])
