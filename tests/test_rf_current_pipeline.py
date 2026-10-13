"""Ensure existing RF entry points actually use the verified new experiment."""
import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import reproduce

sys.path.insert(0, str(reproduce.SOURCE))
import evaluation as ev
sys.path.insert(0, str(reproduce.REPO / 'rf_midpoint_experiment'))
import train_rf_midpoint as midpoint


class CurrentRFPipelineTests(unittest.TestCase):
    def test_all_entry_points_share_verified_grids_and_model_parameters(self):
        config=json.loads((reproduce.REPO/'rf_midpoint_experiment/midpoint_config.json').read_text(encoding='utf-8'))
        for rep in ('logmel','mfcc_delta'):
            for mode in ('clean_matched','augmented'):
                key=f'{rep}__forest__{mode}'
                grid=config['grids'][key]
                self.assertEqual(reproduce.CONFIGS[key]['grid'],grid)
                self.assertEqual(ev.parameter_grid(rep,'forest',mode),grid)
                for value in grid:
                    expected=midpoint.candidate(midpoint.local_config(config,rep,mode),value)
                    params=ev.model_for('forest',value,123).get_params()
                    self.assertEqual({name:params[name] for name in expected},expected)
        with self.assertRaises(ValueError):ev.model_for('forest',48,123)
        self.assertEqual(ev.parameter_grid('logmel','forest','multi_snr_augmented'),[.01,.1,.3])
        import optimize_rf, optimize_rf_midpoint
        self.assertIs(optimize_rf.main,optimize_rf_midpoint.main)

    def test_current_fold_matches_verified_runner_for_all_four_configurations(self):
        rng=np.random.default_rng(73)
        groups=np.repeat(np.arange(12),20)
        y=np.tile(np.arange(10),24)
        clean=rng.normal(size=(240,12)).astype('float32')
        clean[:,0]+=y
        variants={c:clean+rng.normal(0,.1,clean.shape).astype('float32') for c in ev.CONDITIONS}
        config=json.loads((reproduce.REPO/'rf_midpoint_experiment/midpoint_config.json').read_text(encoding='utf-8'))
        train,test=list(ev.grouped_folds(groups,np.arange(len(y)),10,ev.SEED))[0]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); cache=root/'data';cache.mkdir()
            for rep in ('logmel','mfcc_delta'):
                np.savez_compressed(cache/f'{rep}.npz',y=y,groups=groups,**variants)
                for mode in ('clean_matched','augmented'):
                    key=f'{rep}__forest__{mode}'
                    dest=root/'integrated'/key;(dest/'models').mkdir(parents=True)
                    integrated=ev.train_fold(variants,y,groups,'forest',mode,config['grids'][key],1,train,test,dest,rep)
                    reference=midpoint.run(cache,root/'reference',config,rep,mode,1,None)
                    self.assertEqual(integrated['selected'],reference['selected'])
                    np.testing.assert_allclose(integrated['inner_scores'],reference['inner_scores'],atol=1e-12,rtol=0)
                    self.assertEqual(integrated['train_people'],reference['train_people'])
                    self.assertEqual(integrated['inner_valid_people'],reference['inner_valid_people'])
                    with np.load(dest/'predictions_01.npz') as actual, np.load(root/'reference'/key/'fold_01'/'predictions.npz') as expected:
                        for c in ev.CONDITIONS:
                            np.testing.assert_allclose(actual[c],expected[c],atol=1e-12,rtol=0)
                    with (dest/'models/fold_01.pkl').open('rb') as f:bundle=pickle.load(f)
                    self.assertEqual(bundle['estimator'].max_depth,16)
                    self.assertIsInstance(bundle['estimator'].max_features,float)


if __name__=='__main__':unittest.main()
