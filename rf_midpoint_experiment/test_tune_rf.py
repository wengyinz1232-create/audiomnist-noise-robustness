import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import tune_rf as rf

class ProtocolTests(unittest.TestCase):
    def test_grid_selection_preserves_best_scores_when_feasible(self):
        from select_config_grids import select
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for rep in ['logmel','mfcc_delta']:
                for mode in ['clean_matched','augmented']:
                    name=f'{rep}__forest__{mode}'
                    rows=([[.5,.9,.8,.7]]*2+[[.5,.8,.9,.7]]*5+[[.5,.7,.8,.9]]*3
                          if name=='logmel__forest__clean_matched' else [[.5,.9,.8,.7]]*10)
                    for fold,row in enumerate(rows,1):
                        rf.save(root/name/f'fold_{fold:02}'/'scores.json',dict(fold=fold,values=[.01,.1,.3,1.0],means=row))
            result=select(root)
            self.assertEqual(result['status'],'EXPLORATORY_FOUR_CONFIGURATIONS_PASS')
            special=result['configurations']['logmel__forest__clean_matched']['recommended']
            self.assertEqual(special['grid'],[.1,.3,1.0])
            self.assertEqual(special['audit']['counts'],[2,5,3])
            self.assertTrue(all(x['recommended']['mean_loss_vs_full_screen']==0 for x in result['configurations'].values()))
            (root/'mfcc_delta__forest__augmented'/'fold_10'/'scores.json').unlink()
            with self.assertRaises(ValueError):select(root)

    def test_configuration_specific_grids_are_explicit(self):
        from train_rf_midpoint import local_config,validate_config
        grids={f'{r}__forest__{m}':[.01,.1,.3] for r in ['logmel','mfcc_delta'] for m in ['clean_matched','augmented']}
        grids['logmel__forest__clean_matched']=[.1,.3,1.0]
        config=dict(parameter='max_features',grids=grids)
        validate_config(config)
        self.assertEqual(local_config(config,'logmel','clean_matched')['grid'],[.1,.3,1.0])
        self.assertEqual(local_config(config,'mfcc_delta','augmented')['grid'],[.01,.1,.3])
        incomplete=dict(config,grids={k:v for k,v in grids.items() if k!='mfcc_delta__forest__augmented'})
        with self.assertRaises(ValueError): validate_config(incomplete)

    def test_midpoint_runner_never_forces_middle(self):
        from train_rf_midpoint import run,candidate,validate_config
        config=dict(parameter='max_features',grid=[.01,.1,1.0],fixed=dict(max_depth=4,min_samples_leaf=2,n_estimators=5))
        validate_config(config)
        with self.assertRaises(ValueError):
            validate_config(dict(config,grid=[.1,.01,1.0]))
        rng=np.random.default_rng(77)
        groups=np.repeat(np.arange(60),20); y=np.arange(1200)%10
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); cache=root/'cache'; cache.mkdir()
            np.savez(cache/'logmel.npz',y=y,groups=groups,**{c:rng.normal(size=(1200,8)).astype('float32') for c in rf.CONDITIONS})
            screen=root/'screen'; folder=screen/'logmel__forest__clean_matched'/'fold_01'
            for value,s in zip(config['grid'],[.9,.8,.7]):
                p=candidate(config,value)
                rf.save(folder/(rf.key(p)+'.json'),dict(params=p,inner=[dict(score=s)]*3))
            result=run(str(cache),str(root/'out'),config,'logmel','clean_matched',1,str(screen))
            self.assertEqual(result['selected'],.01)
            np.testing.assert_allclose(result['inner_means'],[.9,.8,.7],atol=1e-15,rtol=0)
            self.assertTrue(result['reload_verified'])

    def test_midpoint_audit_rejects_plateau_as_natural_peak(self):
        from audit_midpoints import audit
        got=audit([[.5,.9,.9]]*10)
        self.assertEqual(got['counts'],[0,10,0])
        self.assertEqual(got['strict_middle_wins'],0)
        self.assertEqual(got['middle_tie_folds'],list(range(1,11)))
        got=audit([[.9,.8,.7]]*3+[[.8,.9,.7]]*4+[[.7,.8,.9]]*3)
        self.assertEqual(got['counts'],[3,4,3])
        self.assertTrue(got['middle_unique_mode'])
        self.assertEqual(got['middle_tie_folds'],[])

    def test_outer_test_changes_cannot_affect_selection_inputs(self):
        rng=np.random.default_rng(11)
        groups=np.repeat(np.arange(60),20); y=np.arange(1200)%10
        variants={c:rng.normal(size=(1200,8)).astype('float32') for c in rf.CONDITIONS}
        _,te=next(rf.ev.grouped_folds(groups,np.arange(1200),10,rf.SEED))
        captured=[]
        class Captured(Exception): pass
        def capture(prepared,*args):
            captured.append(copy.deepcopy(prepared))
            raise Captured()
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)
            for changed in (False,True):
                yy=y.copy(); data={c:x.copy() for c,x in variants.items()}
                if changed:
                    yy[te]=(yy[te]+1)%10
                    for c in data: data[c][te]=9999
                np.savez(folder/'logmel.npz',y=yy,groups=groups,**data)
                with patch.object(rf,'select_stages',side_effect=capture),self.assertRaises(Captured):
                    rf.run_fold(str(folder),str(folder/'output'),'logmel','augmented',1)
        for left,right in zip(*captured):
            for i in (0,1,3): np.testing.assert_array_equal(left[i],right[i])
            for l,r in zip(left[2],right[2]): np.testing.assert_array_equal(l,r)

    def test_independent_probability_metrics(self):
        from summarize_rf import independent_metrics
        y=np.arange(10); p=np.eye(10)
        got=independent_metrics(y,p)
        self.assertEqual(got,dict(macro_f1=1.,accuracy=1.,macro_precision=1.,balanced_accuracy=1.,log_loss=0.))
        got=independent_metrics(y,np.full((10,10),.1))
        self.assertAlmostEqual(got['accuracy'],.1)
        self.assertAlmostEqual(got['macro_f1'],2/110)
        self.assertAlmostEqual(got['log_loss'],np.log(10))

    def test_prefix_matches_independent_forests(self):
        rng=np.random.default_rng(42)
        x=rng.normal(size=(150,12)).astype('float32'); y=np.arange(150)%10
        large,_,_=rf.fit(rf.params(8,1,0.3,240),x,y,123)
        for n in (60,120):
            small,_,_=rf.fit(rf.params(8,1,0.3,n),x,y,123)
            prefix=copy.copy(large); prefix.estimators_=large.estimators_[:n]; prefix.n_estimators=n
            np.testing.assert_allclose(small.predict_proba(x),prefix.predict_proba(x),atol=1e-12,rtol=1e-12)

    def test_nested_speaker_isolation(self):
        groups=np.repeat(np.arange(60),20); idx=np.arange(len(groups))
        tests=[]
        for f,(tr,te) in enumerate(rf.ev.grouped_folds(groups,idx,10,rf.SEED),1):
            tests.extend(te)
            self.assertEqual(len(set(groups[te])),6)
            for fit,valid in rf.ev.grouped_folds(groups,tr,3,rf.SEED+f):
                self.assertEqual(len(set(groups[fit])),36)
                self.assertEqual(len(set(groups[valid])),18)
                self.assertFalse(set(groups[fit])&set(groups[valid]))
                self.assertFalse(set(groups[tr])&set(groups[te]))
        self.assertEqual(sorted(tests),idx.tolist())

    def test_selection_incumbent_ties_and_known_maximum(self):
        ps=[rf.params(8),rf.params(16),rf.params(None)]
        cache={rf.key(p):dict(inner=[dict(score=s)]*3) for p,s in zip(ps,[.9,.9,.8])}
        self.assertEqual(rf.pick(ps,cache),ps[0])
        cache[rf.key(ps[2])]['inner']=[dict(score=.91)]*3
        self.assertEqual(rf.pick(ps,cache),ps[2])

    def test_staged_selection_no_test_input_and_resume(self):
        rng=np.random.default_rng(5)
        x=rng.normal(size=(100,8)).astype('float32'); y=np.arange(100)%10
        prepared=[(x,y,[x[:20],x[20:40]],y[:20])]*3
        # Use small trees solely for this workflow test; stage C remains real 240 trees.
        with tempfile.TemporaryDirectory() as d:
            with patch.object(rf,'STRUCTURES',[rf.params(4)]),patch.object(rf,'BASELINE',[rf.params(4)]):
                first=rf.select_stages(prepared,1,Path(d))
                with patch.object(rf,'fit',side_effect=AssertionError('resume unexpectedly refit')):
                    second=rf.select_stages(prepared,1,Path(d))
                self.assertEqual(first,second)
                self.assertGreaterEqual(np.mean([v['score'] for v in first[3][rf.key(first[1])]['inner']]),
                                        np.mean([v['score'] for v in first[3][rf.key(first[0])]['inner']]))

if __name__=='__main__':
    unittest.main()
