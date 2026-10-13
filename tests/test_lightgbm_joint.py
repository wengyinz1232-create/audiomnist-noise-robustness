"""Protocol tests for multi-parameter selection and saved evidence."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import reproduce
import optimize_lightgbm as joint
sys.path.insert(0,str(reproduce.SOURCE))
import evaluation as ev


class JointLightGBMTests(unittest.TestCase):
    def test_grid_and_model_contract(self):
        self.assertEqual(len(joint.GRID),27)
        self.assertEqual(len({tuple(v.items()) for v in joint.GRID}),27)
        for value in joint.GRID:
            params=ev.model_for('lightgbm_goss_joint',value,123).get_params()
            for k,v in {**value,**joint.PLAN['fixed']}.items():self.assertEqual(params[k],v)
            self.assertEqual(params['random_state'],123)
        self.assertEqual(len(joint.KEYS),4)

    def test_real_fold_roundtrip_and_selection_leakage(self):
        rng=np.random.default_rng(21)
        groups=np.repeat(np.arange(20),10);y=np.tile(np.arange(10),20)
        clean=rng.normal(size=(200,8)).astype('float32');clean[:,0]+=y
        variants={c:clean+rng.normal(0,.1,clean.shape).astype('float32') for c in ev.CONDITIONS}
        grid=[{'learning_rate':.08,'n_estimators':n,'num_leaves':7} for n in (2,3,4)]
        tr,te=list(ev.grouped_folds(groups,np.arange(len(y)),10,ev.SEED))[0]
        with tempfile.TemporaryDirectory() as tmp,patch.object(joint,'GRID',grid):
            for mode in ('clean_matched','augmented'):
                folder=Path(tmp)/mode
                config={'seed':ev.SEED,'representation':'logmel','mode':mode}
                with patch.object(ev,'fit',wraps=ev.fit) as fit:
                    row=ev.train_fold(variants,y,groups,'lightgbm_goss_joint',mode,grid,1,tr,te,folder,'logmel')
                    self.assertEqual(fit.call_count,10)
                audit=joint.verify_fold(reproduce.SOURCE,folder,config,row,variants,y,groups)
                self.assertEqual(audit['status'],'PASS')
                changed={k:v.copy() for k,v in variants.items()}
                changed['pink_10'][:]=999
                for condition in ev.CONDITIONS:changed[condition][te]=-999
                again=ev.train_fold(changed,y,groups,'lightgbm_goss_joint',mode,grid,1,tr,te,Path(tmp)/(mode+'-changed'),'logmel')
                self.assertEqual(row['inner_scores'],again['inner_scores'])
                self.assertEqual(row['selected'],again['selected'])
                bad=copy.deepcopy(row);bad['inner_scores'][0][0]+=100
                with self.assertRaises(AssertionError):
                    joint.verify_fold(reproduce.SOURCE,folder,config,bad,variants,y,groups)

    def test_marginal_midpoints_are_not_forced(self):
        low=joint.GRID[0]; middle={k:v[1] for k,v in joint.PLAN['search'].items()}
        rows=[{'selected':low}]*5+[{'selected':middle}]*5
        for item in joint.selection_counts(rows).values():
            self.assertFalse(item['middle_unique_mode'])
            self.assertFalse(item['middle_strict_majority'])
        rows=[{'selected':low}]*4+[{'selected':middle}]*6
        for item in joint.selection_counts(rows).values():
            self.assertTrue(item['middle_unique_mode'])
            self.assertTrue(item['middle_strict_majority'])

    def test_summary_rejects_missing_fold_evidence(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(reproduce,'check_data'),patch.object(joint,'protocol',return_value={'representation':'logmel'}):
            root=Path(tmp);(root/'data').mkdir()
            np.savez(root/'data/logmel.npz',y=np.arange(10),groups=np.arange(10),
                     **{c:np.zeros((10,2)) for c in ev.CONDITIONS})
            with self.assertRaises(FileNotFoundError):joint.summarize(root)


if __name__=='__main__':unittest.main()
