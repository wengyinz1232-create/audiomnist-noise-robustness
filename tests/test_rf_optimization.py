import tempfile
import unittest
from pathlib import Path
import numpy as np
import rf_depth_diagnostic as rf


class RFOptimizationTests(unittest.TestCase):
    def test_shared_active_depth_default(self):
        self.assertEqual(rf.DEFAULT_GRID,[4,8,16])
        self.assertIn('exploratory',rf.GRID_ORIGIN)

    def test_invalid_grids(self):
        for grid in ([6,12], [6,6,12], [12,6,24], [0,6,12]):
            with self.assertRaises(ValueError):
                rf.validate_grid(grid)

    def test_outer_data_does_not_select_depth_and_artifacts_reload(self):
        rng=np.random.default_rng(42)
        groups=np.repeat(np.arange(12),20)
        y=np.tile(np.arange(10),24)
        x=rng.normal(size=(240,8)).astype('float32')
        x[:,0]+=y
        variants={c:x+rng.normal(0,.1,x.shape).astype('float32') for c in rf.CONDITIONS}
        train=np.arange(200); test=np.arange(200,240)
        with tempfile.TemporaryDirectory() as tmp:
            first=rf.train_fold(variants,y,groups,'logmel','augmented',[2,4,6],1,train,test,Path(tmp)/'a')
            altered={c:v.copy() for c,v in variants.items()}
            for v in altered.values(): v[test]*=100
            changed=y.copy(); changed[test]=(changed[test]+3)%10
            second=rf.train_fold(altered,changed,groups,'logmel','augmented',[2,4,6],1,train,test,Path(tmp)/'b')
            self.assertEqual(first['inner_scores'],second['inner_scores'])
            self.assertEqual(first['selected'],second['selected'])
            self.assertFalse(set(first['train_people']) & set(first['test_people']))
            for limit,details in zip([2,4,6],first['inner_tree_diagnostics']):
                self.assertTrue(all(max(d['actual_depths'])<=limit for d in details))
            summary=rf.summarize([first],[2,4,6])
            self.assertEqual(summary['status'],'PARTIAL')
            self.assertIsNone(summary['middle_unique_mode'])
            self.assertIsNone(summary['metrics']['pink_10']['macro_f1']['fold_sd'])


if __name__=='__main__': unittest.main()
