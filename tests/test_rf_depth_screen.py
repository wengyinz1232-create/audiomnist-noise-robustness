import unittest
import numpy as np
import diagnose_rf_depth as screen


class DepthScreenTests(unittest.TestCase):
    def test_excluded_rows_cannot_change_screen(self):
        rng=np.random.default_rng(13)
        groups=np.repeat(np.arange(9),10)
        y=np.tile(np.arange(10),9)
        variants={c:rng.normal(size=(90,4)).astype('float32') for c in ('clean','white_10')}
        idx=np.arange(60)
        a=screen.screen(variants,y,groups,idx,[2,3,4],'clean_matched',1)
        changed={c:x.copy() for c,x in variants.items()}
        for x in changed.values(): x[60:]=np.nan
        yy=y.copy(); yy[60:]=-999
        b=screen.screen(changed,yy,groups,idx,[2,3,4],'clean_matched',1)
        self.assertEqual([r['objective'] for r in a],[r['objective'] for r in b])
        self.assertEqual(len(a),9)
        for row in a:
            self.assertFalse(set(row['train_people']) & set(row['valid_people']))
            self.assertTrue(set(row['train_people']+row['valid_people']) <= set(range(6)))
        summary=screen.aggregate(a,[2,3,4])
        self.assertEqual(len(summary),3)
        self.assertTrue(all(r['actual_depth_max']<=r['depth'] for r in summary))


if __name__=='__main__': unittest.main()
