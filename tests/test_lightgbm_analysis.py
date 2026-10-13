import copy
import itertools
import unittest
import analyze_lightgbm_search as analysis


def fixtures():
    grid=[dict(zip(analysis.PARAMETERS,v)) for v in itertools.product([.03,.08,.15],[70,150,300],[7,31,127])]
    docs=[]
    for key in sorted(analysis.EXPECTED):
        for fold in range(1,11):
            scores=[[.7+i*.001]*3 for i in range(27)]
            docs.append({'config':{'key':key,'grid':grid},'result':{'fold':fold,'grid':grid,
                'inner_scores':scores,'inner_means':[sum(s)/3 for s in scores],
                'selected':grid[-1],'seconds':10,'metrics':{'pink_10':{'macro_f1':.99}}}})
    return docs


class AnalysisTests(unittest.TestCase):
    def test_diagnosis_uses_no_outer_metrics_and_does_not_force_middle(self):
        docs=fixtures();report=analysis.analyze(docs)
        for group in report['configurations'].values():
            for parameter in group['parameters'].values():
                self.assertEqual(parameter['winner_counts'],[0,0,10])
                self.assertEqual(parameter['next_action'],'INVESTIGATE_HIGHER_RANGE')
                self.assertFalse(parameter['middle_strict_majority'])
        changed=copy.deepcopy(docs)
        for doc in changed:doc['result']['metrics']={'all_test_data':'corrupted and ignored'}
        self.assertEqual(report,analysis.analyze(changed))

    def test_partial_conflicting_and_forged_evidence_rejected(self):
        docs=fixtures()
        with self.assertRaises(ValueError):analysis.analyze(docs[:-1])
        self.assertEqual(analysis.analyze(docs[:-1],True)['status'],'PROVISIONAL')
        with self.assertRaises(ValueError):analysis.analyze(docs+[docs[0]])
        changed=copy.deepcopy(docs);changed[0]['result']['selected']=changed[0]['config']['grid'][13]
        with self.assertRaises(ValueError):analysis.analyze(changed)
        changed=copy.deepcopy(docs);changed[0]['config']['environment']='changed'
        with self.assertRaises(ValueError):analysis.analyze(changed)


if __name__=='__main__':unittest.main()
