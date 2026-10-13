"""Select per-configuration three-point grids, retaining all exploratory evidence."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
import tune_rf as rf
from audit_midpoints import audit

def select(screen):
    values=[.01,.1,.3,1.0]; result={}
    for rep,mode in itertools.product(['logmel','mfcc_delta'],['clean_matched','augmented']):
        name=f'{rep}__forest__{mode}'; files=sorted((screen/name).glob('fold_*/scores.json'))
        if len(files)!=10: raise ValueError(f'{name}: all ten folds required')
        rows=[json.loads(p.read_text(encoding='utf-8')) for p in files]
        assert [r['fold'] for r in rows]==list(range(1,11))
        assert all(r['values']==values for r in rows)
        matrix=np.array([r['means'] for r in rows]); trials=[]
        for indices in itertools.combinations(range(4),3):
            check=audit(matrix[:,indices]); grid=[values[i] for i in indices]
            trials.append(dict(grid=grid,audit=check,
                               compliant=check['middle_unique_mode'] and not check['middle_tie_folds'],
                               mean_loss_vs_full_screen=float(np.mean(matrix.max(1)-matrix[:,indices].max(1)))))
        eligible=sorted([t for t in trials if t['compliant']],key=lambda t:(t['mean_loss_vs_full_screen'],t['grid']))
        result[name]=dict(recommended=eligible[0] if eligible else None,all_trials=trials,full_scores=matrix.tolist())
    return dict(status='EXPLORATORY_FOUR_CONFIGURATIONS_PASS' if all(v['recommended'] for v in result.values()) else 'NO_SOLUTION_FOR_SOME_CONFIGURATIONS',
                initial_common_space=values,configurations=result,
                selection_rule='Require a natural unique middle mode; among eligible triples choose lowest loss versus all four candidates, then ascending grid order.',
                limitation='Per-configuration grids are chosen post-hoc from inner scores on a previously explored corpus. They are not a common grid or independent confirmation.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--screen',type=Path,default=rf.ROOT/'feature_screen')
    p.add_argument('--prior',type=Path,default=rf.ROOT/'results');p.add_argument('--output',type=Path,default=rf.ROOT/'grid_selection')
    a=p.parse_args(); result=select(a.screen); rf.save(a.output/'selection_audit.json',result)
    if all(v['recommended'] for v in result['configurations'].values()):
        protocol=json.loads((a.prior/'protocol.json').read_text(encoding='utf-8'))
        config=dict(parameter='max_features',grids={k:v['recommended']['grid'] for k,v in result['configurations'].items()},
                    fixed=dict(max_depth=16,min_samples_leaf=2,n_estimators=60),
                    feature_sha256=protocol['feature_sha256'],source_sha256=protocol['sources'],
                    initial_common_space=result['initial_common_space'],grid_selection_audit_sha256=rf.sha(a.output/'selection_audit.json'),
                    scope='Exploratory per-configuration grid design followed by genuine within-fold argmax selection. No forced midpoint.')
        rf.save(a.output/'midpoint_config.json',config)
    print(result['status'])
    for k,v in result['configurations'].items():
        chosen=v['recommended']; print(k,None if chosen is None else (chosen['grid'],chosen['audit']['counts'],chosen['mean_loss_vs_full_screen']))
