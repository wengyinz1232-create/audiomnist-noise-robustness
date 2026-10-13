"""Explore prospective three-point depth grids from INNER scores; never edit results.

This is a post-hoc exploratory grid screen, not an independent nested-CV result.
An eligible grid must have a unique middle mode in all FOUR complete configurations.
Middle selections tied to an endpoint do not count as evidence of a natural peak.
"""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
import tune_rf as rf

DEPTHS=[4,8,16,48,128,None]

def audit(means):
    means=np.asarray(means,dtype=float)
    assert means.shape==(10,3)
    winners=means.argmax(axis=1)
    counts=np.bincount(winners,minlength=3).tolist()
    middle_ties=[i+1 for i,row in enumerate(means) if winners[i]==1 and min(row[1]-row[0],row[1]-row[2])<=1e-12]
    return dict(counts=counts,middle_unique_mode=counts[1]>max(counts[0],counts[2]),
                middle_tie_folds=middle_ties,
                strict_middle_wins=int(np.sum((means[:,1]>means[:,0]+1e-12)&(means[:,1]>means[:,2]+1e-12))),
                selected_inner_mean=float(means.max(axis=1).mean()),
                means=means.tolist())

def screen(root):
    configurations=list(itertools.product(['logmel','mfcc_delta'],['clean_matched','augmented']))
    available={}; missing={}
    for rep,mode in configurations:
        name=f'{rep}__forest__{mode}'; rows=[]
        for fold in range(1,11):
            values=[]
            for depth in DEPTHS:
                p=root/name/f'fold_{fold:02}'/'candidates'/(rf.key(rf.params(depth))+'.json')
                if not p.exists(): break
                d=json.loads(p.read_text(encoding='utf-8'))
                assert d['params']==rf.params(depth)
                values.append(float(np.mean([x['score'] for x in d['inner']])))
            if len(values)==len(DEPTHS): rows.append(values)
        if len(rows)!=10: missing[name]=len(rows)
        available[name]=np.asarray(rows)
    if missing:
        return dict(status='INCOMPLETE_INNER_SCREEN',complete_fold_counts={k:len(v) for k,v in available.items()},
                    note='No midpoint success claim is allowed before all four configurations have ten complete inner results.')
    trials=[]
    for indices in itertools.combinations(range(len(DEPTHS)),3):
        detail={name:audit(rows[:,indices]) for name,rows in available.items()}
        compliant=all(x['middle_unique_mode'] and not x['middle_tie_folds'] for x in detail.values())
        trials.append(dict(grid=[DEPTHS[i] for i in indices],configurations=detail,compliant=compliant,
                           average_selected_inner_score=float(np.mean([x['selected_inner_mean'] for x in detail.values()]))))
    eligible=[t for t in trials if t['compliant']]
    eligible.sort(key=lambda t:t['average_selected_inner_score'],reverse=True)
    return dict(status='EXPLORATORY_MATCH_FOUND' if eligible else 'NO_COMMON_GRID_MEETS_CRITERION',
                recommended=eligible[0] if eligible else None,all_trials=trials,
                fixed=dict(n_estimators=60,min_samples_leaf=2,max_features='sqrt'),
                criterion='Middle is unique most frequent argmax in each of four configurations; no middle-endpoint ties.',
                limitation='Grid screened using previous inner results across outer partitions. This is exploratory and cannot independently validate the selected grid.')

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--results',type=Path,default=rf.ROOT/'results')
    p.add_argument('--output',type=Path,default=rf.ROOT/'midpoint_audit.json'); a=p.parse_args()
    result=screen(a.results); rf.save(a.output,result)
    print(json.dumps({k:v for k,v in result.items() if k!='all_trials'},ensure_ascii=False,indent=2))
