"""Training-only depth screening on a fixed outer training split, never its test data."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import rf_depth_diagnostic as rf

DEPTHS = [4, 8, 12, 16, 20, 24, 32, 48]


def screen(variants, y, groups, indices, depths, mode, fold, emit=None):
    """Indices are the only rows this function may fit or validate on."""
    rows = []
    classes = np.arange(10)
    for j, (fit_idx, valid_idx) in enumerate(rf.ev.grouped_folds(groups, indices, 3, rf.SEED+fold)):
        x, yy = rf.ev.training_rows(variants, y, fit_idx, mode)
        stats = rf.ev.scaling(x)
        x = rf.ev.transform(x, stats)
        valid = {c: rf.ev.transform(variants[c][valid_idx], stats) for c in ('clean','white_10')}
        for depth in depths:
            model, detail = rf.fitted(depth, x, yy, rf.SEED+100*fold+j)
            scores = {c:rf.ev.measure(y[valid_idx],rf.ev.proba(model,v,classes),classes)['macro_f1']
                      for c,v in valid.items()}
            row = dict(inner_fold=j+1, depth=depth, objective=float(np.mean(list(scores.values()))),
                       condition_scores=scores, train_people=np.unique(groups[fit_idx]).tolist(),
                       valid_people=np.unique(groups[valid_idx]).tolist(), **detail)
            rows.append(row)
            if emit: emit(rows)
    return rows


def aggregate(rows, depths):
    summary=[]
    for depth in depths:
        entries=[r for r in rows if r['depth']==depth]
        if len(entries)!=3: continue
        actual=[d for r in entries for d in r['actual_depths']]
        leaves=[d for r in entries for d in r['leaf_counts']]
        summary.append(dict(depth=depth, inner_mean=float(np.mean([r['objective'] for r in entries])),
                            inner_sd=float(np.std([r['objective'] for r in entries],ddof=1)),
                            actual_depth_mean=float(np.mean(actual)),actual_depth_max=max(actual),
                            leaf_count_mean=float(np.mean(leaves)),
                            cap_fraction=float(np.mean([r['depth_limit_fraction'] for r in entries])),
                            fit_seconds_mean=float(np.mean([r['fit_seconds'] for r in entries]))))
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-dir',type=Path,required=True)
    p.add_argument('--work-dir',type=Path,default=rf.reproduce.REPO/'.rf-depth-screen')
    p.add_argument('--outer-fold',type=int,choices=range(1,11),default=1)
    p.add_argument('--depths',type=int,nargs='+',default=DEPTHS)
    a=p.parse_args()
    if len(a.depths)<3 or a.depths!=sorted(set(a.depths)) or min(a.depths)<1:
        raise ValueError('Need at least three distinct increasing positive depths.')
    cache=a.cache_dir.resolve(); work=a.work_dir.resolve()
    for protected in (cache,rf.reproduce.SOURCE,rf.reproduce.REPO/'reproduced_results'):
        if work==protected or protected in work.parents: raise ValueError('Use a separate output directory.')
    inventory=json.loads((cache/'inventory.json').read_text(encoding='utf-8'))
    for name,field in [('features.py','frontend_sha256'),('prepare_data.py','prepare_sha256')]:
        assert rf.reproduce.sha(rf.reproduce.SOURCE/name)==inventory['signature'][field]
    metadata=dict(protocol='training-only-depth-screen-v1',outer_fold=a.outer_fold,depths=a.depths,
                  runner_sha256=rf.reproduce.sha(__file__),rf_runner_sha256=rf.reproduce.sha(rf.__file__),
                  evaluation_sha256=rf.reproduce.sha(rf.reproduce.SOURCE/'evaluation.py'),
                  source_feature_hashes=inventory['cache_sha256'],seed=rf.SEED,
                  limitations=['Only selected outer training speakers are used.',
                               'Screening is exploratory; inner scores are not unbiased test estimates.',
                               'Do not claim midpoint compliance from this screen.',
                               'A global grid informed by this screen is not independent of all other outer folds.'])
    path=work/'protocol.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=metadata:
        raise ValueError('Protocol changed; use a new work directory.')
    rf.reproduce.save(path,metadata)
    for rep in ('logmel','mfcc_delta'):
        feature=cache/(rep+'.npz')
        assert rf.reproduce.sha(feature)==inventory['cache_sha256'][rep]
        with np.load(feature) as data:
            y,groups=data['y'],data['groups']
            # Do not even load pink/W20 arrays: neither is a screening objective.
            variants={c:data[c] for c in ('clean','white_10')}
        assert len(y)==12000 and len(np.unique(groups))==60
        train,test=list(rf.ev.grouped_folds(groups,np.arange(len(y)),10,rf.SEED))[a.outer_fold-1]
        # Physically remove outer-test rows before passing arrays to the screening function.
        variants={c:x[train] for c,x in variants.items()}; yy=y[train]; gg=groups[train]
        for mode in ('clean_matched','augmented'):
            dest=work/f'{rep}__{mode}.json'
            if dest.exists():
                old=json.loads(dest.read_text(encoding='utf-8'))
                if old.get('status')=='COMPLETE':
                    print('RESUME',rep,mode,flush=True); continue
            def emit(rows):
                rf.reproduce.save(dest,dict(status='PARTIAL',representation=rep,mode=mode,
                                           training_people=np.unique(gg).tolist(),rows=rows))
                print(rep,mode,'inner',rows[-1]['inner_fold'],'depth',rows[-1]['depth'],flush=True)
            rows=screen(variants,yy,gg,np.arange(len(yy)),a.depths,mode,a.outer_fold,emit)
            summary=aggregate(rows,a.depths)
            rf.reproduce.save(dest,dict(status='COMPLETE',representation=rep,mode=mode,
                                       training_people=np.unique(gg).tolist(),rows=rows,summary=summary))
            with dest.with_suffix('.csv').open('w',newline='',encoding='utf-8') as f:
                w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    print('TRAINING_ONLY_SCREEN_COMPLETE',flush=True)


if __name__=='__main__': main()
