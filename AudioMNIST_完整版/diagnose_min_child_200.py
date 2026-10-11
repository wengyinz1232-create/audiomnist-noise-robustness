"""Inner-only diagnostic of [5, 20, 200]; fixed two-leaf, 70-round GOSS."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import json
import pickle
import time
from pathlib import Path
import numpy as np
from threadpoolctl import ThreadpoolController
from common import ROOT, SEED, sha, save_json
from evaluation import grouped_folds, training_rows, scaling, transform, fit, model_for, proba, measure

OUT=ROOT/'results'/'diagnosis_min_child_200_20261011'
GRID=[5,20,200]


def main():
    if os.name=='nt': ThreadpoolController._snapshot_loaded_modules=lambda self,kernel:None
    OUT.mkdir(exist_ok=True)
    signatures={p.name:sha(p) for p in [Path(__file__),ROOT/'evaluation.py',ROOT/'common.py',ROOT/'features.py',ROOT/'data'/'mfcc_delta.npz']}
    sources={m:ROOT/'runs'/'alt_min_child'/f'mfcc_delta__lightgbm_goss_min_child__{m}'/'result.json' for m in ('clean_matched','augmented')}
    plan={'grid':GRID,'signature':signatures,'reference_hashes':{m:sha(p) for m,p in sources.items()},
          'fixed_settings':'model_for(lightgbm_goss_min_child): num_leaves=2, n_estimators=70, learning_rate=.08; unchanged other settings',
          'objective':'mean(clean macro-F1, white_10 macro-F1)',
          'scope':'Exploratory inner-only diagnostic; no outer-test scoring; stop after this grid regardless of outcome.',
          'reuse_validation':'Cache/common/frontend hashes, speaker folds, every saved outer-model hash and constructor parameters, plus fresh CANDIDATE 20 anchor fit in first inner fold of each mode. Historical evaluation.py hash differs, so no claim of complete historical source identity.'}
    if (OUT/'plan.json').exists(): assert json.loads((OUT/'plan.json').read_text())==plan
    save_json(OUT/'plan.json',plan)
    with np.load(ROOT/'data'/'mfcc_delta.npz') as data:
        y,groups=data['y'],data['groups'];variants={c:data[c] for c in ('clean','white_10')}
    classes=np.unique(y);outer=list(grouped_folds(groups,np.arange(len(y)),10,SEED));summaries={}
    for mode,source in sources.items():
        old=json.loads(source.read_text());config=old['config'];refs={r['fold']:r for r in old['folds']}
        assert config['grid']==[5,20,80] and len(old['folds'])==10 and set(refs)==set(range(1,11))
        assert config['seed']==SEED and config['samples']==len(y)
        for key in ('mfcc_delta.npz','common.py','features.py'): assert config['signature'][key]==signatures[key]
        path=OUT/f'{mode}.json';doc=json.loads(path.read_text()) if path.exists() else {'plan':plan,'rows':[],'reuse_checks':[]}
        assert doc['plan']==plan
        for number,(train,_) in enumerate(outer,1):
            ref=refs[number];inner=list(grouped_folds(groups,train,3,SEED+number))
            assert ref['train_people']==np.unique(groups[train]).tolist()
            assert ref['inner_valid_people']==[np.unique(groups[v]).tolist() for _,v in inner]
            modelpath=source.parent/'models'/f'fold_{number:02}.pkl'
            assert sha(modelpath)==ref['model_sha256']
            with modelpath.open('rb') as stream: saved=pickle.load(stream)['estimator']
            assert saved.get_params()==model_for('lightgbm_goss_min_child',ref['selected'],SEED+number).get_params()
            for j,(tr,va) in enumerate(inner):
                if any(r['fold']==number and r['inner']==j and r['value']==200 for r in doc['rows']):continue
                xx,yy=training_rows(variants,y,tr,mode);stats=scaling(xx);xx=transform(xx,stats)
                xv=[transform(variants[c][va],stats) for c in ('clean','white_10')]
                seed=SEED+100*number+j
                if number==1 and j==0:
                    check,w=fit('lightgbm_goss_min_child',20,xx,yy,seed)
                    score=float(np.mean([measure(y[va],proba(check,z,classes),classes)['macro_f1'] for z in xv]))
                    difference=abs(score-ref['inner_scores'][1][j]);assert difference<1e-12
                    doc['reuse_checks'].append({'fold':number,'inner':j,'value':20,'absolute_score_difference':difference,'warnings':w})
                start=time.perf_counter();model,w=fit('lightgbm_goss_min_child',200,xx,yy,seed)
                scores=[measure(y[va],proba(model,z,classes),classes)['macro_f1'] for z in xv]
                for k,v in enumerate(GRID[:2]):
                    doc['rows'].append({'fold':number,'inner':j,'value':v,'score':ref['inner_scores'][k][j],'source':'saved alt_min_child inner score'})
                doc['rows'].append({'fold':number,'inner':j,'value':200,'score':float(np.mean(scores)),'condition_scores':scores,'source':'new fit','warnings':w,'seconds':time.perf_counter()-start})
                save_json(path,doc);print('DONE',mode,'fold',number,'inner',j+1,flush=True)
        assert len(doc['rows'])==90 and len({(r['fold'],r['inner'],r['value']) for r in doc['rows']})==90
        means=[[float(np.mean([r['score'] for r in doc['rows'] if r['fold']==f and r['value']==v])) for v in GRID] for f in range(1,11)]
        chosen=[int(np.argmax(s)) for s in means];counts=[chosen.count(k) for k in range(3)]
        summary={'grid':GRID,'counts':counts,'middle_unique':counts[1]>max(counts[0],counts[2]),'fold_means':means,'mean_scores':np.mean(means,axis=0).tolist(),'exact_winner_ties':sum(s.count(max(s))>1 for s in means),'warnings_new_fits':sum(len(r.get('warnings',[])) for r in doc['rows'])}
        doc['summary']=summary;save_json(path,doc);summaries[mode]=summary
        print('COMPLETE',mode,counts,flush=True)
    save_json(OUT/'summary.json',{'status':'complete_inner_diagnostic','configurations':summaries,'new_candidate_fits':60,'anchor_verification_fits':2,'reused_inner_scores':120,'outer_test_evaluations':0,'both_middle_unique':all(s['middle_unique'] for s in summaries.values()),'limitations':'Exploratory grid after prior results. Not a full outer rerun or independent confirmation; historical source identity not fully proven.'})


if __name__=='__main__':main()
