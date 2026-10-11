"""Freeze the agreed experiment selection and plot saved outer-fold results."""
import csv
import json
import os
from statistics import mean, stdev
from common import ROOT, CONDITIONS, sha, save_json
from build_experiment_index import build

OUT=ROOT/'results'/'selected_experiments'
NEW='logistic_selected_grids_20261011'


def main():
    build();OUT.mkdir(exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(OUT/'matplotlib-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    catalog=json.loads((ROOT/'results/current/catalog.json').read_text(encoding='utf-8'))
    strict=[('logmel','logistic'),('logmel','forest'),('logmel','lightgbm_goss'),('mfcc_delta','forest')]
    selected={f'main/{rep}__{model}__{mode}':'matched_grid_comparison' for rep,model in strict for mode in ('clean_matched','augmented')}
    selected.update({f'{NEW}/mfcc_delta__logistic__{mode}':'separately_tuned_logistic' for mode in ('clean_matched','augmented')})
    selected['alt_min_child/mfcc_delta__lightgbm_goss_min_child__clean_matched']='standalone_lightgbm_alternative'
    selected['main/mfcc_delta__lightgbm_goss__augmented']='standalone_lightgbm_original'
    records={};manifest=[];exclusions=[];metric_rows=[]
    for entry in catalog:
        if entry['id'] not in selected:
            exclusions.append({'id':entry['id'],'source':entry['source'],'original_role':entry['role'],'middle_unique_mode':entry['middle_unique_mode'],
                               'reason':'Historical reference, unresolved grid failure, or supplementary experiment outside agreed selection; retained in all-run catalog.'})
            continue
        assert entry['middle_unique_mode'],entry['id']
        path=ROOT/entry['source'];assert sha(path)==entry['sha256']
        d=json.loads(path.read_text());rows={r['fold']:r for r in d['folds']}
        assert len(rows)==len(d['folds'])==10 and set(rows)==set(range(1,11))
        records[entry['id']]=(d,rows)
        manifest.append({**entry,'selected_role':selected[entry['id']]})
        for condition in CONDITIONS:
            v=[rows[f]['metrics'][condition]['macro_f1'] for f in range(1,11)]
            metric_rows.append({'id':entry['id'],'role':selected[entry['id']],'condition':condition,'macro_f1_mean':mean(v),'fold_sample_sd':stdev(v)})
    assert len(manifest)==12 and len(exclusions)==len(catalog)-12
    def values(key,condition):return [records[key][1][f]['metrics'][condition]['macro_f1'] for f in range(1,11)]
    labels=['Clean','White 20 dB','White 10 dB','Pink 10 dB']
    colors=['#bf6b28','#2364a3'];titles={'logistic':'Logistic regression','forest':'Random forest','lightgbm_goss':'LightGBM-GOSS'}
    figures=[]
    def save(fig,name,caption):
        fig.text(.5,.015,caption,ha='center',va='bottom',fontsize=9)
        fig.tight_layout(rect=(0,.07,1,.94))
        for ext in ('png','pdf'):fig.savefig(OUT/f'{name}.{ext}')
        plt.close(fig);figures.append({'name':name,'caption':caption,'png':f'{name}.png','pdf':f'{name}.pdf'})
    def curve(ax,key,label,color):
        scores=[values(key,c) for c in CONDITIONS]
        ax.errorbar(range(4),[mean(v) for v in scores],yerr=[stdev(v) for v in scores],marker='o',capsize=3,color=color,label=label)
        ax.set_xticks(range(4),labels,rotation=15);ax.set_ylim(0,1.05);ax.set_ylabel('Macro-F1');ax.grid(axis='y',alpha=.2)
    paired=[]
    fig,axes=plt.subplots(2,2,figsize=(12,8));fig.suptitle('Matched-grid comparisons | unseen speakers | 10 outer folds',fontsize=15)
    for ax,(rep,model) in zip(axes.flat,strict):
        clean=f'main/{rep}__{model}__clean_matched';aug=f'main/{rep}__{model}__augmented'
        a,ar=records[aug];b,br=records[clean]
        assert a['config']['grid']==b['config']['grid']
        for f in range(1,11):
            assert ar[f]['test_people']==br[f]['test_people'] and ar[f]['inner_valid_people']==br[f]['inner_valid_people']
        curve(ax,clean,'Clean training',colors[0]);curve(ax,aug,'White-noise augmentation',colors[1])
        ax.set_title(('Log-Mel' if rep=='logmel' else 'MFCC + delta')+' | '+titles[model]);ax.legend(fontsize=8)
        for condition in CONDITIONS:
            diff=[x-y for x,y in zip(values(aug,condition),values(clean,condition))]
            paired.append({'representation':rep,'model':model,'condition':condition,'mean_difference':mean(diff),'paired_fold_sd':stdev(diff),'positive_folds':sum(x>0 for x in diff),'negative_folds':sum(x<0 for x in diff)})
    save(fig,'01_matched_robustness','Error bars: sample SD across folds, not confidence intervals. Same grid within each training-mode pair.')
    fig,axes=plt.subplots(1,2,figsize=(12,5));fig.suptitle('Augmentation effect | differences on the same speaker folds',fontsize=15)
    names=['Log-Mel / Logistic','Log-Mel / Forest','Log-Mel / LightGBM','MFCC + delta / Forest']
    for ax,condition in zip(axes,('clean','pink_10')):
        r=[next(x for x in paired if x['representation']==rep and x['model']==model and x['condition']==condition) for rep,model in strict]
        ax.errorbar([x['mean_difference'] for x in r],range(4),xerr=[x['paired_fold_sd'] for x in r],fmt='o',capsize=4,color=colors[1])
        ax.axvline(0,color='#777777',linestyle='--');ax.set_yticks(range(4),names);ax.invert_yaxis();ax.set_title('Clean recordings' if condition=='clean' else 'Pink noise 10 dB')
        ax.set_xlabel('Macro-F1: augmented minus clean training');ax.grid(axis='x',alpha=.2)
    save(fig,'02_matched_paired_effects','Error bars: sample SD of within-fold differences. Axis ranges differ between panels; folds are not independent.')
    fig,ax=plt.subplots(figsize=(10,5));fig.suptitle('Updated MFCC + delta logistic regression | separately tuned pipelines',fontsize=14)
    curve(ax,f'{NEW}/mfcc_delta__logistic__clean_matched','Clean training: C = [0.0001, 0.003, 0.01]',colors[0])
    curve(ax,f'{NEW}/mfcc_delta__logistic__augmented','Augmented: C = [0.01, 0.03, 0.1]',colors[1]);ax.legend(fontsize=9)
    save(fig,'03_updated_logistic','Different search grids; not an identical-grid augmentation-only comparison. Error bars: fold sample SD.')
    fig,axes=plt.subplots(1,2,figsize=(12,5));fig.suptitle('MFCC + delta LightGBM | two standalone configurations',fontsize=15)
    keys=['alt_min_child/mfcc_delta__lightgbm_goss_min_child__clean_matched','main/mfcc_delta__lightgbm_goss__augmented']
    for ax,key,title,color in zip(axes,keys,['Clean alternative: 2 leaves; tune min_child_samples','Original augmented model: tune num_leaves'],colors):
        curve(ax,key,title,color);ax.set_title(title,fontsize=10)
    save(fig,'04_lightgbm_standalone','Model configurations differ. Do not attribute the between-panel difference solely to augmentation. Bars: fold sample SD.')
    findings={'matched_pairs':4,'matched_pink_gain_positive_in_all_ten_folds':all(r['positive_folds']==10 for r in paired if r['condition']=='pink_10'),
              'matched_clean_effects':[{k:r[k] for k in ('representation','model','mean_difference','positive_folds','negative_folds')} for r in paired if r['condition']=='clean'],
              'limitations':['Selection is agreed scope, not a claim that all original configurations passed.','New logistic grids and earlier supplemental designs followed exploration of this data.','The LightGBM panels are standalone, not a matched causal comparison.','All noise is synthetic; fold SD is not an independent confidence interval.','Original ensemble is not recomputed using updated logistic models.']}
    save_json(OUT/'selection.json',{'selected':manifest,'not_selected_but_retained':exclusions,'figures':figures,'findings':findings})
    save_json(OUT/'paired_effects.json',paired)
    for name,rows in [('selection.csv',[{k:e[k] for k in ('id','source','selected_role','grid','selected_counts','middle_unique_mode')} for e in manifest]),('figure_data.csv',metric_rows),('paired_effects.csv',paired)]:
        with (OUT/name).open('w',newline='',encoding='utf-8-sig') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader()
            writer.writerows({k:json.dumps(v) if isinstance(v,list) else v for k,v in r.items()} for r in rows)
    print('SELECTION_COMPLETE',len(manifest),'configurations;',len(figures),'figures; all ten pink gains:',findings['matched_pink_gain_positive_in_all_ten_folds'])


if __name__=='__main__':main()
