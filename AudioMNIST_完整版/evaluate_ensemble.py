"""Evaluate a fixed equal-probability ensemble on saved outer-fold predictions.

This is a supplementary analysis, not a fourth tuned algorithm. It uses only
the three MFCC+delta augmented models with the same ten speaker test folds.
No model is loaded or fitted, and no ensemble weight is selected from scores.
"""
import argparse
import json
from pathlib import Path
from statistics import mean, stdev
import numpy as np
from common import ROOT, save_json
from evaluation import measure

MODELS = ('logistic', 'forest', 'lightgbm_goss')
CONDITIONS = ('clean', 'white_20', 'white_10', 'pink_10')


def evaluate(main_runs):
    records = {}
    for model in MODELS:
        folder = main_runs/f'mfcc_delta__{model}__augmented'
        doc = json.loads((folder/'result.json').read_text(encoding='utf-8'))
        assert doc['config']['representation'] == 'mfcc_delta'
        assert doc['config']['model'] == model and doc['config']['mode'] == 'augmented'
        assert len(doc['folds']) == 10
        grid = doc['config']['grid']
        counts = [sum(row['selected'] == value for row in doc['folds']) for value in grid]
        assert counts[1] > max(counts[0], counts[2]), f'Grid midpoint fails for {model}'
        records[model] = folder
    per_fold = []
    for fold in range(1,11):
        data = {}
        for model, folder in records.items():
            with np.load(folder/f'predictions_{fold:02}.npz') as archive:
                data[model] = {key: archive[key] for key in ('indices','y','classes',*CONDITIONS)}
        reference = data[MODELS[0]]
        for model in MODELS[1:]:
            for key in ('indices','y','classes'):
                assert np.array_equal(data[model][key], reference[key]), f'{key}, fold {fold}'
        for condition in CONDITIONS:
            individual = {model: measure(reference['y'], data[model][condition], reference['classes'])
                          for model in MODELS}
            averaged = np.mean([data[model][condition] for model in MODELS], axis=0)
            ensemble = measure(reference['y'], averaged, reference['classes'])
            per_fold.append({'fold':fold,'condition':condition,
                             'individual_macro_f1':{model:individual[model]['macro_f1'] for model in MODELS},
                             'ensemble_macro_f1':ensemble['macro_f1'],
                             'ensemble_accuracy':ensemble['accuracy'],
                             'ensemble_balanced_accuracy':ensemble['balanced_accuracy'],
                             'ensemble_confusion':ensemble['confusion']})
    summary = {}
    for condition in CONDITIONS:
        rows = [row for row in per_fold if row['condition'] == condition]
        ensemble_values = [row['ensemble_macro_f1'] for row in rows]
        baseline_values = [row['individual_macro_f1']['logistic'] for row in rows]
        differences = [a-b for a,b in zip(ensemble_values,baseline_values)]
        summary[condition] = {'ensemble_macro_f1_mean':mean(ensemble_values),
                              'ensemble_macro_f1_fold_sd':stdev(ensemble_values),
                              'logistic_macro_f1_mean':mean(baseline_values),
                              'ensemble_minus_logistic_mean':mean(differences),
                              'paired_difference_fold_sd':stdev(differences),
                              'positive_folds':sum(x>0 for x in differences),
                              'negative_folds':sum(x<0 for x in differences)}
    return {'method':'Unweighted arithmetic mean of three aligned out-of-fold probability vectors',
            'models':MODELS,'representation':'mfcc_delta','training_mode':'augmented',
            'summary':summary,'per_fold':per_fold,
            'limitations':'Post-hoc supplementary comparison on previously explored folds; not independent confirmation or a fourth formally tuned algorithm.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main-runs',type=Path,default=ROOT/'runs'/'main')
    parser.add_argument('--out',type=Path,default=ROOT/'results'/'ensemble_equal_weight.json')
    args = parser.parse_args()
    result = evaluate(args.main_runs)
    save_json(args.out,result)
    print('ENSEMBLE_COMPLETE',args.out)


if __name__ == '__main__': main()
