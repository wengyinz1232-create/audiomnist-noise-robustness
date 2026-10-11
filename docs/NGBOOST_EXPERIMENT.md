# NGBoost paired training experiment

This is an actual training entry point, separate from the original twelve configurations. Its scope is **MFCC+delta with clean + white-10 augmentation**, compared against the current seven-leaf LightGBM–GOSS configuration. Results do not cover clean-only training or Log-Mel NGBoost.

## Fixed protocol

- Same 12,000 recordings, seed 20261009, ten speaker-disjoint outer folds and three inner folds as the original experiment. Each outer fold has 54 training and six unseen test speakers.
- Both models select their parameter using the mean of clean and white-10 inner macro-F1. Pink noise and outer test labels never select a stage or parameter.
- NGBoost 0.5.11: ten-class categorical distribution, natural gradient, depth-3 regression trees, minimum leaf size 20, learning rate 0.03, row and column sampling 0.5. Tune stages over **50, 100, 200**.
- LightGBM 4.7.0: 70 rounds, seven leaves, learning rate 0.08, GOSS rates 0.2/0.1, minimum leaf size 20. Tune L2 over **0, 0.1, 1**, exactly as the existing MFCC configuration.
- The stage grid is a finite computational budget, not equal model capacity or proof of optimal NGBoost tuning. Do not adjust it using outer results. Report boundary selections and midpoint-audit failures honestly.
- For each NGBoost inner fold, train 200 stages once and evaluate the three prefixes. With fixed seeds and no early stopping this is equivalent to independently fitting each stage count; a test verifies this equivalence. Three trajectories plus one final refit use four actual fit calls per outer fold. LightGBM uses nine inner fits plus one refit. Both evaluate nine inner candidate scores.
- Odd folds run NGBoost first; even folds run LightGBM first. A paired fold runs sequentially on the same runner. Both have BLAS limited to one thread; LightGBM can use two tree-building threads, while this pinned NGBoost implementation fits base learners serially. Workflow times describe these implementations, not an intrinsic algorithmic speed ratio.
- Save and reload all twenty outer models, validate probabilities, save outer predictions and recompute metrics. No additional full-data deployment models are fitted. Keep the original twelve-configuration results intact.

## Run locally

Use a fresh Python 3.12 environment:

```bash
python -m pip install -r requirements-ngboost.txt
python -m unittest discover -s tests -v
python reproduce.py prepare --work-dir .ngboost-comparison
python compare_ngboost.py plan
python compare_ngboost.py fold --work-dir .ngboost-comparison --fold 1
# Repeat --fold 2 through --fold 10, then:
python compare_ngboost.py summarize --work-dir .ngboost-comparison
```

The fold command refuses to overwrite a completed fold. Failed folds leave `failure.json`; no summary is produced by silently dropping them. Use a fresh directory after changing source or protocol.

## Run on GitHub

Actions → **NGBoost paired comparison** → Run workflow. The workflow prepares features once, runs ten independent paired-fold jobs (at most ten concurrently), then validates and summarizes all folds. No paid service is enabled by the workflow. It consumes normal Actions time and stores evidence for 14 days.

The `ngboost-comparison-evidence` artifact contains `reports/summary.csv`, `paired_differences.csv`, `comparison.json`, `RESULTS.md`, plus per-fold results, predictions and models. Only load pickle files from your own trusted run. A failed job still attempts to upload partial evidence.

Final-fit time is separate from total workflow time. Workflow time includes search, scoring, refitting, outer evaluation, serialization and reload checks, but not shared feature extraction. Evaluation timing includes transforms and four batch predictions, not device-level per-request latency. Serialized model size is not peak RAM. Differences are NGBoost minus LightGBM; lower log loss is better. Fold SD and win counts are descriptive because the training sets overlap.

## Interpretation rules

Report measured performance even if it favors NGBoost or fails to show a clear difference. A small difference or mixed fold wins cannot establish universal superiority. A boundary-selected stage grid limits any claim that the NGBoost family itself is inferior. Poorer scores under this budget could reflect underfitting, settings or computation limits rather than an inherent defect. Check loss and probability behavior before claiming numerical instability. Neither log loss nor a categorical distribution alone proves calibration or out-of-distribution uncertainty quality.

Methods: [NGBoost, ICML 2020](https://proceedings.mlr.press/v119/duan20a.html), [NGBoost classification](https://stanfordmlgroup.github.io/ngboost/1-useage.html), [stage selection](https://stanfordmlgroup.github.io/ngboost/2-tuning.html), [LightGBM parameters](https://lightgbm.readthedocs.io/en/stable/Parameters.html).
