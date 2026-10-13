# Current random-forest training

The existing RF implementation now tunes `max_features` directly. This applies to `AudioMNIST_完整版/evaluation.py`, `reproduce.py train/all`, the Full reproduction workflow, and the `optimize_rf.py` entry point.

Fixed settings: 60 bootstrap trees, maximum depth 16, minimum leaf size 2, Gini splitting, and minimum split size 2.

| Configuration | Feature-fraction grid | Recorded ten-fold selection counts |
|---|---|---|
| Log-Mel clean | `[0.1, 0.3, 1.0]` | `[2,5,3]` |
| Log-Mel augmented | `[0.01, 0.1, 0.3]` | `[0,10,0]` |
| MFCC+delta clean | `[0.01, 0.1, 0.3]` | `[0,10,0]` |
| MFCC+delta augmented | `[0.01, 0.1, 0.3]` | `[0,10,0]` |

`AudioMNIST_完整版/rf_settings.json` supplies grids and fixed settings to the main pipeline. Tests ensure these agree with the verified experiment configuration. Every outer fold still selects the actual highest clean/W10 inner macro-F1. Exact ties use candidate order; midpoint counts are observations, never forced choices. These grids were designed after exploration of the same folds and are not independent confirmation. The Log-Mel CT count is a unique mode, not a strict majority.

## Run through the existing pipeline

### GitHub Actions

Select **RF reproduction** in Actions and choose **Run workflow** on `main`.
The workflow prepares fresh features, trains the four RF configurations in parallel,
and verifies every saved model and prediction. Each configuration includes ten outer
folds plus the main pipeline's all-data tuning/refit (440 model fits in total).
Four `rf-evidence-*` artifacts retain inner scores, selections, outer metrics,
prediction probabilities, data fingerprints and verification records for 14 days.
Model pickle files are checked on the runner but are not included in these artifacts.
The workflow also runs when its own YAML file changes on `main`; unrelated pushes
do not start RF training. An artifact from a failed job can contain partial results;
confirm all four training jobs succeeded before treating the run as complete.

Use a fresh working directory because the source fingerprint and parameter meaning have changed:

```bash
python -m pip install -r requirements-reproduce.txt
python reproduce.py prepare --work-dir .reproduction-rf-current
python reproduce.py train --work-dir .reproduction-rf-current --config logmel__forest__clean_matched logmel__forest__augmented mfcc_delta__forest__clean_matched mfcc_delta__forest__augmented
python reproduce.py verify --work-dir .reproduction-rf-current --config logmel__forest__clean_matched logmel__forest__augmented mfcc_delta__forest__clean_matched mfcc_delta__forest__augmented
```

`python reproduce.py all --work-dir .reproduction-rf-current` trains all twelve model configurations using the updated RF settings. The main pipeline retains an additional all-data tuning/refit for each configuration. The previously verified RF experiment consisted of 40 outer models and had no all-data fit; changing the entry point does not retroactively create a new result batch.

## RF-only entry

```bash
python optimize_rf.py --cache-dir .reproduction-rf-current/data --output .rf-midpoint --workers 4
```

`optimize_rf_midpoint.py` remains an equivalent alias. These RF-only entry points run four configurations and produce the original verified experiment's result format. They do not add all-data models. The former depth-only `--depths`, `--representation`, `--mode` and `--fold` options belong to the explicitly historical `rf_depth_diagnostic.py`, documented in [the depth diagnostic notes](RF_DEPTH_DIAGNOSTIC.md).

The optional multi-SNR mode in the older pipeline inherits each representation's augmented grid, but was not part of the four-configuration midpoint result.

## Validation and evidence

Repository tests compare all four integrated RF configurations against the verified trainer on deterministic synthetic speaker-grouped data, including all candidate inner scores, chosen parameters, speaker partitions, saved models and probabilities under all four conditions. Configuration checks, staged-runtime checks and the separate RF protocol suite also run in Code checks.

Historical source snapshots, archived cloud results and the verified experimental evidence are retained as historical records. They are not the current training implementation and must not be relabeled as a fresh run. Previously generated runtimes must be regenerated in a new directory; a hash mismatch should not be bypassed.
