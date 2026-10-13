# RF depth search with capacity diagnostics

This document describes the earlier depth diagnostic. The current four-configuration
midpoint experiment uses `optimize_rf_midpoint.py` and explicit per-configuration
`max_features` grids. See `rf_midpoint_experiment/使用说明.txt` and the README.
Its grids were selected post-hoc from inner-score exploration and must be reported
as exploratory; the locked-grid runner always chooses the actual score maximum.

`optimize_rf.py` is a new experiment, not a replacement for the archived
`[4,48,128]` reproduction. The default `[4,8,16]` was chosen after a training-only
screen of `[4,8,12,16,20,24,32,48]` on outer fold 1's 54 training speakers,
using three inner folds and all four representation/mode combinations.
These three limits actively restrict tree growth and produce distinct inner
scores. This defines a capacity-controlled RF, not a claim to maximise RF
performance over unrestricted depth. Neither universal optimality nor course
midpoint compliance has been established.
Sixty trees, minimum leaf size 2, square-root feature sampling, seeds, features,
training modes and the manual 10-by-3 speaker-disjoint protocol are unchanged.
Use the same grid for both representations and modes. Do not modify the grid
based on outer test scores or force midpoint selection by breaking ties differently.

Install `requirements-reproduce.txt`. Create features with the existing command:

```bash
python reproduce.py prepare --work-dir .reproduction
python optimize_rf.py --cache-dir .reproduction/data --representation mfcc_delta --mode augmented --fold 1
```

The second command is a single-fold smoke run, not a complete evaluation.
To finish all four RF configurations (run sequentially):

```bash
python optimize_rf.py --cache-dir .reproduction/data --representation logmel --mode clean_matched
python optimize_rf.py --cache-dir .reproduction/data --representation logmel --mode augmented
python optimize_rf.py --cache-dir .reproduction/data --representation mfcc_delta --mode clean_matched
python optimize_rf.py --cache-dir .reproduction/data --representation mfcc_delta --mode augmented
```

Results default to `.rf-depth-v3/`, separated from historical outputs.
`--depths 4 8 16` specifies an explicit ascending grid. Changed grids or sources
require a new `--work-dir`; identical runs resume after hash checks. Do not run
multiple processes against the same configuration directory.

Each fold records all inner scores, each fitted tree's actual depth/leaf count,
fraction reaching the depth cap, fit seconds, selected depth and near-best
ties (1e-12 diagnostic tolerance; selection itself uses exact argmax). Selected
models are reloaded and their probabilities checked on all four test conditions.
Saved predictions and models have SHA-256 hashes. Summary reports partial/full
status, selection counts and midpoint status only when all ten folds exist.
One-fold runs have no sample SD. Four complete configurations use 400 actual
fits and yield 40 outer models; no full-data deployment fit is performed.

Check whether candidate depths actually change inner validation scores, whether
trees reach their limits, and whether midpoint selection reflects distinct scores
rather than a plateau. Boundary selection or ties must be reported, not hidden.
The screen selected 16/16/8/16 for Log-Mel CT/AT and MFCC CT/AT respectively
within this grid, so three configurations still prefer the upper boundary.
Do not describe these data as satisfying the midpoint criterion. In MFCC AT,
16 and 24 differ by only 0.000116 in mean inner F1; in both CT configurations,
24/32/48 have identical scores and tree summaries. Larger depth alone is
unlikely to create a well-separated, central optimum across every configuration.
If midpoint compliance is essential, a separately specified regularisation
parameter (e.g. minimum leaf size) may be a better tuning target; it is not
changed here.

To repeat the screening without fitting or scoring outer test recordings:

```bash
python diagnose_rf_depth.py --cache-dir .reproduction/data --work-dir .rf-depth-screen
```

The screen only loads clean/W10 features and removes outer-test rows before
passing arrays into its inner loop. Scores are exploratory inner-validation
scores, not test results or independent confirmation. Completed configurations
can resume; interrupted configurations restart their screening. Use a new work
directory if sources or the protocol change. Applying a grid informed by fold 1
to all ten folds is still exploratory, because its screening speakers include
test speakers of other folds. For independent evaluation, lock the protocol and
use new external speakers, or perform the entire screening within each outer
training split. A global pilot must not be described as fully nested selection.

Reusing this previously explored corpus is exploratory. New report results require
complete runs and metric verification; the old RF result tables remain historical.

```bash
python -m unittest discover -s tests -v
```
