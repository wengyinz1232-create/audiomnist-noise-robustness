# LightGBM joint tuning

The current optimization experiment jointly searches learning rate `[0.03,0.08,0.15]`,
boosting rounds `[70,150,300]` and leaves `[7,31,127]`. The Cartesian product has
27 candidates, shared by Log-Mel/MFCC+delta and clean/augmented training.
GOSS rates stay at 0.2/0.1; minimum leaf samples 20, L2 1, L1 0.

Run **LightGBM joint tuning** in GitHub Actions on `main`. Adding/changing its own
workflow YAML on main also triggers a run. Forty jobs, at most eight concurrent,
each execute one outer speaker fold with a three-fold manual inner search.
The objective is mean clean/W10 macro-F1. P10 and outer test speakers never enter
selection. Actual argmax wins; ties use the deterministic Cartesian-product order.
This run uses 3,280 fits and forty outer models, without an all-data final fit.

Each fold reloads its model, recomputes all four condition predictions and metrics,
checks speaker splits, model settings, data/source fingerprints, and argmax selection.
The final job requires all forty fold artifacts and verifies predictions, metrics,
selection and one-time test coverage again. Models are checked on the runner;
pickle files are not uploaded. The evidence artifact includes fold scores,
probabilities, audits, mean/sample-SD metrics and marginal parameter counts.

The assignment midpoint target is assessed per parameter and configuration.
Unique mode and strict majority (at least six of ten folds) are separate fields.
Neither is forced. A computational PASS is not a claim that every midpoint target
was met. Boundaries may justify further exploratory grid revisions; all such
revisions must be disclosed, and the original results must remain traceable.
These data were explored previously, so this is not independent confirmation.

Local equivalent after `python reproduce.py prepare --work-dir runtime`:

```bash
python optimize_lightgbm.py fold --work-dir runtime --config logmel__lightgbm_goss_joint__clean_matched --fold 1
# Run all four configurations and folds 1 through 10 before aggregation.
python optimize_lightgbm.py summarize --work-dir runtime
```

The older one-dimensional LightGBM searches remain in `reproduce.py` for historical
comparisons. Use `optimize_lightgbm.py` and this workflow for the new joint search.
NGBoost paired comparisons keep their original frozen comparator protocol.

## Evidence-based iteration

Use `python analyze_lightgbm_search.py --evidence downloaded-artifacts --output diagnosis.json`
after downloading the fold ZIPs or the aggregate evidence. `--allow-partial` labels
early diagnostics as provisional; a full report requires all forty folds.
The analyzer checks audit hashes, protocol consistency, complete Cartesian grids
and genuine argmax selection. It reads only inner scores for tuning advice.
It reports marginal winner counts, best inner scores at each fixed value, losses
from restricting a parameter to its middle value, winning-score gaps and the
learning-rate/rounds interaction. Boundary dominance suggests further range
investigation; dispersed choices suggest checking interactions first. These are
diagnostics, not automatic grid replacements or independent test estimates.
