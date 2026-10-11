AudioMNIST teammate handoff: code, raw audio, feature caches and saved experiments.
Install dependencies: python -m pip install -r requirements.txt
Current result index (no training): python build_experiment_index.py
Start with results/current/status.json and summary_current_12.csv.
All 20 complete runs: results/current/catalog.json and summary_all_runs.csv.
Inner-only and stopped diagnostics are listed separately in diagnostics.json.
Historical results/summary.json remains unchanged by the index builder.
The legacy summarize command overwrites that historical summary; prefer the index builder.
The current 12-slot view replaces the MFCC logistic and LightGBM runs. All 12
selected configurations pass the unique-middle-mode check; original failures remain historical.
New LightGBM fixes 7 leaves and 70 rounds, tuning shared reg_lambda=[0,0.1,1].
Full L2 rerun entry: python rerun_lightgbm_l2.py
Existing selected_experiments figures predate the L2 update; prefer results/current.
New logistic grids differ by training mode; compare separately tuned pipelines.
Reproduce new logistic runs: python rerun_logistic_selected_grids.py
Verify a saved run: python verify_saved_run.py PATH_TO_RESULT_JSON --out AUDIT.json
Original main comparisons: python run_pipeline.py compare
Original MFCC ensemble: python run_pipeline.py ensemble
Those commands still use runs/main and do NOT incorporate the new logistic models.
The two-leaf LightGBM clean alternative is not a matched pair with original augmented LightGBM.
To retrain the original design, use a NEW run ID:
python run_pipeline.py train --run-id rerun_latest --jobs 3
Prepare absent caches: python run_pipeline.py prepare
Download an absent raw archive: python run_pipeline.py prepare --download
Do not resume historical main with changed source fingerprints. See README.md.
No independent confirmation claim: data and grids have been explored previously.
Build a code-only archive: python package_code.py
The archive does not include data, caches, models or results. Do not submit the whole folder.
