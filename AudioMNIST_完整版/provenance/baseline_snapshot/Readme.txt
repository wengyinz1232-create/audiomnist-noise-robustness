Python 3.12 recommended. Install: python -m pip install -r requirements.txt
Run from this folder: python run_pipeline.py all --jobs 3
Missing source archive: python run_pipeline.py prepare --download
This handoff contains code, raw audio and instructions only, not generated outputs.
Running the pipeline creates features, trained models, predictions and results.
Generated summaries: TRAINING_RESULTS.md, VERIFICATION.json, RUN_MANIFEST.json.
Main entry runs raw-data preparation, features, manual 10x3 grouped nested CV,
hyperparameter search, training, saved-model checks and results summarisation.
Training is resumable. All paths are relative to this folder, not the old project.
Do not upload this entire teammate folder: the course prohibits submitting data.
No report or submission is made. See README.md for limitations and pending checks.
