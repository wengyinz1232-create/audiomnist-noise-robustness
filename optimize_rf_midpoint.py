"""Run all four RF configurations with the explicitly locked midpoint grids."""
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent / 'rf_midpoint_experiment'
sys.path.insert(0, str(PACKAGE))
from train_rf_midpoint import main

if __name__ == '__main__':
    if not any(arg == '--config' or arg.startswith('--config=') for arg in sys.argv[1:]):
        sys.argv.extend(['--config', str(PACKAGE / 'midpoint_config.json')])
    main()
