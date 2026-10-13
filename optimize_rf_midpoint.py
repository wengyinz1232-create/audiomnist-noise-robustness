"""Run all four RF configurations with the explicitly locked midpoint grids."""
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent / 'rf_midpoint_experiment'
sys.path.insert(0, str(PACKAGE))
from train_rf_midpoint import main as train_main

def main():
    if not any(arg == '--config' or arg.startswith('--config=') for arg in sys.argv[1:]):
        sys.argv.extend(['--config', str(PACKAGE / 'midpoint_config.json')])
    if not any(arg == '--output' or arg.startswith('--output=') for arg in sys.argv[1:]):
        sys.argv.extend(['--output', str(PACKAGE.parent / '.rf-midpoint')])
    train_main()

if __name__ == '__main__':
    main()
