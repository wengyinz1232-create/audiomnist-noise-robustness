"""Smoke-test WAV -> features -> saved model CLI with a known source example.

This source recording occurs in the training dataset: this is an integration
test, NOT an accuracy claim or an independent held-out evaluation.
"""
import json
import subprocess
import sys
import zipfile
from common import ROOT, save_json


def main():
    examples=ROOT/'examples';examples.mkdir(exist_ok=True)
    with zipfile.ZipFile(ROOT/'data_raw'/'audiomnist.zip') as archive:
        name=next(n for n in archive.namelist() if n.endswith('/7_01_0.wav'))
        (examples/'7_01_0.wav').write_bytes(archive.read(name))
    model=ROOT/'runs'/'main'/'logmel__logistic__augmented'/'models'/'fold_00.pkl'
    result=subprocess.run([sys.executable,str(ROOT/'predict.py'),'--model',str(model),
        '--wav',str(examples/'7_01_0.wav')],capture_output=True,text=True,check=True)
    prediction=json.loads(result.stdout)
    assert len(prediction['probabilities'])==10
    assert abs(sum(prediction['probabilities'].values())-1)<1e-6
    save_json(examples/'prediction_smoke_test.json',{'source':name,'true_digit':7,
        'model':str(model.relative_to(ROOT)),'prediction':prediction,
        'claim':'CLI integration works. This is NOT held-out performance.'})
    print('PREDICTION_CLI_PASS',prediction['prediction'])


if __name__=='__main__':main()
