"""Predict one WAV with the three trusted full-fit MFCC augmented models.

This deploys the fixed equal-weight supplementary ensemble. A CLI prediction
is not an independent accuracy estimate. Load pickle files only from trusted
project output.
"""
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
from common import ROOT, sha
from evaluation import proba, transform
from features import audio_features

MODELS = ('logistic', 'forest', 'lightgbm_goss')


def predict(models_root, wav_path):
    rate, wave = wavfile.read(wav_path)
    if wave.ndim != 1:
        raise ValueError('A mono WAV is required')
    if wave.dtype == np.int16:
        wave = wave.astype(np.float32)/32768
    elif np.issubdtype(wave.dtype, np.floating):
        wave = wave.astype(np.float32)
    else:
        raise ValueError('Use 16-bit PCM or floating-point WAV')
    divisor = np.gcd(rate,16000)
    wave = resample_poly(wave,16000//divisor,rate//divisor)
    if len(wave) > 16000:
        raise ValueError('Recording must be no longer than 1 second')
    wave = np.pad(wave,(0,16000-len(wave)))
    features = audio_features(wave[None,:])['mfcc_delta']
    probabilities = []
    classes = None
    for model in MODELS:
        path = models_root/f'mfcc_delta__{model}__augmented'/'models'/'fold_00.pkl'
        with path.open('rb') as stream:
            bundle = pickle.load(stream)
        if bundle['source_signature'] != sha(ROOT/'features.py'):
            raise ValueError(f'Feature frontend mismatch: {path}')
        if bundle['representation'] != 'mfcc_delta' or bundle['mode'] != 'augmented':
            raise ValueError(f'Wrong full-fit model: {path}')
        if classes is None:
            classes = bundle['classes']
        elif not np.array_equal(classes,bundle['classes']):
            raise ValueError('Model class order differs')
        probabilities.append(proba(bundle['estimator'],transform(features,bundle['scaler']),classes)[0])
    averaged = np.mean(probabilities,axis=0)
    return {'prediction':int(classes[np.argmax(averaged)]),
            'probabilities':{str(int(label)):float(value) for label,value in zip(classes,averaged)},
            'method':'equal average of three full-fit MFCC augmented model probabilities',
            'note':'One prediction is not a held-out evaluation result.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models-root',type=Path,default=ROOT/'runs'/'main')
    parser.add_argument('--wav',type=Path,required=True)
    args = parser.parse_args()
    print(json.dumps(predict(args.models_root,args.wav),indent=2))


if __name__ == '__main__': main()
