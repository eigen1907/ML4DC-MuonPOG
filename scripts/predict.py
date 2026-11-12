"""Predict LS anomaly scores using a saved checkpoint and its fixed threshold."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.models.config import EXPERIMENTS, OUT
from scripts.data.dials import load_dataset
from scripts.models.core import (QUANTILE, feature_terms, histogram_terms, load_checkpoint,
                                 mean_score, restore_model, threshold_for, window_mean)
from scripts.models.preprocessing import prepare


def predict_terms(spec, saved, dataset):
    """Reconstruct with saved scaling; never fit a scaler or recalibrate a cut."""
    if spec.KIND == 'feature':
        features = [f'{me}__{kind}' for me in dataset['me_names'] for kind in ('mean', 'std')]
        if saved['features'] != features:
            raise ValueError('Checkpoint feature names/order differ from the input MEs')
        return feature_terms(dataset['features'], saved)
    if (saved['me_names'] != dataset['me_names']
            or tuple(saved['bin_sizes']) != tuple(hist.shape[1] for hist in dataset['histograms'])
            or len(saved['axes']) != len(dataset['axes'])
            or any(not np.array_equal(old[key], current[key])
                   for old, current in zip(saved['axes'], dataset['axes'])
                   for key in ('low', 'high', 'edges'))):
        raise ValueError('Checkpoint ME names, binning or axes differ from the input MEs')
    inputs, _ = prepare(dataset, scaler=saved['scaler'])
    model = restore_model(spec, saved)
    terms = histogram_terms(model, inputs, dataset['histograms'])[spec.SCORE]
    if spec.KIND == 'window':
        frame = dataset['frame'].copy()
        frame['anomaly_score'] = mean_score(terms)
        values = window_mean(frame)
        return values[:, None], values
    return terms, None


def predict(checkpoint, data, output=None):
    spec, saved, checkpoint = load_checkpoint(checkpoint)
    cut = threshold_for(checkpoint, saved, spec)
    output = Path(output) if output is not None else OUT/'predict'/spec.NAME
    if (output/'scores.parquet').exists() or (output/'metrics.json').exists():
        raise FileExistsError(f'{output}: saved predictions exist; choose a different --output directory')
    dataset = load_dataset(Path(data), split=False)
    terms, values = predict_terms(spec, saved, dataset)
    values = mean_score(terms) if values is None else np.asarray(values, dtype=float)
    scores = dataset['frame'].copy()
    for split, runs in saved.get('run_split', {}).items():
        scores.loc[scores.run_number.isin(runs), 'split'] = split
    scores['anomaly_score'] = values
    for index in range(terms.shape[1]):
        scores[f'term_{index}'] = terms[:, index]
    scores['flagged'] = pd.array(np.where(scores.dcs_pass & np.isfinite(values),
                                         values > cut, None), dtype='boolean')
    output.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(output/'scores.parquet', index=False)
    metrics = dict(experiment=spec.NAME, threshold=cut, quantile=saved.get('quantile', QUANTILE),
                   checkpoint=os.path.relpath(checkpoint.resolve(), output.resolve()),
                   description=EXPERIMENTS[spec.NAME], signature=saved.get('signature'))
    (output/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    print(f'Saved {len(scores):,} LS to {output}; fixed threshold = {cut:.6g}', flush=True)
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True, help='DIALS cache root containing dataset/run folders')
    parser.add_argument('--output', type=Path, help='Prediction directory; existing scores are preserved')
    args = parser.parse_args()
    torch.set_num_threads(4)
    predict(args.checkpoint, args.data, args.output)


if __name__ == '__main__':
    main()
