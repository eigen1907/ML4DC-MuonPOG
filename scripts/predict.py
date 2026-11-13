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
from scripts.models.core import (QUANTILE, feature_terms, load_checkpoint,
                                 mean_score, restore_model, score_values, threshold_for)
from scripts.models.networks import histogram_scores
from scripts.models.preprocessing import condition_values, load_pileup, model_features, prepare


def predict_terms(spec, saved, dataset):
    """Saved scalers and conditions only; missing pile-up remains unscored."""
    if spec.KIND == 'feature':
        features = [f'{me}__{kind}' for me in dataset['me_names'] for kind in ('mean', 'std')]
        if saved['features'] != features:
            raise ValueError('Checkpoint feature names/order differ from the input MEs')
        return feature_terms(dataset['features'], saved)
    if (saved['me_names'] != dataset['me_names'] or len(saved['axes']) != len(dataset['axes'])
            or any(not np.array_equal(old[key], current[key])
                   for old, current in zip(saved['axes'], dataset['axes'])
                   for key in ('low', 'high', 'edges'))):
        raise ValueError('Checkpoint MEs or histogram axes differ from the input')
    if 'pileup' not in dataset:
        dataset['pileup'] = load_pileup(dataset['frame'])
    settings = saved['settings']
    if spec.KIND in ('shared_gmm', 'kde') and settings['score'] not in ('variance_residual', 'mmd'):
        raise ValueError('Unknown density-reconstruction score in checkpoint')
    conditions, _, available = condition_values(
        dataset['pileup'], scaler=saved['condition_scaler'],
        transform=settings.get('condition_transform', 'linear'))
    conditions = torch.as_tensor(conditions, dtype=torch.float32)
    model = restore_model(spec, saved)
    n_mes, n_rows = len(dataset['me_names']), len(dataset['frame'])
    terms = np.full((n_rows, n_mes), np.nan)
    if not available.any():
        return terms, None
    if spec.KIND == 'mdn':
        if tuple(saved['bin_sizes']) != tuple(h.shape[1] for h in dataset['histograms']):
            raise ValueError('Checkpoint bin counts differ from the input')
        inputs, _ = prepare(dataset, scaler=saved['hist_scaler'])
    else:
        features, mask, feature_me, names = model_features(dataset, spec, settings,
                                                         basis=saved.get('basis'))
        if list(names) != saved['features'] or not np.array_equal(feature_me, saved['feature_me']):
            raise ValueError('Checkpoint feature names/order differ from the input')
        mask = np.asarray(mask, dtype=bool)
        scaled = saved['scaler'].transform(features)
        inputs = torch.from_numpy(np.where(mask, scaled, 0).astype(np.float32, copy=False))
        del scaled
    inputs = torch.as_tensor(inputs, dtype=torch.float32)
    if inputs.shape[1] != saved['input_dim']:
        raise ValueError('Checkpoint input size differs from the prepared inputs')
    rows = np.flatnonzero(available)
    batch_size = int(settings.get('batch_size', 1024))
    model.eval()
    kernels = None
    if settings.get('score') == 'mmd':
        from scripts.models.kde_scores import mmd_kernels

        kernels = mmd_kernels(saved['axes'], [h.shape[1] for h in dataset['histograms']],
                              bandwidth_bins=settings.get('mmd_bandwidth_bins', (1., 4., 16.)))
    with torch.no_grad():
        for first in range(0, len(rows), batch_size):
            selected = rows[first:first + batch_size]
            if settings.get('mask_me'):
                observed = torch.as_tensor(np.asarray(dataset['totals'])[selected] > 0)
                prediction = model.predict_heldout(inputs[selected], conditions[selected], observed)
            else:
                prediction = model(inputs[selected], conditions[selected])
            if spec.KIND == 'mdn':
                result = histogram_scores(prediction, [h[selected] for h in dataset['histograms']])
                terms[selected] = result[settings.get('score', 'deviance')].numpy()
            elif spec.KIND in ('shared_gmm', 'kde'):
                from scripts.models.shared_gaussian import decode_weights, histogram_terms

                weights = saved['scaler'].inverse_transform(prediction.numpy())
                if spec.KIND == 'shared_gmm':
                    probabilities = decode_weights(weights, saved['basis'])
                else:
                    from scripts.models.kde import decode

                    probabilities = decode(weights, saved['basis'])
                if kernels is not None:
                    from scripts.models.kde_scores import mmd_terms

                    terms[selected] = mmd_terms(
                        [h[selected] for h in dataset['histograms']], probabilities, kernels,
                        count_corrected=settings.get('mmd_count_corrected', True))
                else:
                    terms[selected] = histogram_terms(
                        [h[selected] for h in dataset['histograms']], probabilities,
                        saved['bin_variance'], saved['probability_floor'])
            else:
                errors = (prediction.numpy() - inputs[selected].numpy()) ** 2
                for me in range(n_mes):
                    columns = np.asarray(feature_me) == me
                    observed = mask[selected][:, columns]
                    counts = observed.sum(1)
                    terms[selected, me] = np.divide(
                        np.where(observed, errors[:, columns], 0).sum(1), counts,
                        out=np.full(len(selected), np.nan), where=counts > 0)
    return terms, score_values(terms, saved) if saved.get('score_calibration') is not None else None


def predict(checkpoint, data, output=None, *, years=None, prompt_reco=False):
    spec, saved, checkpoint = load_checkpoint(checkpoint)
    cut = threshold_for(checkpoint, saved, spec)
    output = Path(output) if output is not None else OUT/'predict'/spec.NAME
    if (output/'scores.parquet').exists() or (output/'metrics.json').exists():
        raise FileExistsError(f'{output}: saved predictions exist; choose a different --output directory')
    dataset = load_dataset(Path(data), split=False, years=years, prompt_reco=prompt_reco)
    terms, values = predict_terms(spec, saved, dataset)
    values = mean_score(terms) if values is None else np.asarray(values, dtype=float)
    scores = dataset['frame'].copy()
    for split, runs in saved.get('run_split', {}).items():
        scores.loc[scores.run_number.isin(runs), 'split'] = split
    if spec.KIND != 'feature':
        scores['pileup'] = np.asarray(dataset['pileup'], dtype=float)
        scores['oms_available'] = np.isfinite(scores.pileup) & scores.pileup.ge(0)
    scores['anomaly_score'] = values
    for index in range(terms.shape[1]):
        scores[f'term_{index}'] = terms[:, index]
    scores['flagged'] = pd.array(np.where(scores.dcs_pass & np.isfinite(values), values > cut, None), dtype='boolean')
    output.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(output/'scores.parquet', index=False)
    metrics = dict(experiment=spec.NAME, threshold=cut, quantile=saved.get('quantile', QUANTILE),
                   checkpoint=os.path.relpath(checkpoint.resolve(), output.resolve()),
                   description=EXPERIMENTS[spec.NAME], signature=saved.get('signature'),
                   years=sorted(scores.dataset.str.extract(r'/Run(20\d\d)')[0].astype(int).unique().tolist()),
                   prompt_reco=prompt_reco)
    if saved.get('score_variant'):
        metrics['score_variant'] = saved['score_variant']
    if spec.KIND != 'feature':
        metrics.update(settings=saved['settings'], oms_available_ls=int(scores.oms_available.sum()))
    (output/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    print(f'Saved {len(scores):,} LS to {output}; fixed threshold = {cut:.6g}', flush=True)
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True, help='DIALS cache root containing dataset/run folders')
    parser.add_argument('--output', type=Path, help='Prediction directory; existing scores are preserved')
    parser.add_argument('--years', type=int, nargs='+', help='Use only these cached years; default: all')
    parser.add_argument('--prompt-reco', action='store_true', help='Match the PromptReco-only training cohort')
    args = parser.parse_args()
    torch.set_num_threads(4)
    predict(args.checkpoint, args.data, args.output, years=args.years, prompt_reco=args.prompt_reco)


if __name__ == '__main__':
    main()
