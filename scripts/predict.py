"""Score cached LS using a saved plain AE and its fixed Good threshold."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.data.dials import assign_quality_splits, fingerprint, load_dataset
from scripts.data.grouping import grouping_scope, replay_grouping
from scripts.models.config import EXPERIMENTS, OUT
from scripts.models.core import THRESHOLD_QUANTILE, load_checkpoint, score_terms, restore_model, threshold_for
from scripts.models.preprocessing import model_features


def evaluation(scores):
    test = scores.loc[scores.split.eq('test')]
    result = {}
    for label in ('good', 'bad', 'unknown'):
        rows = test.loc[test.label.eq(label)]
        valid = np.isfinite(rows.anomaly_score)
        flags = rows.loc[valid, 'flagged'].fillna(False).to_numpy(dtype=bool)
        result[label] = dict(total_ls=len(rows), scored=int(valid.sum()),
                           unscored=int((~valid).sum()), flagged=int(flags.sum()),
                           fraction=float(flags.mean()) if len(flags) else None)
    return {'all': result}


def predict_weights(spec, saved, features, mask, *, model=None):
    """Predict Gaussian parameters without observing the target ME."""
    scaled = saved['scaler'].transform(features)
    inputs = torch.from_numpy(np.where(mask, scaled, 0).astype(np.float32))
    model = restore_model(spec, saved) if model is None else model
    with torch.inference_mode():
        if saved['settings'].get('masked'):
            owners = np.asarray(saved['feature_me'])
            observed = torch.from_numpy(np.column_stack([
                mask[:, owners == me].any(1) for me in range(len(saved['me_names']))]))
            predicted = np.empty_like(features, dtype=np.float32)
            for me in range(len(saved['me_names'])):
                columns = owners == me
                predicted[:, columns] = model(inputs, observed, me).numpy()[:, columns]
        else:
            predicted = model(inputs).numpy()
    return saved['scaler'].inverse_transform(predicted)


def decode_predictions(predicted, basis, executor):
    """Decode the predicted mixtures on four independent row chunks."""
    from scripts.models.free_gaussian import decode_free

    bounds = np.linspace(0, len(predicted), 5, dtype=int)
    slices = [slice(first, last) for first, last in zip(bounds[:-1], bounds[1:]) if last > first]

    def decode(rows):
        return decode_free(predicted[rows], basis)

    pieces = list(executor.map(decode, slices))
    return [np.concatenate([piece[me] for piece in pieces]) for me in range(len(basis))]


def predict_terms(spec, saved, dataset, *, return_diagnostics=False, indices=None):
    if (saved['me_names'] != dataset['me_names'] or len(saved['axes']) != len(dataset['axes'])
            or any(not np.array_equal(old['edges'], current['edges'])
                   for old, current in zip(saved['axes'], dataset['axes']))):
        raise ValueError('Input MEs or histogram axes differ from the checkpoint')
    features, mask, owners, names = model_features(dataset, spec, saved['settings'], basis=saved.get('basis'))
    if list(names) != saved['features'] or not np.array_equal(owners, saved['feature_me']):
        raise ValueError('Feature names/order differ from the checkpoint')
    scaled = saved['scaler'].transform(features)
    inputs = torch.from_numpy(np.where(mask, scaled, 0).astype(np.float32))
    del scaled
    if inputs.shape[1] != saved['input_dim']:
        raise ValueError('Feature count differs from the checkpoint')
    model = restore_model(spec, saved)
    n_mes = len(dataset['me_names'])
    terms = np.full((len(inputs), n_mes), np.nan)
    gaussian_model = spec.KIND == 'gmm'
    loss = saved['settings'].get('loss')
    count_model = gaussian_model and loss == 'input_histogram_deviance'
    weighted_count = count_model and saved['settings'].get('loss_normalization') == 'me_bin'
    mse_model = gaussian_model and loss == 'input_histogram_mse'
    if gaussian_model:
        if not (count_model or mse_model) or saved.get('score_mode') != (
                'norm_deviance_max' if weighted_count else 'deviance_max' if count_model else 'loss_norm_max'):
            raise ValueError('GMM loss and score definitions must match the checkpoint')
        if mse_model or weighted_count:
            if (mse_model and saved['settings'].get('loss_normalization') != 'bin') or len(saved.get('loss_scales', ())) != n_mes:
                raise ValueError('Normalized scoring requires saved Good-Train bin scales')
            for scale, axis in zip(saved['loss_scales'], saved['axes']):
                scale = np.asarray(scale)
                if (scale.shape != (len(axis['edges'])-1,)
                        or not np.isfinite(scale).all() or (scale <= 0).any()):
                    raise ValueError('Expected positive finite native-bin loss scales')
        if weighted_count:
            scales = np.asarray(saved.get('me_loss_scales'), dtype=float)
            if scales.shape != (n_mes,) or not np.isfinite(scales).all() or (scales <= 0).any():
                raise ValueError('Weighted deviance needs a positive fixed scale for each ME')
    diagnostic_keys = (('pearson', 'p_deviance', 'p_pearson', 'expected_low_fraction', 'native_count')
                       + (('raw_term',) if weighted_count else ())
                       if count_model else ('raw_term',) if mse_model else ())
    diagnostics = ({key: np.full_like(terms, np.nan) for key in diagnostic_keys}
                   if return_diagnostics else {})
    rows = np.flatnonzero(mask.any(1))
    wanted = None
    if indices is not None:
        indices = np.asarray(indices)
        if (indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer)
                or (indices < 0).any() or (indices >= len(inputs)).any()
                or len(np.unique(indices)) != len(indices)):
            raise ValueError('Expected unique in-range observation indices')
        wanted = np.zeros(len(inputs), dtype=bool)
        wanted[indices] = True
    batch_size = saved['settings']['batch_size']
    pool = ThreadPoolExecutor(max_workers=4) if spec.KIND == 'gmm' else nullcontext(None)
    with torch.inference_mode(), pool as executor:
        for first in range(0, len(rows), batch_size):
            selected = rows[first:first + batch_size]
            # Preserve the full-data NN batch shape for exact calibration replay.
            # Skip unrelated batches; only requested rows are returned below.
            if wanted is not None and not wanted[selected].any():
                continue
            if spec.KIND == 'gmm':
                from scripts.models.histogram_statistics import count_statistics, deviance

                parameters = predict_weights(spec, saved, features[selected], mask[selected], model=model)
                probabilities = decode_predictions(parameters, saved['basis'], executor)
                for me, probability in enumerate(probabilities):
                    counts = dataset['histograms'][me][selected]
                    observed = counts.sum(1) > 0
                    if mse_model:
                        from scripts.models.loss_normalization import histogram_error

                        target = np.divide(counts, counts.sum(1, keepdims=True),
                                           out=np.zeros_like(counts, dtype=float), where=observed[:, None])
                        delta = target-probability.astype(float)
                        terms[selected, me] = np.where(observed,
                            histogram_error(delta, saved['loss_scales'][me]), np.nan)
                        if diagnostics:
                            diagnostics['raw_term'][selected, me] = np.where(observed,
                                histogram_error(delta), np.nan)
                    else:
                        if diagnostics:
                            statistics = count_statistics(counts, probability)
                            for key, values in diagnostics.items():
                                values[selected, me] = statistics['deviance' if key == 'raw_term' else key]
                        if weighted_count:
                            from scripts.models.histogram_statistics import weighted_deviance

                            value = weighted_deviance(counts, probability,
                                saved['loss_scales'][me], saved['me_loss_scales'][me])
                        else:
                            value = statistics['deviance'] if diagnostics else deviance(counts, probability)
                        terms[selected, me] = np.where(observed, value, np.nan)
            else:
                predicted = model(inputs[selected]).numpy()
                errors = (predicted - inputs[selected].numpy()) ** 2
                for me in range(n_mes):
                    columns = np.asarray(owners) == me
                    observed = mask[selected][:, columns]
                    counts = observed.sum(1)
                    terms[selected, me] = np.divide(
                        np.where(observed, errors[:, columns], 0).sum(1), counts,
                        out=np.full(len(selected), np.nan), where=counts > 0)
    if wanted is not None:
        terms[~wanted] = np.nan
        for values in diagnostics.values():
            values[~wanted] = np.nan
    return (terms, diagnostics) if return_diagnostics else terms


def predict(checkpoint, data, output=None, *, years=None, prompt_reco=False, reuse=False):
    spec, saved, checkpoint = load_checkpoint(checkpoint)
    cut = threshold_for(checkpoint, saved, spec)
    relative = Path(grouping_scope(saved.get('ls_grouping'))) / spec.NAME
    output = Path(output) if output is not None else OUT/'predict'/relative
    existing = any((output/name).exists() for name in ('scores.parquet', 'metrics.json'))
    if existing and not reuse:
        raise FileExistsError(f'Saved predictions exist: {output}')
    if existing and not all((output/name).is_file() for name in ('scores.parquet', 'metrics.json')):
        raise ValueError(f'Incomplete prediction snapshot: {output}')
    dataset = load_dataset(Path(data), split=False, years=years, prompt_reco=prompt_reco)
    dataset['frame'], dataset['run_split'] = assign_quality_splits(
        dataset['frame'], saved['run_split'], policy=saved['split_policy'])
    dataset.update(split_policy=saved['split_policy'])
    if saved.get('ls_grouping'):
        dataset = replay_grouping(dataset, saved['ls_grouping'])
    identity = dict(checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    data_fingerprint=fingerprint(dataset), prompt_reco=prompt_reco,
                    threshold=cut)
    if existing:
        metadata = json.loads((output/'metrics.json').read_text())
        if any(metadata.get(key) != value for key, value in identity.items()):
            raise ValueError('Saved predictions use different data/model/settings')
        print(f'Compatible predictions preserved: {output}', flush=True)
        return pd.read_parquet(output/'scores.parquet')
    terms, diagnostics = predict_terms(spec, saved, dataset, return_diagnostics=True)
    values = score_terms(terms, saved)
    scores = dataset['frame'].assign(anomaly_score=values)
    for me in range(terms.shape[1]):
        scores[f'term_{me}'] = terms[:, me]
        for key, diagnostic_values in diagnostics.items():
            scores[f'{key}_{me}'] = diagnostic_values[:, me]
    scores['flagged'] = pd.array(np.where(np.isfinite(values), values > cut, None), dtype='boolean')
    output.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(output/'scores.parquet', index=False)
    metrics = dict(experiment=spec.NAME, description=EXPERIMENTS[spec.NAME],
                   quantile=saved.get('quantile', THRESHOLD_QUANTILE), signature=saved['signature'],
                   checkpoint=os.path.relpath(checkpoint.resolve(), output.resolve()),
                   years=sorted(scores.dataset.str.extract(r'/Run(20\d\d)')[0].astype(int).unique().tolist()),
                   settings=saved['settings'], calibration=saved['calibration'],
                   score_mode=saved.get('score_mode', 'mean'),
                   me_scales=np.asarray(saved['me_scales']).tolist() if 'me_scales' in saved else None,
                   basis_sha256=saved.get('basis_sha256'),
                   split_policy=saved['split_policy'],
                   excluded_zero_eta_ls=dataset['excluded_zero_eta_ls'],
                   ls_grouping=dataset.get('ls_grouping'),
                   grouping_summary=dataset.get('grouping_summary'),
                   prefit_cache=dataset.get('prefit_cache', {}),
                   observation_unit='ls_group' if dataset.get('ls_grouping') else 'ls',
                   split_counts={str(k): int(v) for k, v in scores.split.value_counts().items()},
                   evaluation=evaluation(scores), **identity)
    if spec.KIND == 'gmm' and saved['settings']['loss'] == 'input_histogram_deviance':
        weighted_count = saved['settings'].get('loss_normalization') == 'me_bin'
        metrics['count_statistics'] = dict(
            target='Raw native-bin ME counts; expected counts = native_count * Predict probability',
            score='Maximum ME/bin-weighted Input–Predict deviance' if weighted_count else 'Maximum ME total Input–Predict deviance',
            pvalue='RAW deviance/Pearson diagnostic chi-square survival probability with native_bins - 1 degrees of freedom; never the weighted score',
            limitation='Not calibrated detector-quality probabilities; sparse counts, model error and inter-ME correlations require validation',
            expected_low_fraction='Fraction of native bins with expected count below 5')
        if weighted_count:
            metrics['count_statistics']['raw_deviance'] = 'raw_term_m stores the unweighted current-model deviance for each ME'
            metrics.update(loss_normalization=saved['loss_normalization'],
                           normalization_quantile=saved.get('normalization_quantile'),
                           me_loss_scales=np.asarray(saved['me_loss_scales']).tolist())
    elif spec.KIND == 'gmm':
        metrics.update(loss_normalization=saved['loss_normalization'],
                       normalization_quantile=saved.get('normalization_quantile'))
    (output/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    unit = 'LS groups' if dataset.get('ls_grouping') else 'LS'
    print(f'Saved {len(scores):,} {unit} to {output}; fixed threshold = {cut:.6g}', flush=True)
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--years', type=int, nargs='+')
    parser.add_argument('--prompt-reco', action='store_true')
    parser.add_argument('--reuse', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    predict(args.checkpoint, args.data, args.output, years=args.years,
            prompt_reco=args.prompt_reco, reuse=args.reuse)


if __name__ == '__main__':
    main()
