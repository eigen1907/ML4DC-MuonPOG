"""Compare four frozen KDE changes on the existing cache; never fetch data.

Run after training models 07 and 08. Score-only variants reuse model 06 exactly.
All candidates are declared here before Test evaluation; there is no Test tuning.
"""

import copy
import gc
import json
import os

import numpy as np
import pandas as pd
import torch

from scripts.data.dials import load_dataset
from scripts.models.config import DIALS, IDS, OUT, REGIONS, SEED
from scripts.models.core import (QUANTILE, calibrate, load_checkpoint, mean_score,
                                 restore_model, score_values)
from scripts.models.preprocessing import condition_values, load_pileup, model_features
from scripts.models.kde import decode
from scripts.models.kde_scores import fit_ecdf
from scripts.predict import predict_terms
from scripts.plotting.scores import plot_stage


def evaluate(scores, threshold, baseline):
    """RR run labels are reference labels, not independently verified LS truth."""
    test = scores.split.eq('test') & scores.dcs_pass
    values = scores.anomaly_score.to_numpy()
    flagged = values > threshold
    result = {}
    for scope, extra in [('all', np.ones(len(scores), bool)),
                         ('no2026C', ~scores.era.eq('2026C').to_numpy())]:
        result[scope] = {}
        for label in ('good', 'bad', 'unknown'):
            selected = (test & scores.label.eq(label)).to_numpy() & extra
            mask = selected & np.isfinite(values)
            result[scope][label] = dict(ls=int(mask.sum()), flagged=int(flagged[mask].sum()),
                                       unscored=int((selected & ~np.isfinite(values)).sum()),
                                       fraction=float(flagged[mask].mean()) if mask.any() else None)
    bad = (test & scores.label.eq('bad')).to_numpy()
    result['bad_runs'] = []
    for run in sorted(scores.loc[bad, 'run_number'].unique()):
        mask = bad & scores.run_number.eq(run).to_numpy()
        result['bad_runs'].append(dict(run=int(run), ls=int((mask & np.isfinite(values)).sum()),
                                       unscored=int((mask & ~np.isfinite(values)).sum()),
                                       flagged=int(flagged[mask].sum())))
    result['regions'] = {}
    result['regions_all_splits'] = {}
    for region, runs in REGIONS.items():
        mask = (test & scores.run_number.isin(runs)).to_numpy() & np.isfinite(values)
        result['regions'][region] = dict(ls=int(mask.sum()), flagged=int(flagged[mask].sum()),
                                         fraction=float(flagged[mask].mean()) if mask.any() else None)
        all_roles = (scores.dcs_pass & scores.run_number.isin(runs)).to_numpy() & np.isfinite(values)
        result['regions_all_splits'][region] = dict(
            ls=int(all_roles.sum()), flagged=int(flagged[all_roles].sum()),
            fraction=float(flagged[all_roles].mean()) if all_roles.any() else None,
            split_ls={key: int((all_roles & scores.split.eq(key).to_numpy()).sum())
                      for key in ('train', 'validation', 'test')})
    result['bad_retention'] = {}
    for name, reference in baseline.items():
        known = bad & reference
        result['bad_retention'][name] = dict(previously_flagged=int(known.sum()),
                                             retained=int((known & flagged).sum()),
                                             lost=int((known & ~flagged).sum()),
                                             newly_flagged=int((bad & ~reference & flagged).sum()))
    return result


def save_result(name, spec, saved, data, terms, values, calibration, baseline,
                *, checkpoint=None):
    scores, cut = calibrate(data['frame'], terms, calibration, values)
    if checkpoint is None:
        checkpoint = OUT/'model'/name/'model.pt'
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        saved.update(threshold=cut, quantile=QUANTILE)
        torch.save(saved, checkpoint)
    else:
        np.testing.assert_allclose(cut, saved['threshold'], rtol=1e-5)
        cut = saved['threshold']
        scores['flagged'] = pd.array(np.where(
            scores.dcs_pass & np.isfinite(scores.anomaly_score), scores.anomaly_score > cut, None),
            dtype='boolean')
    scores['pileup'] = data['pileup']
    scores['oms_available'] = np.isfinite(data['pileup'])
    folder = OUT/'predict'/name
    folder.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(folder/'scores.parquet', index=False)
    metrics = dict(experiment=spec.NAME, variant=name, threshold=cut, quantile=QUANTILE,
                   checkpoint=os.path.relpath(checkpoint.resolve(), folder.resolve()),
                   settings=saved['settings'], signature=saved['signature'],
                   score_variant=saved.get('score_variant'), years=[2024, 2025, 2026],
                   prompt_reco=True, evaluation=evaluate(scores, cut, baseline))
    reference = scores.loc[calibration, 'anomaly_score'].dropna().to_numpy()
    metrics['calibration'] = dict(ls=len(reference), flagged_fraction=float((reference > cut).mean()),
                                  tied_at_threshold=int((reference == cut).sum()))
    (folder/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    history = dict(saved['history'], training_years=saved['training_years'])
    plot_stage(scores, history, cut, OUT/'plots'/name)
    print('RESULT', name, json.dumps(metrics['evaluation']['no2026C']), flush=True)
    return metrics


def subset(data, rows):
    return dict(frame=data['frame'].iloc[rows].reset_index(drop=True),
                histograms=[h[rows] for h in data['histograms']],
                totals=data['totals'][rows], axes=data['axes'],
                me_names=data['me_names'], pileup=data['pileup'][rows])


def reconstruction(spec, saved, data):
    features, mask, _, _ = model_features(data, spec, saved['settings'], saved['basis'])
    x = torch.from_numpy(np.where(mask, saved['scaler'].transform(features), 0).astype(np.float32))
    conditions, _, _ = condition_values(data['pileup'], scaler=saved['condition_scaler'])
    model = restore_model(spec, saved)
    with torch.no_grad():
        if saved['settings'].get('mask_me'):
            prediction = model.predict_heldout(x, conditions, torch.as_tensor(data['totals'] > 0))
        else:
            prediction = model(x, conditions)
    return decode(saved['scaler'].inverse_transform(prediction.numpy()), saved['basis'])


def reconstruction_plot(name, spec, saved, data):
    """Actual per-LS AE output, unlike model 06's pooled smoothing reference."""
    import matplotlib.pyplot as plt
    import mplhep as hep

    hep.style.use('CMS')
    probabilities = reconstruction(spec, saved, data)
    for row, item in data['frame'].iterrows():
        fig, axes = plt.subplots(4, 2, figsize=(16, 18), layout='constrained')
        for ax, axis, counts, prediction in zip(axes.flat, data['axes'], data['histograms'], probabilities):
            x = (axis['edges'][:-1]+axis['edges'][1:])/2
            total = counts[row].sum()
            original = counts[row]/total if total else np.zeros(len(x))
            ax.step(x, original, where='mid', color='black', label='Observed')
            ax.plot(x, prediction[row], color='tab:blue', label='Predicted')
            ax.text(.03, .96, axis['stem'].rsplit('__', 1)[-1], transform=ax.transAxes, va='top')
            ax.set(xlabel='ME value', ylabel='Fraction')
        axes.flat[0].legend(loc='upper right')
        hep.cms.label(ax=axes.flat[0], text='Work in Progress', data=True, com=None, rlabel='')
        axes.flat[1].text(1, 1.02, f'Run {item.run_number}, LS {item.ls_number}',
                          transform=axes.flat[1].transAxes, ha='right', va='bottom')
        folder = OUT/'plots'/name/'reconstruction'
        folder.mkdir(parents=True, exist_ok=True)
        fig.savefig(folder/f'{item.run_number}_{item.ls_number}.png', dpi=140, bbox_inches='tight')
        plt.close(fig)


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    spec, anchor, _ = load_checkpoint(OUT/'model'/'06_ae_kde')
    names = ['06_ae_kde_top1', '06_ae_kde_top2', '06_ae_kde_mmd',
             '07_ae_kde_adaptive', '08_ae_kde_masked']
    for name in names:
        if (OUT/'predict'/name/'scores.parquet').exists():
            raise FileExistsError(f'{name}: study output exists; preserve it or move it before rerunning')
    # Frame/order and frozen roles must match the protected reference exactly.
    data = load_dataset(DIALS, split=False, years=[2024, 2025, 2026], prompt_reco=True)
    data['pileup'] = load_pileup(data['frame'])
    frame = data['frame']
    for split, runs in anchor['run_split'].items():
        frame.loc[frame.run_number.isin(runs), 'split'] = split
    original = pd.read_parquet(OUT/'predict'/'06_ae_kde'/'scores.parquet')
    pd.testing.assert_frame_equal(frame[IDS], original[IDS])
    calibration = (frame.label.eq('good') & frame.dcs_pass & frame.run_number.isin(
        anchor['signature']['calibration_runs'])).to_numpy()
    monitor = np.flatnonzero(frame.label.eq('good') & frame.dcs_pass & frame.run_number.isin(
        anchor['signature']['monitor_runs']))
    monitor = np.random.default_rng(SEED).choice(monitor, min(4096, len(monitor)), replace=False)
    baseline = {}
    for name in ['01_ae_2feat', '06_ae_kde']:
        reference = pd.read_parquet(OUT/'predict'/name/'scores.parquet')
        pd.testing.assert_frame_equal(frame[IDS], reference[IDS])
        baseline[name] = reference.flagged.fillna(False).to_numpy(dtype=bool)
    terms = original[[f'term_{me}' for me in range(len(data['me_names']))]].to_numpy()
    ranks = fit_ecdf(terms, monitor)
    models, results = {'06_ae_kde': (spec, anchor)}, {}
    # A metric-only control separates rank normalization from top-k aggregation.
    control = copy.deepcopy(anchor)
    control.update(score_calibration=ranks, score_aggregation='mean', score_variant='rank mean control')
    control_scores, control['threshold'] = calibrate(
        frame, terms, calibration, score_values(terms, control))
    controls = dict(rank_mean=dict(threshold=control['threshold'],
                                   evaluation=evaluate(control_scores, control['threshold'], baseline)))
    models['06_ae_kde_rankmean'] = (spec, control)
    del control_scores
    for mode in ['top1', 'top2']:
        name = f'06_ae_kde_{mode}'
        saved = copy.deepcopy(anchor)
        saved.update(score_calibration=ranks, score_aggregation=mode, score_variant=mode)
        values = score_values(terms, saved)
        results[name] = save_result(name, spec, saved, data, terms, values, calibration, baseline)
        models[name] = (spec, saved)
    saved = copy.deepcopy(anchor)
    saved['settings'].update(score='mmd', mmd_bandwidth_bins=(1., 4., 16.), mmd_count_corrected=True)
    saved['score_variant'] = 'count-corrected multiscale MMD'
    terms, values = predict_terms(spec, saved, data)
    # Match ME scales on Good monitor; final cut uses disjoint Good calibration.
    saved.update(score_calibration=fit_ecdf(terms, monitor), score_aggregation='mean')
    values = score_values(terms, saved)
    name = '06_ae_kde_mmd'
    results[name] = save_result(name, spec, saved, data, terms, values, calibration, baseline)
    models[name] = (spec, saved)
    del original, reference, terms, values
    data.pop('_shape_cache', None)
    gc.collect()
    for name in names[-2:]:
        current_spec, saved, checkpoint = load_checkpoint(OUT/'model'/name)
        assert saved['signature'] == anchor['signature'] and saved['run_split'] == anchor['run_split']
        terms, values = predict_terms(current_spec, saved, data)
        results[name] = save_result(name, current_spec, saved, data, terms, values,
                                    calibration, baseline, checkpoint=checkpoint)
        models[name] = (current_spec, saved)
        del terms, values
        data.pop('_shape_cache', None)
        gc.collect()
    # Same 128 monitoring LS and marginal thinning protocol as the first KDE study.
    rows = monitor[np.argsort(data['totals'][monitor].min(1), kind='stable')[-128:]]
    base = subset(data, rows)
    stress = []
    for fraction in [1., .1, .01]:
        simulated = {name: [] for name in models}
        for repeat in range(3):
            rng = np.random.default_rng(SEED+repeat)
            sample = {key: base[key] for key in ['frame', 'axes', 'me_names', 'pileup']}
            sample['histograms'] = [rng.binomial(h, fraction).astype(np.int32) for h in base['histograms']]
            sample['totals'] = np.column_stack([h.sum(1) for h in sample['histograms']])
            for name, (current_spec, saved) in models.items():
                terms, values = predict_terms(current_spec, saved, sample)
                values = mean_score(terms) if values is None else values
                simulated[name].extend((values/saved['threshold']).tolist())
        row = dict(retained_fraction=fraction, unique_ls=128, draws=3, models={})
        for name, values in simulated.items():
            values = np.asarray(values)
            valid = np.isfinite(values)
            row['models'][name] = dict(scored=int(valid.sum()), flagged_fraction=float((values[valid] > 1).mean()),
                                        median_score_ratio=float(np.median(values[valid])))
        stress.append(row)
        print('STRESS', json.dumps(row), flush=True)
    # Fixed runs, highest-stat LS in each; no anomaly-score-based example selection.
    examples = []
    for run in [397332, 396153]:
        rows = np.flatnonzero(frame.run_number.eq(run) & frame.dcs_pass)
        if len(rows):
            examples.append(rows[np.argmax(data['totals'][rows].min(1))])
    examples = subset(data, examples)
    for name in ['06_ae_kde', '07_ae_kde_adaptive', '08_ae_kde_masked']:
        reconstruction_plot(name, *models[name], examples)
    for name, metrics in results.items():
        metrics['low_statistics_stress'] = stress
        metrics['notes'] = [
            'Exploratory comparison: these Bad runs have already been inspected in previous studies.',
            'No Test/Bad tuning; frozen split and Good calibration quantile 0.975.',
            'RR run verdicts are reference labels, not LS-level ground truth.',
            'Regions in training/validation are descriptive only; regions_all_splits records these roles.',
            'Count-null MMD moments assume fixed predicted p and independent multinomial draws; not p-values.',
            'Thinning is marginal and cannot preserve event-level inter-ME correlations.',
            'Rank scores have finite tail resolution; larger unseen residuals can tie.']
        (OUT/'predict'/name/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    summary = dict(results={name: item['evaluation'] for name, item in results.items()},
                   controls=controls,
                   low_statistics_stress=stress)
    (OUT/'predict'/'kde_study.json').write_text(json.dumps(summary, indent=2)+'\n')
    print('Completed four KDE comparisons.', flush=True)


if __name__ == '__main__':
    main()
