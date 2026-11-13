"""Compare models 07 (beta-binomial eta loss) and 08 (eta fault MLP)."""

import hashlib
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd
import torch

from scripts.compare_mmd import evaluation, load_stress_sample
from scripts.data.dials import _integers
from scripts.models.config import DIALS, FEATURE_AE, IDS, KDE_MMD_AE, OUT, SEED, SPECIAL_RUNS, dials_folder
from scripts.models.core import QUANTILE, load_checkpoint
from scripts.models.model_08_eta_fault_classifier import NAME as CLASSIFIER, fit_classifier, predict_classifier
from scripts.models.model_07_eta_betabinomial import NAME as HYPOTHESIS, fit_reference, score as hypothesis_score
from scripts.models.preprocessing import MONITOR_LIMIT
from scripts.plotting.scores import plot_stage, save


def load_eta(source, anchor):
    """Load only native eta bins for the already frozen cached LS inventory."""
    index = next(i for i, name in enumerate(anchor['me_names']) if name.endswith('__GlbMuon_Glb_eta'))
    axis = anchor['axes'][index]
    counts = np.empty((len(source), len(axis['edges'])-1), dtype=np.int32)
    for number, ((dataset, run), rows) in enumerate(source.groupby(['dataset', 'run_number'], sort=False), 1):
        path = dials_folder(dataset, run)/f'{anchor["me_names"][index]}.parquet'
        cached = pd.read_parquet(path, columns=['ls_number', 'data', 'x_min', 'x_max', 'x_bin'])
        if cached.ls_number.duplicated().any():
            raise ValueError(f'Duplicate cached LS: {path}')
        cached = cached.set_index('ls_number').loc[rows.ls_number]
        expected = np.asarray([axis['low'], axis['high'], len(axis['edges'])-1])
        if not np.equal(cached[['x_min', 'x_max', 'x_bin']].to_numpy(), expected).all():
            raise ValueError(f'Eta axes differ: {path}')
        counts[rows.index] = _integers(np.stack(cached.data.to_numpy()), np.int32, f'eta bins: {path}')
        if number % 300 == 0:
            print(f'Loaded eta for {number} runs', flush=True)
    return counts, axis


def classifier_score(counts, pu, saved):
    # Monotone softplus keeps tail resolution and nonnegative plot scores.
    logits = predict_classifier(counts, pu, saved, return_logits=True)
    observed = (counts.sum(1) > 0) & np.isfinite(pu) & (pu >= 0)
    if not np.isfinite(logits[observed]).all():
        raise FloatingPointError('Nonfinite classifier prediction for an observed eta histogram')
    values = np.full(len(logits), np.nan)
    values[observed] = np.logaddexp(0., logits[observed])
    return values


def report(frame, cut, references):
    result = evaluation(frame, cut, references)
    for scope, extra in [('all', np.ones(len(frame), bool)),
                         ('no2026C', ~frame.era.eq('2026C').to_numpy())]:
        for label, item in result[scope].items():
            selected = (frame.split.eq('test') & frame.dcs_pass & frame.label.eq(label)).to_numpy() & extra
            item['total_ls'] = int(selected.sum())
            item['fraction_total'] = item['flagged']/item['total_ls'] if item['total_ls'] else None
    bad = (frame.split.eq('test') & frame.dcs_pass & frame.label.eq('bad')).to_numpy()
    missed = bad & ~references[KDE_MMD_AE]
    values = frame.anomaly_score.to_numpy()
    result['mmd_misses'] = dict(ls=int(missed.sum()), recovered=int((missed & (values > cut)).sum()),
                                unscored=int((missed & ~np.isfinite(values)).sum()))
    result['bad_runs'] = []
    for run in sorted(frame.loc[bad, 'run_number'].unique()):
        selected = bad & frame.run_number.eq(run).to_numpy()
        result['bad_runs'].append(dict(run=int(run), total_ls=int(selected.sum()),
                                       flagged=int((values[selected] > cut).sum()),
                                       unscored=int((~np.isfinite(values[selected])).sum())))
    return result


def publish(name, source, values, calibration, references, model_path, details=None, history=None):
    selected = values[calibration & np.isfinite(values)]
    if not len(selected):
        raise ValueError('No usable Good calibration eta observations')
    cut = float(np.quantile(selected, QUANTILE))
    if not np.isfinite(cut) or cut <= 0:
        raise ValueError('Expected a positive finite Good calibration threshold')
    columns = [c for c in source if not c.startswith('term_') and c not in ['flagged', 'anomaly_score']]
    scores = source[columns].copy()
    scores['anomaly_score'] = values
    scores['flagged'] = pd.array(np.where(scores.dcs_pass & np.isfinite(values), values > cut, None),
                                  dtype='boolean')
    if details is not None:
        for region, label in enumerate(['central', 'forward']):
            for key, output in [('region_scores', 'score'), ('observed_fraction', 'observed'),
                                ('normal_probability', 'expected'), ('efficiency_loss', 'loss')]:
                scores[f'{label}_{output}'] = details[key][:, region]
    folder = OUT/'predict'/name
    folder.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(folder/'scores.parquet', index=False)
    metrics = dict(experiment=name, threshold=cut, quantile=QUANTILE, input='eta-only native histogram',
                   experimental_model=os.path.relpath(model_path.resolve(), folder.resolve()),
                   calibration=dict(ls=len(selected), flagged_fraction=float((selected > cut).mean()),
                                    tied_at_threshold=int((selected == cut).sum())),
                   evaluation=report(scores, cut, references))
    (folder/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    plot_stage(scores, history, cut, OUT/'plots'/name)
    print('RESULT', name, json.dumps(metrics['evaluation']['no2026C']), flush=True)
    return cut, metrics


def synthetic_power(counts, pu, frame, axis, classifier, scorers):
    """A held-out Good-Test shape diagnostic, never fitting or choosing settings."""
    selected = np.flatnonzero(frame.label.eq('good') & frame.split.eq('test') & frame.dcs_pass
                             & frame.era.str.startswith(('2024', '2025')) & (counts.sum(1) > 0))
    rng = np.random.default_rng(SEED+200)
    pool = rng.choice(selected, min(4096, len(selected)), replace=False)
    rows = pool[np.argsort(counts[pool].sum(1), kind='stable')[-64:]]
    reference = classifier['reference']
    p = (counts[rows] + 8*reference['raw_reference'])/(counts[rows].sum(1, keepdims=True)+8)
    p = p @ reference['probabilities'].T
    p /= p.sum(1, keepdims=True)
    centers = (axis['edges'][:-1]+axis['edges'][1:])/2
    masks = dict(central=np.abs(centers)<.9, forward=np.abs(centers)>1.2,
                 left=centers<0, right=centers>0)
    result = []
    for n in [1, 3, 10, 30, 100, 300, 1000]:
        normal = rng.multinomial(n, p).astype(np.int32)
        normal_rates = {name: float((predict(normal, pu[rows]) > cut).mean())
                        for name, (predict, cut) in scorers.items()}
        for region, mask in masks.items():
            for loss in [.25, .5, .75, .95]:
                altered = p*(1-loss*mask)
                altered /= altered.sum(1, keepdims=True)
                observed = rng.multinomial(n, altered).astype(np.int32)
                row = dict(n=n, region=region, loss=loss, parents=len(rows),
                           seen_family=region in ['central', 'forward'], models={})
                for name, (predict, cut) in scorers.items():
                    row['models'][name] = dict(normal_flagged=normal_rates[name],
                                               fault_flagged=float((predict(observed, pu[rows]) > cut).mean()))
                result.append(row)
    return result


def power_plot(rows):
    hep.style.use('CMS')
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), layout='constrained')
    for ax, (name, label) in zip(axes, [(HYPOTHESIS, '07 Beta-binomial'), (CLASSIFIER, '08 MLP classifier')]):
        for region, color in [('central', 'tab:blue'), ('forward', 'tab:orange')]:
            selected = [row for row in rows if row['region']==region and row['loss']==.95]
            ax.plot([row['n'] for row in selected], [row['models'][name]['fault_flagged'] for row in selected],
                    marker='o', color=color, label=f'{region.title()} loss: 95%')
        ax.plot([row['n'] for row in selected], [row['models'][name]['normal_flagged'] for row in selected],
                marker='o', color='black', linestyle='--', label='Normal')
        ax.set(xscale='log', ylim=(0, 1.05), xlabel='Muon count', ylabel='Flagged fraction')
        ax.text(.04, .86, label, transform=ax.transAxes, va='top')
        ax.legend(loc='lower right')
    hep.cms.label(ax=axes[0], text='Work in Progress', data=False, com=None, rlabel='')
    save(fig, OUT/'plots'/'eta_power.png')


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    names = [HYPOTHESIS, CLASSIFIER]
    for name in names:
        for area in ['model', 'predict', 'plots']:
            if (OUT/area/name).exists():
                raise FileExistsError(f'Experimental output already exists: {area}/{name}')
    _, anchor, _ = load_checkpoint(OUT/'model'/KDE_MMD_AE)
    source = pd.read_parquet(OUT/'predict'/KDE_MMD_AE/'scores.parquet').reset_index(drop=True)
    if source.duplicated(IDS).any():
        raise ValueError('Duplicate frozen observation IDs')
    old = pd.read_parquet(OUT/'predict'/FEATURE_AE/'scores.parquet')
    pd.testing.assert_frame_equal(source[IDS+['label', 'split', 'dcs_pass']], old[IDS+['label', 'split', 'dcs_pass']])
    references = {KDE_MMD_AE: source.flagged.fillna(False).to_numpy(dtype=bool),
                  FEATURE_AE: old.flagged.fillna(False).to_numpy(dtype=bool)}
    counts, axis = load_eta(source, anchor)
    pu = source.pileup.to_numpy(dtype=float)
    n = counts.sum(1)
    valid = (n > 0) & np.isfinite(pu) & (pu >= 0)
    train = np.flatnonzero(source.label.eq('good') & source.split.eq('train') & source.dcs_pass
                           & ~source.run_number.isin(SPECIAL_RUNS) & valid)
    train = np.sort(np.random.default_rng(SEED).choice(train, min(16384, len(train)), replace=False))
    train_mask = np.zeros(len(source), bool)
    train_mask[train] = True
    monitor = np.flatnonzero(source.label.eq('good') & source.dcs_pass & source.run_number.isin(
        anchor['signature']['monitor_runs']))
    monitor = np.random.default_rng(SEED).choice(monitor, min(MONITOR_LIMIT, len(monitor)), replace=False)
    calibration = (source.label.eq('good') & source.dcs_pass & source.run_number.isin(
        anchor['signature']['calibration_runs'])).to_numpy()
    assert not calibration[train].any() and not calibration[monitor].any()
    assert not np.intersect1d(train, monitor).size
    assert source.loc[monitor, 'split'].eq('validation').all()
    source['eta_count'] = n
    selection = dict(seed=SEED, eta_counts_sha256=hashlib.sha256(memoryview(counts)).hexdigest(),
                     train_ls=len(train), train_runs=int(source.iloc[train].run_number.nunique()),
                     monitor_ls=len(monitor), calibration_ls=int(calibration.sum()),
                     source_signature=anchor['signature'], scope='eta-only experimental prototypes')
    reference = fit_reference(counts, axis['edges'], train_mask, pileup=pu)
    result = hypothesis_score(counts, reference, pileup=pu)
    model_path = OUT/'model'/HYPOTHESIS/'model.json'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    hcut, hmetrics = publish(HYPOTHESIS, source, result['score'], calibration, references,
                             model_path, details=result)
    reference.update(threshold=hcut, quantile=QUANTILE, selection=selection)
    model_path.write_text(json.dumps(reference, indent=2)+'\n')
    print('Training matched-count eta classifier on Good simulations', flush=True)
    classifier = fit_classifier(counts, pu, train, monitor, axis)
    values = classifier_score(counts, pu, classifier)
    history = dict(train=[row['train_loss'] for row in classifier['history']],
                   validation=[row['monitor_loss'] for row in classifier['history']],
                   best_epoch=classifier['best_epoch'], training_years=[2024, 2025])
    model_path = OUT/'model'/CLASSIFIER/'model.pt'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    ccut, cmetrics = publish(CLASSIFIER, source, values, calibration, references,
                             model_path, history=history)
    classifier.update(threshold=ccut, quantile=QUANTILE, selection=selection,
                      score='softplus(logit), ranking only')
    torch.save(classifier, model_path)
    scorers = {HYPOTHESIS: (lambda h, p: hypothesis_score(h, reference, p)['score'], hcut),
               CLASSIFIER: (lambda h, p: classifier_score(h, p, classifier), ccut)}
    sample = load_stress_sample(anchor, source, DIALS)
    eta_index = next(i for i, name in enumerate(sample['me_names']) if name.endswith('__GlbMuon_Glb_eta'))
    stress = []
    for fraction in [1., .1, .01]:
        gathered = {name: [] for name in names}
        for repeat in range(3):
            rng = np.random.default_rng(SEED+repeat)
            # Draw every ME to preserve the prior study's RNG protocol.
            simulated = [rng.binomial(h, fraction).astype(np.int32) for h in sample['histograms']]
            for name, (predict, cut) in scorers.items():
                gathered[name].extend((predict(simulated[eta_index], sample['pileup'])/cut).tolist())
        row = dict(retained_fraction=fraction, unique_ls=128, draws=3, models={})
        for name, scores in gathered.items():
            scores = np.asarray(scores)
            observed = np.isfinite(scores)
            row['models'][name] = dict(scored=int(observed.sum()),
                                       flagged_fraction=float((scores[observed] > 1).mean()))
        stress.append(row)
        print('STRESS', json.dumps(row), flush=True)
    power = synthetic_power(counts, pu, source, axis, classifier, scorers)
    power_plot(power)
    notes = ['Experimental only: existing models/checkpoints/scores remain unchanged.',
             'These Test runs have already been inspected; this is not a fresh blind evaluation.',
             'Both prototypes use eta only; retained baseline/MMD use eight MEs.',
             'Same 16384 Good TRAIN parents and measured PU; no real Bad training/tuning.',
             'Statistics model uses four TRAIN-PU strata and beta-binomial working overdispersion.',
             'Classifier uses same-N/PU synthetic pairs; monitor BCE selects epochs.',
             'Fixed thresholds use independent Good calibration; no Test setting selection.',
             'N=0 or missing PU is unscored, not Good; fraction_total retains unscored in its denominator.',
             'RR Bad summaries certify run LS ranges, but these eight MEs need not expose every fault.',
             'Synthetic power uses 64 held-out Good-Test 2024/25 templates; left/right are unseen families.',
             'Synthetic histograms and marginal thinning do not preserve event-level cross-ME correlations.',
             'Scores are neither calibrated detector-failure probabilities nor valid p-values.']
    results = {HYPOTHESIS: hmetrics, CLASSIFIER: cmetrics}
    for name, metrics in results.items():
        metrics.update(selection=selection, notes=notes, low_statistics_stress=stress,
                       synthetic_power=power)
        (OUT/'predict'/name/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    summary = dict(results=results, selection=selection, notes=notes)
    (OUT/'predict'/'eta_comparison.json').write_text(json.dumps(summary, indent=2)+'\n')
    print('Completed both eta-only experiments.', flush=True)


if __name__ == '__main__':
    main()
