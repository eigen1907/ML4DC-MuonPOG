"""Compare top-1/2/3 MMD aggregation without retraining or fetching data."""

import copy
import json
import os

import numpy as np
import pandas as pd
import torch

from scripts.data.dials import _integers
from scripts.models.config import DIALS, FEATURE_AE, IDS, KDE_MMD_AE, OUT, SEED, dials_folder
from scripts.models.core import QUANTILE, load_checkpoint, score_values
from scripts.models.preprocessing import MONITOR_LIMIT
from scripts.predict import predict_terms
from scripts.plotting.scores import plot_stage

TOP_NS = (1, 2, 3)


def load_stress_sample(saved, scores, data_root):
    """Read just 4,096 frozen monitoring LS; retain the same 128 high-stat LS."""
    columns = IDS + ['era', 'label', 'split', 'dcs_pass', 'pileup']
    frame = scores[columns].sort_values(IDS, kind='stable').reset_index(drop=True)
    available = frame.loc[frame.label.eq('good') & frame.dcs_pass & frame.run_number.isin(
        saved['signature']['monitor_runs'])].reset_index(drop=True)
    if available.empty or not available.split.eq('validation').all():
        raise ValueError('Expected the saved Good monitoring roles')
    selected = np.random.default_rng(SEED).choice(len(available), min(MONITOR_LIMIT, len(available)),
                                                replace=False)
    monitor = available.iloc[selected].reset_index(drop=True)
    if len(monitor) < 128 or monitor.duplicated(IDS).any():
        raise ValueError('Expected at least 128 unique Good monitoring LS')
    axes, names = saved['axes'], saved['me_names']
    histograms = [np.empty((len(monitor), len(axis['edges'])-1), dtype=np.int32) for axis in axes]
    for (dataset, run), group in monitor.groupby(['dataset', 'run_number'], sort=False):
        requested = group.ls_number.to_numpy()
        folder = dials_folder(dataset, run, root=data_root)
        for me, (name, axis) in enumerate(zip(names, axes)):
            path = folder/f'{name}.parquet'
            cache = pd.read_parquet(path, columns=['ls_number', 'data', 'x_min', 'x_max', 'x_bin'],
                                    filters=[('ls_number', 'in', requested.tolist())])
            if cache.ls_number.duplicated().any():
                raise ValueError(f'Duplicate cached LS: {path}')
            cache = cache.set_index('ls_number').loc[requested]
            expected = np.array([axis['low'], axis['high'], len(axis['edges'])-1])
            if not np.equal(cache[['x_min', 'x_max', 'x_bin']].to_numpy(), expected).all():
                raise ValueError(f'Cached axes differ from checkpoint: {path}')
            counts = _integers(np.stack(cache.data.to_numpy()), np.int32, f'bin counts: {path}')
            histograms[me][group.index] = counts
    totals = np.column_stack([h.sum(1, dtype=np.int64) for h in histograms])
    top = np.argsort(totals.min(1), kind='stable')[-128:]
    frame = monitor.iloc[top].reset_index(drop=True)
    pileup = frame.pileup.to_numpy(dtype=float)
    if not np.isfinite(pileup).all() or (pileup < 0).any():
        raise ValueError('Expected saved finite pile-up for the monitoring sample')
    return dict(frame=frame, histograms=[h[top] for h in histograms], totals=totals[top],
                axes=axes, me_names=names, pileup=pileup,
                median_min_me_count=float(np.median(totals[top].min(1))))


def evaluation(scores, cut, references):
    values = scores.anomaly_score.to_numpy()
    flagged = values > cut
    test = (scores.split.eq('test') & scores.dcs_pass).to_numpy()
    result = {}
    for scope, extra in [('all', np.ones(len(scores), bool)),
                         ('no2026C', ~scores.era.eq('2026C').to_numpy())]:
        result[scope] = {}
        for label in ['good', 'bad', 'unknown']:
            mask = test & extra & scores.label.eq(label).to_numpy()
            observed = mask & np.isfinite(values)
            result[scope][label] = dict(ls=int(observed.sum()),
                                       flagged=int(flagged[observed].sum()),
                                       unscored=int((mask & ~np.isfinite(values)).sum()),
                                       fraction=float(flagged[observed].mean()) if observed.any() else None)
    bad = test & scores.label.eq('bad').to_numpy()
    result['bad_retention'] = {}
    for name, old in references.items():
        known = bad & old
        result['bad_retention'][name] = dict(previously_flagged=int(known.sum()),
                                            retained=int((known & flagged).sum()),
                                            lost=int((known & ~flagged).sum()),
                                            newly_flagged=int((bad & ~old & flagged).sum()))
    example = scores.run_number.eq(395719) & scores.ls_number.eq(95)
    result['example_395719_95'] = dict(score=float(values[example][0]),
                                      flagged=bool(flagged[example][0]))
    return result


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    spec, anchor, _ = load_checkpoint(OUT/'model'/KDE_MMD_AE)
    if spec.NAME != KDE_MMD_AE or anchor['settings'].get('mmd_top_n') is not None:
        raise ValueError('Expected the original all-ME MMD checkpoint as the reference')
    source = pd.read_parquet(OUT/'predict'/KDE_MMD_AE/'scores.parquet')
    for n in TOP_NS:
        for area in ['model', 'predict', 'plots']:
            path = OUT/area/KDE_MMD_AE/f'top{n}'
            if path.exists():
                raise FileExistsError(f'{path}: comparison already exists; preserve it before rerunning')
    columns = [f'term_{i}' for i in range(len(anchor['me_names']))]
    terms = source[columns].to_numpy()
    np.testing.assert_array_equal(score_values(terms, anchor), source.anomaly_score.to_numpy())
    calibration = (source.label.eq('good') & source.split.eq('validation') & source.dcs_pass
                   & source.run_number.isin(anchor['signature']['calibration_runs'])).to_numpy()
    np.testing.assert_allclose(np.nanquantile(source.anomaly_score[calibration], QUANTILE),
                               anchor['threshold'], rtol=1e-12)
    references = {KDE_MMD_AE: source.flagged.fillna(False).to_numpy(dtype=bool)}
    old = pd.read_parquet(OUT/'predict'/FEATURE_AE/'scores.parquet')
    pd.testing.assert_frame_equal(source[IDS], old[IDS])
    pd.testing.assert_frame_equal(source[['label', 'split', 'dcs_pass']],
                                  old[['label', 'split', 'dcs_pass']])
    references[FEATURE_AE] = old.flagged.fillna(False).to_numpy(dtype=bool)
    results = dict(mean=dict(threshold=anchor['threshold'],
                             evaluation=evaluation(source, anchor['threshold'], references)))
    models = dict(mean=anchor)
    for n in TOP_NS:
        name = f'top{n}'
        saved = copy.deepcopy(anchor)
        saved['settings']['mmd_top_n'] = n
        values = score_values(terms, saved)
        selected = values[calibration & np.isfinite(values)]
        cut = float(np.quantile(selected, QUANTILE))
        saved.update(threshold=cut, quantile=QUANTILE)
        checkpoint = OUT/'model'/KDE_MMD_AE/name/'model.pt'
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(saved, checkpoint)
        scores = source.copy()
        scores['anomaly_score'] = values
        scores['flagged'] = pd.array(np.where(scores.dcs_pass & np.isfinite(values), values > cut, None),
                                      dtype='boolean')
        folder = OUT/'predict'/KDE_MMD_AE/name
        folder.mkdir(parents=True, exist_ok=True)
        scores.to_parquet(folder/'scores.parquet', index=False)
        metrics = dict(experiment=KDE_MMD_AE, top_n=n, threshold=cut, quantile=QUANTILE,
                       settings=saved['settings'], signature=saved['signature'],
                       checkpoint=os.path.relpath(checkpoint.resolve(), folder.resolve()),
                       years=[2024, 2025, 2026], prompt_reco=True,
                       calibration=dict(ls=len(selected), flagged_fraction=float((selected > cut).mean()),
                                        tied_at_threshold=int((selected == cut).sum())),
                       evaluation=evaluation(scores, cut, references))
        (folder/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
        # The inherited AE loss curve is unchanged and remains in the parent folder.
        plot_stage(scores, None, cut, OUT/'plots'/KDE_MMD_AE/name)
        models[name], results[name] = saved, metrics
        print('RESULT', name, json.dumps(metrics['evaluation']['no2026C']), flush=True)
    sample = load_stress_sample(anchor, source, DIALS)
    stress = []
    for fraction in [1., .1, .01]:
        values = {name: [] for name in models}
        coverage = []
        for repeat in range(3):
            rng = np.random.default_rng(SEED+repeat)
            simulated = {key: sample[key] for key in ['frame', 'axes', 'me_names', 'pileup']}
            simulated['histograms'] = [rng.binomial(h, fraction).astype(np.int32)
                                       for h in sample['histograms']]
            simulated['totals'] = np.column_stack([h.sum(1) for h in simulated['histograms']])
            current, _ = predict_terms(spec, anchor, simulated)
            coverage.append(float(np.isfinite(current).mean()))
            for name, saved in models.items():
                values[name].extend((score_values(current, saved)/saved['threshold']).tolist())
        row = dict(retained_fraction=fraction, unique_ls=len(sample['frame']), draws=3,
                   observed_me_fraction=float(np.mean(coverage)), models={})
        for name, scores in values.items():
            scores = np.asarray(scores)
            observed = np.isfinite(scores)
            row['models'][name] = dict(scored=int(observed.sum()),
                                       flagged_fraction=float((scores[observed] > 1).mean()),
                                       median_score_ratio=float(np.median(scores[observed])))
        stress.append(row)
        print('STRESS', json.dumps(row), flush=True)
    summary = dict(results=results, low_statistics_stress=stress,
                   stress_selection=dict(unique_ls=len(sample['frame']), seed=SEED,
                                         median_min_me_count=sample['median_min_me_count']),
                   notes=['Same AE weights, count correction, ECDF and frozen cohort for every N.',
                          'N=1/2/3 are predeclared; thresholds use Good calibration only.',
                          'No default-model change; this previously inspected Test is exploratory.',
                          'RR labels are run-level proxies, not verified per-LS truth.',
                          'Thinning is marginal and cannot preserve event-level inter-ME correlations.'])
    (OUT/'predict'/KDE_MMD_AE/'topn.json').write_text(json.dumps(summary, indent=2)+'\n')
    for name, item in results.items():
        if name != 'mean':
            item.update(low_statistics_stress=stress, notes=summary['notes'])
            (OUT/'predict'/KDE_MMD_AE/name/'metrics.json').write_text(json.dumps(item, indent=2)+'\n')
    print('Completed top-N MMD comparison.', flush=True)


if __name__ == '__main__':
    main()
