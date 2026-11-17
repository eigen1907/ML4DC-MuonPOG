"""Replay the fixed Good-run case study from saved 1 pb raw-deviance results.

These examples were chosen after inspecting Test; they do not select training
data or tune the model. RR component states are run totals, not per-LS verdicts.
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import mplhep as hep
import numpy as np
import pandas as pd
import torch

from scripts.models.config import MAIN_PREDICTIONS, OUT, RR_DATA
from scripts.data.grouping import grouping_scope
from scripts.models.core import load_checkpoint
from scripts.models.free_gaussian import decode_free
from scripts.models.histogram_statistics import deviance, deviance_bins
from scripts.models.preprocessing import model_features
from scripts.plotting.components import cached_prefit_rows, inference, load_histograms
from scripts.plotting.scores import quality_labels, save
from scripts.predict import predict_weights


# Fixed historical cases and controls; changing data selection is not automatic.
CASE_ROWS = ((395729, 841), (395392, 1045), (393168, 145), (393168, 160),
             (393168, 346), (391699, 1), (384382, 185), (384382, 200),
             (403166, 550), (395098, 28), (395098, 32), (396222, 38), (396222, 42))
OTHER_PEAKS = {391699: (1, 886), 395098: (28, 31), 396222: (38, 41), 384382: (194, 274)}
STA_HITS = 'NumberOfValidRecHitsPerTrack_sta'
GLB_HITS = 'NumberOfMeanRecHitsPerTrack_glb'
PT = 'GlbMuon_Glb_pt'


def component_status(counts):
    states = {name for name, count in counts.items() if count > 0 and name != 'EMPTY'}
    return next(iter(states)).title() if len(states) == 1 else 'Mixed' if states else 'Unknown'


def run_status(good):
    """Keep the RR counts so Mixed is never mistaken for an LS-level state."""
    registry = {}
    for year in sorted(good.era.str[:4].unique()):
        for row in json.loads((RR_DATA/f'{year}.json').read_text())['runs']:
            key = (year, int(row['run_number']))
            if key in registry:
                raise ValueError('Multiple cached RR records for one run')
            registry[key] = row
    records = []
    for (era, run), group in good.groupby(['era', 'run_number'], sort=True):
        row = registry.get((era[:4], int(run)), {})
        record = dict(run_number=int(run), era=era, rr_dataset=row.get('rr_dataset', ''),
                      observations=len(group), original_ls=int(group.n_ls.sum()),
                      median_score=float(group.score.median()), flagged=int(group.flagged.sum()))
        for component in ('muon', 'dt', 'csc', 'rpc', 'gem'):
            counts = row.get('components', {}).get(component, {})
            record[component] = '; '.join(f'{key} {value}' for key, value in counts.items()
                                          if value > 0 and key != 'EMPTY') or 'Unknown'
            record[component+'_status'] = component_status(counts)
        records.append(record)
    return pd.DataFrame(records)


def scatter_axes():
    fig, ax = plt.subplots(figsize=(14, 8.5), layout='constrained')
    ax.set(xscale='log', yscale='log', xlabel='Entries', ylabel='Score', ylim=(.03, 10))
    ax.set_xticks([20000, 50000, 100000, 200000, 500000],
                  labels=['20,000', '50,000', '100,000', '200,000', '500,000'])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.axhline(1, color='black', linestyle='--')
    hep.cms.label(ax=ax, text='Work in Progress', data=True, com=None, rlabel='2024–2026 Test Good')
    return fig, ax


def overview(good, statuses, folder):
    selected = pd.Series(False, index=good.index)
    for run, (first, last) in OTHER_PEAKS.items():
        selected |= good.run_number.eq(run) & good.ls_number.ge(first) & good.ls_end.le(last)
    parts = [(~good.run_number.isin([395729, 393168]) & ~good.era.eq('2026C') & ~selected,
              'Other Good', '#aaaaaa', 8),
             (selected, 'Selected cases', '#b2182b', 24),
             (good.run_number.eq(395729), '395729 (2025D)', '#dd7417', 15),
             (good.run_number.eq(393168), '393168 (2025C)', '#2275b5', 15),
             (good.era.eq('2026C'), '2026C', '#8b54a2', 15)]
    if not np.all(np.stack([mask.to_numpy() for mask, *_ in parts]).sum(0) == 1):
        raise ValueError('Study overview categories must cover Test Good exactly once')
    fig, ax = scatter_axes()
    for mask, label, color, size in parts:
        rows = good.loc[mask]
        ax.scatter(rows.eta_entries, rows.score, s=size, color=color, alpha=.85,
                   linewidths=0, rasterized=True, label=label)
    ax.legend(loc='upper left')
    save(fig, folder/'overview.png')
    colors = {'Good': '#27844c', 'Bad': '#c93232', 'Mixed': '#dca526',
              'Standby': '#dca526', 'Unknown': '#aaaaaa', 'Notset': '#aaaaaa'}
    for component in ('rpc', 'gem'):
        labels = good.run_number.map(statuses.set_index('run_number')[component+'_status'])
        fig, ax = scatter_axes()
        for status in ('Good', 'Bad', 'Mixed', 'Standby', 'Notset', 'Unknown'):
            rows = good.loc[labels.eq(status)]
            if len(rows):
                ax.scatter(rows.eta_entries, rows.score, s=12, color=colors[status], alpha=.85,
                           linewidths=0, rasterized=True, label=f'{component.upper()} {status}')
        ax.legend(loc='upper left', markerscale=1.5)
        save(fig, folder/f'{component}_status.png')


def example_arrays(dataset, fitted, predicted, run, ls, me):
    rows = np.flatnonzero(dataset['frame'].run_number.eq(run) & dataset['frame'].ls_number.eq(ls))
    if len(rows) != 1:
        raise ValueError(f'Expected exactly one saved Good example: {run}/{ls}')
    index = rows[0]
    counts = dataset['histograms'][me][index]
    edges = np.asarray(dataset['axes'][me]['edges'])
    return (dataset['frame'].iloc[index], edges, counts/counts.sum(), fitted[me][index],
            predicted[me][index], deviance_bins(counts, predicted[me][index]))


def example_plots(good, dataset, fitted, predicted, folder):
    me = {suffix: next(i for i, name in enumerate(dataset['me_names']) if name.endswith(suffix))
          for suffix in (STA_HITS, GLB_HITS, PT)}

    def arrays(run, ls, suffix=STA_HITS):
        return example_arrays(dataset, fitted, predicted, run, ls, me[suffix])

    fig, (ax, dx) = plt.subplots(2, 1, figsize=(13, 10), sharex=True,
                                gridspec_kw={'height_ratios': (3, 1)}, layout='constrained')
    for run, ls, color in ((395729, 841, '#d55e00'), (395392, 1045, '#0072b2')):
        row, edges, actual, _, prediction, terms = arrays(run, ls)
        ax.stairs(actual, edges, color=color, linewidth=1.8, label=f'ME input: {run}')
        ax.stairs(prediction, edges, color=color, linewidth=1.8, linestyle='--', label=f'Predict: {run}')
        dx.stairs(terms, edges, color=color, linewidth=1.6)
    ax.set(ylabel='Bin fraction', xlim=(0, 60))
    ax.text(.025, .95, 'StaMuon Hits', transform=ax.transAxes, va='top')
    ax.text(.025, .84, '395729: LS 841–843\n395392: LS 1045–1047', transform=ax.transAxes, va='top')
    ax.legend(loc='upper right')
    dx.set(xlabel='Hits / track', ylabel='Deviance')
    hep.cms.label(ax=ax, text='Work in Progress', data=True, com=None, rlabel='2025D, Good')
    save(fig, folder/'395729_sta_hits.png')

    # Same-run transitions: native inputs above and the unchanged saved score below.
    for run, suffix, selections, xlim, ylim, name in (
            (393168, STA_HITS, (145, 160, 346), (0, 60), None, '393168_sta_hits.png'),
            (384382, PT, (185, 200), (2, 80), (0, .1), '384382_pt.png')):
        fig, (ax, tx) = plt.subplots(2, 1, figsize=(13, 10),
                                    gridspec_kw={'height_ratios': (2.5, 1)}, layout='constrained')
        for ls, color in zip(selections, ('#0072b2', '#d55e00', '#009e73')):
            row, edges, actual, _, prediction, _ = arrays(run, ls, suffix)
            label = f'LS {ls}–{int(row.ls_end)}'
            ax.stairs(actual, edges, color=color, linewidth=1.8, label='ME input: '+label)
            if suffix == PT:
                ax.stairs(prediction, edges, color=color, linewidth=1.8, linestyle='--', label='Predict: '+label)
        ax.set(xlabel='Hits / track' if suffix == STA_HITS else r'Muon $p_T$ [GeV]',
               ylabel='Bin fraction', xlim=xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.text(.025, .95, 'StaMuon Hits' if suffix == STA_HITS else 'Global Muon', transform=ax.transAxes, va='top')
        ax.legend(loc='upper right')
        rows = good.loc[good.run_number.eq(run)]
        tx.scatter((rows.ls_number+rows.ls_end)/2, rows.score, s=25, color='#0072b2')
        tx.set(xlabel='Lumisection', ylabel='Score', ylim=(0, 1.15) if suffix == STA_HITS else (0, .2))
        if suffix == STA_HITS:
            tx.axhline(1, color='black', linestyle='--')
        hep.cms.label(ax=ax, text='Work in Progress', data=True, com=None, rlabel=f'Run {run}, {rows.era.iloc[0]}, Good')
        save(fig, folder/name)

    for run, ls, xlim, name in ((391699, 1, (0, 60), '391699_sta_hits.png'),
                                (403166, 550, (0, 8), '2026C_sta_hits.png')):
        row, edges, actual, fit, prediction, terms = arrays(run, ls)
        fig, (ax, dx) = plt.subplots(2, 1, figsize=(13, 10), sharex=True,
                                    gridspec_kw={'height_ratios': (3, 1)}, layout='constrained')
        for values, color, label, style in ((actual, 'black', 'ME input', '-'),
                (fit, '#0072b2', 'Prefit', '--'), (prediction, '#d55e00', 'Predict', '-')):
            ax.stairs(values, edges, color=color, linewidth=1.8, linestyle=style, label=label)
        ax.set(ylabel='Bin fraction', xlim=xlim)
        ax.text(.025, .95, 'StaMuon Hits', transform=ax.transAxes, va='top')
        ax.legend(loc='upper right')
        dx.stairs(terms, edges, color='#9467bd', linewidth=1.8)
        dx.set(xlabel='Hits / track', ylabel='Deviance')
        hep.cms.label(ax=ax, text='Work in Progress', data=True, com=None,
                      rlabel=f'Run {run}, LS {ls}–{int(row.ls_end)}, {row.era}')
        save(fig, folder/name)

    fig, axes = plt.subplots(2, 1, figsize=(13, 13), sharex=True, layout='constrained')
    for ax, run, first, following in ((axes[0], 395098, 28, 32), (axes[1], 396222, 38, 42)):
        row, edges, actual, _, prediction, _ = arrays(run, first)
        next_prediction = arrays(run, following)[4]
        for values, color, label, style in ((actual, 'black', f'ME input: LS {first}–{int(row.ls_end)}', '-'),
                (prediction, '#d55e00', 'Predict: first group', '-'),
                (next_prediction, '#0072b2', 'Predict: next group', '--')):
            ax.stairs(values, edges, color=color, linewidth=1.8, linestyle=style, label=label)
        ax.set(ylabel='Bin fraction', xlim=(1, 10), ylim=(0, .04))
        ax.text(.025, .95, 'StaMuon Hits', transform=ax.transAxes, va='top')
        ax.legend(loc='upper right')
        hep.cms.label(ax=ax, text='Work in Progress', data=True, com=None, rlabel=f'Run {run}, Good')
    axes[-1].set_xlabel('Hits / track')
    save(fig, folder/'startup_sta_hits.png')


def example_summary(dataset, fitted, predicted, threshold):
    records = []
    for i, row in dataset['frame'].iterrows():
        for me, (stem, axis, histograms) in enumerate(zip(dataset['me_names'], dataset['axes'], dataset['histograms'])):
            counts = histograms[i]
            n = int(counts.sum())
            actual, fit, prediction = counts/n, fitted[me][i], predicted[me][i]
            centers = (np.asarray(axis['edges'])[:-1]+axis['edges'][1:])/2
            d = float(deviance(counts, prediction))
            records.append(dict(dataset=row.dataset, run_number=int(row.run_number), ls_number=int(row.ls_number),
                ls_end=int(row.ls_end), n_ls=int(row.n_ls), era=row.era, ME=stem, native_count=n,
                eta_entries=int(row.eta_entries), score_over_threshold=row.anomaly_score/threshold,
                deviance=d, deviance_per_count=d/n, input_mean=float(actual@centers),
                predict_mean=float(prediction@centers), prefit_mean=float(fit@centers),
                input_std=float(np.sqrt(actual@(centers-actual@centers)**2)),
                predict_std=float(np.sqrt(prediction@(centers-prediction@centers)**2)),
                input_predict_TV=float(abs(actual-prediction).sum()/2),
                input_prefit_TV=float(abs(actual-fit).sum()/2), input_prefit_deviance=float(deviance(counts, fit))))
    return pd.DataFrame(records)


def startup_sensitivity(spec, saved, dataset):
    """Change context only; keep the first group's observed Standalone target fixed."""
    features, mask, owners, _ = model_features(dataset, spec, saved['settings'], basis=saved['basis'])
    target = next(i for i, name in enumerate(saved['me_names']) if name.endswith(STA_HITS))
    context = next(i for i, name in enumerate(saved['me_names']) if name.endswith(GLB_HITS))
    rows = []
    for run in (395098, 396222):
        first, following = np.flatnonzero(dataset['frame'].run_number.eq(run))
        changed, changed_mask = features[first].copy(), mask[first].copy()
        columns = np.asarray(owners) == context
        changed[columns], changed_mask[columns] = features[following, columns], mask[following, columns]
        inputs = np.stack((features[first], features[following], changed))
        masks = np.stack((mask[first], mask[following], changed_mask))
        decoded = predict_weights(spec, saved, inputs, masks)
        probability = decode_free(decoded, saved['basis'], saved['settings']['max_scale'])[target]
        counts = np.repeat(dataset['histograms'][target][first:first+1], len(inputs), axis=0)
        bin_two = np.searchsorted(saved['axes'][target]['edges'], 2, side='right')-1
        for label, value, q in zip(('original', 'all_next', 'global_hits_next'), deviance(counts, probability), probability):
            rows.append(dict(run_number=run, fixed_target_ls=int(dataset['frame'].iloc[first].ls_number),
                             context=label, deviance=float(value),
                             observed_two_hit_fraction=float(counts[0, bin_two]/counts[0].sum()),
                             predicted_two_hit_fraction=float(q[bin_two])))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=MAIN_PREDICTIONS)
    args = parser.parse_args()
    source = args.input
    metadata = json.loads((source/'metrics.json').read_text())
    spec, saved, checkpoint = load_checkpoint(source/metadata['checkpoint'])
    if (saved['score_mode'] != 'deviance_max'
            or grouping_scope(saved.get('ls_grouping')) != 'ls_group_1pb'):
        raise ValueError('This historical study requires raw-deviance predictions with 1 pb^-1 grouping')
    protected = {path: hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in (checkpoint, source/'scores.parquet', source/'metrics.json')}
    if protected[checkpoint] != metadata['checkpoint_sha256']:
        raise ValueError('Prediction metadata refers to another checkpoint')
    frame = pd.read_parquet(source/'scores.parquet')
    good = frame.loc[frame.split.eq('test') & quality_labels(frame).eq('Good')].copy()
    good['score'] = good.anomaly_score/saved['threshold']
    positions = []
    for run, ls in CASE_ROWS:
        selected = good.loc[good.run_number.eq(run) & good.ls_number.eq(ls)]
        if len(selected) != 1:
            raise ValueError(f'Fixed study example missing or ambiguous: {run}/{ls}')
        positions.append(int(selected.index[0]))
    selected = frame.loc[positions].copy()
    dataset = load_histograms(saved, selected)
    dataset['_prefit_override'] = cached_prefit_rows(saved, metadata, frame, selected)
    torch.set_num_threads(4)
    features, _, predicted = inference(spec, saved, dataset)
    fitted = decode_free(features, saved['basis'], saved['settings']['max_scale'])
    hep.style.use('CMS')
    relative = source.resolve().relative_to((OUT/'predict').resolve())
    folder = OUT/'plots'/relative/'good_run_study'
    tables = source/'good_run_study'
    tables.mkdir(parents=True, exist_ok=True)
    statuses = run_status(good)
    statuses.to_csv(tables/'run_status.csv', index=False)
    example_summary(dataset, fitted, predicted, saved['threshold']).to_csv(tables/'examples.csv', index=False)
    startup_sensitivity(spec, saved, dataset).to_csv(tables/'startup_sensitivity.csv', index=False)
    overview(good, statuses, folder)
    example_plots(good, dataset, fitted, predicted, folder/'examples')
    for path, digest in protected.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise AssertionError(f'Frozen source changed: {path}')
    print(f'Good-run study saved: {folder}; 13 examples, frozen inference verified')


if __name__ == '__main__':
    main()
