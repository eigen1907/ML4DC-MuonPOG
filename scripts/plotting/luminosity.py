"""Plot cached luminosity and Global Muon entries per LS and per run."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd

from scripts.models.config import OUT, TWO_FEATURE_AE


LUMI_FIELD = 'recorded_lumi_per_lumisection'


def plot_distribution(frame, unit, output):
    values = frame[LUMI_FIELD].to_numpy(dtype=float)
    maximum = max(float(values.max())*1.02, .01)
    # Run totals span orders of magnitude. A short linear region retains zeros.
    bins = (np.r_[0., np.geomspace(.01, max(maximum, 7.1), 81)] if unit == 'Run'
            else np.linspace(0, maximum, 101))
    fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')
    years = sorted(frame.year.unique())
    for year in years:
        selected = frame.loc[frame.year.eq(year), LUMI_FIELD]
        ax.hist(selected, bins=bins, histtype='step', linewidth=2,
                label=f'{year} ({len(selected):,})')
        print(f'{year} recorded luminosity/{unit}: count={len(selected):,}, '
              f'zeros={selected.eq(0).sum():,}, median={selected.median():.6g}, '
              f'mean={selected.mean():.6g} pb^-1', flush=True)
    ax.set(xlabel=f'Recorded luminosity / {unit} [pb$^{{-1}}$]',
           ylabel='Runs' if unit == 'Run' else 'LS',
           xlim=(bins[0], bins[-1]))
    ax.set_ylim(bottom=0)
    if unit == 'Run':
        ax.axvline(1, color='black', linestyle=':', linewidth=1.4, label='1 pb$^{-1}$')
        ax.set_xscale('symlog', linthresh=.01)
        ax.axvline(7, color='black', linestyle='--', linewidth=1.2, label='7 pb$^{-1}$')
    ax.legend(loc='upper left')
    hep.cms.label(ax=ax, text='Work in Progress', data=True, com=None,
                  rlabel=f'{years[0]}–{years[-1]}')
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {output}', flush=True)


def plot_glbmuon(frame, unit, output):
    """Count ROOT entries in the Global Muon eta ME, not collision events."""
    entries = frame.eta_entries.to_numpy()
    if not np.isfinite(entries).all() or (entries < 0).any() or (entries != np.floor(entries)).any():
        raise ValueError('Expected nonnegative integer Global Muon eta entries')
    maximum = max(1, int(entries.max()))
    bins = np.unique(np.r_[-.5, np.rint(np.geomspace(1, maximum + 1, 81)) - .5])
    fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')
    years = sorted(frame.year.unique())
    for year in years:
        selected = frame.loc[frame.year.eq(year), 'eta_entries']
        counts, _, _ = ax.hist(selected, bins=bins, histtype='step', linewidth=2,
                               label=f'{year} ({len(selected):,})')
        if int(counts.sum()) != len(selected):
            raise AssertionError('Histogram bins omitted Global Muon entries')
        print(f'{year} Global Muon entries/{unit}: count={len(selected):,}, '
              f'median={selected.median():,.1f}, sum={int(selected.sum()):,}', flush=True)
    ax.set(xlabel=f'Global Muon entries / {unit}', ylabel='Runs' if unit == 'Run' else 'LS',
           xlim=(bins[0], bins[-1]))
    ax.set_xscale('symlog', linthresh=1)
    ax.set_ylim(bottom=0)
    ax.axvline(1000, color='black', linestyle='--', linewidth=1.2, label='1,000 entries')
    ax.legend(loc='upper left')
    hep.cms.label(ax=ax, text='Work in Progress', data=True, com=None,
                  rlabel=f'{years[0]}–{years[-1]}')
    fig.savefig(output, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path,
                        default=OUT/'predict'/'ls_no_group'/TWO_FEATURE_AE,
                        help='Ungrouped prediction directory defining the original LS sample')
    args = parser.parse_args()
    path = args.input/'scores.parquet'
    frame = pd.read_parquet(path)
    if 'n_ls' in frame and not frame.n_ls.eq(1).all():
        raise ValueError('Use ls_no_group predictions for a per-LS distribution')
    frame = frame[['run_number', 'ls_number', 'era', 'eta_entries']].copy()
    keys = ['run_number', 'ls_number']
    if frame.empty or frame.duplicated(keys).any():
        raise ValueError('Expected a nonempty sample of unique original run/LS pairs')
    frame['year'] = frame.era.str[:4].astype(int)
    metadata = []
    for year in sorted(frame.year.unique()):
        raw = json.loads((Path('data/oms')/f'{year}.json').read_text())
        table = pd.DataFrame(raw['lumisections']).rename(columns={'lumisection_number': 'ls_number'})
        metadata.append(table[keys + [LUMI_FIELD]].assign(year=year))
    metadata = pd.concat(metadata, ignore_index=True)
    frame = frame.merge(metadata.drop(columns='year'), on=keys, how='left',
                        validate='one_to_one', indicator=True)
    if not frame._merge.eq('both').all():
        raise ValueError(f'OMS metadata missing for {frame._merge.eq("left_only").sum():,} LS')
    values = metadata[LUMI_FIELD].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Expected finite nonnegative per-LS luminosity values')

    # OMS values are integrated luminosities in pb^-1; do not multiply by LS duration.
    hep.style.use('CMS')
    folder = OUT/'plots'/'metadata'
    plot_distribution(frame, 'LS', folder/'luminosity_ls.png')
    # Use every OMS LS in each selected run, including LS excluded from ML.
    selected = metadata.loc[metadata.run_number.isin(frame.run_number.unique())]
    runs = selected.groupby(['year', 'run_number'], as_index=False)[LUMI_FIELD].sum()
    if len(runs) != frame.run_number.nunique():
        raise ValueError('Expected one OMS luminosity total per selected run')
    np.testing.assert_allclose(runs[LUMI_FIELD].sum(), selected[LUMI_FIELD].sum())
    plot_distribution(runs, 'Run', folder/'luminosity_run.png')
    plot_glbmuon(frame, 'LS', folder/'glbmuon_ls.png')
    muon_runs = frame.groupby(['year', 'run_number'], as_index=False).eta_entries.sum()
    if int(muon_runs.eta_entries.sum()) != int(frame.eta_entries.sum()):
        raise AssertionError('Global Muon run sums do not conserve LS entries')
    plot_glbmuon(muon_runs, 'Run', folder/'glbmuon_run.png')
    print(f'OMS matched all {len(frame):,} model LS; run totals include '
          f'all {len(selected):,} cached OMS LS in {len(runs):,} selected runs', flush=True)


if __name__ == '__main__':
    main()
