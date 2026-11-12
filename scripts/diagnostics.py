"""Data-only flow coverage and the historical Region-B rate/shape diagnostic."""

import numpy as np
import pandas as pd

from config import BASE_DCS, DM_DCS, HIST_DCS, IDS, OUT, REGIONS, SPECIAL_RUNS
from plots import plot_run_trend


def write_diagnostics(dataset, frames, cuts):
    frame = dataset['frame']
    entries, totals = dataset['root_entries'], dataset['totals']
    selected = (frame.label.eq('good') & frame.split.eq('train') & frame.dcs_pass
                & ~frame.run_number.isin(SPECIAL_RUNS)).to_numpy()
    folder = OUT/'predict/diagnostics'
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, me in enumerate(dataset['me_names']):
        e, n = entries[selected, i], totals[selected, i]
        rows.append(dict(me=me, root_fills=int(e.sum()), inrange_counts=int(n.sum()),
                         inrange_fraction=float(n.sum()/e.sum()) if e.sum() else np.nan,
                         nonempty_fills_zero_inrange_ls=int(((e > 0) & (n == 0)).sum())))
    pd.DataFrame(rows).to_csv(folder/'flow_coverage.csv', index=False)
    for run in REGIONS['B']:
        stem = 'region_b' if len(REGIONS['B']) == 1 else f'region_b_{run}'
        _run_trend(dataset, frames, cuts, run, stem)


def _run_trend(dataset, frames, cuts, run, stem):
    frame = dataset['frame']
    indices = np.flatnonzero(frame.run_number.eq(run))
    if not len(indices):
        return
    indices = indices[np.argsort(frame.ls_number.iloc[indices].to_numpy())]
    me_index = {name.rsplit('__', 1)[-1]: i for i, name in enumerate(dataset['me_names'])}
    pt = me_index['GlbMuon_Glb_pt']
    phi = me_index['GlbMuon_Glb_phi']
    global_hits = me_index['NumberOfMeanRecHitsPerTrack_glb']
    standalone_hits = me_index['NumberOfValidRecHitsPerTrack_sta']
    entries, totals = dataset['root_entries'], dataset['totals']
    counts = dataset['histograms'][pt][indices]
    edges = dataset['axes'][pt]['edges']
    centers = (edges[:-1]+edges[1:])/2
    n = counts.sum(1)
    mean_pt = np.divide(counts@centers, n, out=np.full(len(n), np.nan), where=n > 0)
    target_ids = pd.MultiIndex.from_frame(frame.iloc[indices][IDS])
    score_labels = {BASE_DCS: '2feat + DCS', HIST_DCS: 'Hist MSE', DM_DCS: 'DM-K1'}
    scores = {name: frames[name].set_index(IDS).anomaly_score.reindex(target_ids).to_numpy()/cuts[name]
              for name in score_labels}
    count_series = {'GlbMuon ROOT fills': entries[indices, phi],
                    'StaMuon ROOT fills': entries[indices, standalone_hits],
                    'GlobalHits in-range fills': totals[indices, global_hits]}
    score_series = {score_labels[name]: values for name, values in scores.items()}
    era = frame.era.iloc[indices[0]]
    ls = frame.ls_number.iloc[indices].to_numpy()
    plot_run_trend(ls, count_series, mean_pt, score_series, OUT/f'plots/diagnostics/{stem}.png',
                   label=f'Run {run} (Good), {era}')
    summary = pd.DataFrame(dict(ls_number=ls, global_root_fills=count_series['GlbMuon ROOT fills'],
                               standalone_root_fills=count_series['StaMuon ROOT fills'],
                               global_hits_inrange=count_series['GlobalHits in-range fills'],
                               mean_muon_pt=mean_pt, **scores))
    summary.to_csv(OUT/f'predict/diagnostics/{stem}.csv', index=False)
