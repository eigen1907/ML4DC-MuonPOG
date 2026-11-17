"""Combine consecutive, equally labelled LS before constructing model inputs."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.models.config import IDS
from scripts.models.preprocessing import _moments


def grouping_scope(policy):
    """Name the output directory for a saved LS grouping policy."""
    if not policy:
        return 'ls_no_group'
    if policy.get('version') == 'contiguous-quality-v1':
        return f'ls_group_{policy["min_eta_entries"]}glb'
    if policy.get('version') == 'recorded-luminosity-quality-v1':
        return f'ls_group_{policy["min_recorded_lumi_pb"]:g}pb'
    raise ValueError('Unknown LS grouping policy')


def replay_grouping(dataset, policy):
    """Reproduce a checkpoint's grouping without changing its saved policy."""
    scope = grouping_scope(policy)
    if scope == 'ls_no_group':
        return dataset
    if policy.get('version') == 'recorded-luminosity-quality-v1':
        grouped = group_lumisections(dataset, min_recorded_lumi=policy['min_recorded_lumi_pb'])
    else:
        grouped = group_lumisections(dataset, policy['min_eta_entries'])
    if grouped['ls_grouping'] != policy:
        raise ValueError('LS grouping policy differs from the checkpoint')
    return grouped


def _recorded_luminosity(frame):
    """Match cached OMS integrated luminosity in pb^-1 to every original LS."""
    years = frame.dataset.str.extract(r'/Run(20\d\d)')[0]
    if years.isna().any():
        raise ValueError('Cannot determine OMS metadata years from dataset names')
    field = 'recorded_lumi_per_lumisection'
    keys = ['run_number', 'ls_number']
    tables = []
    for year in sorted(years.unique()):
        payload = json.loads((Path('data/oms')/f'{year}.json').read_text())
        table = pd.DataFrame(payload['lumisections']).rename(columns={'lumisection_number': 'ls_number'})
        tables.append(table[keys + [field]])
    matched = frame[keys].merge(pd.concat(tables, ignore_index=True), on=keys,
                                how='left', validate='many_to_one', sort=False)
    values = matched[field].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Every luminosity-grouping LS needs finite nonnegative OMS recorded luminosity')
    return values


def group_lumisections(dataset, min_entries=1000, *, min_recorded_lumi=None):
    """Sum whole LS to an eta-entry or recorded-luminosity target, without overlap.

    Run, split, raw quality changes and missing LS end a segment. A short tail
    joins its preceding completed group in the same segment. A whole segment
    below the target is excluded and counted, rather than crossing a boundary.
    The source dataset and cache files are not modified.
    """
    from scripts.data.dials import validation_roles

    if type(min_entries) is not int or min_entries <= 0:
        raise ValueError('Expected a positive integer eta-entry target')
    by_luminosity = min_recorded_lumi is not None
    if by_luminosity and (isinstance(min_recorded_lumi, bool)
                or not np.isfinite(min_recorded_lumi) or min_recorded_lumi <= 0):
        raise ValueError('Expected a positive finite recorded-luminosity target in pb^-1')
    if dataset.get('ls_grouping'):
        raise ValueError('LS have already been grouped')
    frame = dataset['frame']
    if frame.empty or frame.duplicated(IDS).any() or not frame.eta_entries.gt(0).all():
        raise ValueError('Expected unique positive-eta LS before grouping')
    order = frame.sort_values(IDS, kind='stable').index.to_numpy()
    if not np.array_equal(frame.index, np.arange(len(frame))):
        raise ValueError('Expected a consecutive dataset frame index')
    frame = frame.iloc[order].reset_index(drop=True)
    if by_luminosity:
        frame = frame.assign(recorded_lumi_pb=_recorded_luminosity(frame))
    identity = np.array_equal(order, np.arange(len(frame)))
    keys = ['dataset', 'run_number', 'split', 'rr_label', 'dcs_pass', 'muon_certified']
    boundaries = (frame[keys].ne(frame[keys].shift()).any(axis=1)
                  | frame.ls_number.diff().ne(1)).to_numpy()
    segments = np.flatnonzero(boundaries)
    eta = frame.eta_entries.to_numpy(dtype=np.int64)
    measure = frame.recorded_lumi_pb.to_numpy(dtype=float) if by_luminosity else eta
    target = float(min_recorded_lumi) if by_luminosity else min_entries
    # Decimal OMS values may sum a few floating-point ulps below the target.
    threshold = target*(1 - 1e-12) if by_luminosity else target
    # Partitions cover every source row, including excluded short segments.
    partitions, keep, appended = [], [], 0
    for begin, end in zip(segments, np.r_[segments[1:], len(frame)]):
        first, count, completed = begin, 0, len(partitions)
        for row in range(begin, end):
            count += float(measure[row]) if by_luminosity else int(measure[row])
            if count >= threshold:
                partitions.append([first, row + 1])
                keep.append(True)
                first, count = row + 1, 0
        if first < end:
            if len(partitions) > completed:
                partitions[-1][1] = end
                appended += 1
            else:
                partitions.append([begin, end])
                keep.append(False)
    bounds = np.asarray(partitions, dtype=np.int64)
    keep = np.asarray(keep, dtype=bool)
    if not keep.any():
        raise ValueError('No contiguous quality-compatible LS segment reaches the grouping target')
    first, last = bounds[keep].T
    sizes = last - first
    grouped = frame.iloc[first].copy().reset_index(drop=True)
    grouped['ls_end'] = frame.ls_number.to_numpy()[last - 1]
    grouped['n_ls'] = sizes
    if by_luminosity:
        grouped['recorded_lumi_pb'] = np.add.reduceat(measure, bounds[:, 0])[keep]

    def sums(values):
        values = np.asarray(values)
        if len(values) != len(frame):
            raise ValueError('Histogram and metadata row counts differ')
        ordered = values if identity else values[order]
        return np.add.reduceat(ordered, bounds[:, 0], axis=0, dtype=np.int64)[keep]

    histograms = [sums(histogram) for histogram in dataset['histograms']]
    root_entries = sums(dataset['root_entries'])
    for name in ('eta_entries', 'phi_entries'):
        if name in frame:
            grouped[name] = sums(dataset['frame'][name].to_numpy())
    totals = np.column_stack([histogram.sum(1, dtype=np.int64) for histogram in histograms])
    features = np.column_stack([_moments(histogram, axis['low'], axis['high'])
                               for histogram, axis in zip(histograms, dataset['axes'])])
    complete = grouped.recorded_lumi_pb.ge(threshold) if by_luminosity else grouped.eta_entries.ge(min_entries)
    if not complete.all():
        raise AssertionError('An incomplete LS group survived')

    labels = pd.Series(np.select(
        [frame.rr_label.eq('bad'), ~frame.dcs_pass, ~frame.muon_certified,
         frame.rr_label.eq('good')],
        ['RR Bad', 'DCS Bad', 'JSON Bad', 'Good'], default='RR Unknown'))

    def coverage(values):
        result = {}
        for value in values.unique():
            original = values.eq(value).to_numpy()
            selected = original[first]
            retained = int(sizes[selected].sum())
            result[str(value)] = dict(original_ls=int(original.sum()),
                                      retained_ls=retained, groups=int(selected.sum()),
                                      excluded_ls=int(original.sum()) - retained)
        return result

    result = dict(dataset)
    result.pop('_shape_cache', None)
    roles = dataset.get('validation_roles')
    if roles is None:
        roles = (validation_roles(frame, quality_split=True)
                 if frame.split.eq('validation').any()
                 else dict(monitor_runs=[], calibration_runs=[]))
    policy = (dict(version='recorded-luminosity-quality-v1', min_recorded_lumi_pb=target,
                   source_field='recorded_lumi_per_lumisection',
                   tail='append_previous', short_segments='exclude') if by_luminosity else
              dict(version='contiguous-quality-v1', min_eta_entries=min_entries,
                   tail='append_previous', short_segments='exclude'))
    result.update(frame=grouped, histograms=histograms, totals=totals,
                  root_entries=root_entries, features=features, validation_roles=roles,
                  ls_grouping=policy,
                  grouping_summary=dict(original_ls=len(frame), retained_ls=int(sizes.sum()),
                                        groups=len(grouped), excluded_ls=int(len(frame)-sizes.sum()),
                                        excluded_segments=int((~keep).sum()),
                                        appended_tail_segments=appended,
                                        by_split=coverage(frame.split), by_quality=coverage(labels)))
    criterion = f'recorded luminosity >= {target:g} pb^-1' if by_luminosity else f'eta entries >= {min_entries:,}'
    print(f'Grouped {len(frame):,} LS into {len(grouped):,} observations '
          f'({criterion}); excluded '
          f'{result["grouping_summary"]["excluded_ls"]:,} LS in incomplete segments', flush=True)
    return result
