"""Per-ME count-model checks; none of these plots defines the LS alarm."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import mplhep as hep
import numpy as np
import pandas as pd

from scripts.models.config import SEED
from scripts.plotting.scores import quality_labels, save


TOYS = 255


def me_label(stem):
    name = stem.removeprefix('Muons__').split('__')[-1]
    return {'NumberOfMeanRecHitsPerTrack_glb': 'GlbMuon Hits',
            'NumberOfValidRecHitsPerTrack_sta': 'StaMuon Hits'}.get(name, name)


def raw_column(saved, me):
    prefix = 'raw_term' if saved['settings'].get('loss_normalization') == 'me_bin' else 'term'
    return f'{prefix}_{me}'


def count_checks(frame, saved, tables, folder):
    """Describe Test Good p-values and count dependence without fitting scales."""
    good = frame.split.eq('test') & quality_labels(frame).eq('Good')
    colors = plt.get_cmap('tab10')(np.arange(len(saved['me_names'])))
    cdf, cdf_axes = plt.subplots(2, 1, figsize=(13, 13), layout='constrained')
    trend, axes = plt.subplots(3, 1, figsize=(13, 16), sharex=True, layout='constrained')
    rows = []
    for me, (stem, axis, color) in enumerate(zip(saved['me_names'], saved['axes'], colors)):
        dof = len(axis['edges'])-2
        counts = frame[f'native_count_{me}']
        observed = good & counts.gt(0)
        for ax, name in zip(cdf_axes, ('deviance', 'pearson')):
            p = np.sort(frame.loc[observed, f'p_{name}_{me}'].to_numpy(float))
            if len(p):
                ax.step(np.r_[0., p, 1.], np.r_[0., np.arange(1, len(p)+1)/len(p), 1.],
                        where='post', color=color, label=me_label(stem))
        selected = frame.loc[observed]
        if selected.empty:
            continue
        # Native N differs by ME. Quantiles retain count resolution even when
        # a luminosity grouping puts almost every observation in one decade.
        edges = np.unique(np.quantile(counts[observed], np.linspace(0, 1, 11)))
        groups = np.searchsorted(edges[1:-1], counts[observed], side='right')
        record = []
        for group in np.unique(groups):
            part = selected.iloc[np.flatnonzero(groups == group)]
            n = part[f'native_count_{me}'].to_numpy(float)
            d = part[raw_column(saved, me)].to_numpy(float)
            x2 = part[f'pearson_{me}'].to_numpy(float)
            item = dict(ME=stem, native_count_min=float(n.min()), native_count_max=float(n.max()),
                observations=len(part), native_count_median=float(np.median(n)), degrees_of_freedom=dof,
                median_deviance_over_df=float(np.median(d/dof)),
                median_pearson_over_df=float(np.median(x2/dof)),
                fraction_p_deviance_below_001=float((part[f'p_deviance_{me}'] < .01).mean()),
                fraction_p_pearson_below_001=float((part[f'p_pearson_{me}'] < .01).mean()),
                median_expected_low_fraction=float(part[f'expected_low_fraction_{me}'].median()))
            rows.append(item);record.append(item)
        for ax, value in zip(axes, ('median_deviance_over_df', 'fraction_p_deviance_below_001',
                                    'median_expected_low_fraction')):
            ax.plot([x['native_count_median'] for x in record], [x[value] for x in record],
                    marker='o', color=color, label=me_label(stem))
    for ax, name in zip(cdf_axes, ('Deviance', 'Pearson')):
        ax.plot([0, 1], [0, 1], color='black', linestyle='--')
        ax.set(xlim=(-.01, 1.01), ylim=(0, 1.02), xlabel=f'{name} p-value (nominal)', ylabel='Good fraction')
    hep.cms.label(ax=cdf_axes[0], text='Work in Progress', data=True, com=None, rlabel='Test Good')
    handles, labels = cdf_axes[0].get_legend_handles_labels()
    cdf.legend(handles, labels, loc='outside lower center', ncol=2)
    save(cdf, folder/'pvalue_cdf.png')
    for ax in axes:
        ax.set_xscale('log')
    axes[0].set_yscale('symlog', linthresh=1)
    axes[0].axhline(1, color='black', linestyle='--')
    axes[0].set_ylabel('Median deviance / df')
    axes[1].axhline(.01, color='black', linestyle='--')
    axes[1].set(ylabel='Fraction with p < 0.01', ylim=(0, 1))
    axes[2].set(xlabel='Native count', ylabel='Sparse-bin fraction', ylim=(0, 1))
    hep.cms.label(ax=axes[0], text='Work in Progress', data=True, com=None, rlabel='Test Good')
    handles, labels = axes[0].get_legend_handles_labels()
    trend.legend(handles, labels, loc='outside lower center', ncol=2)
    save(trend, folder/'count_dependence.png')
    pd.DataFrame(rows).to_csv(tables/'count_summary.csv', index=False)


def toy_sample(frame):
    """A small fixed illustration, not a representative p-value calibration set."""
    labels = quality_labels(frame)
    test = frame.split.eq('test') & np.isfinite(frame.anomaly_score)
    positions = []
    # Preserve the completed illustrative sample. The plot's actual tested
    # count is per ME; these selection strata use Global Muon eta entries.
    for label, limit in (('Good', 12), ('RR Bad', 4)):
        for low, high in ((0, 100), (100, 1000), (1000, 10000), (10000, np.inf)):
            candidates = frame.loc[test & labels.eq(label) & frame.eta_entries.ge(low)
                                   & frame.eta_entries.lt(high)].sort_values(['run_number', 'ls_number', 'dataset'])
            if len(candidates):
                positions.extend(candidates.iloc[np.unique(np.linspace(
                    0, len(candidates)-1, min(limit, len(candidates))).astype(int))].index)
    return frame.loc[sorted(set(positions))].copy()


def fixed_predict_toys(spec, saved, metadata, frame, tables):
    """Test the fixed count model, without NN or training uncertainty."""
    selected = toy_sample(frame)
    if selected.empty:
        return pd.DataFrame()
    path = tables/'toys.csv'
    from scripts.models.free_gaussian import decode_free
    from scripts.models.histogram_statistics import count_statistics, normalize_probability
    from scripts.models.preprocessing import model_features
    from scripts.plotting.components import cached_prefit_rows, load_histograms
    from scripts.predict import predict_weights

    dataset = load_histograms(saved, selected)
    dataset['_prefit_override'] = cached_prefit_rows(saved, metadata, frame, selected)
    features, mask, _, _ = model_features(dataset, spec, saved['settings'], basis=saved['basis'])
    predicted = predict_weights(spec, saved, features, mask)
    probabilities = decode_free(predicted, saved['basis'], saved['settings']['max_scale'])
    rng, rows = np.random.default_rng(SEED), []
    labels = quality_labels(selected).to_numpy()
    for me, (stem, counts, probability) in enumerate(zip(saved['me_names'], dataset['histograms'], probabilities)):
        probability = normalize_probability(probability)
        observed = count_statistics(counts, probability)
        np.testing.assert_allclose(observed['deviance'], selected[raw_column(saved, me)],
                                   rtol=2e-4, atol=1e-5, equal_nan=True)
        for i, row in enumerate(selected.itertuples()):
            n = int(counts[i].sum())
            if n == 0:
                continue
            toys = rng.multinomial(n, probability[i], size=TOYS)
            simulated = count_statistics(toys, np.broadcast_to(probability[i], toys.shape))
            record = dict(dataset=row.dataset, run_number=int(row.run_number), ls_number=int(row.ls_number),
                quality=labels[i], ME=stem, native_count=n, simulations=TOYS,
                expected_low_fraction=float(observed['expected_low_fraction'][i]))
            for name in ('deviance', 'pearson'):
                value = float(observed[name][i])
                record[name] = value
                record[f'p_{name}_approx'] = float(observed[f'p_{name}'][i])
                record[f'p_{name}_fixed_predict_mc'] = float(
                    (1+np.count_nonzero(simulated[name] >= value))/(TOYS+1))
            rows.append(record)
    result = pd.DataFrame(rows)
    result.to_csv(path, index=False)
    print(f'Fixed-Predict MC: {len(selected)} observations, {len(result)} ME tests, {TOYS} toys', flush=True)
    return result


def toy_plot(result, folder):
    if result.empty:
        return
    fig, axes = plt.subplots(2, 1, figsize=(13, 13), layout='constrained')
    resolution = 1/(TOYS+1)
    for ax, name in zip(axes, ('deviance', 'pearson')):
        for label, color in (('Good', '#27834b'), ('RR Bad', '#c43c3c')):
            selected = result.quality.eq(label)
            unresolved = np.isclose(result[f'p_{name}_fixed_predict_mc'], resolution)
            for floor, marker in ((False, 'o'), (True, 'v')):
                points = selected & (unresolved == floor)
                if points.any():
                    ax.scatter(result.loc[points, f'p_{name}_approx'],
                               result.loc[points, f'p_{name}_fixed_predict_mc'],
                               s=18 if not floor else 40, marker=marker, alpha=.8, linewidths=0, color=color)
        ax.plot([0, 1], [0, 1], color='black', linestyle='--')
        ax.axhline(resolution, color='gray', linestyle=':')
        ax.set_xscale('symlog', linthresh=resolution)
        ax.set_yscale('symlog', linthresh=resolution)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel=f'{name.capitalize()} p-value (nominal)',
               ylabel='Fixed-Predict toy p-value')
        ax.legend(handles=[Line2D([], [], color='#27834b', marker='o', linestyle='', label='Good'),
                           Line2D([], [], color='#c43c3c', marker='o', linestyle='', label='RR Bad'),
                           Line2D([], [], color='gray', marker='v', linestyle='', label=f'0 / {TOYS} toys')],
                  loc='best')
    hep.cms.label(ax=axes[0], text='Work in Progress', data=True, com=None, rlabel='Test subset')
    save(fig, folder/'chi2_vs_toys.png')


def plot_checks(spec, saved, metadata, frame, source, folder):
    """Regenerate count checks alone from frozen predictions and small fixed-Predict toys."""
    source, folder = Path(source), Path(folder)/'checks'
    tables = source/'checks'
    tables.mkdir(parents=True, exist_ok=True)
    hep.style.use('CMS')
    count_checks(frame, saved, tables, folder)
    toy_plot(fixed_predict_toys(spec, saved, metadata, frame, tables), folder)
