"""Plot high-statistics Test examples for each quality and flag category."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import mplhep as hep
import numpy as np
import pandas as pd
import torch

from scripts.models.config import IDS, MAIN_PREDICTIONS, OUT, dials_folder
from scripts.models.core import load_checkpoint, score_terms
from scripts.models.free_gaussian import component_parameters, decode_free
from scripts.models.gaussian import gaussian_bin_probabilities
from scripts.models.histogram_statistics import count_statistics, deviance_bins, normalize_probability
from scripts.models.preprocessing import model_features
from scripts.plotting.scores import quality_labels


CASE_SCORE = 'deviance_max'
CASE_QUALITIES = {'good': 'Good', 'json_bad': 'JSON Bad',
                  'dcs_bad': 'DCS Bad', 'rr_bad': 'RR Bad'}


def load_histograms(saved, frame):
    """Read and sum native bins over each selected LS or contiguous LS group."""
    frame = frame.reset_index(drop=True)
    if saved.get('ls_grouping') and not {'ls_end', 'n_ls'}.issubset(frame.columns):
        raise ValueError('Grouped predictions must retain their original LS intervals')
    histograms = [np.zeros((len(frame), len(axis['edges'])-1), dtype=np.int64) for axis in saved['axes']]
    for (dataset, run), positions in frame.groupby(['dataset', 'run_number']).groups.items():
        for m, stem in enumerate(saved['me_names']):
            table = pd.read_parquet(dials_folder(dataset, run)/(stem+'.parquet'), columns=['ls_number', 'data'])
            if table.ls_number.duplicated().any():
                raise ValueError('Duplicate cached LS')
            table = table.set_index('ls_number').sort_index()
            for position in positions:
                row = frame.loc[position]
                first = int(row.ls_number)
                last = int(row.get('ls_end', first))
                expected = np.arange(first, last + 1)
                selected = table.loc[first:last]
                if (last < first or len(expected) != int(row.get('n_ls', 1))
                        or not np.array_equal(selected.index.to_numpy(), expected)):
                    raise ValueError('Cached LS coverage differs from the saved LS group')
                histograms[m][position] = np.stack(selected.data).sum(axis=0, dtype=np.int64)
    return dict(frame=frame, histograms=histograms, me_names=saved['me_names'], axes=saved['axes'], cache_prefit=False,
                totals=np.column_stack([h.sum(1, dtype=np.int64) for h in histograms]))


def inference(spec, saved, dataset):
    from scripts.predict import predict_weights

    features, mask, owners, names = model_features(dataset, spec, saved['settings'], basis=saved['basis'])
    if list(names) != saved['features'] or not np.array_equal(owners, saved['feature_me']):
        raise ValueError('Feature order differs from checkpoint')
    decoded = predict_weights(spec, saved, features, mask)
    predictions = decode_free(decoded, saved['basis'], saved['settings']['max_scale'])
    frame = dataset['frame']
    if saved['settings']['loss'] == 'input_histogram_deviance':
        statistics = [count_statistics(counts, predicted)
                      for counts, predicted in zip(dataset['histograms'], predictions)]
        terms = np.column_stack([item['deviance'] for item in statistics])
        if saved['settings'].get('loss_normalization') == 'me_bin':
            from scripts.models.histogram_statistics import weighted_deviance

            if not np.allclose(terms, frame[[f'raw_term_{m}' for m in range(len(predictions))]],
                               rtol=2e-4, atol=1e-6, equal_nan=True):
                raise ValueError('Frozen inference does not reproduce stored raw deviance')
            terms = np.where(dataset['totals'] > 0, np.column_stack([
                weighted_deviance(counts, predicted, scale, me_scale)
                for counts, predicted, scale, me_scale in zip(dataset['histograms'], predictions,
                                                              saved['loss_scales'], saved['me_loss_scales'])]), np.nan)
        pearson = np.column_stack([item['pearson'] for item in statistics])
        if not np.allclose(pearson, frame[[f'pearson_{m}' for m in range(len(predictions))]],
                          rtol=2e-4, atol=1e-6, equal_nan=True):
            raise ValueError('Frozen inference does not reproduce stored Pearson statistics')
    else:
        from scripts.models.loss_normalization import histogram_error

        valid = dataset['totals'] > 0
        residuals = []
        for me, (counts, predicted) in enumerate(zip(dataset['histograms'], predictions)):
            target = np.divide(counts, counts.sum(1, keepdims=True),
                               out=np.zeros_like(counts, dtype=float), where=valid[:, me, None])
            residuals.append(target-predicted.astype(float))
        terms = np.where(valid, np.column_stack([
            histogram_error(delta, scale) for delta, scale in zip(residuals, saved['loss_scales'])]), np.nan)
        raw_terms = np.where(valid, np.column_stack([histogram_error(delta) for delta in residuals]), np.nan)
        if not np.allclose(raw_terms, frame[[f'raw_term_{m}' for m in range(len(predictions))]],
                           rtol=2e-4, atol=1e-10, equal_nan=True):
            raise ValueError('Frozen inference does not reproduce stored raw MSE terms')
    if not np.allclose(terms, frame[[f'term_{m}' for m in range(len(predictions))]],
                       rtol=2e-4, atol=1e-6, equal_nan=True):
        raise ValueError('Frozen inference does not reproduce stored ME scores')
    if not np.allclose(score_terms(terms, saved), frame.anomaly_score, rtol=2e-4, atol=1e-10, equal_nan=True):
        raise ValueError('Frozen inference does not reproduce stored LS scores')
    return features, decoded, predictions


def cached_prefit_rows(saved, metadata, canonical, selected):
    """Replay exact full-cache rows, rather than repeating a free fit for plots."""
    kind = f'free{len(saved["basis"][0]["weights"])}'
    records = metadata.get('prefit_cache') or saved.get('prefit_cache') or {}
    record = records.get(kind)
    if record is None:
        raise ValueError('Free-Gaussian examples require the saved full prefit cache')
    from scripts.models.preprocessing import _prefit_controls

    positions = selected.index.to_numpy(dtype=int)
    identity = IDS + [name for name in ('ls_end', 'n_ls') if name in canonical]
    if not canonical.iloc[positions][identity].reset_index(drop=True).equals(
            selected[identity].reset_index(drop=True)):
        raise ValueError('Selected examples do not match canonical prefit row positions')
    path = Path(record['path'])
    with np.load(path, allow_pickle=False) as stored:
        manifest = json.loads(str(stored['manifest'].item()))
        key = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        if (key != record['key'] or path.stem != key or manifest['rows'] != len(canonical)
                or manifest['me_names'] != saved['me_names']
                or manifest['controls'] != _prefit_controls(saved['settings'], True)):
            raise ValueError('Saved prefit cache identity differs from the model or prediction rows')
        return dict(features=stored['features'][positions], valid=stored['valid'][positions],
                    feature_me=stored['feature_me'], feature_names=stored['feature_names'].tolist())


def observation_basis(reference, parameters, position):
    """Build component curves from one LS's fitted or predicted free parameters."""
    means, scales = parameters['means'][position], parameters['scales'][position]
    probabilities = gaussian_bin_probabilities(
        np.ones((len(means), 1)), means[:, None], scales[:, None],
        reference['edges'], periodic=reference['periodic']).T
    probabilities = np.maximum(probabilities, 1e-12)
    probabilities /= probabilities.sum(0, keepdims=True)
    return dict(reference, means=means, scales=scales, probabilities=probabilities)


def smooth_gaussians(axis, basis):
    """Sample the saved Gaussians finely, on the native-bin fraction scale."""
    subdivisions = 16
    edges = np.asarray(axis['edges'])
    fine_edges = (edges[:-1, None] + np.diff(edges)[:, None]
                  * np.arange(subdivisions)/subdivisions).ravel()
    fine_edges = np.append(fine_edges, edges[-1])
    normalized = np.interp(fine_edges, edges, basis['edges'])
    probabilities = gaussian_bin_probabilities(
        np.ones((len(basis['means']), 1)), basis['means'][:, None],
        basis['scales'][:, None], normalized, periodic=basis['periodic']).T
    # Rebinning must reproduce the saved basis (apart from its tiny bin floor).
    assert np.allclose(probabilities.reshape(len(edges)-1, subdivisions, -1).sum(1),
                       basis['probabilities'], rtol=1e-7, atol=1e-10)
    return (fine_edges[:-1]+fine_edges[1:])/2, probabilities*subdivisions


def axis_label(stem):
    for suffix, label in (('_pt', r'Muon $p_T$ [GeV]'), ('_eta', r'Muon $\eta$'), ('_phi', r'Muon $\phi$')):
        if stem.endswith(suffix):
            return label
    if 'RecHitsPerTrack' in stem:
        return 'Hits / track'
    if stem.endswith(('GlbGlbMuon_HM', 'GlbGlbMuon_LM')):
        return 'Dimuon mass [GeV]'
    if stem.endswith('relPFIso_R03'):
        return 'Relative isolation'
    return stem.rsplit('__', 1)[-1]


def plot_me(row, stem, axis, counts, basis, fitted_weights, predicted_weights, path, *,
            predicted_basis=None, residual_scale=None, deviance_scale=None):
    predicted_basis = basis if predicted_basis is None else predicted_basis
    n = int(counts.sum())
    fit_components = basis['probabilities']*fitted_weights[None, :]
    predicted_components = predicted_basis['probabilities']*predicted_weights[None, :]
    fit_sum = fit_components.sum(1) if n else np.full(len(counts), np.nan)
    predicted_sum = predicted_components.sum(1)
    observed = counts/n if n else np.full(len(counts), np.nan)
    edges = np.asarray(axis['edges'])
    smooth_x, smooth_basis = smooth_gaussians(axis, basis)
    predict_x, smooth_predicted_basis = smooth_gaussians(axis, predicted_basis)
    assert np.array_equal(smooth_x, predict_x)
    smooth_prefit = smooth_basis*fitted_weights[None, :] if n else np.full_like(smooth_basis, np.nan)
    smooth_predict = smooth_predicted_basis*predicted_weights[None, :]
    fig, (fit_ax, predict_ax, diff) = plt.subplots(3, 1, figsize=(13.5, 13), sharex=True,
                                   gridspec_kw={'height_ratios': (3, 3, 1), 'hspace': .04}, layout='constrained')
    peak = max(smooth_predict.sum(1).max(), np.nanmax(observed) if n else 0,
               np.nanmax(smooth_prefit.sum(1)) if n else 0)
    for ax, components, color, name, palette in (
            (fit_ax, smooth_prefit, '#1874c6', 'Prefit',
             ('#159947', '#00a878', '#00a6b5', '#087cc5', '#2443b3')),
            (predict_ax, smooth_predict, '#bc4b37', 'Predict',
             ('#dab000', '#f3a000', '#ef7024', '#dc3730', '#b71025'))):
        means = basis['means'] if name == 'Prefit' else predicted_basis['means']
        shades = np.empty(len(means))
        shades[np.argsort(means, kind='stable')] = np.linspace(0, 1, len(shades))
        component_colors = LinearSegmentedColormap.from_list(name, palette)(shades)
        ax.stairs(observed, edges, color='black', linewidth=1.4, label='ME input', zorder=4)
        for k in range(components.shape[1]):
            ax.plot(smooth_x, components[:, k], color=component_colors[k], alpha=.9, linewidth=1.25, linestyle='--',
                    label=f'Gaussian {name.lower()}' if k == 0 else None)
        ax.plot(smooth_x, components.sum(1), color=color, linewidth=2.1,
                label=f'Gaussian {name.lower()} sum', zorder=3)
        ax.set_ylabel('Bin fraction')
        ax.set_ylim(0, peak*1.7 if peak else 1)
        ax.legend(loc='upper right')
        ax.tick_params(labelbottom=False)
    if not n:
        fit_ax.text(.5, .5, 'No entries: prefit and diff unavailable', transform=fit_ax.transAxes, ha='center')
    status = 'Non-Flagged' if row.category.startswith('non_flagged_') else 'Flagged'
    quality = quality_labels(pd.DataFrame([row])).iloc[0]
    fit_ax.text(.025, .97, f'{stem.rsplit("__", 1)[-1]}\nN = {n:,}; {quality}; {status}',
            transform=fit_ax.transAxes, va='top', bbox=dict(facecolor='white', alpha=.85, edgecolor='none'))
    last = int(row.get('ls_end', row.ls_number))
    ls = str(row.ls_number) if last == row.ls_number else f'{row.ls_number}–{last}'
    hep.cms.label(ax=fit_ax, text='Work in Progress', data=True, com=None,
                  rlabel=f'Run {row.run_number}, LS {ls}, {row.era}')
    # Scoring uses every native bin, including bins outside the displayed pT range.
    if residual_scale is None:
        # Sum(residual**2) is this ME's deviance, not a Gaussian pull.
        expected = n*normalize_probability(predicted_sum)
        contributions = deviance_bins(counts, predicted_sum)
        if deviance_scale is not None:
            scale = np.asarray(deviance_scale, dtype=float)
            if scale.shape != (len(counts),) or not np.isfinite(scale).all() or (scale <= 0).any():
                raise ValueError('Expected positive finite ME/bin deviance scales')
            contributions = contributions/scale
        residual = (np.sign(counts-expected)*np.sqrt(contributions)
                    if n else np.full(len(counts), np.nan))
        ylabel = 'Weighted\nresidual' if deviance_scale is not None else 'Deviance\nresidual'
    else:
        scale = np.asarray(residual_scale, dtype=float)
        if (scale.shape not in ((), (len(counts),)) or not np.isfinite(scale).all() or (scale <= 0).any()):
            raise ValueError('Expected positive finite native-bin residual scales')
        residual = (observed-predicted_sum)/scale
        ylabel = 'Norm. diff'
    diff.stairs(residual, edges, color='#7546a3', linewidth=1.8, zorder=4)
    diff.axhline(0, color='0.5', linewidth=.8)
    diff.set(xlabel=axis_label(stem), ylabel=ylabel)
    diff.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2), useMathText=True)
    if stem.endswith('_pt'):
        diff.set_xlim(0, 80)
    fig.savefig(path, dpi=150, bbox_inches='tight');plt.close(fig)
    return observed, fit_sum, predicted_sum, residual


def case_scores(scores, saved):
    """Use each model's adopted max-score decisions for every case category."""
    mode = saved.get('score_mode')
    if mode not in (CASE_SCORE, 'loss_norm_max', 'norm_deviance_max'):
        raise ValueError('Expected the saved MSE or deviance max-score decisions')
    cut = float(saved['threshold'])
    values = scores.anomaly_score.to_numpy(float)
    if not np.array_equal(scores.flagged.to_numpy(bool), values > cut):
        raise ValueError('Decisions differ from the saved threshold')
    return scores.assign(case_threshold=cut, case_score_mode=mode)


def case_inventory(scores):
    """Select one highest-statistics Test observation for each of eight cases."""
    scores = scores.reset_index(drop=True)
    examples, positions, inventory = [], [], []
    valid = scores.split.eq('test') & np.isfinite(scores.anomaly_score)
    labels = quality_labels(scores)
    mode = scores.case_score_mode.iloc[0] if len(scores) and 'case_score_mode' in scores else CASE_SCORE
    for flagged in (True, False):
        for suffix, quality in CASE_QUALITIES.items():
            category = ('flagged_' if flagged else 'non_flagged_') + suffix
            candidates = scores.loc[valid & labels.eq(quality)
                                    & scores.flagged.eq(flagged).fillna(False)]
            record = dict(category=category, quality=quality, flagged=flagged,
                          score_mode=mode, available=not candidates.empty,
                          candidates=len(candidates),
                          reason='' if len(candidates) else 'No scored Test observation in this category')
            if not candidates.empty:
                row = candidates.sort_values(['eta_entries', 'run_number', 'ls_number', 'dataset'],
                    ascending=[False, True, True, True], kind='stable').iloc[0].copy()
                row['category'] = category
                examples.append(row)
                positions.append(int(row.name))
                record.update(dataset=row.dataset, run_number=int(row.run_number),
                              ls_number=int(row.ls_number), ls_end=int(row.get('ls_end', row.ls_number)),
                              n_ls=int(row.get('n_ls', 1)), eta_entries=int(row.eta_entries))
            inventory.append(record)
    return pd.DataFrame(examples, index=positions), pd.DataFrame(inventory)


def select_examples(scores):
    return case_inventory(scores)[0]


def plot_examples(saved, scores, dataset, features, decoded, folder):
    """Show actual Test categories with each LS's fitted and predicted parameters."""
    scores = scores.reset_index(drop=True)
    if not scores[IDS].equals(dataset['frame'][IDS].reset_index(drop=True)):
        raise ValueError('Plot inputs and predictions are not row-aligned')
    examples = select_examples(scores)
    if examples.empty:
        return pd.DataFrame()
    positions = examples.index.to_numpy(dtype=int)
    mode = saved.get('score_mode', CASE_SCORE)
    is_deviance = saved['settings']['loss'] == 'input_histogram_deviance'
    weighted_count = is_deviance and saved['settings'].get('loss_normalization') == 'me_bin'
    maximum = saved['settings']['max_scale']
    fitted_parameters = component_parameters(features[positions], saved['basis'], maximum)
    predicted_parameters = component_parameters(decoded[positions], saved['basis'], maximum)
    probabilities = decode_free(decoded[positions], saved['basis'], maximum)
    summary = []
    for i, (_, row) in enumerate(examples.iterrows()):
        output = Path(folder)/row.category
        output.mkdir(parents=True, exist_ok=True)
        for me, (stem, reference) in enumerate(zip(saved['me_names'], saved['basis'])):
            fit_basis = observation_basis(reference, fitted_parameters[me], i)
            predict_basis = observation_basis(reference, predicted_parameters[me], i)
            weights = [fitted_parameters[me]['weights'][i], predicted_parameters[me]['weights'][i]]
            observed, fit, prediction, residual = plot_me(
                row, stem, saved['axes'][me], dataset['histograms'][me][positions[i]],
                fit_basis, weights[0], weights[1], output/(stem+'.png'),
                predicted_basis=predict_basis,
                residual_scale=None if is_deviance else np.sqrt(
                    saved['threshold']*np.asarray(saved['loss_scales'][me], dtype=float)),
                deviance_scale=(np.asarray(saved['loss_scales'][me], dtype=float)*saved['me_loss_scales'][me]
                                if weighted_count else None))
            if not np.allclose(prediction, probabilities[me][i], rtol=2e-5, atol=2e-7):
                raise ValueError('Plotted Gaussian sum differs from the decoder')
            counts = dataset['histograms'][me][positions[i]]
            n = int(counts.sum())
            details = {}
            if is_deviance:
                statistics = count_statistics(counts, prediction)
                raw_column = f'raw_term_{me}' if weighted_count else f'term_{me}'
                if not np.isclose(statistics['deviance'], row[raw_column],
                                  rtol=2e-4, atol=1e-5, equal_nan=True):
                    raise ValueError('Plotted Input–Predict deviance differs from the stored ME score')
                if not np.isclose(statistics['pearson'], row[f'pearson_{me}'],
                                  rtol=2e-4, atol=1e-5, equal_nan=True):
                    raise ValueError('Plotted Pearson statistic differs from its stored value')
                value = np.sum(residual**2) if n else np.nan
                if not np.isclose(value, row[f'term_{me}'], rtol=2e-4, atol=1e-5, equal_nan=True):
                    raise ValueError('Plotted deviance residual does not reproduce the ME score')
                details = dict(input_predict_deviance=float(statistics['deviance']), pearson=float(statistics['pearson']),
                               p_deviance=float(statistics['p_deviance']), p_pearson=float(statistics['p_pearson']))
                if weighted_count:
                    details['weighted_deviance'] = value
            else:
                mse = np.mean((observed-prediction)**2) if n else np.nan
                normalized = np.mean(residual**2) if n else np.nan
                if not np.isclose(mse, row[f'raw_term_{me}'], rtol=2e-4, atol=1e-10, equal_nan=True):
                    raise ValueError('Plotted raw MSE differs from the stored ME score')
                if not np.isclose(normalized, row[f'term_{me}']/saved['threshold'],
                                  rtol=2e-4, atol=1e-10, equal_nan=True):
                    raise ValueError('Plotted normalized MSE differs from the stored ME score')
                details = dict(input_predict_MSE=mse, normalized_MSE=normalized)
            summary.append(dict(category=row.category, run=int(row.run_number), ls=int(row.ls_number),
                                ls_end=int(row.get('ls_end', row.ls_number)), n_ls=int(row.get('n_ls', 1)),
                                me=stem, native_N=n, score_mode=mode,
                                score_over_threshold=row.anomaly_score/row.case_threshold,
                                fit_TV=.5*np.abs(observed-fit).sum() if n else np.nan,
                                predict_TV=.5*np.abs(observed-prediction).sum() if n else np.nan,
                                **details))
        print(f'Saved {len(saved["me_names"])} MEs: GMM/{row.category}, '
              f'{row.run_number}/{row.ls_number}', flush=True)
    return pd.DataFrame(summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path,
                        default=MAIN_PREDICTIONS,
                        help='GMM prediction directory')
    args = parser.parse_args()
    torch.set_num_threads(4)
    hep.style.use('CMS')
    metadata = json.loads((args.input/'metrics.json').read_text())
    spec, saved, checkpoint = load_checkpoint(args.input/metadata['checkpoint'])
    if spec.KIND != 'gmm':
        raise ValueError('Expected GMM predictions')
    try:
        relative = args.input.resolve().relative_to((OUT/'predict').resolve())
    except ValueError:
        relative = Path(spec.NAME)
    source = args.input/'scores.parquet'
    inputs = (checkpoint, source, args.input/'metrics.json')
    protected = [hashlib.sha256(path.read_bytes()).digest() for path in inputs]
    canonical = pd.read_parquet(source)
    scores = case_scores(canonical, saved)
    selected, inventory = case_inventory(scores)
    inventory.to_csv(source.parent/'cases.csv', index=False)
    folder = OUT/'plots'/relative
    for row in inventory.itertuples():
        case_folder = folder/row.category
        case_folder.mkdir(parents=True, exist_ok=True)
        if not row.available:
            for old in case_folder.glob('*.png'):
                old.unlink()
    if selected.empty:
        print('No scored Test examples; availability recorded in cases.csv', flush=True)
        return
    dataset = load_histograms(saved, selected)
    dataset['_prefit_override'] = cached_prefit_rows(saved, metadata, canonical, selected)
    features, decoded, _ = inference(spec, saved, dataset)
    summary = plot_examples(saved, dataset['frame'], dataset, features, decoded, folder)
    summary.to_csv(source.parent/'component_examples.csv', index=False)
    assert protected == [hashlib.sha256(path.read_bytes()).digest() for path in inputs]
    print('Saved Gaussian components; checkpoint and scores unchanged', flush=True)


if __name__ == '__main__':
    main()
