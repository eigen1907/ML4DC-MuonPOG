"""Evaluate masked GMM MSE/deviance models and deviance p-value diagnostics."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

from scripts.data.dials import assign_quality_splits, fingerprint, load_dataset
from scripts.data.grouping import grouping_scope, replay_grouping
from scripts.models.config import (DIALS, GMM_MSE, GMM_DEVIANCE, GMM_NORM_DEVIANCE,
                                   OUT, REFERENCE_SCORE, MAIN_PREDICTIONS)
from scripts.models.core import NORMALIZATION_QUANTILE, THRESHOLD_QUANTILE, load_checkpoint, normalized_max
from scripts.plotting.scores import history_for, plot_stage, quality_labels


DIAGNOSTIC_PREFIXES = ('term_', 'raw_term_', 'pearson_', 'p_deviance_',
                       'p_pearson_', 'expected_low_fraction_', 'native_count_')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def roles(frame, saved):
    validation = frame.split.eq('validation') & frame.label.eq('good')
    monitor = (validation & frame.run_number.isin(saved['signature']['monitor_runs'])).to_numpy()
    calibration = (validation & frame.run_number.isin(saved['signature']['calibration_runs'])).to_numpy()
    if not monitor.any() or not calibration.any() or (monitor & calibration).any():
        raise ValueError('Expected separate nonempty Good monitoring and calibration runs')
    return monitor, calibration


def normal_scales(terms, monitor):
    scales = np.nanquantile(terms[monitor], NORMALIZATION_QUANTILE, axis=0)
    if not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError('Every ME needs a positive finite Good-monitoring error scale')
    return scales


def threshold(values, calibration, quantile=THRESHOLD_QUANTILE):
    if not np.isfinite(quantile) or not 0 < quantile < 1:
        raise ValueError('Expected a calibration quantile in (0, 1)')
    selected = values[calibration & np.isfinite(values)]
    cut = float(np.quantile(selected, quantile)) if len(selected) else np.nan
    if not np.isfinite(cut) or cut <= 0:
        raise ValueError('Expected a positive finite Good-calibration threshold')
    return cut


def summary(frame, values, cut, mode):
    labels = quality_labels(frame).to_numpy()
    test = frame.split.eq('test').to_numpy() & np.isfinite(values)
    flagged = values > cut
    weights = frame.n_ls.to_numpy() if 'n_ls' in frame else np.ones(len(frame), dtype=int)
    rows = []
    for unit, weight in (('observations', np.ones(len(frame), dtype=int)), ('original_ls', weights)):
        for label in ('Good', 'RR Bad', 'JSON Bad', 'DCS Bad', '2026C Good'):
            selected = test & (labels == ('Good' if label == '2026C Good' else label))
            if label == '2026C Good':
                selected &= frame.era.eq('2026C').to_numpy()
            total = int(weight[selected].sum())
            count = int(weight[selected & flagged].sum())
            rows.append(dict(score=mode, unit=unit, quality=label, threshold=cut,
                total=total, flagged=count, flag_fraction=count/total if total else None))
    return rows


def write_scales(path, names, scales, terms, monitor):
    pd.DataFrame(dict(ME=names, normal_MSE_q975=scales,
        observed_monitoring_points=np.isfinite(terms[monitor]).sum(0))).to_csv(path, index=False)


def raw_deviance_column(saved, me):
    """Keep count-model diagnostics separate from the weighted alarm score."""
    prefix = 'raw_term' if saved['settings'].get('loss_normalization') == 'me_bin' else 'term'
    return f'{prefix}_{me}'


def validate_statistics(saved, frame, terms):
    """Check the raw-count score and its explicitly approximate chi-square p-values."""
    train = np.flatnonzero(frame.split.eq('train'))
    if (saved['settings'].get('loss') != 'input_histogram_deviance'
            or saved['settings'].get('loss_normalization') not in (None, 'me_bin')
            or saved['quantile'] != THRESHOLD_QUANTILE or saved['training_ls'] != len(train)
            or not frame.iloc[train].label.eq('good').all()
            or not frame.iloc[train].quality_pass.all() or not frame.iloc[train].muon_certified.all()):
        raise ValueError('Expected raw-count deviance trained only on selected Good observations')
    for me, axis in enumerate(saved['axes']):
        raw = frame[raw_deviance_column(saved, me)].to_numpy(float)
        native_count = frame[f'native_count_{me}'].to_numpy(float)
        observed = native_count > 0
        if not np.isfinite(native_count).all() or (native_count < 0).any():
            raise ValueError('Expected nonnegative finite native histogram totals')
        np.testing.assert_array_equal(np.isfinite(terms[:, me]), observed)
        np.testing.assert_array_equal(np.isfinite(raw), observed)
        if (terms[observed, me] < 0).any():
            raise ValueError('Deviance must be nonnegative')
        dof = len(axis['edges'])-2
        if dof < 1:
            raise ValueError('Chi-square diagnostics need at least two native bins')
        for statistic, p_column in ((raw, f'p_deviance_{me}'),
                                    (frame[f'pearson_{me}'].to_numpy(float), f'p_pearson_{me}')):
            np.testing.assert_array_equal(np.isfinite(statistic), observed)
            if (statistic[observed] < 0).any():
                raise ValueError('Count statistics must be nonnegative')
            np.testing.assert_allclose(frame[p_column], chi2.sf(statistic, dof),
                                       rtol=1e-12, atol=1e-15, equal_nan=True)
        low = frame[f'expected_low_fraction_{me}'].to_numpy(float)
        np.testing.assert_array_equal(np.isfinite(low), observed)
        if ((low[observed] < 0) | (low[observed] > 1)).any():
            raise ValueError('Expected a fraction of sparse expected bins in [0, 1]')


def validate_weighted_deviance(saved, frame, terms):
    """Validate frozen 03 Good-Train weights, not a chi-square transformation."""
    info = saved['loss_normalization']
    train = np.flatnonzero(frame.split.eq('train'))
    if (saved['settings'].get('loss_normalization') != 'me_bin' or info['mode'] != 'me_bin'
            or info['reference'] != 'Frozen 03 Good Train deviance contributions'
            or info['source_experiment'] != GMM_DEVIANCE
            or info['quantile'] != NORMALIZATION_QUANTILE
            or info['training_observations'] != len(train)
            or info['training_rows_sha256'] != hashlib.sha256(train.astype('<i8').tobytes()).hexdigest()
            or info['floor_fraction'] != .01 or info['numerical_floor'] != 1e-6):
        raise ValueError('Expected frozen Good-Train deviance normalization provenance')
    source = Path(info['source_checkpoint'])
    digest = info['source_checkpoint_sha256']
    if (not isinstance(digest, str) or len(digest) != 64
            or any(char not in '0123456789abcdef' for char in digest)):
        raise ValueError('Expected the frozen source checkpoint SHA256')
    if source.is_file():
        if sha256(source) != digest:
            raise ValueError('Normalization source checkpoint has changed')
        _, original, _ = load_checkpoint(source)
        if (original['experiment'] != GMM_DEVIANCE or original['signature'] != saved['signature']
                or original['run_split'] != saved['run_split']
                or original['basis_sha256'] != saved['basis_sha256']):
            raise ValueError('Normalization source uses another Train cohort or basis')
    n_mes = len(saved['axes'])
    me_scales = np.asarray(saved['me_loss_scales'], dtype=float)
    if (len(saved['loss_scales']) != n_mes or len(info['per_me']) != n_mes
            or me_scales.shape != (n_mes,) or not np.isfinite(me_scales).all()
            or (me_scales < info['numerical_floor']).any()):
        raise ValueError('Every ME needs positive fixed bin and ME deviance scales')
    for me, (scale, me_scale, axis, record) in enumerate(zip(
            saved['loss_scales'], me_scales, saved['axes'], info['per_me'])):
        scale = np.asarray(scale, dtype=float)
        expected_floor = max(info['numerical_floor'], info['floor_fraction']*record['baseline_me_scale'])
        observed_train = int((frame.iloc[train][f'native_count_{me}'] > 0).sum())
        if (scale.shape != (len(axis['edges'])-1,) or not np.isfinite(scale).all()
                or record['me'] != saved['me_names'][me] or record['observed'] != observed_train
                or record['native_bins'] != len(scale) or record['quantile'] != info['quantile']
                or not np.isfinite(record['baseline_me_scale']) or record['baseline_me_scale'] < 0
                or not np.isclose(record['floor'], expected_floor, rtol=1e-12, atol=0)
                or (scale < record['floor']).any()):
            raise ValueError('Malformed fixed Good-Train deviance bin scales')
        raw = frame[f'raw_term_{me}'].to_numpy(float)
        observed = np.isfinite(raw)
        if (np.any(terms[observed, me] < raw[observed]/(scale.max()*me_scale)*(1-1e-8)-1e-12)
                or np.any(terms[observed, me] > raw[observed]/(scale.min()*me_scale)*(1+1e-8)+1e-12)):
            raise ValueError('Weighted deviance falls outside its positive scale bounds')


def validate_mse(saved, frame, terms):
    """Check the Good-Train bin scales shared by MSE learning and scoring."""
    train = np.flatnonzero(frame.split.eq('train'))
    info = saved['loss_normalization']
    if (saved['settings'].get('loss') != 'input_histogram_mse'
            or saved['settings'].get('loss_normalization') != 'bin' or info['mode'] != 'bin'
            or info['reference'] != 'Good Train mean Prefit histogram'
            or info['quantile'] != NORMALIZATION_QUANTILE
            or info['training_observations'] != len(train) or saved['training_ls'] != len(train)
            or info['training_rows_sha256'] != hashlib.sha256(train.astype('<i8').tobytes()).hexdigest()
            or not frame.iloc[train].label.eq('good').all()
            or not frame.iloc[train].quality_pass.all() or not frame.iloc[train].muon_certified.all()):
        raise ValueError('Expected bin-normalized MSE trained only on selected Good observations')
    if len(saved['loss_scales']) != len(saved['axes']) or len(info['per_me']) != len(saved['axes']):
        raise ValueError('Every ME needs its Train bin scales and provenance')
    for me, (scale, record, axis) in enumerate(zip(saved['loss_scales'], info['per_me'], saved['axes'])):
        scale = np.asarray(scale, dtype=float)
        if (scale.shape != (len(axis['edges'])-1,) or not np.isfinite(scale).all() or (scale <= 0).any()
                or record['me'] != saved['me_names'][me] or record['quantile'] != info['quantile']
                or not 0 < record['observed'] <= len(train)
                or not np.isfinite(record['floor']) or record['floor'] <= 0
                or (scale < record['floor']).any()):
            raise ValueError('Malformed positive Train bin scale or provenance')
        raw = frame[f'raw_term_{me}'].to_numpy(float)
        observed = np.isfinite(raw)
        np.testing.assert_array_equal(np.isfinite(terms[:, me]), observed)
        if np.isinf(raw).any() or (raw[observed] < 0).any() or (terms[observed, me] < 0).any():
            raise ValueError('Expected nonnegative raw and bin-normalized ME errors')
        if (np.any(terms[observed, me] < raw[observed]/scale.max()*(1-1e-8)-1e-12)
                or np.any(terms[observed, me] > raw[observed]/scale.min()*(1+1e-8)+1e-12)):
            raise ValueError('Normalized MSE falls outside its positive bin-scale bounds')


def reference_scores(source, metadata, frame, saved, checkpoint, scope):
    from scripts.models.model_00_ref_score import reference_terms

    output = OUT/'predict'/scope/REFERENCE_SCORE
    output.mkdir(parents=True, exist_ok=True)
    model_output = OUT/'model'/scope/REFERENCE_SCORE
    reference_path = model_output/'reference.npz'
    arrays = {}
    for i, (basis, axis) in enumerate(zip(saved['basis'], saved['axes'])):
        arrays[f'template_{i}'] = basis['probabilities'] @ basis['weights']
        arrays[f'edges_{i}'] = np.asarray(axis['edges'])
    reference_matches = False
    if reference_path.is_file():
        with np.load(reference_path) as reference:
            reference_matches = (set(reference.files) == set(arrays)
                and all(np.array_equal(reference[key], value) for key, value in arrays.items()))
    identity = dict(source_checkpoint_sha256=sha256(checkpoint),
                    source_scores_sha256=sha256(source/'scores.parquet'),
                    data_fingerprint=metadata['data_fingerprint'],
                    scoring='fixed_train_reference_normalized_max_v1')
    metrics_path, score_path = output/'metrics.json', output/'scores.parquet'
    reusable = False
    if metrics_path.is_file() and score_path.is_file():
        old = json.loads(metrics_path.read_text())
        # The reference depends on native data and the Train template, not
        # on NN weights, its objective, or the GMM prediction diagnostics.
        reusable = (reference_matches and old.get('scores_sha256') == sha256(score_path)
            and old.get('data_fingerprint') == metadata['data_fingerprint']
            and old.get('signature') == saved['signature']
            and old.get('split_policy') == saved['split_policy']
            and old.get('ls_grouping') == saved.get('ls_grouping')
            and old.get('scoring') == identity['scoring']
            and old.get('quantile') == THRESHOLD_QUANTILE
            and old.get('normalization_quantile') == NORMALIZATION_QUANTILE
            and (model_output/'scoring.json').is_file())
    columns = [name for name in frame if name not in ['anomaly_score', 'flagged']
               and not name.startswith(DIAGNOSTIC_PREFIXES)]
    if reusable:
        scores = pd.read_parquet(score_path)
        if not scores[columns].equals(frame[columns]):
            raise ValueError('Reference snapshot and GMM source rows differ')
        terms = scores[[f'term_{i}' for i in range(len(saved['me_names']))]].to_numpy(float)
        print(f'Compatible reference terms reused: {output}', flush=True)
    else:
        dataset = load_dataset(DIALS, split=False, years=metadata['years'],
                               prompt_reco=metadata['prompt_reco'])
        dataset['frame'], dataset['run_split'] = assign_quality_splits(
            dataset['frame'], saved['run_split'], policy=saved['split_policy'])
        dataset['split_policy'] = saved['split_policy']
        if saved.get('ls_grouping'):
            dataset = replay_grouping(dataset, saved['ls_grouping'])
        if (fingerprint(dataset) != metadata['data_fingerprint']
                or not dataset['frame'][columns].equals(frame[columns])):
            raise ValueError('Cached data no longer match the frozen prediction snapshot')
        terms = reference_terms(dataset, saved['basis'])
    monitor, calibration = roles(frame, saved)
    scales = normal_scales(terms, monitor)
    values = normalized_max(terms, scales)
    cut = threshold(values, calibration)
    if reusable:
        np.testing.assert_allclose(scores.anomaly_score, values, rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(scores.flagged.to_numpy(bool), values > cut)
        np.testing.assert_allclose(old['threshold'], cut, rtol=1e-12, atol=0)
        plot_stage(scores, None, cut, OUT/'plots'/scope/REFERENCE_SCORE)
        print(f'Compatible reference snapshot preserved: {output}', flush=True)
        return
    scores = frame[columns].assign(anomaly_score=values,
        flagged=pd.array(np.where(np.isfinite(values), values > cut, None), dtype='boolean'))
    for me in range(terms.shape[1]):
        scores[f'term_{me}'] = terms[:, me]
    scores.to_parquet(score_path, index=False)
    write_scales(output/'scales.csv', saved['me_names'], scales, terms, monitor)
    pd.DataFrame(summary(frame, values, cut, REFERENCE_SCORE)).assign(quantile=THRESHOLD_QUANTILE).to_csv(
        output/'summary.csv', index=False)
    record = dict(experiment=REFERENCE_SCORE, description='Fixed Good-Train GMM reference; no AE inference',
        checkpoint=os.path.relpath(checkpoint.resolve(), output.resolve()), quantile=THRESHOLD_QUANTILE,
        normalization_quantile=NORMALIZATION_QUANTILE, score_mode='norm_max',
        threshold=cut, years=metadata['years'], prompt_reco=metadata['prompt_reco'],
        signature=saved['signature'], split_policy=saved['split_policy'],
        ls_grouping=saved.get('ls_grouping'), grouping_summary=saved.get('grouping_summary'),
        observation_unit=metadata.get('observation_unit', 'ls_group' if saved.get('ls_grouping') else 'ls'),
        calibration=dict(ls=int(calibration.sum()),
        flagged_fraction=float((values[calibration] > cut).mean())),
        scores_sha256=sha256(score_path), **identity)
    metrics_path.write_text(json.dumps(record, indent=2)+'\n')
    model_output.mkdir(parents=True, exist_ok=True)
    if reference_path.is_file():
        with np.load(reference_path) as reference:
            if set(reference.files) != set(arrays):
                raise ValueError('Stored reference arrays differ from the source Train basis')
            for key, value in arrays.items():
                np.testing.assert_array_equal(reference[key], value)
    else:
        np.savez(reference_path, **arrays)
    scoring = dict(me_names=saved['me_names'], me_scales=scales.tolist(), threshold=cut,
        quantile=THRESHOLD_QUANTILE, normalization_quantile=NORMALIZATION_QUANTILE,
        score_mode='norm_max', run_split=saved['run_split'], split_policy=saved['split_policy'],
        validation_roles={key: saved['signature'][key] for key in ('monitor_runs', 'calibration_runs')},
        ls_grouping=saved.get('ls_grouping'), **identity)
    (model_output/'scoring.json').write_text(json.dumps(scoring, indent=2)+'\n')
    plot_stage(scores, None, cut, OUT/'plots'/scope/REFERENCE_SCORE)
    print(f'Reference-only results saved: {output} (threshold={cut:.6g})', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=MAIN_PREDICTIONS, help='GMM prediction directory')
    args = parser.parse_args()
    source = args.input
    metadata = json.loads((source/'metrics.json').read_text())
    spec, saved, checkpoint = load_checkpoint(source/metadata['checkpoint'])
    modes = {GMM_MSE: 'loss_norm_max', GMM_DEVIANCE: 'deviance_max',
             GMM_NORM_DEVIANCE: 'norm_deviance_max'}
    mode = modes.get(spec.NAME)
    if (spec.KIND != 'gmm' or mode is None
            or metadata.get('experiment') != spec.NAME
            or saved.get('score_mode') != mode or saved['settings'] != spec.SETTINGS
            or sha256(checkpoint) != metadata['checkpoint_sha256']
            or metadata.get('quantile') != saved.get('quantile')
            or saved.get('quantile') != THRESHOLD_QUANTILE
            or metadata.get('score_mode') != saved.get('score_mode')):
        raise ValueError('Expected matching q99 GMM predictions')
    protected = {path: sha256(path) for path in (checkpoint, source/'scores.parquet', source/'metrics.json')}
    frame = pd.read_parquet(source/'scores.parquet')
    scope = grouping_scope(saved.get('ls_grouping'))
    _, calibration = roles(frame, saved)
    terms = frame[[f'term_{i}' for i in range(len(saved['me_names']))]].to_numpy(float)
    if spec.NAME in (GMM_DEVIANCE, GMM_NORM_DEVIANCE):
        validate_statistics(saved, frame, terms)
        if spec.NAME == GMM_NORM_DEVIANCE:
            validate_weighted_deviance(saved, frame, terms)
    else:
        validate_mse(saved, frame, terms)
    scales = np.ones(terms.shape[1])
    np.testing.assert_allclose(scales, saved['me_scales'], rtol=1e-12, atol=0)
    values = normalized_max(terms, scales)
    cut = saved['threshold']
    if not np.isclose(metadata['threshold'], cut, rtol=1e-12, atol=0):
        raise ValueError('Prediction metadata and checkpoint thresholds differ')
    np.testing.assert_allclose(values, frame.anomaly_score, rtol=1e-12, atol=1e-12)
    if not np.isclose(threshold(values, calibration), cut, rtol=1e-12, atol=0):
        raise ValueError('Good calibration does not reproduce the saved threshold')
    np.testing.assert_array_equal(values > cut, frame.flagged.to_numpy(bool))
    pd.DataFrame(summary(frame, values, cut, saved['score_mode'])).assign(quantile=saved['quantile']).to_csv(
        source/'summary.csv', index=False)
    plot_stage(frame, history_for(checkpoint), cut, OUT/'plots'/scope/spec.NAME)
    if spec.NAME in (GMM_DEVIANCE, GMM_NORM_DEVIANCE):
        from scripts.plotting.validation import plot_checks

        plot_checks(spec, saved, metadata, frame, source, OUT/'plots'/scope/spec.NAME)
    reference_scores(source, metadata, frame, saved, checkpoint, scope)
    for path, digest in protected.items():
        if sha256(path) != digest:
            raise AssertionError(f'Frozen source changed: {path}')
    print(f'GMM plots saved: output/plots/{scope}/{spec.NAME}', flush=True)


if __name__ == '__main__':
    main()
