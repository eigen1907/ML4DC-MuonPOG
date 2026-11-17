"""Shared moments, histogram inputs and training-only reference selections."""

import hashlib
import json
from pathlib import Path
import tempfile
from zipfile import BadZipFile

import numpy as np

from scripts.models.config import OUT, SEED, SPLIT_POLICY


MONITOR_LIMIT = 4096


def _prefit_identity(dataset, basis):
    """Hash the actual observations, independent of their quality labels."""
    from scripts.models.shared_gaussian import basis_fingerprint

    data = hashlib.sha256()
    columns = ['dataset', 'run_number', 'ls_number']
    columns += [name for name in ('ls_end', 'n_ls') if name in dataset['frame']]
    data.update(dataset['frame'][columns].to_csv(index=False).encode())
    data.update(json.dumps(dataset.get('ls_grouping'), sort_keys=True).encode())
    for name, histogram in zip(dataset['me_names'], dataset['histograms']):
        histogram = np.ascontiguousarray(histogram)
        data.update(name.encode())
        data.update(f'{histogram.shape}:{histogram.dtype}'.encode())
        data.update(memoryview(histogram))
    return data.hexdigest(), basis_fingerprint(basis)


def _prefit_controls(settings, free):
    controls = dict(version='prefit-v1', free=bool(free),
                    weight_iterations=int(settings['weight_iterations']), prior=0)
    if free:
        controls.update(iterations=int(settings.get('prefit_iterations', 60)),
                        lr=float(settings.get('prefit_lr', .04)),
                        batch_size=int(settings.get('prefit_batch_size', 512)),
                        device=str(settings.get('prefit_device', 'cpu')),
                        max_scale=float(settings.get('max_scale', .5)),
                        tolerance=float(settings.get('prefit_tolerance', 1e-6)),
                        dtype='float32', solver='Adam')
    else:
        controls.update(dtype='float64-EM/float32-output', solver='EM')
    return controls


def _validate_prefit(result, dataset, basis, free):
    rows, n_mes = len(dataset['frame']), len(basis)
    widths = [len(item['weights']) * (3 if free else 1) for item in basis]
    owner = np.repeat(np.arange(n_mes, dtype=np.int64), widths)
    names = [f'{item["stem"]}__g{i}_{kind}' for item in basis
             for kind in (('weight', 'mean', 'log_sigma') if free else ('weight',))
             for i in range(len(item['weights']))]
    features, valid = result['features'], result['valid']
    if (features.shape != (rows, len(owner)) or features.dtype != np.float32
            or valid.shape != (rows, n_mes) or valid.dtype != bool
            or result['feature_me'].dtype != np.int64
            or not np.array_equal(result['feature_me'], owner)
            or list(result['feature_names']) != names):
        raise ValueError('Prefit arrays, ME owners or feature names differ from this dataset')
    for me, histogram in enumerate(dataset['histograms']):
        if not np.array_equal(valid[:, me], histogram.sum(1, dtype=np.int64) > 0):
            raise ValueError('Cached prefit observation mask differs from native counts')
    for first in range(0, rows, 4096):
        observed = valid[first:first + 4096, owner]
        if not np.isfinite(features[first:first + 4096][observed]).all():
            raise ValueError('Nonfinite measured prefit features')


def _cached_prefit(dataset, basis, settings, free, identity=None):
    """Share fixed/free prefit results between training, prediction and models."""
    from scripts.models.shared_gaussian import fit_weights

    identity = _prefit_identity(dataset, basis) if identity is None else identity
    controls = _prefit_controls(settings, free)
    manifest = dict(data_sha256=identity[0], basis_sha256=identity[1],
                    controls=controls, rows=len(dataset['frame']), me_names=dataset['me_names'])
    key = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    kind = f'{"free" if free else "fixed"}{len(basis[0]["weights"])}'
    path = OUT/'dataset'/'prefit'/kind/(key+'.npz')
    use_disk = dataset.get('cache_prefit', True)
    cached = dataset.get('_shape_cache')
    if cached is not None and cached[0] == key:
        return cached[1]
    if use_disk and path.is_file():
        try:
            with np.load(path, allow_pickle=False) as stored:
                if json.loads(str(stored['manifest'].item())) != manifest:
                    raise ValueError('Prefit cache identity differs')
                result = dict(features=stored['features'], valid=stored['valid'],
                              feature_me=stored['feature_me'],
                              feature_names=stored['feature_names'].tolist(),
                              diagnostics=json.loads(str(stored['diagnostics'].item())))
            _validate_prefit(result, dataset, basis, free)
        except (OSError, ValueError, KeyError, EOFError, BadZipFile) as error:
            raise ValueError(f'Invalid prefit cache: {path}; remove it and retry') from error
        print(f'Loaded {kind} prefit cache: {path}', flush=True)
    else:
        if free:
            from scripts.models.free_gaussian import prefit_free

            fixed = _cached_prefit(dataset, basis, settings, False, identity)
            result = prefit_free(dataset['histograms'], basis, settings,
                                initial_weights=fixed['features'])
        else:
            result = fit_weights(dataset['histograms'], basis,
                                 iterations=settings['weight_iterations'])
        _validate_prefit(result, dataset, basis, free)
        if use_disk:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.tmp', delete=False) as handle:
                    temporary = Path(handle.name)
                    np.savez(handle, features=result['features'], valid=result['valid'],
                             feature_me=result['feature_me'],
                             feature_names=np.asarray(result['feature_names']),
                             manifest=json.dumps(manifest, sort_keys=True),
                             diagnostics=json.dumps(result.get('diagnostics', [])))
                temporary.replace(path)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            print(f'Saved {kind} prefit cache: {path}', flush=True)
    dataset['_shape_cache'] = (key, result)
    if use_disk:
        dataset.setdefault('prefit_cache', {})[kind] = dict(path=str(path), key=key)
    return result


def _moments(counts, low, high, order=2):
    """Bin-center moments on the signed native axis, rescaled affinely to 0–1.

    Eta is not folded: opposite detector hemispheres remain distinguishable.
    The third feature is population skewness. Empty or zero-variance
    histograms use zero for undefined skewness, following the empty-histogram
    mean/std convention. No bias or low-statistics correction is applied.
    """
    if order not in (2, 3):
        raise ValueError('Expected two or three moments per ME')
    low, high = float(low), float(high)
    edges = np.linspace(low, high, counts.shape[1] + 1, dtype=np.float32)
    centers = (edges[:-1] + edges[1:]) / 2
    positions = (centers - low) / (high - low)
    counts = counts.astype(np.float64)
    totals = counts.sum(axis=1)
    weights = np.divide(counts, totals[:, None], out=np.zeros_like(counts), where=totals[:, None] > 0)
    mean = weights @ positions
    std = np.sqrt((weights * (positions - mean[:, None]) ** 2).sum(axis=1))
    moments = [mean, std]
    if order > 2:
        valid = std > 0
        centered = np.divide(positions - mean[:, None], std[:, None],
                             out=np.zeros_like(weights), where=valid[:, None])
        squared = centered * centered
        moments.append((weights * squared * centered).sum(axis=1))
    return np.column_stack(moments)



def training_context(dataset):
    """Keep Good train, monitoring and calibration roles separate and reproducible."""
    from scripts.data.dials import fingerprint, validation_roles

    frame = dataset['frame'].copy()
    if dataset.get('split_policy') != SPLIT_POLICY:
        raise ValueError('Expected the shared RR/DCS/Muon-certified split policy')
    if not frame.eta_entries.gt(0).all():
        raise ValueError('Zero eta-entry LS must be excluded from all ML splits')
    roles = dataset.get('validation_roles') or validation_roles(frame, quality_split=True)
    if (frame.split.isin(('train', 'validation')) & ~frame.quality_pass).any():
        raise ValueError('Train/Validation contains LS failing the quality split policy')
    train = frame.split.eq('train').to_numpy()
    monitor = (frame.split.eq('validation') & frame.run_number.isin(roles['monitor_runs'])).to_numpy()
    calibration = (frame.split.eq('validation') & frame.run_number.isin(roles['calibration_runs'])).to_numpy()
    certified = frame.muon_certified.to_numpy(dtype=bool)
    fit_monitor = np.flatnonzero(monitor)
    fit_monitor = np.random.default_rng(SEED).choice(fit_monitor, min(MONITOR_LIMIT, len(fit_monitor)), replace=False)
    if (train & monitor).any() or (train & calibration).any() or (monitor & calibration).any():
        raise ValueError('Training, monitoring and calibration overlap')
    signature = dict(version='unconditioned-signed-moments-v3',
                     data_fingerprint=fingerprint(dataset), seed=SEED,
                     split_policy=dataset['split_policy'],
                     selection='Zero eta ROOT entries excluded from all ML splits; Train/Validation require '
                               'RR Good AND DCS-pass AND Muon-certified; remaining positive-eta LS Test',
                     muon_certification_sha256=hashlib.sha256(certified.astype(np.uint8).tobytes()).hexdigest(),
                     training_ls=int(train.sum()), monitor_runs=roles['monitor_runs'],
                     calibration_runs=roles['calibration_runs'])
    if dataset.get('ls_grouping'):
        signature['ls_grouping'] = dataset['ls_grouping']
    return dict(frame=frame, train=train, monitor_indices=fit_monitor,
                calibration=calibration, signature=signature)


def load_muon_certification(frame):
    """Return inclusive nominal and separate 2025/2026 low-PU certification.

    Files are public CMS JSON snapshots stored verbatim by the metadata muon
    recipe. Missing annual files are an error, not an uncertified verdict.
    This mask selects fitting LS; the split policy keeps the raw RR verdict
    separately when assigning operational Good/Bad labels.
    """
    from scripts.data.metadata import validate_lumi_mask

    certified = np.zeros(len(frame), dtype=bool)
    years = frame.dataset.str.extract(r'/Run(20\d\d)', expand=False)
    if years.isna().any():
        raise ValueError('Dataset year is required for Muon certification')
    for year in sorted(years.unique()):
        paths = [Path('data/rr') / f'{year}_muon.json']
        if year in ('2025', '2026'):
            paths.append(Path('data/rr') / f'{year}_lowPU.json')
        annual_positions = np.flatnonzero(years.eq(year).to_numpy())
        annual = frame.iloc[annual_positions][['run_number', 'ls_number']]
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f'Missing Muon certification: {path}; '
                                        f'run bash runs/data/muon_cert.sh --years {year}')
            intervals = json.loads(path.read_bytes())
            validate_lumi_mask(intervals)
            for run, positions in annual.groupby('run_number', sort=False).indices.items():
                rows = annual_positions[positions]
                ls = annual.iloc[positions].ls_number.to_numpy()
                for first, last in intervals.get(str(int(run)), []):
                    certified[rows] |= (ls >= first) & (ls <= last)
    return certified


def model_features(dataset, spec, settings, basis=None):
    """Return targets, observation masks and the ME owning each feature."""
    if spec.KIND == 'moments':
        order = settings.get('moment_order', 2)
        if order not in (2, 3):
            raise ValueError('Expected two or three moments per ME')
        kinds = ('mean', 'std', 'skewness')[:order]
        if order == 2:
            values = np.asarray(dataset['features'], dtype=np.float32)
        else:
            values = np.empty((len(dataset['features']), order * len(dataset['me_names'])), dtype=np.float32)
            for me, (histogram, axis) in enumerate(zip(dataset['histograms'], dataset['axes'])):
                for start in range(0, len(values), 4096):
                    stop = start + 4096
                    values[start:stop, order*me:order*(me+1)] = _moments(
                        histogram[start:stop], axis['low'], axis['high'],
                        order=order)
        totals = dataset.get('totals')
        if totals is None:
            totals = np.column_stack([histogram.sum(axis=1) for histogram in dataset['histograms']])
        mask = np.repeat(np.asarray(totals) > 0, order, axis=1)
        return values, mask, np.repeat(
            np.arange(len(dataset['me_names'])), order), [
            f'{me}__{kind}' for me in dataset['me_names'] for kind in kinds]
    if spec.KIND == 'gmm':
        if basis is None:
            raise ValueError('Density features require a trained basis/reference')
        result = dataset.get('_prefit_override')
        if result is None:
            result = _cached_prefit(dataset, basis, settings, bool(settings.get('free')))
        else:
            # Plotting selects exact rows from the full prefit cache. In
            # particular, GPU free fits must not be repeated in tiny batches.
            _validate_prefit(result, dataset, basis, bool(settings.get('free')))
        owner = result['feature_me']
        return result['features'], result['valid'][:, owner], owner, result['feature_names']
    raise ValueError('Unknown feature representation')
