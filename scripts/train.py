"""Train one AE on certified Good LS and calibrate its fixed threshold."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
import torch

from scripts.data.dials import load_dataset
from scripts.data.grouping import group_lumisections, grouping_scope
from scripts.models.config import DEFAULT_LS_MODE, DIALS, EXPERIMENTS, GMM_DEVIANCE, LS_MODES, OUT, SEED
from scripts.models.core import NORMALIZATION_QUANTILE, THRESHOLD_QUANTILE, calibrate, load_checkpoint, model_spec, restore_model
from scripts.models.preprocessing import model_features, training_context


def feature_loss(predicted, observed, mask, feature_me, n_mes):
    """Average observed feature residuals within each ME, then across MEs."""
    indices = feature_me[None, :].expand(len(observed), -1)
    sums = observed.new_zeros((len(observed), n_mes)).scatter_add_(
        1, indices, torch.where(mask, (predicted - observed).square(), 0))
    counts = observed.new_zeros((len(observed), n_mes)).scatter_add_(1, indices, mask.to(observed.dtype))
    valid = counts > 0
    per_me = sums / counts.clamp_min(1)
    per_ls = torch.where(valid, per_me, 0).sum(1) / valid.sum(1).clamp_min(1)
    return per_ls[valid.any(1)].mean()


def fit_network(inputs, masks, owners, train, monitor, spec, *, scaler=None,
                basis=None, histograms=None, loss_scales=None, me_loss_scales=None):
    settings = spec.SETTINGS
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = spec.build_model(inputs.shape[1], settings)
    free = settings.get('free', False)
    device = torch.device(settings.get('training_device', 'cpu') if free else 'cpu')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings['lr'],
                                 weight_decay=settings['weight_decay'])
    batch_size = settings['batch_size']
    feature_me = torch.as_tensor(owners, dtype=torch.long)
    n_mes = int(feature_me.max()) + 1
    if basis is not None:
        mean, scale = torch.from_numpy(scaler.mean_), torch.from_numpy(scaler.scale_)
        if not free:
            raise ValueError('Gaussian AEs require free parameters')
        mean, scale = mean.to(device=device, dtype=torch.float32), scale.to(device=device, dtype=torch.float32)

    # Keep zero-copy CPU count tensors; only Train/monitoring batches enter loss.
    if free:
        if histograms is None or len(histograms) != n_mes:
            raise ValueError('Count deviance needs the native input histograms')
        native_counts = [torch.from_numpy(np.asarray(counts)) for counts in histograms]
        if loss_scales is not None:
            loss_scales = [torch.as_tensor(value, dtype=torch.float32, device=device) for value in loss_scales]
        if me_loss_scales is not None:
            me_loss_scales = torch.as_tensor(me_loss_scales, dtype=torch.float32, device=device)

    def free_objective(rows, monitoring):
        from scripts.models.free_gaussian import probabilities_torch

        width = settings['components'] * 3
        values = inputs[rows].to(device)
        observed = masks[rows].reshape(-1, n_mes, width).any(-1).to(device)

        def residual(predicted, me, selected):
            from scripts.models.histogram_statistics import deviance

            columns = slice(me*width, (me+1)*width)
            raw = predicted[:, columns]*scale[columns] + mean[columns]
            actual = native_counts[me][rows[selected.cpu().numpy()]].to(device=device, dtype=torch.float32)
            decoded = probabilities_torch(raw, basis[me])
            if settings.get('loss_normalization') == 'me_bin':
                from scripts.models.histogram_statistics import weighted_deviance

                return weighted_deviance(actual, decoded, loss_scales[me], me_loss_scales[me])
            if settings.get('loss_normalization') == 'bin':
                from scripts.models.loss_normalization import histogram_error

                fractions = actual/actual.sum(1, keepdim=True).clamp_min(1)
                return histogram_error(fractions-decoded, loss_scales[me])
            return deviance(actual, decoded)

        if monitoring:
            errors = values.new_zeros((len(rows), n_mes))
            for me in range(n_mes):
                selected = observed[:, me]
                if selected.any():
                    predicted = model(values[selected], observed[selected], me)
                    errors[selected, me] = residual(predicted, me, selected)
            return (errors.sum(1) / observed.sum(1).clamp_min(1)).mean()
        target_me = torch.rand(observed.shape, device=device).masked_fill(~observed, -1).argmax(1)
        predicted = model(values, observed, target_me)
        errors = values.new_zeros(len(rows))
        for me in range(n_mes):
            selected = target_me == me
            if selected.any():
                errors[selected] = residual(predicted[selected], me, selected)
        return errors.mean()

    def objective(rows, *, monitoring=False):
        if free:
            return free_objective(rows, monitoring)
        predicted = model(inputs[rows])
        return feature_loss(predicted, inputs[rows], masks[rows], feature_me, n_mes)

    history = dict(train=[], validation=[], best_epoch=0,
                   loss='Hidden-ME Input–Predict count deviance' if basis is not None else 'ME-balanced feature MSE')
    if settings.get('loss_normalization') == 'me_bin':
        history['loss'] = 'Hidden-ME Input–Predict deviance; Good Train ME/bin-normalized'
    elif loss_scales is not None:
        history['loss'] = 'Hidden-ME Input–Predict histogram MSE; Train bin-normalized'
    best, state, stale = float('inf'), None, 0
    for epoch in range(1, settings['epochs'] + 1):
        model.train()
        total = 0.
        shuffled = rng.permutation(train)
        for first in range(0, len(shuffled), batch_size):
            rows = shuffled[first:first + batch_size]
            optimizer.zero_grad(set_to_none=True)
            value = objective(rows)
            if not torch.isfinite(value):
                raise FloatingPointError('Nonfinite training loss')
            value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            total += float(value.detach()) * len(rows)
        model.eval()
        checked = 0.
        with torch.no_grad():
            for first in range(0, len(monitor), batch_size):
                rows = monitor[first:first + batch_size]
                checked += float(objective(rows, monitoring=True)) * len(rows)
        checked /= len(monitor)
        if not np.isfinite(checked):
            raise FloatingPointError('Nonfinite monitoring loss')
        history['train'].append(total / len(train))
        history['validation'].append(checked)
        print(f'{spec.NAME} epoch {epoch:03d}: train {total/len(train):.7g}, monitor {checked:.7g}', flush=True)
        if checked < best * (1 - 1e-4):
            best, state, stale = checked, copy.deepcopy(model.state_dict()), 0
            history['best_epoch'] = epoch
        else:
            stale += 1
            if stale >= settings['patience']:
                break
    model.load_state_dict(state)
    return dict(state_dict={key: value.detach().cpu() for key, value in state.items()},
                history=history, training_device=str(device))


def fit(dataset, context, spec, *, basis=None, normalization=None):
    settings = dict(spec.SETTINGS)
    saved = dict(settings=settings, signature=context['signature'], experiment=spec.NAME,
                 me_names=dataset['me_names'], axes=dataset['axes'], run_split=dataset['run_split'],
                 excluded_zero_eta_ls=dataset['excluded_zero_eta_ls'])
    saved.update(ls_grouping=dataset.get('ls_grouping'),
                 grouping_summary=dataset.get('grouping_summary'))
    if spec.KIND == 'gmm':
        from scripts.models.shared_gaussian import basis_fingerprint, fit_basis

        if settings.get('loss_normalization') == 'me_bin':
            from scripts.models.preprocessing import _prefit_controls

            source = OUT/'model'/grouping_scope(dataset.get('ls_grouping'))/GMM_DEVIANCE/'model.pt'
            if not source.is_file():
                raise FileNotFoundError(f'Train 03 for this LS mode before fitting 04 normalization: {source}')
            source_spec, baseline, source = load_checkpoint(source)
            if (baseline['signature'] != context['signature'] or baseline['run_split'] != dataset['run_split']
                    or baseline['me_names'] != dataset['me_names']
                    or _prefit_controls(baseline['settings'], True) != _prefit_controls(settings, True)
                    or any(baseline['settings'][key] != settings[key] for key in ('components', 'scale_floor_bins'))):
                raise ValueError('Train the matching 03 deviance model before estimating 04 normalization')
            basis = baseline['basis']
        saved['basis'] = basis if basis is not None else fit_basis(
            dataset['histograms'], dataset['axes'], context['train'],
            components=settings['components'], iterations=settings['basis_iterations'],
            reference_iterations=settings['reference_iterations'],
            scale_floor_bins=settings['scale_floor_bins'])
        saved['basis_sha256'] = basis_fingerprint(saved['basis'])
    features, mask, owners, names = model_features(dataset, spec, settings, basis=saved.get('basis'))
    features, mask = np.asarray(features, dtype=np.float32), np.asarray(mask, dtype=bool)
    if features.shape != mask.shape or not np.isfinite(features[mask]).all():
        raise ValueError('Expected finite aligned observed features')
    available = mask.any(1)
    train = np.flatnonzero(context['train'] & available)
    monitor = context['monitor_indices']
    monitor = monitor[available[monitor]]
    if not len(train) or not len(monitor) or np.intersect1d(train, monitor).size:
        raise ValueError('Expected nonempty disjoint Train and monitoring LS')
    if (mask[train].sum(0) == 0).any():
        raise ValueError('A feature has no observed Good training values')
    scaler = StandardScaler().fit(np.where(mask[train], features[train], np.nan))
    scaled = scaler.transform(features)
    inputs = torch.from_numpy(np.where(mask, scaled, 0).astype(np.float32))
    del scaled
    saved.update(scaler=scaler, features=list(names), feature_me=owners,
                 input_dim=inputs.shape[1], training_ls=len(train), monitoring_ls=len(monitor),
                 prefit_cache=dataset.get('prefit_cache', {}))
    if settings.get('masked'):
        saved['masking'] = dict(unit='whole ME', training='one random observed target per LS',
                               inference='one pass per target ME', availability_bits=len(dataset['me_names']))
    loss_scales, me_loss_scales = None, None
    if settings.get('loss_normalization') == 'bin':
        rows_sha = hashlib.sha256(train.astype('<i8').tobytes()).hexdigest()
        if normalization is None:
            from scripts.models.free_gaussian import decode_free
            from scripts.models.loss_normalization import fit_me_loss_reference

            references, loss_scales, records = [], [], []
            width = settings['components']*3
            valid = mask[:, ::width]
            for me, item in enumerate(saved['basis']):
                observed = train[valid[train, me]]
                columns = slice(me*width, (me+1)*width)

                def batches():
                    with ThreadPoolExecutor(max_workers=4) as executor:
                        for first in range(0, len(observed), settings['batch_size']):
                            parts = [part for part in np.array_split(observed[first:first+settings['batch_size']], 4) if len(part)]
                            yield np.concatenate(list(executor.map(
                                lambda rows: decode_free(features[rows, columns], [item])[0], parts)))

                record = fit_me_loss_reference(batches, quantile=NORMALIZATION_QUANTILE)
                references.append(record.pop('mean'))
                loss_scales.append(record.pop('bin_scales'))
                records.append(dict(me=dataset['me_names'][me], **record))
            normalization = dict(loss_scales=loss_scales, loss_reference=references,
                loss_normalization=dict(mode='bin', reference='Good Train mean Prefit histogram',
                    quantile=NORMALIZATION_QUANTILE, floor_fraction=.01, numerical_floor=1e-12,
                    training_observations=len(train), training_rows_sha256=rows_sha, per_me=records))
        info = normalization['loss_normalization']
        if (info['mode'] != 'bin' or info['quantile'] != NORMALIZATION_QUANTILE
                or info['training_rows_sha256'] != rows_sha or info['training_observations'] != len(train)
                or [record['me'] for record in info['per_me']] != dataset['me_names']):
            raise ValueError('Saved bin normalization does not match the Good Train selection')
        loss_scales = normalization['loss_scales']
        if len(loss_scales) != len(dataset['axes']):
            raise ValueError('Every ME needs its fixed native-bin loss scale')
        for value, axis in zip(loss_scales, dataset['axes']):
            value = np.asarray(value)
            if value.shape != (len(axis['edges'])-1,) or not np.isfinite(value).all() or (value <= 0).any():
                raise ValueError('Expected positive fixed native-bin loss scales')
        saved.update(copy.deepcopy(normalization))
    elif settings.get('loss_normalization') == 'me_bin':
        from scripts.models.free_gaussian import decode_free
        from scripts.models.histogram_statistics import deviance_bins
        from scripts.models.loss_normalization import fit_deviance_reference

        np.testing.assert_array_equal(scaler.mean_, baseline['scaler'].mean_)
        np.testing.assert_array_equal(scaler.scale_, baseline['scaler'].scale_)
        source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
        teacher = restore_model(source_spec, baseline)
        width = settings['components']*3
        observed_mes = torch.from_numpy(mask[:, ::width].copy())
        loss_scales, me_loss_scales, records = [], [], []
        for me, item in enumerate(saved['basis']):
            observed = train[mask[train, me*width]]
            columns = slice(me*width, (me+1)*width)

            def batches():
                with torch.inference_mode():
                    for first in range(0, len(observed), settings['batch_size']):
                        rows = observed[first:first+settings['batch_size']]
                        outputs = teacher(inputs[rows], observed_mes[rows], me).numpy()
                        parameters = baseline['scaler'].inverse_transform(outputs)
                        q = decode_free(parameters[:, columns], [item], baseline['settings']['max_scale'])[0]
                        yield deviance_bins(dataset['histograms'][me][rows], q)

            record = fit_deviance_reference(batches, observed=len(observed), quantile=NORMALIZATION_QUANTILE)
            loss_scales.append(record.pop('bin_scales'))
            me_loss_scales.append(record.pop('me_scale'))
            records.append(dict(me=dataset['me_names'][me], **record))
            print(f'{spec.NAME}: frozen Train deviance scales ready for {dataset["me_names"][me]}', flush=True)
        if hashlib.sha256(source.read_bytes()).hexdigest() != source_sha:
            raise ValueError('Source deviance checkpoint changed while fitting normalization')
        me_loss_scales = np.asarray(me_loss_scales, dtype=np.float64)
        saved.update(loss_scales=loss_scales, me_loss_scales=me_loss_scales,
            loss_normalization=dict(mode='me_bin', reference='Frozen 03 Good Train deviance contributions',
                quantile=NORMALIZATION_QUANTILE, floor_fraction=records[0]['floor_fraction'],
                numerical_floor=records[0]['numerical_floor'],
                source_checkpoint=str(source), source_checkpoint_sha256=source_sha,
                source_experiment=GMM_DEVIANCE, training_observations=len(train),
                training_rows_sha256=hashlib.sha256(train.astype('<i8').tobytes()).hexdigest(), per_me=records))
    saved.update(fit_network(inputs, torch.from_numpy(mask), owners, train, monitor, spec,
                             scaler=scaler, basis=saved.get('basis'),
                             histograms=dataset['histograms'], loss_scales=loss_scales, me_loss_scales=me_loss_scales))
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, choices=[f'model_{name}' for name in EXPERIMENTS])
    parser.add_argument('--data', type=Path, default=DIALS)
    parser.add_argument('--years', type=int, nargs='+')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--ls-mode', choices=LS_MODES, default=DEFAULT_LS_MODE,
                        help='Original LS, 1,000 Global Muon eta entries, or recorded-luminosity groups')
    args = parser.parse_args()
    spec = model_spec(args.model)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    dataset = load_dataset(args.data, split=True, years=args.years, quality_split=True)
    if args.ls_mode == 'ls_group_1000glb':
        dataset = group_lumisections(dataset)
    elif args.ls_mode in ('ls_group_7pb', 'ls_group_1pb'):
        dataset = group_lumisections(dataset, min_recorded_lumi=7. if args.ls_mode == 'ls_group_7pb' else 1.)
    relative = Path(grouping_scope(dataset.get('ls_grouping'))) / spec.NAME
    folder = args.output or OUT/'model'/relative
    path = folder/'model.pt'
    context = training_context(dataset)
    score_mode = ('norm_deviance_max' if spec.SETTINGS.get('loss_normalization') == 'me_bin' else
                  'loss_norm_max' if spec.SETTINGS.get('loss_normalization') == 'bin' else
                  'deviance_max' if spec.KIND == 'gmm' else 'mean')
    basis, normalization = None, None
    if path.exists():
        saved = torch.load(path, map_location='cpu', weights_only=False)
        same_data = (saved['experiment'] == spec.NAME and saved['signature'] == context['signature']
                     and saved['run_split'] == dataset['run_split'])
        if not same_data:
            raise ValueError('Saved model uses different data or settings; choose another --output')
        if saved['settings'] != spec.SETTINGS:
            controls = lambda settings: {k: v for k, v in settings.items() if k not in ('loss', 'loss_normalization')}
            if spec.KIND != 'gmm' or controls(saved['settings']) != controls(spec.SETTINGS):
                raise ValueError('Saved model uses different feature/model settings; choose another --output')
            basis = saved['basis']
            if spec.SETTINGS.get('loss_normalization') == 'bin' and saved.get('loss_normalization', {}).get('mode') == 'bin':
                normalization = {key: saved[key] for key in ('loss_scales', 'loss_reference', 'loss_normalization')}
            print('Objective changed: retraining from seed with the same Train basis and cached prefits', flush=True)
            saved = None
        elif (saved.get('score_mode', 'mean') == score_mode
              and saved['quantile'] == THRESHOLD_QUANTILE):
            print(f'Compatible checkpoint preserved: {path}', flush=True)
            return
        else:
            print(f'Compatible NN preserved; recalibrating: {path}', flush=True)
    else:
        saved = None
    if saved is None:
        saved = fit(dataset, context, spec, basis=basis, normalization=normalization)
        saved.update(split_policy=dataset['split_policy'],
                     training_years=sorted(dataset['frame'].loc[context['train']].dataset.str.extract(
                         r'/Run(20\d\d)')[0].astype(int).unique().tolist()))
    from scripts.predict import predict_terms
    saved.update(score_mode=score_mode,
                 normalization_quantile=NORMALIZATION_QUANTILE if 'loss_scales' in saved else None)
    scales = np.ones(len(saved['me_names'])) if spec.KIND == 'gmm' else None
    if scales is not None:
        saved['me_scales'] = scales
    # Calibrate on Validation; the predict command scores the full dataset once.
    calibration_rows = np.flatnonzero(dataset['frame'].split.eq('validation'))
    terms = predict_terms(spec, saved, dataset, indices=calibration_rows)
    scores, threshold = calibrate(dataset['frame'], terms, context['calibration'],
                                  scales=scales, quantile=THRESHOLD_QUANTILE)
    reference = scores.loc[context['calibration'], 'anomaly_score'].dropna().to_numpy()
    saved.update(threshold=threshold, quantile=THRESHOLD_QUANTILE,
                 calibration=dict(ls=len(reference), flagged_fraction=float((reference > threshold).mean())))
    folder.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save(saved, temporary)
    temporary.replace(path)
    print(f'Checkpoint saved: {path} (threshold={threshold:.6g})', flush=True)


if __name__ == '__main__':
    main()
