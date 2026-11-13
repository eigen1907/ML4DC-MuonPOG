"""Single-eta-histogram detector trained only on Good data and simulated losses.

Normal/fault pairs have exactly the same N and pile-up. Both histograms are
multinomial draws from the same Good-parent pilot, with only the fault shape
reweighted. This is conditional simulation, not independent clean measurements.
Scores rank resemblance to the simulated faults; they are not Bad probabilities.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn

from scripts.models.kde import train_reference
from scripts.models.preprocessing import _rows


SETTINGS = dict(seed=42, train_limit=16384, epochs=40, patience=6, batch_size=512,
                lr=.001, weight_decay=1e-4, prior_strength=8., bandwidth_bins=1.,
                min_fraction=.01, severity=(.25, .95))


def _settings(settings):
    settings = {} if settings is None else dict(settings)
    if settings.keys() - SETTINGS.keys():
        raise ValueError(f'Unknown classifier settings: {settings.keys() - SETTINGS.keys()}')
    settings = SETTINGS | settings
    for key in ('seed', 'train_limit', 'epochs', 'patience', 'batch_size'):
        if (isinstance(settings[key], (bool, np.bool_))
                or not isinstance(settings[key], (int, np.integer))
                or settings[key] < (0 if key == 'seed' else 1)):
            raise ValueError(f'Invalid integer setting: {key}')
    for key in ('lr', 'weight_decay', 'prior_strength', 'bandwidth_bins', 'min_fraction'):
        if not np.isfinite(settings[key]) or settings[key] < 0:
            raise ValueError(f'Invalid finite nonnegative setting: {key}')
    severity = np.asarray(settings['severity'], dtype=float)
    if (settings['lr'] <= 0 or settings['bandwidth_bins'] < 1
            or not 0 < settings['min_fraction'] <= 1 or severity.shape != (2,)
            or not np.isfinite(severity).all() or not 0 < severity[0] <= severity[1] < 1):
        raise ValueError('Expected positive learning rate, native-bin smoothing and loss range in (0,1)')
    return settings


def _observations(hist, pu):
    hist, pu = np.asarray(hist), np.asarray(pu, dtype=float)
    if (hist.ndim != 2 or hist.shape[1] < 2 or pu.shape != (len(hist),)
            or not np.isfinite(hist).all() or (hist < 0).any()
            or (not np.issubdtype(hist.dtype, np.integer)
                and (np.abs(hist - np.rint(hist)) > 1e-5).any())):
        raise ValueError('Expected aligned nonnegative integer eta histograms and pile-up')
    totals = hist.sum(1, dtype=float)
    valid = (totals > 0) & np.isfinite(pu) & (pu >= 0)
    return hist, pu, totals, valid


def _inputs(hist, pu):
    hist, pu, totals, valid = _observations(hist, pu)
    values = np.zeros((len(hist), hist.shape[1] + 2), dtype=np.float32)
    np.divide(hist, totals[:, None], out=values[:, :-2], where=totals[:, None] > 0)
    values[:, -2] = np.log1p(totals)
    values[:, -1] = np.where(np.isfinite(pu) & (pu >= 0), pu, 0)
    return values, valid


def _regions(axis, bins):
    edges = np.asarray(axis['edges'] if 'edges' in axis else
                       np.linspace(axis['low'], axis['high'], bins + 1), dtype=float)
    if (edges.shape != (bins + 1,) or not np.isfinite(edges).all()
            or (np.diff(edges) <= 0).any() or not edges[0] < 0 < edges[-1]):
        raise ValueError('Expected the signed physical eta axis, not a folded/normalized axis')
    centers = (edges[:-1] + edges[1:]) / 2
    regions = np.asarray([np.abs(centers) < .9, np.abs(centers) > 1.2])
    if not regions.any(1).all() or regions.all(1).any():
        raise ValueError('Eta bins must resolve both central and forward loss regions')
    return regions


def synthetic_pairs(hist, pu, indices, reference, axis, seed, settings=None):
    """Generate balanced normal/loss pairs from selected Good-parent histograms.

    Retained N uses a log-uniform fraction of each parent's N, rounded to at
    least one observation; no observed low-N parents are discarded. Reference
    kernels and prior must have been fitted on TRAIN only. Central/forward
    losses reduce efficiency by 25--95% by default, then renormalize the shape;
    both classes are drawn at the SAME N/PU. No rate-loss label cue is present.
    """
    settings = _settings(settings)
    hist, pu = np.asarray(hist), np.asarray(pu, dtype=float)
    if hist.ndim != 2 or pu.shape != (len(hist),):
        raise ValueError('Expected aligned eta histograms and pile-up')
    rows = _rows(indices, len(hist))
    counts, _, totals, valid = _observations(hist[rows], pu[rows])
    rows, counts, totals = rows[valid], counts[valid].astype(float), totals[valid]
    if not len(rows):
        raise ValueError('No nonempty selected Good histogram with observed pile-up')
    raw = np.asarray(reference['raw_reference'], dtype=float)
    kernel = np.asarray(reference['probabilities'], dtype=float)
    if raw.shape != (hist.shape[1],) or kernel.shape != (hist.shape[1], hist.shape[1]):
        raise ValueError('Training reference and eta bins differ')
    pilot = (counts + settings['prior_strength'] * raw) / (totals[:, None] + settings['prior_strength'])
    pilot = np.maximum(pilot @ kernel.T, 0)
    pilot /= pilot.sum(1, keepdims=True)
    rng = np.random.default_rng(seed)
    fraction = np.exp(rng.uniform(np.log(settings['min_fraction']), 0, len(rows)))
    n = np.maximum(1, np.rint(totals * fraction).astype(np.int64))
    family = rng.integers(0, 2, len(rows))
    severity = rng.uniform(*settings['severity'], len(rows))
    affected = _regions(axis, hist.shape[1])[family]
    fault = pilot * (1 - severity[:, None] * affected)
    fault /= fault.sum(1, keepdims=True)
    paired_counts = np.concatenate((rng.multinomial(n, pilot), rng.multinomial(n, fault)))
    paired_pu = np.tile(pu[rows], 2)
    features, _ = _inputs(paired_counts, paired_pu)
    labels = np.repeat(np.asarray([0., 1.], dtype=np.float32), len(rows))
    info = dict(source_indices=np.tile(rows, 2), counts=paired_counts, n=np.tile(n, 2),
                pileup=paired_pu, family=np.concatenate((np.full(len(rows), 'normal'),
                            np.asarray(['central', 'forward'])[family])),
                severity=np.concatenate((np.zeros(len(rows)), severity)))
    return features, labels, info


def _network(bins):
    return nn.Sequential(nn.Linear(bins + 2, 64), nn.SiLU(), nn.Linear(64, 16),
                         nn.SiLU(), nn.Linear(16, 1))


def _scaled(values, scaler):
    return torch.from_numpy(((values - scaler['mean']) / scaler['scale']).astype(np.float32))


def fit_classifier(hist, pu, train_indices, monitor_indices, axis, settings=None):
    """Fit on Good TRAIN simulations; stop on fixed Good MONITOR simulations.

    The caller supplies the frozen split. Training and monitoring selections
    must be disjoint; no real Bad samples or Test data enter model selection.
    The saved dictionary contains only CPU weights and simple reference/scaler
    arrays. Final thresholds belong to a separate Good calibration selection.
    """
    settings = _settings(settings)
    hist, pu = np.asarray(hist), np.asarray(pu, dtype=float)
    if hist.ndim != 2 or pu.shape != (len(hist),):
        raise ValueError('Expected aligned eta histograms and pile-up')
    train, monitor = (_rows(rows, len(hist)) for rows in (train_indices, monitor_indices))
    if np.intersect1d(train, monitor).size:
        raise ValueError('Good training and monitoring LS must be disjoint')
    rng = np.random.default_rng(settings['seed'])
    if len(train) > settings['train_limit']:
        train = np.sort(rng.choice(train, settings['train_limit'], replace=False))
    _, _, _, train_valid = _observations(hist[train], pu[train])
    _, _, _, monitor_valid = _observations(hist[monitor], pu[monitor])
    train, monitor = train[train_valid], monitor[monitor_valid]
    if not len(train) or not len(monitor):
        raise ValueError('Training and monitoring need nonempty eta histograms and observed pile-up')
    _regions(axis, hist.shape[1])
    reference = train_reference([hist[train]], [axis], np.ones(len(train), dtype=bool),
                                bandwidth_bins=settings['bandwidth_bins'])[0]
    training_values, _ = _inputs(hist[train], pu[train])
    fitted_scaler = StandardScaler().fit(training_values)
    scaler = dict(mean=fitted_scaler.mean_, scale=fitted_scaler.scale_)
    validation, labels, _ = synthetic_pairs(hist, pu, monitor, reference, axis,
                              settings['seed'] + 10000, settings)
    validation = _scaled(validation, scaler)
    labels = torch.from_numpy(labels)
    observed_monitor, _ = _inputs(hist[monitor], pu[monitor])
    observed_monitor = _scaled(observed_monitor, scaler)
    torch.manual_seed(settings['seed'])
    model = _network(hist.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['lr'],
                                  weight_decay=settings['weight_decay'])
    loss_function = nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(settings['seed'])
    history, best_loss, best_state, best_epoch = [], np.inf, None, 0
    for epoch in range(1, settings['epochs'] + 1):
        values, target, _ = synthetic_pairs(hist, pu, train, reference, axis,
                                           settings['seed'] + epoch, settings)
        values, target = _scaled(values, scaler), torch.from_numpy(target)
        order = torch.randperm(len(values), generator=generator)
        model.train()
        total_loss = 0.
        for start in range(0, len(values), settings['batch_size']):
            rows = order[start:start + settings['batch_size']]
            optimizer.zero_grad()
            loss = loss_function(model(values[rows]).squeeze(1), target[rows])
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(rows)
        model.eval()
        with torch.no_grad():
            monitor_loss = float(loss_function(model(validation).squeeze(1), labels))
            good_scores = model(observed_monitor).squeeze(1).sigmoid().numpy()
        history.append(dict(epoch=epoch, train_loss=total_loss / len(values),
                            monitor_loss=monitor_loss, monitor_good_mean=float(good_scores.mean()),
                            monitor_good_q975=float(np.quantile(good_scores, .975))))
        if monitor_loss < best_loss - 1e-5:
            best_loss, best_epoch = monitor_loss, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        elif epoch - best_epoch >= settings['patience']:
            print(f'eta classifier stopped at epoch {epoch}; best monitor BCE {best_loss:.5f}', flush=True)
            break
        if epoch == 1 or epoch % 5 == 0:
            print(f'eta classifier epoch {epoch:02d}: train BCE {total_loss / len(values):.5f}, '
                  f'monitor BCE {monitor_loss:.5f}, best {best_loss:.5f}', flush=True)
    return dict(state_dict=best_state, scaler=scaler, reference=reference, axis=dict(axis),
                settings=settings, history=history, best_epoch=best_epoch, bins=hist.shape[1],
                train_indices=train, monitor_indices=monitor,
                score='softplus(logit), ranking only')


def predict_classifier(hist, pu, saved, batch_size=4096, return_logits=False):
    """Return softplus(logit) rank scores; N=0 and missing PU remain NaN."""
    hist, pu, _, valid = _observations(hist, pu)
    if (hist.shape[1] != saved['bins'] or isinstance(batch_size, bool)
            or not isinstance(batch_size, (int, np.integer)) or batch_size < 1):
        raise ValueError('Expected saved eta binning and a positive batch size')
    model = _network(saved['bins'])
    model.load_state_dict(saved['state_dict'])
    model.eval()
    scores = np.full(len(hist), np.nan)
    rows = np.flatnonzero(valid)
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            selected = rows[start:start + batch_size]
            values, _ = _inputs(hist[selected], pu[selected])
            logits = model(_scaled(values, saved['scaler'])).squeeze(1)
            values = logits.numpy().astype(np.float64)
            scores[selected] = values if return_logits else np.logaddexp(0., values)
    return scores
