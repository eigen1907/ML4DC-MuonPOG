"""Per-observation Gaussian fits and decoding with free weights, means and widths."""

from time import perf_counter

import numpy as np
from scipy.special import ndtr
import torch


def _components_torch(means, scales, edges, periodic):
    """Native-bin masses, reusing each edge CDF and wrapping phi."""
    probability = means.new_zeros((len(means), len(edges)-1, means.shape[1]))
    for image in (-2, -1, 0, 1, 2) if periodic else (0,):
        z = (edges[None, :, None]-means[:, None, :]-image)/scales[:, None, :]
        if means.device.type == 'mps':
            # Pairwise tails avoid gather's nondeterministic MPS backward.
            low, high = z[:, :-1], z[:, 1:]
            right = low > 0
            probability = probability + .5*(
                torch.erfc(-torch.where(right, -low, high)/np.sqrt(2))
                - torch.erfc(-torch.where(right, -high, low)/np.sqrt(2)))
            continue
        positive = z > 0
        tail = torch.erfc(torch.where(positive, z, -z)/np.sqrt(2))
        lower, upper = tail[:, :-1], tail[:, 1:]
        mass = .5*torch.where(positive[:, :-1], lower-upper, upper-lower)
        crossing = ~positive[:, :-1] & positive[:, 1:]
        first = (~positive).sum(1, keepdim=True).clamp_max(len(edges)-1)
        upper_cdf = torch.erfc(-torch.gather(z, 1, first)/np.sqrt(2))
        probability = probability + torch.where(crossing, .5*(upper_cdf-lower), mass)
    probability = probability.clamp_min(0)
    probability = probability/probability.sum(1, keepdim=True).clamp_min(1e-30)
    probability = probability.clamp_min(1e-12)
    return probability/probability.sum(1, keepdim=True)


def parameters_torch(features, item, max_scale=.5):
    """Project one ME's [weights, means, log-sigma] block to valid parameters."""
    k = len(item['weights'])
    if features.ndim != 2 or features.shape[1] != 3*k:
        raise ValueError('Expected a weight/mean/log-sigma block for one ME')
    weights = features[:, :k].clamp_min(0)
    total = weights.sum(1, keepdim=True)
    reference = torch.as_tensor(item['weights'], dtype=features.dtype, device=features.device)
    weights = torch.where(total > 0, weights/total.clamp_min(1e-30), reference)
    means = features[:, k:2*k]
    means = means.remainder(1) if item['periodic'] else means.clamp(0, 1)
    scales = features[:, 2*k:].clamp(np.log(item['floor']), np.log(max_scale)).exp()
    return weights, means, scales


def probabilities_torch(features, item, max_scale=.5):
    """Decode a single ME block differentiably, on CPU or MPS in float32."""
    weights, means, scales = parameters_torch(features, item, max_scale)
    edges = torch.as_tensor(item['edges'], dtype=features.dtype, device=features.device)
    return (_components_torch(means, scales, edges, item['periodic'])*weights[:, None, :]).sum(2)


def component_parameters(features, basis, max_scale=.5):
    """Decode NumPy feature rows, retaining separate Gaussian parameters."""
    features = np.asarray(features)
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError('Expected finite observation-by-parameter features')
    result, first = [], 0
    for item in basis:
        k = len(item['weights'])
        values = features[:, first:first+3*k].astype(np.float64)
        if values.shape[1] != 3*k:
            raise ValueError('Feature count differs from the Gaussian basis')
        weights = np.maximum(values[:, :k], 0)
        total = weights.sum(1, keepdims=True)
        weights = np.divide(weights, total, out=np.broadcast_to(item['weights'], weights.shape).copy(),
                            where=total > 0)
        means = values[:, k:2*k]
        means = means % 1 if item['periodic'] else np.clip(means, 0, 1)
        scales = np.exp(np.clip(values[:, 2*k:], np.log(item['floor']), np.log(max_scale)))
        result.append(dict(weights=weights, means=means, scales=scales))
        first += 3*k
    if first != features.shape[1]:
        raise ValueError('Feature count differs from the Gaussian basis')
    return result


def decode_free(features, basis, max_scale=.5):
    """Return normalized native-bin mixtures without fitting evaluated data."""
    result = []
    for values, item in zip(component_parameters(features, basis, max_scale), basis):
        means, scales = values['means'][:, None, :], values['scales'][:, None, :]
        edges = np.asarray(item['edges'])
        probability = np.zeros((len(means), len(edges)-1, means.shape[-1]))
        for image in (-2, -1, 0, 1, 2) if item['periodic'] else (0,):
            low = (edges[:-1][None, :, None]-means-image)/scales
            high = (edges[1:][None, :, None]-means-image)/scales
            right = low > 0
            probability += ndtr(np.where(right, -low, high))-ndtr(np.where(right, -high, low))
        probability = np.maximum(probability, 0)
        probability /= np.maximum(probability.sum(1, keepdims=True), 1e-300)
        probability = np.maximum(probability, 1e-12)
        probability /= probability.sum(1, keepdims=True)
        result.append((probability*values['weights'][:, None, :]).sum(2).astype(np.float32))
    return result


def prefit_free(histograms, basis, settings, initial_weights=None):
    """Fit independent LS mixtures in batches; no Dirichlet or shape prior.

    Train parameters initialize every fit. All sixteen means and widths can
    move throughout their valid range. The finite iteration budget is recorded;
    it must not be confused with proof that every LS reached convergence.
    """
    from scripts.models.shared_gaussian import fit_weights

    if not histograms or len(histograms) != len(basis):
        raise ValueError('Expected one histogram array and reference per ME')
    size = len(histograms[0])
    iterations = int(settings.get('prefit_iterations', 60))
    batch = int(settings.get('prefit_batch_size', 512))
    maximum = float(settings.get('max_scale', .5))
    device = torch.device(settings.get('prefit_device', 'cpu'))
    if iterations < 1 or batch < 1 or maximum <= 0:
        raise ValueError('Expected positive fitting settings')
    if initial_weights is None:
        initial_weights = fit_weights(histograms, basis, iterations=settings.get('weight_iterations', 60))['features']
    initial_weights = np.asarray(initial_weights)
    if (initial_weights.shape != (size, sum(len(item['weights']) for item in basis))
            or not np.isfinite(initial_weights).all() or (initial_weights < 0).any()):
        raise ValueError('Fixed-prefit initialization has a different shape')
    features, owners, names, diagnostics = [], [], [], []
    valid = np.zeros((size, len(basis)), dtype=bool)
    first = 0
    for me, (histogram, item) in enumerate(zip(histograms, basis)):
        started = perf_counter()
        histogram = np.asarray(histogram)
        k = len(item['weights'])
        if histogram.ndim != 2 or len(histogram) != size or histogram.shape[1] != len(item['edges'])-1:
            raise ValueError('Histogram dimensions differ from the reference')
        if maximum < item['floor']:
            raise ValueError('Maximum sigma is below the minimum sigma')
        output = np.zeros((size, 3*k), dtype=np.float32)
        gains, stable, fitted = 0., 0, 0
        for start in range(0, size, batch):
            stop = min(start+batch, size)
            counts = histogram[start:stop]
            if not np.isfinite(counts).all() or (counts < 0).any():
                raise ValueError('Expected finite nonnegative histogram counts')
            present = counts.sum(1) > 0
            valid[start:stop, me] = present
            if not present.any():
                continue
            rows = np.flatnonzero(present)+start
            counts_t = torch.as_tensor(counts[present], dtype=torch.float32, device=device)
            fractions = counts_t/counts_t.sum(1, keepdim=True)
            initial = initial_weights[rows, first:first+k]
            mu = np.clip(item['means'], 1e-6, 1-1e-6)
            logits = torch.tensor(np.log(np.maximum(initial, 1e-12)), dtype=torch.float32, device=device, requires_grad=True)
            locations = torch.tensor(np.broadcast_to(np.log(mu/(1-mu)), (len(rows), k)).copy(),
                                     dtype=torch.float32, device=device, requires_grad=True)
            log_scales = torch.tensor(np.broadcast_to(np.log(item['scales']), (len(rows), k)).copy(),
                                      dtype=torch.float32, device=device, requires_grad=True)
            edges = torch.as_tensor(item['edges'], dtype=torch.float32, device=device)
            optimizer = torch.optim.Adam([logits, locations, log_scales], lr=settings.get('prefit_lr', .04))
            best_loss = counts_t.new_full((len(rows),), float('inf'))
            best = counts_t.new_zeros((len(rows), 3*k))
            first_loss, previous, last_change = None, None, None
            for step in range(iterations+1):
                optimizer.zero_grad(set_to_none=True)
                weights = logits.softmax(1)
                means = locations.sigmoid()
                scales = log_scales.clamp(np.log(item['floor']), np.log(maximum)).exp()
                probability = (_components_torch(means, scales, edges, item['periodic'])*weights[:, None, :]).sum(2)
                loss = -(fractions*probability.clamp_min(1e-30).log()).sum(1)
                if first_loss is None:
                    first_loss = loss.detach()
                improved = loss.detach() < best_loss
                values = torch.cat((weights, means, scales.log()), 1).detach()
                best = torch.where(improved[:, None], values, best)
                best_loss = torch.minimum(best_loss, loss.detach())
                if previous is not None:
                    last_change = (loss.detach()-previous).abs()/previous.abs().clamp_min(1)
                previous = loss.detach()
                if step == iterations:
                    break
                loss.sum().backward()
                optimizer.step()
                with torch.no_grad():
                    log_scales.clamp_(np.log(item['floor']), np.log(maximum))
            if not torch.isfinite(best).all() or not torch.isfinite(best_loss).all():
                raise FloatingPointError('Nonfinite free-prefit parameters')
            values = best.cpu().numpy()
            order = np.argsort(values[:, k:2*k], axis=1, kind='stable')
            output[rows] = np.concatenate([np.take_along_axis(values[:, j*k:(j+1)*k], order, axis=1)
                                           for j in range(3)], axis=1)
            gains += float((first_loss-best_loss).sum().cpu())
            stable += int((last_change < settings.get('prefit_tolerance', 1e-6)).sum().cpu())
            fitted += len(rows)
            if start and start//batch % 20 == 0:
                print(f'Free prefit {item["stem"]}: {stop:,}/{size:,} LS, {perf_counter()-started:.0f}s', flush=True)
        features.append(output)
        owners.extend([me]*(3*k))
        names.extend(f'{item["stem"]}__g{i}_{kind}' for kind in ('weight', 'mean', 'log_sigma') for i in range(k))
        diagnostics.append(dict(me=item['stem'], observed_ls=fitted, seconds=perf_counter()-started,
            iterations=iterations, device=str(device), mean_NLL_improvement=gains/max(1, fitted),
            final_step_stable_fraction=stable/max(1, fitted)))
        print(f'Free prefit {item["stem"]}: complete in {diagnostics[-1]["seconds"]:.1f}s', flush=True)
        first += k
    return dict(features=np.concatenate(features, axis=1), valid=valid,
                feature_me=np.asarray(owners, dtype=np.int64), feature_names=names, diagnostics=diagnostics)
