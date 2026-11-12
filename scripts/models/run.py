"""Run five Muon AE experiments on cached histograms; no entries cut."""

import argparse
import json
import pickle

import numpy as np
import pandas as pd
import torch

from scripts.config import (BASE, BASE_DCS, DM_DCS, DM_WINDOW, EXPERIMENTS, HIST_DCS,
                            OUT, SEED, SPECIAL_RUNS, WINDOW_SIZE)
from scripts.data.dataset import fingerprint, load_dataset, load_run_labels, validation_roles
from scripts.models.autoencoders import HistogramAE, fit_base, fit_histogram, prepare
from scripts.plotting.scores import plot_stage
from scripts.models.scoring import QUANTILE, base_terms, calibrate, histogram_terms, window_mean

STAGES = tuple(EXPERIMENTS)
MONITOR_LIMIT = 4096


def history_for(stage):
    if stage in (BASE, BASE_DCS):
        saved = pickle.loads((OUT/'model'/stage/'model.pkl').read_bytes())
        return dict(train=saved['train_losses'], validation=saved['validation_losses'], loss='Feature MSE')
    if stage in (HIST_DCS, DM_DCS):
        return torch.load(OUT/'model'/stage/'model.pt', map_location='cpu', weights_only=False)['history']
    return None


def save_stage(stage, frame, terms, calibration, signature, values=None):
    scores, cut = calibrate(frame, terms, calibration, values)
    folder = OUT/'predict'/stage
    folder.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(folder/'scores.parquet', index=False)
    (folder/'metrics.json').write_text(json.dumps(dict(signature=signature, threshold=cut,
        quantile=QUANTILE, description=EXPERIMENTS[stage],
        source_model=DM_DCS if stage == DM_WINDOW else stage,
        training_selection=('Historical Good training LS, without DCS selection' if stage == BASE
                            else signature['selection']),
        interpretation='Exploratory LS score under RR run labels; not P(bad)'), indent=2)+'\n')
    plot_stage(scores, history_for(stage), cut, OUT/'plots'/stage)
    print(stage, 'complete', flush=True)
    return scores


def read_saved_scores(path):
    """Display current RR labels with historical scores, cuts and splits unchanged."""
    scores = pd.read_parquet(path)
    try:
        labels = load_run_labels()
    except FileNotFoundError:
        scores.loc[scores.label.str.startswith('bad_', na=False), 'label'] = 'bad'
    else:
        scores['label'] = scores.run_number.map(labels).fillna('unknown')
    return scores


def draw_saved():
    for stage in STAGES:
        folder = OUT/'predict'/stage
        scores = read_saved_scores(folder/'scores.parquet')
        cut = json.loads((folder/'metrics.json').read_text())['threshold']
        plot_stage(scores, history_for(stage), cut, OUT/'plots'/stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plots-only', action='store_true')
    args = parser.parse_args()
    if args.plots_only:
        draw_saved()
        return
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    dataset = load_dataset()
    frame = dataset['frame'].copy()
    roles = validation_roles(frame)
    normal = frame.label.eq('good') & ~frame.run_number.isin(SPECIAL_RUNS)
    train = (normal & frame.split.eq('train') & frame.dcs_pass).to_numpy()
    monitor = (normal & frame.split.eq('validation') & frame.dcs_pass & frame.run_number.isin(roles['monitor_runs'])).to_numpy()
    calibration = (normal & frame.split.eq('validation') & frame.dcs_pass & frame.run_number.isin(roles['calibration_runs'])).to_numpy()
    fit_monitor = np.flatnonzero(monitor)
    fit_monitor = np.random.default_rng(SEED).choice(fit_monitor, min(MONITOR_LIMIT, len(fit_monitor)), replace=False)
    if (train & monitor).any() or (train & calibration).any() or (monitor & calibration).any():
        raise ValueError('Training, monitoring and calibration overlap')
    signature = dict(version='five-ae-dcs-v1', data_fingerprint=fingerprint(dataset), seed=SEED,
                     selection='Good training runs AND DCS-pass; no entry/statistics cut',
                     training_ls=int(train.sum()), monitor_runs=roles['monitor_runs'],
                     calibration_runs=roles['calibration_runs'])
    features = [f'{me}__{kind}' for me in dataset['me_names'] for kind in ('mean', 'std')]
    original_path = OUT/'model'/BASE/'model.pkl'
    if not original_path.exists():
        print('Reproducing historical feature-AE recipe; no original checkpoint was provided', flush=True)
        original = fit_base(dataset['features'], frame.split.eq('train').to_numpy(),
                            np.flatnonzero(frame.split.eq('validation')))
        original.update(features=features, run_split=dataset['run_split'], origin='Historical recipe refit')
        _, historical_scores = base_terms(dataset['features'], original)
        original['threshold'] = float(np.quantile(historical_scores[frame.split.eq('validation')], .95))
        original_path.parent.mkdir(parents=True, exist_ok=True)
        original_path.write_bytes(pickle.dumps(original))
    original = pickle.loads(original_path.read_bytes())
    if original['features'] != features or original['run_split'] != dataset['run_split']:
        raise ValueError('RR labels or run split changed; existing models/results are preserved. Use --plots-only for historical results or start a new fit in a separate output directory')
    cached = {}
    for stage in (BASE_DCS, HIST_DCS, DM_DCS):
        path = OUT/'model'/stage/('model.pkl' if stage == BASE_DCS else 'model.pt')
        if path.exists():
            cached[stage] = (pickle.loads(path.read_bytes()) if stage == BASE_DCS else
                             torch.load(path, map_location='cpu', weights_only=False))
            if cached[stage].get('signature') != signature:
                raise ValueError(f'{stage}: RR labels or training selection changed; existing results are preserved. Use --plots-only or start a separate fit')
    terms, values = base_terms(dataset['features'], original)
    save_stage(BASE, frame, terms, calibration, signature, values)
    path = OUT/'model'/BASE_DCS/'model.pkl'
    if path.exists():
        selected = cached[BASE_DCS]
    else:
        print('Training DCS-only feature AE on', int(train.sum()), 'LS', flush=True)
        selected = fit_base(dataset['features'], train, fit_monitor)
        selected.update(features=features, run_split=dataset['run_split'], signature=signature)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(selected))
    terms, values = base_terms(dataset['features'], selected)
    save_stage(BASE_DCS, frame, terms, calibration, signature, values)
    inputs, scaler = prepare(dataset, train)
    frames = {}
    for stage, distribution, score in ((HIST_DCS, 'mse', 'mse'), (DM_DCS, 'dm', 'weighted')):
        path = OUT/'model'/stage/'model.pt'
        if path.exists():
            saved = cached[stage]
            same_axes = (len(saved['axes']) == len(dataset['axes']) and all(
                np.array_equal(old[key], current[key])
                for old, current in zip(saved['axes'], dataset['axes'])
                for key in ('low', 'high', 'edges')))
            if (saved['me_names'] != dataset['me_names']
                    or tuple(saved['bin_sizes']) != tuple(h.shape[1] for h in dataset['histograms'])
                    or not same_axes
                    or any(not np.array_equal(getattr(saved['scaler'], key), getattr(scaler, key))
                           for key in ('mean_', 'scale_'))):
                raise ValueError(f'{stage}: cached ME layout, axes or training scaling changed')
            model = HistogramAE(saved['bin_sizes'], distribution)
            model.load_state_dict(saved['state_dict'])
        else:
            print('Training', stage, 'on', int(train.sum()), 'DCS-only Good LS', flush=True)
            model, history = fit_histogram(dataset, inputs, train, fit_monitor, distribution)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(dict(signature=signature, state_dict=model.state_dict(), scaler=scaler, history=history,
                            bin_sizes=model.bin_sizes, me_names=dataset['me_names'], axes=dataset['axes']), path)
        terms = histogram_terms(model, inputs, dataset['histograms'])[score]
        frames[stage] = save_stage(stage, frame, terms, calibration, signature)
    values = window_mean(frames[DM_DCS])
    save_stage(DM_WINDOW, frame, values[:, None], calibration, signature, values)
    (OUT/'model'/DM_WINDOW).mkdir(parents=True, exist_ok=True)
    (OUT/'model'/DM_WINDOW/'source.json').write_text(json.dumps(dict(
        source_model=DM_DCS, window=WINDOW_SIZE, centered=True), indent=2)+'\n')
    print('Five-stage study complete:', OUT/'predict', flush=True)


if __name__ == '__main__':
    main()
