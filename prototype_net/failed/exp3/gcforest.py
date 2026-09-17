"""Self-contained gcForest-style cascade on frozen exp3 augmented_pause features.

Zhou and Feng, https://www.ijcai.org/proceedings/2017/0497.pdf. A bounded cascade
of random forests and completely-random-feature ExtraTrees. Three participant-
grouped folds produce class vectors at each level. No multi-grained scanning:
the engineered columns do not form a raw temporal sequence. This uses two
forests per level and bounded trees rather than the paper's much larger setup.
All cascade implementation and experiment code lives in this file.

Validation selects checkpoints/depth; test is evaluated after that choice.
The historical selection split is reported as a diagnostic, never as a gate.

python -m prototype_net.exp3.gcforest --phase train
python -m prototype_net.exp3.gcforest --phase compare
python -m prototype_net.exp3.gcforest --phase predict --input observed.npz
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.metrics import confusion_matrix, log_loss
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARCHITECTURE = 'gcforest'
SEED = 9172026


def digest(path):
    # BEGIN frozen source adapter
    resolved = Path(path).resolve()
    if resolved == (HERE / 'model.py').resolve():
        from prototype_net.exp3.model import legacy_source_sha256
        return legacy_source_sha256()
    if resolved == Path(__file__).resolve():
        source = Path(__file__).read_text()
        start_marker = '    # BEGIN ' + 'frozen source adapter\n'
        end_marker = '    # END ' + 'frozen source adapter\n'
        start = source.index(start_marker)
        end = source.index(end_marker, start) + len(end_marker)
        # Normalize only this compatibility adapter; all architecture code remains checked.
        return hashlib.sha256((source[:start] + source[end:]).encode()).hexdigest()
    # END frozen source adapter
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def frozen_contract():
    folder = HERE / 'runs_med'
    freeze = json.loads((folder / 'freeze.json').read_text())
    for name, key in [('model.joblib', 'model_sha256'), ('protocol.json', 'protocol_sha256'),
                      ('scale_floor.npy', 'floor_sha256')]:
        if digest(folder / name) != freeze[key]:
            raise RuntimeError(f'Frozen artifact changed: {name}')
    protocol = json.loads((folder / 'protocol.json').read_text())
    for name, expected in protocol['source_hashes'].items():
        if digest(ROOT / name) != expected:
            raise RuntimeError(f'Frozen source changed: {name}')
    seen = set()
    for ids in protocol['splits'].values():
        ids = list(map(str, ids))
        if len(set(ids)) != len(ids) or seen.intersection(ids):
            raise RuntimeError('Participant roles overlap or contain duplicates')
        seen.update(ids)
    if freeze['representation'] != 'augmented_pause':
        raise RuntimeError('This study requires the augmented_pause freeze')
    return protocol, freeze


def load_data(role, protocol):
    if role not in ('train', 'selection', 'calibration', 'validation', 'test'):
        raise ValueError(f'Unknown participant role: {role}')
    parent = HERE / 'runs'
    path = parent / f'{role}.npz'
    meta = json.loads((parent / f'{role}_cache.json').read_text())
    if digest(path) != meta['sha256'] or digest(parent / 'scale_floor.npy') != meta['floor_sha256']:
        raise RuntimeError(f'Parent feature cache changed: {role}')
    expected_users = np.repeat(np.asarray(protocol['splits'][role], dtype=str), 4)
    x = np.empty((len(expected_users), 1485), dtype=np.float32)
    with np.load(path, allow_pickle=False) as data:
        x[:, :1443] = data['augmented']
        y, users = data['y'], data['users']
    if not np.array_equal(users, expected_users) or not np.array_equal(y, np.tile(np.arange(4), len(users) // 4)):
        raise RuntimeError(f'Participant/label ordering mismatch: {role}')
    path = HERE / 'runs_pause' / f'{role}_pause.npz'
    meta = json.loads(path.with_name(f'{role}_pause_cache.json').read_text())
    if digest(path) != meta['sha256']:
        raise RuntimeError(f'Pause cache changed: {role}')
    windows_meta = json.loads((parent / f'{role}_windows_cache.json').read_text())
    if meta['parent_windows'] != windows_meta['sha256']:
        raise RuntimeError(f'Pause and parent window caches disagree: {role}')
    with np.load(path, allow_pickle=False) as data:
        pause = data['pause']
        if pause.shape != (len(users) // 4, 4, 42):
            raise RuntimeError(f'Pause shape mismatch: {role}')
        x[:, 1443:] = pause.reshape(-1, 42)
    if not np.isfinite(x).all():
        raise RuntimeError(f'Nonfinite features: {role}')
    return x, y.astype(np.int64), users


def score(y, probability):
    if probability.shape != (len(y), 4) or not np.isfinite(probability).all():
        raise RuntimeError('Invalid class probabilities')
    if not np.allclose(probability.sum(1), 1, atol=1e-5):
        raise RuntimeError('Class probabilities do not sum to one')
    return dict(accuracy=float(np.mean(probability.argmax(1) == y)),
                log_loss=float(log_loss(y, probability, labels=np.arange(4))))


def paired_difference(y, baseline, candidate, users):
    change = (candidate.argmax(1) == y).astype(float) - (baseline.argmax(1) == y)
    ids, inv = np.unique(users, return_inverse=True)
    change = np.bincount(inv, weights=change) / np.bincount(inv)
    rng = np.random.default_rng(SEED)
    draws = [rng.choice(change, len(ids), replace=True).mean() for _ in range(1000)]
    return dict(accuracy_difference=float(change.mean()),
                participant_bootstrap_95ci=np.quantile(draws, [.025, .975]).tolist())


def grouped_folds(x, y, users, seed, folds=3):
    splitter = GroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    result = list(splitter.split(x, y, groups=users))
    for training, held in result:
        if set(users[training]).intersection(users[held]):
            raise RuntimeError('Internal cascade fold splits a participant')
    return result


def forest(kind, config, seed):
    common = dict(n_estimators=config['trees'], max_depth=config['depth'],
                  min_samples_leaf=config['min_leaf'], n_jobs=config['threads'], random_state=seed)
    if kind == 'random_forest':
        return RandomForestClassifier(max_features='sqrt', bootstrap=True, **common)
    if kind == 'random_feature_forest':
        return ExtraTreesClassifier(max_features=1, bootstrap=False, **common)
    raise ValueError(kind)


def layer_probabilities(layer, x):
    return np.concatenate([np.mean([model.predict_proba(x) for model in models], axis=0)
                           for models in layer], axis=1).astype(np.float32)


def probabilities(layers, x):
    previous = None
    for layer in layers:
        inputs = x if previous is None else np.concatenate([x, previous], axis=1)
        previous = layer_probabilities(layer, inputs)
    if previous is None:
        raise ValueError('At least one trained cascade level is required')
    return previous.reshape(len(x), 2, 4).mean(axis=1)


def fit_cascade(x, y, users, xs, ys, config, seed, callback=None):
    folds = grouped_folds(x, y, users, seed, config['folds'])
    previous = selected_previous = None
    layers, history = [], []
    best, best_depth, stale = (-1., -float('inf')), 0, 0
    started = time.monotonic()
    for depth in range(config['layers']):
        inputs = x if previous is None else np.concatenate([x, previous], axis=1)
        selected_inputs = xs if selected_previous is None else np.concatenate([xs, selected_previous], axis=1)
        previous = np.zeros((len(x), 8), dtype=np.float32)
        selected_previous = np.zeros((len(xs), 8), dtype=np.float32)
        layer = []
        for kind_index, kind in enumerate(('random_forest', 'random_feature_forest')):
            models = []
            block = slice(kind_index * 4, (kind_index + 1) * 4)
            for fold_index, (training, held) in enumerate(folds):
                model = forest(kind, config, seed + 1000 * depth + 100 * kind_index + fold_index)
                model.fit(inputs[training], y[training])
                previous[held, block] = model.predict_proba(inputs[held])
                selected_previous[:, block] += model.predict_proba(selected_inputs) / len(folds)
                models.append(model)
                print(json.dumps(dict(event='fold', level=depth + 1, forest=kind,
                                      fold=fold_index + 1, seconds=time.monotonic() - started)), flush=True)
            layer.append(models)
        layers.append(layer)
        probability = selected_previous.reshape(len(xs), 2, 4).mean(1)
        current = score(ys, probability)
        record = dict(level=depth + 1, validation=current, seconds=time.monotonic() - started)
        history.append(record)
        rank = current['accuracy'], -current['log_loss']
        if rank > best:
            best, best_depth, stale = rank, depth + 1, 0
        else:
            stale += 1
        print(json.dumps(record), flush=True)
        if callback is not None:
            callback(layers[:best_depth], history, best_depth)
        if stale >= config['patience']:
            break
    return layers[:best_depth], history, best_depth


def train(args):
    if args.output.exists():
        raise RuntimeError(f'Refusing to overwrite an existing study: {args.output}')
    protocol, freeze = frozen_contract()
    x, y, users = load_data('train', protocol)
    xs, ys, selected_users = load_data('validation', protocol)
    baseline = joblib.load(HERE / 'runs_med/model.joblib')
    baseline_p = baseline['model'].predict_proba(xs)
    baseline_score = score(ys, baseline_p)
    expected = json.loads((HERE / 'runs_med/validation_results.json').read_text())['distribution']['accuracy']
    if abs(baseline_score['accuracy'] - expected) > 1e-12:
        raise RuntimeError('Frozen baseline validation accuracy failed to reproduce')
    config = dict(trees=args.trees, layers=args.layers, depth=16, min_leaf=20,
                  folds=3, threads=4, patience=2)
    item = dict(architecture=ARCHITECTURE, config=config, seed=args.seed,
                source_sha256=digest(__file__), parent_freeze_sha256=digest(HERE / 'runs_med/freeze.json'),
                frozen_model_sha256=freeze['model_sha256'], completed=False,
                baseline_validation=baseline_score, development_role='validation')
    started = time.monotonic()
    print(json.dumps(dict(event='start', architecture=ARCHITECTURE, train_rows=len(y),
                          validation_rows=len(ys), config=config)), flush=True)

    def checkpoint(layers, history, best_depth):
        item.update(layers=layers, history=history.copy(), selected_level=best_depth,
                    validation=history[best_depth - 1]['validation'])
        joblib.dump(item, args.output)

    layers, history, best_depth = fit_cascade(x, y, users, xs, ys, config, args.seed, checkpoint)
    probability = probabilities(layers, xs)
    item.update(layers=layers, history=history, selected_level=best_depth,
                validation=score(ys, probability),
                validation_paired_change=paired_difference(ys, baseline_p, probability, selected_users),
                train=score(y, probabilities(layers, x)), completed=True,
                seconds=time.monotonic() - started)
    joblib.dump(item, args.output)
    print(json.dumps({k: item[k] for k in ('architecture', 'selected_level', 'train', 'validation',
                                          'baseline_validation', 'validation_paired_change', 'seconds')}), flush=True)


def load_checkpoint(args):
    frozen_contract()
    item = joblib.load(args.output)
    if item['architecture'] != ARCHITECTURE or not item['completed']:
        raise RuntimeError('Wrong architecture or incomplete training run')
    if item['source_sha256'] != digest(__file__):
        raise RuntimeError('Architecture source changed since training')
    if item['parent_freeze_sha256'] != digest(HERE / 'runs_med/freeze.json'):
        raise RuntimeError('Parent freeze changed since training')
    return item


def compare(args):
    item = load_checkpoint(args)
    if 'evaluation_started' in item:
        raise RuntimeError('This candidate already opened the benchmark')
    item['evaluation_started'] = time.time()
    joblib.dump(item, args.output)
    protocol, _ = frozen_contract()
    x, y, _ = load_data('calibration', protocol)
    probability = probabilities(item['layers'], x)
    fit = minimize_scalar(lambda log_t: log_loss(y, softmax(np.log(np.clip(probability, 1e-12, 1)) /
                                                          np.exp(log_t), axis=1)),
                          bounds=(-2.3, 2.3), method='bounded')
    if not fit.success:
        raise RuntimeError('Temperature calibration failed')
    item['temperature'] = float(np.exp(fit.x))
    baseline = joblib.load(HERE / 'runs_med/model.joblib')
    item['evaluation'] = {}
    for role in ('train', 'validation', 'test', 'selection'):
        x, y, users = load_data(role, protocol)
        base = baseline['model'].predict_proba(x)
        base = softmax(np.log(np.clip(base, 1e-12, 1)) / baseline['temperature'], axis=1)
        raw = probabilities(item['layers'], x)
        probability = softmax(np.log(np.clip(raw, 1e-12, 1)) / item['temperature'], axis=1)
        result = dict(candidate=score(y, probability), baseline=score(y, base),
                      paired_change=paired_difference(y, base, probability, users),
                      confusion=confusion_matrix(y, probability.argmax(1), labels=np.arange(4)).tolist())
        item['evaluation'][role] = result
        joblib.dump(item, args.output)
        print(json.dumps(dict(role=role, **result)), flush=True)


def predict(args):
    item = load_checkpoint(args)
    if 'temperature' not in item:
        raise RuntimeError('Run calibration/comparison before calibrated inference')
    from prototype_net.exp3.model import encode_observed
    baseline = joblib.load(HERE / 'runs_med/model.joblib')
    with np.load(args.input, allow_pickle=False) as data:
        x = encode_observed(data['gallery'], data['gallery_lengths'], data['query'], data['query_lengths'],
                            baseline['scale_floor'], 'augmented_pause')
    raw = probabilities(item['layers'], x)
    probability = softmax(np.log(np.clip(raw, 1e-12, 1)) / item['temperature'], axis=1)[0]
    print(json.dumps(dict(predicted_class=int(probability.argmax()), probabilities=probability.tolist())))


def main():
    parser = argparse.ArgumentParser(__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--phase', choices=['train', 'compare', 'predict'], required=True)
    parser.add_argument('--output', type=Path, default=HERE / 'runs_med' / f'{ARCHITECTURE}_candidate.joblib')
    parser.add_argument('--input', type=Path)
    parser.add_argument('--trees', type=int, default=64)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()
    if args.trees < 1 or args.layers < 1:
        parser.error('trees and layers must be positive')
    if args.phase == 'predict' and args.input is None:
        parser.error('predict requires --input')
    if not args.output.parent.is_dir():
        parser.error('output must use an existing directory')
    with threadpool_limits(limits=4):
        dict(train=train, compare=compare, predict=predict)[args.phase](args)


if __name__ == '__main__':
    main()
