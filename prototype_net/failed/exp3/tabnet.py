"""Self-contained TabNet experiment on frozen exp3 augmented_pause features.

Arik and Pfister, https://arxiv.org/abs/1908.07442. Sequential sparsemax masks,
shared and independent GLU transforms, ghost batch normalization, and sparse
attention regularization. Training-only quantile preprocessing adapts the model
to these skewed engineered features. All TabNet implementation code lives here.

Validation selects checkpoints/depth; test is evaluated after that choice.
The historical selection split is reported as a diagnostic, never as a gate.

python -m prototype_net.exp3.tabnet --phase train
python -m prototype_net.exp3.tabnet --phase compare
python -m prototype_net.exp3.tabnet --phase predict --input observed.npz
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.metrics import confusion_matrix, log_loss
from sklearn.preprocessing import QuantileTransformer
from threadpoolctl import threadpool_limits
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARCHITECTURE = 'tabnet'
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


# Architecture-specific layers. No architecture helper modules are imported.
class Sparsemax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        value = value - value.max(dim=-1, keepdim=True).values
        ordered = value.sort(dim=-1, descending=True).values
        ranks = torch.arange(1, value.shape[-1] + 1, dtype=value.dtype, device=value.device)
        partial = ordered.cumsum(-1) - 1
        support = (ranks * ordered > partial).sum(-1, keepdim=True).clamp_min(1)
        threshold = partial.gather(-1, support - 1) / support
        probability = (value - threshold).clamp_min(0)
        ctx.save_for_backward(probability)
        return probability

    @staticmethod
    def backward(ctx, gradient):
        probability, = ctx.saved_tensors
        support = probability > 0
        result = gradient * support
        return result - support * result.sum(-1, keepdim=True) / support.sum(-1, keepdim=True).clamp_min(1)


def sparsemax(value):
    return Sparsemax.apply(value)


class GhostBatchNorm(nn.Module):
    def __init__(self, features, virtual_batch=128):
        super().__init__()
        self.virtual_batch = virtual_batch
        self.norm = nn.BatchNorm1d(features, momentum=.02)

    def forward(self, x):
        if not self.training:
            return self.norm(x)
        chunks = x.chunk(max(1, int(np.ceil(len(x) / self.virtual_batch))), dim=0)
        return torch.cat([self.norm(chunk) for chunk in chunks], dim=0)


class GatedLayer(nn.Module):
    def __init__(self, linear, width):
        super().__init__()
        self.linear = linear
        self.norm = GhostBatchNorm(2 * width)

    def forward(self, x):
        left, right = self.norm(self.linear(x)).chunk(2, dim=-1)
        return left * torch.sigmoid(right)


class FeatureTransformer(nn.Module):
    def __init__(self, shared, width):
        super().__init__()
        self.layers = nn.ModuleList([GatedLayer(layer, width) for layer in shared] +
                                   [GatedLayer(nn.Linear(width, 2 * width, bias=False), width) for _ in range(2)])

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            next_value = layer(x)
            x = next_value if i == 0 else (x + next_value) * (2 ** -.5)
        return x


class AttentiveTransformer(nn.Module):
    def __init__(self, attention, features):
        super().__init__()
        self.linear = nn.Linear(attention, features, bias=False)
        self.norm = GhostBatchNorm(features)

    def forward(self, attention, prior):
        return sparsemax(prior * self.norm(self.linear(attention)))


class TabNet(nn.Module):
    def __init__(self, features=1485, decision=32, attention=32, steps=3, gamma=1.5):
        super().__init__()
        self.decision, self.gamma = decision, gamma
        width = decision + attention
        self.input_norm = nn.BatchNorm1d(features, momentum=.01)
        shared = nn.ModuleList([nn.Linear(features, 2 * width, bias=False),
                                nn.Linear(width, 2 * width, bias=False)])
        self.initial = FeatureTransformer(shared, width)
        self.transforms = nn.ModuleList([FeatureTransformer(shared, width) for _ in range(steps)])
        self.attentive = nn.ModuleList([AttentiveTransformer(attention, features) for _ in range(steps)])
        self.head = nn.Linear(decision, 4, bias=False)

    @torch.no_grad()
    def initialize(self, x):
        # Initialize only the input BN statistics from training examples.
        self.input_norm.running_mean.copy_(x.mean(0))
        self.input_norm.running_var.copy_(x.var(0, unbiased=False).clamp_min(1e-5))

    def forward(self, x):
        x = self.input_norm(x)
        prior = torch.ones_like(x)
        attention = self.initial(x)[:, self.decision:]
        decision = x.new_zeros((len(x), self.decision))
        penalty = x.new_zeros(())
        for attentive, transform in zip(self.attentive, self.transforms):
            mask = attentive(attention, prior)
            penalty = penalty - (mask * mask.clamp_min(1e-15).log()).sum(1).mean()
            prior = prior * (self.gamma - mask)
            value = transform(mask * x)
            decision = decision + F.relu(value[:, :self.decision])
            attention = value[:, self.decision:]
        return self.head(decision), penalty / len(self.transforms)


def model_config():
    return dict(features=1485, decision=32, attention=32, steps=3, gamma=1.5)


def make_model(config):
    return TabNet(**config)


def optimizer_config():
    return dict(lr=.01, weight_decay=0.)


def regularization_strength():
    return .001


def probabilities(model, x, device, batch_size=4096):
    model.eval()
    result = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            logits, _ = model(torch.from_numpy(x[start:start + batch_size]).to(device))
            result.append(logits.softmax(-1).cpu().numpy())
    return np.concatenate(result)


def choose_device(requested):
    if requested == 'auto':
        return 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    if requested == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS is unavailable')
    return requested


def restore(item, device):
    model = make_model(item['config'])
    model.load_state_dict(item['state'])
    return model.to(device)


def train(args):
    if args.output.exists():
        raise RuntimeError(f'Refusing to overwrite an existing study: {args.output}')
    protocol, freeze = frozen_contract()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    x, y, _ = load_data('train', protocol)
    xs, ys, users = load_data('validation', protocol)
    baseline = joblib.load(HERE / 'runs_med/model.joblib')
    baseline_p = baseline['model'].predict_proba(xs)
    baseline_score = score(ys, baseline_p)
    expected = json.loads((HERE / 'runs_med/validation_results.json').read_text())['distribution']['accuracy']
    if abs(baseline_score['accuracy'] - expected) > 1e-12:
        raise RuntimeError('Frozen baseline validation accuracy failed to reproduce')
    normalizer = QuantileTransformer(n_quantiles=256, output_distribution='normal',
                                    subsample=20000, random_state=args.seed, copy=False)
    x = normalizer.fit_transform(x)
    xs = normalizer.transform(xs)
    device = choose_device(args.device)
    config = model_config()
    model = make_model(config).to(device)
    initial = x[rng.choice(len(x), min(4096, len(x)), replace=False)]
    model.initialize(torch.from_numpy(initial).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_config())
    item = dict(architecture=ARCHITECTURE, config=config, normalizer=normalizer,
                source_sha256=digest(__file__), parent_freeze_sha256=digest(HERE / 'runs_med/freeze.json'),
                frozen_model_sha256=freeze['model_sha256'], seed=args.seed, batch_size=args.batch_size,
                epochs_limit=args.epochs, patience=args.patience, optimizer=optimizer_config(),
                device=device, completed=False, baseline_validation=baseline_score, development_role='validation', history=[])
    best, stale = (-1., -float('inf')), 0
    started = time.monotonic()
    print(json.dumps(dict(event='start', architecture=ARCHITECTURE, device=device,
                          train_rows=len(y), validation_rows=len(ys), config=config)), flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.
        for indices in np.array_split(rng.permutation(len(x)), int(np.ceil(len(x) / args.batch_size))):
            xb = torch.from_numpy(x[indices]).to(device)
            yb = torch.from_numpy(y[indices]).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, penalty = model(xb)
            loss = F.cross_entropy(logits, yb) + regularization_strength() * penalty
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
        probability = probabilities(model, xs, device)
        current = score(ys, probability)
        record = dict(epoch=epoch, training_loss=total_loss / len(y), validation=current,
                      seconds=time.monotonic() - started)
        if epoch % 5 == 0:
            record['train'] = score(y, probabilities(model, x, device))
        item['history'].append(record)
        print(json.dumps(record), flush=True)
        rank = current['accuracy'], -current['log_loss']
        if rank > best:
            best, stale = rank, 0
            item.update(state=copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()}),
                        selected_epoch=epoch, validation=current,
                        validation_paired_change=paired_difference(ys, baseline_p, probability, users))
        else:
            stale += 1
        joblib.dump(item, args.output)
        if stale >= args.patience:
            break
    model = restore(item, device)
    item['train'] = score(y, probabilities(model, x, device))
    item['completed'] = True
    item['seconds'] = time.monotonic() - started
    joblib.dump(item, args.output)
    print(json.dumps({k: item[k] for k in ('architecture', 'selected_epoch', 'train', 'validation',
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
    device = choose_device(args.device)
    model = restore(item, device)
    x, y, _ = load_data('calibration', protocol)
    probability = probabilities(model, item['normalizer'].transform(x), device)
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
        raw = probabilities(model, item['normalizer'].transform(x), device)
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
    # Reuse the existing frozen feature encoder; all architecture code is local.
    from prototype_net.exp3.model import encode_observed
    baseline = joblib.load(HERE / 'runs_med/model.joblib')
    with np.load(args.input, allow_pickle=False) as data:
        x = encode_observed(data['gallery'], data['gallery_lengths'], data['query'], data['query_lengths'],
                            baseline['scale_floor'], 'augmented_pause')
    device = choose_device(args.device)
    raw = probabilities(restore(item, device), item['normalizer'].transform(x), device)
    probability = softmax(np.log(np.clip(raw, 1e-12, 1)) / item['temperature'], axis=1)[0]
    print(json.dumps(dict(predicted_class=int(probability.argmax()), probabilities=probability.tolist())))


def main():
    parser = argparse.ArgumentParser(__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--phase', choices=['train', 'compare', 'predict'], required=True)
    parser.add_argument('--output', type=Path, default=HERE / 'runs_med' / f'{ARCHITECTURE}_candidate.joblib')
    parser.add_argument('--input', type=Path)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--patience', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 2:
        parser.error('epochs/patience must be positive and batch-size must be at least two')
    if args.phase == 'predict' and args.input is None:
        parser.error('predict requires --input')
    if not args.output.parent.is_dir():
        parser.error('output must use an existing directory')
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):
        dict(train=train, compare=compare, predict=predict)[args.phase](args)


if __name__ == '__main__':
    main()
