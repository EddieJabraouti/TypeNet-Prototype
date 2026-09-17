"""Gallery-relative binary deviation study, with the readable frozen model archived below.

Legacy commands execute only the hash-verified original source in an isolated
namespace. Binary commands use --binary and the implementation below the archive.
"""
import hashlib as _hashlib
from pathlib import Path as _Path
import json as _json

_LEGACY_SOURCE = r'''"""Best exp3 model: distribution-preserving aggregation, plus raw-series tests.

Ten clean baseline windows versus ten query windows. The parent freeze is the
1,443-d augmented tree. This run scores MED-deconvolved press-to-press pause
descriptors versus each person's clean baseline and concatenates them onto the
parent. No generator pause mask. No maximum-entropy density.

Hyperparameters match the regularized ten-window reference: 15 leaves, depth 4,
min_samples_leaf 200, L2 50, 600 iterations, lr 0.04.

Data imports only: Aalto loaders (`paper_typenet.nn`) and the synthetic
timing generator (`prototype_net.perturbation_generator.perturb`).

    python -m prototype_net.exp3.model --phase develop
    python -m prototype_net.exp3.model --phase compare
    python -m prototype_net.exp3.model observed.npz
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, log_loss, roc_auc_score, roc_curve
from threadpoolctl import threadpool_limits

from paper_typenet.nn import events_to_features
from prototype_net.perturbation_generator.perturb import (
    CLASS_NAMES, PerturbationConfig, StructuredTimingPerturber,
)

csv.field_size_limit(10_000_000)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / 'data/Keystrokes/files'
MANIFEST = ROOT / 'prototype_net/weights/protocol/synthetic_impairment_protocol_v4_64x512_unseen_triplet_manifest.json'
PARENT = HERE / 'runs'
PAUSE = HERE / 'runs_pause'
HOLD = HERE / 'runs_hold'
OUT = HERE / 'runs_med'
SEED = 9172026
METRIC_SEED = 9162026
WINDOWS = 10
SERIES_DIM = 42
PAUSE_DIM = SERIES_DIM
HOLD_DIM = SERIES_DIM
MED_DIM = SERIES_DIM
MED_LEN = 5
MED_ITERS = 40
WORD_KEYS = (13, 32)
CORR_KEYS = (8, 46)
QUANTILES = [.05, .1, .25, .5, .75, .9, .95]
EDGES = [-np.inf, -3, -2, -1, -.5, 0, .5, 1, 2, 3, np.inf]
MODEL_CONFIG = dict(
    max_leaf_nodes=15, max_depth=4, min_samples_leaf=200, l2_regularization=50.,
    max_features=1., learning_rate=.04, max_iter=600, early_stopping=False,
    random_state=SEED,
)
MIN_COUNTS = dict(train=12000, selection=800, calibration=600, validation=800, test=2000)
WORKERS = 8


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def calibrated(p, temperature):
    return softmax(np.log(np.clip(p, 1e-12, 1)) / temperature, axis=1)


def metrics(y, p, users):
    pred, conf = p.argmax(1), p.max(1)
    correct = pred == y
    bins, ece = [], 0.
    for lo, hi in zip(np.linspace(0, 1, 16)[:-1], np.linspace(0, 1, 16)[1:]):
        mask = (conf >= lo) & ((conf < hi) if hi < 1 else (conf <= hi))
        if mask.any():
            acc, confidence = float(correct[mask].mean()), float(conf[mask].mean())
            ece += mask.mean() * abs(acc - confidence)
            bins.append(dict(lower=float(lo), upper=float(hi), count=int(mask.sum()),
                             accuracy=acc, confidence=confidence))
    unique, inv = np.unique(users, return_inverse=True)
    by_user = np.bincount(inv, weights=correct) / np.bincount(inv)
    rng = np.random.default_rng(METRIC_SEED)
    boots = [np.mean(rng.choice(by_user, len(by_user), replace=True)) for _ in range(1000)]
    fpr, tpr, _ = roc_curve(y != 0, 1 - p[:, 0])
    ix = np.argmin(np.abs(fpr - (1 - tpr)))
    matrix = confusion_matrix(y, pred, labels=np.arange(4))
    return dict(
        accuracy=float(correct.mean()),
        accuracy_cluster_bootstrap_95ci=np.quantile(boots, [.025, .975]).tolist(),
        log_loss=float(log_loss(y, p, labels=np.arange(4))),
        multiclass_brier=float(np.mean(np.sum((p - np.eye(4)[y]) ** 2, axis=1))),
        ece_15_equal_width=float(ece), mean_confidence=float(conf.mean()),
        pooled_detection_eer=float((fpr[ix] + 1 - tpr[ix]) / 2),
        per_class_recall=dict(zip(CLASS_NAMES, (matrix.diagonal() / matrix.sum(1)).tolist())),
        confusion_counts=matrix.tolist(), reliability=bins, samples=len(y),
        participants=len(unique),
    )


def paired_difference(y, baseline, candidate, users):
    delta = (candidate.argmax(1) == y).astype(float) - (baseline.argmax(1) == y)
    unique, idx = np.unique(users, return_inverse=True)
    per_user = np.bincount(idx, weights=delta) / np.bincount(idx)
    rng = np.random.default_rng(SEED)
    bootstrap = [rng.choice(per_user, len(per_user), replace=True).mean() for _ in range(1000)]
    return dict(accuracy_difference=float(delta.mean()),
                participant_bootstrap_95ci=np.quantile(bootstrap, [.025, .975]).tolist())


def channels(x, length):
    n = int(length)
    h = np.maximum(np.rint(x[:n, 0].astype(float) * 1000) / 1000, .001)
    p = np.maximum(np.rint(x[:n - 1, 2].astype(float) * 1000) / 1000, .001)
    k = np.rint(x[:n, 4] * 255).astype(int)
    return np.log(h), np.log(p), k


def summary(v):
    v = np.asarray(v, dtype=float)
    if len(v) == 0:
        return np.zeros(19)
    quant = np.quantile(v, QUANTILES)
    center = v - np.mean(v)
    variance = np.mean(center ** 2)
    extras = [np.mean(v), np.std(v), np.mean(np.abs(v - quant[3])),
              quant[6] - quant[0], np.mean(np.clip(center, -3, 3) ** 3)]
    for lag in (1, 2, 3, 5):
        extras.append(np.mean(center[:-lag] * center[lag:]) if len(v) > lag else 0.)
    extras += [np.mean(np.abs(np.diff(v))) if len(v) > 1 else 0.,
               np.mean(v > quant[3] + 1), variance]
    return np.r_[quant, extras]


def enrollment(galleries, lengths):
    cs = [channels(g, n) for g, n in zip(galleries, lengths)]
    all_h = np.concatenate([c[0] for c in cs])
    all_p = np.concatenate([c[1] for c in cs])
    all_k = np.concatenate([c[2] for c in cs])
    global_mean = np.mean(all_h)
    centers = {int(k): (all_h[all_k == k].sum() + 5 * global_mean) /
               ((all_k == k).sum() + 5) for k in np.unique(all_k)}
    gallery_rows = []
    for bh, bp, bk in cs:
        br = bh - np.array([centers.get(int(k), global_mean) for k in bk])
        gallery_rows.append(np.r_[summary(bh), summary(bp), summary(br)])
    return np.array(gallery_rows), all_h, all_p, centers, global_mean


def session_vector(x, length, reference):
    gallery_rows, gh, gp, centers, global_mean = reference
    h, p, keys = channels(x, length)
    residual = h - np.array([centers.get(int(k), global_mean) for k in keys])
    raw = np.r_[summary(h), summary(p), summary(residual)]
    delta = raw[None, :] - gallery_rows
    features = [raw, np.median(delta, axis=0), np.std(gallery_rows, axis=0),
                np.quantile(delta, .1, axis=0), np.quantile(delta, .9, axis=0)]
    for v, base in ((h, gh), (p, gp)):
        q = np.quantile(base, QUANTILES)
        features.append(np.mean(v[:, None] > q, axis=0))
        features.append(np.array([np.mean(v > q[3] + np.log(t)) for t in (2, 3, 5)]))
    for lag in (1, 2, 3, 5):
        features.append(np.array([np.std(residual[lag:] - residual[:-lag])
                                  if len(residual) > lag else 0.]))
    return np.nan_to_num(np.concatenate(features), nan=0, posinf=10, neginf=-10).astype('float32')


def bundle_vector(rows):
    return np.r_[np.mean(rows, axis=0), np.std(rows, axis=0)].astype('float32')


def series_intervals(x, length, kind):
    n = int(length)
    if kind == 'hold':
        values = np.maximum(np.rint(x[:n, 0].astype(float) * 1000) / 1000, .001)
        keys = np.rint(x[:n, 4] * 255).astype(int)
    elif kind == 'press':
        values = np.maximum(np.rint(x[:n - 1, 2].astype(float) * 1000) / 1000, .001)
        keys = np.rint(x[:n - 1, 4] * 255).astype(int)
    else:
        raise ValueError(f'Unknown series: {kind}')
    return values, keys


def collect_series(windows, lengths, kind):
    values, keys = [], []
    for x, n in zip(windows, lengths):
        v, k = series_intervals(x, n, kind)
        values.append(v)
        keys.append(k)
    return np.concatenate(values), np.concatenate(keys)


def gate_values(values, keys, wanted):
    return values if wanted is None else values[np.isin(keys, wanted)]


def rate_excess(values, t95, t99):
    if len(values) == 0:
        return np.zeros(5, dtype=float)
    return np.array([
        np.mean(values > t95),
        np.mean(values > t99),
        np.mean(np.maximum(values - t95, 0.)),
        np.mean(np.maximum(values - t99, 0.)),
        np.max(values),
    ], dtype=float)


def window_series_stats(windows, lengths, t95, wanted, kind):
    rates, excess = [], []
    for x, n in zip(windows, lengths):
        values, keys = series_intervals(x, n, kind)
        values = gate_values(values, keys, wanted)
        if len(values) == 0:
            rates.append(0.)
            excess.append(0.)
        else:
            rates.append(float(np.mean(values > t95)))
            excess.append(float(np.mean(np.maximum(values - t95, 0.))))
    r, e = np.asarray(rates), np.asarray(excess)
    return [float(r.mean()), float(r.std()), float(r.max()),
            float(e.mean()), float(e.std()), float(e.max())]


def series_vector(gallery, gallery_lengths, query, query_lengths, kind):
    """Observed timings vs this person's clean baseline. No generator mask."""
    g_v, g_k = collect_series(gallery, gallery_lengths, kind)
    q_v, q_k = collect_series(query, query_lengths, kind)
    t50 = float(np.quantile(g_v, .50))
    t95 = float(np.quantile(g_v, .95))
    t99 = float(np.quantile(g_v, .99))
    parts = []
    for wanted in (None, WORD_KEYS, CORR_KEYS):
        baseline = rate_excess(gate_values(g_v, g_k, wanted), t95, t99)
        observed = rate_excess(gate_values(q_v, q_k, wanted), t95, t99)
        parts.extend(observed.tolist())
        parts.extend((observed[:4] - baseline[:4]).tolist())
    parts.extend(window_series_stats(query, query_lengths, t95, None, kind))
    parts.extend(window_series_stats(query, query_lengths, t95, WORD_KEYS, kind))
    denom = np.maximum([t50, t95, t99], 1e-3)
    parts.extend((np.quantile(q_v, [.5, .95, .99]) / denom).tolist())
    result = np.asarray(parts, dtype='float32')
    if result.shape != (SERIES_DIM,) or not np.isfinite(result).all():
        raise ValueError(f'{kind} vector must be finite and {SERIES_DIM}-d')
    return result


def pause_vector(gallery, gallery_lengths, query, query_lengths):
    return series_vector(gallery, gallery_lengths, query, query_lengths, 'press')


def hold_vector(gallery, gallery_lengths, query, query_lengths):
    return series_vector(gallery, gallery_lengths, query, query_lengths, 'hold')


def sample_kurtosis(y):
    y = np.asarray(y, dtype=float)
    if len(y) < 4:
        return 0.
    y = y - np.mean(y)
    second = np.mean(y * y)
    if second < 1e-18:
        return 0.
    return float(np.mean(y ** 4) / (second ** 2))


def fir(x, f):
    x = np.asarray(x, dtype=float)
    length = len(f)
    if len(x) < length:
        return np.empty(0, dtype=float)
    return np.lib.stride_tricks.sliding_window_view(x, length) @ f


def kurtosis_med(x, length=MED_LEN, iters=MED_ITERS):
    """Enrollment-only Wiggins MED. Falls back to a delta if kurtosis does not rise."""
    x = np.asarray(x, dtype=float)
    identity = np.zeros(length, dtype=float)
    identity[0] = 1.
    if not np.isfinite(x).all() or len(x) < length + 2:
        return identity
    x = x - np.median(x)
    mad = float(np.median(np.abs(x)))
    if mad < 1e-8:
        return identity
    x = x / mad
    design = np.lib.stride_tricks.sliding_window_view(x, length)
    f = identity.copy()
    for _ in range(iters):
        y = design @ f
        grad = design.T @ (y ** 3)
        nrm = float(np.linalg.norm(grad))
        if nrm < 1e-12:
            break
        updated = grad / nrm
        if np.dot(f, updated) < 0:
            updated = -updated
        if np.linalg.norm(updated - f) < 1e-8:
            f = updated
            break
        f = updated
    if sample_kurtosis(design @ f) < sample_kurtosis(design @ identity):
        return identity
    return f


def apply_med(values, keys, f):
    y = fir(values, f)
    if len(y) == 0:
        return np.abs(values - np.median(values)), np.asarray(keys)
    lag = int(np.argmax(np.abs(f)))
    return np.abs(y), np.asarray(keys)[lag:lag + len(y)]


def contrast_vector(g_v, g_k, q_v, q_k, query_parts):
    t50 = float(np.quantile(g_v, .50))
    t95 = float(np.quantile(g_v, .95))
    t99 = float(np.quantile(g_v, .99))
    parts = []
    for wanted in (None, WORD_KEYS, CORR_KEYS):
        baseline = rate_excess(gate_values(g_v, g_k, wanted), t95, t99)
        observed = rate_excess(gate_values(q_v, q_k, wanted), t95, t99)
        parts.extend(observed.tolist())
        parts.extend((observed[:4] - baseline[:4]).tolist())
    for wanted in (None, WORD_KEYS):
        rates, excess = [], []
        for values, keys in query_parts:
            gated = gate_values(values, keys, wanted)
            if len(gated) == 0:
                rates.append(0.)
                excess.append(0.)
            else:
                rates.append(float(np.mean(gated > t95)))
                excess.append(float(np.mean(np.maximum(gated - t95, 0.))))
        r, e = np.asarray(rates), np.asarray(excess)
        parts.extend([float(r.mean()), float(r.std()), float(r.max()),
                      float(e.mean()), float(e.std()), float(e.max())])
    denom = np.maximum([t50, t95, t99], 1e-3)
    parts.extend((np.quantile(q_v, [.5, .95, .99]) / denom).tolist())
    result = np.asarray(parts, dtype='float32')
    if result.shape != (SERIES_DIM,) or not np.isfinite(result).all():
        raise ValueError(f'Contrast vector must be finite and {SERIES_DIM}-d')
    return result


def med_vector(gallery, gallery_lengths, query, query_lengths):
    """MED on raw press-to-press vs this person's clean baseline. Filter from gallery only."""
    g_parts = [series_intervals(x, n, 'press') for x, n in zip(gallery, gallery_lengths)]
    f = kurtosis_med(np.concatenate([v for v, _ in g_parts]))
    g_dec = [apply_med(v, k, f) for v, k in g_parts]
    q_dec = [apply_med(*series_intervals(x, n, 'press'), f) for x, n in zip(query, query_lengths)]
    g_v, g_k = np.concatenate([v for v, _ in g_dec]), np.concatenate([k for _, k in g_dec])
    q_v, q_k = np.concatenate([v for v, _ in q_dec]), np.concatenate([k for _, k in q_dec])
    return contrast_vector(g_v, g_k, q_v, q_k, q_dec)


def scale_floor(training_gallery):
    return np.maximum(.1 * np.std(training_gallery.reshape(-1, 57).astype(float), axis=0), 1e-4)


def hist(values):
    counts = np.stack([np.sum((values >= lo) & (values < hi), axis=-2)
                       for lo, hi in zip(EDGES[:-1], EDGES[1:])], axis=-1).astype(float) + .5
    return counts / counts.sum(-1, keepdims=True)


def entropy(p):
    return -np.sum(p * np.log(p), axis=-1)


def comparison_distribution(gallery, queries, floor):
    """gallery (10,57), queries (examples,10,57) -> (examples,825)."""
    g = np.asarray(gallery, dtype=float)
    q = np.asarray(queries, dtype=float)
    if g.shape != (10, 57) or q.ndim != 3 or q.shape[1:] != (10, 57):
        raise ValueError('Ten individual window summaries per side required')
    if not np.isfinite(g).all() or not np.isfinite(q).all() or np.shape(floor) != (57,) or np.any(floor <= 0):
        raise ValueError('Finite summaries and positive training-only scale floors required')
    scale = np.maximum(np.quantile(g, .75, axis=0) - np.quantile(g, .25, axis=0), floor)
    within = ((g[:, None] - g[None, :]) / scale)[~np.eye(10, dtype=bool)]
    pairs = (q[:, :, None] - g[None, None, :]) / scale
    flat = pairs.reshape(len(q), 100, 57)
    tail = np.quantile(np.abs(within), .95, axis=0)
    exceed = np.abs(pairs) > tail
    per_query = exceed.mean(axis=2)
    parts = [np.quantile(flat, QUANTILES, axis=1).transpose(1, 0, 2).reshape(len(q), -1),
             (flat > 0).mean(axis=1), exceed.mean(axis=(1, 2)),
             per_query.std(axis=1), np.quantile(per_query, .9, axis=1), per_query.max(axis=1)]
    hp, hg = hist(flat), hist(within)
    mix = (hp + hg) / 2
    js = entropy(mix) - (entropy(hp) + entropy(hg)) / 2
    parts += [entropy(hp) - entropy(hg), np.maximum(js, 0)]
    distances = np.stack([np.abs(q[:, :, None, i:i + 7] - g[None, None, :, i:i + 7]).mean(-1)
                          for i in (0, 19, 38)], axis=-1).reshape(len(q), 100, 3)
    parts += [np.quantile(distances, QUANTILES, axis=1).transpose(1, 0, 2).reshape(len(q), -1),
              distances.mean(axis=1), distances.std(axis=1)]
    result = np.concatenate(parts, axis=1).astype('float32')
    assert result.shape == (len(q), 825) and np.isfinite(result).all()
    return result


def typical_timing_shift(gallery, queries, index):
    g = np.exp(gallery[:, :, index].astype(float)) * 1000
    q = np.exp(queries[:, :, :, index].astype(float)) * 1000
    return np.median((q[:, :, :, None] - g[:, None, None, :]).reshape(len(g), 4, 100), axis=-1)


def ordered_blocks(path):
    sessions = {}
    with path.open(newline='', encoding='utf8', errors='replace') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            try:
                event = tuple(int(row[k]) for k in ('PRESS_TIME', 'RELEASE_TIME', 'KEYCODE', 'KEYSTROKE_ID'))
                sid = row['TEST_SECTION_ID']
            except (ValueError, KeyError, TypeError):
                continue
            sessions.setdefault(sid, set()).add(event)
    sessions = [sorted(events, key=lambda e: (e[0], e[3])) for events in sessions.values() if len(events) >= 6]
    sessions.sort(key=lambda events: (events[0][0], events[0][3]))
    midpoint = len(sessions) // 2
    if midpoint == 0:
        raise ValueError('Insufficient distinct sentences')
    left, right = sessions[:midpoint], sessions[midpoint:]
    if max(e[1] for s in left for e in s) >= min(e[0] for s in right for e in s):
        raise ValueError('Baseline and query periods overlap')

    def windows(group):
        result = []
        for sentence in group:
            for start in range(0, len(sentence), 50):
                events = sentence[start:start + 50]
                if len(events) < 6:
                    continue
                x, n = events_to_features(events, 50)
                if not np.isfinite(x).all():
                    raise ValueError('Nonfinite source timings')
                key = hashlib.sha256(x[:n].tobytes()).hexdigest()
                result.append((x, n, key))
        return result

    return windows(left), windows(right)


def rng_for(role, uid, cls):
    value = hashlib.sha256(f'accumulated:{SEED}:{role}:{uid}:{cls}'.encode()).digest()
    return np.random.default_rng(int.from_bytes(value[:8], 'little'))


def shuffled_role(roles, role, seed):
    return sorted(roles[role]['participant_ids'],
                  key=lambda u: hashlib.sha256(f'{seed}:{role}:{u}'.encode()).hexdigest())


def original_exp3_opened(roles):
    val, test = shuffled_role(roles, 'selection', METRIC_SEED), shuffled_role(roles, 'final_test', METRIC_SEED)
    return set(val[:3400]), set(test[:4000])


def classifier():
    return HistGradientBoostingClassifier(**MODEL_CONFIG)


def temperature(y, raw):
    fit = minimize_scalar(lambda t: log_loss(y, calibrated(raw, np.exp(t))),
                          bounds=(-2.3, 2.3), method='bounded')
    if not fit.success:
        raise RuntimeError('Temperature fit failed')
    return float(np.exp(fit.x))


def order_ids(ids):
    return sorted((str(u) for u in ids), key=lambda u: hashlib.sha256(f'{SEED}:{u}'.encode()).hexdigest())


def held_role(uid):
    value = int.from_bytes(hashlib.sha256(f'{SEED}:heldrole:{uid}'.encode()).digest()[:8], 'little') / 2 ** 64
    if value < .20:
        return 'calibration'
    if value < .45:
        return 'validation'
    return 'test'


def scan_pool(name, pool, used_hashes):
    ids, skipped, hashes, lengths = [], [], {}, {}
    scanned = 0
    for uid in pool:
        try:
            g, q = ordered_blocks(DATA / f'{uid}_keystrokes.txt')
            if min(len(g), len(q)) < WINDOWS:
                raise ValueError('Fewer than ten windows on either side of sentence boundary')
            rows = g[-WINDOWS:] + q[:WINDOWS]
            keys = [v[2] for v in rows]
            if len(set(keys)) != len(keys):
                raise ValueError('Repeated source window within participant')
            if any(k in used_hashes for k in keys):
                raise ValueError('Repeated source window across participants')
        except (OSError, ValueError) as exc:
            skipped.append(dict(id=uid, reason=str(exc)))
            scanned += 1
            if scanned % 500 == 0:
                print('REGISTER_PROGRESS', name, 'kept', len(ids), 'skipped', len(skipped), flush=True)
            continue
        for k in keys:
            used_hashes[k] = uid
        ids.append(uid)
        hashes[uid] = digest(DATA / f'{uid}_keystrokes.txt')
        lengths[uid] = [int(v[1]) for v in rows]
        scanned += 1
        if scanned % 500 == 0:
            print('REGISTER_PROGRESS', name, 'kept', len(ids), 'skipped', len(skipped), flush=True)
    print('REGISTER', name, len(ids), 'skipped', len(skipped), flush=True)
    return dict(ids=ids, skipped=skipped, hashes=hashes, lengths=lengths)


def role_audit(ids, skipped, hashes, lengths):
    values = [n for uid in ids for n in lengths[uid]]
    return dict(retained=len(ids), skipped=skipped, source_sha256=hashes,
                keys_per_window_quantiles=np.quantile(values, [0, .25, .5, .75, 1]).tolist() if values else [])


def parent_source():
    return 'prototype_net/perturbation_generator/perturb.py'


def register():
    if (OUT / 'protocol.json').exists():
        p = json.loads((OUT / 'protocol.json').read_text())
        for file, sha in p['source_hashes'].items():
            if digest(ROOT / file) != sha:
                raise RuntimeError(f'Frozen source changed: {file}')
        return p
    parent_p = json.loads((PARENT / 'protocol.json').read_text())
    parent_f = json.loads((PARENT / 'freeze.json').read_text())
    if digest(PARENT / 'protocol.json') != parent_f['protocol_sha256']:
        raise RuntimeError('Parent protocol changed')
    if digest(PARENT / 'model.joblib') != parent_f['model_sha256']:
        raise RuntimeError('Parent freeze changed')
    if digest(PARENT / 'scale_floor.npy') != parent_f['floor_sha256']:
        raise RuntimeError('Parent scale floor changed')
    for role, need in MIN_COUNTS.items():
        if len(parent_p['splits'][role]) < need:
            raise RuntimeError(f'Only {len(parent_p["splits"][role])} eligible {role} participants; required at least {need}')
    p = dict(parent_p)
    p.update(
        study='med_v1',
        parent_dir='prototype_net/exp3/runs',
        parent_protocol_sha256=parent_f['protocol_sha256'],
        parent_representation=parent_f['representation'],
        parent_search=json.loads((PARENT / 'search.json').read_text()),
        pause_search=json.loads((PAUSE / 'search.json').read_text()),
        representations=['med', 'augmented', 'augmented_med', 'augmented_pause', 'augmented_pause_med'],
        source_hashes={
            str(Path(__file__).relative_to(ROOT)): digest(__file__),
            parent_source(): digest(ROOT / parent_source()),
        },
        selection_rule='Select among med, augmented, augmented+med, augmented+pause, and augmented+pause+med by highest selection accuracy; tie-break log loss. Summary is fitted as a comparator, not as a selection candidate. Same frozen ten-window hyperparameters, 600 iterations.',
        med='42 features: Wiggins kurtosis MED (length-5 FIR, 40 iterations) estimated on this person\'s clean gallery press-to-press, applied to gallery and query, then the same rate/excess/persistence recipe as the crude pause vector on absolute deconvolved spikes. Filter is enrollment-only. No generator pause_mask. No maximum-entropy density.',
        comparison='Same people and synthetic draws as the full-Aalto freeze. MED and pause features are computed from the same rng_for seeds. Validation/test remain unused until compare.',
    )
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PARENT / 'availability_selection.json', OUT / 'availability_selection.json')
    write_json(OUT / 'protocol.json', p)
    return p


def person_source(task):
    role, uid, sha = task
    path = DATA / f'{uid}_keystrokes.txt'
    if digest(path) != sha:
        raise RuntimeError(f'Source changed: {uid}')
    g, q = ordered_blocks(path)
    g, q = g[-WINDOWS:], q[:WINDOWS]
    hashes = [h for _, _, h in g + q]
    if len(set(hashes)) != 20:
        raise RuntimeError('Repeated baseline/query source window')
    return role, uid, g, q


def observed_windows(role, uid, q, cls, perturber):
    rng = rng_for(role, uid, cls)
    profile = perturber.sample_profile(cls - 1, rng) if cls else None
    return [(perturber.perturb(x, n, cls - 1, rng, profile=profile) if cls else x, n)
            for x, n, _ in q]


def make_pause(task):
    role, uid, g, q = person_source(task)
    g_x = np.stack([x for x, n, _ in g])
    g_n = np.asarray([n for _, n, _ in g], dtype=int)
    perturber = StructuredTimingPerturber(PerturbationConfig())
    pauses = []
    for cls in range(4):
        observed = observed_windows(role, uid, q, cls, perturber)
        q_x = np.stack([x for x, n in observed])
        q_n = np.asarray([n for _, n in observed], dtype=int)
        pauses.append(pause_vector(g_x, g_n, q_x, q_n))
    return np.stack(pauses)


def make_med(task):
    role, uid, g, q = person_source(task)
    g_x = np.stack([x for x, n, _ in g])
    g_n = np.asarray([n for _, n, _ in g], dtype=int)
    perturber = StructuredTimingPerturber(PerturbationConfig())
    meds = []
    for cls in range(4):
        observed = observed_windows(role, uid, q, cls, perturber)
        q_x = np.stack([x for x, n in observed])
        q_n = np.asarray([n for _, n in observed], dtype=int)
        meds.append(med_vector(g_x, g_n, q_x, q_n))
    return np.stack(meds)


def load_vector_cache(folder, role, name, dim, n_people):
    path, meta = folder / f'{role}_{name}.npz', folder / f'{role}_{name}_cache.json'
    if not path.exists() or not meta.exists():
        raise RuntimeError(f'{name} cache missing: {role}')
    m = json.loads(meta.read_text())
    if digest(path) != m['sha256']:
        raise RuntimeError(f'Modified {name} cache: {role}')
    with np.load(path, allow_pickle=False) as src:
        value = np.array(src[name])
    if value.shape != (n_people, 4, dim) or not np.isfinite(value).all():
        raise RuntimeError(f'{name} cache shape mismatch: {role}')
    return value


def make_person(task):
    role, uid, g, q = person_source(task)
    ref = enrollment(np.stack([x for x, n, _ in g]), [n for _, n, _ in g])
    gallery = np.nan_to_num(ref[0], nan=0, posinf=10, neginf=-10).astype('float32')
    perturber = StructuredTimingPerturber(PerturbationConfig())
    query, bundle, holds = [], [], []
    g_x = np.stack([x for x, n, _ in g])
    g_n = np.asarray([n for _, n, _ in g], dtype=int)
    for cls in range(4):
        observed = observed_windows(role, uid, q, cls, perturber)
        rows = np.stack([session_vector(x, n, ref) for x, n in observed])
        query.append(rows[:, :57])
        bundle.append(bundle_vector(rows))
        q_x = np.stack([x for x, n in observed])
        q_n = np.asarray([n for _, n in observed], dtype=int)
        holds.append(hold_vector(g_x, g_n, q_x, q_n))
    return gallery, np.stack(query), np.stack(bundle), np.stack(holds)


def load_parent_windows(role):
    path, meta = PARENT / f'{role}_windows.npz', PARENT / f'{role}_windows_cache.json'
    if not path.exists() or not meta.exists():
        raise RuntimeError(f'Parent window cache missing: {role}')
    m = json.loads(meta.read_text())
    if digest(path) != m['sha256']:
        raise RuntimeError(f'Modified parent window cache: {role}')
    return dict(np.load(path, allow_pickle=False))


def load_windows(p, role):
    path, meta = OUT / f'{role}_med.npz', OUT / f'{role}_med_cache.json'
    parent = load_parent_windows(role)
    n_people = len(p['splits'][role])
    if len(parent['gallery']) != n_people:
        raise RuntimeError(f'Parent windows do not match split: {role}')
    parent['pause'] = load_vector_cache(PAUSE, role, 'pause', PAUSE_DIM, n_people)
    if meta.exists():
        m = json.loads(meta.read_text())
        if digest(path) != m['sha256']:
            raise RuntimeError(f'Modified med cache: {role}')
        with np.load(path, allow_pickle=False) as src:
            parent['med'] = np.array(src['med'])
        if parent['med'].shape != (n_people, 4, MED_DIM):
            raise RuntimeError(f'Med cache shape mismatch: {role}')
        return parent
    if path.exists():
        raise RuntimeError(f'Incomplete med cache {role}; inspect before recovery')
    audit = json.loads((OUT / 'availability_selection.json').read_text())[role]
    tasks = [(role, u, audit['source_sha256'][u]) for u in p['splits'][role]]
    meds = []
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for i, med in enumerate(pool.map(make_med, tasks, chunksize=8)):
            meds.append(med)
            if (i + 1) % 500 == 0:
                print('MED', role, i + 1, len(tasks), round(time.monotonic() - started), flush=True)
    med = np.stack(meds)
    if med.shape != (len(tasks), 4, MED_DIM) or not np.isfinite(med).all():
        raise RuntimeError(f'Nonfinite med features: {role}')
    np.savez_compressed(path, med=med)
    write_json(meta, dict(sha256=digest(path), seconds=time.monotonic() - started, parent_windows=digest(PARENT / f'{role}_windows.npz')))
    parent['med'] = med
    return parent


def dataset(p, role, floor):
    d = load_windows(p, role)
    parent_path, parent_meta = PARENT / f'{role}.npz', PARENT / f'{role}_cache.json'
    started = time.monotonic()
    if parent_path.exists():
        pm = json.loads(parent_meta.read_text())
        if digest(parent_path) != pm['sha256']:
            raise RuntimeError(f'Modified parent feature cache: {role}')
        if pm['floor_sha256'] != digest(PARENT / 'scale_floor.npy'):
            raise RuntimeError(f'Parent feature cache uses a different scale floor: {role}')
        old = dict(np.load(parent_path, allow_pickle=False))
        if not np.array_equal(old['y'], d['y']):
            raise RuntimeError(f'Parent labels do not match med windows: {role}')
        extra = old['distribution_only']
        bundle = old['summary']
        augmented = old['augmented']
        print('FEATURES', role, 'reused parent', extra.shape[0], flush=True)
    else:
        rows = []
        for i, (g, q) in enumerate(zip(d['gallery'], d['query'])):
            rows.append(comparison_distribution(g, q, floor))
            if (i + 1) % 2000 == 0:
                print('FEATURES', role, i + 1, len(d['gallery']), round(time.monotonic() - started), flush=True)
        extra = np.concatenate(rows)
        bundle = d['bundle'].reshape(-1, 618)
        augmented = np.concatenate((bundle, extra), axis=1)
    pause = d['pause'].reshape(-1, PAUSE_DIM)
    med = d['med'].reshape(-1, MED_DIM)
    if pause.shape[0] != len(d['y']) or med.shape[0] != len(d['y']):
        raise RuntimeError(f'Pause/MED rows do not match labels: {role}')
    return dict(summary=bundle, pause=pause, med=med, augmented=augmented,
                augmented_pause=np.concatenate((augmented, pause), axis=1),
                augmented_med=np.concatenate((augmented, med), axis=1),
                augmented_pause_med=np.concatenate((augmented, pause, med), axis=1),
                y=d['y'], users=d['users'])


def diagnostics(p):
    report, samples = {}, {}
    for role in ('train', 'selection'):
        d = load_windows(p, role)
        report[role] = {}
        for name, index in [('hold', 3), ('press_to_press', 22)]:
            shift = typical_timing_shift(d['gallery'], d['query'], index)
            samples[f'{role}_{name}'] = shift
            rng = np.random.default_rng(SEED)
            paired = shift[:, 1] - shift[:, 0]
            boot = [np.median(rng.choice(paired, len(paired), replace=True)) for _ in range(1000)]
            report[role][name] = dict(
                quantiles_ms=np.quantile(shift, [.1, .5, .9], axis=0).tolist(),
                normal_vs_mild_auc=float(roc_auc_score(np.repeat([0, 1], len(shift)),
                                                       np.r_[shift[:, 0], shift[:, 1]])),
                paired_mild_minus_normal_median_ms=float(np.median(paired)),
                participant_bootstrap_95ci_ms=np.quantile(boot, [.025, .975]).tolist(),
            )
        rates = d['med'][:, :, 0]
        report[role]['med_rate95_all'] = dict(
            class_medians=np.median(rates, axis=0).tolist(),
            normal_vs_mild_auc=float(roc_auc_score(np.repeat([0, 1], len(rates)),
                                                   np.r_[rates[:, 0], rates[:, 1]])),
            paired_mild_minus_normal_median=float(np.median(rates[:, 1] - rates[:, 0])),
        )
        pause_rates = d['pause'][:, :, 0]
        report[role]['pause_rate95_all'] = dict(
            class_medians=np.median(pause_rates, axis=0).tolist(),
            normal_vs_mild_auc=float(roc_auc_score(np.repeat([0, 1], len(pause_rates)),
                                                   np.r_[pause_rates[:, 0], pause_rates[:, 1]])),
        )
    write_json(OUT / 'signal_audit.json', report)
    np.savez_compressed(OUT / 'signal_audit_samples.npz', **samples)
    print('SIGNAL_AUDIT', json.dumps(report), flush=True)


def develop(p):
    if (OUT / 'search.json').exists() or (OUT / 'freeze.json').exists():
        raise RuntimeError('Already started; no repeat fitting')
    write_json(OUT / 'search.json', [])
    shutil.copy2(PARENT / 'scale_floor.npy', OUT / 'scale_floor.npy')
    floor = np.load(OUT / 'scale_floor.npy')
    if digest(OUT / 'scale_floor.npy') != json.loads((PARENT / 'freeze.json').read_text())['floor_sha256']:
        raise RuntimeError('Copied scale floor does not match the parent freeze')
    diagnostics(p)
    tr, sel = dataset(p, 'train', floor), dataset(p, 'selection', floor)
    history, winner, best = [], None, (-1., -float('inf'))
    parent_search = {row['representation']: row for row in p.get('parent_search', [])}
    parent_search.update({row['representation']: row for row in p.get('pause_search', [])})
    for name in ['summary', *p['representations']]:
        started = time.monotonic()
        model = classifier().fit(tr[name], tr['y'])
        prob = model.predict_proba(sel[name])
        tp = model.predict_proba(tr[name])
        score = dict(accuracy=float(np.mean(prob.argmax(1) == sel['y'])),
                     log_loss=float(log_loss(sel['y'], prob)))
        record = dict(representation=name, features=tr[name].shape[1], selection=score,
                      train_accuracy=float(np.mean(tp.argmax(1) == tr['y'])),
                      seconds=time.monotonic() - started)
        if name in parent_search:
            expected = parent_search[name]['selection']['accuracy']
            record['parent_selection_accuracy'] = expected
            if abs(score['accuracy'] - expected) > 1e-3:
                raise RuntimeError(f'{name} selection accuracy {score["accuracy"]} != parent {expected}')
        history.append(record)
        write_json(OUT / 'search.json', history)
        print('CANDIDATE', json.dumps(record), flush=True)
        joblib.dump(dict(model=model, representation=name), OUT / f'{name}.joblib')
        if name == 'summary':
            continue
        rank = score['accuracy'], -score['log_loss']
        if rank > best:
            best, winner = rank, dict(model=model, representation=name)
    cal = dataset(p, 'calibration', floor)
    baseline = joblib.load(OUT / 'summary.joblib')
    winner['temperature'] = temperature(cal['y'], winner['model'].predict_proba(cal[winner['representation']]))
    baseline['temperature'] = temperature(cal['y'], baseline['model'].predict_proba(cal['summary']))
    winner['scale_floor'] = floor
    baseline['scale_floor'] = floor
    joblib.dump(winner, OUT / 'model.joblib')
    joblib.dump(baseline, OUT / 'baseline.joblib')
    for name in ('augmented', 'augmented_pause'):
        if winner['representation'] != name and (OUT / f'{name}.joblib').exists():
            previous = joblib.load(OUT / f'{name}.joblib')
            previous['temperature'] = temperature(cal['y'], previous['model'].predict_proba(cal[name]))
            previous['scale_floor'] = floor
            joblib.dump(previous, OUT / f'{name}.joblib')
    name = winner['representation']
    development = {
        r: metrics(d['y'], calibrated(winner['model'].predict_proba(d[name]), winner['temperature']), d['users'])
        for r, d in [('train', tr), ('selection', sel), ('calibration', cal)]
    }
    write_json(OUT / 'development.json', development)
    write_json(OUT / 'freeze.json', dict(
        model_sha256=digest(OUT / 'model.joblib'), baseline_sha256=digest(OUT / 'baseline.joblib'),
        protocol_sha256=digest(OUT / 'protocol.json'), floor_sha256=digest(OUT / 'scale_floor.npy'),
        representation=name, time=time.time(), parent_protocol_sha256=p['parent_protocol_sha256'],
    ))
    print('FROZEN', name, flush=True)


def compare(p):
    if (OUT / 'comparison_started.json').exists():
        raise RuntimeError('Comparison already opened')
    f = json.loads((OUT / 'freeze.json').read_text())
    assert digest(OUT / 'model.joblib') == f['model_sha256']
    assert digest(OUT / 'baseline.joblib') == f['baseline_sha256']
    assert digest(OUT / 'protocol.json') == f['protocol_sha256']
    assert digest(OUT / 'scale_floor.npy') == f['floor_sha256']
    write_json(OUT / 'comparison_started.json', f)
    item = joblib.load(OUT / 'model.joblib')
    old = joblib.load(OUT / 'baseline.joblib')
    results = {}
    for role in ('validation', 'test'):
        d = dataset(p, role, item['scale_floor'])
        op = calibrated(old['model'].predict_proba(d['summary']), old['temperature'])
        raw = item['model'].predict_proba(d[item['representation']])
        prob = calibrated(raw, item['temperature'])
        results[role] = dict(
            classifier=metrics(d['y'], op, d['users']),
            distribution=metrics(d['y'], prob, d['users']),
            raw=metrics(d['y'], raw, d['users']),
            paired_change=paired_difference(d['y'], op, prob, d['users']),
        )
        if item['representation'] != 'augmented' and (OUT / 'augmented.joblib').exists():
            previous = joblib.load(OUT / 'augmented.joblib')
            ap = calibrated(previous['model'].predict_proba(d['augmented']), previous['temperature'])
            results[role]['previous_augmented'] = metrics(d['y'], ap, d['users'])
            results[role]['paired_change_vs_augmented'] = paired_difference(d['y'], ap, prob, d['users'])
        if item['representation'] != 'augmented_pause' and (OUT / 'augmented_pause.joblib').exists():
            previous = joblib.load(OUT / 'augmented_pause.joblib')
            pp = calibrated(previous['model'].predict_proba(d['augmented_pause']), previous['temperature'])
            results[role]['previous_pause'] = metrics(d['y'], pp, d['users'])
            results[role]['paired_change_vs_pause'] = paired_difference(d['y'], pp, prob, d['users'])
        np.savez_compressed(OUT / f'{role}_predictions.npz', y=d['y'], users=d['users'], p=prob)
        write_json(OUT / f'{role}_results.json', results[role])
        print('RESULT', role, json.dumps(results[role]), flush=True)
    write_json(OUT / 'results.json', results)
    dev = json.loads((OUT / 'development.json').read_text())
    search = json.loads((OUT / 'search.json').read_text())
    n = {role: len(p['splits'][role]) for role in ('train', 'selection', 'calibration', 'validation', 'test')}
    lines = [
        '# MED pause experiment on the full Aalto freeze', '',
        'Same people and synthetic draws as the full-Aalto parent freeze. MED is a length-5 Wiggins kurtosis filter estimated on each person\'s clean gallery press-to-press, applied to gallery and query, then scored with the same 42-d pause recipe. No generator pause mask. No maximum-entropy density. Selection is among med, augmented, augmented+med, augmented+pause, and augmented+pause+med by 4-way accuracy, then log loss. Chance accuracy is 25%.',
        '',
        f"Cohort: train {n['train']:,}; selection {n['selection']:,}; calibration {n['calibration']:,}; validation {n['validation']:,}; test {n['test']:,}.",
        '',
        'Selection candidates:',
    ]
    for rec in search:
        extra = f"; parent {rec['parent_selection_accuracy']:.2%}" if 'parent_selection_accuracy' in rec else ''
        lines.append(
            f"- {rec['representation']} ({rec['features']} features): selection {rec['selection']['accuracy']:.2%}, log loss {rec['selection']['log_loss']:.4f}; train {rec['train_accuracy']:.2%} (4-way){extra}"
        )
    lines += [
        '',
        f'Selected: **{item["representation"]}**. Train accuracy {dev["train"]["accuracy"]:.2%}; selection accuracy {dev["selection"]["accuracy"]:.2%} (4-way, chance 0.25).',
        '',
    ]
    if (PARENT / 'results.json').exists():
        parent_results = json.loads((PARENT / 'results.json').read_text())
        lines += ['## Parent freeze (augmented, no raw-series concat)', '']
        for role, r in parent_results.items():
            b = r['distribution']
            recall = b['per_class_recall']
            lines.append(
                f'- {role.title()}: {b["accuracy"]:.2%}; EER {b["pooled_detection_eer"]:.2%}. Recall normal {100 * recall["normal"]:.1f}%, mild {100 * recall["mild"]:.1f}%, moderate {100 * recall["moderate"]:.1f}%, severe {100 * recall["severe"]:.1f}%.'
            )
        lines += ['']
    lines += [
        '## Four-class accuracy versus the 618-feature summary',
        '',
    ]
    for role, r in results.items():
        a, b, z = r['classifier'], r['distribution'], r['paired_change']
        ci = z['participant_bootstrap_95ci']
        recall = b['per_class_recall']
        lines += [
            f'- **{role.title()}:** {a["accuracy"]:.2%} → **{b["accuracy"]:.2%}**; change {100 * z["accuracy_difference"]:+.2f} pp (95% participant-bootstrap CI {100 * ci[0]:+.2f}, {100 * ci[1]:+.2f}).',
            f'  EER {a["pooled_detection_eer"]:.2%} → {b["pooled_detection_eer"]:.2%}; ECE {100 * a["ece_15_equal_width"]:.2f} → {100 * b["ece_15_equal_width"]:.2f} pp; log loss {a["log_loss"]:.4f} → {b["log_loss"]:.4f}.',
            f'  Recall: normal {100 * recall["normal"]:.1f}%, mild {100 * recall["mild"]:.1f}%, moderate {100 * recall["moderate"]:.1f}%, severe {100 * recall["severe"]:.1f}%.',
        ]
        if 'paired_change_vs_augmented' in r:
            pa, pz = r['previous_augmented'], r['paired_change_vs_augmented']
            pci = pz['participant_bootstrap_95ci']
            lines.append(
                f'  Versus previous augmented: {pa["accuracy"]:.2%} → {b["accuracy"]:.2%}; change {100 * pz["accuracy_difference"]:+.2f} pp (95% CI {100 * pci[0]:+.2f}, {100 * pci[1]:+.2f}).'
            )
        if 'paired_change_vs_pause' in r:
            pa, pz = r['previous_pause'], r['paired_change_vs_pause']
            pci = pz['participant_bootstrap_95ci']
            lines.append(
                f'  Versus previous pause concat: {pa["accuracy"]:.2%} → {b["accuracy"]:.2%}; change {100 * pz["accuracy_difference"]:+.2f} pp (95% CI {100 * pci[0]:+.2f}, {100 * pci[1]:+.2f}).'
            )
    if (PAUSE / 'results.json').exists() and (PAUSE / 'search.json').exists():
        pause_search = json.loads((PAUSE / 'search.json').read_text())
        pause_results = json.loads((PAUSE / 'results.json').read_text())
        pause_sel = next(r for r in pause_search if r['representation'] == 'augmented_pause')
        lines += ['', '## Pause concat on the same parent', '']
        lines.append(
            f"Selected pause representation: augmented_pause. Selection {pause_sel['selection']['accuracy']:.2%}; train {pause_sel['train_accuracy']:.2%} (4-way)."
        )
        for role, r in pause_results.items():
            b, z = r['distribution'], r['paired_change_vs_augmented']
            ci = z['participant_bootstrap_95ci']
            recall = b['per_class_recall']
            lines.append(
                f'- {role.title()}: {r["previous_augmented"]["accuracy"]:.2%} → {b["accuracy"]:.2%}; change {100 * z["accuracy_difference"]:+.2f} pp (95% CI {100 * ci[0]:+.2f}, {100 * ci[1]:+.2f}). Recall normal {100 * recall["normal"]:.1f}%, mild {100 * recall["mild"]:.1f}%, moderate {100 * recall["moderate"]:.1f}%, severe {100 * recall["severe"]:.1f}%.'
            )
    lines += ['',
              'MED descriptors are absolute spikes after an enrollment-only kurtosis deconvolution of press-to-press. They are not a fitted maximum-entropy density. These synthetic classes are not clinically validated severity labels.']
    text = '\n'.join(lines) + '\n'
    (HERE / 'RESULTS.md').write_text(text)
    (OUT / 'RESULTS.md').write_text(text)


def load_frozen():
    p = json.loads((OUT / 'protocol.json').read_text())
    f = json.loads((OUT / 'freeze.json').read_text())
    if digest(OUT / 'model.joblib') != f['model_sha256'] or digest(OUT / 'protocol.json') != f['protocol_sha256']:
        raise RuntimeError('Frozen artifacts changed')
    for file, sha in p['source_hashes'].items():
        if digest(ROOT / file) != sha:
            raise RuntimeError(f'Frozen source changed: {file}')
    return joblib.load(OUT / 'model.joblib')


def encode_observed(gallery, gallery_lengths, query, query_lengths, floor, representation):
    if gallery.shape != (10, 50, 5) or query.shape != (10, 50, 5):
        raise ValueError('Ten separate windows per side required')
    for x, lengths in ((gallery, gallery_lengths), (query, query_lengths)):
        if np.shape(lengths) != (10,) or not np.issubdtype(np.asarray(lengths).dtype, np.integer):
            raise ValueError('Ten integer lengths required')
        if not np.isfinite(x).all() or np.any((np.asarray(lengths) < 6) | (np.asarray(lengths) > 50)):
            raise ValueError('Finite windows with 6–50 events required')
    ref = enrollment(gallery, gallery_lengths)
    g = np.nan_to_num(ref[0], nan=0, posinf=10, neginf=-10).astype('float32')
    rows = np.stack([session_vector(x, n, ref) for x, n in zip(query, query_lengths)])
    extra = comparison_distribution(g, rows[:, :57][None], floor)
    bundle = bundle_vector(rows)
    pause = pause_vector(gallery, gallery_lengths, query, query_lengths)
    hold = hold_vector(gallery, gallery_lengths, query, query_lengths)
    med = med_vector(gallery, gallery_lengths, query, query_lengths)
    if representation == 'summary':
        return bundle[None]
    if representation == 'pause':
        return pause[None]
    if representation == 'hold':
        return hold[None]
    if representation == 'med':
        return med[None]
    if representation == 'distribution_only':
        return extra
    if representation == 'augmented':
        return np.concatenate((bundle[None], extra), axis=1)
    if representation == 'augmented_pause':
        return np.concatenate((bundle[None], extra, pause[None]), axis=1)
    if representation == 'augmented_hold':
        return np.concatenate((bundle[None], extra, hold[None]), axis=1)
    if representation == 'augmented_med':
        return np.concatenate((bundle[None], extra, med[None]), axis=1)
    if representation == 'augmented_pause_med':
        return np.concatenate((bundle[None], extra, pause[None], med[None]), axis=1)
    raise ValueError(f'Unknown representation: {representation}')


def predict(path):
    item = load_frozen()
    with np.load(path, allow_pickle=False) as d:
        x = encode_observed(d['gallery'], d['gallery_lengths'], d['query'], d['query_lengths'],
                            item['scale_floor'], item['representation'])
    with threadpool_limits(limits=4):
        prob = calibrated(item['model'].predict_proba(x), item['temperature'])[0]
    print(json.dumps(dict(predicted_class=CLASS_NAMES[prob.argmax()],
                          probabilities=dict(zip(CLASS_NAMES, prob.tolist()))), indent=2))


def main():
    parser = argparse.ArgumentParser(__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input', nargs='?', help='NPZ with gallery/query windows for inference')
    parser.add_argument('--phase', choices=['develop', 'compare'])
    parser.add_argument('--device', default='mps', help='Requested device. The kept model is sklearn HGB and fits on CPU; MPS has no sklearn backend.')
    args = parser.parse_args()
    try:
        import torch
        mps = bool(torch.backends.mps.is_available())
    except Exception:
        mps = False
    print(f'DEVICE requested={args.device} mps_available={mps} classifier=sklearn-cpu', flush=True)
    if args.input:
        predict(args.input)
        return
    if args.phase is None:
        parser.error('provide observed.npz or --phase develop|compare')
    p = register()
    with threadpool_limits(limits=4):
        (develop if args.phase == 'develop' else compare)(p)


if __name__ == '__main__':
    main()
'''


def legacy_source_sha256():
    value = _hashlib.sha256(_LEGACY_SOURCE.encode()).hexdigest()
    protocol = _json.loads((_Path(__file__).resolve().parent / 'runs_med/protocol.json').read_text())
    expected = protocol['source_hashes']['prototype_net/exp3/model.py']
    if value != expected:
        raise RuntimeError('Archived frozen model source changed')
    return value


legacy_source_sha256()
_legacy_namespace = {'__file__': __file__, '__name__': 'prototype_net.exp3.model'}
exec(compile(_LEGACY_SOURCE, __file__ + ':frozen', 'exec'), _legacy_namespace)
_legacy_digest = _legacy_namespace['digest']


def _verified_legacy_digest(path):
    if _Path(path).resolve() == _Path(__file__).resolve():
        return legacy_source_sha256()
    return _legacy_digest(path)


_legacy_namespace['digest'] = _verified_legacy_digest
for _key, _value in _legacy_namespace.items():
    if not _key.startswith('__'):
        globals()[_key] = _value
_legacy_main = _legacy_namespace['main']

# Binary implementation follows. It uses full-file hashes, never the legacy digest.

import sys as _sys
import tempfile as _tempfile
from sklearn.metrics import roc_auc_score as _binary_auc
from prototype_net.perturbation_generator import ClinicalCalibration, ClinicalDeviationGenerator

_BINARY_NAMES = ('untouched', 'cognitive', 'motor', 'combined')
_BINARY_SEED = 9202026


def _binary_raw_digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _binary_worker_init(calibration, floor, pause_shape=1., hazard_multiplier=1.):
    global _binary_generator, _binary_floor
    threadpool_limits(limits=1)
    _binary_generator = ClinicalDeviationGenerator(calibration, pause_shape=pause_shape,
                                                   hazard_multiplier=hazard_multiplier)
    _binary_floor = floor


def _binary_encode(gallery, lengths, query, query_lengths, reference=None):
    reference = enrollment(gallery, lengths) if reference is None else reference
    g = np.nan_to_num(reference[0], nan=0, posinf=10, neginf=-10).astype('float32')
    rows = np.stack([session_vector(x, n, reference) for x, n in zip(query, query_lengths)])
    return np.concatenate((bundle_vector(rows), comparison_distribution(g, rows[:, :57][None], _binary_floor)[0],
                           pause_vector(gallery, lengths, query, query_lengths))).astype('float32')


def _binary_person(task):
    role, uid, draw = task
    path = DATA / f'{uid}_keystrokes.txt'
    gallery, query = ordered_blocks(path)
    gallery, query = gallery[-10:], query[:10]
    gx = np.stack([x for x, n, h in gallery]); gn = np.array([n for x, n, h in gallery])
    qx = np.stack([x for x, n, h in query]); qn = np.array([n for x, n, h in query])
    g_before, q_before = gx.copy(), qx.copy()
    reference = enrollment(gx, gn)
    rows = [_binary_encode(gx, gn, qx, qn, reference)]
    details = []
    for kind in _BINARY_NAMES[1:]:
        seed = int.from_bytes(hashlib.sha256(f'binary:{_BINARY_SEED}:{role}:{uid}:{kind}:{draw}'.encode()).digest()[:8], 'little')
        observed, detail = _binary_generator.perturb(gx, gn, qx, qn, kind, np.random.default_rng(seed))
        rows.append(_binary_encode(gx, gn, observed, qn, reference))
        details.append(detail)
    if not np.array_equal(gx, g_before) or not np.array_equal(qx, q_before):
        raise RuntimeError('The generator mutated untouched enrollment/query data')
    result = np.stack(rows)
    if result.shape != (4, 1485) or not np.isfinite(result).all():
        raise RuntimeError('Invalid binary features')
    return result, details, _binary_raw_digest(path)


def _binary_role(role, protocol, calibration, floor, cache, workers):
    ids = list(protocol['splits'][role])
    source_role = 'train' if role in ('fit', 'development', 'threshold_calibration') else role
    availability = json.loads((OUT / 'availability_selection.json').read_text())[source_role]['source_sha256']
    expected = [availability[uid] for uid in ids]
    path = cache / f'{role}.joblib'
    contract = dict(source=_binary_raw_digest(__file__), generator=_binary_raw_digest(ROOT / 'prototype_net/perturbation_generator/__init__.py'),
                    legacy_generator=_binary_raw_digest(ROOT / parent_source()), clinical=calibration.source_sha256,
                    protocol=_binary_raw_digest(OUT / 'protocol.json'), ids=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                    floor=hashlib.sha256(np.asarray(floor).tobytes()).hexdigest(), seed=_BINARY_SEED)
    if path.exists():
        value = joblib.load(path)
        if value['contract'] != contract or value['ids'] != ids or value['source_hashes'] != expected:
            raise RuntimeError(f'Stale binary cache: {path}; choose another scratch cache directory')
        if any(_binary_raw_digest(DATA / f'{uid}_keystrokes.txt') != sha for uid, sha in zip(ids, expected)):
            raise RuntimeError('Raw source changed since feature caching')
        return value
    x = np.empty((len(ids), 4, 1485), dtype=np.float32)
    counts, hashes = [], []
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers, initializer=_binary_worker_init, initargs=(calibration, floor)) as pool:
        for index, (row, details, sha) in enumerate(pool.map(_binary_person, [(source_role, uid, 0) for uid in ids], chunksize=20)):
            if sha != expected[index]:
                raise RuntimeError(f'Raw source changed for {ids[index]}')
            x[index] = row; counts.append(details); hashes.append(sha)
            if (index + 1) % 1000 == 0:
                print(json.dumps(dict(event='features', role=role, people=index+1, total=len(ids), seconds=time.monotonic()-started)), flush=True)
    value = dict(x=x, ids=ids, details=counts, source_hashes=hashes, contract=contract)
    joblib.dump(value, path)
    return value


def _binary_metrics(probabilities, threshold):
    p = np.asarray(probabilities).reshape(-1, 4)
    flagged = p >= threshold
    correctness = .5 * (~flagged[:, 0]) + flagged[:, 1:].mean(1) * .5
    y = np.tile([0, 1, 1, 1], len(p))
    weights = np.tile([2., 2/3, 2/3, 2/3], len(p))
    rng = np.random.default_rng(_BINARY_SEED)
    # Independent participant resampling, preserving all four examples per person.
    quantities = np.c_[correctness, flagged[:, 0], flagged[:, 1:]]
    boots = np.stack([quantities[rng.integers(len(p), size=len(p))].mean(0) for _ in range(1000)])
    lo, hi = np.quantile(boots, [.025, .975], axis=0)
    return dict(participants=len(p), threshold=float(threshold), balanced_accuracy=float(correctness.mean()),
                accuracy_95ci=[float(lo[0]), float(hi[0])], false_positive_rate=float(flagged[:, 0].mean()),
                false_positive_95ci=[float(lo[1]), float(hi[1])],
                sensitivity=float(flagged[:, 1:].mean()),
                mechanism_sensitivity={name:dict(value=float(flagged[:, i].mean()), ci=[float(lo[i+1]), float(hi[i+1])])
                                       for i, name in enumerate(_BINARY_NAMES[1:], 1)},
                auroc=float(_binary_auc(y, p.ravel(), sample_weight=weights)))


def _binary_source_contract():
    return {name: _binary_raw_digest(ROOT / name) for name in (
        'prototype_net/exp3/model.py', 'prototype_net/perturbation_generator/__init__.py', parent_source(),
        'data/clinical/ad_mci_writing/Participant_level.csv')}


def _binary_training_partition(protocol):
    # Membership is determined before viewing data, scores or labels.
    original = protocol['splits']
    sets = {r:set(map(str, ids)) for r, ids in original.items()}
    if any(sets[r] & sets[s] for i, r in enumerate(sets) for s in list(sets)[:i]):
        raise RuntimeError('Original participant roles overlap')
    ids = sorted(map(str, original['train']),
                 key=lambda uid: hashlib.sha256(f'binary-training-partition:9232026:{uid}'.encode()).digest())
    if len(ids) != len(set(ids)) or len(ids) < 100:
        raise RuntimeError('Insufficient or repeated training participants')
    fit_end, dev_end = int(.70 * len(ids)), int(.85 * len(ids))
    return dict(splits=dict(fit=ids[:fit_end], development=ids[fit_end:dev_end],
                            threshold_calibration=ids[dev_end:]))


def _binary_fit_gallery(uid):
    gallery, _ = ordered_blocks(DATA / f'{uid}_keystrokes.txt')
    gallery = gallery[-10:]
    return enrollment(np.stack([x for x,n,h in gallery]), np.array([n for x,n,h in gallery]))[0].astype('float32')


def _binary_threshold(normal, false_positive_target=.05):
    normal = np.asarray(normal, dtype=float)
    if normal.ndim != 1 or not len(normal) or not np.isfinite(normal).all():
        raise ValueError('Finite training-calibration negative scores required')
    if not 0 < false_positive_target < 1:
        raise ValueError('Invalid false-positive target')
    allowed = int(np.floor(false_positive_target * len(normal)))
    return float(np.nextafter(np.sort(normal)[::-1][allowed], np.inf))


def _binary_development_case(task):
    uid, draw, shape = task
    gallery, query = ordered_blocks(DATA / f'{uid}_keystrokes.txt')
    gallery, query = gallery[-10:], query[:10]
    gx = np.stack([x for x,n,h in gallery]); gn = np.array([n for x,n,h in gallery])
    qx = np.stack([x for x,n,h in query]); qn = np.array([n for x,n,h in query])
    before = qx.copy(); gallery_before = gx.copy()
    reference = enrollment(gx, gn)
    generator = ClinicalDeviationGenerator(_binary_generator.calibration, pause_shape=shape)
    rows = [_binary_encode(gx, gn, qx, qn, reference)]
    changes = []
    for kind in _BINARY_NAMES[1:]:
        seed = int.from_bytes(hashlib.sha256(f'binary-development:9232026:{uid}:{kind}:{draw}'.encode()).digest()[:8], 'little')
        observed, detail = generator.perturb(gx, gn, qx, qn, kind, np.random.default_rng(seed))
        if not np.array_equal(observed[:,:,4], qx[:,:,4]):
            raise RuntimeError('Generator changed keys')
        for z, original, n in zip(observed, qx, qn):
            if not np.array_equal(z[n:], original[n:]):
                raise RuntimeError('Generator changed padding')
            if not np.allclose(z[:n-1,1], z[:n-1,2]-z[:n-1,0], atol=1e-5) or not np.allclose(z[:n-1,3], z[:n-1,1]+z[1:n,0], atol=1e-5):
                raise RuntimeError('Generator broke timing identities')
        if kind == 'cognitive' and not np.array_equal(observed[:,:,0], qx[:,:,0]):
            raise RuntimeError('Cognitive generator changed holds')
        # Diagnostic attenuation only; these strengths are not clinical severity labels.
        for strength in (.125, .25, .5, 1.):
            z = qx.copy()
            z[:,:,:4] = qx[:,:,:4] + strength * (observed[:,:,:4] - qx[:,:,:4])
            rows.append(_binary_encode(gx, gn, z, qn, reference))
            ratios = []
            for a,b,n in zip(z,qx,qn):
                keep = b[:n,0] > 0
                ratios.extend(np.log(np.maximum(a[:n,0][keep],1e-12)/b[:n,0][keep]))
            delta = np.concatenate([a[:n-1,2]-b[:n-1,2] for a,b,n in zip(z,qx,qn)])
            changes.append([float(np.var(ratios)), float(np.sqrt(np.mean(delta**2))),
                            float(np.max(delta)), int(np.count_nonzero(delta))])
    if not np.array_equal(qx,before) or not np.array_equal(gx,gallery_before):
        raise RuntimeError('Generator mutated raw class zero or gallery')
    rows = np.stack(rows)
    if rows.shape != (13,1485) or not np.isfinite(rows).all():
        raise RuntimeError('Invalid development features')
    return rows, changes


def _binary_development_checks(item, partition, args, fit):
    from sklearn.tree import DecisionTreeClassifier
    # Fixed size, strength grid and draws. Only original training participants.
    ids = partition['splits']['development'][:300]
    fit_ids = set(partition['splits']['fit'])
    if fit_ids & set(ids) or set(ids) & set(partition['splits']['threshold_calibration']):
        raise RuntimeError('Development and parameter-fitting identities overlap')
    cols = [7,8,26,27,64,65,83,84]  # hold/press mean and std, absolute and gallery-relative
    stump = DecisionTreeClassifier(max_depth=1, min_samples_leaf=200, random_state=9232026)
    x = fit['x'].reshape(-1,1485)
    y = np.tile([0,1,1,1],len(fit['ids']))
    stump.fit(x[:,cols], y, sample_weight=np.tile([2.,2/3,2/3,2/3],len(fit['ids'])))
    checks = {}
    for name, draw, shape in [('draw0',0,1.),('draw1',1,1.),('draw2',2,1.),
                              ('pause_shape_half',0,.5),('pause_shape_double',0,2.)]:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_binary_worker_init,
                                 initargs=(item['calibration'],item['scale_floor'])) as pool:
            values = list(pool.map(_binary_development_case, [(uid,draw,shape) for uid in ids], chunksize=10))
        features = np.stack([v[0] for v in values]); changes = np.array([v[1] for v in values])
        with threadpool_limits(limits=4):
            scores = item['model'].predict_proba(features.reshape(-1,1485))[:,1].reshape(len(ids),13)
        simple = stump.predict_proba(features.reshape(-1,1485)[:,cols])[:,1].reshape(len(ids),13)
        threshold = item['threshold']; flags = scores >= threshold
        strata = {}
        for k, kind in enumerate(_BINARY_NAMES[1:]):
            full = flags[:,1+4*k+3]
            for j, strength in enumerate((.125,.25,.5,1.)):
                col = 1+4*k+j
                labels = np.r_[np.zeros(len(ids)),np.ones(len(ids))]
                difference = full.astype(float)-flags[:,col].astype(float)
                rng = np.random.default_rng(9232026)
                boot = np.array([difference[rng.integers(len(ids),size=len(ids))].mean() for _ in range(1000)])
                ci = np.quantile(boot,[.025,.975]).tolist()
                strata[f'{kind}:{strength}'] = dict(sensitivity=float(flags[:,col].mean()),
                    balanced_accuracy=float(((~flags[:,0]).mean()+flags[:,col].mean())/2),
                    auroc=float(_binary_auc(labels,np.r_[scores[:,0],scores[:,col]])),
                    simple_stump_auroc=float(_binary_auc(labels,np.r_[simple[:,0],simple[:,col]])),
                    sensitivity_drop_vs_full=float(difference.mean()), drop_95ci=ci,
                    material_drop=bool(ci[0] > .10),
                    mean_log_hold_change_variance=float(changes[:,col-1,0].mean()),
                    press_change_rms_quantiles=np.quantile(changes[:,col-1,1],[.05,.5,.95]).tolist(),
                    maximum_added_press_seconds=float(changes[:,col-1,2].max()))
        checks[name] = dict(participants=len(ids), false_positive_rate=float(flags[:,0].mean()), strata=strata)
        print(json.dumps(dict(event='development_check',name=name,results=checks[name])),flush=True)
    failures = [f'{name}/{stratum}' for name,c in checks.items() for stratum,v in c['strata'].items() if v['material_drop']]
    return dict(participants=ids, checks=checks, invariants_passed=True,
                decision_rule='Flag a material sensitivity loss when the paired participant-bootstrap 95% lower bound exceeds 10 percentage points versus full strength. Diagnostic rule, not a clinical acceptance standard.',
                material_drop_cases=failures,
                coverage_status='insufficient_weak_change_coverage' if failures else 'no_material_drop_detected_on_this_suite',
                clinically_validated=False,
                note='Attenuation and pause shapes are stress tests, not clinically fitted progression. Simple stump is a shortcut diagnostic, not a replacement learner.')


def binary_study(args):
    if args.output.exists():
        raise RuntimeError('Refusing to overwrite an existing binary study')
    load_frozen()
    original = json.loads((OUT / 'protocol.json').read_text())
    partition = _binary_training_partition(original)
    args.cache.mkdir(parents=True, exist_ok=True)
    # Persist the protocol before loading feature data or producing any model scores.
    design = dict(version='training_only_v2', partition=partition,
                  config=MODEL_CONFIG, false_positive_target=.05,
                  development_suite=dict(people=300, strengths=[.125,.25,.5,1.],draws=[0,1,2],pause_shapes=[.5,1.,2.]),
                  heldout_policy='No validation/test observation may fit parameters, choose thresholds, or guide development.')
    design_path = args.output.with_suffix('.protocol.json')
    if design_path.exists():
        raise RuntimeError('Refusing to overwrite an existing study protocol')
    write_json(design_path,design)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        galleries = list(pool.map(_binary_fit_gallery,partition['splits']['fit'],chunksize=30))
    floor = scale_floor(np.stack(galleries)); del galleries
    calibration = ClinicalCalibration.from_csv(ROOT / 'data/clinical/ad_mci_writing/Participant_level.csv')
    fit = _binary_role('fit',partition,calibration,floor,args.cache,args.workers)
    x = fit['x'].reshape(-1,1485)
    y = np.tile([0,1,1,1],len(fit['ids']))
    model = HistGradientBoostingClassifier(**MODEL_CONFIG)
    with threadpool_limits(limits=4):
        model.fit(x,y,sample_weight=np.tile([2.,2/3,2/3,2/3],len(fit['ids'])))
        train_p = model.predict_proba(x)[:,1].reshape(-1,4)
    item = dict(model=model,scale_floor=floor,calibration=calibration,seed=_BINARY_SEED,
                source_hashes=_binary_source_contract(),parent_model_sha256=_binary_raw_digest(OUT/'model.joblib'),
                parent_protocol_sha256=_binary_raw_digest(OUT/'protocol.json'),
                design=design,design_sha256=_binary_raw_digest(design_path),representation='augmented_pause',
                positive_mechanisms=_BINARY_NAMES[1:],completed=False,
                class_weights='untouched 2; each of three positives 2/3',
                interpretation='No injected deviation versus injected deviation relative to personal enrollment; not diagnosis')
    calibration_data = _binary_role('threshold_calibration',partition,calibration,floor,args.cache,args.workers)
    with threadpool_limits(limits=4):
        cp = model.predict_proba(calibration_data['x'].reshape(-1,1485))[:,1].reshape(-1,4)
    threshold = _binary_threshold(cp[:,0])
    item.update(threshold=threshold,threshold_source='original train / threshold_calibration negatives only',
                threshold_rule='Score strictly above (floor(.05*N)+1)th largest training-calibration negative; flag >= threshold',
                results={r:dict(at_half=_binary_metrics(p,.5),operating=_binary_metrics(p,threshold))
                         for r,p in [('fit',train_p),('threshold_calibration',cp)]})
    if item['results']['threshold_calibration']['operating']['false_positive_rate'] > .05:
        raise RuntimeError('Training-calibration FPR constraint failed')
    joblib.dump(item,args.output)
    print(json.dumps(dict(event='training_threshold_locked',threshold=threshold,results=item['results'])),flush=True)
    del calibration_data
    development = _binary_role('development',partition,calibration,floor,args.cache,args.workers)
    with threadpool_limits(limits=4):
        dp = model.predict_proba(development['x'].reshape(-1,1485))[:,1].reshape(-1,4)
    item['results']['development'] = dict(at_half=_binary_metrics(dp,.5),operating=_binary_metrics(dp,threshold))
    del development
    item['generation_validation'] = _binary_development_checks(item,partition,args,fit)
    item['completed'] = True
    item['heldout_evaluated'] = False
    joblib.dump(item,args.output)
    summary = {k:v for k,v in item.items() if k not in ('model','scale_floor','calibration')}
    write_json(args.output.with_suffix('.json'),summary)
    print(json.dumps(dict(event='complete_training_only',results=item['results'],coverage=item['generation_validation']['coverage_status'])),flush=True)


def binary_evaluate(args):
    # Read-only model evaluation. Never writes back model, parameters or threshold.
    item = joblib.load(args.output)
    if not item.get('completed') or item.get('design',{}).get('version') != 'training_only_v2':
        raise RuntimeError('Evaluation requires a completed training-only v2 artifact')
    if item['source_hashes'] != _binary_source_contract():
        raise RuntimeError('Binary source or calibration data changed')
    target = args.output.with_suffix('.evaluation.json')
    if target.exists():
        raise RuntimeError('Heldout evaluation already exists; do not repeatedly consult it during development')
    original = json.loads((OUT/'protocol.json').read_text())
    if item['parent_protocol_sha256'] != _binary_raw_digest(OUT/'protocol.json'):
        raise RuntimeError('Original protocol changed')
    before = _binary_raw_digest(args.output)
    args.cache.mkdir(parents=True,exist_ok=True)
    results = {}
    for role in ('validation','test'):
        data = _binary_role(role,original,item['calibration'],item['scale_floor'],args.cache,args.workers)
        with threadpool_limits(limits=4):
            p = item['model'].predict_proba(data['x'].reshape(-1,1485))[:,1].reshape(-1,4)
        results[role] = _binary_metrics(p,item['threshold'])
    if before != _binary_raw_digest(args.output):
        raise RuntimeError('Evaluation modified the frozen artifact')
    write_json(target,dict(model_sha256=before,threshold=item['threshold'],results=results,
                          interpretation='Descriptive evaluation on a historically examined cohort, not pristine confirmation. No fitting.'))
    print(json.dumps(results),flush=True)


def binary_predict(args):
    item = joblib.load(args.output)
    if not item.get('completed'):
        raise RuntimeError('Binary study is incomplete')
    if item['source_hashes'] != _binary_source_contract():
        raise RuntimeError('Binary source or clinical calibration data changed')
    if item['parent_model_sha256'] != _binary_raw_digest(OUT / 'model.joblib') or item['parent_protocol_sha256'] != _binary_raw_digest(OUT / 'protocol.json'):
        raise RuntimeError('Parent frozen artifacts changed')
    _binary_worker_init(item['calibration'], item['scale_floor'])
    with np.load(args.input, allow_pickle=False) as d:
        # Reuse the frozen input validation and exact feature contract.
        x = encode_observed(d['gallery'], d['gallery_lengths'], d['query'], d['query_lengths'],
                            item['scale_floor'], 'augmented_pause')
    with threadpool_limits(limits=4):
        score = float(item['model'].predict_proba(x)[0, 1])
    print(json.dumps(dict(deviation_score=score, threshold=item['threshold'], flag=score >= item['threshold'],
                          meaning='Deviation from enrolled baseline; not a diagnosis', calibrated_probability=False)))


def binary_main():
    parser = argparse.ArgumentParser(description='Untouched Aalto gallery/query versus class-1-only clinical deviations')
    parser.add_argument('--binary', action='store_true')
    parser.add_argument('--phase', choices=('study', 'evaluate', 'predict'), required=True)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', type=Path, default=OUT / 'binary_deviation_trainonly.joblib')
    parser.add_argument('--cache', type=Path, default=Path(_tempfile.gettempdir()) / 'exp3_binary_trainonly')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1 or not args.output.parent.is_dir():
        parser.error('Positive workers and an existing output directory required')
    if args.phase == 'predict' and args.input is None:
        parser.error('Prediction requires --input with personal gallery and query windows')
    {'study':binary_study, 'evaluate':binary_evaluate, 'predict':binary_predict}[args.phase](args)


if __name__ == '__main__':
    binary_main() if '--binary' in _sys.argv else _legacy_main()
