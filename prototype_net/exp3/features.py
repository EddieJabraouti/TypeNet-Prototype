"""Timing summaries available at inference, with no clean query twin or trace.

Only hold and press-to-press timings enter the model. Every class is rounded
to milliseconds, and the final transition is always discarded. This prevents
the current generator's precision and terminal-transition artifacts becoming
class labels. Keycodes only index clean enrollment statistics, never labels.
"""
from __future__ import annotations

import numpy as np


def channels(x, length):
    n = int(length)
    h = np.maximum(np.rint(x[:n, 0].astype(float) * 1000) / 1000, .001)
    p = np.maximum(np.rint(x[:n-1, 2].astype(float) * 1000) / 1000, .001)
    k = np.rint(x[:n, 4] * 255).astype(int)
    return np.log(h), np.log(p), k


def summary(v):
    v = np.asarray(v, dtype=float)
    if len(v) == 0:
        return np.zeros(19)
    quant = np.quantile(v, [.05, .1, .25, .5, .75, .9, .95])
    center = v - np.mean(v)
    variance = np.mean(center**2)
    extras = [np.mean(v), np.std(v), np.mean(np.abs(v - quant[3])),
              quant[6]-quant[0], np.mean(np.clip(center, -3, 3)**3)]
    for lag in (1, 2, 3, 5):
        extras.append(np.mean(center[:-lag]*center[lag:]) if len(v)>lag else 0.)
    extras += [np.mean(np.abs(np.diff(v))) if len(v)>1 else 0.,
               np.mean(v > quant[3]+1), variance]
    return np.r_[quant, extras]


def enrollment(galleries, lengths):
    cs = [channels(g, n) for g, n in zip(galleries, lengths)]
    all_h = np.concatenate([c[0] for c in cs])
    all_p = np.concatenate([c[1] for c in cs])
    all_k = np.concatenate([c[2] for c in cs])
    # Shrink key-specific estimates toward this person's overall hold time.
    global_mean = np.mean(all_h)
    centers = {int(k): (all_h[all_k == k].sum()+5*global_mean)/
               ((all_k == k).sum()+5) for k in np.unique(all_k)}
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
    # Compare to every enrollment session, then summarize those comparisons.
    delta = raw[None, :] - gallery_rows
    features = [raw, np.median(delta, axis=0), np.std(gallery_rows, axis=0),
                np.quantile(delta, .1, axis=0), np.quantile(delta, .9, axis=0)]
    for v, base in ((h, gh), (p, gp)):
        q = np.quantile(base, [.05, .1, .25, .5, .75, .9, .95])
        features.append(np.mean(v[:, None] > q, axis=0))
        features.append(np.array([np.mean(v > q[3]+np.log(t)) for t in (2, 3, 5)]))
    # Motor noise is correlated; compare hold residual change at several lags.
    for lag in (1, 2, 3, 5):
        features.append(np.array([np.std(residual[lag:]-residual[:-lag])
                                  if len(residual)>lag else 0.]))
    return np.nan_to_num(np.concatenate(features), nan=0, posinf=10, neginf=-10).astype('float32')


def bundle_vector(rows):
    """Five observed query sessions, all under the same synthetic user profile."""
    return np.r_[np.mean(rows, axis=0), np.std(rows, axis=0)].astype('float32')
