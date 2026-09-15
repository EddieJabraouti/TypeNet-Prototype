"""Identity-conditioned timing residual, classified with a 4-way head.

TypeNet is frozen and used only as "who this person is" (gallery mean
embedding). A trainable LSTM reads raw holds/gaps. The class head sees the
query-minus-gallery residual in that timing space, never TypeNet embeddings
and never perturber internals (no pause masks, delays, or profiles).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from paper_typenet import nn as typenet

CLASS_NAMES = ("normal", "mild", "moderate", "severe")
CLASS_COUNT = len(CLASS_NAMES)
SEVERITY_NAMES = CLASS_NAMES[1:]
RESIDUAL_HEAD_HIDDEN = 256
RESIDUAL_HEAD_DROPOUT = 0.2


@dataclass(frozen=True)
class FourWayAccuracy:
    overall: float
    by_class: Mapping[str, float]
    confusion: Mapping[str, Mapping[str, float]] | None = None


class IdentityResidualNet(nn.Module):
    """FiLM a raw-timing LSTM with frozen identity; classify the residual."""

    def __init__(
        self,
        hidden_size: int = 128,
        identity_dim: int = typenet.PAPER_EMBEDDING_DIM,
        magnitude_prior: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = typenet.KerasStyleLSTM(
            typenet.PAPER_FEATURE_COUNT,
            hidden_size,
            recurrent_dropout=0.2,
        )
        self.film_proj = nn.Linear(identity_dim, 2 * hidden_size)
        self.attn = nn.Linear(hidden_size, 1)
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, RESIDUAL_HEAD_HIDDEN),
            nn.ReLU(),
            nn.Dropout(RESIDUAL_HEAD_DROPOUT),
            nn.Linear(RESIDUAL_HEAD_HIDDEN, CLASS_COUNT),
        )
        self.magnitude_prior = float(magnitude_prior)

    def _film(self, hidden: Tensor, identity: Tensor) -> Tensor:
        gamma, beta = self.film_proj(identity).chunk(2, dim=-1)
        gamma = torch.tanh(gamma)
        while gamma.dim() < hidden.dim():
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
        return hidden * (1.0 + gamma) + beta

    def _encode(
        self, inputs: Tensor, lengths: Tensor, identity: Tensor
    ) -> tuple[Tensor, Tensor]:
        sequence, hidden = self.lstm(inputs, lengths)
        return self._film(sequence, identity), self._film(hidden, identity)

    def _attend(self, residual_seq: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        time = residual_seq.shape[1]
        valid = torch.arange(time, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(
            1
        )
        scores = self.attn(residual_seq).squeeze(-1)
        if self.magnitude_prior != 0.0:
            magnitude = residual_seq.norm(dim=-1).masked_fill(~valid, 0.0)
            peak = magnitude.max(dim=1, keepdim=True).values.clamp_min(1e-6)
            scores = scores + self.magnitude_prior * (magnitude / peak)
        fill = -1e4
        weights = torch.softmax(scores.masked_fill(~valid, fill), dim=1)
        weights = weights * valid.to(dtype=weights.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        pooled = (weights.unsqueeze(-1) * residual_seq).sum(dim=1)
        return pooled, weights

    def _collapse_stats(
        self, residual_seq: Tensor, weights: Tensor, lengths: Tensor
    ) -> tuple[Tensor, Tensor]:
        entropy = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(dim=1)
        denom = lengths.clamp(min=2).to(dtype=entropy.dtype).log()
        attn_entropy = entropy / denom.clamp_min(1e-6)
        valid = torch.arange(residual_seq.shape[1], device=lengths.device).unsqueeze(
            0
        ) < lengths.unsqueeze(1)
        normalized = F.normalize(residual_seq, dim=-1, eps=1e-6)
        cosine = torch.matmul(normalized, normalized.transpose(-1, -2))
        pair_mask = valid.unsqueeze(2) & valid.unsqueeze(1)
        diagonal = torch.eye(cosine.shape[-1], dtype=torch.bool, device=cosine.device)
        pair_mask = pair_mask & ~diagonal
        pair_count = pair_mask.sum(dim=(1, 2)).clamp_min(1).to(dtype=cosine.dtype)
        residual_cosine = (cosine.masked_fill(~pair_mask, 0.0).sum(dim=(1, 2))) / pair_count
        return attn_entropy, residual_cosine

    def classify(
        self,
        gallery: Tensor,
        gallery_lengths: Tensor,
        query: Tensor,
        query_lengths: Tensor,
        identity: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        batch, gallery_size, time, features = gallery.shape
        gallery_hidden = self._encode(
            gallery.reshape(batch * gallery_size, time, features),
            gallery_lengths.reshape(batch * gallery_size),
            identity.repeat_interleave(gallery_size, dim=0),
        )[1]
        gallery_mean = gallery_hidden.reshape(batch, gallery_size, -1).mean(dim=1)
        query_seq, _ = self._encode(query, query_lengths, identity)
        residual = query_seq - gallery_mean.unsqueeze(1)
        pooled, weights = self._attend(residual, query_lengths)
        logits = self.head(torch.cat((pooled, pooled.abs()), dim=-1))
        attn_entropy, residual_cosine = self._collapse_stats(
            residual, weights, query_lengths
        )
        return logits, attn_entropy, residual_cosine, gallery_mean

    def forward(
        self,
        gallery: Tensor,
        gallery_lengths: Tensor,
        query: Tensor,
        query_lengths: Tensor,
        identity: Tensor,
    ) -> Tensor:
        return self.classify(
            gallery, gallery_lengths, query, query_lengths, identity
        )[0]


def class_loss(logits: Tensor, labels: Tensor, bce_weight: float) -> Tensor:
    """4-way CE plus optional BCE on P(not normal). Session labels only."""

    classification = F.cross_entropy(logits, labels)
    if bce_weight == 0.0:
        return classification
    impaired = (labels != 0).to(dtype=logits.dtype)
    not_normal = (1.0 - torch.softmax(logits, dim=-1)[:, 0]).clamp(1e-6, 1.0 - 1e-6)
    detection = F.binary_cross_entropy(not_normal, impaired)
    return classification + bce_weight * detection


def batch_pairwise_cosine(vectors: Tensor) -> Tensor:
    """Mean off-diagonal cosine. 1 means every row points the same way."""

    if vectors.shape[0] < 2:
        return vectors.new_zeros(())
    normalized = F.normalize(vectors, dim=-1, eps=1e-6)
    gram = torch.matmul(normalized, normalized.transpose(0, 1))
    count = vectors.shape[0]
    off_diagonal = gram.sum() - gram.diag().sum()
    return off_diagonal / (count * (count - 1))


def vicreg_variance(vectors: Tensor, gamma: float = 1.0) -> Tensor:
    """Hinge that keeps per-dimension batch std above gamma (VICReg)."""

    if vectors.shape[0] < 2:
        return vectors.new_zeros(())
    std = torch.sqrt(vectors.var(dim=0, unbiased=False) + 1e-4)
    return F.relu(gamma - std).mean()


def collapse_penalty(
    attn_entropy: Tensor,
    residual_cosine: Tensor,
    gallery_mean: Tensor,
    cosine_weight: float,
    entropy_floor: float,
    entropy_weight: float,
    gallery_cosine_weight: float,
    gallery_variance_weight: float,
) -> tuple[Tensor, Tensor]:
    """Penalties on leftover alignment, peaked attention, and shared baselines.

    Uses only residual geometry and attention weights. No perturber internals.
    """

    gallery_cosine = batch_pairwise_cosine(gallery_mean)
    penalty = residual_cosine.new_zeros(())
    if cosine_weight != 0.0:
        penalty = penalty + cosine_weight * residual_cosine.mean()
    if entropy_weight != 0.0 and entropy_floor > 0.0:
        penalty = penalty + entropy_weight * F.relu(entropy_floor - attn_entropy).mean()
    if gallery_cosine_weight != 0.0:
        penalty = penalty + gallery_cosine_weight * gallery_cosine
    if gallery_variance_weight != 0.0:
        penalty = penalty + gallery_variance_weight * vicreg_variance(gallery_mean)
    return penalty, gallery_cosine


def identity_from_gallery(
    encoder: typenet.TypeNetEncoder,
    gallery: Tensor,
    gallery_lengths: Tensor,
) -> Tensor:
    """Gallery-only TypeNet mean. Detached so identity never trains."""

    encoder.eval()
    batch, gallery_size, time, features = gallery.shape
    with torch.no_grad():
        embeddings = encoder(
            gallery.reshape(batch * gallery_size, time, features),
            gallery_lengths.reshape(batch * gallery_size),
        )
    return embeddings.reshape(batch, gallery_size, -1).mean(dim=1).detach()


def train_one_epoch_residual(
    encoder: typenet.TypeNetEncoder,
    residual_net: IdentityResidualNet,
    sampler,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_size: int,
    steps_per_epoch: int,
    max_gradient_norm: float,
    bce_weight: float,
    cosine_weight: float = 0.0,
    entropy_floor: float = 0.0,
    entropy_weight: float = 0.0,
    gallery_cosine_weight: float = 0.0,
    gallery_variance_weight: float = 0.0,
) -> tuple[float, float, float, float, float, float]:
    encoder.eval()
    residual_net.train()
    losses: list[float] = []
    accuracies: list[float] = []
    gradient_norms: list[float] = []
    attn_entropies: list[float] = []
    residual_cosines: list[float] = []
    gallery_cosines: list[float] = []
    for _ in range(steps_per_epoch):
        (
            gallery_features,
            gallery_lengths,
            query_features,
            query_lengths,
            labels,
        ) = sampler.sample(batch_size, device)
        if gallery_features.ndim != 4:
            raise AssertionError("Residual training expects gallery [batch, G, T, 5]")
        identity = identity_from_gallery(encoder, gallery_features, gallery_lengths)
        logits, attn_entropy, residual_cosine, gallery_mean = residual_net.classify(
            gallery_features,
            gallery_lengths,
            query_features,
            query_lengths,
            identity,
        )
        penalty, gallery_cosine = collapse_penalty(
            attn_entropy,
            residual_cosine,
            gallery_mean,
            cosine_weight,
            entropy_floor,
            entropy_weight,
            gallery_cosine_weight,
            gallery_variance_weight,
        )
        loss = class_loss(logits, labels, bce_weight) + penalty
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        leaked = [
            name
            for name, parameter in encoder.named_parameters()
            if parameter.grad is not None
        ]
        if leaked:
            raise AssertionError(
                "TypeNet received gradients under residual training: "
                + ", ".join(leaked)
            )
        gradient_norm = nn.utils.clip_grad_norm_(
            residual_net.parameters(), max_norm=max_gradient_norm
        )
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        accuracies.append(
            float((logits.argmax(dim=1) == labels).float().mean().detach().cpu())
        )
        gradient_norms.append(float(gradient_norm.detach().cpu()))
        attn_entropies.append(float(attn_entropy.mean().detach().cpu()))
        residual_cosines.append(float(residual_cosine.mean().detach().cpu()))
        gallery_cosines.append(float(gallery_cosine.detach().cpu()))
    return (
        float(np.mean(losses)),
        float(np.mean(accuracies)),
        float(np.mean(gradient_norms)),
        float(np.mean(attn_entropies)),
        float(np.mean(residual_cosines)),
        float(np.mean(gallery_cosines)),
    )


def load_complete_user_matrices(
    paths: Sequence[Path],
    sequence_length: int,
    eval_users: int,
) -> tuple[np.ndarray, np.ndarray]:
    selected_paths = list(paths[:eval_users] if eval_users > 0 else paths)
    features: list[np.ndarray] = []
    lengths: list[np.ndarray] = []
    for path in selected_paths:
        try:
            user = typenet.load_user_sequences(path, sequence_length)
        except (OSError, ValueError, csv.Error):
            continue
        if len(user.features) < typenet.PAPER_SESSIONS_PER_USER:
            continue
        features.append(user.features[: typenet.PAPER_SESSIONS_PER_USER])
        lengths.append(user.lengths[: typenet.PAPER_SESSIONS_PER_USER])
    if len(features) < 2:
        raise RuntimeError("Evaluation requires at least two complete users")
    return np.stack(features), np.stack(lengths)


def _embed_identity_matrix(
    encoder: typenet.TypeNetEncoder,
    gallery_features: np.ndarray,
    gallery_lengths: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    users, gallery_size, time, feat = gallery_features.shape
    encoder.eval()
    batches: list[np.ndarray] = []
    flat = gallery_features.reshape(-1, time, feat)
    flat_lengths = gallery_lengths.reshape(-1)
    for start in range(0, len(flat), batch_size):
        stop = min(start + batch_size, len(flat))
        with torch.no_grad():
            embedded = encoder(
                torch.from_numpy(flat[start:stop]).to(device),
                torch.from_numpy(flat_lengths[start:stop]).to(device),
            )
        batches.append(embedded.cpu().numpy())
    return (
        np.concatenate(batches)
        .reshape(users, gallery_size, typenet.PAPER_EMBEDDING_DIM)
        .mean(axis=1)
    )


def _residual_logits(
    residual_net: IdentityResidualNet,
    gallery_features: np.ndarray,
    gallery_lengths: np.ndarray,
    query_features: np.ndarray,
    query_lengths: np.ndarray,
    identity: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, float, float]:
    residual_net.eval()
    users, query_count, time, feat = query_features.shape
    gallery_size = gallery_features.shape[1]
    logits_out: list[np.ndarray] = []
    entropies: list[float] = []
    cosines: list[float] = []
    user_batch = max(1, batch_size // max(query_count, 1))
    for start in range(0, users, user_batch):
        stop = min(start + user_batch, users)
        chunk_users = stop - start
        gallery = np.repeat(gallery_features[start:stop], query_count, axis=0)
        gallery_len = np.repeat(gallery_lengths[start:stop], query_count, axis=0)
        ident = np.repeat(identity[start:stop], query_count, axis=0)
        query = query_features[start:stop].reshape(chunk_users * query_count, time, feat)
        query_len = query_lengths[start:stop].reshape(chunk_users * query_count)
        with torch.no_grad():
            logits, attn_entropy, residual_cosine, _gallery_mean = residual_net.classify(
                torch.from_numpy(gallery).to(device),
                torch.from_numpy(gallery_len).to(device),
                torch.from_numpy(query).to(device),
                torch.from_numpy(query_len).to(device),
                torch.from_numpy(ident).to(device),
            )
        logits_out.append(
            logits.cpu()
            .numpy()
            .reshape(chunk_users, query_count, CLASS_COUNT)
        )
        entropies.append(float(attn_entropy.mean().cpu()))
        cosines.append(float(residual_cosine.mean().cpu()))
        if gallery.shape[1] != gallery_size:
            raise AssertionError("Gallery size changed during residual eval")
    return np.concatenate(logits_out, axis=0), float(np.mean(entropies)), float(
        np.mean(cosines)
    )


def _four_way_from_logits(
    logits_by_class: Mapping[str, np.ndarray],
) -> FourWayAccuracy:
    by_class: dict[str, float] = {}
    confusion: dict[str, dict[str, float]] = {}
    matches: list[float] = []
    for class_index, name in enumerate(CLASS_NAMES):
        logits = logits_by_class[name]
        pred = logits.argmax(axis=-1).reshape(-1)
        counts = np.bincount(pred, minlength=CLASS_COUNT).astype(np.float64)
        total = float(counts.sum())
        share = counts / total if total else counts
        confusion[name] = {
            pred_name: float(share[pred_index])
            for pred_index, pred_name in enumerate(CLASS_NAMES)
        }
        recall = float(share[class_index])
        by_class[name] = recall
        matches.append(recall)
    return FourWayAccuracy(
        overall=float(np.mean(matches)),
        by_class=by_class,
        confusion=confusion,
    )


def format_four_way(prefix: str, accuracy: FourWayAccuracy) -> str:
    parts = " ".join(
        f"{name}={accuracy.by_class[name]:.3f}" for name in CLASS_NAMES
    )
    return f"{prefix} acc={accuracy.overall:.3f} ({parts})"


def format_collapse(
    prefix: str, collapse: Mapping[str, tuple[float, float]]
) -> str:
    parts = " ".join(
        f"{name} H={collapse[name][0]:.2f}/cos={collapse[name][1]:.2f}"
        for name in CLASS_NAMES
        if name in collapse
    )
    return f"{prefix} attn {parts}"


def format_confusion(prefix: str, accuracy: FourWayAccuracy) -> str:
    if not accuracy.confusion:
        return f"{prefix} conf (none)"
    rows = []
    for true_name in CLASS_NAMES:
        shares = accuracy.confusion.get(true_name, {})
        cells = " ".join(
            f"{pred_name}={shares.get(pred_name, 0.0):.2f}"
            for pred_name in CLASS_NAMES
        )
        rows.append(f"{true_name}[{cells}]")
    return f"{prefix} conf " + "  ".join(rows)


def evaluate_residual_cohort(
    encoder: typenet.TypeNetEncoder,
    residual_net: IdentityResidualNet,
    paths: Sequence[Path],
    perturber,
    sequence_length: int,
    gallery_size: int,
    eval_users: int,
    eval_batch_size: int,
    device: torch.device,
    seed: int,
) -> tuple[
    np.ndarray,
    dict[str, np.ndarray],
    int,
    FourWayAccuracy,
    dict[str, tuple[float, float]],
]:
    """Same gallery/query protocol as paircls. Labels are session class only.

    Returns clean scores, per-severity scores, user count, and 4-way accuracy.
    Does not import prototype_net.nn (this file is loaded from nn.py as a script).
    """

    feature_matrix, length_matrix = load_complete_user_matrices(
        paths, sequence_length, eval_users
    )
    users = len(feature_matrix)
    gallery_features = feature_matrix[:, :gallery_size]
    gallery_lengths = length_matrix[:, :gallery_size]
    identity = _embed_identity_matrix(
        encoder, gallery_features, gallery_lengths, eval_batch_size, device
    )
    query_features = feature_matrix[
        :,
        typenet.PAPER_QUERY_START : (
            typenet.PAPER_QUERY_START + typenet.PAPER_QUERY_COUNT
        ),
    ]
    query_lengths = length_matrix[
        :,
        typenet.PAPER_QUERY_START : (
            typenet.PAPER_QUERY_START + typenet.PAPER_QUERY_COUNT
        ),
    ]
    logits_by_class: dict[str, np.ndarray] = {}
    collapse: dict[str, tuple[float, float]] = {}
    clean_logits, clean_entropy, clean_cosine = _residual_logits(
        residual_net,
        gallery_features,
        gallery_lengths,
        query_features,
        query_lengths,
        identity,
        eval_batch_size,
        device,
    )
    logits_by_class["normal"] = clean_logits
    collapse["normal"] = (clean_entropy, clean_cosine)
    impaired_by_severity: dict[str, np.ndarray] = {}
    for severity_index, severity_name in enumerate(SEVERITY_NAMES):
        rng = np.random.default_rng(seed)
        perturbed = np.empty_like(query_features)
        for user_index in range(users):
            profile = perturber.sample_profile(severity_index, rng)
            for query_index in range(typenet.PAPER_QUERY_COUNT):
                perturbed[user_index, query_index] = perturber.perturb(
                    query_features[user_index, query_index],
                    int(query_lengths[user_index, query_index]),
                    severity_index,
                    rng,
                    profile=profile,
                )
        logits, entropy, cosine = _residual_logits(
            residual_net,
            gallery_features,
            gallery_lengths,
            perturbed,
            query_lengths,
            identity,
            eval_batch_size,
            device,
        )
        logits_by_class[severity_name] = logits
        collapse[severity_name] = (entropy, cosine)
        impaired_by_severity[severity_name] = 1.0 - _softmax(logits)[..., 0]
    return (
        1.0 - _softmax(clean_logits)[..., 0],
        impaired_by_severity,
        users,
        _four_way_from_logits(logits_by_class),
        collapse,
    )


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)
