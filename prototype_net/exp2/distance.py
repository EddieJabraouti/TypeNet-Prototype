"""Option 2: pairwise distance regression on TypeNet embeddings.

TypeNet still emits a 128-d vector. The loss sees pairs: one unperturbed
baseline embedding and one query embedding (clean or perturbed). The score
is an affine-calibrated Euclidean distance between those two vectors.

A gallery is G individual baselines, never a centroid. Training applies the
regression loss to every (baseline_i, query) pair. Evaluation averages those
pairwise distances. That is mean-of-distances, not distance-to-mean.

Training targets scale linearly with perturbation level:

    normal=0.0  mild=0.3  moderate=0.6  severe=1.0

Default: freeze the TypeNet frontend (input BN, first LSTM, hidden BN) and
train the second LSTM plus the two affine distance parameters, matching the
original TypeNet fine-tune. --freeze-encoder trains only the affine readout.
--unfreeze-encoder trains both LSTMs; BN running stats stay frozen.

Nearest-target 4-way accuracy is a decode of the scalar (not a class head).
Checkpoint selection is validation MAE. Report 4-way accuracy at that epoch
alongside EER.

    python -m prototype_net.exp2.distance --smoke-test
    python -m prototype_net.exp2.distance --device mps
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper_typenet import nn as typenet  # noqa: E402
from prototype_net.exp2 import model as changehead  # noqa: E402
from prototype_net.perturbation_generator.perturb import (  # noqa: E402
    CLASS_COUNT,
    CLASS_NAMES,
    SEVERITY_NAMES,
    PerturbationConfig,
    StructuredTimingPerturber,
    run_deterministic_perturbation_checks,
)

DISTANCE_TARGETS = (0.0, 0.3, 0.6, 1.0)
FROZEN_FRONTEND_NAMES = (
    "input_batch_norm",
    "first_lstm",
    "dropout",
    "hidden_batch_norm",
)


class AffineEuclideanDistance(nn.Module):
    """Map ||baseline - query|| onto the 0/0.3/0.6/1.0 severity axis."""

    def __init__(self) -> None:
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(()))
        self.bias = nn.Parameter(torch.zeros(()))

    def raw(self, baseline: Tensor, query: Tensor) -> Tensor:
        return (baseline - query).square().sum(dim=-1).clamp_min(1e-12).sqrt()

    def forward(self, baseline: Tensor, query: Tensor) -> Tensor:
        return F.softplus(self.log_scale) * self.raw(baseline, query) + self.bias

    def scale(self) -> float:
        return float(F.softplus(self.log_scale).detach().cpu())


class DistanceRegressor(nn.Module):
    """TypeNet embeddings plus pairwise affine Euclidean distance."""

    def __init__(
        self,
        encoder: typenet.TypeNetEncoder,
        freeze_encoder: bool = False,
        unfreeze_all: bool = False,
    ) -> None:
        super().__init__()
        if freeze_encoder and unfreeze_all:
            raise ValueError("Cannot freeze the encoder and unfreeze all layers")
        self.encoder = encoder
        self.distance = AffineEuclideanDistance()
        self.freeze_encoder = freeze_encoder
        self.unfreeze_all = unfreeze_all
        configure_encoder_training(encoder, freeze_encoder, unfreeze_all)

    def set_runtime_mode(self, training: bool) -> None:
        self.distance.train(training)
        if self.freeze_encoder or not training:
            self.encoder.eval()
            return
        self.encoder.train()
        for module_name in changehead.BATCH_NORM_MODULE_NAMES:
            getattr(self.encoder, module_name).eval()
        if self.unfreeze_all:
            self.encoder.first_lstm.train()
            self.encoder.dropout.train()
            self.encoder.second_lstm.train()
            return
        for module_name in FROZEN_FRONTEND_NAMES:
            getattr(self.encoder, module_name).eval()
        self.encoder.second_lstm.train()

    def embed(self, inputs: Tensor, lengths: Tensor) -> Tensor:
        if self.freeze_encoder:
            with torch.no_grad():
                return self.encoder(inputs, lengths)
        return self.encoder(inputs, lengths)

    def pair_distances(
        self,
        gallery: Tensor,
        gallery_lengths: Tensor,
        query: Tensor,
        query_lengths: Tensor,
    ) -> Tensor:
        """Return [batch, gallery_size] distances, one per baseline/query pair."""

        batch, gallery_size, time, features = gallery.shape
        combined = torch.cat(
            (gallery.reshape(batch * gallery_size, time, features), query), dim=0
        )
        combined_lengths = torch.cat(
            (gallery_lengths.reshape(batch * gallery_size), query_lengths), dim=0
        )
        embeddings = self.embed(combined, combined_lengths)
        baselines = embeddings[: batch * gallery_size].reshape(
            batch, gallery_size, typenet.PAPER_EMBEDDING_DIM
        )
        queries = embeddings[batch * gallery_size :].unsqueeze(1)
        return self.distance(baselines, queries)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]


def configure_encoder_training(
    encoder: typenet.TypeNetEncoder,
    freeze_encoder: bool,
    unfreeze_all: bool,
) -> None:
    if freeze_encoder:
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)
        encoder.eval()
        return
    for parameter in encoder.parameters():
        parameter.requires_grad_(True)
    if unfreeze_all:
        return
    for module_name in FROZEN_FRONTEND_NAMES:
        module = getattr(encoder, module_name)
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        module.eval()


class GalleryQueryDistanceSampler:
    """Same-user clean baselines plus a query labelled by perturbation level."""

    def __init__(
        self,
        store: typenet.KeystrokeStore,
        perturber: StructuredTimingPerturber,
        seed: int,
        gallery_size: int,
    ) -> None:
        if len(store) < 1:
            raise ValueError("At least one training user is required")
        if gallery_size < 1:
            raise ValueError("gallery_size must be at least one")
        self.store = store
        self.perturber = perturber
        self.rng = np.random.default_rng(seed)
        self.gallery_size = gallery_size
        self.sessions_needed = gallery_size + 1

    def balanced_class_indices(self, batch_size: int) -> np.ndarray:
        labels = np.arange(batch_size, dtype=np.int64) % CLASS_COUNT
        self.rng.shuffle(labels)
        return labels

    def _sample_one(
        self, class_index: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        for _ in range(100):
            user_index = int(self.rng.integers(len(self.store)))
            try:
                user = self.store.get(user_index)
            except (OSError, ValueError, csv.Error):
                continue
            if len(user.features) < self.sessions_needed:
                continue
            chosen = self.rng.choice(
                len(user.features), size=self.sessions_needed, replace=False
            )
            gallery_indices = chosen[: self.gallery_size]
            query_index = int(chosen[self.gallery_size])
            query = user.features[query_index]
            query_length = int(user.lengths[query_index])
            if class_index > 0:
                query = self.perturber.perturb(
                    query,
                    query_length,
                    class_index - 1,
                    self.rng,
                )
            return (
                user.features[gallery_indices],
                user.lengths[gallery_indices].astype(np.int64),
                query,
                query_length,
            )
        raise RuntimeError("Could not sample a valid gallery-query pair")

    def sample(
        self, batch_size: int, device: torch.device
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        labels = self.balanced_class_indices(batch_size)
        galleries: list[np.ndarray] = []
        gallery_lengths: list[np.ndarray] = []
        queries: list[np.ndarray] = []
        query_lengths: list[int] = []
        for class_index in labels:
            gallery, lengths, query, query_length = self._sample_one(
                int(class_index)
            )
            galleries.append(gallery)
            gallery_lengths.append(lengths)
            queries.append(query)
            query_lengths.append(query_length)
        return (
            torch.from_numpy(np.stack(galleries)).to(device),
            torch.from_numpy(np.stack(gallery_lengths)).to(device),
            torch.from_numpy(np.stack(queries)).to(device),
            torch.tensor(query_lengths, dtype=torch.long, device=device),
            torch.from_numpy(labels).to(device),
        )


def targets_from_labels(labels: Tensor, distance_targets: Sequence[float]) -> Tensor:
    table = torch.tensor(
        list(distance_targets), dtype=torch.float32, device=labels.device
    )
    return table[labels]


def nearest_target_classes(
    distances: np.ndarray, distance_targets: Sequence[float]
) -> np.ndarray:
    targets = np.asarray(distance_targets, dtype=np.float64)
    return np.abs(
        np.asarray(distances, dtype=np.float64)[..., None] - targets
    ).argmin(axis=-1)


def four_way_from_distances(
    distances_by_class: Mapping[str, np.ndarray],
    distance_targets: Sequence[float],
) -> changehead.FourWayAccuracy:
    logits_by_class = {}
    targets = np.asarray(distance_targets, dtype=np.float64)
    for name, distances in distances_by_class.items():
        logits_by_class[name] = -np.abs(
            np.asarray(distances, dtype=np.float64)[..., None] - targets
        )
    return changehead._four_way_from_logits(logits_by_class)


def pearson_correlation(first: np.ndarray, second: np.ndarray) -> float:
    x = np.asarray(first, dtype=np.float64).reshape(-1)
    y = np.asarray(second, dtype=np.float64).reshape(-1)
    if x.size < 2:
        return 0.0
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt((x * x).sum() * (y * y).sum()))
    if denom <= 0.0:
        return 0.0
    return float((x * y).sum() / denom)


def spearman_correlation(first: np.ndarray, second: np.ndarray) -> float:
    x = np.asarray(first, dtype=np.float64).reshape(-1)
    y = np.asarray(second, dtype=np.float64).reshape(-1)
    x_rank = np.argsort(np.argsort(x, kind="mergesort"), kind="mergesort")
    y_rank = np.argsort(np.argsort(y, kind="mergesort"), kind="mergesort")
    return pearson_correlation(x_rank.astype(np.float64), y_rank.astype(np.float64))


def mean_distance_by_class(
    distances_by_class: Mapping[str, np.ndarray],
) -> dict[str, float]:
    return {
        name: float(np.mean(np.asarray(distances)))
        for name, distances in distances_by_class.items()
    }


def format_mean_distances(prefix: str, means: Mapping[str, float]) -> str:
    parts = " ".join(f"{name}={means[name]:.3f}" for name in CLASS_NAMES)
    return f"{prefix} mean-d ({parts})"


def train_one_epoch(
    model: DistanceRegressor,
    sampler: GalleryQueryDistanceSampler,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_size: int,
    steps_per_epoch: int,
    max_gradient_norm: float,
    distance_targets: Sequence[float],
) -> tuple[float, float, float, float]:
    model.set_runtime_mode(True)
    losses: list[float] = []
    maes: list[float] = []
    accuracies: list[float] = []
    gradient_norms: list[float] = []
    trainable = model.trainable_parameters()
    for _ in range(steps_per_epoch):
        gallery, gallery_lengths, query, query_lengths, labels = sampler.sample(
            batch_size, device
        )
        pair_predicted = model.pair_distances(
            gallery, gallery_lengths, query, query_lengths
        )
        pair_targets = (
            targets_from_labels(labels, distance_targets)
            .unsqueeze(1)
            .expand_as(pair_predicted)
            .contiguous()
        )
        loss = F.smooth_l1_loss(pair_predicted, pair_targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        frozen_with_gradients = [
            name
            for name, parameter in model.encoder.named_parameters()
            if not parameter.requires_grad and parameter.grad is not None
        ]
        if frozen_with_gradients:
            raise AssertionError(
                "Frozen encoder received gradients: "
                + ", ".join(frozen_with_gradients)
            )
        gradient_norm = nn.utils.clip_grad_norm_(
            trainable, max_norm=max_gradient_norm
        )
        optimizer.step()
        predicted = pair_predicted.mean(dim=1).detach().cpu().numpy()
        target_np = pair_targets[:, 0].detach().cpu().numpy()
        losses.append(float(loss.detach().cpu()))
        maes.append(float(np.mean(np.abs(predicted - target_np))))
        pred_class = nearest_target_classes(predicted, distance_targets)
        accuracies.append(
            float(np.mean(pred_class == labels.detach().cpu().numpy()))
        )
        gradient_norms.append(float(gradient_norm.detach().cpu()))
    return (
        float(np.mean(losses)),
        float(np.mean(maes)),
        float(np.mean(accuracies)),
        float(np.mean(gradient_norms)),
    )


@torch.inference_mode()
def _embed_matrix(
    model: DistanceRegressor,
    features: np.ndarray,
    lengths: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.set_runtime_mode(False)
    users, sessions, time, feat = features.shape
    flat = features.reshape(-1, time, feat)
    flat_lengths = lengths.reshape(-1)
    batches: list[np.ndarray] = []
    for start in range(0, len(flat), batch_size):
        stop = min(start + batch_size, len(flat))
        embedded = model.embed(
            torch.from_numpy(flat[start:stop]).to(device),
            torch.from_numpy(flat_lengths[start:stop]).to(device),
        )
        batches.append(embedded.cpu().numpy())
    return np.concatenate(batches).reshape(
        users, sessions, typenet.PAPER_EMBEDDING_DIM
    )


@torch.inference_mode()
def _mean_pairwise_distances(
    model: DistanceRegressor,
    galleries: np.ndarray,
    queries: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Mean affine distance from each query to every individual gallery vector."""

    model.set_runtime_mode(False)
    users, gallery_size, dim = galleries.shape
    query_count = queries.shape[1]
    baseline_flat = np.repeat(galleries[:, None, :, :], query_count, axis=1).reshape(
        -1, dim
    )
    query_flat = np.repeat(queries[:, :, None, :], gallery_size, axis=2).reshape(
        -1, dim
    )
    scores: list[np.ndarray] = []
    for start in range(0, len(query_flat), batch_size):
        stop = min(start + batch_size, len(query_flat))
        predicted = model.distance(
            torch.from_numpy(baseline_flat[start:stop]).to(device),
            torch.from_numpy(query_flat[start:stop]).to(device),
        )
        scores.append(predicted.cpu().numpy())
    return (
        np.concatenate(scores)
        .reshape(users, query_count, gallery_size)
        .mean(axis=2)
    )


@torch.inference_mode()
def evaluate_cohort(
    model: DistanceRegressor,
    paths: Sequence[Path],
    perturber: StructuredTimingPerturber,
    sequence_length: int,
    gallery_size: int,
    eval_users: int,
    eval_batch_size: int,
    device: torch.device,
    seed: int,
    distance_targets: Sequence[float],
) -> tuple[
    changehead.FourWayAccuracy,
    dict[str, typenet.PerUserEERMetrics],
    float,
    float,
    float,
    dict[str, float],
    float,
]:
    feature_matrix, length_matrix = changehead.load_complete_user_matrices(
        paths, sequence_length, eval_users
    )
    users = len(feature_matrix)
    gallery_features = feature_matrix[:, :gallery_size]
    gallery_lengths = length_matrix[:, :gallery_size]
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
    gallery_embeddings = _embed_matrix(
        model, gallery_features, gallery_lengths, eval_batch_size, device
    )
    distances_by_class: dict[str, np.ndarray] = {}
    target_by_class: dict[str, np.ndarray] = {}
    clean_queries = _embed_matrix(
        model, query_features, query_lengths, eval_batch_size, device
    )
    clean_distances = _mean_pairwise_distances(
        model, gallery_embeddings, clean_queries, eval_batch_size, device
    )
    distances_by_class["normal"] = clean_distances
    target_by_class["normal"] = np.full_like(
        clean_distances, distance_targets[0]
    )
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
        query_embeddings = _embed_matrix(
            model, perturbed, query_lengths, eval_batch_size, device
        )
        distances = _mean_pairwise_distances(
            model, gallery_embeddings, query_embeddings, eval_batch_size, device
        )
        distances_by_class[severity_name] = distances
        target_by_class[severity_name] = np.full_like(
            distances, distance_targets[severity_index + 1]
        )
    accuracy = four_way_from_distances(distances_by_class, distance_targets)
    per_user = changehead.evaluate_all_per_user_metrics(
        distances_by_class["normal"],
        {name: distances_by_class[name] for name in SEVERITY_NAMES},
    )
    predicted = np.concatenate(
        [distances_by_class[name] for name in CLASS_NAMES], axis=1
    ).reshape(-1)
    targets = np.concatenate(
        [target_by_class[name] for name in CLASS_NAMES], axis=1
    ).reshape(-1)
    mae = float(np.mean(np.abs(predicted - targets)))
    pearson = pearson_correlation(predicted, targets)
    spearman = spearman_correlation(predicted, targets)
    means = mean_distance_by_class(distances_by_class)
    impaired_flat = np.concatenate(
        [distances_by_class[name] for name in SEVERITY_NAMES], axis=1
    ).reshape(-1)
    threshold, _eer = typenet.equal_error_threshold(
        distances_by_class["normal"].reshape(-1), impaired_flat
    )
    return accuracy, per_user, mae, pearson, spearman, means, float(threshold)


def optimizer_param_groups(
    model: DistanceRegressor, learning_rate: float, encoder_lr_multiplier: float
) -> list[dict[str, object]]:
    distance_parameters = list(model.distance.parameters())
    encoder_parameters = [
        parameter for parameter in model.encoder.parameters() if parameter.requires_grad
    ]
    if not encoder_parameters:
        return [{"params": distance_parameters, "lr": learning_rate}]
    return [
        {
            "params": encoder_parameters,
            "lr": learning_rate * encoder_lr_multiplier,
        },
        {"params": distance_parameters, "lr": learning_rate},
    ]


def encoder_scope(model: DistanceRegressor) -> str:
    if model.freeze_encoder:
        return "affine-only"
    if model.unfreeze_all:
        return "all-lstms+affine"
    return "second_lstm+affine"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Option 2 distance regression: pairwise Euclidean between each "
            "unperturbed baseline vector and the query, targets 0/0.3/0.6/1.0. "
            "Not a cognitive-impairment diagnostic."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "data" / "Keystrokes" / "files",
    )
    parser.add_argument(
        "--pretrained-weights",
        type=Path,
        default=changehead.PRETRAINED_WEIGHTS,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=changehead.EXP2_DIR / "distance_v4_64x512_unseen_checkpoint.pt",
    )
    parser.add_argument(
        "--weights-output",
        type=Path,
        default=changehead.EXP2_DIR / "distance_v4_64x512_unseen_weights.pt",
    )
    parser.add_argument(
        "--reuse-split-manifest",
        type=Path,
        default=changehead.PROTOCOL_MANIFEST,
        help="Locked v4 train/val/test identities. Required except --smoke-test.",
    )
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Train only the affine distance readout on frozen TypeNet embeddings.",
    )
    parser.add_argument(
        "--unfreeze-encoder",
        action="store_true",
        help="Train both TypeNet LSTMs. BN running stats stay frozen.",
    )
    parser.add_argument(
        "--encoder-lr-multiplier",
        type=float,
        default=0.1,
        help="LR multiplier for trainable encoder params.",
    )
    parser.add_argument(
        "--distance-targets",
        type=float,
        nargs=4,
        metavar=("NORMAL", "MILD", "MODERATE", "SEVERE"),
        default=DISTANCE_TARGETS,
        help="Regression targets for the four perturbation levels.",
    )
    parser.add_argument("--sequence-length", type=int, default=typenet.PAPER_SEQUENCE_LENGTH)
    parser.add_argument("--gallery-size", type=int, default=typenet.PAPER_GALLERY_POOL)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps-per-epoch", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.025)
    parser.add_argument("--lr-decay-factor", type=float, default=0.5)
    parser.add_argument("--lr-decay-patience", type=int, default=2)
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-6)
    parser.add_argument("--max-gradient-norm", type=float, default=1.0)
    parser.add_argument("--validate-every", type=int, default=5)
    parser.add_argument("--minimum-epochs", type=int, default=20)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--eval-users", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--cache-users", type=int, default=-1)
    parser.add_argument(
        "--severity-factors",
        type=float,
        nargs=3,
        metavar=("MILD", "MODERATE", "SEVERE"),
        default=(0.25, 0.55, 1.00),
    )
    parser.add_argument(
        "--pause-probabilities",
        type=float,
        nargs=3,
        metavar=("MILD", "MODERATE", "SEVERE"),
        default=(0.004, 0.012, 0.030),
    )
    parser.add_argument("--temporal-correlation", type=float, default=0.65)
    parser.add_argument("--hold-location-ratio", type=float, default=1.20)
    parser.add_argument("--hold-log-variability", type=float, default=0.30)
    parser.add_argument("--press-location-ratio", type=float, default=1.05)
    parser.add_argument("--press-log-variability", type=float, default=0.12)
    parser.add_argument("--profile-log-variability", type=float, default=0.35)
    parser.add_argument("--session-log-variability", type=float, default=0.15)
    parser.add_argument("--speed-matched-probability", type=float, default=0.50)
    parser.add_argument("--word-boundary-pause-multiplier", type=float, default=3.0)
    parser.add_argument("--correction-pause-multiplier", type=float, default=5.0)
    parser.add_argument("--pause-scale", type=float, default=2.0)
    parser.add_argument("--pause-tail-shape", type=float, default=1.8)
    parser.add_argument("--maximum-pause-seconds", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke-test", action="store_true")
    return parser


def apply_smoke_settings(args: argparse.Namespace) -> None:
    changehead.apply_smoke_settings(args)
    args.gallery_size = 2


def validate_arguments(args: argparse.Namespace) -> None:
    if args.sequence_length != typenet.PAPER_SEQUENCE_LENGTH:
        raise ValueError("The pretrained encoder requires M=50")
    positive_values = (
        args.epochs,
        args.steps_per_epoch,
        args.batch_size,
        args.learning_rate,
        args.max_gradient_norm,
        args.validate_every,
        args.eval_batch_size,
    )
    if any(value <= 0 for value in positive_values):
        raise ValueError("Epoch, batch, LR, gradient, and evaluation values must be positive")
    if args.eval_users < 0:
        raise ValueError("--eval-users cannot be negative")
    factors = tuple(args.severity_factors)
    probabilities = tuple(args.pause_probabilities)
    if not (factors[0] < factors[1] < factors[2]):
        raise ValueError("Severity factors must increase mild < moderate < severe")
    if not (
        0.0 <= probabilities[0] <= probabilities[1] <= probabilities[2] <= 1.0
    ):
        raise ValueError("Pause probabilities must increase within [0, 1]")
    if args.lr_decay_patience < 1:
        raise ValueError("--lr-decay-patience must be at least one")
    if not 0.0 < args.lr_decay_factor < 1.0:
        raise ValueError("--lr-decay-factor must be between zero and one")
    if not 0.0 < args.encoder_lr_multiplier <= 1.0:
        raise ValueError("--encoder-lr-multiplier must be in (0, 1]")
    if not 1 <= args.gallery_size <= typenet.PAPER_GALLERY_POOL:
        raise ValueError("--gallery-size must be between 1 and 10")
    targets = tuple(args.distance_targets)
    if len(targets) != CLASS_COUNT:
        raise ValueError("--distance-targets needs four values")
    if not (targets[0] < targets[1] < targets[2] < targets[3]):
        raise ValueError(
            "Distance targets must increase normal < mild < moderate < severe"
        )
    if args.freeze_encoder and args.unfreeze_encoder:
        raise ValueError("Use only one of --freeze-encoder and --unfreeze-encoder")
    if not args.smoke_test and (
        args.reuse_split_manifest is None or not args.reuse_split_manifest.is_file()
    ):
        raise FileNotFoundError(
            "Locked protocol manifest is required unless --smoke-test"
        )


def checkpoint_payload(
    model: DistanceRegressor,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    sampler: GalleryQueryDistanceSampler,
    args: argparse.Namespace,
    perturbation_config: PerturbationConfig,
    best_epoch: int,
    selection_accuracy: changehead.FourWayAccuracy,
    selection_mae: float,
    selection_eer: float,
    selection_threshold: float,
    train_accuracy_at_best: float,
    train_mae_at_best: float,
    protocol_manifest: Mapping[str, object],
    selection_pearson: float | None = None,
    selection_spearman: float | None = None,
    test_accuracy: changehead.FourWayAccuracy | None = None,
    test_mae: float | None = None,
    test_per_user: Mapping[str, typenet.PerUserEERMetrics] | None = None,
    selection_per_user: Mapping[str, typenet.PerUserEERMetrics] | None = None,
    training_in_progress: bool = False,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "model_state_dict": model.encoder.state_dict(),
        "distance_state_dict": model.distance.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": scheduler.state_dict(),
        "sampler_rng_state": sampler.rng.bit_generator.state,
        "torch_rng_state": torch.get_rng_state(),
        "best_epoch": best_epoch,
        "selection_mae": selection_mae,
        "selection_eer": selection_eer,
        "selection_threshold": selection_threshold,
        "train_accuracy_at_best": train_accuracy_at_best,
        "train_mae_at_best": train_mae_at_best,
        "selection_accuracy": changehead.four_way_payload(selection_accuracy),
        "training_in_progress": training_in_progress,
        "freeze_encoder": model.freeze_encoder,
        "unfreeze_all": model.unfreeze_all,
        "distance_targets": list(args.distance_targets),
        "pretrained_weights": str(args.pretrained_weights),
        "protocol_manifest": str(args.reuse_split_manifest),
        "protocol_fingerprints": changehead.protocol_fingerprints(protocol_manifest)
        if protocol_manifest.get("roles")
        else {},
        "perturbation_config": {
            field: getattr(perturbation_config, field)
            for field in perturbation_config.__dataclass_fields__
        },
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    if selection_pearson is not None:
        payload["selection_pearson"] = selection_pearson
    if selection_spearman is not None:
        payload["selection_spearman"] = selection_spearman
    if selection_per_user is not None:
        payload["selection_per_user"] = changehead.per_user_metrics_payload(
            selection_per_user
        )
    if test_accuracy is not None:
        payload["test_accuracy"] = changehead.four_way_payload(test_accuracy)
    if test_mae is not None:
        payload["test_mae"] = test_mae
    if test_per_user is not None:
        payload["test_per_user"] = changehead.per_user_metrics_payload(test_per_user)
    return payload


def main() -> None:
    args = build_argument_parser().parse_args()
    if args.smoke_test:
        apply_smoke_settings(args)
    validate_arguments(args)
    typenet.set_reproducible_seed(args.seed)
    device = typenet.resolve_device(args.device)
    print(f"device={device}")

    perturbation_config = changehead.perturbation_config_from_args(args)
    perturber = StructuredTimingPerturber(perturbation_config)
    run_deterministic_perturbation_checks(perturber, args.sequence_length)
    distance_targets = tuple(args.distance_targets)

    if args.smoke_test:
        protocol = changehead.smoke_protocol(args.data_dir, args.sequence_length)
        print("smoke split")
    else:
        protocol = changehead.protocol_from_existing_manifest(
            args.reuse_split_manifest, args.data_dir
        )
        print(f"reusing split {args.reuse_split_manifest.name}")

    fingerprints = (
        changehead.protocol_fingerprints(protocol.manifest)
        if protocol.manifest.get("roles")
        else {}
    )
    print(
        f"train={len(protocol.train):,}  val={len(protocol.selection):,}  "
        f"test={len(protocol.final_test):,}"
        + (
            f"  held-out={len(protocol.historical_quarantine):,}"
            if protocol.historical_quarantine
            else ""
        )
    )

    encoder = changehead.load_pretrained_encoder(args.pretrained_weights, device)
    model = DistanceRegressor(
        encoder,
        freeze_encoder=args.freeze_encoder,
        unfreeze_all=args.unfreeze_encoder,
    ).to(device)
    param_groups = optimizer_param_groups(
        model, args.learning_rate, args.encoder_lr_multiplier
    )
    trainable_count = sum(
        parameter.numel()
        for group in param_groups
        for parameter in group["params"]
    )
    store = typenet.KeystrokeStore(
        protocol.train, args.sequence_length, args.cache_users
    )
    sampler = GalleryQueryDistanceSampler(
        store, perturber, args.seed, args.gallery_size
    )
    class_check = sampler.balanced_class_indices(args.batch_size)
    class_counts = np.bincount(class_check, minlength=CLASS_COUNT)
    if int(class_counts.max() - class_counts.min()) > 1:
        raise AssertionError("Class sampling is not balanced")
    optimizer = torch.optim.Adam(param_groups)
    print(
        f"distance-reg {encoder_scope(model)}  params={trainable_count:,}  "
        f"targets={','.join(f'{value:g}' for value in distance_targets)}  "
        f"classes={','.join(CLASS_NAMES)}  chance=0.25  "
        f"select=val MAE  loss=SmoothL1 on each (baseline_i, query) pair  "
        f"score=mean pairwise ||g_i-q||  "
        f"{changehead.format_optimizer_lrs(optimizer)}  "
        f"batch={args.batch_size}  steps={args.steps_per_epoch}  "
        f"G={args.gallery_size}"
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_decay_factor,
        patience=args.lr_decay_patience - 1,
        min_lr=args.minimum_learning_rate,
        threshold=0.0,
    )

    best_epoch = 0
    best_validation_mae = math.inf
    best_validation_eer = math.inf
    best_threshold: float | None = None
    best_state: dict[str, object] | None = None
    best_selection_accuracy: changehead.FourWayAccuracy | None = None
    best_selection_per_user: dict[str, typenet.PerUserEERMetrics] | None = None
    best_selection_pearson: float | None = None
    best_selection_spearman: float | None = None
    best_train_accuracy: float | None = None
    best_train_mae: float | None = None
    checks_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        loss, mae, accuracy, grad_norm = train_one_epoch(
            model,
            sampler,
            optimizer,
            device,
            args.batch_size,
            args.steps_per_epoch,
            args.max_gradient_norm,
            distance_targets,
        )
        if epoch % args.validate_every != 0 and epoch != args.epochs:
            continue
        print(
            f"ep {epoch:03d}  loss={loss:.3f}  mae={mae:.3f}  "
            f"acc={accuracy:.3f}  gnorm={grad_norm:.3f}  "
            f"scale={model.distance.scale():.4f}  "
            f"{changehead.format_optimizer_lrs(optimizer)}"
        )
        (
            validation_accuracy,
            validation_per_user,
            validation_mae,
            validation_pearson,
            validation_spearman,
            validation_means,
            validation_threshold,
        ) = evaluate_cohort(
            model,
            protocol.selection,
            perturber,
            args.sequence_length,
            args.gallery_size,
            args.eval_users,
            args.eval_batch_size,
            device,
            args.seed + 10_000,
            distance_targets,
        )
        validation_eer = validation_per_user["overall"].mean
        print(
            f"  {changehead.format_four_way('val nearest-target', validation_accuracy)}"
        )
        print(f"  {changehead.format_confusion('val', validation_accuracy)}")
        print(
            f"  val mae={validation_mae:.3f}  pearson={validation_pearson:.3f}  "
            f"spearman={validation_spearman:.3f}"
        )
        print(f"  {format_mean_distances('val', validation_means)}")
        print(f"  {changehead.format_cohort_per_user('val', validation_per_user)}")

        previous_lrs = [group["lr"] for group in optimizer.param_groups]
        scheduler.step(validation_mae)
        current_lrs = [group["lr"] for group in optimizer.param_groups]
        if current_lrs != previous_lrs:
            before = " ".join(f"{lr:.4g}" for lr in previous_lrs)
            after = " ".join(f"{lr:.4g}" for lr in current_lrs)
            print(f"  lr {before} -> {after}")

        if validation_mae < best_validation_mae:
            best_epoch = epoch
            best_validation_mae = validation_mae
            best_validation_eer = validation_eer
            best_threshold = validation_threshold
            best_state = {
                "encoder": copy.deepcopy(model.encoder.state_dict()),
                "distance": copy.deepcopy(model.distance.state_dict()),
            }
            best_selection_accuracy = validation_accuracy
            best_selection_per_user = validation_per_user
            best_selection_pearson = validation_pearson
            best_selection_spearman = validation_spearman
            best_train_accuracy = accuracy
            best_train_mae = mae
            checks_without_improvement = 0
            if args.checkpoint is not None:
                args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    checkpoint_payload(
                        model,
                        optimizer,
                        scheduler,
                        sampler,
                        args,
                        perturbation_config,
                        best_epoch,
                        validation_accuracy,
                        best_validation_mae,
                        best_validation_eer,
                        validation_threshold,
                        accuracy,
                        mae,
                        protocol.manifest,
                        selection_pearson=validation_pearson,
                        selection_spearman=validation_spearman,
                        selection_per_user=validation_per_user,
                        training_in_progress=True,
                    ),
                    args.checkpoint,
                )
            if args.weights_output is not None:
                args.weights_output.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, args.weights_output)
        else:
            checks_without_improvement += 1

        if (
            args.early_stopping_patience > 0
            and epoch >= args.minimum_epochs
            and checks_without_improvement >= args.early_stopping_patience
        ):
            print(f"early stop after {checks_without_improvement} flat checks")
            break

    if (
        best_state is None
        or best_threshold is None
        or best_selection_accuracy is None
        or best_train_accuracy is None
        or best_train_mae is None
    ):
        raise RuntimeError("No validated distance-regression checkpoint was produced")
    model.encoder.load_state_dict(best_state["encoder"])
    model.distance.load_state_dict(best_state["distance"])
    model.set_runtime_mode(False)

    print(
        f"best ep={best_epoch}  "
        f"{changehead.format_four_way('val nearest-target', best_selection_accuracy)}  "
        f"train acc={best_train_accuracy:.3f}  "
        f"train mae={best_train_mae:.3f}  "
        f"val mae={best_validation_mae:.3f}  "
        f"val EER {best_validation_eer * 100:.2f}%"
    )

    print("test")
    (
        test_accuracy,
        test_per_user,
        test_mae,
        test_pearson,
        test_spearman,
        test_means,
        _test_threshold,
    ) = evaluate_cohort(
        model,
        protocol.final_test,
        perturber,
        args.sequence_length,
        args.gallery_size,
        args.eval_users,
        args.eval_batch_size,
        device,
        args.seed + 30_000,
        distance_targets,
    )
    print(changehead.format_four_way("test nearest-target", test_accuracy))
    print(changehead.format_confusion("test", test_accuracy))
    print(
        f"test mae={test_mae:.3f}  pearson={test_pearson:.3f}  "
        f"spearman={test_spearman:.3f}"
    )
    print(format_mean_distances("test", test_means))
    print(changehead.format_cohort_per_user("test", test_per_user))

    if args.checkpoint is not None:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            checkpoint_payload(
                model,
                optimizer,
                scheduler,
                sampler,
                args,
                perturbation_config,
                best_epoch,
                best_selection_accuracy,
                best_validation_mae,
                best_validation_eer,
                best_threshold,
                best_train_accuracy,
                best_train_mae,
                protocol.manifest,
                selection_pearson=best_selection_pearson,
                selection_spearman=best_selection_spearman,
                test_accuracy=test_accuracy,
                test_mae=test_mae,
                test_per_user=test_per_user,
                selection_per_user=best_selection_per_user,
                training_in_progress=False,
            ),
            args.checkpoint,
        )
        print("sealed checkpoint")
    if args.weights_output is not None:
        args.weights_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "encoder": model.encoder.state_dict(),
                "distance": model.distance.state_dict(),
                "objective": "distance-regression",
            },
            args.weights_output,
        )

    result = {
        "best_epoch": best_epoch,
        "train_acc_at_best": best_train_accuracy,
        "train_mae_at_best": best_train_mae,
        "model_selection_accuracy": changehead.four_way_payload(
            best_selection_accuracy
        ),
        "model_selection_mae": best_validation_mae,
        "model_selection_pearson": best_selection_pearson,
        "model_selection_spearman": best_selection_spearman,
        "test_accuracy": changehead.four_way_payload(test_accuracy),
        "test_mae": test_mae,
        "test_pearson": test_pearson,
        "test_spearman": test_spearman,
        "model_selection_per_user_eer": best_validation_eer,
        "model_selection_per_user": (
            changehead.per_user_metrics_payload(best_selection_per_user)
            if best_selection_per_user is not None
            else None
        ),
        "final_test_fingerprint": fingerprints.get("final_test"),
        "test_per_user": changehead.per_user_metrics_payload(test_per_user),
        "freeze_encoder": model.freeze_encoder,
        "unfreeze_all": model.unfreeze_all,
        "distance_targets": list(distance_targets),
    }
    print("RESULT_JSON " + json.dumps(result, sort_keys=True))
    if args.smoke_test:
        print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
