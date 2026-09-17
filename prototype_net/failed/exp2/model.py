"""Option 1: 4-way change-head probe of TypeNet sequence layers.

TypeNet has no separate projection after the second LSTM: the 128-d last
hidden state *is* the biometric embedding. This experiment keeps that vector
and replaces gallery-distance verification with an MLP class head of 4 logits
(normal/baseline, mild, moderate, severe). There is no gallery, no pair head,
and no residual comparison.

Default: freeze the pretrained encoder and train only the class head. That
answers whether the current sequence representation already separates
perturbation levels. Pass --unfreeze-encoder to also train the LSTMs.

The head is a residual MLP: LayerNorm, a 256-d expansion, then seven
pre-norm Linear-ReLU residual blocks (eight hidden layers, dropout 0.2).
Skip connections keep that depth trainable so a near-chance result is not
an underfit probe.

This is a diagnostic, not a change-from-baseline detector. If 4-way accuracy
is far above chance (0.25) on a single session, the perturber leaves an
absolute signature. If it stays near chance, a gallery/comparison system
(Option 2 or 3) is required.

Reuses the locked v4 unseen 80/20 split. Val/test 4-way accuracy is scored
on query sessions 10-14, matching residual/paircls. Report 4-way accuracy at
the selected checkpoint alongside EER.

    python -m prototype_net.exp2.model --smoke-test
    python -m prototype_net.exp2.model --unfreeze-encoder
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
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
from prototype_net.perturbation_generator.perturb import (  # noqa: E402
    CLASS_COUNT,
    CLASS_NAMES,
    SEVERITY_NAMES,
    PerturbationConfig,
    StructuredTimingPerturber,
    run_deterministic_perturbation_checks,
)

PROTOCOL_MANIFEST = (
    REPO_ROOT
    / "prototype_net"
    / "weights"
    / "protocol"
    / "synthetic_impairment_protocol_v4_64x512_unseen_triplet_manifest.json"
)
PRETRAINED_WEIGHTS = (
    REPO_ROOT
    / "paper_typenet"
    / "weights"
    / "typenet_68k_m50_g10_64x512_best_weights.pt"
)
EXP2_DIR = Path(__file__).resolve().parent
BATCH_NORM_MODULE_NAMES = ("input_batch_norm", "hidden_batch_norm")
HEAD_HIDDEN_DIM = 256
HEAD_HIDDEN_LAYERS = 8
HEAD_DROPOUT = 0.2


class ResidualMLPBlock(nn.Module):
    """Pre-norm Linear-ReLU residual so stacked hidden layers stay trainable."""

    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs + self.dropout(F.relu(self.linear(self.norm(inputs))))


def build_class_head(
    input_dim: int,
    hidden_dim: int,
    hidden_layers: int,
    dropout: float,
) -> nn.Sequential:
    """Deep residual MLP on the TypeNet vector."""

    if hidden_layers < 1:
        raise ValueError("Class head needs at least one hidden layer")
    if hidden_dim < CLASS_COUNT:
        raise ValueError("Class head hidden size is smaller than the number of classes")
    blocks: list[nn.Module] = [
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
    ]
    for _ in range(hidden_layers - 1):
        blocks.append(ResidualMLPBlock(hidden_dim, dropout))
    blocks.append(nn.Linear(hidden_dim, CLASS_COUNT))
    return nn.Sequential(*blocks)


@dataclass(frozen=True)
class FourWayAccuracy:
    overall: float
    by_class: Mapping[str, float]
    confusion: Mapping[str, Mapping[str, float]] | None = None


@dataclass(frozen=True)
class ProtocolSplits:
    train: list[Path]
    selection: list[Path]
    final_test: list[Path]
    historical_quarantine: list[Path]
    reserve: list[Path]
    manifest: Mapping[str, object]


class ChangeHeadNet(nn.Module):
    """Frozen-or-trainable TypeNet encoder plus a deep residual MLP class head."""

    def __init__(
        self,
        encoder: typenet.TypeNetEncoder,
        freeze_encoder: bool = True,
        head_hidden_dim: int = HEAD_HIDDEN_DIM,
        head_hidden_layers: int = HEAD_HIDDEN_LAYERS,
        head_dropout: float = HEAD_DROPOUT,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = build_class_head(
            typenet.PAPER_EMBEDDING_DIM,
            head_hidden_dim,
            head_hidden_layers,
            head_dropout,
        )
        self.freeze_encoder = freeze_encoder
        self.head_hidden_dim = head_hidden_dim
        self.head_hidden_layers = head_hidden_layers
        self.head_dropout = head_dropout
        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(False)

    def set_runtime_mode(self, training: bool) -> None:
        self.head.train(training)
        if self.freeze_encoder or not training:
            self.encoder.eval()
            return
        self.encoder.train()
        for module_name in BATCH_NORM_MODULE_NAMES:
            getattr(self.encoder, module_name).eval()

    def forward(self, inputs: Tensor, lengths: Tensor) -> Tensor:
        if self.freeze_encoder:
            with torch.no_grad():
                embedding = self.encoder(inputs, lengths)
        else:
            embedding = self.encoder(inputs, lengths)
        return self.head(embedding)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]


class SessionClassSampler:
    """Single-session 4-way samples. Class 0 is clean; 1-3 are perturbed."""

    def __init__(
        self,
        store: typenet.KeystrokeStore,
        perturber: StructuredTimingPerturber,
        seed: int,
    ) -> None:
        if len(store) < 1:
            raise ValueError("At least one training user is required")
        self.store = store
        self.perturber = perturber
        self.rng = np.random.default_rng(seed)

    def balanced_class_indices(self, batch_size: int) -> np.ndarray:
        labels = np.arange(batch_size, dtype=np.int64) % CLASS_COUNT
        self.rng.shuffle(labels)
        return labels

    def _sample_one(self, class_index: int) -> tuple[np.ndarray, int]:
        for _ in range(100):
            user_index = int(self.rng.integers(len(self.store)))
            try:
                user = self.store.get(user_index)
            except (OSError, ValueError, csv.Error):
                continue
            if len(user.features) < 1:
                continue
            session_index = int(self.rng.integers(len(user.features)))
            features = user.features[session_index]
            length = int(user.lengths[session_index])
            if class_index > 0:
                features = self.perturber.perturb(
                    features,
                    length,
                    class_index - 1,
                    self.rng,
                )
            return features, length
        raise RuntimeError("Could not sample a valid labelled session")

    def sample(
        self, batch_size: int, device: torch.device
    ) -> tuple[Tensor, Tensor, Tensor]:
        labels = self.balanced_class_indices(batch_size)
        features = []
        lengths = []
        for class_index in labels:
            session, length = self._sample_one(int(class_index))
            features.append(session)
            lengths.append(length)
        return (
            torch.from_numpy(np.stack(features)).to(device),
            torch.tensor(lengths, dtype=torch.long, device=device),
            torch.from_numpy(labels).to(device),
        )


def load_pretrained_encoder(
    path: Path, device: torch.device
) -> typenet.TypeNetEncoder:
    if not path.is_file():
        raise FileNotFoundError(f"Pretrained TypeNet weights not found: {path}")
    payload = torch.load(path, map_location=device, weights_only=False)
    state_dict = payload.get("model_state_dict", payload)
    if not isinstance(state_dict, Mapping):
        raise ValueError(f"{path} does not contain a model state dictionary")
    encoder = typenet.TypeNetEncoder().to(device)
    encoder.load_state_dict(state_dict, strict=True)
    return encoder


def train_one_epoch(
    model: ChangeHeadNet,
    sampler: SessionClassSampler,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_size: int,
    steps_per_epoch: int,
    max_gradient_norm: float,
) -> tuple[float, float, float]:
    model.set_runtime_mode(True)
    losses: list[float] = []
    accuracies: list[float] = []
    gradient_norms: list[float] = []
    trainable = model.trainable_parameters()
    for _ in range(steps_per_epoch):
        features, lengths, labels = sampler.sample(batch_size, device)
        logits = model(features, lengths)
        loss = F.cross_entropy(logits, labels)
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
        losses.append(float(loss.detach().cpu()))
        accuracies.append(
            float((logits.argmax(dim=1) == labels).float().mean().detach().cpu())
        )
        gradient_norms.append(float(gradient_norm.detach().cpu()))
    return float(np.mean(losses)), float(np.mean(accuracies)), float(
        np.mean(gradient_norms)
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


def format_per_user(prefix: str, metrics: typenet.PerUserEERMetrics) -> str:
    return f"{prefix} {metrics.mean * 100:.2f}%±{metrics.standard_deviation * 100:.1f}"


def format_cohort_per_user(
    prefix: str, per_user_metrics: Mapping[str, typenet.PerUserEERMetrics]
) -> str:
    parts = [
        format_per_user(name, per_user_metrics[name])
        for name in ("overall", *SEVERITY_NAMES)
    ]
    return f"{prefix}  " + "  ".join(parts)


def per_user_eer_metrics(
    normal_by_user: np.ndarray, impaired_by_user: np.ndarray
) -> typenet.PerUserEERMetrics:
    normal = np.asarray(normal_by_user)
    impaired = np.asarray(impaired_by_user)
    if normal.ndim != 2 or impaired.ndim != 2:
        raise ValueError("Per-user EER requires [users, queries] score matrices")
    if normal.shape[0] != impaired.shape[0]:
        raise ValueError("Clean and impaired per-user scores must cover the same users")
    return typenet.paper_protocol_per_user_eer(
        normal.reshape(-1),
        impaired.reshape(-1),
        user_count=int(normal.shape[0]),
    )


def evaluate_all_per_user_metrics(
    normal: np.ndarray, impaired_by_severity: Mapping[str, np.ndarray]
) -> dict[str, typenet.PerUserEERMetrics]:
    impaired = np.concatenate(
        [impaired_by_severity[name] for name in SEVERITY_NAMES], axis=1
    )
    metrics = {"overall": per_user_eer_metrics(normal, impaired)}
    for severity_name in SEVERITY_NAMES:
        metrics[severity_name] = per_user_eer_metrics(
            normal, impaired_by_severity[severity_name]
        )
    return metrics


def per_user_metrics_payload(
    metrics: Mapping[str, typenet.PerUserEERMetrics],
) -> dict[str, dict[str, float]]:
    return {
        name: {"mean": value.mean, "standard_deviation": value.standard_deviation}
        for name, value in metrics.items()
    }


def four_way_payload(accuracy: FourWayAccuracy) -> dict[str, object]:
    payload: dict[str, object] = {
        "overall": accuracy.overall,
        "by_class": dict(accuracy.by_class),
    }
    if accuracy.confusion:
        payload["confusion"] = {
            true_name: dict(shares) for true_name, shares in accuracy.confusion.items()
        }
    return payload


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


@torch.inference_mode()
def _class_logits(
    model: ChangeHeadNet,
    features: np.ndarray,
    lengths: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.set_runtime_mode(False)
    users, query_count, time, feat = features.shape
    flat = features.reshape(-1, time, feat)
    flat_lengths = lengths.reshape(-1)
    batches: list[np.ndarray] = []
    for start in range(0, len(flat), batch_size):
        stop = min(start + batch_size, len(flat))
        logits = model(
            torch.from_numpy(flat[start:stop]).to(device),
            torch.from_numpy(flat_lengths[start:stop]).to(device),
        )
        batches.append(logits.cpu().numpy())
    return np.concatenate(batches).reshape(users, query_count, CLASS_COUNT)


@torch.inference_mode()
def evaluate_cohort(
    model: ChangeHeadNet,
    paths: Sequence[Path],
    perturber: StructuredTimingPerturber,
    sequence_length: int,
    eval_users: int,
    eval_batch_size: int,
    device: torch.device,
    seed: int,
) -> tuple[FourWayAccuracy, dict[str, typenet.PerUserEERMetrics], float]:
    feature_matrix, length_matrix = load_complete_user_matrices(
        paths, sequence_length, eval_users
    )
    users = len(feature_matrix)
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
    clean_logits = _class_logits(
        model, query_features, query_lengths, eval_batch_size, device
    )
    logits_by_class["normal"] = clean_logits
    normal_scores = 1.0 - _softmax(clean_logits)[..., 0]
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
        logits = _class_logits(
            model, perturbed, query_lengths, eval_batch_size, device
        )
        logits_by_class[severity_name] = logits
        impaired_by_severity[severity_name] = 1.0 - _softmax(logits)[..., 0]
    accuracy = _four_way_from_logits(logits_by_class)
    per_user = evaluate_all_per_user_metrics(normal_scores, impaired_by_severity)
    impaired_flat = np.concatenate(
        [impaired_by_severity[name] for name in SEVERITY_NAMES], axis=1
    ).reshape(-1)
    threshold, _eer = typenet.equal_error_threshold(
        normal_scores.reshape(-1), impaired_flat
    )
    return accuracy, per_user, float(threshold)


def paths_from_participant_ids(
    data_dir: Path, participant_ids: Sequence[str]
) -> list[Path]:
    return [data_dir / f"{participant_id}_keystrokes.txt" for participant_id in participant_ids]


def protocol_from_existing_manifest(
    manifest_path: Path, data_dir: Path
) -> ProtocolSplits:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    roles = manifest.get("roles")
    if not isinstance(roles, Mapping):
        raise TypeError(f"Invalid roles in {manifest_path}")

    def role_paths(role_name: str) -> list[Path]:
        role = roles.get(role_name, {})
        if not isinstance(role, Mapping):
            return []
        participant_ids = role.get("participant_ids", [])
        if not isinstance(participant_ids, list):
            raise TypeError(f"Invalid {role_name} participant_ids in {manifest_path}")
        return paths_from_participant_ids(
            data_dir, [str(participant_id) for participant_id in participant_ids]
        )

    return ProtocolSplits(
        train=role_paths("train"),
        selection=role_paths("selection"),
        final_test=role_paths("final_test"),
        historical_quarantine=role_paths("historical_quarantine"),
        reserve=role_paths("reserve"),
        manifest=manifest,
    )


def protocol_fingerprints(manifest: Mapping[str, object]) -> dict[str, str]:
    roles = manifest["roles"]
    if not isinstance(roles, Mapping):
        raise TypeError("Protocol manifest roles are invalid")
    return {
        str(role): str(details["identity_fingerprint"])
        for role, details in roles.items()
        if isinstance(details, Mapping)
    }


def perturbation_config_from_args(args: argparse.Namespace) -> PerturbationConfig:
    return PerturbationConfig(
        severity_factors=tuple(args.severity_factors),
        pause_probabilities=tuple(args.pause_probabilities),
        temporal_correlation=args.temporal_correlation,
        hold_location_ratio=args.hold_location_ratio,
        hold_log_variability=args.hold_log_variability,
        press_location_ratio=args.press_location_ratio,
        press_log_variability=args.press_log_variability,
        profile_log_variability=args.profile_log_variability,
        session_log_variability=args.session_log_variability,
        speed_matched_probability=args.speed_matched_probability,
        word_boundary_pause_multiplier=args.word_boundary_pause_multiplier,
        correction_pause_multiplier=args.correction_pause_multiplier,
        pause_scale=args.pause_scale,
        pause_tail_shape=args.pause_tail_shape,
        maximum_pause_seconds=args.maximum_pause_seconds,
    )


def format_optimizer_lrs(optimizer: torch.optim.Optimizer) -> str:
    return " ".join(
        f"lr{index}={group['lr']:.4g}"
        for index, group in enumerate(optimizer.param_groups)
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Option 1 change-head: 4-way CE on TypeNet sequence features. "
            "Not a cognitive-impairment diagnostic."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "data" / "Keystrokes" / "files",
    )
    parser.add_argument("--pretrained-weights", type=Path, default=PRETRAINED_WEIGHTS)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=EXP2_DIR / "changehead_v4_64x512_unseen_checkpoint.pt",
    )
    parser.add_argument(
        "--weights-output",
        type=Path,
        default=EXP2_DIR / "changehead_v4_64x512_unseen_weights.pt",
    )
    parser.add_argument(
        "--reuse-split-manifest",
        type=Path,
        default=PROTOCOL_MANIFEST,
        help="Locked v4 train/val/test identities. Required except --smoke-test.",
    )
    parser.add_argument(
        "--unfreeze-encoder",
        action="store_true",
        help="Train TypeNet LSTMs as well as the class head. BN running stats stay frozen.",
    )
    parser.add_argument(
        "--encoder-lr-multiplier",
        type=float,
        default=0.1,
        help="LR multiplier for encoder params under --unfreeze-encoder.",
    )
    parser.add_argument(
        "--head-hidden-dim",
        type=int,
        default=HEAD_HIDDEN_DIM,
        help="Width of each MLP hidden layer on the TypeNet vector.",
    )
    parser.add_argument(
        "--head-hidden-layers",
        type=int,
        default=HEAD_HIDDEN_LAYERS,
        help="Hidden layers in the residual MLP head. Default 8; first expands, the rest are residual blocks.",
    )
    parser.add_argument(
        "--head-dropout",
        type=float,
        default=HEAD_DROPOUT,
        help="Dropout after each hidden layer. 0 disables it.",
    )
    parser.add_argument("--sequence-length", type=int, default=typenet.PAPER_SEQUENCE_LENGTH)
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
    args.epochs = 1
    args.steps_per_epoch = 1
    args.batch_size = 8
    args.eval_users = 2
    args.eval_batch_size = 30
    args.cache_users = 12
    args.validate_every = 1
    args.minimum_epochs = 0
    args.early_stopping_patience = 0
    args.checkpoint = None
    args.weights_output = None
    args.reuse_split_manifest = None


def smoke_protocol(data_dir: Path, sequence_length: int) -> ProtocolSplits:
    paths = typenet.discover_user_files(data_dir)[:24]
    complete: list[Path] = []
    for path in paths:
        try:
            user = typenet.load_user_sequences(path, sequence_length)
        except (OSError, ValueError, csv.Error):
            continue
        if len(user.features) >= typenet.PAPER_SESSIONS_PER_USER:
            complete.append(path)
        if len(complete) >= 12:
            break
    if len(complete) < 8:
        raise RuntimeError("Smoke test needs at least eight complete users")
    return ProtocolSplits(
        train=complete[:6],
        selection=complete[6:8],
        final_test=complete[8:10] if len(complete) >= 10 else complete[6:8],
        historical_quarantine=[],
        reserve=[],
        manifest={"roles": {}, "locked_config_sha256": "smoke"},
    )


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
    if args.head_hidden_dim < 32:
        raise ValueError("--head-hidden-dim must be at least 32")
    if args.head_hidden_layers < 1:
        raise ValueError("--head-hidden-layers must be at least one")
    if not 0.0 <= args.head_dropout < 1.0:
        raise ValueError("--head-dropout must be in [0, 1)")
    if not args.smoke_test and (
        args.reuse_split_manifest is None or not args.reuse_split_manifest.is_file()
    ):
        raise FileNotFoundError(
            "Locked protocol manifest is required unless --smoke-test"
        )


def optimizer_param_groups(
    model: ChangeHeadNet, learning_rate: float, encoder_lr_multiplier: float
) -> list[dict[str, object]]:
    head_parameters = list(model.head.parameters())
    if model.freeze_encoder:
        return [{"params": head_parameters, "lr": learning_rate}]
    encoder_parameters = [
        parameter for parameter in model.encoder.parameters() if parameter.requires_grad
    ]
    if not encoder_parameters:
        raise AssertionError("Unfrozen encoder has no trainable parameters")
    return [
        {"params": encoder_parameters, "lr": learning_rate * encoder_lr_multiplier},
        {"params": head_parameters, "lr": learning_rate},
    ]


def checkpoint_payload(
    model: ChangeHeadNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    sampler: SessionClassSampler,
    args: argparse.Namespace,
    perturbation_config: PerturbationConfig,
    best_epoch: int,
    selection_accuracy: FourWayAccuracy,
    selection_eer: float,
    selection_threshold: float,
    train_accuracy_at_best: float,
    protocol_manifest: Mapping[str, object],
    test_accuracy: FourWayAccuracy | None = None,
    test_per_user: Mapping[str, typenet.PerUserEERMetrics] | None = None,
    selection_per_user: Mapping[str, typenet.PerUserEERMetrics] | None = None,
    training_in_progress: bool = False,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "model_state_dict": model.encoder.state_dict(),
        "head_state_dict": model.head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": scheduler.state_dict(),
        "sampler_rng_state": sampler.rng.bit_generator.state,
        "torch_rng_state": torch.get_rng_state(),
        "best_epoch": best_epoch,
        "selection_eer": selection_eer,
        "selection_threshold": selection_threshold,
        "train_accuracy_at_best": train_accuracy_at_best,
        "selection_accuracy": four_way_payload(selection_accuracy),
        "training_in_progress": training_in_progress,
        "freeze_encoder": model.freeze_encoder,
        "pretrained_weights": str(args.pretrained_weights),
        "protocol_manifest": str(args.reuse_split_manifest),
        "protocol_fingerprints": protocol_fingerprints(protocol_manifest)
        if protocol_manifest.get("roles")
        else {},
        "perturbation_config": asdict(perturbation_config),
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    if selection_per_user is not None:
        payload["selection_per_user"] = per_user_metrics_payload(selection_per_user)
    if test_accuracy is not None:
        payload["test_accuracy"] = four_way_payload(test_accuracy)
    if test_per_user is not None:
        payload["test_per_user"] = per_user_metrics_payload(test_per_user)
    return payload


def main() -> None:
    args = build_argument_parser().parse_args()
    if args.smoke_test:
        apply_smoke_settings(args)
    validate_arguments(args)
    typenet.set_reproducible_seed(args.seed)
    device = typenet.resolve_device(args.device)
    print(f"device={device}")

    perturbation_config = perturbation_config_from_args(args)
    perturber = StructuredTimingPerturber(perturbation_config)
    run_deterministic_perturbation_checks(perturber, args.sequence_length)

    if args.smoke_test:
        protocol = smoke_protocol(args.data_dir, args.sequence_length)
        print("smoke split")
    else:
        protocol = protocol_from_existing_manifest(
            args.reuse_split_manifest, args.data_dir
        )
        print(f"reusing split {args.reuse_split_manifest.name}")

    fingerprints = (
        protocol_fingerprints(protocol.manifest) if protocol.manifest.get("roles") else {}
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

    encoder = load_pretrained_encoder(args.pretrained_weights, device)
    model = ChangeHeadNet(
        encoder,
        freeze_encoder=not args.unfreeze_encoder,
        head_hidden_dim=args.head_hidden_dim,
        head_hidden_layers=args.head_hidden_layers,
        head_dropout=args.head_dropout,
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
    sampler = SessionClassSampler(store, perturber, args.seed)
    class_check = sampler.balanced_class_indices(args.batch_size)
    class_counts = np.bincount(class_check, minlength=CLASS_COUNT)
    if int(class_counts.max() - class_counts.min()) > 1:
        raise AssertionError("Class sampling is not balanced")
    optimizer = torch.optim.Adam(param_groups)
    head_count = sum(parameter.numel() for parameter in model.head.parameters())
    scope = "encoder+head" if args.unfreeze_encoder else "mlp-head"
    hidden = "-".join(
        [str(typenet.PAPER_EMBEDDING_DIM)]
        + [str(args.head_hidden_dim)] * args.head_hidden_layers
        + [str(CLASS_COUNT)]
    )
    print(
        f"change-head {scope}  params={trainable_count:,}  "
        f"head={head_count:,} ({hidden} drop={args.head_dropout:g})  "
        f"classes={','.join(CLASS_NAMES)}  chance=0.25  "
        f"select=val 4-way acc  loss=CE  "
        f"{format_optimizer_lrs(optimizer)}  "
        f"batch={args.batch_size}  steps={args.steps_per_epoch}"
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.lr_decay_factor,
        patience=args.lr_decay_patience - 1,
        min_lr=args.minimum_learning_rate,
        threshold=0.0,
    )

    best_epoch = 0
    best_validation_acc = -math.inf
    best_validation_eer = math.inf
    best_threshold: float | None = None
    best_state: dict[str, object] | None = None
    best_selection_accuracy: FourWayAccuracy | None = None
    best_selection_per_user: dict[str, typenet.PerUserEERMetrics] | None = None
    best_train_accuracy: float | None = None
    checks_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        loss, accuracy, grad_norm = train_one_epoch(
            model,
            sampler,
            optimizer,
            device,
            args.batch_size,
            args.steps_per_epoch,
            args.max_gradient_norm,
        )
        if epoch % args.validate_every != 0 and epoch != args.epochs:
            continue
        print(
            f"ep {epoch:03d}  loss={loss:.3f}  acc={accuracy:.3f}  "
            f"gnorm={grad_norm:.3f}  {format_optimizer_lrs(optimizer)}"
        )
        validation_accuracy, validation_per_user, validation_threshold = evaluate_cohort(
            model,
            protocol.selection,
            perturber,
            args.sequence_length,
            args.eval_users,
            args.eval_batch_size,
            device,
            args.seed + 10_000,
        )
        validation_eer = validation_per_user["overall"].mean
        print(f"  {format_four_way('val', validation_accuracy)}")
        print(f"  {format_confusion('val', validation_accuracy)}")
        print(f"  {format_cohort_per_user('val', validation_per_user)}")

        previous_lrs = [group["lr"] for group in optimizer.param_groups]
        scheduler.step(validation_accuracy.overall)
        current_lrs = [group["lr"] for group in optimizer.param_groups]
        if current_lrs != previous_lrs:
            before = " ".join(f"{lr:.4g}" for lr in previous_lrs)
            after = " ".join(f"{lr:.4g}" for lr in current_lrs)
            print(f"  lr {before} -> {after}")

        if validation_accuracy.overall > best_validation_acc:
            best_epoch = epoch
            best_validation_acc = validation_accuracy.overall
            best_validation_eer = validation_eer
            best_threshold = validation_threshold
            best_state = {
                "encoder": copy.deepcopy(model.encoder.state_dict()),
                "head": copy.deepcopy(model.head.state_dict()),
            }
            best_selection_accuracy = validation_accuracy
            best_selection_per_user = validation_per_user
            best_train_accuracy = accuracy
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
                        best_validation_eer,
                        validation_threshold,
                        accuracy,
                        protocol.manifest,
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
    ):
        raise RuntimeError("No validated change-head checkpoint was produced")
    model.encoder.load_state_dict(best_state["encoder"])
    model.head.load_state_dict(best_state["head"])
    model.set_runtime_mode(False)

    print(
        f"best ep={best_epoch}  "
        f"{format_four_way('val', best_selection_accuracy)}  "
        f"train acc={best_train_accuracy:.3f}  "
        f"val EER {best_validation_eer * 100:.2f}%"
    )

    print("test")
    test_accuracy, test_per_user, _test_threshold = evaluate_cohort(
        model,
        protocol.final_test,
        perturber,
        args.sequence_length,
        args.eval_users,
        args.eval_batch_size,
        device,
        args.seed + 30_000,
    )
    print(format_four_way("test", test_accuracy))
    print(format_confusion("test", test_accuracy))
    print(format_cohort_per_user("test", test_per_user))

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
                best_validation_eer,
                best_threshold,
                best_train_accuracy,
                protocol.manifest,
                test_accuracy=test_accuracy,
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
                "head": model.head.state_dict(),
                "objective": "changehead",
            },
            args.weights_output,
        )

    result = {
        "best_epoch": best_epoch,
        "train_acc_at_best": best_train_accuracy,
        "model_selection_accuracy": four_way_payload(best_selection_accuracy),
        "test_accuracy": four_way_payload(test_accuracy),
        "model_selection_per_user_eer": best_validation_eer,
        "model_selection_per_user": (
            per_user_metrics_payload(best_selection_per_user)
            if best_selection_per_user is not None
            else None
        ),
        "final_test_fingerprint": fingerprints.get("final_test"),
        "test_per_user": per_user_metrics_payload(test_per_user),
        "freeze_encoder": model.freeze_encoder,
    }
    print("RESULT_JSON " + json.dumps(result, sort_keys=True))
    if args.smoke_test:
        print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
