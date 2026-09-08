"""Research-only synthetic timing-impairment prototype built on TypeNet

The pretrained TypeNet architecture remains unchanged. Input batch
normalization, the first LSTM, inter-layer dropout, and hidden batch
normalization are frozen in evaluation mode. Only the second LSTM, which emits
the 128-dimensional embedding, is fine-tuned.

Each training triplet contains:

    anchor   = clean session A
    positive = different clean session B from the same user
    negative = severity-conditioned perturbed copy of session B

Run a short real-data check:

    python prototype_net/nn.py --smoke-test
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper_typenet import nn as typenet  # noqa: E402


SEVERITY_NAMES = ("mild", "moderate", "severe")
SEVERITY_COUNT = len(SEVERITY_NAMES)
FROZEN_MODULE_NAMES = (
    "input_batch_norm",
    "first_lstm",
    "dropout",
    "hidden_batch_norm",
)


@dataclass(frozen=True)
class PerturbationConfig:
    severity_factors: tuple[float, float, float] = (0.25, 0.50, 1.00)
    pause_probabilities: tuple[float, float, float] = (0.01, 0.04, 0.10)
    temporal_correlation: float = 0.80
    pause_scale: float = 2.0
    minimum_timing_scale: float = 0.001


@dataclass(frozen=True)
class ImpairmentScores:
    normal: np.ndarray
    impaired_by_severity: Mapping[str, np.ndarray]
    user_count: int

    @property
    def impaired(self) -> np.ndarray:
        return np.concatenate(
            [self.impaired_by_severity[name] for name in SEVERITY_NAMES]
        )


@dataclass(frozen=True)
class DetectionMetrics:
    threshold: float
    false_impairment_rate: float
    missed_impairment_rate: float
    sensitivity: float
    specificity: float
    balanced_accuracy: float
    auroc: float
    diagnostic_eer: float
    diagnostic_eer_threshold: float


@dataclass(frozen=True)
class ProtocolSplits:
    train: list[Path]
    legacy_quarantine: list[Path]
    selection: list[Path]
    calibration: list[Path]
    final_test: list[Path]
    reserve: list[Path]
    manifest: Mapping[str, object]


class StructuredTimingPerturber:
    """Severity-conditioned additive timing perturbations.

    Noise is added to the independent HL and IL timing quantities. PL and RL
    are then reconstructed to preserve the physical timing identities:

        PL[t] = HL[t] + IL[t]
        RL[t] = IL[t] + HL[t + 1]

    Keycodes and zero padding are never changed.
    """

    def __init__(self, config: PerturbationConfig) -> None:
        self.config = config

    @staticmethod
    def _correlated_absolute_noise(
        length: int, correlation: float, rng: np.random.Generator
    ) -> np.ndarray:
        if length <= 0:
            return np.empty(0, dtype=np.float32)
        innovations = rng.standard_normal(length)
        correlated = np.empty(length, dtype=np.float64)
        correlated[0] = innovations[0]
        innovation_scale = math.sqrt(max(0.0, 1.0 - correlation**2))
        for index in range(1, length):
            correlated[index] = (
                correlation * correlated[index - 1]
                + innovation_scale * innovations[index]
            )
        return np.abs(correlated).astype(np.float32)

    @staticmethod
    def _robust_positive_scale(values: np.ndarray, minimum: float) -> float:
        finite_positive = values[np.isfinite(values) & (values > 0)]
        if finite_positive.size == 0:
            return minimum
        return max(float(np.median(finite_positive)), minimum)

    def perturb(
        self,
        features: np.ndarray,
        length: int,
        severity_index: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if severity_index not in range(SEVERITY_COUNT):
            raise ValueError(f"severity_index must be 0..{SEVERITY_COUNT - 1}")
        if features.ndim != 2 or features.shape[1] != typenet.PAPER_FEATURE_COUNT:
            raise ValueError("Expected a [time, 5] TypeNet feature matrix")

        valid_length = min(max(int(length), 1), len(features))
        perturbed = np.asarray(features, dtype=np.float32).copy()
        factor = self.config.severity_factors[severity_index]
        pause_probability = self.config.pause_probabilities[severity_index]

        hold_scale = self._robust_positive_scale(
            perturbed[:valid_length, 0],
            self.config.minimum_timing_scale,
        )
        transition_length = max(valid_length - 1, 0)
        transition_scale = self._robust_positive_scale(
            perturbed[:transition_length, 2],
            self.config.minimum_timing_scale,
        )

        hold_delay = (
            factor
            * hold_scale
            * self._correlated_absolute_noise(
                valid_length, self.config.temporal_correlation, rng
            )
        )
        inter_delay = (
            factor
            * transition_scale
            * self._correlated_absolute_noise(
                transition_length, self.config.temporal_correlation, rng
            )
        )
        if transition_length:
            pause_draws = rng.random(transition_length)
            pause_magnitudes = rng.exponential(
                scale=self.config.pause_scale * transition_scale,
                size=transition_length,
            )
            inter_delay += (
                (pause_draws < pause_probability)
                * factor
                * pause_magnitudes
            ).astype(np.float32)

        hold_latency = np.maximum(
            perturbed[:valid_length, 0] + hold_delay, 0.0
        )
        inter_latency = (
            perturbed[:transition_length, 1] + inter_delay
            if transition_length
            else np.empty(0, dtype=np.float32)
        )

        perturbed[:valid_length, 0] = hold_latency
        if transition_length:
            perturbed[:transition_length, 1] = inter_latency
            perturbed[:transition_length, 2] = (
                hold_latency[:transition_length] + inter_latency
            )
            perturbed[:transition_length, 3] = (
                inter_latency + hold_latency[1:valid_length]
            )

        # The final key has no forward transition; padding remains exactly zero.
        perturbed[valid_length - 1, 1:4] = 0.0
        if valid_length < len(perturbed):
            perturbed[valid_length:] = 0.0
        return perturbed


class ImpairmentTripletSampler:
    """Random same-user clean/clean/synthetic triplets with balanced severity."""

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

    def balanced_severity_indices(self, batch_size: int) -> np.ndarray:
        severity = np.arange(batch_size, dtype=np.int64) % SEVERITY_COUNT
        self.rng.shuffle(severity)
        return severity

    def _sample_clean_pair(
        self,
    ) -> tuple[np.ndarray, int, np.ndarray, int]:
        for _ in range(100):
            user_index = int(self.rng.integers(len(self.store)))
            try:
                user = self.store.get(user_index)
            except (OSError, ValueError, csv.Error):
                continue
            if len(user.features) < 2:
                continue
            anchor_index, positive_index = self.rng.choice(
                len(user.features), size=2, replace=False
            )
            return (
                user.features[anchor_index],
                int(user.lengths[anchor_index]),
                user.features[positive_index],
                int(user.lengths[positive_index]),
            )
        raise RuntimeError("Could not sample a valid same-user clean pair")

    def sample(
        self, batch_size: int, device: torch.device
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        severity_indices = self.balanced_severity_indices(batch_size)
        samples: list[
            tuple[np.ndarray, int, np.ndarray, int, np.ndarray, int]
        ] = []
        for severity_index in severity_indices:
            anchor, anchor_length, positive, positive_length = (
                self._sample_clean_pair()
            )
            negative = self.perturber.perturb(
                positive,
                positive_length,
                int(severity_index),
                self.rng,
            )
            samples.append(
                (
                    anchor,
                    anchor_length,
                    positive,
                    positive_length,
                    negative,
                    positive_length,
                )
            )

        def feature_tensor(position: int) -> Tensor:
            return torch.from_numpy(
                np.stack([sample[position] for sample in samples])
            ).to(device)

        def length_tensor(position: int) -> Tensor:
            return torch.tensor(
                [sample[position] for sample in samples],
                dtype=torch.long,
                device=device,
            )

        return (
            feature_tensor(0),
            length_tensor(1),
            feature_tensor(2),
            length_tensor(3),
            feature_tensor(4),
            length_tensor(5),
            torch.from_numpy(severity_indices).to(device),
        )


def load_pretrained_encoder(path: Path, device: torch.device) -> typenet.TypeNetEncoder:
    if not path.is_file():
        raise FileNotFoundError(f"Pretrained TypeNet weights not found: {path}")
    payload = torch.load(path, map_location=device, weights_only=False)
    state_dict = payload.get("model_state_dict", payload)
    if not isinstance(state_dict, Mapping):
        raise ValueError(f"{path} does not contain a model state dictionary")
    encoder = typenet.TypeNetEncoder().to(device)
    encoder.load_state_dict(state_dict, strict=True)
    return encoder


def freeze_intermediate_representation(encoder: typenet.TypeNetEncoder) -> None:
    for parameter in encoder.parameters():
        parameter.requires_grad_(True)
    for module_name in FROZEN_MODULE_NAMES:
        module = getattr(encoder, module_name)
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        module.eval()


def set_fine_tuning_mode(encoder: typenet.TypeNetEncoder) -> None:
    encoder.train()
    for module_name in FROZEN_MODULE_NAMES:
        getattr(encoder, module_name).eval()
    encoder.second_lstm.train()


def assert_freeze_configuration(encoder: typenet.TypeNetEncoder) -> None:
    trainable_names = [
        name for name, parameter in encoder.named_parameters() if parameter.requires_grad
    ]
    if not trainable_names or not all(
        name.startswith("second_lstm.") for name in trainable_names
    ):
        raise AssertionError(
            "Only second_lstm parameters may be trainable; got "
            + ", ".join(trainable_names)
        )
    for module_name in FROZEN_MODULE_NAMES:
        if getattr(encoder, module_name).training:
            raise AssertionError(f"Frozen module remained in training mode: {module_name}")


def triplet_losses(
    anchor: Tensor, positive: Tensor, negative: Tensor, margin: float
) -> Tensor:
    positive_distance_squared = (anchor - positive).square().sum(dim=1)
    negative_distance_squared = (anchor - negative).square().sum(dim=1)
    return torch.relu(
        positive_distance_squared - negative_distance_squared + margin
    )


def train_one_epoch(
    encoder: typenet.TypeNetEncoder,
    sampler: ImpairmentTripletSampler,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_size: int,
    steps_per_epoch: int,
    margin: float,
    max_gradient_norm: float,
) -> tuple[float, float, float, dict[str, float]]:
    set_fine_tuning_mode(encoder)
    assert_freeze_configuration(encoder)
    losses: list[float] = []
    gradient_norms: list[float] = []
    severity_losses: dict[str, list[float]] = {
        name: [] for name in SEVERITY_NAMES
    }

    for _ in range(steps_per_epoch):
        (
            anchor,
            anchor_lengths,
            positive,
            positive_lengths,
            negative,
            negative_lengths,
            severity_indices,
        ) = sampler.sample(batch_size, device)

        combined = torch.cat((anchor, positive, negative), dim=0)
        combined_lengths = torch.cat(
            (anchor_lengths, positive_lengths, negative_lengths), dim=0
        )
        embeddings = encoder(combined, combined_lengths)
        expected_shape = (
            batch_size * 3,
            typenet.PAPER_EMBEDDING_DIM,
        )
        if embeddings.shape != expected_shape:
            raise AssertionError(
                f"Expected embedding shape {expected_shape}, got "
                f"{tuple(embeddings.shape)}"
            )
        anchor_embedding, positive_embedding, negative_embedding = embeddings.chunk(3)
        per_sample_loss = triplet_losses(
            anchor_embedding,
            positive_embedding,
            negative_embedding,
            margin,
        )
        loss = per_sample_loss.mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        frozen_with_gradients = [
            name
            for name, parameter in encoder.named_parameters()
            if not parameter.requires_grad and parameter.grad is not None
        ]
        if frozen_with_gradients:
            raise AssertionError(
                "Frozen parameters received gradients: "
                + ", ".join(frozen_with_gradients)
            )
        gradient_norm = nn.utils.clip_grad_norm_(
            [parameter for parameter in encoder.parameters() if parameter.requires_grad],
            max_norm=max_gradient_norm,
        )
        optimizer.step()

        losses.append(float(loss.detach().cpu()))
        gradient_norms.append(float(gradient_norm.detach().cpu()))
        for severity_index, severity_name in enumerate(SEVERITY_NAMES):
            mask = severity_indices == severity_index
            severity_losses[severity_name].append(
                float(per_sample_loss[mask].mean().detach().cpu())
            )

    return (
        float(np.mean(losses)),
        float(np.mean(gradient_norms)),
        float(np.max(gradient_norms)),
        {
            name: float(np.mean(values))
            for name, values in severity_losses.items()
        },
    )


@torch.inference_mode()
def embed_feature_matrix(
    encoder: typenet.TypeNetEncoder,
    features: np.ndarray,
    lengths: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    encoder.eval()
    batches: list[np.ndarray] = []
    for start in range(0, len(features), batch_size):
        stop = min(start + batch_size, len(features))
        batch_features = torch.from_numpy(features[start:stop]).to(device)
        batch_lengths = torch.from_numpy(lengths[start:stop]).to(device)
        batches.append(encoder(batch_features, batch_lengths).cpu().numpy())
    return np.concatenate(batches)


def mean_gallery_distances(
    galleries: np.ndarray, queries: np.ndarray
) -> np.ndarray:
    """Return [users, queries] mean Euclidean distances to clean galleries."""

    scores = np.zeros(queries.shape[:2], dtype=np.float32)
    for gallery_index in range(galleries.shape[1]):
        scores += np.linalg.norm(
            galleries[:, gallery_index, None, :] - queries,
            axis=-1,
        )
    return scores / galleries.shape[1]


def evaluate_impairment_scores(
    encoder: typenet.TypeNetEncoder,
    paths: Sequence[Path],
    perturber: StructuredTimingPerturber,
    sequence_length: int,
    gallery_size: int,
    eval_users: int,
    eval_batch_size: int,
    device: torch.device,
    seed: int,
) -> ImpairmentScores:
    selected_paths = list(paths[:eval_users] if eval_users > 0 else paths)
    clean_user_features: list[np.ndarray] = []
    clean_user_lengths: list[np.ndarray] = []
    skipped = 0
    for path in selected_paths:
        try:
            user = typenet.load_user_sequences(path, sequence_length)
        except (OSError, ValueError, csv.Error):
            skipped += 1
            continue
        if len(user.features) < typenet.PAPER_SESSIONS_PER_USER:
            skipped += 1
            continue
        clean_user_features.append(
            user.features[: typenet.PAPER_SESSIONS_PER_USER]
        )
        clean_user_lengths.append(
            user.lengths[: typenet.PAPER_SESSIONS_PER_USER]
        )
    if len(clean_user_features) < 2:
        raise RuntimeError("Evaluation requires at least two complete users")
    if skipped:
        print(f"Skipped {skipped} incomplete/invalid evaluation users.")

    feature_matrix = np.stack(clean_user_features)
    length_matrix = np.stack(clean_user_lengths)
    users = len(feature_matrix)
    clean_embeddings = embed_feature_matrix(
        encoder,
        feature_matrix.reshape(-1, sequence_length, typenet.PAPER_FEATURE_COUNT),
        length_matrix.reshape(-1),
        eval_batch_size,
        device,
    ).reshape(
        users,
        typenet.PAPER_SESSIONS_PER_USER,
        typenet.PAPER_EMBEDDING_DIM,
    )
    galleries = clean_embeddings[:, :gallery_size]
    clean_queries = clean_embeddings[
        :,
        typenet.PAPER_QUERY_START : (
            typenet.PAPER_QUERY_START + typenet.PAPER_QUERY_COUNT
        ),
    ]
    normal_scores = mean_gallery_distances(galleries, clean_queries).reshape(-1)

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
    impaired_by_severity: dict[str, np.ndarray] = {}
    for severity_index, severity_name in enumerate(SEVERITY_NAMES):
        rng = np.random.default_rng(seed)
        perturbed_queries = np.empty_like(query_features)
        for user_index in range(users):
            for query_index in range(typenet.PAPER_QUERY_COUNT):
                perturbed_queries[user_index, query_index] = perturber.perturb(
                    query_features[user_index, query_index],
                    int(query_lengths[user_index, query_index]),
                    severity_index,
                    rng,
                )
        perturbed_embeddings = embed_feature_matrix(
            encoder,
            perturbed_queries.reshape(
                -1, sequence_length, typenet.PAPER_FEATURE_COUNT
            ),
            query_lengths.reshape(-1),
            eval_batch_size,
            device,
        ).reshape(
            users,
            typenet.PAPER_QUERY_COUNT,
            typenet.PAPER_EMBEDDING_DIM,
        )
        impaired_by_severity[severity_name] = mean_gallery_distances(
            galleries, perturbed_embeddings
        ).reshape(-1)

    return ImpairmentScores(
        normal=normal_scores,
        impaired_by_severity=impaired_by_severity,
        user_count=users,
    )


def binary_auroc(normal_scores: np.ndarray, impaired_scores: np.ndarray) -> float:
    """Compute P(impaired score > normal score), averaging tied ranks."""

    normal = np.asarray(normal_scores, dtype=np.float64).reshape(-1)
    impaired = np.asarray(impaired_scores, dtype=np.float64).reshape(-1)
    values = np.concatenate((normal, impaired))
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        average_rank = ((start + 1) + stop) / 2.0
        ranks[order[start:stop]] = average_rank
        start = stop
    impaired_rank_sum = ranks[len(normal) :].sum()
    return float(
        (
            impaired_rank_sum
            - len(impaired) * (len(impaired) + 1) / 2.0
        )
        / (len(impaired) * len(normal))
    )


def detection_metrics(
    normal_scores: np.ndarray,
    impaired_scores: np.ndarray,
    threshold: float,
) -> DetectionMetrics:
    normal = np.asarray(normal_scores).reshape(-1)
    impaired = np.asarray(impaired_scores).reshape(-1)
    false_impairment_rate = float(np.mean(normal > threshold))
    missed_impairment_rate = float(np.mean(impaired <= threshold))
    diagnostic_threshold, diagnostic_eer = typenet.equal_error_threshold(
        normal, impaired
    )
    sensitivity = 1.0 - missed_impairment_rate
    specificity = 1.0 - false_impairment_rate
    return DetectionMetrics(
        threshold=float(threshold),
        false_impairment_rate=false_impairment_rate,
        missed_impairment_rate=missed_impairment_rate,
        sensitivity=sensitivity,
        specificity=specificity,
        balanced_accuracy=(sensitivity + specificity) / 2.0,
        auroc=binary_auroc(normal, impaired),
        diagnostic_eer=diagnostic_eer,
        diagnostic_eer_threshold=diagnostic_threshold,
    )


def evaluate_all_metrics(
    scores: ImpairmentScores, threshold: float
) -> dict[str, DetectionMetrics]:
    metrics = {
        "overall": detection_metrics(scores.normal, scores.impaired, threshold)
    }
    for severity_name in SEVERITY_NAMES:
        metrics[severity_name] = detection_metrics(
            scores.normal,
            scores.impaired_by_severity[severity_name],
            threshold,
        )
    return metrics


def run_deterministic_perturbation_checks(
    perturber: StructuredTimingPerturber,
    sequence_length: int,
) -> None:
    length = min(12, sequence_length)
    clean = np.zeros(
        (sequence_length, typenet.PAPER_FEATURE_COUNT), dtype=np.float32
    )
    clean[:length, 0] = 0.10
    clean[: length - 1, 1] = 0.05
    clean[: length - 1, 2] = 0.15
    clean[: length - 1, 3] = 0.15
    clean[:length, 4] = np.linspace(32, 90, length) / 255.0

    perturbation_magnitudes: list[float] = []
    for severity_index in range(SEVERITY_COUNT):
        perturbed = perturber.perturb(
            clean,
            length,
            severity_index,
            np.random.default_rng(1234),
        )
        np.testing.assert_array_equal(perturbed[:, 4], clean[:, 4])
        np.testing.assert_array_equal(
            perturbed[length:], np.zeros_like(perturbed[length:])
        )
        if np.any(perturbed[:length, 0] < 0):
            raise AssertionError("Perturbation generated negative hold latency")
        np.testing.assert_allclose(
            perturbed[: length - 1, 2],
            perturbed[: length - 1, 0] + perturbed[: length - 1, 1],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            perturbed[: length - 1, 3],
            perturbed[: length - 1, 1] + perturbed[1:length, 0],
            atol=1e-6,
        )
        perturbation_magnitudes.append(
            float(np.linalg.norm(perturbed[:, :4] - clean[:, :4]))
        )
    if not (
        perturbation_magnitudes[0]
        < perturbation_magnitudes[1]
        < perturbation_magnitudes[2]
    ):
        raise AssertionError(
            "Perturbation magnitude did not increase monotonically by severity"
        )


def frozen_parameter_snapshot(
    encoder: typenet.TypeNetEncoder,
) -> dict[str, Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in encoder.named_parameters()
        if not parameter.requires_grad
    }


def assert_frozen_parameters_unchanged(
    encoder: typenet.TypeNetEncoder, before: Mapping[str, Tensor]
) -> None:
    current = dict(encoder.named_parameters())
    changed = [
        name
        for name, original in before.items()
        if not torch.equal(original, current[name].detach().cpu())
    ]
    if changed:
        raise AssertionError("Frozen parameters changed: " + ", ".join(changed))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def identity_fingerprint(paths: Sequence[Path]) -> str:
    participant_ids = [
        typenet.participant_id_from_path(path) for path in paths
    ]
    payload = "\n".join(participant_ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_locked_protocol_splits(
    all_paths: Sequence[Path], args: argparse.Namespace
) -> ProtocolSplits:
    """Create disjoint cohorts while quarantining all historically inspected IDs."""

    selected = list(
        all_paths[: args.max_users] if args.max_users > 0 else all_paths
    )
    legacy_count = (
        args.legacy_validation_users + args.quarantined_test_users
    )
    required = (
        args.train_users
        + legacy_count
        + args.validation_users
        + args.calibration_users
        + args.test_users
    )
    if len(selected) < required:
        raise ValueError(
            f"Locked protocol requires {required:,} users, found {len(selected):,}"
        )

    train_stop = args.train_users
    legacy_stop = train_stop + legacy_count
    train = selected[:train_stop]
    legacy_quarantine = selected[train_stop:legacy_stop]
    candidates = selected[legacy_stop:]
    candidates.sort(
        key=lambda path: hashlib.sha256(
            (
                args.protocol_salt
                + ":"
                + typenet.participant_id_from_path(path)
            ).encode("utf-8")
        ).digest()
    )

    selection_stop = args.validation_users
    calibration_stop = selection_stop + args.calibration_users
    test_stop = calibration_stop + args.test_users
    selection = candidates[:selection_stop]
    calibration = candidates[selection_stop:calibration_stop]
    final_test = candidates[calibration_stop:test_stop]
    reserve = candidates[test_stop:]

    roles = {
        "train": train,
        "legacy_quarantine": legacy_quarantine,
        "selection": selection,
        "calibration": calibration,
        "final_test": final_test,
        "reserve": reserve,
    }
    all_role_ids: list[str] = []
    for paths in roles.values():
        all_role_ids.extend(
            typenet.participant_id_from_path(path) for path in paths
        )
    if len(all_role_ids) != len(set(all_role_ids)):
        raise AssertionError("Protocol cohorts contain duplicate participant IDs")

    manifest_roles = {
        role: {
            "count": len(paths),
            "identity_fingerprint": identity_fingerprint(paths),
            "participant_ids": [
                typenet.participant_id_from_path(path) for path in paths
            ],
        }
        for role, paths in roles.items()
    }
    locked_config = {
        "sequence_length": args.sequence_length,
        "gallery_size": args.gallery_size,
        "margin": args.margin,
        "epochs": args.epochs,
        "steps_per_epoch": args.steps_per_epoch,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "severity_factors": list(args.severity_factors),
        "pause_probabilities": list(args.pause_probabilities),
        "temporal_correlation": args.temporal_correlation,
        "pause_scale": args.pause_scale,
        "seed": args.seed,
    }
    config_json = json.dumps(
        locked_config, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    manifest: dict[str, object] = {
        "protocol_version": "synthetic-impairment-v1-locked",
        "protocol_salt": args.protocol_salt,
        "purpose": (
            "Identity-disjoint synthetic-proxy evaluation; legacy cohorts are "
            "quarantined because prior TypeNet experiments exposed them."
        ),
        "pretrained_weights_sha256": sha256_file(args.pretrained_weights),
        "prototype_source_sha256": sha256_file(Path(__file__)),
        "typenet_source_sha256": sha256_file(Path(typenet.__file__)),
        "locked_config": locked_config,
        "locked_config_sha256": hashlib.sha256(config_json).hexdigest(),
        "roles": manifest_roles,
    }
    return ProtocolSplits(
        train=train,
        legacy_quarantine=legacy_quarantine,
        selection=selection,
        calibration=calibration,
        final_test=final_test,
        reserve=reserve,
        manifest=manifest,
    )


def write_or_verify_protocol_manifest(
    manifest: Mapping[str, object], path: Path | None
) -> None:
    if path is None:
        return
    serialized = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing != serialized:
            raise RuntimeError(
                f"Locked protocol manifest differs from existing file: {path}"
            )
        print(f"Verified locked protocol manifest: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized, encoding="utf-8")
    print(f"Wrote locked protocol manifest: {path}")


def lock_final_test_access(
    manifest: Mapping[str, object], manifest_path: Path | None
) -> None:
    if manifest_path is None:
        return
    lock_path = manifest_path.with_name(
        manifest_path.name + ".final_test_opened"
    )
    if lock_path.exists():
        raise RuntimeError(
            "The one-shot final-test cohort for this protocol has already "
            f"been opened: {lock_path}"
        )
    lock_payload = {
        "protocol_version": manifest["protocol_version"],
        "locked_config_sha256": manifest["locked_config_sha256"],
        "final_test_fingerprint": protocol_fingerprints(manifest)[
            "final_test"
        ],
        "warning": (
            "This final-test cohort is burned. Future model or configuration "
            "changes require a new protocol salt and untouched reserve cohort."
        ),
    }
    with lock_path.open("x", encoding="utf-8") as handle:
        json.dump(lock_payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def protocol_fingerprints(
    manifest: Mapping[str, object],
) -> dict[str, str]:
    roles = manifest["roles"]
    if not isinstance(roles, Mapping):
        raise TypeError("Protocol manifest roles are invalid")
    return {
        str(role): str(details["identity_fingerprint"])
        for role, details in roles.items()
        if isinstance(details, Mapping)
    }


def format_metrics(prefix: str, metrics: DetectionMetrics) -> str:
    return (
        f"{prefix}: post-hoc/oracle EER={metrics.diagnostic_eer * 100:.3f}% "
        f"AUROC={metrics.auroc * 100:.3f}% "
        f"false-impairment={metrics.false_impairment_rate * 100:.3f}% "
        f"missed-impairment={metrics.missed_impairment_rate * 100:.3f}% "
        f"sensitivity={metrics.sensitivity * 100:.3f}% "
        f"specificity={metrics.specificity * 100:.3f}%"
    )


def checkpoint_payload(
    encoder: typenet.TypeNetEncoder,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    sampler: ImpairmentTripletSampler,
    args: argparse.Namespace,
    perturbation_config: PerturbationConfig,
    best_epoch: int,
    selection_eer: float,
    selection_threshold: float,
    training_in_progress: bool,
    protocol_manifest: Mapping[str, object],
    calibration_eer: float | None = None,
    calibration_threshold: float | None = None,
    calibration_metrics: Mapping[str, DetectionMetrics] | None = None,
    test_metrics: Mapping[str, DetectionMetrics] | None = None,
    final_test_accessed: bool = False,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "model_state_dict": encoder.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": scheduler.state_dict(),
        "sampler_rng_state": sampler.rng.bit_generator.state,
        "torch_rng_state": torch.get_rng_state(),
        "best_epoch": best_epoch,
        "selection_eer": selection_eer,
        "selection_threshold": selection_threshold,
        "calibration_eer": calibration_eer,
        "threshold": calibration_threshold,
        "training_in_progress": training_in_progress,
        "final_test_accessed": final_test_accessed,
        "pretrained_weights": str(args.pretrained_weights),
        "protocol_manifest": str(args.protocol_manifest),
        "protocol_fingerprints": protocol_fingerprints(protocol_manifest),
        "locked_config_sha256": protocol_manifest["locked_config_sha256"],
        "perturbation_config": asdict(perturbation_config),
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    if calibration_metrics is not None:
        payload["calibration_metrics"] = {
            name: asdict(metrics)
            for name, metrics in calibration_metrics.items()
        }
    if test_metrics is not None:
        payload["test_metrics"] = {
            name: asdict(metrics) for name, metrics in test_metrics.items()
        }
    return payload


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune TypeNet against research-only synthetic timing "
            "perturbations; this is not a cognitive-impairment diagnostic."
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
        default=(
            REPO_ROOT
            / "paper_typenet"
            / "weights"
            / "typenet_68k_m50_g10_32x512_best_weights.pt"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "prototype_net" / "weights" / "prototype_checkpoint.pt",
    )
    parser.add_argument(
        "--weights-output",
        type=Path,
        default=REPO_ROOT / "prototype_net" / "weights" / "prototype_weights.pt",
    )
    parser.add_argument(
        "--protocol-manifest",
        type=Path,
        default=(
            REPO_ROOT
            / "prototype_net"
            / "weights"
            / "synthetic_impairment_protocol_manifest.json"
        ),
    )
    parser.add_argument("--sequence-length", type=int, default=50)
    parser.add_argument("--gallery-size", type=int, default=10)
    parser.add_argument("--margin", type=float, default=1.5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps-per-epoch", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--lr-decay-factor", type=float, default=0.5)
    parser.add_argument("--lr-decay-patience", type=int, default=2)
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-6)
    parser.add_argument("--max-gradient-norm", type=float, default=1.0)
    parser.add_argument("--validate-every", type=int, default=5)
    parser.add_argument("--minimum-epochs", type=int, default=20)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--train-users", type=int, default=68_000)
    parser.add_argument(
        "--legacy-validation-users",
        type=int,
        default=1_000,
        help="Previously used pretraining-validation identities to quarantine",
    )
    parser.add_argument(
        "--quarantined-test-users",
        type=int,
        default=1_000,
        help="Previously inspected TypeNet-test identities to quarantine",
    )
    parser.add_argument(
        "--validation-users",
        type=int,
        default=1_000,
        help="Fresh model-selection identities",
    )
    parser.add_argument(
        "--calibration-users",
        type=int,
        default=1_000,
        help="Fresh identities used once to fit the global threshold",
    )
    parser.add_argument("--test-users", type=int, default=1_000)
    parser.add_argument("--max-users", type=int, default=0)
    parser.add_argument(
        "--eval-users",
        type=int,
        default=0,
        help="Limit each locked evaluation cohort; zero uses the entire cohort",
    )
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--cache-users", type=int, default=-1)
    parser.add_argument(
        "--severity-factors",
        type=float,
        nargs=3,
        metavar=("MILD", "MODERATE", "SEVERE"),
        default=(0.25, 0.50, 1.00),
    )
    parser.add_argument(
        "--pause-probabilities",
        type=float,
        nargs=3,
        metavar=("MILD", "MODERATE", "SEVERE"),
        default=(0.01, 0.04, 0.10),
    )
    parser.add_argument("--temporal-correlation", type=float, default=0.80)
    parser.add_argument("--pause-scale", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--protocol-salt",
        default="synthetic-impairment-prototype-v1-2026-09-07",
        help="Locked salt for deterministic hash-randomized holdout assignment",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke-test", action="store_true")
    return parser


def apply_smoke_settings(args: argparse.Namespace) -> None:
    args.max_users = 12
    args.train_users = 6
    args.legacy_validation_users = 0
    args.quarantined_test_users = 0
    args.validation_users = 2
    args.calibration_users = 2
    args.test_users = 2
    args.epochs = 1
    args.steps_per_epoch = 1
    args.batch_size = 6
    args.eval_users = 0
    args.eval_batch_size = 30
    args.cache_users = 12
    args.validate_every = 1
    args.minimum_epochs = 0
    args.early_stopping_patience = 0
    args.checkpoint = None
    args.weights_output = None
    args.protocol_manifest = None


def validate_arguments(args: argparse.Namespace) -> None:
    if args.sequence_length != typenet.PAPER_SEQUENCE_LENGTH:
        raise ValueError("The pretrained encoder requires M=50")
    if not 1 <= args.gallery_size <= typenet.PAPER_GALLERY_POOL:
        raise ValueError("--gallery-size must be between 1 and 10")
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
    cohort_counts = (
        args.train_users,
        args.validation_users,
        args.calibration_users,
        args.test_users,
    )
    if any(value < 2 for value in cohort_counts):
        raise ValueError(
            "Train, selection, calibration, and test cohorts need at least two users"
        )
    if args.legacy_validation_users < 0 or args.quarantined_test_users < 0:
        raise ValueError("Quarantined cohort sizes cannot be negative")
    if args.eval_users < 0:
        raise ValueError("--eval-users cannot be negative")
    if not args.protocol_salt:
        raise ValueError("--protocol-salt cannot be empty")
    factors = tuple(args.severity_factors)
    probabilities = tuple(args.pause_probabilities)
    if not (factors[0] < factors[1] < factors[2]):
        raise ValueError("Severity factors must increase mild < moderate < severe")
    if not (
        0.0 <= probabilities[0] <= probabilities[1] <= probabilities[2] <= 1.0
    ):
        raise ValueError("Pause probabilities must increase within [0, 1]")
    if not 0.0 <= args.temporal_correlation < 1.0:
        raise ValueError("--temporal-correlation must be in [0, 1)")
    if args.lr_decay_patience < 1:
        raise ValueError("--lr-decay-patience must be at least one")
    if not 0.0 < args.lr_decay_factor < 1.0:
        raise ValueError("--lr-decay-factor must be between zero and one")
    if args.early_stopping_patience < 0:
        raise ValueError("--early-stopping-patience cannot be negative")


def main() -> None:
    args = build_argument_parser().parse_args()
    if args.smoke_test:
        apply_smoke_settings(args)
    validate_arguments(args)
    typenet.set_reproducible_seed(args.seed)
    device = typenet.resolve_device(args.device)
    print(
        "RESEARCH-ONLY: synthetic timing perturbations are not clinically "
        "validated and cannot diagnose cognitive impairment."
    )
    print(f"Device: {device}")

    perturbation_config = PerturbationConfig(
        severity_factors=tuple(args.severity_factors),
        pause_probabilities=tuple(args.pause_probabilities),
        temporal_correlation=args.temporal_correlation,
        pause_scale=args.pause_scale,
    )
    perturber = StructuredTimingPerturber(perturbation_config)
    run_deterministic_perturbation_checks(perturber, args.sequence_length)

    all_paths = typenet.discover_user_files(args.data_dir)
    protocol = build_locked_protocol_splits(all_paths, args)
    write_or_verify_protocol_manifest(
        protocol.manifest, args.protocol_manifest
    )
    fingerprints = protocol_fingerprints(protocol.manifest)
    print(
        f"Found {len(all_paths):,} users; locked identity-disjoint protocol: "
        f"{len(protocol.train):,} train, "
        f"{len(protocol.legacy_quarantine):,} legacy quarantine, "
        f"{len(protocol.selection):,} selection, "
        f"{len(protocol.calibration):,} calibration, "
        f"{len(protocol.final_test):,} one-shot final test, "
        f"{len(protocol.reserve):,} reserve."
    )
    print(
        "Cohort fingerprints: "
        f"selection={fingerprints['selection'][:12]} "
        f"calibration={fingerprints['calibration'][:12]} "
        f"final-test={fingerprints['final_test'][:12]}"
    )
    if args.checkpoint is not None and args.checkpoint.exists():
        existing_checkpoint = torch.load(
            args.checkpoint, map_location="cpu", weights_only=False
        )
        if existing_checkpoint.get("final_test_accessed", False):
            raise RuntimeError(
                "Refusing to reuse a checkpoint whose one-shot final test has "
                "already been accessed. Use a new locked protocol and output path."
            )

    encoder = load_pretrained_encoder(args.pretrained_weights, device)
    freeze_intermediate_representation(encoder)
    set_fine_tuning_mode(encoder)
    assert_freeze_configuration(encoder)
    trainable_parameters = [
        parameter for parameter in encoder.parameters() if parameter.requires_grad
    ]
    trainable_count = sum(parameter.numel() for parameter in trainable_parameters)
    print(
        f"Loaded pretrained TypeNet; fine-tuning second_lstm only "
        f"({trainable_count:,} trainable parameters)."
    )
    frozen_before = frozen_parameter_snapshot(encoder)

    store = typenet.KeystrokeStore(
        protocol.train, args.sequence_length, args.cache_users
    )
    sampler = ImpairmentTripletSampler(store, perturber, args.seed)
    severity_check = sampler.balanced_severity_indices(args.batch_size)
    severity_counts = np.bincount(severity_check, minlength=SEVERITY_COUNT)
    if int(severity_counts.max() - severity_counts.min()) > 1:
        raise AssertionError("Severity sampling is not balanced")

    optimizer = torch.optim.Adam(trainable_parameters, lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_decay_factor,
        patience=args.lr_decay_patience - 1,
        min_lr=args.minimum_learning_rate,
    )

    best_epoch = 0
    best_validation_eer = math.inf
    best_threshold: float | None = None
    best_state: dict[str, Tensor] | None = None
    checks_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        (
            loss,
            mean_gradient_norm,
            maximum_gradient_norm,
            severity_losses,
        ) = train_one_epoch(
            encoder,
            sampler,
            optimizer,
            device,
            args.batch_size,
            args.steps_per_epoch,
            args.margin,
            args.max_gradient_norm,
        )
        severity_text = " ".join(
            f"{name}={severity_losses[name]:.4f}" for name in SEVERITY_NAMES
        )
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} loss={loss:.6f} "
            f"lr={optimizer.param_groups[0]['lr']:.8f} "
            f"gradient mean/max={mean_gradient_norm:.4f}/{maximum_gradient_norm:.4f} "
            f"[{severity_text}]"
        )

        if epoch % args.validate_every != 0 and epoch != args.epochs:
            continue
        validation_scores = evaluate_impairment_scores(
            encoder,
            protocol.selection,
            perturber,
            args.sequence_length,
            args.gallery_size,
            args.eval_users,
            args.eval_batch_size,
            device,
            args.seed + 10_000,
        )
        validation_threshold, validation_eer = typenet.equal_error_threshold(
            validation_scores.normal, validation_scores.impaired
        )
        validation_metrics = evaluate_all_metrics(
            validation_scores, validation_threshold
        )
        print(
            format_metrics(
                "  model-selection overall",
                validation_metrics["overall"],
            )
        )

        previous_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(validation_eer)
        current_lr = optimizer.param_groups[0]["lr"]
        if current_lr < previous_lr:
            print(f"  learning-rate decay: {previous_lr:.8f} -> {current_lr:.8f}")

        if validation_eer < best_validation_eer:
            best_epoch = epoch
            best_validation_eer = validation_eer
            best_threshold = validation_threshold
            best_state = copy.deepcopy(encoder.state_dict())
            checks_without_improvement = 0
            if args.checkpoint is not None:
                args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    checkpoint_payload(
                        encoder,
                        optimizer,
                        scheduler,
                        sampler,
                        args,
                        perturbation_config,
                        best_epoch,
                        best_validation_eer,
                        best_threshold,
                        training_in_progress=True,
                        protocol_manifest=protocol.manifest,
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
            print(
                "Early stopping after "
                f"{checks_without_improvement} non-improving validation checks."
            )
            break

    if best_state is None or best_threshold is None:
        raise RuntimeError("No validated prototype checkpoint was produced")
    encoder.load_state_dict(best_state)
    assert_frozen_parameters_unchanged(encoder, frozen_before)

    calibration_scores = evaluate_impairment_scores(
        encoder,
        protocol.calibration,
        perturber,
        args.sequence_length,
        args.gallery_size,
        args.eval_users,
        args.eval_batch_size,
        device,
        args.seed + 20_000,
    )
    calibration_threshold, calibration_eer = typenet.equal_error_threshold(
        calibration_scores.normal, calibration_scores.impaired
    )
    calibration_metrics = evaluate_all_metrics(
        calibration_scores, calibration_threshold
    )
    print(
        f"Best model-selection checkpoint: epoch={best_epoch} "
        f"EER={best_validation_eer * 100:.3f}% "
        f"selection-only tau={best_threshold:.6f}"
    )
    print(
        f"Frozen calibration threshold: tau={calibration_threshold:.6f} "
        f"pooled EER={calibration_eer * 100:.3f}%"
    )

    # Persist the selected model and frozen calibration threshold before opening
    # the one-shot final-test cohort.
    if args.checkpoint is not None:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            checkpoint_payload(
                encoder,
                optimizer,
                scheduler,
                sampler,
                args,
                perturbation_config,
                best_epoch,
                best_validation_eer,
                best_threshold,
                training_in_progress=False,
                protocol_manifest=protocol.manifest,
                calibration_eer=calibration_eer,
                calibration_threshold=calibration_threshold,
                calibration_metrics=calibration_metrics,
                final_test_accessed=False,
            ),
            args.checkpoint,
        )
        print("Sealed selected model and threshold before final-test access.")

    lock_final_test_access(protocol.manifest, args.protocol_manifest)
    print(
        "Opening one-shot final-test cohort "
        f"{fingerprints['final_test'][:12]}."
    )
    test_scores = evaluate_impairment_scores(
        encoder,
        protocol.final_test,
        perturber,
        args.sequence_length,
        args.gallery_size,
        args.eval_users,
        args.eval_batch_size,
        device,
        args.seed + 30_000,
    )
    test_metrics = evaluate_all_metrics(
        test_scores, calibration_threshold
    )
    for metric_name in ("overall", *SEVERITY_NAMES):
        print(
            format_metrics(
                f"Final test {metric_name} at frozen calibration tau",
                test_metrics[metric_name],
            )
        )

    if args.checkpoint is not None:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            checkpoint_payload(
                encoder,
                optimizer,
                scheduler,
                sampler,
                args,
                perturbation_config,
                best_epoch,
                best_validation_eer,
                best_threshold,
                training_in_progress=False,
                protocol_manifest=protocol.manifest,
                calibration_eer=calibration_eer,
                calibration_threshold=calibration_threshold,
                calibration_metrics=calibration_metrics,
                test_metrics=test_metrics,
                final_test_accessed=True,
            ),
            args.checkpoint,
        )
        print(f"Saved checkpoint: {args.checkpoint}")
    if args.weights_output is not None:
        args.weights_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(encoder.state_dict(), args.weights_output)
        print(f"Saved weights: {args.weights_output}")

    result = {
        "best_epoch": best_epoch,
        "model_selection_eer": best_validation_eer,
        "calibration_eer": calibration_eer,
        "threshold": calibration_threshold,
        "final_test_fingerprint": fingerprints["final_test"],
        "test": {
            name: asdict(metrics) for name, metrics in test_metrics.items()
        },
    }
    print("RESULT_JSON " + json.dumps(result, sort_keys=True))
    if args.smoke_test:
        print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
