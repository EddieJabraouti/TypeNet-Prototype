"""Research evidence-calibrated timing-impairment prototype built on TypeNet

Default fine-tune freezes the TypeNet frontend (input BN, first LSTM,
dropout, hidden BN) and updates only the second LSTM. `--unfreeze-all`
trains every encoder parameter from the same pretrained weights; batch-norm
affine scales move, but running mean/var stay in eval() on the 68k stats.

Default protocol: identities used to train those 68k TypeNet weights are
excluded. The remaining unseen pool (~86k) is hash-split once (80% develop /
20% test; 10% of develop is validation). Later methods (paircls, LoRA, FFT)
start from a fresh copy of the same pretrained TypeNet and reuse that split so
comparisons are identity-matched. Val/test are scored, not discarded.

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
import pathlib
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn


REPO_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = REPO_ROOT / "prototype_net" / "weights"
PROTOCOL_DIR = WEIGHTS_DIR / "protocol"
TRIPLET_DIR = WEIGHTS_DIR / "triplet"
PAIRCLS_DIR = WEIGHTS_DIR / "paircls"
PAIRCLS_SIDE_DIR = WEIGHTS_DIR / "paircls_side"
ABORTED_DIR = WEIGHTS_DIR / "aborted"
INVALID_EXPERIMENTS_DIR = WEIGHTS_DIR / "invalid experiments"


def find_weights_file(name: str) -> Path | None:
    if not name:
        return None
    matches = sorted(
        path
        for path in WEIGHTS_DIR.rglob(name)
        if path.is_file() and INVALID_EXPERIMENTS_DIR not in path.parents
    )
    return matches[0] if matches else None


if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper_typenet import nn as typenet  # noqa: E402


SEVERITY_NAMES = ("mild", "moderate", "severe")
SEVERITY_COUNT = len(SEVERITY_NAMES)
CLASS_NAMES = ("normal",) + SEVERITY_NAMES
CLASS_COUNT = len(CLASS_NAMES)
PAIR_HEAD_HIDDEN_DIM = 256
PAIR_HEAD_DROPOUT = 0.2
SIDE_HIDDEN_DIM = 64
SIDE_OUTPUT_DIM = 128
FROZEN_MODULE_NAMES = (
    "input_batch_norm",
    "first_lstm",
    "dropout",
    "hidden_batch_norm",
)
BATCH_NORM_MODULE_NAMES = ("input_batch_norm", "hidden_batch_norm")
FRONTEND_MODULE_NAMES = (
    "input_batch_norm",
    "first_lstm",
    "hidden_batch_norm",
)


@dataclass(frozen=True)
class PerturbationConfig:
    """Priors constrained by open clinical keystroke evidence.

    The severe hold-time location ratio (1.20) lies between the participant-
    level ratios observed in Tappy (1.07) and neuroQWERTY (1.28). Press-to-press
    location changes remain small because neither dataset showed a material
    mean transition-time shift. Cognitive effects are represented primarily by
    sparse pauses and shorter production bursts, consistent with the open
    AD/MCI writing-process data (Zenodo 5942517), rather than by slowing every
    transition.
    """

    severity_factors: tuple[float, float, float] = (0.25, 0.55, 1.00)
    pause_probabilities: tuple[float, float, float] = (0.004, 0.012, 0.030)
    temporal_correlation: float = 0.65
    hold_location_ratio: float = 1.20
    hold_log_variability: float = 0.30
    press_location_ratio: float = 1.05
    press_log_variability: float = 0.12
    profile_log_variability: float = 0.35
    session_log_variability: float = 0.15
    speed_matched_probability: float = 0.50
    word_boundary_pause_multiplier: float = 3.0
    correction_pause_multiplier: float = 5.0
    pause_scale: float = 2.0
    pause_tail_shape: float = 1.8
    maximum_pause_seconds: float = 3.0
    minimum_timing_scale: float = 0.001


@dataclass(frozen=True)
class SyntheticImpairmentProfile:
    """Participant-level latent motor/cognitive burden."""

    severity_index: int
    motor_burden: float
    cognitive_burden: float
    speed_matched: bool


@dataclass(frozen=True)
class ImpairmentScores:
    """Per-query impairment scores kept as [users, queries] for per-user EER.

    Pair classification uses P(not normal). The triplet path uses mean
    Euclidean gallery distance. Higher means more impaired either way.
    """

    normal: np.ndarray
    impaired_by_severity: Mapping[str, np.ndarray]
    user_count: int

    @property
    def impaired(self) -> np.ndarray:
        return np.concatenate(
            [self.impaired_by_severity[name] for name in SEVERITY_NAMES],
            axis=1,
        )

    @property
    def normal_flat(self) -> np.ndarray:
        return np.asarray(self.normal).reshape(-1)

    @property
    def impaired_flat(self) -> np.ndarray:
        return np.asarray(self.impaired).reshape(-1)


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
    historical_quarantine: list[Path]
    selection: list[Path]
    calibration: list[Path]
    final_test: list[Path]
    reserve: list[Path]
    manifest: Mapping[str, object]


class StructuredTimingPerturber:
    """Evidence-calibrated, severity-conditioned timing perturbations.

    Hold latency (HL) and press-to-press latency (PL) are perturbed on the log
    scale. Inter-key (IL) and release (RL) latencies are reconstructed to
    preserve the physical timing identities:

        IL[t] = PL[t] - HL[t]
        RL[t] = IL[t] + HL[t + 1]

    Motor variation is temporally correlated and bidirectional. Cognitive
    pauses are sparse, heavy-tailed, and more likely after word boundaries or
    correction keys. Half of participant profiles are speed-matched so the
    model cannot depend only on a global slowdown. Keycodes and zero padding
    are never changed.
    """

    def __init__(self, config: PerturbationConfig) -> None:
        self.config = config

    @staticmethod
    def _correlated_standard_noise(
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
        correlated -= np.median(correlated)
        return correlated.astype(np.float32)

    @staticmethod
    def _robust_positive_scale(values: np.ndarray, minimum: float) -> float:
        finite_positive = values[np.isfinite(values) & (values > 0)]
        if finite_positive.size == 0:
            return minimum
        return max(float(np.median(finite_positive)), minimum)

    @staticmethod
    def _bounded_lognormal_multiplier(
        log_variability: float,
        rng: np.random.Generator,
        lower: float,
        upper: float,
    ) -> float:
        value = math.exp(log_variability * float(rng.standard_normal()))
        return float(np.clip(value, lower, upper))

    @staticmethod
    def _log_logistic(
        size: int, shape: float, rng: np.random.Generator
    ) -> np.ndarray:
        if size <= 0:
            return np.empty(0, dtype=np.float32)
        uniform = np.clip(
            rng.random(size), np.finfo(np.float64).eps, 1.0 - 1e-12
        )
        return np.power(uniform / (1.0 - uniform), 1.0 / shape).astype(
            np.float32
        )

    def sample_profile(
        self, severity_index: int, rng: np.random.Generator
    ) -> SyntheticImpairmentProfile:
        if severity_index not in range(SEVERITY_COUNT):
            raise ValueError(f"severity_index must be 0..{SEVERITY_COUNT - 1}")
        severity = self.config.severity_factors[severity_index]
        heterogeneity = self._bounded_lognormal_multiplier(
            self.config.profile_log_variability,
            rng,
            lower=0.35,
            upper=1.75,
        )
        motor_fraction = float(rng.beta(2.0, 2.0))
        motor_weight = 0.4 + 1.2 * motor_fraction
        cognitive_weight = 1.6 - 1.2 * motor_fraction
        return SyntheticImpairmentProfile(
            severity_index=severity_index,
            motor_burden=severity * heterogeneity * motor_weight,
            cognitive_burden=severity * heterogeneity * cognitive_weight,
            speed_matched=bool(
                rng.random() < self.config.speed_matched_probability
            ),
        )

    def _timing_multipliers(
        self,
        length: int,
        burden: float,
        location_ratio: float,
        log_variability: float,
        speed_matched: bool,
        rng: np.random.Generator,
    ) -> np.ndarray:
        noise = self._correlated_standard_noise(
            length, self.config.temporal_correlation, rng
        )
        location = (
            0.0 if speed_matched else burden * math.log(location_ratio)
        )
        log_multipliers = location + burden * log_variability * noise
        return np.clip(np.exp(log_multipliers), 0.35, 3.0).astype(np.float32)

    @staticmethod
    def _keycodes(features: np.ndarray, valid_length: int) -> np.ndarray:
        return np.rint(features[:valid_length, 4] * 255.0).astype(np.int64)

    def _pause_probabilities(
        self,
        features: np.ndarray,
        valid_length: int,
        severity_index: int,
        relative_cognitive_burden: float,
    ) -> np.ndarray:
        transition_length = max(valid_length - 1, 0)
        probabilities = np.full(
            transition_length,
            self.config.pause_probabilities[severity_index]
            * relative_cognitive_burden,
            dtype=np.float64,
        )
        if transition_length == 0:
            return probabilities
        preceding_key = self._keycodes(features, valid_length)[
            :transition_length
        ]
        probabilities[np.isin(preceding_key, (13, 32))] *= (
            self.config.word_boundary_pause_multiplier
        )
        probabilities[np.isin(preceding_key, (8, 46))] *= (
            self.config.correction_pause_multiplier
        )
        return np.clip(probabilities, 0.0, 0.35)

    def perturb(
        self,
        features: np.ndarray,
        length: int,
        severity_index: int,
        rng: np.random.Generator,
        profile: SyntheticImpairmentProfile | None = None,
    ) -> np.ndarray:
        if severity_index not in range(SEVERITY_COUNT):
            raise ValueError(f"severity_index must be 0..{SEVERITY_COUNT - 1}")
        if features.ndim != 2 or features.shape[1] != typenet.PAPER_FEATURE_COUNT:
            raise ValueError("Expected a [time, 5] TypeNet feature matrix")
        if profile is None:
            profile = self.sample_profile(severity_index, rng)
        elif profile.severity_index != severity_index:
            raise ValueError("Profile severity does not match severity_index")

        valid_length = min(max(int(length), 1), len(features))
        perturbed = np.asarray(features, dtype=np.float32).copy()
        transition_length = max(valid_length - 1, 0)
        transition_scale = self._robust_positive_scale(
            perturbed[:transition_length, 2],
            self.config.minimum_timing_scale,
        )
        session_multiplier = self._bounded_lognormal_multiplier(
            self.config.session_log_variability,
            rng,
            lower=0.50,
            upper=1.50,
        )
        motor_burden = profile.motor_burden * session_multiplier
        cognitive_burden = profile.cognitive_burden * session_multiplier

        hold_latency = np.maximum(
            perturbed[:valid_length, 0]
            * self._timing_multipliers(
                valid_length,
                motor_burden,
                self.config.hold_location_ratio,
                self.config.hold_log_variability,
                profile.speed_matched,
                rng,
            ),
            0.0,
        )
        press_latency = np.maximum(
            perturbed[:transition_length, 2]
            * self._timing_multipliers(
                transition_length,
                motor_burden,
                self.config.press_location_ratio,
                self.config.press_log_variability,
                profile.speed_matched,
                rng,
            ),
            0.0,
        )

        if transition_length:
            severity = self.config.severity_factors[severity_index]
            relative_cognitive_burden = cognitive_burden / severity
            pause_probabilities = self._pause_probabilities(
                perturbed,
                valid_length,
                severity_index,
                relative_cognitive_burden,
            )
            pause_mask = rng.random(transition_length) < pause_probabilities
            pause_median = (
                self.config.pause_scale
                * transition_scale
                * max(cognitive_burden, 0.10)
            )
            pause_delays = (
                pause_median
                * self._log_logistic(
                    transition_length, self.config.pause_tail_shape, rng
                )
            )
            pause_delays = np.minimum(
                pause_delays, self.config.maximum_pause_seconds
            )
            press_latency += (pause_mask * pause_delays).astype(np.float32)

        perturbed[:valid_length, 0] = hold_latency
        if transition_length:
            inter_latency = press_latency - hold_latency[:transition_length]
            perturbed[:transition_length, 1] = inter_latency
            perturbed[:transition_length, 2] = press_latency
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


def gallery_query_pair_features(gallery_mean: Tensor, query: Tensor) -> Tensor:
    if gallery_mean.shape != query.shape:
        raise ValueError(
            "Gallery mean and query embeddings must share shape, got "
            f"{tuple(gallery_mean.shape)} and {tuple(query.shape)}"
        )
    return torch.cat(
        (
            gallery_mean,
            query,
            (gallery_mean - query).abs(),
            gallery_mean * query,
        ),
        dim=-1,
    )


class GalleryQueryPairHead(nn.Module):
    """4-way classifier over a mean-gallery / query embedding pair."""

    def __init__(
        self,
        embedding_dim: int = typenet.PAPER_EMBEDDING_DIM,
        hidden_dim: int = PAIR_HEAD_HIDDEN_DIM,
        dropout: float = PAIR_HEAD_DROPOUT,
        side_dim: int = 0,
    ) -> None:
        super().__init__()
        self.side_dim = side_dim
        pair_dim = embedding_dim * 4 + side_dim * 4
        self.net = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, CLASS_COUNT),
        )

    def forward(
        self,
        gallery_mean: Tensor,
        query: Tensor,
        side_gallery_mean: Tensor | None = None,
        side_query: Tensor | None = None,
    ) -> Tensor:
        features = gallery_query_pair_features(gallery_mean, query)
        if self.side_dim > 0:
            if side_gallery_mean is None or side_query is None:
                raise ValueError("Side embeddings are required for this pair head")
            features = torch.cat(
                (
                    features,
                    gallery_query_pair_features(side_gallery_mean, side_query),
                ),
                dim=-1,
            )
        return self.net(features)


class RawTimingSideEncoder(nn.Module):
    """Trainable residual encoder on raw 5-d timings; last layer starts at zero."""

    def __init__(
        self,
        hidden_size: int = SIDE_HIDDEN_DIM,
        output_size: int = SIDE_OUTPUT_DIM,
    ) -> None:
        super().__init__()
        self.lstm = typenet.KerasStyleLSTM(
            typenet.PAPER_FEATURE_COUNT,
            hidden_size,
            recurrent_dropout=0.2,
        )
        self.proj = nn.Linear(hidden_size, output_size)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, inputs: Tensor, lengths: Tensor) -> Tensor:
        _, hidden = self.lstm(inputs, lengths)
        return self.proj(hidden)


def pair_not_normal_probability(logits: Tensor) -> Tensor:
    """Binary impairment score: 1 - P(normal)."""

    return 1.0 - torch.softmax(logits, dim=-1)[:, 0]


def paircls_loss(
    logits: Tensor, labels: Tensor, bce_weight: float
) -> Tensor:
    """4-way CE plus optional BCE on P(not normal)."""

    classification = nn.functional.cross_entropy(logits, labels)
    if bce_weight == 0.0:
        return classification
    impaired = (labels != 0).to(dtype=logits.dtype)
    detection = nn.functional.binary_cross_entropy(
        pair_not_normal_probability(logits).clamp(1e-6, 1.0 - 1e-6),
        impaired,
    )
    return classification + bce_weight * detection


class GalleryQueryClassSampler:
    """Same-user gallery/query pairs with balanced 4-way class labels."""

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


def load_checkpoint_payload(
    path: Path, map_location: str | torch.device = "cpu"
) -> dict[str, object]:
    """Load a checkpoint, including ones pickled with Windows path objects."""

    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except Exception:
        original_windows_path = getattr(pathlib, "WindowsPath", None)
        pathlib.WindowsPath = pathlib.PosixPath
        try:
            payload = torch.load(path, map_location=map_location, weights_only=False)
        finally:
            if original_windows_path is not None:
                pathlib.WindowsPath = original_windows_path
    if not isinstance(payload, dict):
        raise TypeError(f"{path} does not contain a checkpoint dictionary")
    return payload


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


def build_pair_head(
    device: torch.device, side_dim: int = 0
) -> GalleryQueryPairHead:
    return GalleryQueryPairHead(side_dim=side_dim).to(device)


def build_side_encoder(device: torch.device) -> RawTimingSideEncoder:
    return RawTimingSideEncoder().to(device)


def load_pair_head(
    payload: Mapping[str, object], device: torch.device
) -> GalleryQueryPairHead | None:
    state = payload.get("pair_head_state_dict")
    if not isinstance(state, Mapping):
        return None
    weight = state.get("net.0.weight")
    side_dim = 0
    if isinstance(weight, Tensor) and weight.shape[1] > typenet.PAPER_EMBEDDING_DIM * 4:
        side_dim = SIDE_OUTPUT_DIM
    pair_head = build_pair_head(device, side_dim=side_dim)
    pair_head.load_state_dict(state, strict=True)
    return pair_head


def load_side_encoder(
    payload: Mapping[str, object], device: torch.device
) -> RawTimingSideEncoder | None:
    state = payload.get("side_encoder_state_dict")
    if not isinstance(state, Mapping):
        return None
    side_encoder = build_side_encoder(device)
    side_encoder.load_state_dict(state, strict=True)
    return side_encoder


def encoder_is_fully_unfrozen(encoder: typenet.TypeNetEncoder) -> bool:
    return all(parameter.requires_grad for parameter in encoder.parameters())


def configure_encoder_training(
    encoder: typenet.TypeNetEncoder, unfreeze_all: bool
) -> None:
    for parameter in encoder.parameters():
        parameter.requires_grad_(True)
    if unfreeze_all:
        return
    freeze_intermediate_representation(encoder)


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
    for module_name in BATCH_NORM_MODULE_NAMES:
        getattr(encoder, module_name).eval()
    if encoder_is_fully_unfrozen(encoder):
        encoder.first_lstm.train()
        encoder.dropout.train()
        encoder.second_lstm.train()
        return
    for module_name in FROZEN_MODULE_NAMES:
        getattr(encoder, module_name).eval()
    encoder.second_lstm.train()


def assert_freeze_configuration(encoder: typenet.TypeNetEncoder) -> None:
    trainable_names = [
        name for name, parameter in encoder.named_parameters() if parameter.requires_grad
    ]
    if encoder_is_fully_unfrozen(encoder):
        for module_name in BATCH_NORM_MODULE_NAMES:
            if getattr(encoder, module_name).training:
                raise AssertionError(
                    f"Batch-norm running stats must stay in eval(): {module_name}"
                )
        if not encoder.first_lstm.training or not encoder.second_lstm.training:
            raise AssertionError("Both LSTMs must be in train() for full fine-tune")
        if not encoder.dropout.training:
            raise AssertionError("Dropout must be in train() for full fine-tune")
        return
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


def optimizer_param_groups(
    encoder: typenet.TypeNetEncoder,
    pair_head: GalleryQueryPairHead | None,
    learning_rate: float,
    unfreeze_all: bool,
    frontend_lr_multiplier: float,
    extra_parameters: Sequence[nn.Parameter] | None = None,
) -> list[dict[str, object]]:
    extra = list(extra_parameters or ())
    if not unfreeze_all:
        parameters = [
            parameter for parameter in encoder.parameters() if parameter.requires_grad
        ]
        if pair_head is not None:
            parameters.extend(pair_head.parameters())
        parameters.extend(extra)
        return [{"params": parameters, "lr": learning_rate}]

    frontend_parameters: list[nn.Parameter] = []
    head_parameters: list[nn.Parameter] = []
    for name, parameter in encoder.named_parameters():
        if not parameter.requires_grad:
            continue
        module_name = name.split(".", 1)[0]
        if module_name in FRONTEND_MODULE_NAMES:
            frontend_parameters.append(parameter)
        else:
            head_parameters.append(parameter)
    if pair_head is not None:
        head_parameters.extend(pair_head.parameters())
    head_parameters.extend(extra)
    if not frontend_parameters or not head_parameters:
        raise AssertionError("Full fine-tune requires frontend and head parameter groups")
    return [
        {
            "params": frontend_parameters,
            "lr": learning_rate * frontend_lr_multiplier,
        },
        {"params": head_parameters, "lr": learning_rate},
    ]


def format_optimizer_lrs(optimizer: torch.optim.Optimizer) -> str:
    return " ".join(
        f"lr{index}={group['lr']:.4g}"
        for index, group in enumerate(optimizer.param_groups)
    )


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


def train_one_epoch_paircls(
    encoder: typenet.TypeNetEncoder,
    pair_head: GalleryQueryPairHead,
    sampler: GalleryQueryClassSampler,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_size: int,
    steps_per_epoch: int,
    max_gradient_norm: float,
    bce_weight: float,
    side_encoder: RawTimingSideEncoder | None = None,
) -> tuple[float, float]:
    set_fine_tuning_mode(encoder)
    pair_head.train()
    if side_encoder is not None:
        side_encoder.train()
    assert_freeze_configuration(encoder)
    losses: list[float] = []
    accuracies: list[float] = []

    for _ in range(steps_per_epoch):
        (
            gallery_features,
            gallery_lengths,
            query_features,
            query_lengths,
            labels,
        ) = sampler.sample(batch_size, device)
        batch, gallery_size, time, features = gallery_features.shape
        combined = torch.cat(
            (
                gallery_features.reshape(batch * gallery_size, time, features),
                query_features,
            ),
            dim=0,
        )
        combined_lengths = torch.cat(
            (gallery_lengths.reshape(batch * gallery_size), query_lengths),
            dim=0,
        )
        embeddings = encoder(combined, combined_lengths)
        gallery_embeddings = embeddings[: batch * gallery_size].reshape(
            batch, gallery_size, typenet.PAPER_EMBEDDING_DIM
        )
        query_embeddings = embeddings[batch * gallery_size :]
        side_gallery_mean = None
        side_query = None
        if side_encoder is not None:
            side_embeddings = side_encoder(combined, combined_lengths)
            side_gallery = side_embeddings[: batch * gallery_size].reshape(
                batch, gallery_size, SIDE_OUTPUT_DIM
            )
            side_gallery_mean = side_gallery.mean(dim=1)
            side_query = side_embeddings[batch * gallery_size :]
        logits = pair_head(
            gallery_embeddings.mean(dim=1),
            query_embeddings,
            side_gallery_mean=side_gallery_mean,
            side_query=side_query,
        )
        loss = paircls_loss(logits, labels, bce_weight)

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
        trainable = [
            parameter
            for parameter in encoder.parameters()
            if parameter.requires_grad
        ] + list(pair_head.parameters())
        if side_encoder is not None:
            trainable.extend(side_encoder.parameters())
        nn.utils.clip_grad_norm_(trainable, max_norm=max_gradient_norm)
        optimizer.step()

        losses.append(float(loss.detach().cpu()))
        accuracies.append(
            float((logits.argmax(dim=1) == labels).float().mean().detach().cpu())
        )

    return float(np.mean(losses)), float(np.mean(accuracies))


@torch.inference_mode()
def classify_pair_scores(
    pair_head: GalleryQueryPairHead,
    galleries: np.ndarray,
    queries: np.ndarray,
    batch_size: int,
    device: torch.device,
    side_galleries: np.ndarray | None = None,
    side_queries: np.ndarray | None = None,
) -> np.ndarray:
    """Score [users, queries] as P(not normal) from mean-gallery / query pairs."""

    pair_head.eval()
    users, query_count, dim = queries.shape
    gallery_mean = np.repeat(galleries.mean(axis=1), query_count, axis=0)
    query_flat = queries.reshape(-1, dim)
    side_gallery_mean = None
    side_query_flat = None
    if side_galleries is not None and side_queries is not None:
        side_gallery_mean = np.repeat(
            side_galleries.mean(axis=1), query_count, axis=0
        )
        side_query_flat = side_queries.reshape(-1, side_queries.shape[-1])
    scores: list[np.ndarray] = []
    for start in range(0, len(query_flat), batch_size):
        stop = min(start + batch_size, len(query_flat))
        side_g = None
        side_q = None
        if side_gallery_mean is not None and side_query_flat is not None:
            side_g = torch.from_numpy(side_gallery_mean[start:stop]).to(device)
            side_q = torch.from_numpy(side_query_flat[start:stop]).to(device)
        logits = pair_head(
            torch.from_numpy(gallery_mean[start:stop]).to(device),
            torch.from_numpy(query_flat[start:stop]).to(device),
            side_gallery_mean=side_g,
            side_query=side_q,
        )
        scores.append(pair_not_normal_probability(logits).cpu().numpy())
    return np.concatenate(scores).reshape(users, query_count)


@torch.inference_mode()
def embed_feature_matrix(
    encoder: nn.Module,
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
    pair_head: GalleryQueryPairHead | None = None,
    side_encoder: RawTimingSideEncoder | None = None,
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
    side_galleries = None
    side_clean_queries = None
    if side_encoder is not None:
        clean_side = embed_feature_matrix(
            side_encoder,
            feature_matrix.reshape(
                -1, sequence_length, typenet.PAPER_FEATURE_COUNT
            ),
            length_matrix.reshape(-1),
            eval_batch_size,
            device,
        ).reshape(
            users,
            typenet.PAPER_SESSIONS_PER_USER,
            SIDE_OUTPUT_DIM,
        )
        side_galleries = clean_side[:, :gallery_size]
        side_clean_queries = clean_side[
            :,
            typenet.PAPER_QUERY_START : (
                typenet.PAPER_QUERY_START + typenet.PAPER_QUERY_COUNT
            ),
        ]
    if pair_head is None:
        normal_scores = mean_gallery_distances(galleries, clean_queries)
    else:
        normal_scores = classify_pair_scores(
            pair_head,
            galleries,
            clean_queries,
            eval_batch_size,
            device,
            side_galleries=side_galleries,
            side_queries=side_clean_queries,
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
    impaired_by_severity: dict[str, np.ndarray] = {}
    for severity_index, severity_name in enumerate(SEVERITY_NAMES):
        rng = np.random.default_rng(seed)
        perturbed_queries = np.empty_like(query_features)
        for user_index in range(users):
            profile = perturber.sample_profile(severity_index, rng)
            for query_index in range(typenet.PAPER_QUERY_COUNT):
                perturbed_queries[user_index, query_index] = perturber.perturb(
                    query_features[user_index, query_index],
                    int(query_lengths[user_index, query_index]),
                    severity_index,
                    rng,
                    profile=profile,
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
        if pair_head is None:
            impaired_by_severity[severity_name] = mean_gallery_distances(
                galleries, perturbed_embeddings
            )
        else:
            side_perturbed = None
            if side_encoder is not None:
                side_perturbed = embed_feature_matrix(
                    side_encoder,
                    perturbed_queries.reshape(
                        -1, sequence_length, typenet.PAPER_FEATURE_COUNT
                    ),
                    query_lengths.reshape(-1),
                    eval_batch_size,
                    device,
                ).reshape(users, typenet.PAPER_QUERY_COUNT, SIDE_OUTPUT_DIM)
            impaired_by_severity[severity_name] = classify_pair_scores(
                pair_head,
                galleries,
                perturbed_embeddings,
                eval_batch_size,
                device,
                side_galleries=side_galleries,
                side_queries=side_perturbed,
            )

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
        "overall": detection_metrics(
            scores.normal_flat, scores.impaired_flat, threshold
        )
    }
    for severity_name in SEVERITY_NAMES:
        metrics[severity_name] = detection_metrics(
            scores.normal_flat,
            scores.impaired_by_severity[severity_name].reshape(-1),
            threshold,
        )
    return metrics


def per_user_eer_metrics(
    normal_by_user: np.ndarray, impaired_by_user: np.ndarray
) -> typenet.PerUserEERMetrics:
    """Fit one tau per user from that user's gallery-vs-query scores.

    Each row is one subject. The clean scores are that subject's query
    sessions compared to their own n-session gallery. Impaired scores are
    the same queries after perturbation. Tau is the subject-specific EER
    threshold, then subject EERs are averaged.
    """
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
    scores: ImpairmentScores,
) -> dict[str, typenet.PerUserEERMetrics]:
    metrics = {"overall": per_user_eer_metrics(scores.normal, scores.impaired)}
    for severity_name in SEVERITY_NAMES:
        metrics[severity_name] = per_user_eer_metrics(
            scores.normal, scores.impaired_by_severity[severity_name]
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
    keycodes = np.asarray(
        (84, 104, 101, 32, 111, 116, 104, 101, 114, 8, 32, 97),
        dtype=np.float32,
    )
    clean[:length, 4] = keycodes[:length] / 255.0
    clean_before = clean.copy()

    perturbation_magnitudes: list[float] = []
    for severity_index in range(SEVERITY_COUNT):
        perturbed = perturber.perturb(
            clean,
            length,
            severity_index,
            np.random.default_rng(1234),
        )
        repeated = perturber.perturb(
            clean,
            length,
            severity_index,
            np.random.default_rng(1234),
        )
        np.testing.assert_array_equal(perturbed, repeated)
        np.testing.assert_array_equal(clean, clean_before)
        np.testing.assert_array_equal(perturbed[:, 4], clean[:, 4])
        np.testing.assert_array_equal(
            perturbed[length:], np.zeros_like(perturbed[length:])
        )
        if not np.isfinite(perturbed).all():
            raise AssertionError("Perturbation generated non-finite values")
        if np.any(perturbed[:length, 0] < 0):
            raise AssertionError("Perturbation generated negative hold latency")
        if np.any(perturbed[: length - 1, 2] < 0):
            raise AssertionError("Perturbation generated negative press latency")
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

    speed_matched_profile = SyntheticImpairmentProfile(
        severity_index=2,
        motor_burden=1.0,
        cognitive_burden=1.0,
        speed_matched=True,
    )
    speed_matched = perturber.perturb(
        clean,
        length,
        2,
        np.random.default_rng(4321),
        profile=speed_matched_profile,
    )
    hold_change = speed_matched[:length, 0] - clean[:length, 0]
    if not (np.any(hold_change < 0.0) and np.any(hold_change > 0.0)):
        raise AssertionError(
            "Speed-matched motor variation must be bidirectional"
        )

    context_probabilities = perturber._pause_probabilities(
        clean,
        length,
        severity_index=2,
        relative_cognitive_burden=1.0,
    )
    base_probability = perturber.config.pause_probabilities[2]
    if not math.isclose(
        context_probabilities[3],
        base_probability * perturber.config.word_boundary_pause_multiplier,
    ):
        raise AssertionError("Word-boundary pause weighting is invalid")
    if not math.isclose(
        context_probabilities[9],
        base_probability * perturber.config.correction_pause_multiplier,
    ):
        raise AssertionError("Correction pause weighting is invalid")


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


def _manifest_role_ids(manifest: Mapping[str, object], role_name: str) -> set[str]:
    roles = manifest.get("roles", {})
    if not isinstance(roles, Mapping):
        raise TypeError("Invalid roles in protocol manifest")
    role = roles.get(role_name, {})
    if not role:
        return set()
    if not isinstance(role, Mapping):
        raise TypeError(f"Invalid {role_name} role in protocol manifest")
    role_ids = role.get("participant_ids", [])
    if not isinstance(role_ids, list):
        raise TypeError(f"Invalid {role_name} participant_ids in protocol manifest")
    return {str(participant_id) for participant_id in role_ids}


def exposed_manifest_ids(paths: Sequence[Path]) -> set[str]:
    """Load identities whose outcomes were exposed by earlier protocols."""

    exposed_ids: set[str] = set()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Quarantine manifest not found: {path}")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        participant_ids = manifest.get("participant_ids", [])
        if not isinstance(participant_ids, list):
            raise TypeError(f"Invalid participant_ids in {path}")
        exposed_ids.update(str(participant_id) for participant_id in participant_ids)
        for role_name in (
            "legacy_quarantine",
            "historical_quarantine",
            "selection",
            "calibration",
            "final_test",
        ):
            exposed_ids.update(_manifest_role_ids(manifest, role_name))
    return exposed_ids


def pretrain_train_ids(paths: Sequence[Path]) -> set[str]:
    """Load identities that trained the frozen TypeNet encoder."""

    train_ids: set[str] = set()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Pretrain exclusion manifest not found: {path}")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        train_ids.update(_manifest_role_ids(manifest, "train"))
    return train_ids


def excluded_identity_ids(args: argparse.Namespace) -> set[str]:
    excluded = (
        exposed_manifest_ids(args.quarantine_manifests)
        if args.quarantine_manifests
        else set()
    )
    if args.exclude_pretrain and args.pretrain_manifests:
        excluded |= pretrain_train_ids(args.pretrain_manifests)
    return excluded


def hash_sorted_paths(paths: Sequence[Path], salt: str) -> list[Path]:
    return sorted(
        paths,
        key=lambda path: hashlib.sha256(
            (salt + ":" + typenet.participant_id_from_path(path)).encode("utf-8")
        ).digest(),
    )


def random_pool_split_counts(
    pool_size: int, dev_fraction: float, val_fraction_of_dev: float
) -> tuple[int, int, int]:
    """Return (train, val, test) for an 80/20 split with val inside the 80%."""

    if not 0.5 <= dev_fraction < 1.0:
        raise ValueError("--dev-fraction must be in [0.5, 1)")
    if not 0.0 < val_fraction_of_dev < 0.5:
        raise ValueError("--val-fraction-of-dev must be in (0, 0.5)")
    develop = int(round(pool_size * dev_fraction))
    test = pool_size - develop
    val = int(round(develop * val_fraction_of_dev))
    train = develop - val
    if min(train, val, test) < 2:
        raise ValueError(
            f"Unseen pool of {pool_size:,} is too small for the 80/20 split"
        )
    return train, val, test


def build_locked_protocol_splits(
    all_paths: Sequence[Path], args: argparse.Namespace
) -> ProtocolSplits:
    """Create disjoint cohorts while quarantining pretrained and inspected IDs."""

    selected = list(
        all_paths[: args.max_users] if args.max_users > 0 else all_paths
    )
    excluded_ids = excluded_identity_ids(args)
    if args.exclude_pretrain:
        args.legacy_validation_users = 0
        args.quarantined_test_users = 0
        historical_quarantine = [
            path
            for path in selected
            if typenet.participant_id_from_path(path) in excluded_ids
        ]
        candidates = [
            path
            for path in selected
            if typenet.participant_id_from_path(path) not in excluded_ids
        ]
        legacy_quarantine: list[Path] = []
        needed = (
            args.train_users
            + args.validation_users
            + args.calibration_users
            + args.test_users
        )
        if len(candidates) < needed:
            raise ValueError(
                f"Locked unseen protocol requires {needed:,} identities outside "
                f"pretrain/eval quarantine, found {len(candidates):,}"
            )
        candidates = hash_sorted_paths(candidates, args.protocol_salt)
        train = candidates[: args.train_users]
        rest = candidates[args.train_users :]
    else:
        legacy_count = (
            args.legacy_validation_users + args.quarantined_test_users
        )
        fixed_prefix_count = args.train_users + legacy_count
        if len(selected) < fixed_prefix_count:
            raise ValueError(
                f"Locked protocol requires at least {fixed_prefix_count:,} users, "
                f"found {len(selected):,}"
            )
        train_stop = args.train_users
        legacy_stop = train_stop + legacy_count
        train = selected[:train_stop]
        legacy_quarantine = selected[train_stop:legacy_stop]
        historical_quarantine = [
            path
            for path in selected[legacy_stop:]
            if typenet.participant_id_from_path(path) in excluded_ids
        ]
        rest = hash_sorted_paths(
            [
                path
                for path in selected[legacy_stop:]
                if typenet.participant_id_from_path(path) not in excluded_ids
            ],
            args.protocol_salt,
        )

    fresh_required = (
        args.validation_users + args.calibration_users + args.test_users
    )
    if len(rest) < fresh_required:
        raise ValueError(
            f"Locked protocol requires {fresh_required:,} unexposed evaluation "
            f"candidates, found {len(rest):,}"
        )
    selection_stop = args.validation_users
    calibration_stop = selection_stop + args.calibration_users
    test_stop = calibration_stop + args.test_users
    selection = rest[:selection_stop]
    calibration = rest[selection_stop:calibration_stop]
    final_test = rest[calibration_stop:test_stop]
    reserve = rest[test_stop:]

    roles = {
        "train": train,
        "legacy_quarantine": legacy_quarantine,
        "historical_quarantine": historical_quarantine,
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
    live_ids = {
        typenet.participant_id_from_path(path)
        for paths in (train, selection, calibration, final_test)
        for path in paths
    }
    leaked = live_ids & excluded_ids
    if leaked:
        raise AssertionError(
            "Train/eval cohorts contain quarantined or pretrained IDs: "
            + ", ".join(sorted(leaked)[:12])
        )

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
        "objective": args.objective,
        "exclude_pretrain": bool(args.exclude_pretrain),
        "random_pool_split": bool(args.random_pool_split),
        "dev_fraction": args.dev_fraction,
        "val_fraction_of_dev": args.val_fraction_of_dev,
        "impairment_score": (
            "p_not_normal"
            if args.objective == "paircls"
            else "mean_gallery_distance"
        ),
        "margin": args.margin,
        "epochs": args.epochs,
        "steps_per_epoch": args.steps_per_epoch,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "unfreeze_all": bool(args.unfreeze_all),
        "frontend_lr_multiplier": args.frontend_lr_multiplier,
        "pair_bce_weight": args.pair_bce_weight,
        "side_encoder": bool(args.side_encoder),
        "perturbation": asdict(perturbation_config_from_args(args)),
        "seed": args.seed,
        "quarantine_manifest_sha256": [
            sha256_file(path) for path in args.quarantine_manifests
        ],
        "pretrain_manifest_sha256": [
            sha256_file(path) for path in args.pretrain_manifests
        ],
    }
    if args.objective == "paircls":
        locked_config["classes"] = list(CLASS_NAMES)
    config_json = json.dumps(
        locked_config, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    manifest: dict[str, object] = {
        "protocol_version": args.protocol_version,
        "protocol_salt": args.protocol_salt,
        "purpose": (
            "Unseen-to-TypeNet pool, one frozen 80/20 split shared across "
            "independent fine-tunes from the same pretrained encoder. "
            "10% of the 80% develop split is validation."
            if args.random_pool_split
            else (
            "Identity-disjoint synthetic impairment protocol. Fine-tune and "
            "evaluation identities are held out from TypeNet pretraining and "
            "from every previously opened eval cohort."
            if args.exclude_pretrain
            else (
                "Identity-disjoint evidence-calibrated synthetic stress test; "
                "every identity exposed by prior experiments is quarantined."
            )
            )
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
        historical_quarantine=historical_quarantine,
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
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized, encoding="utf-8")


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


def per_user_metrics_payload(
    metrics: Mapping[str, typenet.PerUserEERMetrics],
) -> dict[str, dict[str, float]]:
    return {
        name: {
            "mean": value.mean,
            "standard_deviation": value.standard_deviation,
        }
        for name, value in metrics.items()
    }


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
            raise TypeError(f"Invalid {role_name} role in {manifest_path}")
        participant_ids = role.get("participant_ids", [])
        if not isinstance(participant_ids, list):
            raise TypeError(f"Invalid {role_name} participant_ids in {manifest_path}")
        return paths_from_participant_ids(
            data_dir, [str(participant_id) for participant_id in participant_ids]
        )

    return ProtocolSplits(
        train=role_paths("train"),
        legacy_quarantine=role_paths("legacy_quarantine"),
        historical_quarantine=role_paths("historical_quarantine"),
        selection=role_paths("selection"),
        calibration=role_paths("calibration"),
        final_test=role_paths("final_test"),
        reserve=role_paths("reserve"),
        manifest=manifest,
    )


def existing_role_paths(paths: Sequence[Path]) -> list[Path]:
    return [path for path in paths if path.is_file()]


def collect_complete_user_paths(
    paths: Sequence[Path],
    sequence_length: int,
    needed: int,
) -> list[Path]:
    complete: list[Path] = []
    for path in paths:
        if not path.is_file():
            continue
        try:
            user = typenet.load_user_sequences(path, sequence_length)
        except (OSError, ValueError, csv.Error):
            continue
        if len(user.features) < typenet.PAPER_SESSIONS_PER_USER:
            continue
        complete.append(path)
        if needed > 0 and len(complete) >= needed:
            break
    return complete


def checkpoint_payload(
    encoder: typenet.TypeNetEncoder,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    sampler: ImpairmentTripletSampler | GalleryQueryClassSampler,
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
    calibration_per_user: Mapping[str, typenet.PerUserEERMetrics] | None = None,
    selection_per_user: Mapping[str, typenet.PerUserEERMetrics] | None = None,
    test_metrics: Mapping[str, DetectionMetrics] | None = None,
    test_per_user: Mapping[str, typenet.PerUserEERMetrics] | None = None,
    final_test_accessed: bool = False,
    pair_head: GalleryQueryPairHead | None = None,
    side_encoder: RawTimingSideEncoder | None = None,
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
    if pair_head is not None:
        payload["pair_head_state_dict"] = pair_head.state_dict()
    if side_encoder is not None:
        payload["side_encoder_state_dict"] = side_encoder.state_dict()
    if calibration_metrics is not None:
        payload["calibration_metrics"] = {
            name: asdict(metrics)
            for name, metrics in calibration_metrics.items()
        }
    if calibration_per_user is not None:
        payload["calibration_per_user"] = per_user_metrics_payload(
            calibration_per_user
        )
    if selection_per_user is not None:
        payload["selection_per_user"] = per_user_metrics_payload(selection_per_user)
    if test_metrics is not None:
        payload["test_metrics"] = {
            name: asdict(metrics) for name, metrics in test_metrics.items()
        }
    if test_per_user is not None:
        payload["test_per_user"] = per_user_metrics_payload(test_per_user)
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
            / "typenet_68k_m50_g10_64x512_best_weights.pt"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            TRIPLET_DIR
            / "synthetic_impairment_evidence_v4_64x512_unseen_triplet_checkpoint.pt"
        ),
    )
    parser.add_argument(
        "--weights-output",
        type=Path,
        default=(
            TRIPLET_DIR
            / "synthetic_impairment_evidence_v4_64x512_unseen_triplet_weights.pt"
        ),
    )
    parser.add_argument(
        "--protocol-manifest",
        type=Path,
        default=(
            PROTOCOL_DIR
            / "synthetic_impairment_protocol_v4_64x512_unseen_triplet_manifest.json"
        ),
    )
    parser.add_argument(
        "--quarantine-manifests",
        type=Path,
        nargs="*",
        default=(
            INVALID_EXPERIMENTS_DIR
            / "synthetic_impairment_protocol_manifest.json",
            REPO_ROOT
            / "paper_typenet"
            / "weights"
            / "typenet_unbiased_1000_manifest.json",
            INVALID_EXPERIMENTS_DIR
            / "synthetic_impairment_protocol_v2_32x512_manifest.json",
            INVALID_EXPERIMENTS_DIR
            / "synthetic_impairment_protocol_v3_evidence_manifest.json",
            INVALID_EXPERIMENTS_DIR
            / "synthetic_impairment_protocol_v4_64x512_evidence_manifest.json",
            INVALID_EXPERIMENTS_DIR
            / "synthetic_impairment_protocol_v4_64x512_peruser_manifest.json",
            INVALID_EXPERIMENTS_DIR
            / "synthetic_impairment_protocol_v4_64x512_peruser_150e_manifest.json",
            INVALID_EXPERIMENTS_DIR
            / "synthetic_impairment_protocol_v4_64x512_paircls_manifest.json",
        ),
        help="Earlier manifests whose exposed identities must remain quarantined",
    )
    parser.add_argument(
        "--pretrain-manifests",
        type=Path,
        nargs="*",
        default=(
            INVALID_EXPERIMENTS_DIR
            / "synthetic_impairment_protocol_v4_64x512_evidence_manifest.json",
        ),
        help="Manifests whose train role is the frozen TypeNet pretraining set",
    )
    parser.add_argument(
        "--exclude-pretrain",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep TypeNet-pretrain identities out of fine-tune and eval",
    )
    parser.add_argument(
        "--objective",
        choices=("triplet", "paircls"),
        default="triplet",
        help="triplet: v4 gallery-distance fine-tune; paircls: 4-way pair head",
    )
    parser.add_argument(
        "--unfreeze-all",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Train every TypeNet parameter from the pretrained weights. "
            "Batch-norm affine scales move; running mean/var stay frozen."
        ),
    )
    parser.add_argument(
        "--frontend-lr-multiplier",
        type=float,
        default=0.1,
        help="LR multiplier for first LSTM and both batch-norms under --unfreeze-all",
    )
    parser.add_argument(
        "--pair-bce-weight",
        type=float,
        default=0.0,
        help=(
            "Add λ * BCE(P(not normal), impaired) to paircls CE. "
            "Zero keeps 4-way CE only."
        ),
    )
    parser.add_argument(
        "--side-encoder",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Train a zero-init residual LSTM on raw 5-d timings and concat "
            "it into the pair head. TypeNet frontend stays frozen."
        ),
    )
    parser.add_argument(
        "--random-pool-split",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Hash-split the unseen pool 80/20; keep this split for later methods",
    )
    parser.add_argument(
        "--dev-fraction",
        type=float,
        default=0.80,
        help="Fraction of the unseen pool used for train+val",
    )
    parser.add_argument(
        "--val-fraction-of-dev",
        type=float,
        default=0.10,
        help="Fraction of the develop split held out as validation",
    )
    parser.add_argument(
        "--reuse-split-manifest",
        type=Path,
        default=None,
        help=(
            "Load train/val/test identities from an existing split instead of "
            "drawing a new one. Use this for paircls/LoRA on the same 80/20."
        ),
    )
    parser.add_argument(
        "--lock-final-test",
        action="store_true",
        help="Burn this experiment's test cohort (off by default)",
    )
    parser.add_argument("--sequence-length", type=int, default=50)
    parser.add_argument("--gallery-size", type=int, default=10)
    parser.add_argument("--margin", type=float, default=1.5)
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
        default=0,
        help="Optional extra holdout; unused under --random-pool-split",
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
    parser.add_argument(
        "--word-boundary-pause-multiplier", type=float, default=3.0
    )
    parser.add_argument(
        "--correction-pause-multiplier", type=float, default=5.0
    )
    parser.add_argument("--pause-scale", type=float, default=2.0)
    parser.add_argument("--pause-tail-shape", type=float, default=1.8)
    parser.add_argument("--maximum-pause-seconds", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--protocol-version",
        default="synthetic-impairment-v4-64x512-unseen-triplet-8020",
    )
    parser.add_argument(
        "--protocol-salt",
        default="synthetic-impairment-v4-64x512-unseen-triplet-8020-2026-09-13",
        help="Locked salt for deterministic hash-randomized holdout assignment",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto (MLX if present, else Apple GPU/MPS, else CPU), cpu, mps, or MLX",
    )
    parser.add_argument(
        "--restate-checkpoint",
        type=Path,
        default=None,
        help=(
            "Evaluate an existing burned checkpoint under per-user EER on its "
            "selection and calibration cohorts; does not train or open final test"
        ),
    )
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
    args.quarantine_manifests = ()
    args.pretrain_manifests = ()
    args.exclude_pretrain = False
    args.random_pool_split = False
    args.lock_final_test = False
    args.reuse_split_manifest = None


def perturbation_config_from_args(
    args: argparse.Namespace,
) -> PerturbationConfig:
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


def fit_protocol_to_available_users(
    args: argparse.Namespace, all_paths: Sequence[Path]
) -> None:
    """Shrink train so enough unexcluded identities remain for eval."""

    selected = list(
        all_paths[: args.max_users] if args.max_users > 0 else all_paths
    )
    user_count = len(selected)
    eval_needed = (
        args.validation_users + args.calibration_users + args.test_users
    )
    excluded_ids = excluded_identity_ids(args)
    available = [
        path
        for path in selected
        if typenet.participant_id_from_path(path) not in excluded_ids
    ]
    available_count = len(available)
    if args.random_pool_split:
        args.legacy_validation_users = 0
        args.quarantined_test_users = 0
        args.train_users, args.validation_users, args.test_users = (
            random_pool_split_counts(
                available_count, args.dev_fraction, args.val_fraction_of_dev
            )
        )
        args.calibration_users = 0
        print(
            f"unseen pool={available_count:,} of {user_count:,}  "
            f"train={args.train_users:,} ({args.train_users / available_count:.0%})  "
            f"val={args.validation_users:,} ({args.validation_users / available_count:.0%})  "
            f"test={args.test_users:,} ({args.test_users / available_count:.0%})"
        )
        return

    eval_needed = (
        args.validation_users + args.calibration_users + args.test_users
    )
    required = args.train_users + eval_needed
    if available_count >= required:
        return

    print(
        f"usable unseen users={available_count:,} of {user_count:,} "
        f"(wanted {required:,}); shrinking train"
    )
    args.legacy_validation_users = 0
    args.quarantined_test_users = 0
    args.train_users = min(args.train_users, max(2, available_count - eval_needed))
    remaining = available_count - args.train_users
    if remaining < eval_needed:
        per_eval = max(2, remaining // 3)
        args.validation_users = per_eval
        args.calibration_users = per_eval
        args.test_users = remaining - 2 * per_eval
        args.train_users = max(2, available_count - remaining)
    print(
        f"cohorts train={args.train_users:,} sel={args.validation_users:,} "
        f"cal={args.calibration_users:,} test={args.test_users:,}"
    )


def validate_arguments(args: argparse.Namespace) -> None:
    if args.sequence_length != typenet.PAPER_SEQUENCE_LENGTH:
        raise ValueError("The pretrained encoder requires M=50")
    if not 1 <= args.gallery_size <= typenet.PAPER_GALLERY_POOL:
        raise ValueError("--gallery-size must be between 1 and 10")
    if args.margin <= 0.0:
        raise ValueError("--margin must be positive")
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
    if args.train_users < 2 or args.validation_users < 2 or args.test_users < 2:
        raise ValueError(
            "Train, validation, and test cohorts need at least two users"
        )
    if args.calibration_users < 0 or (
        args.calibration_users == 1
    ):
        raise ValueError("--calibration-users must be 0 or at least two")
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
    ratios = (args.hold_location_ratio, args.press_location_ratio)
    if any(ratio < 1.0 for ratio in ratios):
        raise ValueError("Timing location ratios must be at least one")
    nonnegative_values = (
        args.hold_log_variability,
        args.press_log_variability,
        args.profile_log_variability,
        args.session_log_variability,
    )
    if any(value < 0.0 for value in nonnegative_values):
        raise ValueError("Log-variability parameters cannot be negative")
    if not 0.0 <= args.speed_matched_probability <= 1.0:
        raise ValueError("--speed-matched-probability must be within [0, 1]")
    if (
        args.word_boundary_pause_multiplier < 1.0
        or args.correction_pause_multiplier < 1.0
    ):
        raise ValueError("Context pause multipliers must be at least one")
    if args.pause_scale <= 0.0 or args.maximum_pause_seconds <= 0.0:
        raise ValueError("Pause scale and maximum pause must be positive")
    if args.pause_tail_shape <= 1.0:
        raise ValueError("--pause-tail-shape must exceed one")
    if args.lr_decay_patience < 1:
        raise ValueError("--lr-decay-patience must be at least one")
    if not 0.0 < args.lr_decay_factor < 1.0:
        raise ValueError("--lr-decay-factor must be between zero and one")
    if args.objective not in {"triplet", "paircls"}:
        raise ValueError("--objective must be triplet or paircls")
    if not 0.0 < args.frontend_lr_multiplier <= 1.0:
        raise ValueError("--frontend-lr-multiplier must be in (0, 1]")
    if args.pair_bce_weight < 0.0:
        raise ValueError("--pair-bce-weight cannot be negative")
    if args.side_encoder and args.objective != "paircls":
        raise ValueError("--side-encoder requires --objective paircls")
    if not 0.5 <= args.dev_fraction < 1.0:
        raise ValueError("--dev-fraction must be in [0.5, 1)")
    if not 0.0 < args.val_fraction_of_dev < 0.5:
        raise ValueError("--val-fraction-of-dev must be in (0, 0.5)")


def score_cohort(
    encoder: typenet.TypeNetEncoder,
    paths: Sequence[Path],
    perturber: StructuredTimingPerturber,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
    threshold: float | None = None,
    pair_head: GalleryQueryPairHead | None = None,
    side_encoder: RawTimingSideEncoder | None = None,
) -> tuple[ImpairmentScores, float, float, dict[str, DetectionMetrics], dict[str, typenet.PerUserEERMetrics]]:
    scores = evaluate_impairment_scores(
        encoder,
        paths,
        perturber,
        args.sequence_length,
        args.gallery_size,
        args.eval_users,
        args.eval_batch_size,
        device,
        seed,
        pair_head=pair_head,
        side_encoder=side_encoder,
    )
    fitted_threshold, fitted_eer = typenet.equal_error_threshold(
        scores.normal_flat, scores.impaired_flat
    )
    operating_threshold = fitted_threshold if threshold is None else threshold
    return (
        scores,
        fitted_threshold,
        fitted_eer,
        evaluate_all_metrics(scores, operating_threshold),
        evaluate_all_per_user_metrics(scores),
    )


def print_cohort_metrics(
    prefix: str,
    per_user_metrics: Mapping[str, typenet.PerUserEERMetrics],
) -> None:
    print(format_cohort_per_user(prefix, per_user_metrics))


def run_restate(args: argparse.Namespace, device: torch.device) -> None:
    """Re-score a burned checkpoint without training or opening final test."""

    if args.restate_checkpoint is None or not args.restate_checkpoint.is_file():
        raise FileNotFoundError(
            f"Restate checkpoint not found: {args.restate_checkpoint}"
        )
    payload = load_checkpoint_payload(args.restate_checkpoint)
    saved_manifest = payload.get("protocol_manifest")
    saved_manifest_name = (
        pathlib.PureWindowsPath(str(saved_manifest)).name if saved_manifest else ""
    )
    manifest_candidates = [
        Path(str(saved_manifest)) if saved_manifest else None,
        find_weights_file(saved_manifest_name),
        args.protocol_manifest,
    ]
    manifest_path = next(
        (
            candidate
            for candidate in manifest_candidates
            if candidate is not None and candidate.is_file()
        ),
        None,
    )
    if manifest_path is None:
        raise FileNotFoundError(
            "Locked protocol manifest not found for restatement "
            f"(saved path: {saved_manifest})"
        )
    protocol = protocol_from_existing_manifest(manifest_path, args.data_dir)
    fingerprints = protocol_fingerprints(protocol.manifest)
    selection_paths = existing_role_paths(protocol.selection)
    calibration_paths = existing_role_paths(protocol.calibration)
    restatement_source = "locked-manifest"
    if len(selection_paths) < 2 or len(calibration_paths) < 2:
        print(
            "WARNING: locked v4 selection/calibration files are not on this machine. "
            "Scoring a local identity-disjoint proxy of complete users instead. "
            "This is not the burned v4 cohort."
        )
        restatement_source = "local-proxy"
        local_paths = typenet.discover_user_files(args.data_dir)
        if args.max_users > 0:
            local_paths = local_paths[: args.max_users]
        needed = max(args.validation_users + args.calibration_users, 4)
        proxy_paths = collect_complete_user_paths(
            local_paths, args.sequence_length, needed
        )
        if len(proxy_paths) < 4:
            raise RuntimeError(
                "Not enough complete local users to restate per-user EER"
            )
        split = max(2, len(proxy_paths) // 2)
        selection_paths = proxy_paths[:split]
        calibration_paths = proxy_paths[split:]
    print(
        "Restating existing checkpoint under per-user EER; "
        "final-test cohort will not be opened."
    )
    print(
        f"Manifest {manifest_path.name}: source={restatement_source} "
        f"selection={len(selection_paths)} "
        f"calibration={len(calibration_paths)} "
        f"locked-selection={fingerprints['selection'][:12]} "
        f"locked-calibration={fingerprints['calibration'][:12]}"
    )

    perturbation_config = perturbation_config_from_args(args)
    saved_perturbation = payload.get("perturbation_config")
    if isinstance(saved_perturbation, Mapping):
        restored: dict[str, object] = {}
        for field in perturbation_config.__dataclass_fields__:
            value = saved_perturbation.get(field, getattr(perturbation_config, field))
            if field in ("severity_factors", "pause_probabilities"):
                value = tuple(value)
            restored[field] = value
        perturbation_config = PerturbationConfig(**restored)
    perturber = StructuredTimingPerturber(perturbation_config)
    encoder = load_pretrained_encoder(args.pretrained_weights, device)
    encoder.load_state_dict(payload["model_state_dict"], strict=True)
    saved_config = payload.get("config", {})
    unfreeze_all = (
        bool(saved_config.get("unfreeze_all", False))
        if isinstance(saved_config, Mapping)
        else False
    )
    configure_encoder_training(encoder, unfreeze_all)
    encoder.eval()
    pair_head = load_pair_head(payload, device)
    if pair_head is not None:
        pair_head.eval()
    side_encoder = load_side_encoder(payload, device)
    if side_encoder is not None:
        side_encoder.eval()

    selection_scores, selection_threshold, selection_eer, selection_global, selection_per_user = (
        score_cohort(
            encoder,
            selection_paths,
            perturber,
            args,
            device,
            args.seed + 10_000,
            pair_head=pair_head,
            side_encoder=side_encoder,
        )
    )
    print_cohort_metrics("Restated selection", selection_per_user)
    print(f"Restated selection users={selection_scores.user_count}")

    calibration_scores, calibration_threshold, calibration_eer, calibration_global, calibration_per_user = (
        score_cohort(
            encoder,
            calibration_paths,
            perturber,
            args,
            device,
            args.seed + 20_000,
            pair_head=pair_head,
            side_encoder=side_encoder,
        )
    )
    print_cohort_metrics("Restated calibration", calibration_per_user)
    print(f"Restated calibration users={calibration_scores.user_count}")
    result = {
        "restated_checkpoint": str(args.restate_checkpoint),
        "protocol_manifest": str(manifest_path),
        "restatement_source": restatement_source,
        "selection_per_user": per_user_metrics_payload(selection_per_user),
        "calibration_per_user": per_user_metrics_payload(calibration_per_user),
    }
    print("RESULT_JSON " + json.dumps(result, sort_keys=True))


def main() -> None:
    args = build_argument_parser().parse_args()
    if args.smoke_test:
        apply_smoke_settings(args)
    validate_arguments(args)
    typenet.set_reproducible_seed(args.seed)
    device = typenet.resolve_device(args.device)
    print(f"device={device}")

    if args.restate_checkpoint is not None:
        run_restate(args, device)
        return

    perturbation_config = perturbation_config_from_args(args)
    perturber = StructuredTimingPerturber(perturbation_config)
    run_deterministic_perturbation_checks(perturber, args.sequence_length)

    all_paths = typenet.discover_user_files(args.data_dir)
    if args.reuse_split_manifest is not None:
        protocol = protocol_from_existing_manifest(
            args.reuse_split_manifest, args.data_dir
        )
        print(f"reusing split {args.reuse_split_manifest.name}")
    else:
        fit_protocol_to_available_users(args, all_paths)
        protocol = build_locked_protocol_splits(all_paths, args)
        write_or_verify_protocol_manifest(
            protocol.manifest, args.protocol_manifest
        )
    fingerprints = protocol_fingerprints(protocol.manifest)
    print(
        f"users={len(all_paths):,}  train={len(protocol.train):,}  "
        f"val={len(protocol.selection):,}  "
        f"test={len(protocol.final_test):,}  "
        f"reserve={len(protocol.reserve):,}  "
        f"held-out={len(protocol.historical_quarantine):,}"
        + (
            f"  cal={len(protocol.calibration):,}"
            if protocol.calibration
            else ""
        )
    )
    if args.checkpoint is not None and args.checkpoint.exists():
        existing_checkpoint = load_checkpoint_payload(args.checkpoint)
        if existing_checkpoint.get("final_test_accessed", False):
            raise RuntimeError(
                "Refusing to reuse a checkpoint whose one-shot final test has "
                "already been accessed. Use a new locked protocol and output path."
            )

    encoder = load_pretrained_encoder(args.pretrained_weights, device)
    configure_encoder_training(encoder, args.unfreeze_all)
    set_fine_tuning_mode(encoder)
    assert_freeze_configuration(encoder)
    pair_head: GalleryQueryPairHead | None = None
    side_encoder: RawTimingSideEncoder | None = None
    if args.objective == "paircls":
        if args.side_encoder:
            side_encoder = build_side_encoder(device)
            side_encoder.train()
        pair_head = build_pair_head(
            device, side_dim=SIDE_OUTPUT_DIM if side_encoder is not None else 0
        )
        pair_head.train()
    param_groups = optimizer_param_groups(
        encoder,
        pair_head,
        args.learning_rate,
        args.unfreeze_all,
        args.frontend_lr_multiplier,
        extra_parameters=(
            list(side_encoder.parameters()) if side_encoder is not None else None
        ),
    )
    trainable_count = sum(
        parameter.numel()
        for group in param_groups
        for parameter in group["params"]
    )
    scope = "all params" if args.unfreeze_all else "second_lstm"
    if pair_head is None:
        print(f"fine-tune {scope}  params={trainable_count:,}  objective=triplet")
    else:
        pair_head_count = sum(parameter.numel() for parameter in pair_head.parameters())
        side_count = (
            sum(parameter.numel() for parameter in side_encoder.parameters())
            if side_encoder is not None
            else 0
        )
        head_scope = scope if args.unfreeze_all else "second_lstm+pair_head"
        if side_encoder is not None:
            head_scope += "+side"
        print(
            f"fine-tune {head_scope}  params={trainable_count:,} "
            f"(head={pair_head_count:,}"
            + (f"  side={side_count:,}" if side_count else "")
            + ")  objective=paircls"
        )
    frozen_before = frozen_parameter_snapshot(encoder)

    store = typenet.KeystrokeStore(
        protocol.train, args.sequence_length, args.cache_users
    )
    if args.objective == "paircls":
        sampler: ImpairmentTripletSampler | GalleryQueryClassSampler = (
            GalleryQueryClassSampler(
                store, perturber, args.seed, args.gallery_size
            )
        )
        class_check = sampler.balanced_class_indices(args.batch_size)
        class_counts = np.bincount(class_check, minlength=CLASS_COUNT)
        if int(class_counts.max() - class_counts.min()) > 1:
            raise AssertionError("Class sampling is not balanced")
    else:
        sampler = ImpairmentTripletSampler(store, perturber, args.seed)
        severity_check = sampler.balanced_severity_indices(args.batch_size)
        severity_counts = np.bincount(severity_check, minlength=SEVERITY_COUNT)
        if int(severity_counts.max() - severity_counts.min()) > 1:
            raise AssertionError("Severity sampling is not balanced")

    optimizer = torch.optim.Adam(param_groups)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_decay_factor,
        patience=args.lr_decay_patience - 1,
        min_lr=args.minimum_learning_rate,
        threshold=0.0,
    )
    if args.objective == "paircls":
        print(
            f"{format_optimizer_lrs(optimizer)}  "
            f"classes={','.join(CLASS_NAMES)}  "
            f"score=P(not normal)  "
            f"loss=CE+{args.pair_bce_weight:g}*BCE  "
            f"side={'on' if side_encoder is not None else 'off'}  "
            f"batch={args.batch_size}  "
            f"steps={args.steps_per_epoch}  G={args.gallery_size}"
        )
    else:
        print(
            f"{format_optimizer_lrs(optimizer)}  margin={args.margin:g}  "
            f"score=mean gallery distance  batch={args.batch_size}  "
            f"steps={args.steps_per_epoch}"
        )

    best_epoch = 0
    best_validation_eer = math.inf
    best_threshold: float | None = None
    best_state: dict[str, object] | None = None
    best_selection_per_user: dict[str, typenet.PerUserEERMetrics] | None = None
    checks_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        if args.objective == "paircls":
            assert isinstance(sampler, GalleryQueryClassSampler)
            assert pair_head is not None
            loss, accuracy = train_one_epoch_paircls(
                encoder,
                pair_head,
                sampler,
                optimizer,
                device,
                args.batch_size,
                args.steps_per_epoch,
                args.max_gradient_norm,
                args.pair_bce_weight,
                side_encoder=side_encoder,
            )
            epoch_line = (
                f"ep {epoch:03d}  loss={loss:.3f}  acc={accuracy:.3f}  "
                f"{format_optimizer_lrs(optimizer)}"
            )
        else:
            assert isinstance(sampler, ImpairmentTripletSampler)
            loss, _, _, _ = train_one_epoch(
                encoder,
                sampler,
                optimizer,
                device,
                args.batch_size,
                args.steps_per_epoch,
                args.margin,
                args.max_gradient_norm,
            )
            epoch_line = (
                f"ep {epoch:03d}  loss={loss:.3f}  "
                f"{format_optimizer_lrs(optimizer)}"
            )
        if epoch % args.validate_every != 0 and epoch != args.epochs:
            continue
        print(epoch_line)
        (
            _validation_scores,
            validation_threshold,
            _validation_global_eer,
            validation_metrics,
            validation_per_user,
        ) = score_cohort(
            encoder,
            protocol.selection,
            perturber,
            args,
            device,
            args.seed + 10_000,
            pair_head=pair_head,
            side_encoder=side_encoder,
        )
        validation_eer = validation_per_user["overall"].mean
        print_cohort_metrics("  val", validation_per_user)

        previous_lrs = [group["lr"] for group in optimizer.param_groups]
        scheduler.step(validation_eer)
        current_lrs = [group["lr"] for group in optimizer.param_groups]
        if current_lrs != previous_lrs:
            before = " ".join(f"{lr:.4g}" for lr in previous_lrs)
            after = " ".join(f"{lr:.4g}" for lr in current_lrs)
            print(f"  lr {before} -> {after}")

        if validation_eer < best_validation_eer:
            best_epoch = epoch
            best_validation_eer = validation_eer
            best_threshold = validation_threshold
            best_state = {
                "encoder": copy.deepcopy(encoder.state_dict()),
                "pair_head": (
                    copy.deepcopy(pair_head.state_dict())
                    if pair_head is not None
                    else None
                ),
                "side_encoder": (
                    copy.deepcopy(side_encoder.state_dict())
                    if side_encoder is not None
                    else None
                ),
            }
            best_selection_per_user = validation_per_user
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
                        selection_per_user=validation_per_user,
                        pair_head=pair_head,
                        side_encoder=side_encoder,
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

    if best_state is None or best_threshold is None:
        raise RuntimeError("No validated prototype checkpoint was produced")
    encoder.load_state_dict(best_state["encoder"])
    if pair_head is not None:
        pair_head_state = best_state["pair_head"]
        if not isinstance(pair_head_state, dict):
            raise RuntimeError("Best pair-head weights are missing")
        pair_head.load_state_dict(pair_head_state)
        pair_head.eval()
    if side_encoder is not None:
        side_state = best_state.get("side_encoder")
        if not isinstance(side_state, dict):
            raise RuntimeError("Best side-encoder weights are missing")
        side_encoder.load_state_dict(side_state)
        side_encoder.eval()
    assert_frozen_parameters_unchanged(encoder, frozen_before)

    calibration_threshold = best_threshold
    calibration_eer = best_validation_eer
    calibration_metrics = None
    calibration_per_user = None
    if len(protocol.calibration) >= 2:
        (
            _calibration_scores,
            calibration_threshold,
            calibration_eer,
            calibration_metrics,
            calibration_per_user,
        ) = score_cohort(
            encoder,
            protocol.calibration,
            perturber,
            args,
            device,
            args.seed + 20_000,
            pair_head=pair_head,
            side_encoder=side_encoder,
        )
        print_cohort_metrics("cal", calibration_per_user)
    print(f"best ep={best_epoch}  val {best_validation_eer * 100:.2f}%")

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
                calibration_per_user=calibration_per_user,
                selection_per_user=best_selection_per_user,
                final_test_accessed=False,
                pair_head=pair_head,
                side_encoder=side_encoder,
            ),
            args.checkpoint,
        )
        print("sealed checkpoint")

    if args.lock_final_test:
        lock_final_test_access(protocol.manifest, args.protocol_manifest)
    print("test")
    (
        _test_scores,
        _test_threshold,
        _test_fitted_eer,
        test_metrics,
        test_per_user,
    ) = score_cohort(
        encoder,
        protocol.final_test,
        perturber,
        args,
        device,
        args.seed + 30_000,
        threshold=calibration_threshold,
        pair_head=pair_head,
        side_encoder=side_encoder,
    )
    print_cohort_metrics("test", test_per_user)

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
                calibration_per_user=calibration_per_user,
                selection_per_user=best_selection_per_user,
                test_metrics=test_metrics,
                test_per_user=test_per_user,
                final_test_accessed=bool(args.lock_final_test),
                pair_head=pair_head,
                side_encoder=side_encoder,
            ),
            args.checkpoint,
        )
    if args.weights_output is not None:
        args.weights_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "encoder": encoder.state_dict(),
                "pair_head": (
                    pair_head.state_dict() if pair_head is not None else None
                ),
                "side_encoder": (
                    side_encoder.state_dict() if side_encoder is not None else None
                ),
                "objective": args.objective,
            },
            args.weights_output,
        )

    result = {
        "best_epoch": best_epoch,
        "model_selection_per_user_eer": best_validation_eer,
        "model_selection_per_user": (
            per_user_metrics_payload(best_selection_per_user)
            if best_selection_per_user is not None
            else None
        ),
        "calibration_per_user": (
            per_user_metrics_payload(calibration_per_user)
            if calibration_per_user is not None
            else None
        ),
        "final_test_fingerprint": fingerprints["final_test"],
        "test_per_user": per_user_metrics_payload(test_per_user),
    }
    print("RESULT_JSON " + json.dumps(result, sort_keys=True))
    if args.smoke_test:
        print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
