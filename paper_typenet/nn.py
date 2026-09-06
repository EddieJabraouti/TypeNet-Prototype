"""Single-file reproduction of the TypeNet keystroke-biometrics pipeline.

This implementation follows Acien et al., "TypeNet: Deep Learning Keystroke
Biometrics" (arXiv:2101.05570):

* Input: M=50 keystrokes represented by [HL, IL, PL, RL, keycode/255].
* Encoder: masking -> input batch norm -> LSTM(128) -> dropout(0.5) ->
  batch norm -> LSTM(128), with recurrent dropout 0.2 in both LSTMs.
* Output: one unnormalised 128-dimensional embedding.
* Training: random triplets, squared-Euclidean triplet loss, margin 1.5.
* Verification score: mean Euclidean distance from a query to G enrollment
  embeddings (G=10 by default).

The paper computes EERs on its test population and does not state a validation
split or a fixed numeric threshold.  For a deployable train/validation/test
pipeline, this file uses identity-disjoint subjects, selects a single global
threshold at validation EER, freezes it, and reports FAR/FRR/HTER on test.

Only NumPy and PyTorch are required.  Run a small end-to-end check with:

    python paper_typenet/nn.py --smoke-test

Run the paper-scale defaults (68K training identities, 200 epochs, 150
batches/epoch) by omitting --smoke-test.  This is computationally expensive.
Use --max-users, --epochs, and --steps-per-epoch for intermediate experiments.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


csv.field_size_limit(10_000_000)


PAPER_SEQUENCE_LENGTH = 50
PAPER_FEATURE_COUNT = 5
PAPER_EMBEDDING_DIM = 128
PAPER_TRAINABLE_PARAMETERS = 200_458
PAPER_TRAIN_USERS = 68_000
PAPER_SESSIONS_PER_USER = 15
PAPER_GALLERY_POOL = 10
PAPER_QUERY_START = 10
PAPER_QUERY_COUNT = 5


@dataclass(frozen=True)
class UserSequences:
    """Preprocessed fixed-length sessions belonging to one subject."""

    participant_id: str
    features: np.ndarray  # [sessions, M, 5], float32
    lengths: np.ndarray  # [sessions], int64


@dataclass
class VerificationMetrics:
    threshold: float
    far: float
    frr: float
    hter: float


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def participant_id_from_path(path: Path) -> str:
    return path.name.removesuffix("_keystrokes.txt")


def participant_sort_key(path: Path) -> tuple[int, int | str]:
    participant_id = participant_id_from_path(path)
    try:
        return (0, int(participant_id))
    except ValueError:
        return (1, participant_id)


def discover_user_files(data_dir: Path) -> list[Path]:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Keystroke data directory does not exist: {data_dir}")
    paths = sorted(data_dir.glob("*_keystrokes.txt"), key=participant_sort_key)
    if not paths:
        raise FileNotFoundError(
            f"No '<participant>_keystrokes.txt' files found under {data_dir}"
        )
    return paths


def events_to_features(
    events: Sequence[tuple[int, int, int, int]],
    sequence_length: int,
) -> tuple[np.ndarray, int]:
    """Convert (press, release, keycode, keystroke_id) events to TypeNet input.

    Timing features are computed before truncation, so the M-th retained key can
    use the following event when one exists.  The final event has no successor,
    therefore IL/PL/RL are set to zero while its HL and keycode are retained.
    Negative IL values caused by key rollover are intentionally preserved.
    """

    ordered = sorted(events, key=lambda event: (event[0], event[3]))
    count = len(ordered)
    output = np.zeros((sequence_length, PAPER_FEATURE_COUNT), dtype=np.float32)
    if count == 0:
        return output, 0

    press = np.fromiter((event[0] for event in ordered), dtype=np.float64)
    release = np.fromiter((event[1] for event in ordered), dtype=np.float64)
    keycode = np.fromiter((event[2] for event in ordered), dtype=np.float64)

    raw = np.zeros((count, PAPER_FEATURE_COUNT), dtype=np.float64)
    raw[:, 0] = (release - press) / 1_000.0  # Hold latency (HL)
    if count > 1:
        raw[:-1, 1] = (press[1:] - release[:-1]) / 1_000.0  # Inter-key (IL)
        raw[:-1, 2] = (press[1:] - press[:-1]) / 1_000.0  # Press (PL)
        raw[:-1, 3] = (release[1:] - release[:-1]) / 1_000.0  # Release (RL)
    raw[:, 4] = keycode / 255.0

    retained = min(count, sequence_length)
    output[:retained] = raw[:retained].astype(np.float32)
    return output, retained


def load_user_sequences(path: Path, sequence_length: int) -> UserSequences:
    """Read one Aalto 136M-keystrokes TSV and preprocess all its sessions."""

    sessions: dict[str, list[tuple[int, int, int, int]]] = {}                            #{str: [(int, int, int, int, int), (int....)]} dict contains all longitudinal features per string (key press)
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {
            "TEST_SECTION_ID",
            "KEYSTROKE_ID",
            "PRESS_TIME",
            "RELEASE_TIME",
            "KEYCODE",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = required.difference(reader.fieldnames or [])
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")

        for row in reader:
            try:
                session_id = row["TEST_SECTION_ID"]
                event = (
                    int(row["PRESS_TIME"]),
                    int(row["RELEASE_TIME"]),
                    int(row["KEYCODE"]),
                    int(row["KEYSTROKE_ID"]),
                )
            except (KeyError, TypeError, ValueError):
                continue
            sessions.setdefault(session_id, []).append(event)

    encoded: list[np.ndarray] = []
    lengths: list[int] = []
    for events in sessions.values():
        features, length = events_to_features(events, sequence_length)
        if length > 0 and np.isfinite(features).all():
            encoded.append(features)
            lengths.append(length)

    if not encoded:
        raise ValueError(f"No valid keystroke sessions found in {path}")

    return UserSequences(
        participant_id=participant_id_from_path(path),
        features=np.stack(encoded),
        lengths=np.asarray(lengths, dtype=np.int64),
    )


class KeystrokeStore:
    """Lazy user-level loader with a bounded in-memory preprocessing cache."""

    def __init__(
        self,
        paths: Sequence[Path],
        sequence_length: int,
        cache_users: int,
    ) -> None:
        self.paths = list(paths)
        self.sequence_length = sequence_length
        self.cache_users = cache_users
        self._cache: OrderedDict[int, UserSequences] = OrderedDict()

    def __len__(self) -> int:
        return len(self.paths)

    def get(self, index: int) -> UserSequences:
        cached = self._cache.pop(index, None)
        if cached is not None:
            self._cache[index] = cached
            return cached

        user = load_user_sequences(self.paths[index], self.sequence_length)
        if self.cache_users != 0:
            self._cache[index] = user
            if self.cache_users > 0 and len(self._cache) > self.cache_users:
                self._cache.popitem(last=False)
        return user


class TripletBatchSampler:
    """Random TypeNet triplets: two sessions from one user and one impostor."""

    def __init__(self, store: KeystrokeStore, seed: int) -> None:
        if len(store) < 2:
            raise ValueError("Triplet training requires at least two subjects")
        self.store = store
        self.rng = np.random.default_rng(seed)

    def _sample_one(
        self,
    ) -> tuple[np.ndarray, int, np.ndarray, int, np.ndarray, int]:
        for _ in range(100):
            anchor_index, negative_index = self.rng.choice(
                len(self.store), size=2, replace=False
            )
            try:
                anchor_user = self.store.get(int(anchor_index))
                negative_user = self.store.get(int(negative_index))
            except (OSError, ValueError, csv.Error):
                continue
            if len(anchor_user.features) < 2 or len(negative_user.features) < 1:
                continue

            anchor_session, positive_session = self.rng.choice(
                len(anchor_user.features), size=2, replace=False
            )
            negative_session = int(self.rng.integers(len(negative_user.features)))
            return (
                anchor_user.features[anchor_session],
                int(anchor_user.lengths[anchor_session]),
                anchor_user.features[positive_session],
                int(anchor_user.lengths[positive_session]),
                negative_user.features[negative_session],
                int(negative_user.lengths[negative_session]),
            )
        raise RuntimeError("Could not sample a valid triplet after 100 attempts")

    def sample(
        self, batch_size: int, device: torch.device
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        triplets = [self._sample_one() for _ in range(batch_size)]
        anchor = torch.from_numpy(np.stack([item[0] for item in triplets])).to(device)
        anchor_lengths = torch.tensor(
            [item[1] for item in triplets], dtype=torch.long, device=device
        )
        positive = torch.from_numpy(np.stack([item[2] for item in triplets])).to(device)
        positive_lengths = torch.tensor(
            [item[3] for item in triplets], dtype=torch.long, device=device
        )
        negative = torch.from_numpy(np.stack([item[4] for item in triplets])).to(device)
        negative_lengths = torch.tensor(
            [item[5] for item in triplets], dtype=torch.long, device=device
        )
        return (
            anchor,
            anchor_lengths,
            positive,
            positive_lengths,
            negative,
            negative_lengths,
        )


class KerasStyleLSTM(nn.Module):
    """An LSTM with Keras-style variational recurrent dropout.

    PyTorch's built-in LSTM does not implement recurrent dropout.  This module
    uses one recurrent dropout mask per gate, held fixed across time, matching
    the behavior used by the paper's Keras/TensorFlow implementation.
    """

    def __init__(self, input_size: int, hidden_size: int, recurrent_dropout: float) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.recurrent_dropout = recurrent_dropout
        self.kernel = nn.Parameter(torch.empty(input_size, 4 * hidden_size))
        self.recurrent_kernel = nn.Parameter(
            torch.empty(hidden_size, 4 * hidden_size)
        )
        self.bias = nn.Parameter(torch.empty(4 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.kernel)
        nn.init.orthogonal_(self.recurrent_kernel)
        nn.init.zeros_(self.bias)
        # Keras LSTM defaults to unit_forget_bias=True.
        with torch.no_grad():
            self.bias[self.hidden_size : 2 * self.hidden_size].fill_(1.0)

    def forward(self, inputs: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        batch_size, timesteps, _ = inputs.shape
        hidden = inputs.new_zeros((batch_size, self.hidden_size))
        cell = inputs.new_zeros((batch_size, self.hidden_size))

        if self.training and self.recurrent_dropout > 0:
            recurrent_masks = [
                F.dropout(
                    torch.ones_like(hidden),
                    p=self.recurrent_dropout,
                    training=True,
                )
                for _ in range(4)
            ]
        else:
            recurrent_masks = [torch.ones_like(hidden) for _ in range(4)]

        recurrent_chunks = self.recurrent_kernel.chunk(4, dim=1)
        outputs: list[Tensor] = []
        for timestep in range(timesteps):
            projected = inputs[:, timestep] @ self.kernel + self.bias
            input_gate, forget_gate, candidate, output_gate = projected.chunk(4, dim=1)
            input_gate = input_gate + (hidden * recurrent_masks[0]) @ recurrent_chunks[0]
            forget_gate = (
                forget_gate + (hidden * recurrent_masks[1]) @ recurrent_chunks[1]
            )
            candidate = candidate + (hidden * recurrent_masks[2]) @ recurrent_chunks[2]
            output_gate = (
                output_gate + (hidden * recurrent_masks[3]) @ recurrent_chunks[3]
            )

            input_gate = torch.sigmoid(input_gate)
            forget_gate = torch.sigmoid(forget_gate)
            candidate = torch.tanh(candidate)
            output_gate = torch.sigmoid(output_gate)
            next_cell = forget_gate * cell + input_gate * candidate
            next_hidden = output_gate * torch.tanh(next_cell)

            active = (lengths > timestep).unsqueeze(1)
            cell = torch.where(active, next_cell, cell)
            hidden = torch.where(active, next_hidden, hidden)
            outputs.append(torch.where(active, hidden, torch.zeros_like(hidden)))

        return torch.stack(outputs, dim=1), hidden


def masked_batch_norm(inputs: Tensor, lengths: Tensor, layer: nn.BatchNorm1d) -> Tensor:
    """Apply feature-wise batch norm without including zero padding in stats."""

    timesteps = inputs.shape[1]
    valid_mask = (
        torch.arange(timesteps, device=inputs.device).unsqueeze(0)
        < lengths.unsqueeze(1)
    )
    valid_values = inputs[valid_mask]
    if valid_values.numel() == 0:
        raise ValueError("A batch contained no valid timesteps")

    # BatchNorm cannot estimate variance from one value. This only affects
    # pathological one-key, one-sample training batches.
    if layer.training and valid_values.shape[0] == 1:
        normalized = F.batch_norm(
            valid_values,
            layer.running_mean,
            layer.running_var,
            layer.weight,
            layer.bias,
            training=False,
            momentum=0.0,
            eps=layer.eps,
        )
    else:
        normalized = layer(valid_values)

    output = torch.zeros_like(inputs)
    output[valid_mask] = normalized
    return output


class TypeNetEncoder(nn.Module):
    """The complete 200,458-parameter TypeNet encoder from Figure 2."""

    def __init__(
        self,
        input_size: int = PAPER_FEATURE_COUNT,
        hidden_size: int = PAPER_EMBEDDING_DIM,
        dropout: float = 0.5,
        recurrent_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_batch_norm = nn.BatchNorm1d(input_size)
        self.first_lstm = KerasStyleLSTM(
            input_size, hidden_size, recurrent_dropout=recurrent_dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.hidden_batch_norm = nn.BatchNorm1d(hidden_size)
        self.second_lstm = KerasStyleLSTM(
            hidden_size, hidden_size, recurrent_dropout=recurrent_dropout
        )

    def forward(self, inputs: Tensor, lengths: Tensor) -> Tensor:
        if inputs.ndim != 3 or inputs.shape[-1] != PAPER_FEATURE_COUNT:
            raise ValueError(
                f"Expected [batch, time, {PAPER_FEATURE_COUNT}], got {tuple(inputs.shape)}"
            )
        lengths = lengths.clamp(min=1, max=inputs.shape[1])
        normalized_inputs = masked_batch_norm(
            inputs, lengths, self.input_batch_norm
        )
        sequence, _ = self.first_lstm(normalized_inputs, lengths)
        sequence = self.dropout(sequence)
        sequence = masked_batch_norm(
            sequence, lengths, self.hidden_batch_norm
        )
        _, embedding = self.second_lstm(sequence, lengths)
        return embedding


class TypeNetDecider:
    """Distance-threshold verification head fitted at validation EER."""

    def __init__(self, threshold: float | None = None) -> None:
        self.threshold = threshold

    def fit(self, genuine_scores: np.ndarray, impostor_scores: np.ndarray) -> float:
        threshold, _ = equal_error_threshold(genuine_scores, impostor_scores)
        self.threshold = threshold
        return threshold

    def predict(self, scores: np.ndarray | Sequence[float]) -> np.ndarray:
        if self.threshold is None:
            raise RuntimeError("The decider has not been fitted")
        return np.asarray(scores) <= self.threshold

    def evaluate(
        self, genuine_scores: np.ndarray, impostor_scores: np.ndarray
    ) -> VerificationMetrics:
        if self.threshold is None:
            raise RuntimeError("The decider has not been fitted")
        far = float(np.mean(self.predict(impostor_scores)))
        frr = float(np.mean(~self.predict(genuine_scores)))
        return VerificationMetrics(
            threshold=float(self.threshold),
            far=far,
            frr=frr,
            hter=(far + frr) / 2.0,
        )


def triplet_loss(
    anchor: Tensor, positive: Tensor, negative: Tensor, margin: float
) -> Tensor:
    positive_distance_squared = (anchor - positive).square().sum(dim=1)
    negative_distance_squared = (anchor - negative).square().sum(dim=1)
    return F.relu(positive_distance_squared - negative_distance_squared + margin).mean()


def train_one_epoch(
    model: TypeNetEncoder,
    sampler: TripletBatchSampler,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_size: int,
    steps_per_epoch: int,
    margin: float,
    max_gradient_norm: float,
) -> tuple[float, float, float]:
    model.train()
    losses: list[float] = []
    gradient_norms: list[float] = []
    for _ in range(steps_per_epoch):
        (
            anchor,
            anchor_lengths,
            positive,
            positive_lengths,
            negative,
            negative_lengths,
        ) = sampler.sample(batch_size, device)

        # A single encoder call gives all three shared branches the same BN batch.
        combined = torch.cat((anchor, positive, negative), dim=0)
        combined_lengths = torch.cat(
            (anchor_lengths, positive_lengths, negative_lengths), dim=0
        )
        embeddings = model(combined, combined_lengths)
        anchor_embedding, positive_embedding, negative_embedding = embeddings.chunk(3)
        loss = triplet_loss(
            anchor_embedding, positive_embedding, negative_embedding, margin
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=max_gradient_norm
        )
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        gradient_norms.append(float(gradient_norm.detach().cpu()))
    return (
        float(np.mean(losses)),
        float(np.mean(gradient_norms)),
        float(np.max(gradient_norms)),
    )


@torch.inference_mode()
def embed_users(
    model: TypeNetEncoder,
    paths: Sequence[Path],
    sequence_length: int,
    sessions_per_user: int,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, list[str]]:
    """Embed the first 15 valid sessions for each evaluable subject."""

    user_features: list[np.ndarray] = []
    user_lengths: list[np.ndarray] = []
    participant_ids: list[str] = []
    skipped = 0
    for path in paths:
        try:
            user = load_user_sequences(path, sequence_length)
        except (OSError, ValueError, csv.Error):
            skipped += 1
            continue
        if len(user.features) < sessions_per_user:
            skipped += 1
            continue
        user_features.append(user.features[:sessions_per_user])
        user_lengths.append(user.lengths[:sessions_per_user])
        participant_ids.append(user.participant_id)

    if len(user_features) < 2:
        raise RuntimeError(
            "Verification requires at least two users with 15 valid sessions"
        )
    if skipped:
        print(f"Skipped {skipped} evaluation users with incomplete/invalid data.")

    feature_matrix = np.concatenate(user_features, axis=0)
    length_vector = np.concatenate(user_lengths, axis=0)
    embedded_batches: list[np.ndarray] = []
    model.eval()
    for start in range(0, len(feature_matrix), batch_size):
        stop = min(start + batch_size, len(feature_matrix))
        features = torch.from_numpy(feature_matrix[start:stop]).to(device)
        lengths = torch.from_numpy(length_vector[start:stop]).to(device)
        embedded_batches.append(model(features, lengths).cpu().numpy())

    embeddings = np.concatenate(embedded_batches, axis=0)
    embeddings = embeddings.reshape(
        len(participant_ids), sessions_per_user, PAPER_EMBEDDING_DIM
    )
    return embeddings, participant_ids


def pairwise_euclidean(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean matrix without requiring scikit-learn."""

    squared = (
        np.sum(first * first, axis=1, keepdims=True)
        + np.sum(second * second, axis=1, keepdims=True).T
        - 2.0 * (first @ second.T)
    )
    return np.sqrt(np.maximum(squared, 0.0)).astype(np.float32, copy=False)


def verification_scores(
    embeddings: np.ndarray,
    gallery_size: int,
    impostor_query_index: int,
    max_impostors_per_user: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create the paper's genuine and impostor mean-distance scores."""

    users, sessions, _ = embeddings.shape
    if not 1 <= gallery_size <= PAPER_GALLERY_POOL:
        raise ValueError(f"gallery_size must be between 1 and {PAPER_GALLERY_POOL}")
    if sessions < PAPER_QUERY_START + PAPER_QUERY_COUNT:
        raise ValueError("TypeNet evaluation requires 15 sessions per user")
    if not PAPER_QUERY_START <= impostor_query_index < sessions:
        raise ValueError("impostor_query_index must select one of the five queries")

    galleries = embeddings[:, :gallery_size, :]
    genuine_queries = embeddings[
        :, PAPER_QUERY_START : PAPER_QUERY_START + PAPER_QUERY_COUNT, :
    ]
    genuine = np.linalg.norm(
        galleries[:, :, None, :] - genuine_queries[:, None, :, :], axis=-1
    ).mean(axis=1)

    impostor_queries = embeddings[:, impostor_query_index, :]
    score_matrix = np.zeros((users, users), dtype=np.float32)
    for gallery_index in range(gallery_size):
        score_matrix += pairwise_euclidean(
            galleries[:, gallery_index, :], impostor_queries
        )
    score_matrix /= gallery_size

    if max_impostors_per_user <= 0 or max_impostors_per_user >= users - 1:
        off_diagonal = ~np.eye(users, dtype=bool)
        impostor = score_matrix[off_diagonal]
    else:
        rng = np.random.default_rng(seed)
        sampled: list[np.ndarray] = []
        all_users = np.arange(users)
        for claimed_user in range(users):
            candidates = np.delete(all_users, claimed_user)
            selected = rng.choice(
                candidates, size=max_impostors_per_user, replace=False
            )
            sampled.append(score_matrix[claimed_user, selected])
        impostor = np.concatenate(sampled)

    return genuine.reshape(-1), impostor.reshape(-1)


def equal_error_threshold(
    genuine_scores: np.ndarray, impostor_scores: np.ndarray
) -> tuple[float, float]:
    """Return threshold and EER for distance scores accepted when score <= tau."""

    genuine = np.asarray(genuine_scores, dtype=np.float64).reshape(-1)
    impostor = np.asarray(impostor_scores, dtype=np.float64).reshape(-1)
    if genuine.size == 0 or impostor.size == 0:
        raise ValueError("EER requires non-empty genuine and impostor scores")
    if not np.isfinite(genuine).all() or not np.isfinite(impostor).all():
        raise ValueError("EER scores must all be finite")

    genuine.sort()
    impostor.sort()
    thresholds = np.unique(np.concatenate((genuine, impostor)))
    far = np.searchsorted(impostor, thresholds, side="right") / impostor.size
    frr = 1.0 - (
        np.searchsorted(genuine, thresholds, side="right") / genuine.size
    )
    index = int(np.argmin(np.abs(far - frr)))
    eer = float((far[index] + frr[index]) / 2.0)
    return float(thresholds[index]), eer


def split_user_files(
    all_paths: Sequence[Path],
    max_users: int,
    train_users: int,
    validation_users: int,
    test_users: int,
) -> tuple[list[Path], list[Path], list[Path], str]:
    selected = list(all_paths[:max_users] if max_users > 0 else all_paths)
    if len(selected) < 6:
        raise ValueError("At least six users are needed for train/validation/test")

    fixed_split_needed = train_users + validation_users + 2
    if len(selected) >= fixed_split_needed:
        train = selected[:train_users]
        validation = selected[train_users : train_users + validation_users]
        test = selected[train_users + validation_users :]
        if test_users > 0:
            test = test[:test_users]
        mode = (
            f"paper-scale ordering: first {len(train):,} train, next "
            f"{len(validation):,} validation, remaining {len(test):,} test"
        )
    else:
        validation_count = max(2, int(round(len(selected) * 0.15)))
        test_count = max(2, int(round(len(selected) * 0.15)))
        train_count = len(selected) - validation_count - test_count
        if train_count < 2:
            raise ValueError("Subset is too small to allocate at least two train users")
        train = selected[:train_count]
        validation = selected[train_count : train_count + validation_count]
        test = selected[train_count + validation_count :]
        mode = "small-subset 70/15/15 identity-disjoint split"

    if len(test) < 2:
        raise ValueError("At least two test users are required")
    return train, validation, test, mode


def evaluate_split(
    model: TypeNetEncoder,
    paths: Sequence[Path],
    args: argparse.Namespace,
    device: torch.device,
    seed_offset: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    evaluation_paths = list(paths[: args.eval_users] if args.eval_users > 0 else paths)
    embeddings, participant_ids = embed_users(
        model=model,
        paths=evaluation_paths,
        sequence_length=args.sequence_length,
        sessions_per_user=PAPER_SESSIONS_PER_USER,
        batch_size=args.eval_batch_size,
        device=device,
    )
    genuine, impostor = verification_scores(
        embeddings=embeddings,
        gallery_size=args.gallery_size,
        impostor_query_index=args.impostor_query_index,
        max_impostors_per_user=args.max_impostors_per_user,
        seed=args.seed + seed_offset,
    )
    return genuine, impostor, len(participant_ids)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def build_argument_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Train, validate, and test the TypeNet keystroke encoder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=repo_root / "data" / "Keystrokes" / "files",
        help="Directory containing Aalto '*_keystrokes.txt' files.",
    )
    parser.add_argument("--sequence-length", type=int, default=PAPER_SEQUENCE_LENGTH)
    parser.add_argument("--embedding-dim", type=int, default=PAPER_EMBEDDING_DIM)
    parser.add_argument("--margin", type=float, default=1.5)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--recurrent-dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--steps-per-epoch", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.025)
    parser.add_argument(
        "--max-gradient-norm",
        type=float,
        default=1.0,
        help="Clip the global gradient norm to this value before each update.",
    )
    parser.add_argument(
        "--lr-decay-factor",
        type=float,
        default=0.5,
        help="Multiply the learning rate by this factor when validation EER plateaus.",
    )
    parser.add_argument(
        "--lr-decay-patience",
        type=int,
        default=2,
        help="Consecutive validation EER non-improvements before reducing the LR.",
    )
    parser.add_argument(
        "--minimum-learning-rate",
        type=float,
        default=1e-5,
        help="Lower bound for validation-driven learning-rate decay.",
    )
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--train-users", type=int, default=PAPER_TRAIN_USERS)
    parser.add_argument(
        "--validation-users",
        type=int,
        default=1_000,
        help="Held-out identities used only to select the global EER threshold.",
    )
    parser.add_argument(
        "--test-users",
        type=int,
        default=0,
        help="Test identities after validation; zero keeps all remaining users.",
    )
    parser.add_argument(
        "--max-users",
        type=int,
        default=0,
        help="Limit discovered identities; smaller subsets use a 70/15/15 split.",
    )
    parser.add_argument(
        "--eval-users",
        type=int,
        default=1_000,
        help="Maximum identities scored in each validation/test evaluation; zero=all.",
    )
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--gallery-size", type=int, default=10)
    parser.add_argument(
        "--impostor-query-index",
        type=int,
        default=11,
        help="Zero-based session index used as each impostor query (paper code: 11).",
    )
    parser.add_argument(
        "--max-impostors-per-user",
        type=int,
        default=0,
        help="Sample this many impostors per claimed user; zero uses all.",
    )
    parser.add_argument(
        "--cache-users",
        type=int,
        default=2_048,
        help="Preprocessed training users retained in RAM; -1=unbounded, 0=disabled.",
    )
    parser.add_argument("--validate-every", type=int, default=5)
    parser.add_argument(
        "--minimum-epochs",
        type=int,
        default=0,
        help="Do not apply early stopping before this epoch.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Stop after this many validation checks without improvement; zero disables.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional path for the best encoder, threshold, and run configuration.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="Resume model, optimizer, and scheduler state from an interrupted run.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Override scale settings and run one short epoch on 12 real users.",
    )
    return parser


def apply_smoke_test_settings(args: argparse.Namespace) -> None:
    args.max_users = 12
    args.epochs = 1
    args.steps_per_epoch = 1
    args.batch_size = 4
    args.eval_users = 2
    args.eval_batch_size = 30
    args.cache_users = 12
    args.validate_every = 1
    args.minimum_epochs = 0
    args.early_stopping_patience = 0
    args.checkpoint = None


def validate_arguments(args: argparse.Namespace) -> None:
    if args.embedding_dim != PAPER_EMBEDDING_DIM:
        raise ValueError("The paper architecture requires --embedding-dim 128")
    if args.sequence_length < 1:
        raise ValueError("--sequence-length must be positive")
    if args.epochs < 1 or args.steps_per_epoch < 1 or args.batch_size < 1:
        raise ValueError("Epochs, steps-per-epoch, and batch-size must be positive")
    if args.max_gradient_norm <= 0.0:
        raise ValueError("--max-gradient-norm must be positive")
    if not 0.0 < args.lr_decay_factor < 1.0:
        raise ValueError("--lr-decay-factor must be strictly between zero and one")
    if args.lr_decay_patience < 1:
        raise ValueError("--lr-decay-patience must be at least one")
    if not 0.0 <= args.minimum_learning_rate <= args.learning_rate:
        raise ValueError(
            "--minimum-learning-rate must be between zero and --learning-rate"
        )
    if args.validate_every < 1:
        raise ValueError("--validate-every must be positive")
    if not 0 <= args.minimum_epochs <= args.epochs:
        raise ValueError("--minimum-epochs must be between zero and --epochs")
    if args.early_stopping_patience < 0:
        raise ValueError("--early-stopping-patience cannot be negative")
    if args.cache_users < -1:
        raise ValueError("--cache-users must be -1, 0, or a positive integer")


def main() -> None:
    args = build_argument_parser().parse_args()
    if args.smoke_test:
        apply_smoke_test_settings(args)
    validate_arguments(args)
    set_reproducible_seed(args.seed)
    device = resolve_device(args.device)

    all_paths = discover_user_files(args.data_dir)
    train_paths, validation_paths, test_paths, split_description = split_user_files(
        all_paths=all_paths,
        max_users=args.max_users,
        train_users=args.train_users,
        validation_users=args.validation_users,
        test_users=args.test_users,
    )
    print(f"Device: {device}")
    print(f"Found {len(all_paths):,} user files; {split_description}.")

    model = TypeNetEncoder(
        hidden_size=args.embedding_dim,
        dropout=args.dropout,
        recurrent_dropout=args.recurrent_dropout,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"TypeNet trainable parameters: {parameter_count:,}")
    if parameter_count != PAPER_TRAINABLE_PARAMETERS:
        raise RuntimeError(
            f"Architecture mismatch: expected {PAPER_TRAINABLE_PARAMETERS:,} parameters"
        )

    train_store = KeystrokeStore(
        train_paths,
        sequence_length=args.sequence_length,
        cache_users=args.cache_users,
    )
    sampler = TripletBatchSampler(train_store, seed=args.seed)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        eps=args.adam_epsilon,
    )
    learning_rate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_decay_factor,
        # ReduceLROnPlateau's patience=0 means decay on the first bad check.
        patience=args.lr_decay_patience - 1,
        min_lr=args.minimum_learning_rate,
    )

    start_epoch = 1
    best_validation_eer = math.inf
    best_threshold: float | None = None
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    validation_checks_without_improvement = 0
    if args.resume_checkpoint is not None:
        if not args.resume_checkpoint.is_file():
            raise FileNotFoundError(
                f"Resume checkpoint does not exist: {args.resume_checkpoint}"
            )
        resume_state = torch.load(
            args.resume_checkpoint, map_location=device, weights_only=False
        )
        resume_config = resume_state.get("config", {})
        compatibility_fields = (
            "sequence_length",
            "embedding_dim",
            "margin",
            "dropout",
            "recurrent_dropout",
            "steps_per_epoch",
            "batch_size",
            "learning_rate",
            "lr_decay_factor",
            "lr_decay_patience",
            "minimum_learning_rate",
            "max_gradient_norm",
            "gallery_size",
            "validate_every",
        )
        mismatches = [
            field
            for field in compatibility_fields
            if field in resume_config
            and resume_config[field] != getattr(args, field)
        ]
        if mismatches:
            raise ValueError(
                "Resume configuration differs for: " + ", ".join(mismatches)
            )

        model.load_state_dict(resume_state["model_state_dict"])
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        learning_rate_scheduler.load_state_dict(
            resume_state["lr_scheduler_state_dict"]
        )
        best_validation_eer = float(resume_state["validation_eer"])
        best_threshold = float(resume_state["threshold"])
        best_epoch = int(resume_state["best_epoch"])
        best_state = copy.deepcopy(model.state_dict())
        start_epoch = best_epoch + 1
        if start_epoch > args.epochs:
            raise ValueError(
                f"Checkpoint epoch {best_epoch} already reaches --epochs {args.epochs}"
            )

        sampler_rng_state = resume_state.get("sampler_rng_state")
        if sampler_rng_state is not None:
            sampler.rng.bit_generator.state = sampler_rng_state
        else:
            sampler = TripletBatchSampler(
                train_store, seed=args.seed + start_epoch
            )
            print(
                "Checkpoint predates sampler-state saving; "
                f"reseeded triplet sampling with {args.seed + start_epoch}."
            )
        torch_rng_state = resume_state.get("torch_rng_state")
        if torch_rng_state is not None:
            torch.set_rng_state(torch_rng_state.cpu())
        else:
            torch.manual_seed(args.seed + start_epoch)
        print(
            f"Resumed checkpoint epoch {best_epoch}; "
            f"continuing at epoch {start_epoch} with "
            f"lr={optimizer.param_groups[0]['lr']:.8f}."
        )

    for epoch in range(start_epoch, args.epochs + 1):
        loss, mean_gradient_norm, maximum_gradient_norm = train_one_epoch(
            model=model,
            sampler=sampler,
            optimizer=optimizer,
            device=device,
            batch_size=args.batch_size,
            steps_per_epoch=args.steps_per_epoch,
            margin=args.margin,
            max_gradient_norm=args.max_gradient_norm,
        )
        current_learning_rate = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} - "
            f"triplet loss: {loss:.6f} - lr: {current_learning_rate:.8f} - "
            f"gradient norm mean/max: "
            f"{mean_gradient_norm:.4f}/{maximum_gradient_norm:.4f}"
        )

        should_validate = epoch % args.validate_every == 0 or epoch == args.epochs
        if not should_validate:
            continue
        genuine, impostor, validation_user_count = evaluate_split(
            model, validation_paths, args, device, seed_offset=1
        )
        threshold, validation_eer = equal_error_threshold(genuine, impostor)
        print(
            f"  validation ({validation_user_count} users): "
            f"EER={validation_eer * 100:.3f}% tau={threshold:.6f}"
        )
        previous_learning_rate = optimizer.param_groups[0]["lr"]
        learning_rate_scheduler.step(validation_eer)
        updated_learning_rate = optimizer.param_groups[0]["lr"]
        if updated_learning_rate < previous_learning_rate:
            print(
                "  learning-rate decay: "
                f"{previous_learning_rate:.8f} -> {updated_learning_rate:.8f}"
            )
        if validation_eer < best_validation_eer:
            best_validation_eer = validation_eer
            best_threshold = threshold
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            validation_checks_without_improvement = 0
            if args.checkpoint is not None:
                args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": best_state,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "lr_scheduler_state_dict": learning_rate_scheduler.state_dict(),
                        "sampler_rng_state": sampler.rng.bit_generator.state,
                        "torch_rng_state": torch.get_rng_state(),
                        "threshold": best_threshold,
                        "best_epoch": best_epoch,
                        "validation_eer": best_validation_eer,
                        "training_in_progress": True,
                        "config": {
                            key: str(value) if isinstance(value, Path) else value
                            for key, value in vars(args).items()
                        },
                    },
                    args.checkpoint,
                )
        else:
            validation_checks_without_improvement += 1

        should_stop_early = (
            args.early_stopping_patience > 0
            and epoch >= args.minimum_epochs
            and validation_checks_without_improvement
            >= args.early_stopping_patience
        )
        if should_stop_early:
            print(
                "Early stopping: validation EER did not improve for "
                f"{validation_checks_without_improvement} checks."
            )
            break

    if best_state is None or best_threshold is None:
        raise RuntimeError("Training completed without a validation checkpoint")
    model.load_state_dict(best_state)
    decider = TypeNetDecider(best_threshold)

    test_genuine, test_impostor, test_user_count = evaluate_split(
        model, test_paths, args, device, seed_offset=2
    )
    test_metrics = decider.evaluate(test_genuine, test_impostor)
    test_eer_threshold, diagnostic_test_eer = equal_error_threshold(
        test_genuine, test_impostor
    )
    print(
        f"Best validation checkpoint: epoch={best_epoch} "
        f"EER={best_validation_eer * 100:.3f}% tau={best_threshold:.6f}"
    )
    print(
        f"Test ({test_user_count} users) at frozen validation tau: "
        f"FAR={test_metrics.far * 100:.3f}% "
        f"FRR={test_metrics.frr * 100:.3f}% "
        f"HTER={test_metrics.hter * 100:.3f}%"
    )
    print(
        f"Test diagnostic only: EER={diagnostic_test_eer * 100:.3f}% "
        f"test-derived tau={test_eer_threshold:.6f}"
    )

    if args.checkpoint is not None:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_scheduler_state_dict": learning_rate_scheduler.state_dict(),
                "sampler_rng_state": sampler.rng.bit_generator.state,
                "torch_rng_state": torch.get_rng_state(),
                "threshold": best_threshold,
                "best_epoch": best_epoch,
                "validation_eer": best_validation_eer,
                "test_metrics": asdict(test_metrics),
                "training_in_progress": False,
                "config": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
            },
            args.checkpoint,
        )
        print(f"Saved checkpoint: {args.checkpoint}")

    summary = {
        "validation_eer": best_validation_eer,
        "threshold": best_threshold,
        "test_far": test_metrics.far,
        "test_frr": test_metrics.frr,
        "test_hter": test_metrics.hter,
        "diagnostic_test_eer": diagnostic_test_eer,
        "final_learning_rate": optimizer.param_groups[0]["lr"],
    }
    print("RESULT_JSON " + json.dumps(summary, sort_keys=True))
    if args.smoke_test:
        print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
