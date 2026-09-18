from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper_typenet import nn as typenet  # noqa: E402

SEVERITY_NAMES = ("mild", "moderate", "severe")
SEVERITY_COUNT = len(SEVERITY_NAMES)
CLASS_NAMES = ("normal",) + SEVERITY_NAMES
CLASS_COUNT = len(CLASS_NAMES)


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
class PerturbationTrace:
    """Dense generator labels aligned to a TypeNet [T, 5] session.

    Pause channels are defined on forward transitions (key t → t+1) and are
    zero at the final valid key and in padding. Hold log-ratio is defined on
    every valid key.
    """

    features: np.ndarray
    pause_mask: np.ndarray
    pause_delay: np.ndarray
    hold_log_ratio: np.ndarray


def empty_perturbation_trace(features: np.ndarray) -> PerturbationTrace:
    """Clean-session trace: no pauses, no hold residual."""

    time = int(np.asarray(features).shape[0])
    zeros = np.zeros(time, dtype=np.float32)
    return PerturbationTrace(
        features=np.asarray(features, dtype=np.float32).copy(),
        pause_mask=zeros.copy(),
        pause_delay=zeros.copy(),
        hold_log_ratio=zeros.copy(),
    )


class StructuredTimingPerturber:
    """Severity-conditioned timing perturbations for TypeNet [T, 5] sessions."""

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
        return self.perturb_trace(
            features, length, severity_index, rng, profile=profile
        ).features

    def perturb_trace(
        self,
        features: np.ndarray,
        length: int,
        severity_index: int,
        rng: np.random.Generator,
        profile: SyntheticImpairmentProfile | None = None,
    ) -> PerturbationTrace:
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
        original_hold = perturbed[:valid_length, 0].copy()
        time = len(perturbed)
        pause_mask_full = np.zeros(time, dtype=np.float32)
        pause_delay_full = np.zeros(time, dtype=np.float32)
        hold_log_ratio = np.zeros(time, dtype=np.float32)
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
            original_hold
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
            applied_delay = (pause_mask * pause_delays).astype(np.float32)
            press_latency += applied_delay
            pause_mask_full[:transition_length] = pause_mask.astype(np.float32)
            pause_delay_full[:transition_length] = applied_delay

        hold_floor = np.maximum(original_hold, self.config.minimum_timing_scale)
        hold_log_ratio[:valid_length] = np.log(
            np.maximum(hold_latency, self.config.minimum_timing_scale) / hold_floor
        ).astype(np.float32)

        perturbed[:valid_length, 0] = hold_latency
        if transition_length:
            inter_latency = press_latency - hold_latency[:transition_length]
            perturbed[:transition_length, 1] = inter_latency
            perturbed[:transition_length, 2] = press_latency
            perturbed[:transition_length, 3] = (
                inter_latency + hold_latency[1:valid_length]
            )

        perturbed[valid_length - 1, 1:4] = 0.0
        if valid_length < len(perturbed):
            perturbed[valid_length:] = 0.0
            pause_mask_full[valid_length:] = 0.0
            pause_delay_full[valid_length:] = 0.0
            hold_log_ratio[valid_length:] = 0.0
        pause_mask_full[valid_length - 1] = 0.0
        pause_delay_full[valid_length - 1] = 0.0
        return PerturbationTrace(
            features=perturbed,
            pause_mask=pause_mask_full,
            pause_delay=pause_delay_full,
            hold_log_ratio=hold_log_ratio,
        )


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
        trace = perturber.perturb_trace(
            clean,
            length,
            severity_index,
            np.random.default_rng(1234),
        )
        np.testing.assert_array_equal(trace.features, perturbed)
        if trace.pause_mask.shape != (sequence_length,):
            raise AssertionError("Pause mask must be aligned to the session length")
        if np.any(trace.pause_mask[length - 1 :]):
            raise AssertionError("The final key cannot carry a forward pause")
        if np.any(trace.pause_delay[trace.pause_mask < 0.5] != 0.0):
            raise AssertionError("Pause delay must be zero where the pause mask is off")
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
