"""Evidence-calibrated synthetic timing perturbations for prototype-net."""

from prototype_net.perturbation_generator.perturb import (
    CLASS_COUNT,
    CLASS_NAMES,
    SEVERITY_COUNT,
    SEVERITY_NAMES,
    PerturbationConfig,
    PerturbationTrace,
    StructuredTimingPerturber,
    SyntheticImpairmentProfile,
    empty_perturbation_trace,
    run_deterministic_perturbation_checks,
)

__all__ = [
    "CLASS_COUNT",
    "CLASS_NAMES",
    "SEVERITY_COUNT",
    "SEVERITY_NAMES",
    "PerturbationConfig",
    "PerturbationTrace",
    "StructuredTimingPerturber",
    "SyntheticImpairmentProfile",
    "empty_perturbation_trace",
    "run_deterministic_perturbation_checks",
]


# The legacy module above is byte-frozen by existing experiment manifests.
# This opt-in clinical generator lives here so those experiments remain usable.
import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


CLINICAL_GROUPS = ("control", "impaired")
CLINICAL_PAUSE_SECONDS = 2.0


def _positive_covariance(value):
    """Project an estimated covariance onto the positive semidefinite cone."""
    values, vectors = np.linalg.eigh((value + value.T) / 2)
    return (vectors * np.maximum(values, 0.0)) @ vectors.T


@dataclass(frozen=True)
class ClinicalProfile:
    """Persistent participant parameters; never an input to a classifier."""

    group: str
    log_durations: tuple[float, float]


@dataclass
class ClinicalCalibration:
    """Hierarchical duration priors fitted to clinical participant/task rows.

    The two coordinates are log mean fluent-burst duration and log mean
    pause excess above two seconds. Mean pause duration is derived from
    burst duration and pause-time fraction under an alternating renewal model.
    This model assumption is explicit; individual pause/burst dwell-time
    distributions are not available in the participant-level source.

    `participant_ids` in from_csv is the clinical training partition, not
    the synthetic/Aalto participant partition. Always supply it for evaluation.
    """

    groups: dict
    participant_ids: tuple[str, ...]
    source_sha256: str
    excluded_rows: tuple[dict, ...]
    records: tuple[dict, ...]

    @classmethod
    def from_csv(cls, path, participant_ids=None):
        path = Path(path)
        allowed = None if participant_ids is None else set(map(str, participant_ids))
        by_person, excluded, records = {}, [], []
        with path.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream, delimiter=";"):
                uid = row["participant_code"]
                if allowed is not None and uid not in allowed:
                    continue
                if row["group2"] == "gezonde ouderen":
                    group = "control"
                elif row["group2"] == "CI":
                    group = "impaired"
                else:
                    raise ValueError(f"Unknown clinical group for {uid}")
                try:
                    fluent = float(row["duration_of_P_bursts_s_PB2000"])
                    fraction = float(row["proportion_of_pause_time_PT2000"])
                    burst_rate = float(row["number_of_P_bursts_per_minute_PB2000"])
                    if not np.isfinite([fluent, fraction, burst_rate]).all():
                        raise ValueError("Nonfinite measurements")
                    if fluent <= 0 or not 0 < fraction < 1 or burst_rate <= 0:
                        raise ValueError("Invalid measurements")
                    pause = fluent * fraction / (1 - fraction)
                    if pause <= CLINICAL_PAUSE_SECONDS:
                        raise ValueError("Mean pause incompatible with the threshold")
                except ValueError as error:
                    excluded.append(dict(participant=uid, task=row["task"], reason=str(error)))
                    continue
                vector = np.log([fluent, pause - CLINICAL_PAUSE_SECONDS])
                entry = by_person.setdefault(uid, dict(group=group, vectors=[]))
                if entry["group"] != group:
                    raise ValueError(f"Clinical group changes within participant {uid}")
                entry["vectors"].append(vector)
                records.append(dict(participant=uid, group=group, task=row["task"],
                                    fluent_seconds=fluent, pause_fraction=fraction,
                                    bursts_per_minute=burst_rate))
        groups = {}
        for group in CLINICAL_GROUPS:
            people = [np.asarray(p["vectors"]) for p in by_person.values() if p["group"] == group]
            if len(people) < 3:
                raise ValueError(f"At least three usable clinical participants required in {group}")
            means = np.stack([p.mean(0) for p in people])
            residuals = np.concatenate([p - p.mean(0) for p in people])
            degrees = sum(len(p) - 1 for p in people)
            within = residuals.T @ residuals / degrees if degrees else np.zeros((2, 2))
            # Remove finite-session measurement variance from between-person variance.
            between_raw = np.cov(means, rowvar=False) - within * np.mean([1 / len(p) for p in people])
            between = _positive_covariance(between_raw)
            groups[group] = dict(mean=means.mean(0), between_cov=between,
                                 session_cov=_positive_covariance(within),
                                 participants=len(people), tasks=sum(map(len, people)),
                                 covariance_projection=float(np.linalg.norm(between - between_raw)))
        return cls(groups, tuple(sorted(by_person)), hashlib.sha256(path.read_bytes()).hexdigest(),
                   tuple(excluded), tuple(records))

    def sample_profile(self, group, rng):
        if group not in CLINICAL_GROUPS:
            raise ValueError(f"group must be one of {CLINICAL_GROUPS}; these are not severity stages")
        parameters = self.groups[group]
        vector = rng.multivariate_normal(parameters["mean"], parameters["between_cov"])
        return ClinicalProfile(group, tuple(map(float, vector)))

    def sample_session_durations(self, profile, rng):
        if profile.group not in self.groups:
            raise ValueError("Profile group is absent from this calibration")
        vector = np.asarray(profile.log_durations, dtype=float)
        if vector.shape != (2,) or not np.isfinite(vector).all():
            raise ValueError("Profile must contain two finite log durations")
        vector = vector + rng.multivariate_normal(np.zeros(2), self.groups[profile.group]["session_cov"])
        fluent, excess = np.exp(vector)
        if not np.isfinite([fluent, excess]).all() or fluent <= 0 or excess <= 0:
            raise ValueError("Invalid sampled session durations")
        return float(fluent), float(CLINICAL_PAUSE_SECONDS + excess)


class ClinicalTimingGenerator:
    """Opt-in clinical pause/burst generator for whole, contiguous sessions.

    Both groups receive the same procedure. Hold times and key identity are
    retained; source intervals >2s are replaced with fluent intervals sampled
    from this source session before a new pause process is generated. This is
    a resynthesis of planning pauses, not an additive impairment perturbation.

    Generate BEFORE windowing. Do not concatenate independent sentences and
    pretend their omitted boundaries form a continuous session. Supplied real
    boundary intervals are processed like other intervals. Bursts are timed in
    seconds, not numbers of keys, and the process runs through the whole input.

    Exponential fluent dwell times and shifted-exponential pause dwell times
    are maximum-entropy assumptions given the fitted means; their shapes have
    not been clinically identified. Optional gamma shapes allow sensitivity
    tests without claiming additional clinical calibration. Key/word context
    multipliers are deliberately not inferred from aggregate participant data.
    """

    def __init__(self, calibration, *, burst_shape=1.0, pause_shape=1.0):
        if not isinstance(calibration, ClinicalCalibration):
            raise TypeError("Expected a ClinicalCalibration")
        if not np.isfinite([burst_shape, pause_shape]).all() or min(burst_shape, pause_shape) <= 0:
            raise ValueError("Dwell-time shapes must be finite and positive")
        self.calibration = calibration
        self.burst_shape = float(burst_shape)
        self.pause_shape = float(pause_shape)

    def generate_session(self, features, profile, rng):
        source = np.asarray(features)
        if source.ndim != 2 or source.shape[1] != 5 or len(source) < 3:
            raise ValueError("Provide an unpadded, contiguous [events, 5] session with at least three keys")
        if not np.isfinite(source).all() or np.any(source[:, 0] < 0) or np.any(source[:-1, 2] < 0):
            raise ValueError("Hold and press-to-press intervals must be finite and nonnegative")
        fluent_mean, pause_mean = self.calibration.sample_session_durations(profile, rng)
        result = source.astype(np.float32, copy=True)
        gaps = result[:-1, 2].astype(float)
        pool = gaps[(gaps > 0) & (gaps <= CLINICAL_PAUSE_SECONDS)]
        if not len(pool):
            raise ValueError("Source session has no positive fluent intervals at or below two seconds")
        long = gaps > CLINICAL_PAUSE_SECONDS
        gaps[long] = rng.choice(pool, int(long.sum()), replace=True)
        remaining = float(rng.gamma(self.burst_shape, fluent_mean / self.burst_shape))
        for index, gap in enumerate(gaps):
            if remaining <= 0:
                gaps[index] = CLINICAL_PAUSE_SECONDS + rng.gamma(
                    self.pause_shape, (pause_mean - CLINICAL_PAUSE_SECONDS) / self.pause_shape)
                remaining = float(rng.gamma(self.burst_shape, fluent_mean / self.burst_shape))
            else:
                remaining -= gap
        result[:-1, 2] = gaps
        result[:-1, 1] = gaps - result[:-1, 0]
        result[:-1, 3] = result[:-1, 1] + result[1:, 0]
        result[-1, 1:4] = 0
        if not np.isfinite(result).all():
            raise ValueError("Generated timing overflow")
        return result


def measure_clinical_session(features):
    """Observable pause/burst metrics; no profile, masks, or clean twin inputs.

    A pause is a press-to-press interval strictly above two seconds. Report
    first-to-last-key time, not unknown task-start/finish delays. Only interior,
    complete production bursts contribute to mean duration. The initial and
    terminal production intervals are censored and reported separately.
    This explicit event convention approximates, but cannot exactly reproduce,
    the unavailable raw Inputlog task-boundary convention.
    """
    x = np.asarray(features)
    if x.ndim != 2 or x.shape[1] != 5 or len(x) < 3 or not np.isfinite(x).all():
        raise ValueError("Expected a finite unpadded [events, 5] session")
    gaps = x[:-1, 2].astype(float)
    if np.any(gaps < 0) or gaps.sum() <= 0:
        raise ValueError("Nonnegative intervals and positive observation time required")
    mask = gaps > CLINICAL_PAUSE_SECONDS
    where = np.flatnonzero(mask)
    bursts = [float(gaps[left + 1:right].sum()) for left, right in zip(where[:-1], where[1:])]
    total = float(gaps.sum())
    return dict(observation_seconds=total, pause_count=int(mask.sum()),
                pause_time_fraction=float(gaps[mask].sum() / total),
                mean_pause_seconds=float(gaps[mask].mean()) if mask.any() else None,
                mean_complete_burst_seconds=float(np.mean(bursts)) if bursts else None,
                complete_burst_count=len(bursts),
                pause_onsets_per_minute=float(mask.sum() * 60 / total),
                censored_edge_seconds=[float(gaps[:where[0]].sum()), float(gaps[where[-1] + 1:].sum())]
                if len(where) else [total],
                threshold_seconds=CLINICAL_PAUSE_SECONDS)


__all__ += ["CLINICAL_GROUPS", "CLINICAL_PAUSE_SECONDS", "ClinicalProfile",
            "ClinicalCalibration", "ClinicalTimingGenerator", "measure_clinical_session"]


class ClinicalDeviationGenerator:
    """Positive-only deviations relative to untouched personal enrollment.

    Cognitive extra-pause intensity uses the fitted control/impaired fluent
    duration ratio and a Jeffreys-smoothed enrollment pause hazard (0.5 events).
    That smoothing and transferring the clinical ratio to Aalto are assumptions.
    Positive examples condition on at least one injected cognitive pause; they
    represent an observed intervention, not every person with a diagnosis.
    Existing pauses are never removed. Dwell means come from impaired profiles.
    All ten windows share one profile. Missing between-window time is not invented.

    Motor-only and combined examples retain the existing Parkinson's-informed
    motor priors, with equal sampling over the three former stress amplitudes.
    Those amplitudes are not clinically validated cognitive severity labels.
    """

    mechanisms = ("cognitive", "motor", "combined")

    def __init__(self, calibration, *, pause_shape=1.0, hazard_multiplier=1.0):
        if not isinstance(calibration, ClinicalCalibration):
            raise TypeError("Expected clinical calibration")
        if not np.isfinite([pause_shape, hazard_multiplier]).all() or min(pause_shape, hazard_multiplier) <= 0:
            raise ValueError("Positive finite sensitivity parameters required")
        self.calibration = calibration
        self.pause_shape = float(pause_shape)
        self.hazard_multiplier = float(hazard_multiplier)
        self.rate_ratio = float(np.exp(calibration.groups['control']['mean'][0] -
                                       calibration.groups['impaired']['mean'][0]))
        if self.rate_ratio <= 1:
            raise ValueError("Calibration does not support increased pause onset rate")

    def perturb(self, gallery, gallery_lengths, query, query_lengths, mechanism, rng):
        if mechanism not in self.mechanisms:
            raise ValueError("Unknown positive mechanism")
        for x, lengths in ((gallery, gallery_lengths), (query, query_lengths)):
            if np.shape(x) != (10, 50, 5) or np.shape(lengths) != (10,):
                raise ValueError("Ten 50-key windows per side required")
            if not np.isfinite(x).all() or not np.issubdtype(np.asarray(lengths).dtype, np.integer):
                raise ValueError("Finite windows and integer lengths required")
            if np.any(np.asarray(lengths) < 6) or np.any(np.asarray(lengths) > 50):
                raise ValueError("Window lengths must be 6..50")
        result = np.asarray(query, dtype=np.float32).copy()
        details = dict(mechanism=mechanism, injected_pauses=0, motor_stress_index=None)
        if mechanism in ('motor', 'combined'):
            # Disable the old cognitive channel; this branch owns motor effects only.
            from dataclasses import replace
            engine = StructuredTimingPerturber(replace(PerturbationConfig(), pause_probabilities=(0., 0., 0.)))
            stress = int(rng.integers(SEVERITY_COUNT))
            profile = engine.sample_profile(stress, rng)
            result = np.stack([engine.perturb(x, int(n), stress, rng, profile=profile)
                               for x, n in zip(result, query_lengths)])
            for x, original, n in zip(result, query, query_lengths):
                n = int(n)
                existing = original[:n-1, 2] > 2
                x[:n-1, 2][existing] = np.maximum(x[:n-1, 2][existing], original[:n-1, 2][existing])
                x[:n-1, 1] = x[:n-1, 2] - x[:n-1, 0]
                x[:n-1, 3] = x[:n-1, 1] + x[1:n, 0]
            details['motor_stress_index'] = stress
        if mechanism in ('cognitive', 'combined'):
            baseline = np.concatenate([x[:int(n)-1, 2] for x, n in zip(gallery, gallery_lengths)])
            active = float(baseline[(baseline > 0) & (baseline <= 2)].sum())
            if active <= 0:
                raise ValueError("Enrollment has no observed fluent exposure")
            hazard = (np.count_nonzero(baseline > 2) + .5) / active
            # Locations depend on the original query, so motor changes cannot create candidate sites.
            sites = [(w, t, float(x[t, 2])) for w, (x, n) in enumerate(zip(query, query_lengths))
                     for t in range(int(n)-1) if 0 < x[t, 2] <= 2]
            if not sites:
                raise ValueError("Query has no fluent transition on which to inject a pause")
            weights = np.array([s[2] for s in sites])
            expected = hazard * (self.rate_ratio - 1) * weights.sum() * self.hazard_multiplier
            # Exact zero-truncated Poisson inversion; no strength adjustment based on classification.
            from scipy.stats import poisson
            lower = float(np.exp(-expected))
            quantile = min(lower + (1-lower) * rng.random(), np.nextafter(1., 0.))
            count = 1 if expected < 1e-8 else int(poisson.ppf(quantile, expected))
            count = min(max(count, 1), len(sites))
            chosen = rng.choice(len(sites), count, replace=False, p=weights/weights.sum())
            profile = self.calibration.sample_profile('impaired', rng)
            _, pause_mean = self.calibration.sample_session_durations(profile, rng)
            delays = 2 + rng.gamma(self.pause_shape, (pause_mean-2)/self.pause_shape, count)
            for index, delay in zip(chosen, delays):
                w, t, _ = sites[index]
                result[w, t, 2] += delay
            for x, n in zip(result, query_lengths):
                n = int(n)
                x[:n-1, 1] = x[:n-1, 2] - x[:n-1, 0]
                x[:n-1, 3] = x[:n-1, 1] + x[1:n, 0]
            details.update(injected_pauses=count, expected_unconditioned_pauses=float(expected),
                           enrollment_hazard=float(hazard), clinical_rate_ratio=self.rate_ratio)
        if not np.isfinite(result).all() or np.array_equal(result, query):
            raise ValueError("Positive intervention did not produce a finite changed query")
        return result, details


__all__ += ["ClinicalDeviationGenerator"]
