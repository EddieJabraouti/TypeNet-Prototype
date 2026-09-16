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
