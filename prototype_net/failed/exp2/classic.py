"""Option 3: frozen TypeNet embeddings plus scikit-learn 4-way classifiers.

TypeNet stays the pretrained biometric encoder. No neural head. Each sklearn
row is one gallery template and one query:

    concat(query, gallery_i, ||query - gallery_i||_2)

Gallery templates are unperturbed. The query is clean (normal) or perturbed.
A query is scored by averaging the G pair probabilities.

    python -m prototype_net.exp2.classic --smoke-test
    python -m prototype_net.exp2.classic --device mps
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper_typenet import nn as typenet  # noqa: E402
from prototype_net.exp2 import model as changehead  # noqa: E402
from prototype_net.perturbation_generator.perturb import (  # noqa: E402
    CLASS_COUNT,
    CLASS_NAMES,
    SEVERITY_NAMES,
    StructuredTimingPerturber,
    run_deterministic_perturbation_checks,
)

MODEL_NAMES = ("rbf_svm", "elasticnet_logreg")
FEATURE_NAME = "concat(q, g_i, ||q-g_i||)"
EMBEDDING_DIM = typenet.PAPER_EMBEDDING_DIM
PAIR_FEATURE_DIM = EMBEDDING_DIM + EMBEDDING_DIM + 1


def pair_gallery_features(
    galleries: np.ndarray, queries: np.ndarray
) -> np.ndarray:
    """Return [users, queries, gallery, 257] = concat(q, g_i, Euclidean)."""

    users, query_count, dim = queries.shape
    gallery_size = galleries.shape[1]
    if galleries.shape[0] != users or galleries.shape[2] != dim:
        raise ValueError("Gallery and query embeddings must align")
    gallery = np.broadcast_to(
        galleries[:, None, :, :], (users, query_count, gallery_size, dim)
    )
    query = np.broadcast_to(
        queries[:, :, None, :], (users, query_count, gallery_size, dim)
    )
    distance = np.linalg.norm(query - gallery, axis=-1, keepdims=True)
    return np.concatenate((query, gallery, distance), axis=-1)


def flatten_pairs(
    pair_features: np.ndarray, class_index: int
) -> tuple[np.ndarray, np.ndarray]:
    users, queries, gallery, dim = pair_features.shape
    features = pair_features.reshape(-1, dim)
    labels = np.full(users * queries * gallery, class_index, dtype=np.int64)
    return features, labels


def subsample_balanced(
    features: np.ndarray,
    labels: np.ndarray,
    max_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if max_samples <= 0 or len(labels) <= max_samples:
        return features, labels
    rng = np.random.default_rng(seed)
    per_class = max(1, max_samples // CLASS_COUNT)
    chosen: list[np.ndarray] = []
    for class_index in range(CLASS_COUNT):
        indices = np.flatnonzero(labels == class_index)
        if indices.size == 0:
            continue
        take = min(per_class, int(indices.size))
        chosen.append(rng.choice(indices, size=take, replace=False))
    order = rng.permutation(np.concatenate(chosen))
    return features[order], labels[order]


def build_elasticnet_logreg(
    C: float, l1_ratio: float, max_iter: int, seed: int
) -> Pipeline:
    classifier = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=l1_ratio,
        C=C,
        max_iter=max_iter,
        tol=1e-3,
        random_state=seed,
    )
    return Pipeline([("scaler", StandardScaler()), ("clf", classifier)])


def build_rbf_svm(C: float, gamma: str | float, seed: int) -> Pipeline:
    classifier = SVC(
        kernel="rbf",
        C=C,
        gamma=gamma,
        probability=True,
        class_weight=None,
        cache_size=1000,
        random_state=seed,
    )
    return Pipeline([("scaler", StandardScaler()), ("clf", classifier)])


@torch.inference_mode()
def embed_sessions(
    encoder: typenet.TypeNetEncoder,
    features: np.ndarray,
    lengths: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    encoder.eval()
    users, sessions, time, feat = features.shape
    flat = features.reshape(-1, time, feat)
    flat_lengths = lengths.reshape(-1)
    batches: list[np.ndarray] = []
    for start in range(0, len(flat), batch_size):
        stop = min(start + batch_size, len(flat))
        embedded = encoder(
            torch.from_numpy(flat[start:stop]).to(device),
            torch.from_numpy(flat_lengths[start:stop]).to(device),
        )
        batches.append(embedded.cpu().numpy())
    return np.concatenate(batches).reshape(users, sessions, EMBEDDING_DIM)


def perturb_sessions(
    session_features: np.ndarray,
    session_lengths: np.ndarray,
    perturber: StructuredTimingPerturber,
    severity_index: int,
    seed: int,
) -> np.ndarray:
    users, session_count, _, _ = session_features.shape
    rng = np.random.default_rng(seed)
    perturbed = np.empty_like(session_features)
    for user_index in range(users):
        profile = perturber.sample_profile(severity_index, rng)
        for session_index in range(session_count):
            perturbed[user_index, session_index] = perturber.perturb(
                session_features[user_index, session_index],
                int(session_lengths[user_index, session_index]),
                severity_index,
                rng,
                profile=profile,
            )
    return perturbed


def cohort_pair_features_by_class(
    encoder: typenet.TypeNetEncoder,
    paths: Sequence[Path],
    perturber: StructuredTimingPerturber,
    sequence_length: int,
    gallery_size: int,
    user_limit: int,
    eval_batch_size: int,
    device: torch.device,
    seed: int,
) -> dict[str, np.ndarray]:
    feature_matrix, length_matrix = changehead.load_complete_user_matrices(
        paths, sequence_length, user_limit
    )
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
    galleries = embed_sessions(
        encoder, gallery_features, gallery_lengths, eval_batch_size, device
    )
    by_class: dict[str, np.ndarray] = {}
    clean_queries = embed_sessions(
        encoder, query_features, query_lengths, eval_batch_size, device
    )
    by_class["normal"] = pair_gallery_features(galleries, clean_queries)
    for severity_index, severity_name in enumerate(SEVERITY_NAMES):
        perturbed = perturb_sessions(
            query_features, query_lengths, perturber, severity_index, seed
        )
        queries = embed_sessions(
            encoder, perturbed, query_lengths, eval_batch_size, device
        )
        by_class[severity_name] = pair_gallery_features(galleries, queries)
    return by_class


def stack_training_pairs(
    features_by_class: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    blocks: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for class_index, name in enumerate(CLASS_NAMES):
        pair_features, pair_labels = flatten_pairs(
            features_by_class[name], class_index
        )
        blocks.append(pair_features)
        labels.append(pair_labels)
    return np.concatenate(blocks), np.concatenate(labels)


def average_pair_proba(
    pipeline: Pipeline, pair_features: np.ndarray, batch_size: int
) -> np.ndarray:
    """Map [users, queries, gallery, dim] to [users, queries, classes]."""

    users, queries, gallery, dim = pair_features.shape
    if dim != PAIR_FEATURE_DIM:
        raise ValueError(f"Expected {PAIR_FEATURE_DIM}-d pair features, got {dim}")
    flat = pair_features.reshape(-1, dim)
    chunks: list[np.ndarray] = []
    for start in range(0, len(flat), batch_size):
        stop = min(start + batch_size, len(flat))
        chunks.append(pipeline.predict_proba(flat[start:stop]))
    return np.concatenate(chunks).reshape(users, queries, gallery, CLASS_COUNT).mean(
        axis=2
    )


def four_way_from_proba(
    proba_by_class: Mapping[str, np.ndarray],
) -> changehead.FourWayAccuracy:
    logits_by_class = {
        name: np.log(np.clip(proba, 1e-12, 1.0))
        for name, proba in proba_by_class.items()
    }
    return changehead._four_way_from_logits(logits_by_class)


def evaluate_models(
    models: Mapping[str, Pipeline],
    features_by_class: Mapping[str, np.ndarray],
    predict_batch_size: int,
) -> dict[str, dict[str, object]]:
    results: dict[str, dict[str, object]] = {}
    for model_name, pipeline in models.items():
        proba_by_class = {
            class_name: average_pair_proba(
                pipeline, pair_features, predict_batch_size
            )
            for class_name, pair_features in features_by_class.items()
        }
        accuracy = four_way_from_proba(proba_by_class)
        normal_scores = 1.0 - proba_by_class["normal"][..., 0]
        impaired_by_severity = {
            name: 1.0 - proba_by_class[name][..., 0] for name in SEVERITY_NAMES
        }
        per_user = changehead.evaluate_all_per_user_metrics(
            normal_scores, impaired_by_severity
        )
        impaired_flat = np.concatenate(
            [impaired_by_severity[name] for name in SEVERITY_NAMES], axis=1
        ).reshape(-1)
        threshold, _eer = typenet.equal_error_threshold(
            normal_scores.reshape(-1), impaired_flat
        )
        results[model_name] = {
            "accuracy": accuracy,
            "per_user": per_user,
            "threshold": float(threshold),
        }
    return results


def print_model_metrics(
    split: str, model_name: str, payload: Mapping[str, object]
) -> None:
    accuracy = payload["accuracy"]
    per_user = payload["per_user"]
    if not isinstance(accuracy, changehead.FourWayAccuracy):
        raise TypeError("accuracy payload is invalid")
    if not isinstance(per_user, Mapping):
        raise TypeError("per-user payload is invalid")
    print(changehead.format_four_way(f"{split} {model_name}", accuracy))
    print(changehead.format_confusion(f"{split} {model_name}", accuracy))
    print(changehead.format_cohort_per_user(f"{split} {model_name} EER", per_user))


def model_result_payload(payload: Mapping[str, object]) -> dict[str, object]:
    accuracy = payload["accuracy"]
    per_user = payload["per_user"]
    if not isinstance(accuracy, changehead.FourWayAccuracy):
        raise TypeError("accuracy payload is invalid")
    if not isinstance(per_user, Mapping):
        raise TypeError("per-user payload is invalid")
    return {
        "accuracy": changehead.four_way_payload(accuracy),
        "per_user": changehead.per_user_metrics_payload(per_user),
        "per_user_eer": per_user["overall"].mean,
        "threshold": payload["threshold"],
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Option 3: frozen TypeNet embeddings plus Euclidean gallery "
            "distance, classified 4-way with RBF SVM and ElasticNet LogReg. "
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
        "--model-output",
        type=Path,
        default=changehead.EXP2_DIR / "classic_gallerydist_v4_64x512_unseen_models.joblib",
    )
    parser.add_argument(
        "--reuse-split-manifest",
        type=Path,
        default=changehead.PROTOCOL_MANIFEST,
        help="Locked v4 train/val/test identities. Required except --smoke-test.",
    )
    parser.add_argument("--sequence-length", type=int, default=typenet.PAPER_SEQUENCE_LENGTH)
    parser.add_argument("--gallery-size", type=int, default=typenet.PAPER_GALLERY_POOL)
    parser.add_argument(
        "--fit-users",
        type=int,
        default=2000,
        help="Complete training users used to fit sklearn. Zero uses the full train role.",
    )
    parser.add_argument(
        "--svm-max-samples",
        type=int,
        default=20_000,
        help="Balanced subsample size for RBF SVM. Zero uses every training pair.",
    )
    parser.add_argument("--logreg-C", type=float, default=1.0)
    parser.add_argument("--logreg-l1-ratio", type=float, default=0.5)
    parser.add_argument("--logreg-max-iter", type=int, default=2000)
    parser.add_argument("--svm-C", type=float, default=1.0)
    parser.add_argument("--svm-gamma", default="scale")
    parser.add_argument("--eval-users", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--predict-batch-size", type=int, default=8192)
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
    args.fit_users = 6
    args.eval_users = 2
    args.eval_batch_size = 30
    args.predict_batch_size = 256
    args.gallery_size = 2
    args.svm_max_samples = 400
    args.logreg_max_iter = 200
    args.model_output = None
    args.reuse_split_manifest = None
    args.device = "cpu"


def validate_arguments(args: argparse.Namespace) -> None:
    if args.sequence_length != typenet.PAPER_SEQUENCE_LENGTH:
        raise ValueError("The pretrained encoder requires M=50")
    if not 1 <= args.gallery_size <= typenet.PAPER_GALLERY_POOL:
        raise ValueError("--gallery-size must be between 1 and 10")
    if args.fit_users < 0 or args.eval_users < 0:
        raise ValueError("User limits cannot be negative")
    if args.svm_max_samples < 0:
        raise ValueError("--svm-max-samples cannot be negative")
    if args.logreg_C <= 0.0 or args.svm_C <= 0.0:
        raise ValueError("Regularization C values must be positive")
    if not 0.0 <= args.logreg_l1_ratio <= 1.0:
        raise ValueError("--logreg-l1-ratio must be in [0, 1]")
    if args.logreg_max_iter < 1:
        raise ValueError("--logreg-max-iter must be positive")
    gamma = args.svm_gamma
    if isinstance(gamma, str) and gamma not in {"scale", "auto"}:
        try:
            args.svm_gamma = float(gamma)
        except ValueError as exc:
            raise ValueError("--svm-gamma must be scale, auto, or a float") from exc
        if args.svm_gamma <= 0.0:
            raise ValueError("--svm-gamma must be positive")
    factors = tuple(args.severity_factors)
    probabilities = tuple(args.pause_probabilities)
    if not (factors[0] < factors[1] < factors[2]):
        raise ValueError("Severity factors must increase mild < moderate < severe")
    if not (
        0.0 <= probabilities[0] <= probabilities[1] <= probabilities[2] <= 1.0
    ):
        raise ValueError("Pause probabilities must increase within [0, 1]")
    if not args.smoke_test and (
        args.reuse_split_manifest is None or not args.reuse_split_manifest.is_file()
    ):
        raise FileNotFoundError(
            "Locked protocol manifest is required unless --smoke-test"
        )


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
        f"test={len(protocol.final_test):,}  "
        f"fit-users={args.fit_users or 'all'}  G={args.gallery_size}"
    )

    encoder = changehead.load_pretrained_encoder(args.pretrained_weights, device)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    print(
        "TypeNet frozen  no-head  objective=classic-ml  "
        f"features={FEATURE_NAME}"
    )

    train_by_class = cohort_pair_features_by_class(
        encoder,
        protocol.train,
        perturber,
        args.sequence_length,
        args.gallery_size,
        args.fit_users,
        args.eval_batch_size,
        device,
        args.seed,
    )
    train_features, train_labels = stack_training_pairs(train_by_class)
    class_counts = np.bincount(train_labels, minlength=CLASS_COUNT)
    print(
        f"train pairs={len(train_labels):,}  "
        f"dim={train_features.shape[1]}  "
        f"class-counts={','.join(str(int(count)) for count in class_counts)}"
    )

    svm_features, svm_labels = subsample_balanced(
        train_features, train_labels, args.svm_max_samples, args.seed
    )
    models = {
        "rbf_svm": build_rbf_svm(args.svm_C, args.svm_gamma, args.seed),
        "elasticnet_logreg": build_elasticnet_logreg(
            args.logreg_C, args.logreg_l1_ratio, args.logreg_max_iter, args.seed
        ),
    }
    print(
        f"fit rbf_svm samples={len(svm_labels):,}  "
        f"C={args.svm_C:g}  gamma={args.svm_gamma}"
    )
    models["rbf_svm"].fit(svm_features, svm_labels)
    print(
        f"fit elasticnet_logreg samples={len(train_labels):,}  "
        f"C={args.logreg_C:g}  l1_ratio={args.logreg_l1_ratio:g}"
    )
    models["elasticnet_logreg"].fit(train_features, train_labels)

    train_pred = {
        name: pipeline.predict(train_features) for name, pipeline in models.items()
    }
    for name, predicted in train_pred.items():
        accuracy = float(np.mean(predicted == train_labels))
        print(f"train {name} pair acc={accuracy:.3f} (4-way, chance=0.25)")

    val_by_class = cohort_pair_features_by_class(
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
    print("val")
    val_metrics = evaluate_models(models, val_by_class, args.predict_batch_size)
    for name in MODEL_NAMES:
        print_model_metrics("val", name, val_metrics[name])

    print("test")
    test_by_class = cohort_pair_features_by_class(
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
    test_metrics = evaluate_models(models, test_by_class, args.predict_batch_size)
    for name in MODEL_NAMES:
        print_model_metrics("test", name, test_metrics[name])

    if args.model_output is not None:
        args.model_output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "models": models,
                "class_names": CLASS_NAMES,
                "feature": FEATURE_NAME,
                "config": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
            },
            args.model_output,
        )
        print(f"saved {args.model_output}")

    result = {
        "train_pair_accuracy": {
            name: float(np.mean(predicted == train_labels))
            for name, predicted in train_pred.items()
        },
        "val": {name: model_result_payload(val_metrics[name]) for name in MODEL_NAMES},
        "test": {
            name: model_result_payload(test_metrics[name]) for name in MODEL_NAMES
        },
        "final_test_fingerprint": fingerprints.get("final_test"),
        "fit_users": args.fit_users,
        "svm_samples": int(len(svm_labels)),
        "logreg_samples": int(len(train_labels)),
        "feature": FEATURE_NAME,
    }
    print("RESULT_JSON " + json.dumps(result, sort_keys=True))
    if args.smoke_test:
        print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
