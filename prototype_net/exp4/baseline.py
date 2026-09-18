"""Personal-baseline tracking for the shared raw-embedding PD classifier."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import joblib
import torch
from threadpoolctl import threadpool_limits
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.metrics import roc_auc_score

from prototype_net.exp4.model import (ROOT, ROLES, SEED, TypingClassifier, fit_model,
                                      predict, metrics, sha256, source_hashes, write_json)


class PersonalMonitor(TypingClassifier):
    def __init__(self, run, participant_id, family="xgboost", batch_size=50):
        super().__init__(run, family, batch_size)
        self.participant_id = str(participant_id)
        self.baseline = None
        self.last_features = None
        self.state = dict(participant_id=self.participant_id, family=family, batch_size=batch_size,
                          fingerprint=self.fingerprint, batches=[], event_ids=[], last_timestamp_ms=None)

    def observe(self, payload):
        if str(payload["participant_id"]) != self.participant_id:
            raise ValueError("Participant mismatch")
        batch_id, timestamp = str(payload["batch_id"]), float(payload["timestamp_ms"])
        if batch_id in self.state["batches"]:
            raise ValueError("Batch already processed")
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("Invalid batch timestamp")
        previous = self.state["last_timestamp_ms"]
        if previous is not None and timestamp <= previous:
            raise ValueError("Batch must follow the previous batch")
        features, identifiers, windows = self.encode(payload["events"])
        if set(identifiers) & set(self.state["event_ids"]):
            raise ValueError("Events overlap an earlier batch")
        if features is None:
            if windows > self.batch_size:
                raise ValueError(f"Split the input into batches of {self.batch_size} complete windows")
            return dict(status="insufficient_data", participant_id=self.participant_id, batch_id=batch_id,
                        windows=windows, required_windows=self.batch_size)
        probability = self.probabilities(features)
        score = float(probability.mean())
        result = dict(status="baseline_ready" if self.baseline is None else "scored", participant_id=self.participant_id,
            batch_id=batch_id, timestamp_ms=timestamp, windows=windows, keystrokes=windows*50,
            embedding_shape=[windows, 128], classifier_input_shape=list(features.shape), family=self.family,
            model_count=len(self.models), pd_score=score, predicted_class=int(score >= .5),
            model_score_sd=float(probability.mean(1).std(ddof=1)))
        if self.baseline is None:
            self.baseline = features.copy()
            self.state.update(baseline_batch_id=batch_id, baseline_pd_score=score)
        else:
            result.update(baseline_batch_id=self.state["baseline_batch_id"], baseline_pd_score=self.state["baseline_pd_score"],
                          pd_score_change=score-self.state["baseline_pd_score"],
                          **baseline_distance(self.baseline[:, :128], features[:, :128]))
        self.state["batches"].append(batch_id)
        self.state["event_ids"].extend(identifiers)
        self.state["last_timestamp_ms"] = timestamp
        self.last_features = features
        return result


    def save(self, path):
        path = Path(path)
        if self.baseline is None:
            raise ValueError("No baseline to save")
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, baseline=self.baseline, metadata=np.array(json.dumps(self.state, allow_nan=False)))
        temporary.replace(path)


    def restore(self, path):
        with np.load(path, allow_pickle=False) as cache:
            baseline = cache["baseline"].copy()
            state = json.loads(str(cache["metadata"]))
        for key in ("participant_id", "family", "batch_size", "fingerprint"):
            if state[key] != self.state[key]:
                raise ValueError(f"State mismatch: {key}")
        if baseline.shape != (self.batch_size, self.features) or not np.isfinite(baseline).all():
            raise ValueError("Invalid baseline matrix")
        self.baseline, self.state = baseline, state


def baseline_distance(baseline, current):
    baseline, current = np.asarray(baseline, dtype=np.float64), np.asarray(current, dtype=np.float64)
    if (baseline.ndim != 2 or current.ndim != 2 or baseline.shape[1] != current.shape[1]
            or min(len(baseline), len(current)) < 2):
        raise ValueError("Expected matrices with matching coordinates and at least two embeddings")
    if not np.isfinite(baseline).all() or not np.isfinite(current).all():
        raise ValueError("Invalid embeddings")
    bb = np.sum((baseline[:, None] - baseline[None, :])**2, axis=-1)
    cc = np.sum((current[:, None] - current[None, :])**2, axis=-1)
    bc = np.sum((baseline[:, None] - current[None, :])**2, axis=-1)
    off_diagonal = bb[~np.eye(len(baseline), dtype=bool)]
    within = float(np.sqrt(off_diagonal).mean())
    between = float(np.sqrt(bc).mean())
    positive = off_diagonal[off_diagonal > 0]
    bandwidth = float(np.median(positive)) if len(positive) else 1e-12
    mmd = float(np.exp(-bb/bandwidth).mean() + np.exp(-cc/bandwidth).mean() - 2*np.exp(-bc/bandwidth).mean())
    return dict(embedding_distance=between, baseline_within_distance=within,
                embedding_distance_ratio=between/within if within > 0 else None,
                mmd2=max(0., mmd), kernel_bandwidth_squared=bandwidth/2,
                change_interpretation="Descriptive typing change; no clinical change threshold fitted")


class BaselineMonitor:
    """Fixed personal reference alongside a frozen population PD classifier."""
    def __init__(self, engine, participant_id, context_id, baseline_batches=1):
        if not str(participant_id).strip() or not str(context_id).strip():
            raise ValueError("Participant and keyboard/context identifiers are required")
        if type(baseline_batches) is not int or baseline_batches < 1 or engine.batch_size != 6:
            raise ValueError("Use a positive enrollment count and six 50-key sequences per batch")
        if engine.features != 128:
            raise ValueError("BaselineMonitor requires the raw-embedding classifier")
        self.engine = engine
        self.baseline = np.empty((0, 6, 128), dtype=np.float32)
        self.baseline_scores = np.empty((0, len(engine.models)), dtype=np.float64)
        self.state = dict(version=1, participant_id=str(participant_id), context_id=str(context_id),
            baseline_batches=baseline_batches, fingerprint=engine.fingerprint, family=engine.family,
            batches=[], event_ids=[], last_sequence_number=None, positive_streak=0, history=[])

    def observe(self, payload):
        for field in ("participant_id", "context_id"):
            if str(payload[field]) != self.state[field]:
                raise ValueError(f"{field} mismatch")
        number, batch_id = payload["sequence_number"], str(payload["batch_id"])
        previous = self.state["last_sequence_number"]
        if type(number) is not int or number < 0 or (previous is not None and number <= previous):
            raise ValueError("Sequence number must increase in acquisition order")
        if not batch_id.strip() or batch_id in self.state["batches"]:
            raise ValueError("Batch identifier is empty or already processed")
        if len(payload["events"]) != 300:
            raise ValueError("Supply exactly 300 eligible keys in complete 50-key session windows")
        features, identifiers, windows = self.engine.encode(payload["events"])
        if features is None or windows != 6:
            raise ValueError("Expected six complete 50-key windows without discarded keys")
        if set(identifiers) & set(self.state["event_ids"]):
            raise ValueError("Events overlap an earlier batch")
        scores = self.engine.probabilities(features).mean(axis=1).astype(np.float64)
        return self._accept(batch_id, number, identifiers, features[:, :128], scores)

    def _accept(self, batch_id, number, identifiers, embedded, scores):
        enrolling = len(self.baseline) < self.state["baseline_batches"]
        score = float(scores.mean())
        result = dict(participant_id=self.state["participant_id"], batch_id=batch_id,
            sequence_number=number, status="enrolling" if enrolling else "scored",
            enrollment_batches=self.state["baseline_batches"], keystrokes=300,
            embedding_shape=[6, 128], classifier_input_shape=[6, 128],
            pd_score=score, model_score_sd=float(scores.std(ddof=1)) if len(scores) > 1 else 0.,
            target="PD-associated typing versus control", threshold=.5,
            classifier_personalized=False)
        if enrolling:
            self.baseline = np.concatenate([self.baseline, embedded[None]])
            self.baseline_scores = np.concatenate([self.baseline_scores, scores[None]])
            result["status"] = "baseline_ready" if len(self.baseline) == self.state["baseline_batches"] else "enrolling"
        else:
            base_score = float(self.baseline_scores.mean())
            positive = score >= .5
            streak = self.state["positive_streak"] + 1 if positive else 0
            result.update(flag=int(positive), label="pd_pattern" if positive else "control_pattern",
                baseline_pd_score=base_score, pd_score_change=score-base_score,
                positive_streak=streak, repeated_positive=streak >= 2,
                **baseline_distance(self.baseline.reshape(-1, 128), embedded))
            self.state["positive_streak"] = streak
            self.state["history"].append(dict(batch_id=batch_id, sequence_number=number,
                pd_score=score, flag=int(positive), pd_score_change=score-base_score,
                mmd2=result["mmd2"], positive_streak=streak))
        result["baseline_batches_stored"] = len(self.baseline)
        self.state["batches"].append(batch_id)
        self.state["event_ids"].extend(identifiers)
        self.state["last_sequence_number"] = number
        return result

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, baseline=self.baseline, scores=self.baseline_scores,
                metadata=np.array(json.dumps(self.state, allow_nan=False)))
        temporary.replace(path)

    def restore(self, path):
        with np.load(path, allow_pickle=False) as cache:
            baseline, scores = cache["baseline"].copy(), cache["scores"].copy()
            state = json.loads(str(cache["metadata"]))
        for key in ("version", "participant_id", "context_id", "baseline_batches", "fingerprint", "family"):
            if state[key] != self.state[key]:
                raise ValueError(f"State mismatch: {key}")
        count = len(baseline)
        if (baseline.shape != (count, 6, 128) or count > state["baseline_batches"]
                or scores.shape != (count, len(self.engine.models))
                or not np.isfinite(baseline).all() or not np.isfinite(scores).all()
                or np.any((scores < 0) | (scores > 1))):
            raise ValueError("Invalid saved baseline")
        if (len(set(state["batches"])) != len(state["batches"])
                or len(set(state["event_ids"])) != len(state["event_ids"])
                or len(state["event_ids"]) != 300*len(state["batches"])
                or count != min(len(state["batches"]), state["baseline_batches"])
                or len(state["history"]) != len(state["batches"])-count):
            raise ValueError("Inconsistent saved observation history")
        self.baseline, self.baseline_scores, self.state = baseline, scores, state

    def export(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            joblib.dump(dict(variant="baseline", engine=self.engine._bundle(),
                baseline_batches=self.state["baseline_batches"]), stream, compress=3)

    @classmethod
    def load(cls, path, participant_id, context_id):
        bundle = joblib.load(path)
        if bundle["variant"] != "baseline":
            raise ValueError("Expected a baseline variant artifact")
        return cls(TypingClassifier._from_bundle(bundle["engine"]), participant_id,
                   context_id, bundle["baseline_batches"])


def replay_baselines(out, baseline):
    source = Path(__file__).parent / "runs/baseline_embeddings"
    reg = json.loads((source / "registration.json").read_text())
    dataset = source / "300/dataset.npz"
    if sha256(dataset) != reg["scenarios"]["300"]["dataset_sha256"] or sha256(source / "events.npz") != reg["events_sha256"]:
        raise ValueError("Replay data changed")
    model_reg = json.loads((baseline / "registration.json").read_text())
    if (model_reg["encoder_sha256"] != reg["encoder_sha256"] or model_reg["threshold"] != .5
            or model_reg["features"] != 128 or model_reg["augmentation"] != [0]):
        raise ValueError("Model and replay protocol mismatch")
    engine = TypingClassifier(baseline)
    with np.load(dataset, allow_pickle=False) as cache:
        data = {key: cache[key] for key in ("x", "raw", "lineage", "owner", "batch", "split", "y")}
    with np.load(source / "events.npz", allow_pickle=False) as cache:
        events = cache["events"]
    if len(np.unique(data["lineage"])) != data["lineage"].size:
        raise ValueError("Repeated events across replay batches")
    out.mkdir(parents=True, exist_ok=False)
    registration = dict(classifier_run=str(baseline.resolve()),
        classifier_freeze_sha256=sha256(baseline / "freeze.json"), encoder_sha256=reg["encoder_sha256"],
        dataset_sha256=sha256(dataset), events_sha256=sha256(source / "events.npz"),
        baseline_batches=[1, 3], keys_per_batch=300, threshold=.5, synthetic=False,
        classifier="Existing 25 XGBoost folds; 128 TypeNet coordinates only",
        decision="Mean model score >= .5; baseline distance and score change do not alter the flag",
        baseline="Fixed after enrollment; first one or three acquisition-ordered batches",
        time="Recording/acquisition order, not simulated days or verified symptom transitions",
        sources=["https://jmlr.org/papers/v13/gretton12a.html"],
        source_hashes={**source_hashes(), "paper_typenet/nn.py": sha256(ROOT / "paper_typenet/nn.py")})
    write_json(out / "registration.json", registration)
    with zipfile.ZipFile(out / "source.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in registration["source_hashes"]:
            archive.write(ROOT / name, name)
        archive.write(Path(__file__).with_name("README.md"), "README.md")
    rows, scores, owners, batches, roles, labels = [], [], [], [], [], []
    example = None
    error = 0.
    for person_index, uid in enumerate(np.unique(data["owner"])):
        index = np.flatnonzero(data["owner"] == uid)
        role = str(data["split"][index[0]])
        if str(uid) not in model_reg["splits"][role]:
            raise ValueError("Participant split changed")
        monitors = {n: BaselineMonitor(engine, uid, "recorded_keyboard", n) for n in (1, 3)}
        for number in sorted(np.unique(data["batch"][index])):
            ix = index[data["batch"][index] == number]
            raw_events = events[data["lineage"][ix].reshape(-1)]
            if len(ix) != 6 or not np.all(raw_events[:, 0] == uid):
                raise ValueError("Invalid participant batch")
            payload = dict(participant_id=str(uid), context_id="recorded_keyboard",
                batch_id=str(int(number)), sequence_number=int(number), events=[
                    dict(session_id=str(e[1]), event_id=str(e[2]), keydown_ms=float(e[3]),
                         keyup_ms=float(e[4]), keycode=float(e[5])) for e in raw_events])
            features, identifiers, windows = engine.encode(payload["events"])
            if windows != 6:
                raise ValueError("Replay lost input keys")
            error = max(error, float(np.max(np.abs(features[:, :128]-data["x"][ix]))))
            np.testing.assert_allclose(features[:, :128], data["x"][ix], atol=2e-5, rtol=2e-5)
            np.testing.assert_allclose(features, data["x"][ix], atol=2e-5, rtol=2e-5)
            model_scores = engine.probabilities(features).mean(axis=1).astype(np.float64)
            scores.append(model_scores)
            owners.append(str(uid)); batches.append(int(number)); roles.append(role); labels.append(int(data["y"][ix[0]]))
            for n, monitor in monitors.items():
                if example is None and role == "test" and n == 3 and number == 3:
                    monitor.save(out / "example-state.npz")
                    write_json(out / "example-input.json", payload)
                    restored = BaselineMonitor(engine, uid, "recorded_keyboard", n)
                    restored.restore(out / "example-state.npz")
                    example = restored.observe(payload)
                if set(identifiers) & set(monitor.state["event_ids"]):
                    raise ValueError("Replay reused earlier events")
                result = monitor._accept(str(int(number)), int(number), identifiers, features[:, :128], model_scores)
                rows.append(dict(result, role=role, baseline_setting=n))
                if example is not None and example["participant_id"] == uid and n == 3 and number == 3:
                    if result != example:
                        raise ValueError("Restored raw-input replay differs")
        if person_index % 50 == 0:
            print(json.dumps(dict(people=person_index+1, batches=len(scores))), flush=True)
    owners, batches, roles, labels, scores = map(np.asarray, (owners, batches, roles, labels, scores))
    aggregates = []
    for n in (1, 3):
        for role in ROLES:
            mask = (roles == role) & (batches >= n)
            ids = np.unique(owners[mask])
            person_scores = np.stack([scores[mask & (owners == uid)].mean(0) for uid in ids])
            person_labels = np.array([labels[np.flatnonzero(owners == uid)[0]] for uid in ids])
            fold = [metrics(person_labels, person_scores[:, i], "classification") for i in range(25)]
            aggregates.append(dict(baseline_batches=n, role=role, people=len(ids), followup_batches=int(mask.sum()),
                mean_fold_accuracy=float(np.mean([r["accuracy"] for r in fold])),
                sd_fold_accuracy=float(np.std([r["accuracy"] for r in fold], ddof=1)),
                mean_fold_auroc=float(np.mean([r["auroc"] for r in fold])),
                person_ensemble=metrics(person_labels, person_scores.mean(1), "classification"),
                batch_ensemble=metrics(labels[mask], scores[mask].mean(1), "classification")))
    np.savez_compressed(out / "scores.npz", owner=owners, batch=batches, split=roles, y=labels, scores=scores)
    write_json(out / "history.json", rows)
    write_json(out / "results.json", dict(people=len(set(owners)), batches=len(batches), aggregates=aggregates,
        example=example, interpretation="PD/control discrimination on existing participants, not validation of deterioration detection; no new training or threshold tuning.",
        sd_definition="Sample SD across 25 existing fold models on the same participants"))
    write_json(out / "verification.json", dict(raw_input_embedding_max_abs_error=error,
        unique_events=int(data["lineage"].size), restored_observation_matches=True,
        participant_splits_unchanged=True, labels_used_only_for_evaluation=True,
        registration_sha256=sha256(out / "registration.json"), scores_sha256=sha256(out / "scores.npz")))
    print(json.dumps(aggregates, indent=2), flush=True)


def tuning_metrics(y, scores, threshold):
    positive = scores >= threshold
    tp, tn = int(np.sum(positive & (y == 1))), int(np.sum(~positive & (y == 0)))
    fp, fn = int(np.sum(positive & (y == 0))), int(np.sum(~positive & (y == 1)))
    sensitivity, specificity = tp/(tp+fn), tn/(tn+fp)
    return dict(accuracy=(tp+tn)/len(y), balanced_accuracy=(sensitivity+specificity)/2,
                auroc=float(roc_auc_score(y, scores)), sensitivity=sensitivity, specificity=specificity,
                confusion_matrix=[[tn, fp], [fn, tp]])


def tuning_people(owners, labels, scores):
    people, first, inverse = np.unique(owners, return_index=True, return_inverse=True)
    values = np.bincount(inverse, weights=scores)/np.bincount(inverse)
    return people, labels[first], values


def tuning_batch_scores(model, x):
    return predict(model, x.reshape(-1, 128)).reshape(len(x), 6).mean(1).astype(np.float64)


def distance_gate(data, policy):
    if policy["metric"] == "none":
        return np.ones(len(data["batch_owner"]), dtype=bool)
    return data[policy["metric"]] >= policy["threshold"]


def prepare_tuning(out, run):
    source = Path(__file__).parent / "runs/baseline_embeddings"
    original = json.loads((run / "registration.json").read_text())
    paired = json.loads((source / "registration.json").read_text())
    if (original["features"] != 128 or original["augmentation"] != [0]
            or sha256(run / "dataset.npz") != original["dataset_sha256"]
            or sha256(source / "300/dataset.npz") != paired["scenarios"]["300"]["dataset_sha256"]
            or original["encoder_sha256"] != paired["encoder_sha256"]
            or sha256(ROOT / original["encoder"]) != original["encoder_sha256"]):
        raise ValueError("Unexpected source data or encoder")
    with np.load(run / "dataset.npz", allow_pickle=False) as cache:
        original_data = {k: cache[k] for k in cache.files}
    with np.load(source / "300/dataset.npz", allow_pickle=False) as cache:
        data = {k: cache[k] for k in ("x", "owner", "batch", "y", "split", "lineage")}
    if original_data["synthetic"].any() or len(np.unique(data["lineage"])) != data["lineage"].size:
        raise ValueError("Synthetic or reused events")
    batches = defaultdict(list)
    for uid in np.unique(data["owner"]):
        person = data["owner"] == uid
        base = data["x"][person & (data["batch"] == 0)]
        if base.shape != (6, 128):
            raise ValueError("Invalid enrollment matrix")
        for number in sorted(set(data["batch"][person]) - {0}):
            mask = person & (data["batch"] == number)
            role = str(data["split"][mask][0])
            if uid not in original["splits"][role]:
                raise ValueError("Participant split changed")
            x = data["x"][mask]
            distance = baseline_distance(base, x)
            values = dict(batch_x=x, batch_owner=str(uid), batch_y=int(data["y"][mask][0]),
                batch=int(number), split=role, distance=distance["embedding_distance"],
                ratio=distance["embedding_distance_ratio"])
            for key, value in values.items():
                batches[key].append(value)
    batches = {key: np.asarray(value) for key, value in batches.items()}
    if not np.isfinite(batches["distance"]).all() or not np.isfinite(batches["ratio"]).all():
        raise ValueError("Degenerate enrollment variability")
    out.mkdir(parents=True, exist_ok=False)
    training = {key: original_data[key][original_data["split"] == "train"] for key in ("x", "y", "owner")}
    training.update({key: value[batches["split"] == "train"] for key, value in batches.items()})
    holdout = {key: value[batches["split"] != "train"] for key, value in batches.items()}
    np.savez_compressed(out / "train.npz", **training)
    np.savez_compressed(out / "holdout.npz", **holdout)
    registration = dict(seed=SEED, classifier_run=str(run.resolve()), folds=original["folds"],
        original_registration_sha256=sha256(run / "registration.json"),
        original_dataset_sha256=original["dataset_sha256"], paired_dataset_sha256=paired["scenarios"]["300"]["dataset_sha256"],
        encoder=original["encoder"], encoder_sha256=original["encoder_sha256"],
        source_hashes={**source_hashes(), "paper_typenet/nn.py": sha256(ROOT / "paper_typenet/nn.py")},
        train_sha256=sha256(out / "train.npz"), holdout_sha256=sha256(out / "holdout.npz"),
        stage1=dict(max_depth=[1, 2, 3], n_estimators=[75, 150, 300], learning_rate=[.03, .1], min_child_weight=[3, 10]),
        stage2=dict(reg_lambda=[1, 10, 30], reg_alpha=[0, 1], subsample=[.8, 1.], colsample_bytree=[.75, 1.]),
        inner_folds=5, inner_seed_offset=1000, score_thresholds=[round(v, 2) for v in np.arange(.3, .701, .025)],
        distance_quantiles=[.1, .25, .5, .75, .9, .95],
        selection="Two-stage grid: mean inner-validation participant AUROC, then balanced accuracy at .5, then grid order. No selection on outer CV or held-out validation/test.",
        thresholds="Score threshold maximizes inner-OOF participant balanced accuracy; ties prefer .5. Distance rule maximizes same objective with fixed score threshold; ties prefer sensitivity then no gate.",
        decision="Each 300-key follow-up score is zeroed if its distance fails the gate. Scores are averaged per participant before thresholding. Ungated PD score remains separately available.",
        distance="Mean of 36 baseline/current Euclidean distances, or divided by mean of 30 off-diagonal baseline distances. Both thresholds use outer-training distance quantiles only.",
        ensemble="Equal mean scores across 25 fold models, with mean of their selected score thresholds. Gated scores apply each model's own gate before averaging.",
        arms=["current", "tuned", "tuned_distance"], synthetic=False, baseline_keys=300,
        counts={role: dict(people=len(set(batches["batch_owner"][batches["split"] == role])),
                          batches=int(np.sum(batches["split"] == role))) for role in ROLES},
        training_people=len(set(training["owner"])), training_rows=len(training["x"]),
        sources=["https://arxiv.org/html/2101.05570v3", "https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html"],
        interpretation="PD/control labels, not labels of clinical deterioration. Fixed existing participants have been evaluated previously.")
    write_json(out / "registration.json", registration)
    with zipfile.ZipFile(out / "source.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in registration["source_hashes"]:
            archive.write(ROOT / name, name)
        archive.write(Path(__file__).with_name("README.md"), "README.md")
    print(json.dumps(dict(counts=registration["counts"], training_people=registration["training_people"],
                          inner_fits=25*5*(36+24), outer_fits=50)), flush=True)


def tuning_fold(args):
    out, index = args
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        reg = json.loads((out / "registration.json").read_text())
        with np.load(out / "train.npz", allow_pickle=False) as cache:
            data = {key: cache[key] for key in cache.files}
        outer = reg["folds"][index]
        people = np.asarray(outer["fit"])
        labels = np.array([data["y"][np.flatnonzero(data["owner"] == uid)[0]] for uid in people])
        fit_data = {key: value[np.isin(data["owner"], people)] for key, value in data.items() if key in ("x", "owner", "y")}
        batches = {key: value[np.isin(data["batch_owner"], people)] for key, value in data.items() if key not in ("x", "owner", "y")}
        inner, records = [], []
        for fit, score in StratifiedKFold(5, shuffle=True, random_state=SEED+1000+index).split(people, labels):
            inner.append(dict(fit=people[fit].tolist(), score=people[score].tolist()))
        best = None
        for stage in (1, 2):
            anchor = {} if stage == 1 else best["parameters"]
            stage_best = None
            for number, parameters in enumerate(ParameterGrid(reg[f"stage{stage}"])):
                parameters = {**anchor, **parameters}
                oof = np.full(len(batches["batch_owner"]), np.nan)
                scores = []
                for fold in inner:
                    train = np.isin(fit_data["owner"], fold["fit"])
                    score = np.isin(batches["batch_owner"], fold["score"])
                    model = fit_model("xgboost", fit_data["x"][train], fit_data["y"][train],
                        fit_data["owner"][train], "classification", "none", 0, parameters=parameters)
                    oof[score] = tuning_batch_scores(model, batches["batch_x"][score])
                    _, y, prediction = tuning_people(batches["batch_owner"][score], batches["batch_y"][score], oof[score])
                    scores.append(tuning_metrics(y, prediction, .5))
                if not np.isfinite(oof).all():
                    raise ValueError("Missing inner out-of-fold predictions")
                record = dict(stage=stage, candidate=number, parameters=parameters,
                    auroc=float(np.mean([s["auroc"] for s in scores])),
                    balanced_accuracy=float(np.mean([s["balanced_accuracy"] for s in scores])))
                records.append(record)
                key = (record["auroc"], record["balanced_accuracy"], -number)
                if stage_best is None or key > stage_best["key"]:
                    stage_best = dict(key=key, parameters=parameters, oof=oof, record=record)
            best = stage_best
            print(json.dumps(dict(fold=index+1, stage=stage, inner_auroc=best["record"]["auroc"])), flush=True)
        _, y, prediction = tuning_people(batches["batch_owner"], batches["batch_y"], best["oof"])
        thresholds = [(tuning_metrics(y, prediction, t), t) for t in reg["score_thresholds"]]
        chosen, threshold = max(thresholds, key=lambda item: (item[0]["balanced_accuracy"], -abs(item[1]-.5), -item[1]))
        policies = [dict(metric="none", threshold=0.)]
        for metric in ("distance", "ratio"):
            policies.extend(dict(metric=metric, threshold=float(t)) for t in np.unique(np.quantile(batches[metric], reg["distance_quantiles"])))
        gates = []
        for policy in policies:
            _, gate_y, gate_scores = tuning_people(batches["batch_owner"], batches["batch_y"], best["oof"]*distance_gate(batches, policy))
            gates.append(dict(policy=policy, metrics=tuning_metrics(gate_y, gate_scores, threshold)))
        selected = max(range(len(gates)), key=lambda i: (gates[i]["metrics"]["balanced_accuracy"], gates[i]["metrics"]["sensitivity"], -i))
        policy = gates[selected]["policy"]
        models = {"current": fit_model("xgboost", fit_data["x"], fit_data["y"], fit_data["owner"], "classification", "none", 0),
                  "tuned": fit_model("xgboost", fit_data["x"], fit_data["y"], fit_data["owner"], "classification", "none", 0, parameters=best["parameters"])}
        held = np.isin(data["batch_owner"], outer["score"])
        outer_data = {key: value[held] for key, value in data.items() if key not in ("x", "owner", "y")}
        predictions = {key: tuning_batch_scores(model, outer_data["batch_x"]) for key, model in models.items()}
        predictions["tuned_distance"] = predictions["tuned"]*distance_gate(outer_data, policy)
        cv = {}
        for arm, scores in predictions.items():
            _, y, values = tuning_people(outer_data["batch_owner"], outer_data["batch_y"], scores)
            cv[arm] = tuning_metrics(y, values, .5 if arm == "current" else threshold)
        path = out / f"fold{index+1:02d}.joblib"
        if path.exists():
            raise FileExistsError(path)
        joblib.dump(dict(models=models, parameters=best["parameters"], threshold=threshold, policy=policy,
            outer=outer, inner=inner, search=records, gates=gates, inner_oof=best["oof"],
            inner_batch_owner=batches["batch_owner"], inner_batch=batches["batch"],
            threshold_scores=[dict(threshold=t, **m) for m, t in thresholds], cv=cv,
            outer_predictions=predictions, outer_batch_owner=outer_data["batch_owner"], outer_batch=outer_data["batch"],
            registration_sha256=sha256(out / "registration.json")), path, compress=3)
        return dict(index=index+1, sha256=sha256(path), cv=cv, parameters=best["parameters"], threshold=threshold, policy=policy)


def verify_tuning(out):
    reg = json.loads((out / "registration.json").read_text())
    for name, digest in reg["source_hashes"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f"Registered source changed: {name}")
    if sha256(out / "train.npz") != reg["train_sha256"]:
        raise ValueError("Training data changed")
    return reg


def train_tuning(out):
    reg = verify_tuning(out)
    if (out / "freeze.json").exists() or any(out.glob("fold*.joblib")):
        raise FileExistsError("Training output already exists")
    with ProcessPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(tuning_fold, [(out, i) for i in range(len(reg["folds"]))]))
    verify_tuning(out)
    write_json(out / "freeze.json", dict(registration_sha256=sha256(out / "registration.json"), models=records))


def evaluate_tuning(out):
    if (out / "results.json").exists() or (out / "predictions.npz").exists():
        raise FileExistsError("Evaluation output already exists")
    reg = verify_tuning(out)
    freeze = json.loads((out / "freeze.json").read_text())
    if freeze["registration_sha256"] != sha256(out / "registration.json") or sha256(out / "holdout.npz") != reg["holdout_sha256"]:
        raise ValueError("Frozen protocol or held-out data changed")
    with np.load(out / "holdout.npz", allow_pickle=False) as cache:
        data = {key: cache[key] for key in cache.files}
    predictions, thresholds, rows = {arm: [] for arm in reg["arms"]}, [], []
    for record in freeze["models"]:
        path = out / f"fold{record['index']:02d}.joblib"
        if sha256(path) != record["sha256"]:
            raise ValueError("Model changed after freeze")
        bundle = joblib.load(path)
        threshold = bundle["threshold"]
        thresholds.append(threshold)
        scores = {arm: tuning_batch_scores(model, data["batch_x"]) for arm, model in bundle["models"].items()}
        scores["tuned_distance"] = scores["tuned"]*distance_gate(data, bundle["policy"])
        for arm in reg["arms"]:
            predictions[arm].append(scores[arm])
            for role in ("validation", "test"):
                mask = data["split"] == role
                people, y, values = tuning_people(data["batch_owner"][mask], data["batch_y"][mask], scores[arm][mask])
                t = .5 if arm == "current" else threshold
                rows.append(dict(index=record["index"], arm=arm, role=role, people=len(people),
                    **tuning_metrics(y, values, t), batch_metrics=tuning_metrics(data["batch_y"][mask], scores[arm][mask], t)))
    predictions = {key: np.stack(value) for key, value in predictions.items()}
    aggregates = []
    for arm in reg["arms"]:
        for role in ("cv", "validation", "test"):
            group = [record["cv"][arm] for record in freeze["models"]] if role == "cv" else [r for r in rows if r["arm"] == arm and r["role"] == role]
            aggregate = dict(arm=arm, role=role, models=len(group), **{m: dict(mean=float(np.mean([r[m] for r in group])),
                sd=float(np.std([r[m] for r in group], ddof=1))) for m in ("accuracy", "balanced_accuracy", "auroc", "sensitivity", "specificity")})
            if role != "cv":
                mask = data["split"] == role
                scores = predictions[arm].mean(0)[mask]
                people, y, values = tuning_people(data["batch_owner"][mask], data["batch_y"][mask], scores)
                t = .5 if arm == "current" else float(np.mean(thresholds))
                aggregate.update(people=len(people), batches=int(mask.sum()), ensemble_threshold=t,
                    ensemble=tuning_metrics(y, values, t), batch_ensemble=tuning_metrics(data["batch_y"][mask], scores, t))
            aggregates.append(aggregate)
    np.savez_compressed(out / "predictions.npz", **predictions, thresholds=np.asarray(thresholds),
        owner=data["batch_owner"], batch=data["batch"], y=data["batch_y"], split=data["split"])
    write_json(out / "results.json", dict(rows=rows, aggregates=aggregates, counts=reg["counts"],
        freeze_sha256=sha256(out / "freeze.json"), predictions_sha256=sha256(out / "predictions.npz"),
        selected_policies=[r["policy"] for r in freeze["models"]],
        sd_definition="Sample SD across 25 outer-fold models, not 25 independent test populations. CV scores are from untouched outer folds.",
        auroc_definition="For the distance arm, AUROC ranks gated scores (failed gates become zero); these are not calibrated probabilities."))
    print(json.dumps(aggregates, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("observe", "export", "replay", "tune-prepare", "tune-train", "tune-evaluate"), required=True)
    parser.add_argument("--run", type=Path, default=Path(__file__).parent / "runs/embeddings")
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--baseline-batches", type=int, default=1)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and args.phase not in ("tune-train", "tune-evaluate"):
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        if args.phase == "tune-prepare":
            prepare_tuning(args.output, args.run)
            return
        if args.phase == "tune-train":
            train_tuning(args.output)
            return
        if args.phase == "tune-evaluate":
            evaluate_tuning(args.output)
            return
        if args.phase == "replay":
            if args.artifact:
                parser.error("replay requires a registered --run")
            replay_baselines(args.output, args.run)
            return
        if args.phase == "observe":
            if args.input is None or args.state is None:
                parser.error("observe requires --input and --state")
            if len({p.resolve() for p in (args.input, args.state, args.output)}) != 3:
                parser.error("Input, state and output paths must differ")
            payload = json.loads(args.input.read_text())
            participant, context = payload["participant_id"], payload["context_id"]
        else:
            participant, context = "export", "export"
        monitor = (BaselineMonitor.load(args.artifact, participant, context) if args.artifact else
                   BaselineMonitor(TypingClassifier(args.run), participant, context, args.baseline_batches))
        if args.phase == "export":
            monitor.export(args.output)
        else:
            if args.state.exists():
                monitor.restore(args.state)
            result = monitor.observe(payload)
            monitor.save(args.state)
            write_json(args.output, result)


if __name__ == "__main__":
    main()
