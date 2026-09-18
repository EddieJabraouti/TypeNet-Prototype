"""Clinical keystroke experiments. Run with --help; protocol and sources in README.md."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import zipfile
import argparse
import platform
import time
import subprocess
import shutil
import tempfile
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import sklearn
import torch
import xgboost
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, log_loss, mean_absolute_error, r2_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from torch import nn
from torch.nn import functional as F

from prototype_net.failed.exp3.node import ObliviousLayer
from prototype_net.failed.exp3.tabnet import TabNet

ROOT = Path(__file__).resolve().parents[2]
SEED = 9172026
ROLES = ("train", "validation", "test")
MODELS = ("xgboost", "random_forest", "tabnet", "node")
STATS = ("mean", "sd", "q10", "median", "q90", "iqr", "skew")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def summary(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return np.full(len(STATS), np.nan)
    q10, q25, med, q75, q90 = np.quantile(a, [.1, .25, .5, .75, .9])
    sd = a.std()
    skew = np.mean(((a - a.mean()) / sd) ** 3) if sd > 1e-12 else 0.
    return np.array([a.mean(), sd, q10, med, q90, q75 - q25, skew])


def neuroqwerty(base, windows=False):
    ids, labels, targets, features, studies = [], [], [], [], []
    audit = Counter()
    views, owners = [], []
    source = base / "neuroqwerty/neuroQWERTY.zip"
    with zipfile.ZipFile(source) as z:
        for study in ("MIT-CS1PD", "MIT-CS2PD"):
            metadata = csv.DictReader(io.StringIO(z.read(f"{study}/GT_DataPD_{study}.csv").decode()))
            for person in metadata:
                holds, person_views = [], []
                for column, name in person.items():
                    if not column.startswith("file_") or not name:
                        continue
                    audit["recordings"] += 1
                    seen = set()
                    bins = defaultdict(list)
                    rows = csv.reader(io.StringIO(z.read(f"{study}/data_{study}/{name}").decode()))
                    for row in rows:
                        audit["raw_records"] += 1
                        try:
                            key = row[0]
                            if key.startswith("mouse"):
                                audit["mouse_records"] += 1
                                continue
                            hold, release, press = map(float, row[1:])
                            if not all(map(math.isfinite, (hold, release, press))) or not (hold > 0 and release >= press >= 0):
                                raise ValueError("invalid timing")
                            if abs(hold - (release - press)) > .001:
                                raise ValueError("timestamp mismatch")
                            token = (key, hold, release, press)
                            if token in seen:
                                audit["duplicate_records"] += 1
                                continue
                            seen.add(token)
                            holds.append(hold)
                            bins[int(press // 90)].append(hold)
                        except (ValueError, IndexError):
                            audit["invalid_records"] += 1
                    if windows:
                        person_views.extend(summary(b) for b in bins.values() if len(b) >= 30)
                        audit["window_records"] += sum(len(b) for b in bins.values() if len(b) >= 30)
                if not holds:
                    raise ValueError(f"No eligible data for {person['pID']}")
                if person["gt"] not in ("True", "False"):
                    raise ValueError("Unrecognized motor diagnosis")
                # Global pID protects against accidental cross-study duplicates.
                ids.append("nq:" + person["pID"])
                labels.append(int(person["gt"] == "True"))
                targets.append(float(person["updrs108"]))
                studies.append(study)
                features.append(summary(holds))
                if windows:
                    views.extend([summary(holds), *person_views])
                    owners.extend(["nq:" + person["pID"]] * (1 + len(person_views)))
                audit["retained_records"] += len(holds)
    return dict(ids=ids, y=labels, updrs=targets, x=features, strata=studies,
                features=["hold_seconds_" + k for k in STATS], audit=dict(audit), views=views, owners=owners)


WRITING_FEATURES = (
    "time_on_task_s", "number_of_char_per_minute",
    "number_of_pauses_per_minute_PT30", "proportion_of_pause_time_PT2000",
    "number_of_P_bursts_per_minute_PB2000", "duration_of_P_bursts_s_PB2000",
    "pause_time_within_words_mean_s_PT30", "median_char_char",
)


def cognitive_label(value):
    if value.startswith("gezonde"):
        return 0
    if value in ("mci", "mild cognitive impairment", "Alzheimer"):
        return 1
    raise ValueError(f"Unknown cognitive diagnosis {value}")


def writing(base):
    folder = base / "ad_mci_writing"
    with (folder / "Participant_level.csv").open(encoding="latin1", newline="") as f:
        rows = list(csv.DictReader(f, delimiter=";"))
    with (folder / "Linguistic_analysis_word_level.csv").open(encoding="latin1", newline="") as f:
        words = list(csv.DictReader(f, delimiter=";"))
    people = defaultdict(list)
    for r in rows:
        people[r["participant_code"]].append(r)
    word_pauses = defaultdict(list)
    unmatched = Counter()
    for r in words:
        uid = r["Participant"]
        if uid not in people:
            unmatched[uid] += 1
            continue
        if cognitive_label(r["Group1"]) != cognitive_label(people[uid][0]["group1"]):
            raise ValueError(f"Conflicting binary diagnosis for {uid}")
        if r["Task"] not in {p["task"] for p in people[uid]}:
            raise ValueError(f"Unmatched task for {uid}")
        try:
            pause = float(r["BetweenWords"])
            if math.isfinite(pause) and pause >= 0:
                word_pauses[uid].append(pause)
        except ValueError:
            pass
    ids, x, y = [], [], []
    for uid, tasks in sorted(people.items()):
        labels = {cognitive_label(t["group1"]) for t in tasks}
        if len(labels) != 1 or len({t["task"] for t in tasks}) != len(tasks):
            raise ValueError(f"Duplicate task or conflicting diagnosis for {uid}")
        values = np.asarray([[float(t[k]) if t[k].strip() else np.nan for k in WRITING_FEATURES] for t in tasks])
        if np.isinf(values).any():
            raise ValueError("Invalid released writing summary")
        means = [np.mean(c[np.isfinite(c)]) if np.isfinite(c).any() else np.nan for c in values.T]
        x.append([*means, np.mean(word_pauses[uid]) if word_pauses[uid] else np.nan])
        ids.append("writing:" + uid)
        y.append(labels.pop())
    return dict(ids=ids, x=x, y=y, features=[*WRITING_FEATURES, "between_words_mean_ms"],
                audit=dict(participant_task_rows=len(rows), word_rows=len(words),
                           matched_word_pause_rows=sum(map(len, word_pauses.values())),
                           unmatched_word_ids=dict(unmatched),
                           note="Exact-ID joins only; task means per person; word pauses pooled per person. Age bands are not used."))


def tappy(base):
    """All available monthly files; exact record union per stable participant ID."""
    folder = base / "tappy"
    metadata = {}
    conflicts = set()
    user_archives = [folder / "Archived-users.zip", *sorted((folder / "expanded").glob("*sers*.zip"))]
    for path in user_archives:
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                if not name.endswith(".txt") or Path(name).name.startswith("._"):
                    continue
                uid = Path(name).stem.removeprefix("User_")
                info = dict(line.split(": ", 1) for line in z.read(name).decode(errors="replace").splitlines() if ": " in line)
                label = info.get("Parkinsons")
                if label not in ("True", "False"):
                    continue
                if uid in metadata and metadata[uid] != label:
                    conflicts.add(uid)
                metadata[uid] = label
    archives = [folder / "Archived-Data.zip", *sorted((folder / "expanded").glob("*Data*.zip"))]
    audit = Counter()
    ids, labels, features = [], [], []
    # Read one participant at a time: bounded memory even for expanded releases.
    handles = [zipfile.ZipFile(path) for path in archives]
    try:
        files = defaultdict(list)
        for z in handles:
            for name in z.namelist():
                if name.endswith(".txt") and not Path(name).name.startswith("._"):
                    files[Path(name).stem.split("_")[0]].append((z, name))
        for index, (uid, members) in enumerate(sorted(files.items())):
            if uid not in metadata or uid in conflicts:
                audit["excluded_unlabeled_or_conflicting_people"] += 1
                continue
            records = set()
            for z, name in members:
                with z.open(name) as f:
                    for line in f:
                        audit["raw_labeled_records"] += 1
                        try:
                            r = line.decode(errors="replace").strip().split("\t")
                            if len(r) < 8:
                                r = line.decode(errors="replace").strip().split(",")
                            h, latency, flight = (float(r[j]) for j in (4, 6, 7))
                            if r[0] != uid or r[3] not in ("L", "R", "S") or not all(map(math.isfinite, (h, latency, flight))) or h <= 0 or latency <= 0:
                                raise ValueError("invalid timing")
                            # No sequence reconstruction; dates are only deduplication keys.
                            token = (r[1], r[2], r[3], r[5], h, latency, flight)
                            if token in records:
                                audit["duplicate_records"] += 1
                            records.add(token)
                        except (ValueError, IndexError):
                            audit["invalid_records"] += 1
            if records:
                values = np.asarray([r[4:] for r in sorted(records)], dtype=float) / 1000.
                features.append(np.concatenate([summary(values[:, j]) for j in range(3)]))
                ids.append("tappy:" + uid)
                labels.append(int(metadata[uid] == "True"))
                audit["retained_records"] += len(records)
            if index % 50 == 0:
                print(f"Tappy: {index + 1}/{len(files)} identities processed", flush=True)
    finally:
        for z in handles:
            z.close()
    return dict(ids=ids, y=labels, x=features,
                features=[f"{timing}_seconds_{s}" for timing in ("hold", "latency", "flight") for s in STATS],
                audit={**audit, "conflicting_label_ids": sorted(conflicts), "record_archives": [str(p.relative_to(ROOT)) for p in archives]})


def split_indices(ids, labels, seed=SEED):
    """One participant is one row. Shared by all models and descendants."""
    ids, labels = np.asarray(ids), np.asarray(labels)
    if len(set(ids)) != len(ids):
        raise ValueError("Participant IDs must be unique")
    indexes = np.arange(len(ids))
    trainval, test = train_test_split(indexes, test_size=.2, stratify=labels, random_state=seed)
    train, validation = train_test_split(trainval, test_size=.125, stratify=labels[trainval], random_state=seed)
    result = dict(train=np.sort(train), validation=np.sort(validation), test=np.sort(test))
    assert len(set(np.concatenate(list(result.values())))) == len(ids)
    return result


def prepare(out, base=None):
    base = base or ROOT / "data/clinical"
    out.mkdir(parents=True, exist_ok=False)
    (out / "README.md").write_bytes((Path(__file__).parent / "README.md").read_bytes())
    sources = sorted(base.rglob("*.zip*")) + sorted((base / "ad_mci_writing").glob("*.csv"))
    source_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in sources}
    cohorts = {"motor_pd": neuroqwerty(base), "cognitive_ci": writing(base), "tappy_self_report": tappy(base)}
    manifest = dict(seed=SEED, target_fractions=[.7, .1, .2],
                    rounding="ceil(0.20*N) test, ceil(0.125*remaining) validation; rest train",
                    source_hashes=source_hashes, cohorts={}, tasks={})
    for name, cohort in cohorts.items():
        roles = split_indices(cohort["ids"], cohort["y"])
        ids = np.asarray(cohort["ids"])
        x, y = np.asarray(cohort["x"], dtype=np.float64), np.asarray(cohort["y"], dtype=int)
        if np.isinf(x).any():
            raise ValueError("Infinite clinical feature")
        info = dict(features=cohort["features"], audit=cohort["audit"], splits={r: ids[i].tolist() for r, i in roles.items()},
                    counts={r: dict(participants=len(i), positive=int(y[i].sum()), fraction=len(i)/len(ids)) for r, i in roles.items()})
        manifest["cohorts"][name] = info
        endpoints = [(name, y, "classification")]
        if name == "motor_pd":
            endpoint = np.asarray(cohort["updrs"], dtype=float)
            if not np.isfinite(endpoint).all() or (endpoint < 0).any() or (endpoint > 108).any():
                raise ValueError("Invalid UPDRS-III label")
            endpoints.append(("motor_updrs", endpoint, "regression"))
        for task, target, kind in endpoints:
            taskdir = out / task
            taskdir.mkdir()
            hashes = {}
            for role, ix in roles.items():
                p = taskdir / f"{role}.npz"
                np.savez_compressed(p, x=x[ix], y=target[ix], ids=ids[ix],
                                    diagnosis=y[ix], features=np.asarray(cohort["features"]))
                hashes[role] = sha256(p)
            manifest["tasks"][task] = dict(cohort=name, kind=kind, cache_sha256=hashes)
    inventory = {}
    for name in ("Keystrokes", "keystrokes_f"):
        paths = sorted((ROOT / "data" / name / "files").glob("*"))
        inventory[name] = dict(files=len(paths), bytes=sum(p.stat().st_size for p in paths if p.is_file()),
                               role="unlabeled; excluded from clinical supervised targets", clinical_labels=False)
    manifest["unlabeled_inventory"] = inventory
    write_json(out / "manifest.json", manifest)
    print(json.dumps({k: v["counts"] for k, v in manifest["cohorts"].items()}, indent=2))
    return manifest


def fetch_tappy():
    folder = ROOT / "data/clinical/tappy/expanded"
    folder.mkdir(parents=True, exist_ok=True)
    url = "https://data.mendeley.com/public-api/datasets/z39mhdsynx/files?folder_id=root&version=3"
    manifest_path = folder / "download.json"
    if not manifest_path.exists():
        subprocess.run(["curl", "-fsSL", "--max-time", "60", "-H",
                        "Accept: application/vnd.mendeley-public-dataset.1+json", url, "-o", str(manifest_path)], check=True)
    listing = json.loads(manifest_path.read_text())
    archive = folder / "Archived Data.zip"
    if archive.exists():
        with zipfile.ZipFile(archive) as z:
            if z.testzip() is not None:
                raise ValueError("Expanded archive checksum failed")
        return
    parts = []
    for item in sorted(listing, key=lambda r: r["filename"]):
        name = item["filename"]
        if Path(name).name != name:
            raise ValueError("Unsafe archive filename")
        path = folder / name
        details = item["content_details"]
        if not path.exists() or sha256(path) != details["sha256_hash"]:
            temporary = path.with_suffix(path.suffix + ".part")
            subprocess.run(["curl", "-fL", "--retry", "3", "--max-time", "300",
                            details["download_url"], "-o", str(temporary)], check=True)
            if sha256(temporary) != details["sha256_hash"]:
                raise ValueError(f"Download hash mismatch: {name}")
            temporary.replace(path)
        if name.startswith("Archived Data.zip."):
            parts.append(path)
        print(f"Verified {name}", flush=True)
    combined = archive.with_suffix(".part")
    with combined.open("wb") as f:
        for p in parts:
            with p.open("rb") as source:
                shutil.copyfileobj(source, f)
    with zipfile.ZipFile(combined) as z:
        if z.testzip() is not None:
            raise ValueError("Expanded archive checksum failed")
    combined.replace(archive)
    for p in parts:
        p.unlink()


def preprocess(x, state=None):
    if state is None:
        medians = np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0. for c in x.T])
        filled = np.where(np.isnan(x), medians, x)
        scaler = StandardScaler().fit(filled)
        state = dict(medians=medians, scaler=scaler)
    transformed = state["scaler"].transform(np.where(np.isnan(x), state["medians"], x)).astype(np.float32)
    if not np.isfinite(transformed).all():
        raise ValueError("Nonfinite transformed features")
    return transformed, state


def aalto_person(path):
    uid = path.stem.removesuffix("_keystrokes")
    columns = ["PARTICIPANT_ID", "KEYSTROKE_ID", "PRESS_TIME", "RELEASE_TIME"]
    alternate = ROOT / "data/keystrokes_f/files" / path.name
    choices = [path] + ([alternate] if alternate != path and alternate.exists() else [])
    status = "primary"
    for candidate in choices:
        raw = candidate.read_bytes()
        if not all(c.encode() in raw.split(b"\n", 1)[0] for c in columns):
            continue
        data = pd.read_csv(io.BytesIO(raw), sep="\t", quoting=csv.QUOTE_NONE, encoding="latin1", usecols=columns)
        status = "fallback" if candidate != path else "primary"
        break
    else:
        return uid, np.full(7, np.nan), 0, 0, hashlib.sha256(raw).hexdigest(), "unreadable_header"
    count = len(data)
    person = pd.to_numeric(data["PARTICIPANT_ID"], errors="coerce")
    event = pd.to_numeric(data["KEYSTROKE_ID"], errors="coerce")
    data = data[(person == int(uid)) & np.isfinite(event)].copy()
    data = data.drop_duplicates("KEYSTROKE_ID")
    holds = (pd.to_numeric(data["RELEASE_TIME"], errors="coerce") - pd.to_numeric(data["PRESS_TIME"], errors="coerce")) / 1000.
    valid = holds[np.isfinite(holds) & (holds > 0)].to_numpy()
    return uid, summary(valid), count, len(valid), hashlib.sha256(raw).hexdigest(), status


def prepare_aalto(out):
    if (out / "registration.json").exists():
        raise ValueError("Cannot alter external data after registration")
    if not (out / "manifest.json").exists():
        raise ValueError("Run prepare first")
    path = out / "aalto.npz"
    if path.exists():
        raise FileExistsError(path)
    files = {}
    for folder in ("Keystrokes", "keystrokes_f"):
        for p in sorted((ROOT / "data" / folder / "files").glob("*_keystrokes.txt")):
            if not p.name.startswith("._"):
                files.setdefault(p.name, p)
    rows = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(files.values())
        for start in range(0, len(paths), 2000):
            rows.extend(pool.map(aalto_person, paths[start:start+2000]))
            print(f"Aalto: {min(start+2000, len(paths))}/{len(paths)} people", flush=True)
    eligible = [r for r in rows if np.isfinite(r[1]).all()]
    np.savez_compressed(path, ids=np.asarray([r[0] for r in eligible]), x=np.asarray([r[1] for r in eligible]),
                        file_sha256=np.asarray([r[4] for r in eligible]),
                        raw_records=np.asarray([r[2] for r in eligible]), retained_records=np.asarray([r[3] for r in eligible]))
    write_json(out / "aalto.json", dict(source_files=len(files), eligible_people=len(eligible),
                                       excluded_people=[r[0] for r in rows if not np.isfinite(r[1]).all()],
                                       recovered_from_secondary=[r[0] for r in rows if r[5] == "fallback"],
                                       unreadable_header=[r[0] for r in rows if r[5] == "unreadable_header"],
                                       raw_records=sum(r[2] for r in rows), retained_records=sum(r[3] for r in rows),
                                       sha256=sha256(path), labels="none", source_priority=["Keystrokes", "keystrokes_f"]))


def touchscreen():
    path = ROOT / "data/clinical/touchscreen/data.zip"
    if hashlib.md5(path.read_bytes()).hexdigest() != "54be254f9194789102ab4e279cff2182":
        raise ValueError("Touchscreen publisher checksum mismatch")
    result = dict(ids=[], y=[], updrs=[], x=[], views=[], owners=[], audit=Counter())
    with zipfile.ZipFile(path) as z:
        table = pd.read_excel(io.BytesIO(z.read(next(n for n in z.namelist() if n.endswith(".xlsx")))))
        for _, person in table.iterrows():
            uid = f"touchscreen:{int(person['Subject ID']):02d}"
            holds, rows = [], []
            for name in sorted(z.namelist()):
                if not name.endswith(".txt") or f"/S{int(person['Subject ID']):02d}/" not in name:
                    continue
                session, seen = [], set()
                for line in z.read(name).decode().splitlines():
                    if not line.strip():
                        continue
                    result["audit"]["raw_records"] += 1
                    try:
                        fields = line.split(",")
                        press, release = float(fields[1]), float(fields[2].removeprefix("Release "))
                        if not (math.isfinite(press) and math.isfinite(release) and release > press >= 0):
                            raise ValueError("Invalid hold")
                        token = (press, release)
                        if token in seen:
                            result["audit"]["duplicates"] += 1
                            continue
                        seen.add(token)
                        session.append((release-press)/1000)
                    except (ValueError, IndexError):
                        result["audit"]["invalid_records"] += 1
                if session:
                    rows.append(summary(session))
                    holds.extend(session)
                    result["audit"]["recordings"] += 1
            if not holds or person["Group"] not in ("PD", "Control"):
                raise ValueError("Missing touchscreen measurements or label")
            result["ids"].append(uid)
            result["y"].append(int(person["Group"] == "PD"))
            result["updrs"].append(float(person["UPDRS_III Total score"]))
            result["x"].append(summary(holds))
            result["views"].extend([summary(holds), *rows])
            result["owners"].extend([uid] * (1+len(rows)))
            result["audit"]["retained_records"] += len(holds)
    return result


def online_typing():
    path = ROOT / "data/clinical/conll/typing.csv"
    if sha256(path) != "db2166b5211b359c0bea4fb3057d8a41fecfbc98057fdc23c2f8fbe946984403":
        raise ValueError("CoNLL publisher checksum mismatch")
    columns = ["key", "participant_id", "response_id", "diagnosis", "keydown", "keyup"]
    data = pd.read_csv(path, usecols=columns)
    raw = len(data)
    data = data.drop_duplicates(columns)
    result = dict(ids=[], y=[], x=[], views=[], owners=[], audit=Counter(raw_records=raw, duplicates=raw-len(data)))
    for person, group in data.groupby("participant_id", sort=True):
        if group["diagnosis"].nunique() != 1 or group["diagnosis"].iloc[0] not in (0, 1):
            raise ValueError("Conflicting online diagnosis")
        uid = "conll:" + str(person)
        rows, holds = [], []
        for _, session in group.groupby("response_id", sort=True):
            values = (session["keyup"] - session["keydown"]).to_numpy(float)/1000
            valid = np.isfinite(values) & (values > 0) & (session["keydown"].to_numpy(float) >= 0)
            result["audit"]["invalid_records"] += int((~valid).sum())
            if valid.any():
                rows.append(summary(values[valid]))
                holds.extend(values[valid])
                result["audit"]["recordings"] += 1
        if not holds:
            result["audit"]["excluded_empty_people"] += 1
            continue
        result["ids"].append(uid)
        result["y"].append(int(group["diagnosis"].iloc[0]))
        result["x"].append(summary(holds))
        result["views"].extend([summary(holds), *rows])
        result["owners"].extend([uid] * (1+len(rows)))
        result["audit"]["retained_records"] += len(holds)
    return result


def expand(out, baseline):
    previous = json.loads((baseline / "manifest.json").read_text())
    registered = json.loads((baseline / "registration.json").read_text())
    if sha256(baseline / "manifest.json") != registered["manifest_sha256"] or sha256(baseline / "aalto.npz") != registered["aalto_sha256"]:
        raise ValueError("Baseline data changed")
    if not (baseline / "freeze.json").exists():
        raise ValueError("Expansion requires a frozen baseline")
    out.mkdir(parents=True, exist_ok=False)
    (out / "README.md").write_bytes(Path(__file__).with_name("README.md").read_bytes())
    motor = neuroqwerty(ROOT / "data/clinical", windows=True)
    phone, online = touchscreen(), online_typing()
    cohorts = {"motor_pd": motor, "touchscreen_pd": phone, "online_pd": online}
    motor_roles = previous["cohorts"]["motor_pd"]["splits"]
    if set(motor["ids"]) != set(sum(motor_roles.values(), [])):
        raise ValueError("Motor participant population changed")
    phone_roles = {r: np.asarray(phone["ids"])[ix].tolist() for r, ix in split_indices(phone["ids"], phone["y"]).items()}
    online_roles = {r: np.asarray(online["ids"])[ix].tolist() for r, ix in split_indices(online["ids"], online["y"]).items()}
    roles = dict(motor_pd=motor_roles, touchscreen_pd=phone_roles, online_pd=online_roles)
    manifest = dict(seed=SEED, target_fractions=[.7, .1, .2], cohorts={}, tasks={}, workers=4,
                    augmentations=["none", "smote2", "smote8", "smote32", "smote128"], pretraining_epochs=100,
                    baseline_manifest_sha256=sha256(baseline / "manifest.json"), prior_evaluation_exposure=True,
                    views="One pooled summary plus 90-second desktop windows (>=30 holds), or all phone/online sessions; mean prediction per participant.")
    manifest["source_hashes"] = {str(p.relative_to(ROOT)): sha256(p) for p in (
        ROOT / "data/clinical/neuroqwerty/neuroQWERTY.zip", ROOT / "data/clinical/touchscreen/data.zip", ROOT / "data/clinical/conll/typing.csv")}
    for name, cohort in cohorts.items():
        ids, x = np.asarray(cohort["ids"]), np.asarray(cohort["x"])
        views, owners = np.asarray(cohort["views"]), np.asarray(cohort["owners"])
        manifest["cohorts"][name] = dict(splits=roles[name], audit=cohort["audit"],
                                        features=["hold_seconds_"+s for s in STATS], clinical_ground_truth=name != "online_pd")
        endpoints = [(name, cohort["y"], "classification")]
        if "updrs" in cohort:
            endpoints.append((name.replace("_pd", "_updrs"), cohort["updrs"], "regression"))
        for task, target, kind in endpoints:
            folder = out / task
            folder.mkdir()
            hashes, row_counts = {}, {}
            for role in ROLES:
                role_ids = np.asarray(roles[name][role])
                ix = np.asarray([int(np.flatnonzero(ids == uid)[0]) for uid in role_ids])
                mask = np.isin(owners, role_ids)
                path = folder / f"{role}.npz"
                np.savez_compressed(path, ids=role_ids, x=x[ix], y=np.asarray(target)[ix], diagnosis=np.asarray(cohort["y"])[ix],
                                    features=np.asarray(manifest["cohorts"][name]["features"]), views=views[mask], owners=owners[mask])
                hashes[role], row_counts[role] = sha256(path), int(mask.sum())
            manifest["tasks"][task] = dict(kind=kind, cohort=name, cache_sha256=hashes, real_rows=row_counts, use_encoder=name != "touchscreen_pd")
    with np.load(baseline / "aalto.npz", allow_pickle=False) as z:
        external = list(z["x"])
        external_ids = ["aalto:" + str(uid) for uid in z["ids"]]
    manifest["auxiliary"] = dict(aalto_cache_sha256=sha256(baseline / "aalto.npz"), aalto_people=len(external),
                                 excluded_how_we_type="No release events/hold durations in the released typing schema")
    archive = ROOT / "data/auxiliary/ikdd/data.zip"
    people, seen, retained, recordings = defaultdict(list), set(), 0, 0
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError("Corrupt IKDD archive")
        for name in sorted(z.namelist()):
            if not name.endswith(".txt"):
                continue
            raw = z.read(name)
            digest = hashlib.sha256(raw).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            lines = list(csv.reader(io.StringIO(raw.decode("utf-8-sig"))))
            uid = "ikdd:" + lines[0][0].split("_(")[0]
            holds = np.asarray([float(v)/1000 for row in lines[1:] if row and row[0].endswith("-0") for v in row[1:] if v.strip()])
            holds = holds[np.isfinite(holds) & (holds > 0)]
            if not len(holds):
                continue
            people[uid].extend(holds)
            external.append(summary(holds))
            external_ids.append(uid)
            retained += len(holds)
            recordings += 1
    for uid, holds in sorted(people.items()):
        external.append(summary(holds))
        external_ids.append(uid)
    # Only the already assigned Tappy training people can enter the shared encoder.
    tappy = load_role(baseline, previous, "tappy_self_report", "train")
    external.extend(tappy["x"][:, :7])
    external_ids.extend(tappy["ids"])
    manifest["auxiliary"].update(ikdd_archive_sha256=sha256(archive), ikdd_recordings=recordings,
                                  ikdd_people=len(people), ikdd_retained_holds=retained,
                                  tappy_train_cache_sha256=sha256(baseline / "tappy_self_report/train.npz"),
                                  tappy_train_people=len(tappy["ids"]), total_people=len(set(external_ids)), total_rows=len(external))
    np.savez_compressed(out / "aalto.npz", x=np.asarray(external), ids=np.asarray(external_ids))
    write_json(out / "manifest.json", manifest)
    print(json.dumps({"training": manifest["tasks"], "auxiliary": manifest["auxiliary"]}, indent=2))


def pretrain(out):
    path = out / "encoder.joblib"
    if path.exists():
        value = joblib.load(path)
        if value["registration_sha256"] != sha256(out / "registration.json"):
            raise ValueError("Encoder registration changed")
        return value
    with np.load(out / "aalto.npz", allow_pickle=False) as data:
        x, state = preprocess(data["x"])
        people = len(set(data["ids"])) if "ids" in data else len(x)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    model = neural_model("tabnet", 7, 7)
    model.initialize(torch.from_numpy(x))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=1e-4)
    rng = np.random.default_rng(SEED)
    epochs = json.loads((out / "manifest.json").read_text()).get("pretraining_epochs", 10)
    for epoch in range(epochs):
        losses = []
        for indexes in np.array_split(rng.permutation(len(x)), max(1, len(x)//512)):
            target = torch.from_numpy(x[indexes])
            mask = torch.rand_like(target) < .2
            optimizer.zero_grad()
            prediction, penalty = model(target.masked_fill(mask, 0.))
            loss = ((prediction-target).square() * mask).sum() / mask.sum().clamp_min(1) + .001*penalty
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite reconstruction loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            losses.append(float(loss.detach()))
        print(f"Aalto pretraining {epoch+1}/{epochs}: masked MSE {np.mean(losses):.5f}", flush=True)
    value = dict(preprocess=state, state=model.state_dict(), registration_sha256=sha256(out / "registration.json"), people=people, rows=len(x), epochs=epochs)
    joblib.dump(value, path, compress=3)
    return value


def embedding(encoder, raw):
    x, _ = preprocess(raw[:, :7], encoder["preprocess"])
    network = neural_model("tabnet", 7, 7)
    network.load_state_dict(encoder["state"])
    network.head = nn.Identity()
    network.eval()
    with torch.no_grad():
        return network(torch.from_numpy(x))[0].numpy()


def augment(x, y, ids, seed=SEED, multiplier=2):
    rng = np.random.default_rng(seed)
    target = multiplier * max(np.bincount(y))
    extra, targets, parents = [], [], []
    for label in np.unique(y):
        indexes = np.flatnonzero(y == label)
        if len(indexes) < 2:
            raise ValueError("SMOTE requires two real people per class")
        a = x[indexes].astype(float)
        distances = ((a[:, None] - a[None]) ** 2).sum(-1)
        np.fill_diagonal(distances, np.inf)
        neighbors = np.argsort(distances, axis=1, kind="stable")[:, :min(5, len(a) - 1)]
        for _ in range(target - len(a)):
            i = int(rng.integers(len(a)))
            j = int(rng.choice(neighbors[i]))
            weight = float(rng.random())
            extra.append(a[i] + weight * (a[j] - a[i]))
            targets.append(label)
            parents.append([str(ids[indexes[i]]), str(ids[indexes[j]]), weight])
    return np.concatenate([x, np.asarray(extra, dtype=np.float32)]), np.concatenate([y, targets]), parents


def fit_distribution(x, y, ids):
    if x.shape[1] != 7 or not np.isfinite(x).all():
        raise ValueError("Distribution generation requires seven finite hold statistics")
    if any(len(np.unique(y[ids == uid])) != 1 for uid in np.unique(ids)):
        raise ValueError("Conflicting source labels")
    classes = {}
    for label in np.unique(y):
        people = np.unique(ids[y == label])
        if len(people) < 2:
            raise ValueError("Distribution generation requires two people per class")
        groups = [x[ids == uid].astype(float) for uid in people]
        pooled = np.stack([g[0] for g in groups])
        between = np.cov(pooled, rowvar=False)
        within = np.mean([np.cov(g[1:], rowvar=False) if len(g) > 2 else np.zeros((7, 7)) for g in groups], axis=0)
        values, vectors = np.linalg.eigh(between + within)
        bandwidth = .25 * (4 / (9 * len(people))) ** (1 / 11)
        factor = vectors * np.sqrt(np.maximum(values, 0))[None, :] * bandwidth
        classes[int(label)] = dict(people=people, mean=pooled.mean(0),
                                   between_sd=np.sqrt(np.maximum(np.diag(between), 0)),
                                   within_sd=np.sqrt(np.maximum(np.diag(within), 0)),
                                   factor=factor, bandwidth=bandwidth)
    return classes


def sample_distribution(distribution, x, y, ids, count, seed=SEED):
    if count <= 0 or count % 2 or set(distribution) != {0, 1}:
        raise ValueError("Use an even positive count and two classes")
    rng = np.random.default_rng(seed)
    rows, targets, owners, anchors = [], [], [], []
    attempts = 0
    for label in (0, 1):
        people = np.unique(ids[y == label])
        if not len(people):
            raise ValueError("Missing source class")
        indexes = np.array([np.flatnonzero(ids == uid)[0] for uid in people])
        chosen = np.resize(rng.permutation(indexes), count // 2)
        generated = np.empty((len(chosen), 7))
        pending = np.arange(len(chosen))
        for _ in range(1000):
            if not len(pending):
                break
            candidates = x[chosen[pending]] + rng.normal(size=(len(pending), 7)) @ distribution[label]["factor"].T
            mean, sd, q10, median, q90, iqr, _ = candidates.T
            valid = (np.isfinite(candidates).all(1) & (mean > 0) & (sd >= 0) & (q10 > 0)
                     & (q10 <= median) & (median <= q90) & (iqr >= 0) & (iqr <= q90-q10 + 1e-12))
            generated[pending[valid]] = candidates[valid]
            attempts += len(pending)
            pending = pending[~valid]
        if len(pending):
            raise ValueError("Could not generate admissible timing profiles")
        rows.append(generated)
        targets.extend([label] * len(chosen))
        owners.extend(ids[chosen])
        anchors.extend(chosen)
    order = rng.permutation(count)
    return dict(x=np.concatenate(rows)[order], y=np.asarray(targets)[order],
                owners=np.asarray(owners)[order], anchors=np.asarray(anchors)[order], attempts=attempts)


def prepare_distribution(out, baseline):
    previous = json.loads((baseline / "manifest.json").read_text())
    registration = json.loads((baseline / "registration.json").read_text())
    freeze = json.loads((baseline / "freeze.json").read_text())
    if sha256(baseline / "manifest.json") != registration["manifest_sha256"] or sha256(baseline / "registration.json") != freeze["registration_sha256"]:
        raise ValueError("Source registration changed")
    encoder = baseline / "encoder.joblib"
    if sha256(encoder) != freeze["artifacts"]["encoder.joblib"]:
        raise ValueError("Source encoder changed")
    out.mkdir(parents=True, exist_ok=False)
    (out / "motor_pd").mkdir()
    for role in ROLES:
        data = load_role(baseline, previous, "motor_pd", role)
        for uid, pooled in zip(data["ids"], data["x"]):
            if not np.array_equal(pooled, data["views"][np.flatnonzero(data["owners"] == uid)[0]]):
                raise ValueError("First view must be the participant's pooled summary")
        shutil.copyfile(baseline / "motor_pd" / f"{role}.npz", out / "motor_pd" / f"{role}.npz")
    shutil.copyfile(encoder, out / "encoder.joblib")
    manifest = dict(seed=SEED, target_fractions=[.7, .1, .2], workers=4,
                    cohorts={"motor_pd": previous["cohorts"]["motor_pd"]},
                    tasks={"motor_pd": previous["tasks"]["motor_pd"]},
                    augmentations=["none", "rose700", "rose7000"],
                    frozen_encoder_sha256=sha256(encoder), baseline_manifest_sha256=sha256(baseline / "manifest.json"),
                    synthetic_evaluation={"validation": 1000, "test": 2000},
                    generation="Class-conditional Gaussian smoothed bootstrap around pooled participant profiles; kernel uses between-person and within-person covariance. Bandwidth multiplier .25 fixed before fitting.",
                    evaluation="Synthetic held-out profiles use held-out anchors and their labels with a training-only kernel. They measure label-conditioned perturbation robustness, not new independent people. Real participant metrics remain separate.",
                    prior_evaluation_exposure=True)
    write_json(out / "manifest.json", manifest)
    print("Prepared fixed 59/9/17 clinical people; reused frozen TabNet encoder; 700 and 7000 synthetic training profiles.")


def save_distribution(out, manifest):
    path = out / "distribution.joblib"
    if path.exists():
        item = joblib.load(path)
        if item["registration_sha256"] != sha256(out / "registration.json"):
            raise ValueError("Distribution registration changed")
        return {name: sha256(out / name) for name in item["artifacts"]}
    data = load_role(out, manifest, "motor_pd", "train")
    x, y, ids = training_rows(data, np.arange(len(data["ids"])))
    distribution = fit_distribution(x, y, ids)
    artifacts = ["distribution.joblib", "distribution.json"]
    diagnostics = dict(classes={}, synthetic={})
    for label, info in distribution.items():
        diagnostics["classes"][str(label)] = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in info.items() if k != "factor"}
    for augmentation in manifest["augmentations"]:
        if not augmentation.startswith("rose"):
            continue
        count = int(augmentation[4:])
        generated = sample_distribution(distribution, x, y, ids, count)
        name = f"synthetic_train_{count}.npz"
        np.savez_compressed(out / name, **generated)
        artifacts.append(name)
        diagnostics["synthetic"][str(count)] = dict(rows=count, source_people=len(set(generated["owners"])),
            attempts=generated["attempts"], classes={str(label): dict(mean=generated["x"][generated["y"] == label].mean(0).tolist(),
            sd=generated["x"][generated["y"] == label].std(0).tolist()) for label in (0, 1)})
    write_json(out / "distribution.json", diagnostics)
    joblib.dump(dict(distribution=distribution, registration_sha256=sha256(out / "registration.json"), artifacts=artifacts), path, compress=3)
    return {name: sha256(out / name) for name in artifacts}


class NODE(nn.Module):
    def __init__(self, features, outputs):
        super().__init__()
        self.layers = nn.ModuleList([ObliviousLayer(features + i * 16 * outputs, 16, 3, outputs) for i in range(2)])

    @torch.no_grad()
    def initialize(self, x):
        for layer in self.layers:
            layer.initialize(x)
            x = torch.cat([x, layer(x).flatten(1)], dim=1)

    def forward(self, x):
        values = []
        for layer in self.layers:
            value = layer(x)
            values.append(value)
            x = torch.cat([x, value.flatten(1)], dim=1)
        return torch.cat(values, dim=1).mean(1), x.new_zeros(())


def neural_model(name, features, outputs):
    if name == "node":
        return NODE(features, outputs)
    model = TabNet(features=features, decision=8, attention=8, steps=3)
    model.head = nn.Linear(8, outputs, bias=False)
    return model


def fit_model(name, x, y, ids, kind, augmentation, epochs, encoder=None, parameters=None):
    if parameters is not None and name != "xgboost":
        raise ValueError("Parameter overrides are supported for XGBoost only")
    raw = x
    x, state = preprocess(x)
    parents = []
    generation = None
    if augmentation.startswith("smote"):
        if kind != "classification":
            raise ValueError("Synthetic clinical severity labels are prohibited")
        x, y, parents = augment(x, y, ids, multiplier=int(augmentation.removeprefix("smote") or 2))
    elif augmentation.startswith("rose"):
        if kind != "classification":
            raise ValueError("Distribution augmentation requires observed classes")
        distribution = fit_distribution(raw, y, ids)
        generated = sample_distribution(distribution, raw, y, ids, int(augmentation[4:]))
        extra, _ = preprocess(generated["x"], state)
        x, y = np.concatenate([x, extra]), np.concatenate([y, generated["y"]])
        parents = [[str(uid), str(uid), None] for uid in generated["owners"]]
        generation = dict(method="class-conditional smoothed bootstrap", distribution=distribution,
                          anchors=generated["anchors"], attempts=generated["attempts"], seed=SEED,
                          synthetic_sha256=hashlib.sha256(generated["x"].tobytes()).hexdigest())
    if encoder is not None:
        x = np.concatenate([x, embedding(encoder, state["scaler"].inverse_transform(x))], axis=1)
    anchors = np.concatenate([ids, np.asarray([p[0] for p in parents], dtype=str)])
    counts = Counter(anchors)
    weights = np.asarray([1 / counts[uid] for uid in anchors], dtype=np.float32)
    weights *= len(weights) / weights.sum()
    result = dict(name=name, kind=kind, preprocess=state, encoder=encoder, parents=parents,
                  real_people=len(set(ids)), real_rows=len(ids), fit_rows=len(y))
    if generation is not None:
        result["generation"] = generation
    if name == "dummy":
        result["constant"] = float(np.average(y, weights=weights))
    elif name in ("xgboost", "random_forest"):
        if name == "xgboost":
            cls = xgboost.XGBClassifier if kind == "classification" else xgboost.XGBRegressor
            options = dict(n_estimators=150, max_depth=2, learning_rate=.03, min_child_weight=3,
                           reg_lambda=10, subsample=1., colsample_bytree=1., tree_method="hist",
                           n_jobs=1, random_state=SEED)
            if parameters:
                unknown = set(parameters) - set(cls().get_params())
                if unknown:
                    raise ValueError(f"Unknown XGBoost parameters: {sorted(unknown)}")
                options.update(parameters)
            model = cls(**options)
        else:
            cls = RandomForestClassifier if kind == "classification" else RandomForestRegressor
            model = cls(n_estimators=300, max_depth=4, min_samples_leaf=3, max_features=1., n_jobs=1, random_state=SEED)
        result["estimator"] = model.fit(x, y, sample_weight=weights)
    else:
        torch.manual_seed(SEED)
        torch.use_deterministic_algorithms(True)
        model = neural_model(name, x.shape[1], 2 if kind == "classification" else 1)
        tensor = torch.from_numpy(x)
        model.initialize(tensor)
        center = float(y.mean()) if kind == "regression" else 0.
        scale = max(float(y.std()), 1.) if kind == "regression" else 1.
        target = torch.tensor(y, dtype=torch.long) if kind == "classification" else torch.tensor((y-center)/scale, dtype=torch.float32)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=1e-4)
        model.train()
        weighted = torch.from_numpy(weights)
        rng = np.random.default_rng(SEED)
        batch_size = 512 if len(set(ids)) < len(ids) else len(y)
        updates = 0
        for _ in range(epochs):
            for index in np.array_split(rng.permutation(len(y)), max(1, math.ceil(len(y)/batch_size))):
                optimizer.zero_grad()
                output, penalty = model(tensor[index])
                loss = F.cross_entropy(output, target[index], reduction="none") if kind == "classification" else F.mse_loss(output[:, 0], target[index], reduction="none")
                loss = (loss * weighted[index]).mean() + (.001 * penalty if name == "tabnet" else 0.)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
                updates += 1
        result.update(optimizer_updates=updates, batch_size=batch_size, epochs=epochs)
        result.update(state=model.state_dict(), features=x.shape[1], target_center=center, target_scale=scale)
    return result


def predict(model, x):
    raw = x
    x, _ = preprocess(raw, model["preprocess"])
    if model.get("encoder") is not None:
        x = np.concatenate([x, embedding(model["encoder"], model["preprocess"]["scaler"].inverse_transform(x))], axis=1)
    if "constant" in model:
        return np.full(len(x), model["constant"])
    classification = model["kind"] == "classification"
    if "estimator" in model:
        return model["estimator"].predict_proba(x)[:, 1] if classification else model["estimator"].predict(x)
    network = neural_model(model["name"], model["features"], 2 if classification else 1)
    network.load_state_dict(model["state"])
    network.eval()
    with torch.no_grad():
        output, _ = network(torch.from_numpy(x))
    return output.softmax(-1)[:, 1].numpy() if classification else output[:, 0].numpy() * model["target_scale"] + model["target_center"]


def metrics(y, prediction, kind):
    if prediction.shape != y.shape or not np.isfinite(prediction).all():
        raise ValueError("Invalid predictions")
    if kind == "regression":
        return dict(mae=float(mean_absolute_error(y, prediction)),
                    rmse=float(np.sqrt(np.mean((y-prediction)**2))), r2=float(r2_score(y, prediction)))
    if ((prediction < 0) | (prediction > 1)).any():
        raise ValueError("Invalid class probability")
    tn, fp, fn, tp = confusion_matrix(y, prediction >= .5, labels=[0, 1]).ravel()
    return dict(auroc=float(roc_auc_score(y, prediction)), average_precision=float(average_precision_score(y, prediction)),
                accuracy=float(accuracy_score(y, prediction >= .5)), balanced_accuracy=float(balanced_accuracy_score(y, prediction >= .5)),
                sensitivity=float(tp/(tp+fn)), specificity=float(tn/(tn+fp)),
                log_loss=float(log_loss(y, prediction, labels=[0, 1])), confusion_matrix=[[int(tn), int(fp)], [int(fn), int(tp)]])


def intervals(y, prediction, kind, baseline=None):
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(2000):
        if kind == "classification":
            ix = np.concatenate([rng.choice(np.flatnonzero(y == c), np.sum(y == c), replace=True) for c in (0, 1)])
            labels, scores = y[ix], prediction[ix]
            ranks = rankdata(scores)
            positives = labels == 1
            n = int(positives.sum())
            value = (ranks[positives].sum() - n*(n+1)/2)/(n*(len(ix)-n))
        else:
            ix = rng.integers(len(y), size=len(y))
            value = np.mean(abs(y[ix] - prediction[ix]))
            if baseline is not None:
                value = np.mean(abs(y[ix]-baseline[ix])) - value
        draws.append(float(value))
    return dict(ci95=np.quantile(draws, [.025, .975]).tolist(), ci975=np.quantile(draws, [.0125, .9875]).tolist())


def permutation_p(y, prediction):
    ranks = rankdata(prediction)
    positives = int(y.sum())
    observed = ranks[y == 1].sum()
    count = math.comb(len(y), positives)
    if count <= 20000:
        return sum(ranks[list(ix)].sum() >= observed for ix in combinations(range(len(y)), positives)) / count
    rng = np.random.default_rng(SEED)
    exceed = sum(ranks[rng.choice(len(y), positives, replace=False)].sum() >= observed for _ in range(10000))
    return (1 + exceed) / 10001


def source_hashes():
    paths = [Path(__file__), Path(__file__).with_name("baseline.py"), ROOT / "prototype_net/failed/exp3/node.py", ROOT / "prototype_net/failed/exp3/tabnet.py"]
    return {str(p.relative_to(ROOT)): sha256(p) for p in paths}


def load_role(out, manifest, task, role):
    path = out / task / f"{role}.npz"
    info = manifest["tasks"][task]
    if sha256(path) != info["cache_sha256"][role]:
        raise ValueError(f"Changed feature cache: {task}/{role}")
    with np.load(path, allow_pickle=False) as z:
        values = {k: z[k] for k in z.files}
    expected = manifest["cohorts"][info["cohort"]]["splits"][role]
    if list(values["ids"]) != expected:
        raise ValueError("Participant order mismatch")
    if "views" in values:
        if set(values["owners"]) != set(expected) or len(values["owners"]) != len(values["views"]):
            raise ValueError("Window ownership crosses participant roles")
        if values["views"].shape[1] != values["x"].shape[1] or not np.isfinite(values["views"]).all():
            raise ValueError("Invalid window features")
    return values


def training_rows(data, index):
    if "views" not in data:
        return data["x"][index], data["y"][index], data["ids"][index]
    mask = np.isin(data["owners"], data["ids"][index])
    lookup = dict(zip(data["ids"], data["y"]))
    owners = data["owners"][mask]
    return data["views"][mask], np.asarray([lookup[uid] for uid in owners]), owners


def predict_people(model, data, index=None):
    ids = data["ids"] if index is None else data["ids"][index]
    if "constant" in model:
        return np.full(len(ids), model["constant"])
    if "views" not in data:
        return predict(model, data["x"] if index is None else data["x"][index])
    mask = np.isin(data["owners"], ids)
    scores = predict(model, data["views"][mask])
    owners = data["owners"][mask]
    return np.asarray([scores[owners == uid].mean() for uid in ids])


def train_candidate(out, task, name, augmentation, representation, epochs):
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        manifest = json.loads((out / "manifest.json").read_text())
        registration_hash = sha256(out / "registration.json")
        data = load_role(out, manifest, task, "train")
        x, y, ids = data["x"], data["y"], data["ids"]
        kind = manifest["tasks"][task]["kind"]
        folds = list(StratifiedKFold(3, shuffle=True, random_state=SEED).split(x, data["diagnosis"]))
        fold_ids = [{"fit": ids[a].tolist(), "score": ids[b].tolist()} for a, b in folds]
        candidate = f"{name}_{augmentation}_{representation}"
        checkpoint = out / task / f"{candidate}.joblib"
        if checkpoint.exists():
            item = joblib.load(checkpoint)
            if item["registration_sha256"] != registration_hash or item["folds"] != fold_ids:
                raise ValueError("Checkpoint provenance mismatch")
        else:
            auxiliary = joblib.load(out / "encoder.joblib") if representation == "pretrained" else None
            started = time.monotonic()
            oof = np.empty(len(y))
            for fit, score in folds:
                assert not set(ids[fit]) & set(ids[score])
                model = fit_model(name, *training_rows(data, fit), kind, augmentation, epochs, auxiliary)
                assert all(a in ids[fit] and b in ids[fit] for a, b, _ in model["parents"])
                oof[score] = predict_people(model, data, score)
            model = fit_model(name, *training_rows(data, np.arange(len(ids))), kind, augmentation, epochs, auxiliary)
            item = dict(model=model, oof=oof, training_oof=metrics(y, oof, kind), training_fit=metrics(y, predict_people(model, data), kind),
                        folds=fold_ids, registration_sha256=registration_hash, seconds=time.monotonic()-started)
            temporary = checkpoint.with_suffix(".tmp")
            joblib.dump(item, temporary, compress=3)
            temporary.replace(checkpoint)
            print(task, candidate, item["training_oof"], flush=True)
        return str(checkpoint.relative_to(out)), sha256(checkpoint)


def train(out, epochs):
    manifest = json.loads((out / "manifest.json").read_text())
    if (out / "evaluation_opened.json").exists() or (out / "freeze.json").exists():
        raise ValueError("This experiment is sealed; training cannot resume")
    if not (out / "registration.json").exists():
        (out / "README.md").write_bytes(Path(__file__).with_name("README.md").read_bytes())
    registration = dict(seed=SEED, epochs=epochs, models=list(MODELS), augmentations=manifest.get("augmentations", ["none", "smote"]),
                        threshold=.5, folds=3, selection="none; evaluate and report every fixed candidate equally",
                        source_hashes=source_hashes(), manifest_sha256=sha256(out / "manifest.json"),
                        protocol_sha256=sha256(out / "README.md"),
                        aalto_sha256=sha256(out / "aalto.npz") if (out / "aalto.npz").exists() else None,
                        environment=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__,
                                         torch=torch.__version__, xgboost=xgboost.__version__, joblib=joblib.__version__),
                        inference="Descriptive per-candidate results; no winner selection or automatic credibility gate.")
    path = out / "registration.json"
    if path.exists():
        if json.loads(path.read_text()) != registration:
            raise ValueError("Registered protocol changed")
    else:
        write_json(path, registration)
    archive = out / "source.zip"
    if not archive.exists():
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as z:
            for name in registration["source_hashes"]:
                z.write(ROOT / name, name)
            z.write(out / "README.md", "README.md")
    with zipfile.ZipFile(archive) as z:
        for name, digest in registration["source_hashes"].items():
            if hashlib.sha256(z.read(name)).hexdigest() != digest:
                raise ValueError("Archived source mismatch")
    artifacts = {}
    if manifest.get("frozen_encoder_sha256"):
        if sha256(out / "encoder.joblib") != manifest["frozen_encoder_sha256"]:
            raise ValueError("Frozen source encoder changed")
        encoder = joblib.load(out / "encoder.joblib")
    else:
        encoder = pretrain(out) if registration["aalto_sha256"] else None
    if encoder is not None:
        artifacts["encoder.joblib"] = sha256(out / "encoder.joblib")
    if manifest.get("synthetic_evaluation"):
        artifacts.update(save_distribution(out, manifest))
    jobs = []
    for task, info in manifest["tasks"].items():
        representations = ("raw", "pretrained") if encoder is not None and info["cohort"] != "cognitive_ci" and info.get("use_encoder", True) else ("raw",)
        candidates = [("dummy", "none", "raw")] + [(m, a, r) for m in MODELS for a in (registration["augmentations"] if info["kind"] == "classification" else ("none",)) for r in representations]
        jobs.extend((out, task, *candidate, epochs) for candidate in candidates)
    if manifest.get("workers", 1) == 1:
        artifacts.update(train_candidate(*job) for job in jobs)
    else:
        with ProcessPoolExecutor(max_workers=manifest["workers"]) as pool:
            futures = [pool.submit(train_candidate, *job) for job in jobs]
            artifacts.update(future.result() for future in futures)
    write_json(out / "freeze.json", dict(registration_sha256=sha256(path), artifacts=artifacts))


def evaluate(out):
    manifest = json.loads((out / "manifest.json").read_text())
    registration = json.loads((out / "registration.json").read_text())
    freeze = json.loads((out / "freeze.json").read_text())
    if source_hashes() != registration["source_hashes"] or sha256(out / "manifest.json") != registration["manifest_sha256"]:
        raise ValueError("Source or manifest changed after registration")
    if sha256(out / "registration.json") != freeze["registration_sha256"]:
        raise ValueError("Registration changed after freeze")
    if sha256(out / "README.md") != registration["protocol_sha256"]:
        raise ValueError("Registered protocol document changed")
    for name, digest in freeze["artifacts"].items():
        if sha256(out / name) != digest:
            raise ValueError(f"Changed model: {name}")
    write_json(out / "evaluation_opened.json", dict(freeze_sha256=sha256(out / "freeze.json"), opened_at=time.time()))
    report = dict(tasks={})
    for task, info in manifest["tasks"].items():
        kind = info["kind"]
        report["tasks"][task] = {}
        for role in ("validation", "test"):
            data = load_role(out, manifest, task, role)
            result = {}
            predictions = dict(ids=data["ids"], y=data["y"])
            synthetic = None
            if manifest.get("synthetic_evaluation"):
                distribution = joblib.load(out / "distribution.joblib")["distribution"]
                synthetic = sample_distribution(distribution, data["x"], data["y"], data["ids"],
                                                manifest["synthetic_evaluation"][role], SEED + ROLES.index(role))
                if set(synthetic["owners"]) != set(data["ids"]):
                    raise ValueError("Synthetic evaluation ownership mismatch")
                np.savez_compressed(out / task / f"{role}_synthetic.npz", **synthetic)
                synthetic_predictions = dict(owners=synthetic["owners"], y=synthetic["y"])
            for name in sorted(freeze["artifacts"]):
                if Path(name).parent.name != task:
                    continue
                candidate = Path(name).stem
                item = joblib.load(out / name)
                prediction = predict_people(item["model"], data)
                predictions[candidate] = prediction
                result[candidate] = dict(**metrics(data["y"], prediction, kind), uncertainty=intervals(data["y"], prediction, kind))
                if synthetic is not None:
                    scores = predict(item["model"], synthetic["x"])
                    synthetic_predictions[candidate] = scores
                    grouped = np.full(len(data["ids"]), item["model"]["constant"]) if "constant" in item["model"] else np.array([scores[synthetic["owners"] == uid].mean() for uid in data["ids"]])
                    result[candidate]["synthetic"] = dict(rows=len(scores), source_people=len(data["ids"]),
                        row_metrics=metrics(synthetic["y"], scores, kind), participant_metrics=metrics(data["y"], grouped, kind),
                        participant_uncertainty=intervals(data["y"], grouped, kind))
            report["tasks"][task][role] = result
            np.savez_compressed(out / task / f"{role}_predictions.npz", **predictions)
            if synthetic is not None:
                np.savez_compressed(out / task / f"{role}_synthetic_predictions.npz", **synthetic_predictions)
        print(f"Evaluated all {len(result)} candidates for {task}", flush=True)
    write_json(out / "results.json", report)


def comparison(out):
    manifest = json.loads((out / "manifest.json").read_text())
    registration = json.loads((out / "registration.json").read_text())
    freeze = json.loads((out / "freeze.json").read_text())
    results = json.loads((out / "results.json").read_text())
    if sha256(out / "registration.json") != freeze["registration_sha256"]:
        raise ValueError("Registration changed after freeze")
    if sha256(out / "manifest.json") != registration["manifest_sha256"]:
        raise ValueError("Manifest changed after registration")
    rows = []
    for task, info in manifest["tasks"].items():
        validation, test = (results["tasks"][task][r] for r in ("validation", "test"))
        names = {Path(n).stem for n in freeze["artifacts"] if Path(n).parent.name == task}
        if names != set(validation) or names != set(test):
            raise ValueError(f"Incomplete candidate evaluation: {task}")
        counts = {r: len(ids) for r, ids in manifest["cohorts"][info["cohort"]]["splits"].items()}
        for name in sorted(names):
            path = out / task / f"{name}.joblib"
            if sha256(path) != freeze["artifacts"][str(path.relative_to(out))]:
                raise ValueError(f"Changed model: {path}")
            item = joblib.load(path)
            if item["registration_sha256"] != freeze["registration_sha256"]:
                raise ValueError("Checkpoint provenance mismatch")
            model, augmentation, representation = name.rsplit("_", 2)
            fitted = item["model"]
            synthetic = len(fitted["parents"])
            if fitted["real_people"] != counts["train"] or fitted["fit_rows"] != fitted.get("real_rows", counts["train"]) + synthetic:
                raise ValueError("Training population mismatch")
            if model == "dummy":
                data = load_role(out, manifest, task, "train")
                oof = np.empty(len(data["y"]))
                for fold in item["folds"]:
                    fit = np.isin(data["ids"], fold["fit"])
                    score = np.isin(data["ids"], fold["score"])
                    oof[score] = data["y"][fit].mean()
                item["training_fit"] = metrics(data["y"], predict_people(fitted, data), info["kind"])
                item["training_oof"] = metrics(data["y"], oof, info["kind"])
            rows.append(dict(task=task, kind=info["kind"], candidate=name, model=model,
                             augmentation=augmentation, representation=representation, counts=counts,
                             synthetic_rows=synthetic, real_rows=fitted.get("real_rows", counts["train"]), fit_rows=fitted["fit_rows"],
                             training_fit=item["training_fit"], training_oof=item["training_oof"],
                             validation=validation[name], test=test[name]))
    report = dict(seed=registration["seed"], selection="none", rows=rows,
                  provenance={name: sha256(out / name) for name in ("registration.json", "freeze.json", "results.json")})
    write_json(out / "comparison.json", report)
    print(f"Reported all {len(rows)} candidates from stored metrics; no fitting or prediction.")
    if manifest.get("synthetic_evaluation"):
        combined(out)


def combined(out):
    manifest = json.loads((out / "manifest.json").read_text())
    registration = json.loads((out / "registration.json").read_text())
    freeze = json.loads((out / "freeze.json").read_text())
    previous = json.loads((out / "comparison.json").read_text())
    if not manifest.get("synthetic_evaluation") or set(manifest["tasks"]) != {"motor_pd"}:
        raise ValueError("Combined reporting requires the clinical distribution experiment")
    if sha256(out / "manifest.json") != registration["manifest_sha256"] or sha256(out / "registration.json") != freeze["registration_sha256"]:
        raise ValueError("Registered inputs changed")
    for name, digest in {**freeze["artifacts"], **previous["provenance"]}.items():
        if sha256(out / name) != digest:
            raise ValueError(f"Changed input: {name}")
    with zipfile.ZipFile(out / "source.zip") as archive:
        for name, digest in registration["source_hashes"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError("Original training source archive changed")
    target = out / "combined"
    target.mkdir(exist_ok=False)
    count = max(int(a[4:]) for a in manifest["augmentations"] if a.startswith("rose"))
    parts, counts, inputs = [], {}, {}
    for role in ROLES:
        real = load_role(out, manifest, "motor_pd", role)
        x, y, owners = training_rows(real, np.arange(len(real["ids"])))
        path = out / f"synthetic_train_{count}.npz" if role == "train" else out / "motor_pd" / f"{role}_synthetic.npz"
        inputs[str(path.relative_to(out))] = sha256(path)
        with np.load(path, allow_pickle=False) as z:
            synthetic = {key: z[key] for key in z.files}
        labels = dict(zip(real["ids"], real["y"]))
        if set(synthetic["owners"]) != set(real["ids"]) or not np.array_equal(synthetic["y"], [labels[uid] for uid in synthetic["owners"]]):
            raise ValueError("Synthetic source roles or labels changed")
        n = len(x) + len(synthetic["x"])
        owner = np.concatenate([owners, synthetic["owners"]])
        frequencies = Counter(owner)
        weight = np.asarray([1 / frequencies[uid] for uid in owner], dtype=np.float32)
        weight *= len(weight) / weight.sum()
        parts.append(dict(x=np.concatenate([x, synthetic["x"]]), y=np.concatenate([y, synthetic["y"]]),
                          owner=owner, split=np.full(n, role), synthetic=np.r_[np.zeros(len(x), dtype=bool), np.ones(len(synthetic["x"]), dtype=bool)],
                          sample_weight=weight))
        counts[role] = dict(real_rows=len(x), synthetic_rows=len(synthetic["x"]), rows=n, source_people=len(real["ids"]))
    dataset = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    dataset["row_id"] = np.arange(len(dataset["y"]))
    for i, left in enumerate(ROLES):
        for right in ROLES[i+1:]:
            if set(dataset["owner"][dataset["split"] == left]) & set(dataset["owner"][dataset["split"] == right]):
                raise ValueError("Source participant crosses combined partitions")
    np.savez_compressed(target / "dataset.npz", **dataset)
    predictions, rows = {}, []
    for role in ("validation", "test"):
        mask = dataset["split"] == role
        predictions[f"{role}_row_id"] = dataset["row_id"][mask]
        real = load_role(out, manifest, "motor_pd", role)
        path = out / "motor_pd" / f"{role}_synthetic_predictions.npz"
        inputs[str(path.relative_to(out))] = sha256(path)
        with np.load(path, allow_pickle=False) as z:
            saved = {key: z[key] for key in z.files}
        if not np.array_equal(saved["owners"], dataset["owner"][mask][counts[role]["real_rows"]:]):
            raise ValueError("Saved synthetic prediction order changed")
        if not np.array_equal(saved["y"], dataset["y"][mask][counts[role]["real_rows"]:]):
            raise ValueError("Saved synthetic prediction labels changed")
        for row in previous["rows"]:
            name = row["candidate"]
            model = joblib.load(out / "motor_pd" / f"{name}.joblib")["model"]
            scores = np.concatenate([predict(model, real["views"]), saved[name]])
            predictions[f"{role}_{name}"] = scores
            owners = dataset["owner"][mask]
            grouped = np.full(len(real["ids"]), model["constant"]) if "constant" in model else np.asarray([scores[owners == uid].mean() for uid in real["ids"]])
            rows.append(dict(candidate=name, role=role, model=row["model"], augmentation=row["augmentation"],
                             representation=row["representation"], fitted_training_rows=row["fit_rows"],
                             **metrics(dataset["y"][mask], scores, "classification"),
                             source_participant_metrics=metrics(real["y"], grouped, "classification"),
                             source_participant_uncertainty=intervals(real["y"], grouped, "classification")))
    np.savez_compressed(target / "predictions.npz", **predictions)
    write_json(target / "results.json", dict(counts=counts, rows=rows, seed=SEED, threshold=.5,
        metric_unit="Combined real and synthetic rows; uncertainty is reported only at source-participant level.",
        training="Maximum-generation models already fitted these exact combined training rows. Other candidates retain their registered augmentation levels; no retraining or tuning.",
        provenance=dict(input_hashes={**inputs, **{name: sha256(out / name) for name in ("manifest.json", "registration.json", "freeze.json", "comparison.json")}},
                        dataset_sha256=sha256(target / "dataset.npz"), predictions_sha256=sha256(target / "predictions.npz"),
                        trained_source_hashes=registration["source_hashes"], reporting_source_hashes=source_hashes())))
    with zipfile.ZipFile(target / "source.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in source_hashes():
            archive.write(ROOT / name, name)
    print(f"Combined real + synthetic dataset: {counts}; evaluated all {len(previous['rows'])} candidates.")


def typenet_sequences(base, neutral_key, synthetic=128):
    source = base / "neuroqwerty/neuroQWERTY.zip"
    sequences, lengths, lineage, owners, labels, origins, events = [], [], [], [], [], [], []
    audit = Counter()
    symbols = {"space", "period", "comma", "colon", "semicolon", "minus", "plus", "less", "greater",
               "exclam", "exclamdown", "question", "questiondown", "quotedbl", "apostrophe", "grave",
               "parenleft", "parenright", "underscore", "ccedilla", "masculine", "periodcentered"}

    def append(rows, owner, label, origin):
        rows = np.asarray(rows)
        count = min(50, len(rows))
        seq = np.zeros((50, 5), dtype=np.float32)
        seq[:count, 0] = rows[:count, 0]
        seq[:count, 4] = rows[:count, 2] / 255
        n = min(50, len(rows)-1)
        seq[:n, 2] = rows[:n, 1]
        seq[:n, 1] = rows[:n, 1] - rows[:n, 0]
        seq[:n, 3] = rows[:n, 1] + rows[1:n+1, 0] - rows[:n, 0]
        ancestor = np.full(51, -1, dtype=np.int64)
        ancestor[:len(rows)] = rows[:, 3].astype(np.int64)
        sequences.append(seq)
        lengths.append(count)
        lineage.append(ancestor)
        owners.append(owner)
        labels.append(label)
        origins.append(origin)

    with zipfile.ZipFile(source) as archive:
        for study in ("MIT-CS1PD", "MIT-CS2PD"):
            people = csv.DictReader(io.StringIO(archive.read(f"{study}/GT_DataPD_{study}.csv").decode()))
            for person in people:
                owner, label = "nq:" + person["pID"], int(person["gt"] == "True")
                recordings = []
                for column, name in person.items():
                    if not column.startswith("file_") or not name:
                        continue
                    recording, seen = [], set()
                    for row in csv.reader(io.StringIO(archive.read(f"{study}/data_{study}/{name}").decode())):
                        audit["raw_events"] += 1
                        try:
                            key = row[0]
                            if not ((len(key) == 1 and key.isascii() and key.isalnum()) or key in symbols):
                                audit["excluded_nonprinting_or_unmapped_keys"] += 1
                                continue
                            hold, release, press = map(float, row[1:])
                            if not all(map(math.isfinite, (hold, release, press))) or not (hold > 0 and release >= press >= 0):
                                raise ValueError("invalid timing")
                            if abs(hold-(release-press)) > .001:
                                raise ValueError("timestamp mismatch")
                            token = (key, hold, release, press)
                            if token in seen:
                                audit["duplicates"] += 1
                                continue
                            seen.add(token)
                            code = ord(key.upper()) if len(key) == 1 else 32 if key == "space" else neutral_key
                            if code == neutral_key:
                                audit["neutral_symbol_codes"] += 1
                            recording.append((press, release-press, code))
                        except (ValueError, IndexError):
                            audit["invalid_events"] += 1
                    recording.sort(key=lambda row: row[0])
                    if not recording:
                        continue
                    array = np.asarray(recording, dtype=float)
                    event_ids = np.arange(len(events), len(events)+len(array))
                    intervals = np.r_[np.diff(array[:, 0]), 0.]
                    primitives = np.column_stack([array[:, 1], intervals, array[:, 2], event_ids])
                    events.extend((owner, name, str(i), str(p), str(h), str(k)) for i, (p, h, k) in enumerate(array))
                    recordings.append(primitives)
                    for start in range(0, len(primitives), 50):
                        append(primitives[start:start+51], owner, label, False)
                eligible = [r for r in recordings if len(r) >= 26]
                if not eligible:
                    raise ValueError(f"Insufficient contiguous strokes: {owner}")
                starts = np.asarray([len(r)-25 for r in eligible])
                seed = SEED + int(hashlib.sha256(owner.encode()).hexdigest()[:8], 16)
                rng = np.random.default_rng(seed)
                for _ in range(synthetic):
                    pieces = []
                    for size in (25, 26):
                        recording = eligible[int(rng.choice(len(eligible), p=starts/starts.sum()))]
                        start = int(rng.integers(len(recording)-25))
                        pieces.append(recording[start:start+size])
                    append(np.concatenate(pieces), owner, label, True)
    audit["retained_events"] = len(events)
    return dict(sequences=np.asarray(sequences), lengths=np.asarray(lengths), lineage=np.asarray(lineage),
                owner=np.asarray(owners), y=np.asarray(labels), synthetic=np.asarray(origins),
                events=np.asarray(events)), dict(audit)


def typenet_fidelity(data, mask):
    groups = []
    for uid in np.unique(data["owner"][mask]):
        values = []
        for synthetic in (False, True):
            ix = np.flatnonzero(mask & (data["owner"] == uid) & (data["synthetic"] == synthetic))
            strokes = np.concatenate([data["sequences"][i, :data["lengths"][i]] for i in ix])
            # Last real event has no successor. Exclude it from interval diagnostics.
            strokes = strokes[strokes[:, 2] > 0].astype(float)
            values.append(strokes)
        groups.append(values)
    result = {}
    for label in (0, 1):
        subset = [g for uid, g in zip(np.unique(data["owner"][mask]), groups)
                  if data["y"][np.flatnonzero(data["owner"] == uid)[0]] == label]
        stats = []
        for origin in (0, 1):
            stats.append(np.mean([np.r_[g[origin][:, [0, 2]].mean(0),
                       np.quantile(g[origin][:, [0, 2]], [.1, .5, .9], axis=0).ravel(),
                       np.corrcoef(np.log1p(g[origin][:, [0, 2]]).T)[0, 1]] for g in subset], axis=0))
        a, b = stats
        result[str(label)] = dict(real=a.tolist(), synthetic=b.tolist(),
                                  relative_change=((b[:8]-a[:8])/np.maximum(abs(a[:8]), 1e-8)).tolist())
    lags = []
    for origin in (False, True):
        ix = np.flatnonzero(mask & (data["synthetic"] == origin))
        correlations = []
        for i in ix:
            hold = np.log(data["sequences"][i, :data["lengths"][i], 0].astype(float))
            if len(hold) > 3 and hold[:-1].std() > 0 and hold[1:].std() > 0:
                correlations.append(float(np.corrcoef(hold[:-1], hold[1:])[0, 1]))
        lags.append(float(np.mean(correlations)))
    generated = data["sequences"][mask & data["synthetic"]]
    generated_lineage = data["lineage"][mask & data["synthetic"]]
    next_hold = data["events"][generated_lineage[:, 1:], 4].astype(float)
    if not np.allclose(generated[:, :, 3], generated[:, :, 2] + next_hold - generated[:, :, 0], rtol=1e-4, atol=1e-4):
        raise ValueError("Synthetic release identity violated")
    unique = len({hashlib.sha256(row.tobytes()).digest() for row in generated})/len(generated)
    if unique < .99 or not np.isfinite(generated).all():
        raise ValueError("Synthetic sequence quality check failed")
    if not np.allclose(generated[:, :, 1], generated[:, :, 2]-generated[:, :, 0], rtol=1e-4, atol=1e-4):
        raise ValueError("Synthetic timing identity violated")
    return dict(fields=["hold_mean", "press_mean", "hold_q10", "press_q10", "hold_median", "press_median",
                        "hold_q90", "press_q90", "log_hold_press_correlation"], by_label=result,
                mean_log_hold_lag1=dict(real=lags[0], synthetic=lags[1]), unique_fraction=unique)


def prepare_typenet(out, baseline, epochs):
    from paper_typenet.nn import TypeNetEncoder
    out.mkdir(parents=True, exist_ok=False)
    checkpoint = ROOT / "paper_typenet/weights/typenet_68k_m50_g10_64x512_best_weights.pt"
    digest = sha256(checkpoint)
    if digest != "04b7c11aa4a91fcf8f19027b09527c9962f31a1f4ec68913a0359b18e75696b7":
        raise ValueError("Unexpected TypeNet checkpoint")
    encoder = TypeNetEncoder()
    encoder.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    encoder.eval().requires_grad_(False)
    before = {k: v.clone() for k, v in encoder.state_dict().items()}
    data, audit = typenet_sequences(ROOT / "data/clinical", float(encoder.input_batch_norm.running_mean[4])*255)
    for i, ancestors in enumerate(data["lineage"]):
        assert np.all(data["events"][ancestors[ancestors >= 0], 0] == data["owner"][i])
    original = json.loads((baseline / "manifest.json").read_text())
    splits = original["cohorts"]["motor_pd"]["splits"]
    assert len(set(uid for role in ROLES for uid in splits[role])) == sum(map(len, splits.values()))
    assert set(data["owner"]) == set(uid for role in ROLES for uid in splits[role])
    data["split"] = np.asarray([next(role for role in ROLES if uid in splits[role]) for uid in data["owner"]])
    embedded = []
    with torch.inference_mode():
        for start in range(0, len(data["owner"]), 256):
            embedded.append(encoder(torch.from_numpy(data["sequences"][start:start+256]),
                                    torch.from_numpy(data["lengths"][start:start+256])).numpy())
    assert all(torch.equal(v, before[k]) for k, v in encoder.state_dict().items())
    assert sha256(checkpoint) == digest
    stats = np.stack([summary(seq[:length, 0]) for seq, length in zip(data["sequences"], data["lengths"])])
    data["x"] = np.concatenate([np.concatenate(embedded), stats], axis=1).astype(np.float32)
    assert np.isfinite(data["x"]).all()
    counts = {role: dict(people=len(splits[role]), real=int(np.sum((data["split"] == role) & ~data["synthetic"])),
                        synthetic=int(np.sum((data["split"] == role) & data["synthetic"]))) for role in ROLES}
    people = np.asarray(splits["train"])
    labels = np.array([data["y"][np.flatnonzero(data["owner"] == uid)[0]] for uid in people])
    folds = []
    for repeat in range(5):
        for fold, (fit, score) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED+repeat).split(people, labels)):
            folds.append(dict(repeat=repeat+1, fold=fold+1, fit=people[fit].tolist(), score=people[score].tolist()))
    sources = {**source_hashes(), "paper_typenet/nn.py": sha256(ROOT / "paper_typenet/nn.py")}
    np.savez_compressed(out / "dataset.npz", **data)
    registration = dict(seed=SEED, repeats=5, folds=folds, families=list(MODELS), augmentation=[0, 128], epochs=epochs,
                        selection=["combined_cv_accuracy", "combined_cv_auroc", "less_augmentation", "family_order"],
                        threshold=.5, splits=splits, counts=counts, audit=audit, features=135,
                        encoder=str(checkpoint.relative_to(ROOT)), encoder_sha256=digest, encoder_trainable_parameters=0,
                        dataset_sha256=sha256(out / "dataset.npz"), source_hashes=sources,
                        input_sha256=sha256(ROOT / "data/clinical/neuroqwerty/neuroQWERTY.zip"),
                        baseline_manifest_sha256=sha256(baseline / "manifest.json"),
                        fidelity=typenet_fidelity(data, data["split"] == "train"))
    write_json(out / "registration.json", registration)
    with zipfile.ZipFile(out / "source.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sources:
            archive.write(ROOT / path, path)
        archive.write(Path(__file__).parent / "README.md", "README.md")
    print(json.dumps(dict(counts=counts, fidelity=registration["fidelity"])), flush=True)


def typenet_scores(data, mask, prediction):
    owner, y, synthetic = (data[key][mask] for key in ("owner", "y", "synthetic"))
    people = np.unique(owner)
    real = np.array([prediction[(owner == uid) & ~synthetic].mean() for uid in people])
    labels = np.array([y[np.flatnonzero(owner == uid)[0]] for uid in people])
    return dict(combined=metrics(y, prediction, "classification"),
                real_participants=metrics(labels, real, "classification"))


def typenet_fold(args):
    out, index = args
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        registration = json.loads((out / "registration.json").read_text())
        fold = registration["folds"][index]
        with np.load(out / "dataset.npz", allow_pickle=False) as cache:
            keep = cache["split"] == "train"
            data = {key: cache[key][keep] for key in ("x", "y", "owner", "synthetic")}
        fit, score = np.isin(data["owner"], fold["fit"]), np.isin(data["owner"], fold["score"])
        assert not np.any(fit & score) and np.all(fit | score)
        models, results, selected = {}, [], {}
        for family in MODELS:
            candidates = []
            for count in registration["augmentation"]:
                mask = fit & ((~data["synthetic"]) | (count > 0))
                name = f"{family}_{count}"
                model = fit_model(family, data["x"][mask], data["y"][mask], data["owner"][mask],
                                  "classification", "none", registration["epochs"])
                model["real_rows"] = int(np.sum(mask & ~data["synthetic"]))
                model["synthetic_rows"] = int(np.sum(mask & data["synthetic"]))
                model["augmentation"] = f"blocks{count}"
                prediction = predict(model, data["x"][score])
                scores = typenet_scores(data, score, prediction)
                result = dict(candidate=name, family=family, synthetic_per_person=count, scores=scores,
                              real_rows=model["real_rows"], synthetic_rows=model["synthetic_rows"],
                              train_people=model["real_people"])
                results.append(result)
                candidates.append(result)
                models[name] = model
            selected[family] = max(candidates, key=lambda r: (r["scores"]["combined"]["accuracy"],
                                            r["scores"]["combined"]["auroc"], -r["synthetic_per_person"]))["candidate"]
        overall = max(results, key=lambda r: (r["scores"]["combined"]["accuracy"],
                      r["scores"]["combined"]["auroc"], -r["synthetic_per_person"], -MODELS.index(r["family"])))
        target = out / f"fold{index+1:02d}.joblib"
        if target.exists():
            raise FileExistsError(target)
        joblib.dump(dict(models=models, results=results, selected=selected, overall=overall["candidate"],
                         fold=fold, registration_sha256=sha256(out / "registration.json")), target, compress=3)
        return dict(index=index+1, sha256=sha256(target), selected=selected, overall=overall["candidate"])


def typenet_verify(out):
    registration = json.loads((out / "registration.json").read_text())
    for name, digest in registration["source_hashes"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f"Source changed: {name}")
    if sha256(ROOT / registration["encoder"]) != registration["encoder_sha256"]:
        raise ValueError("Frozen encoder changed")
    if sha256(out / "dataset.npz") != registration["dataset_sha256"]:
        raise ValueError("Dataset changed")
    return registration


def train_typenet(out):
    typenet_verify(out)
    if (out / "freeze.json").exists() or list(out.glob("fold*.joblib")):
        raise FileExistsError("Training artifacts already exist")
    from concurrent.futures import as_completed
    records = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(typenet_fold, (out, i)) for i in range(25)]
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(json.dumps(record), flush=True)
    write_json(out / "freeze.json", dict(registration_sha256=sha256(out / "registration.json"),
                                        models=sorted(records, key=lambda r: r["index"])))


def evaluate_typenet(out):
    registration = typenet_verify(out)
    freeze = json.loads((out / "freeze.json").read_text())
    assert freeze["registration_sha256"] == sha256(out / "registration.json")
    if (out / "results.json").exists():
        raise FileExistsError("Evaluation already completed")
    with np.load(out / "dataset.npz", allow_pickle=False) as cache:
        data = {key: cache[key] for key in ("x", "y", "owner", "synthetic", "split")}
    rows, cv, predictions = [], [], {}
    for record in freeze["models"]:
        path = out / f"fold{record['index']:02d}.joblib"
        assert sha256(path) == record["sha256"]
        bundle = joblib.load(path)
        assert bundle["registration_sha256"] == freeze["registration_sha256"]
        cv.extend(dict(index=record["index"], **row) for row in bundle["results"])
        for role in ("validation", "test"):
            mask = data["split"] == role
            cached = {}
            for family, candidate in {**bundle["selected"], "cv_selected": bundle["overall"]}.items():
                if candidate not in cached:
                    scores = predict(bundle["models"][candidate], data["x"][mask])
                    cached[candidate] = typenet_scores(data, mask, scores)
                    predictions[f"fold{record['index']:02d}_{role}_{candidate}"] = scores
                rows.append(dict(index=record["index"], repeat=bundle["fold"]["repeat"], fold=bundle["fold"]["fold"],
                                 family=family, candidate=candidate, role=role, **cached[candidate]))
    aggregates = []
    for role in ("validation", "test"):
        for family in (*MODELS, "cv_selected"):
            group = [row for row in rows if row["role"] == role and row["family"] == family]
            aggregate = dict(role=role, family=family, models=len(group), candidates=dict(Counter(r["candidate"] for r in group)))
            for unit in ("combined", "real_participants"):
                aggregate[unit] = {metric: dict(mean=float(np.mean([r[unit][metric] for r in group])),
                        sd=float(np.std([r[unit][metric] for r in group], ddof=1)))
                        for metric in ("accuracy", "balanced_accuracy", "auroc", "sensitivity", "specificity")}
            aggregates.append(aggregate)
    np.savez_compressed(out / "predictions.npz", **predictions)
    write_json(out / "results.json", dict(rows=rows, cv=cv, aggregates=aggregates, counts=registration["counts"],
        fidelity=registration["fidelity"], freeze_sha256=sha256(out / "freeze.json"),
        predictions_sha256=sha256(out / "predictions.npz"),
        sd_definition="Sample standard deviation across 25 fold models on the same fixed test cohort."))
    print(json.dumps(aggregates, indent=2), flush=True)


def external_sequences(neutral_key):
    from paper_typenet.nn import events_to_features
    path = ROOT / "data/clinical/conll/typing.csv"
    if sha256(path) != "db2166b5211b359c0bea4fb3057d8a41fecfbc98057fdc23c2f8fbe946984403":
        raise ValueError("Online English source changed")
    columns = ["key", "participant_id", "response_id", "diagnosis", "keydown", "keyup"]
    data = pd.read_csv(path, usecols=columns, keep_default_na=False)
    audit = Counter(raw_events=len(data))
    before = len(data)
    data = data.drop_duplicates(columns)
    audit["duplicates"] = before-len(data)
    seqs, lengths, owners, labels = [], [], [], []
    for uid, person in data.groupby("participant_id", sort=True):
        if person["diagnosis"].nunique() != 1 or person["diagnosis"].iloc[0] not in (0, 1):
            raise ValueError("Inconsistent external label")
        label = int(person["diagnosis"].iloc[0])
        for _, response in person.groupby("response_id", sort=True):
            events = []
            for index, row in response.iterrows():
                key = row["key"]
                if key == "spacebar":
                    key = " "
                if not (len(key) == 1 and key.isascii() and key.isprintable()):
                    audit["excluded_nonprinting_keys"] += 1
                    continue
                try:
                    press, release = float(row["keydown"]), float(row["keyup"])
                    if not (math.isfinite(press) and math.isfinite(release) and release > press >= 0):
                        raise ValueError("Invalid timing")
                except (ValueError, TypeError):
                    audit["invalid_events"] += 1
                    continue
                code = ord(key.upper()) if key.isalnum() else 32 if key == " " else neutral_key
                if code == neutral_key:
                    audit["neutral_symbol_codes"] += 1
                events.append((press, release, code, index))
            events.sort(key=lambda e: (e[0], e[3]))
            audit["retained_events"] += len(events)
            audit["retained_responses"] += bool(events)
            for start in range(0, len(events), 50):
                seq, length = events_to_features(events[start:start+51], 50)
                seqs.append(seq)
                lengths.append(length)
                owners.append(f"conll:{uid}")
                labels.append(label)
    return dict(sequences=np.asarray(seqs), lengths=np.asarray(lengths), owner=np.asarray(owners),
                y=np.asarray(labels)), dict(audit)


def validate_external(out, baseline):
    from paper_typenet.nn import TypeNetEncoder
    out.mkdir(parents=True, exist_ok=False)
    reg = json.loads((baseline / "registration.json").read_text())
    freeze = json.loads((baseline / "freeze.json").read_text())
    assert freeze["registration_sha256"] == sha256(baseline / "registration.json")
    with zipfile.ZipFile(baseline / "source.zip") as archive:
        for name, digest in reg["source_hashes"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
    checkpoint = ROOT / reg["encoder"]
    assert sha256(checkpoint) == reg["encoder_sha256"]
    models, info = [], []
    for record in freeze["models"]:
        path = baseline / f"fold{record['index']:02d}.joblib"
        assert sha256(path) == record["sha256"]
        bundle = joblib.load(path)
        candidate = bundle["selected"]["node"]
        assert candidate == record["selected"]["node"]
        models.append(bundle["models"][candidate])
        info.append(dict(index=record["index"], candidate=candidate, checkpoint=str(path), sha256=record["sha256"]))
    assert len(models) == 25
    registration = dict(cohort="Online English", label_source="self-reported PD", seed=SEED, threshold=.5,
        model_count=25, models=info, clinical_freeze_sha256=sha256(baseline / "freeze.json"),
        encoder_sha256=sha256(checkpoint), input_sha256=sha256(ROOT / "data/clinical/conll/typing.csv"),
        fitting="none", selection="none", aggregation="mean sequences within participant; equal-weight 25-model ensemble secondary",
        source_hashes={**source_hashes(), "paper_typenet/nn.py": sha256(ROOT / "paper_typenet/nn.py")})
    write_json(out / "registration.json", registration)
    with zipfile.ZipFile(out / "source.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in registration["source_hashes"]:
            archive.write(ROOT / name, name)
        archive.write(Path(__file__).with_name("README.md"), "README.md")
    network = TypeNetEncoder()
    network.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    network.eval().requires_grad_(False)
    before = {k: v.clone() for k, v in network.state_dict().items()}
    data, audit = external_sequences(float(network.input_batch_norm.running_mean[4])*255)
    features = []
    with torch.inference_mode():
        for start in range(0, len(data["owner"]), 256):
            features.append(network(torch.from_numpy(data["sequences"][start:start+256]),
                                    torch.from_numpy(data["lengths"][start:start+256])).numpy())
    assert all(torch.equal(v, before[k]) for k, v in network.state_dict().items())
    data["x"] = np.concatenate([np.concatenate(features), np.stack([summary(s[:n, 0])
                                  for s, n in zip(data["sequences"], data["lengths"])])], axis=1).astype(np.float32)
    assert np.isfinite(data["x"]).all()
    np.savez_compressed(out / "dataset.npz", **data)
    ids = np.unique(data["owner"])
    y = np.array([data["y"][np.flatnonzero(data["owner"] == uid)[0]] for uid in ids])
    assert not set(ids) & set(sum(reg["splits"].values(), []))
    rows, probabilities, row_scores = [], [], []
    for record, model in zip(info, models):
        score = predict(model, data["x"])
        grouped = np.array([score[data["owner"] == uid].mean() for uid in ids])
        probabilities.append(grouped)
        row_scores.append(score)
        rows.append(dict(**record, participants=metrics(y, grouped, "classification"),
                         sequences=metrics(data["y"], score, "classification")))
    probabilities = np.stack(probabilities)
    ensemble = probabilities.mean(0)
    aggregate = {k: dict(mean=float(np.mean([r["participants"][k] for r in rows])),
                       sd=float(np.std([r["participants"][k] for r in rows], ddof=1)))
                 for k in ("accuracy", "balanced_accuracy", "auroc", "sensitivity", "specificity")}
    rng = np.random.default_rng(SEED)
    boot = []
    for _ in range(2000):
        index = np.concatenate([rng.choice(np.flatnonzero(y == c), int(np.sum(y == c)), replace=True) for c in (0, 1)])
        boot.append([accuracy_score(y[index], ensemble[index] >= .5), roc_auc_score(y[index], ensemble[index])])
    np.savez_compressed(out / "predictions.npz", ids=ids, y=y, probabilities=probabilities, ensemble=ensemble,
                        sequence_probabilities=np.stack(row_scores), owner=data["owner"])
    with np.load(baseline / "dataset.npz", allow_pickle=False) as original:
        seen = {hashlib.sha256(row.tobytes()).digest() for row in original["sequences"][original["lengths"] == 50]}
    overlap = sum(hashlib.sha256(row.tobytes()).digest() in seen for row, n in zip(data["sequences"], data["lengths"]) if n == 50)
    assert overlap == 0
    for record in info:
        assert sha256(Path(record["checkpoint"])) == record["sha256"]
    assert sha256(checkpoint) == registration["encoder_sha256"]
    counts = dict(participants=len(ids), pd=int(y.sum()), controls=int(len(y)-y.sum()), real_sequences=len(data["x"]), synthetic_sequences=0)
    report = dict(counts=counts, audit=audit, results=rows, aggregate=aggregate,
                  ensemble=metrics(y, ensemble, "classification"),
                  ensemble_ci95=dict(accuracy=np.quantile(np.array(boot)[:, 0], [.025, .975]).tolist(),
                                     auroc=np.quantile(np.array(boot)[:, 1], [.025, .975]).tolist()),
                  source_overlap=dict(id_namespace_overlap=0, exact_full_sequence_overlap=overlap,
                                      note="Different studies and ID namespaces; no cross-study identity linkage is available."),
                  unchanged_checkpoints=True, registration_sha256=sha256(out / "registration.json"),
                  dataset_sha256=sha256(out / "dataset.npz"), predictions_sha256=sha256(out / "predictions.npz"))
    write_json(out / "results.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2), flush=True)


def prepare_real(out, baseline, epochs):
    baseline = baseline.resolve()
    out.mkdir(parents=True, exist_ok=False)
    original = json.loads((baseline / "typenet/registration.json").read_text())
    external = json.loads((baseline / "external/results.json").read_text())
    external_reg = json.loads((baseline / "external/registration.json").read_text())
    assert original["encoder_sha256"] == external_reg["encoder_sha256"]
    assert sha256(ROOT / original["encoder"]) == original["encoder_sha256"]
    pieces, inputs = [], {}
    for folder, cohort, digest in (
        ("typenet", "neuroqwerty", original["dataset_sha256"]),
        ("external", "online", external["dataset_sha256"]),
    ):
        path = baseline / folder / "dataset.npz"
        assert sha256(path) == digest
        inputs[str(path.relative_to(ROOT))] = digest
        with np.load(path, allow_pickle=False) as cache:
            keep = ~cache["synthetic"] if "synthetic" in cache else np.ones(len(cache["y"]), dtype=bool)
            piece = {key: cache[key][keep] for key in ("x", "y", "owner", "sequences", "lengths")}
        piece["cohort"] = np.full(len(piece["y"]), cohort)
        pieces.append(piece)
    data = {key: np.concatenate([piece[key] for piece in pieces]) for key in pieces[0]}
    assert np.isfinite(data["x"]).all()
    people = np.unique(data["owner"])
    labels = np.array([data["y"][np.flatnonzero(data["owner"] == uid)[0]] for uid in people])
    splits = {role: people[index].tolist() for role, index in split_indices(people, labels).items()}
    assert len(set(sum(splits.values(), []))) == sum(map(len, splits.values())) == len(people)
    data["split"] = np.asarray([next(role for role in ROLES if uid in splits[role]) for uid in data["owner"]])
    data["synthetic"] = np.zeros(len(data["y"]), dtype=bool)
    seen = {}
    for seq, length, owner in zip(data["sequences"], data["lengths"], data["owner"]):
        if length == 50:
            token = hashlib.sha256(seq.tobytes()).digest()
            assert token not in seen or seen[token] == owner, "Duplicate full sequence across participants"
            seen[token] = owner
    people = np.asarray(splits["train"])
    first = {uid: int(np.flatnonzero(data["owner"] == uid)[0]) for uid in np.unique(data["owner"])}
    assert all(len(np.unique(data["y"][data["owner"] == uid])) == 1 for uid in first)
    strata = np.array([data["y"][first[uid]] for uid in people])
    folds = []
    for repeat in range(5):
        for fold, (fit, score) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED+repeat).split(people, strata)):
            folds.append(dict(repeat=repeat+1, fold=fold+1, fit=people[fit].tolist(), score=people[score].tolist()))
    counts = {}
    for role in ROLES:
        counts[role] = {}
        for cohort in ("all", "neuroqwerty", "online"):
            mask = (data["split"] == role) & ((data["cohort"] == cohort) if cohort != "all" else True)
            ids = np.unique(data["owner"][mask])
            pd_count = sum(int(data["y"][first[uid]]) for uid in ids)
            counts[role][cohort] = dict(people=len(ids), pd=pd_count, controls=len(ids)-pd_count,
                                       real=int(mask.sum()), synthetic=0)
    del data["sequences"], data["lengths"]
    np.savez_compressed(out / "dataset.npz", **data)
    sources = {**source_hashes(), "paper_typenet/nn.py": sha256(ROOT / "paper_typenet/nn.py")}
    registration = dict(seed=SEED, repeats=5, folds=folds, families=list(MODELS), augmentation=[0],
        epochs=epochs, threshold=.5, splits=splits, counts=counts, features=135,
        selection="All four fixed configurations evaluated; no external selection or tuning.",
        stratification="participant label; fresh 70/10/20 split",
        primary="Participant accuracy mean and sample SD over 25 fold models, for each family and cohort",
        secondary="Equal-weight probability ensemble within each family, threshold .5",
        encoder=original["encoder"], encoder_sha256=original["encoder_sha256"], encoder_trainable_parameters=0,
        dataset_sha256=sha256(out / "dataset.npz"), source_hashes=sources, cached_inputs=inputs,
        limitations=["Fresh participant split with fixed seed; prior experiments remain archived.",
                     "Online PD labels are self-reported; neuroQWERTY labels are clinical.",
                     "No cross-study identity linkage available; no shared full sequences detected."],
        audit=dict(synthetic_rows=0, cross_participant_duplicate_full_sequences=0, cached_frozen_features=True))
    write_json(out / "registration.json", registration)
    with zipfile.ZipFile(out / "source.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sources:
            archive.write(ROOT / path, path)
        archive.write(Path(__file__).with_name("README.md"), "README.md")
    print(json.dumps(dict(counts=counts, fits=100, epochs=epochs), indent=2), flush=True)


def prepare_embeddings(out, baseline):
    import ast
    baseline = baseline.resolve()
    original = json.loads((baseline / "registration.json").read_text())
    assert original["features"] == 135 and original["augmentation"] == [0]
    assert original["families"] == list(MODELS) and original["seed"] == SEED
    assert sha256(baseline / "dataset.npz") == original["dataset_sha256"]
    assert sha256(ROOT / original["encoder"]) == original["encoder_sha256"]
    source = "prototype_net/exp4/model.py"
    with zipfile.ZipFile(baseline / "source.zip") as archive:
        archived = archive.read(source)
    assert hashlib.sha256(archived).hexdigest() == original["source_hashes"][source]
    current = {node.name: ast.dump(node) for node in ast.parse(Path(__file__).read_text()).body
               if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    for node in ast.parse(archived).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name != "main":
            assert current[node.name] == ast.dump(node), f"Original function changed: {node.name}"
    for name, digest in original["source_hashes"].items():
        if name != source:
            assert sha256(ROOT / name) == digest
    with np.load(baseline / "dataset.npz", allow_pickle=False) as cache:
        data = {key: cache[key] for key in cache.files}
    assert data["x"].shape[1] == 135 and not data["synthetic"].any()
    data["x"] = data["x"][:, :128].copy()
    assert np.isfinite(data["x"]).all()
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / "dataset.npz", **data)
    sources = {**source_hashes(), "paper_typenet/nn.py": sha256(ROOT / "paper_typenet/nn.py")}
    registration = dict(original, features=128, dataset_sha256=sha256(out / "dataset.npz"),
        source_hashes=sources, cached_inputs={str((baseline / "dataset.npz").relative_to(ROOT)):
                                            original["dataset_sha256"]},
        baseline=str(baseline.relative_to(ROOT)), baseline_registration_sha256=sha256(baseline / "registration.json"),
        baseline_source_sha256=sha256(baseline / "source.zip"),
        representation="First 128 cached frozen TypeNet coordinates only; seven hold summaries removed.",
        stratification="Exact participant splits and CV folds retained from the real-data baseline",
        audit=dict(synthetic_rows=0, cached_frozen_features=True, original_training_functions_unchanged=True))
    write_json(out / "registration.json", registration)
    with zipfile.ZipFile(out / "source.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sources:
            archive.write(ROOT / name, name)
        archive.write(Path(__file__).with_name("README.md"), "README.md")
    print(json.dumps(dict(features=128, counts=registration["counts"], fits=100,
                          epochs=registration["epochs"]), indent=2), flush=True)


def evaluate_real(out):
    registration = typenet_verify(out)
    freeze = json.loads((out / "freeze.json").read_text())
    assert freeze["registration_sha256"] == sha256(out / "registration.json")
    if (out / "results.json").exists():
        raise FileExistsError("Evaluation already completed")
    with np.load(out / "dataset.npz", allow_pickle=False) as cache:
        data = {key: cache[key] for key in cache.files}
    assert not data["synthetic"].any()
    ids = np.unique(data["owner"])
    first = np.array([np.flatnonzero(data["owner"] == uid)[0] for uid in ids])
    labels, roles, cohorts = (data[key][first] for key in ("y", "split", "cohort"))
    scores, rows, cv = defaultdict(list), [], []
    for record in freeze["models"]:
        path = out / f"fold{record['index']:02d}.joblib"
        assert sha256(path) == record["sha256"]
        bundle = joblib.load(path)
        assert bundle["registration_sha256"] == freeze["registration_sha256"]
        cv.extend(dict(index=record["index"], **row) for row in bundle["results"])
        for family in MODELS:
            model = bundle["models"][f"{family}_0"]
            assert model["synthetic_rows"] == 0 and not model["parents"]
            probability = predict(model, data["x"])
            grouped = np.array([probability[data["owner"] == uid].mean() for uid in ids])
            scores[family].append(grouped)
            for role in ("validation", "test"):
                for cohort in ("all", "neuroqwerty", "online"):
                    mask = (roles == role) & ((cohorts == cohort) if cohort != "all" else True)
                    rows.append(dict(index=record["index"], family=family, role=role, cohort=cohort,
                                     people=int(mask.sum()), **metrics(labels[mask], grouped[mask], "classification")))
    aggregates = []
    for family in MODELS:
        scores[family] = np.stack(scores[family])
        for role in ("validation", "test"):
            for cohort in ("all", "neuroqwerty", "online"):
                group = [r for r in rows if r["family"] == family and r["role"] == role and r["cohort"] == cohort]
                mask = (roles == role) & ((cohorts == cohort) if cohort != "all" else True)
                aggregate = {metric: dict(mean=float(np.mean([r[metric] for r in group])),
                                         sd=float(np.std([r[metric] for r in group], ddof=1)))
                             for metric in ("accuracy", "balanced_accuracy", "auroc", "sensitivity", "specificity")}
                aggregates.append(dict(family=family, role=role, cohort=cohort, people=int(mask.sum()), models=len(group),
                    **aggregate, ensemble=metrics(labels[mask], scores[family].mean(0)[mask], "classification")))
    np.savez_compressed(out / "predictions.npz", ids=ids, y=labels, split=roles, cohort=cohorts, **scores)
    write_json(out / "results.json", dict(counts=registration["counts"], rows=rows, cv=cv, aggregates=aggregates,
        freeze_sha256=sha256(out / "freeze.json"), predictions_sha256=sha256(out / "predictions.npz"),
        sd_definition="Sample SD across 25 fold models on the same fixed participants, not independent cohorts."))
    print(json.dumps([r for r in aggregates if r["cohort"] == "all"], indent=2), flush=True)


class TypingClassifier:
    """Frozen TypeNet and population classifier, without participant state."""
    def __init__(self, run=None, family="xgboost", batch_size=6):
        from paper_typenet.nn import TypeNetEncoder
        self.run = Path(run or Path(__file__).parent / "runs/embeddings").resolve()
        if family not in MODELS or batch_size < 2:
            raise ValueError("Invalid model family or batch size")
        self.family, self.batch_size = family, batch_size
        reg = json.loads((self.run / "registration.json").read_text())
        self.features = reg["features"]
        if self.features not in (128, 135):
            raise ValueError("Unsupported monitor feature representation")
        freeze = json.loads((self.run / "freeze.json").read_text())
        if freeze["registration_sha256"] != sha256(self.run / "registration.json"):
            raise ValueError("Registration changed")
        with zipfile.ZipFile(self.run / "source.zip") as archive:
            for name, digest in reg["source_hashes"].items():
                if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                    raise ValueError("Archived training source changed")
        checkpoint = ROOT / reg["encoder"]
        if sha256(checkpoint) != reg["encoder_sha256"]:
            raise ValueError("Encoder changed")
        self.encoder = TypeNetEncoder()
        self.encoder.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        self.encoder.eval().requires_grad_(False)
        self.neutral_key = float(self.encoder.input_batch_norm.running_mean[4]) * 255
        self.models = []
        for record in freeze["models"]:
            path = self.run / f"fold{record['index']:02d}.joblib"
            if sha256(path) != record["sha256"]:
                raise ValueError("Classifier changed")
            bundle = joblib.load(path)
            if bundle["registration_sha256"] != freeze["registration_sha256"]:
                raise ValueError("Classifier registration mismatch")
            self.models.append(bundle["models"][f"{family}_0"])
        if len(self.models) != 25:
            raise ValueError("Expected 25 frozen classifiers")
        self.fingerprint = hashlib.sha256((sha256(self.run / "freeze.json") + reg["encoder_sha256"] + family).encode()).hexdigest()

    def encode(self, events):
        from paper_typenet.nn import events_to_features
        sessions, identifiers = {}, set()
        for order, event in enumerate(events):
            session = str(event["session_id"])
            token = json.dumps([session, str(event["event_id"])], separators=(",", ":"))
            if token in identifiers:
                raise ValueError("Duplicate event in batch")
            identifiers.add(token)
            press, release = float(event["keydown_ms"]), float(event["keyup_ms"])
            if not (math.isfinite(press) and math.isfinite(release) and release > press >= 0):
                raise ValueError("Invalid key timestamps")
            if "key" in event:
                key = event["key"]
                if key == "spacebar":
                    key = " "
                if not (isinstance(key, str) and len(key) == 1 and key.isascii() and key.isprintable()):
                    continue
                code = ord(key.upper()) if key.isalnum() else 32 if key == " " else self.neutral_key
            else:
                code = float(event["keycode"])
                if not math.isfinite(code) or not (0 <= code <= 255 or abs(code-self.neutral_key) < 1e-6):
                    raise ValueError("Expected a standard keycode or the encoder neutral symbol code")
            sessions.setdefault(session, []).append((press, release, code, order))
        sequences = []
        for recording in sessions.values():
            recording.sort(key=lambda row: (row[0], row[3]))
            for start in range(0, len(recording) - 49, 50):
                seq, length = events_to_features(recording[start:start+51], 50)
                assert length == 50
                sequences.append(seq)
        if len(sequences) != self.batch_size:
            return None, sorted(identifiers), len(sequences)
        raw = np.stack(sequences)
        with torch.inference_mode():
            embedded = self.encoder(torch.from_numpy(raw), torch.full((len(raw),), 50, dtype=torch.long)).numpy()
        features = embedded.astype(np.float32)
        if self.features == 135:
            stats = np.stack([summary(sequence[:, 0]) for sequence in raw])
            features = np.concatenate([features, stats], axis=1).astype(np.float32)
        if not np.isfinite(features).all():
            raise ValueError("Nonfinite batch features")
        return features, sorted(identifiers), len(sequences)


    def probabilities(self, features):
        return np.stack([predict(model, features) for model in self.models])

    def classify(self, payload):
        if self.features != 128 or self.family != "xgboost":
            raise ValueError("The standalone variant requires raw-embedding XGBoost")
        if len(payload["events"]) != self.batch_size*50:
            raise ValueError(f"Supply exactly {self.batch_size*50} eligible keys")
        features, _, windows = self.encode(payload["events"])
        if features is None:
            raise ValueError("Input must contain complete 50-key session windows")
        scores = self.probabilities(features).mean(axis=1).astype(np.float64)
        score = float(scores.mean())
        return dict(pd_score=score, flag=int(score >= .5),
            label="pd_pattern" if score >= .5 else "control_pattern", threshold=.5,
            model_score_sd=float(scores.std(ddof=1)) if len(scores) > 1 else 0.,
            model_count=len(scores), keystrokes=windows*50,
            embedding_shape=[windows, 128], classifier_input_shape=[windows, 128],
            target="PD-associated typing versus control")

    def fit(self, x, y, owners, parameters=None):
        """Return a fresh classifier fitted only to the supplied training-fold rows."""
        x, y, owners = np.asarray(x, dtype=np.float32), np.asarray(y), np.asarray(owners)
        if (x.ndim != 2 or x.shape[1] != 128 or y.shape != (len(x),)
                or owners.shape != y.shape or not np.isfinite(x).all()
                or set(np.unique(y)) != {0, 1}):
            raise ValueError("Expected aligned real 128-coordinate training rows, binary labels and owners")
        model = fit_model("xgboost", x, y, owners, "classification", "none", 0, parameters=parameters)
        bundle = self._bundle()
        bundle["models"] = [model]
        content = (model["estimator"].get_booster().save_raw(raw_format="json")
                   + model["preprocess"]["medians"].tobytes()
                   + model["preprocess"]["scaler"].mean_.tobytes()
                   + model["preprocess"]["scaler"].scale_.tobytes())
        bundle["fingerprint"] = hashlib.sha256(self.fingerprint.encode() + content).hexdigest()
        return self._from_bundle(bundle)

    def _bundle(self):
        if self.features != 128 or self.family != "xgboost":
            raise ValueError("Export requires raw-embedding XGBoost")
        return dict(version=1, family=self.family, features=self.features, batch_size=self.batch_size,
            fingerprint=self.fingerprint, encoder={k: v.detach().cpu().clone() for k, v in self.encoder.state_dict().items()},
            models=self.models)

    @classmethod
    def _from_bundle(cls, bundle):
        from paper_typenet.nn import TypeNetEncoder
        if (bundle["version"] != 1 or bundle["family"] != "xgboost" or bundle["features"] != 128
                or type(bundle["batch_size"]) is not int or bundle["batch_size"] < 2
                or not bundle["models"]):
            raise ValueError("Invalid classifier artifact")
        instance = cls.__new__(cls)
        instance.run = None
        for name in ("family", "features", "batch_size", "fingerprint", "models"):
            setattr(instance, name, bundle[name])
        for model in instance.models:
            if (model["name"] != "xgboost" or model["kind"] != "classification"
                    or model["preprocess"]["scaler"].n_features_in_ != 128):
                raise ValueError("Invalid exported classifier head")
        instance.encoder = TypeNetEncoder()
        instance.encoder.load_state_dict(bundle["encoder"])
        instance.encoder.eval().requires_grad_(False)
        instance.neutral_key = float(instance.encoder.input_batch_norm.running_mean[4])*255
        return instance

    def export(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            joblib.dump(dict(variant="classifier", engine=self._bundle()), stream, compress=3)

    @classmethod
    def load(cls, path):
        bundle = joblib.load(path)
        if bundle["variant"] != "classifier":
            raise ValueError("Expected a standalone classifier artifact")
        return cls._from_bundle(bundle["engine"])




def simulate_monitor(out, baseline):
    from prototype_net.exp4.baseline import PersonalMonitor
    from datetime import datetime, timezone
    out.mkdir(parents=True, exist_ok=False)
    baseline = Path(baseline).resolve()
    reg = json.loads((baseline / "registration.json").read_text())
    encoder_path = ROOT / reg["encoder"]
    from paper_typenet.nn import TypeNetEncoder
    encoder = TypeNetEncoder()
    encoder.load_state_dict(torch.load(encoder_path, map_location="cpu", weights_only=True))
    data, audit = typenet_sequences(ROOT / "data/clinical", float(encoder.input_batch_norm.running_mean[4])*255, synthetic=0)
    assert not data["synthetic"].any()
    heldout = set(reg["splits"]["validation"] + reg["splits"]["test"])
    eligible, excluded = {}, []
    for uid in sorted(set(data["owner"])):
        event_rows = data["events"][data["events"][:, 0] == uid]
        recordings = [(name, event_rows[event_rows[:, 1] == name]) for name in sorted(set(event_rows[:, 1]))]
        complete = [(name, rows) for name, rows in recordings if len(rows) >= 2500]
        if uid in heldout and len(complete) >= 2:
            eligible[uid] = complete
        else:
            excluded.append(dict(participant_id=uid, heldout=uid in heldout,
                                 recordings_with_2500_keys=len(complete)))
    registration = dict(batch_size=50, keys_per_sequence=50, family="xgboost", model_count=25,
        baseline_policy="First eligible visit, first 2500 valid keys; reference remains fixed",
        followup_policy="First 2500 valid keys from each later eligible visit; chronological; no reuse",
        clinical_labels_used_by_monitor=False, synthetic_rows=0, eligible_participants=sorted(eligible),
        excluded=excluded, source_audit=audit, clinical_day_claim="Recorded visits, not complete days of passive typing",
        checkpoint_freeze_sha256=sha256(baseline / "freeze.json"), encoder_sha256=sha256(encoder_path),
        input_sha256=sha256(ROOT / "data/clinical/neuroqwerty/neuroQWERTY.zip"),
        source_hashes={**source_hashes(), "paper_typenet/nn.py": sha256(ROOT / "paper_typenet/nn.py")})
    write_json(out / "registration.json", registration)
    with zipfile.ZipFile(out / "source.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in registration["source_hashes"]:
            archive.write(ROOT / path, path)
        archive.write(Path(__file__).with_name("README.md"), "README.md")
    results, matrices, inputs = [], {}, []
    for uid, recordings in eligible.items():
        monitor = PersonalMonitor(baseline, uid)
        state_path = out / f"{uid.replace(':', '_')}.npz"
        for number, (name, recording) in enumerate(recordings):
            timestamp = int(name.split(".")[0])*1000
            payload = dict(participant_id=uid, batch_id=name, timestamp_ms=timestamp,
                events=[dict(session_id=name, event_id=str(row[2]), keydown_ms=float(row[3])*1000,
                             keyup_ms=(float(row[3])+float(row[4]))*1000, keycode=float(row[5]))
                        for row in recording[:2500]])
            inputs.append(payload)
            if number:
                monitor = PersonalMonitor(baseline, uid)
                monitor.restore(state_path)
                before = monitor.baseline.copy()
            result = monitor.observe(payload)
            result.update(recorded_date_utc=datetime.fromtimestamp(timestamp/1000, timezone.utc).isoformat(),
                          source_split=next(role for role in ROLES if uid in reg["splits"][role]),
                          source_label=int(data["y"][np.flatnonzero(data["owner"] == uid)[0]]))
            if number:
                assert np.array_equal(before, monitor.baseline)
            monitor.save(state_path)
            prefix = f"{uid.replace(':', '_')}_batch{number}"
            matrices[prefix + "_features"] = monitor.last_features
            matrices[prefix + "_embeddings"] = monitor.last_features[:, :128]
            results.append(result)
    with (out / "replay.jsonl").open("x") as stream:
        for payload in inputs:
            stream.write(json.dumps(payload, separators=(",", ":"), allow_nan=False)+"\n")
    np.savez_compressed(out / "matrices.npz", **matrices)
    assert sha256(encoder_path) == registration["encoder_sha256"]
    assert sha256(baseline / "freeze.json") == registration["checkpoint_freeze_sha256"]
    report = dict(participants=len(eligible), batches=len(results), baseline_batches=len(eligible),
        followup_batches=len(results)-len(eligible), synthetic_rows=0, results=results,
        interpretation="Functional baseline/follow-up simulation; no longitudinal clinical outcome labels or new accuracy estimate",
        matrices_sha256=sha256(out / "matrices.npz"), replay_sha256=sha256(out / "replay.jsonl"))
    write_json(out / "results.json", report)
    print(json.dumps(report, indent=2), flush=True)


def simulate_impaired(out, baseline):
    from prototype_net.exp4.baseline import PersonalMonitor
    out.mkdir(parents=True, exist_ok=False)
    baseline = Path(baseline).resolve()
    reg = json.loads((baseline / "registration.json").read_text())
    monitor = PersonalMonitor(baseline, "demo:control_to_pd")
    data, audit = typenet_sequences(ROOT / "data/clinical", monitor.neutral_key, synthetic=0)
    assert not data["synthetic"].any()
    control = "nq:95"
    events = data["events"]

    def make_batch(uid, names):
        selected, recordings, remaining = [], [], 50
        for name in sorted(names):
            rows = events[(events[:, 0] == uid) & (events[:, 1] == name)]
            windows = min(len(rows)//50, remaining)
            if not windows:
                continue
            selected.extend(dict(session_id=name, event_id=str(row[2]), keydown_ms=float(row[3])*1000,
                keyup_ms=(float(row[3])+float(row[4]))*1000, keycode=float(row[5])) for row in rows[:windows*50])
            recordings.append(dict(file=name, windows=windows, source_timestamp_ms=int(name.split('.')[0])*1000))
            remaining -= windows
            if not remaining:
                break
        return selected, recordings, remaining

    control_names = sorted(set(events[events[:, 0] == control, 1]))
    cases = [(control, 0, "baseline", [control_names[0]]),
             (control, 0, "control_followup", [control_names[1]])]
    for uid in sorted(set(data["owner"])):
        label = int(data["y"][np.flatnonzero(data["owner"] == uid)[0]])
        if label:
            names = sorted(set(events[events[:, 0] == uid, 1]))
            if make_batch(uid, names)[2] == 0:
                cases.append((uid, label, "pd_followup", names))
    registration = dict(family="xgboost", model_count=25, baseline_participant=control, threshold=.5,
        batch_size=50, keys_per_sequence=50, synthetic_keystrokes=0,
        cases=[dict(source_participant=uid, label=label, role=role) for uid, label, role, _ in cases],
        selection="All PD participants with 50 complete real windows; earliest available recordings, no selection by score",
        scenario="Cross-person demonstration: control baseline followed by control and PD-labeled recordings",
        time_basis="Payload timestamps encode simulated arrival order; source recording timestamps are retained separately",
        baseline_policy="Fixed original control baseline; no fitting or threshold changes",
        checkpoint_freeze_sha256=sha256(baseline / "freeze.json"), encoder_sha256=sha256(ROOT / reg["encoder"]),
        input_sha256=sha256(ROOT / "data/clinical/neuroqwerty/neuroQWERTY.zip"), source_audit=audit,
        source_hashes={**source_hashes(), "paper_typenet/nn.py": sha256(ROOT / "paper_typenet/nn.py")})
    write_json(out / "registration.json", registration)
    with zipfile.ZipFile(out / "source.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in registration["source_hashes"]:
            archive.write(ROOT / path, path)
        archive.write(Path(__file__).with_name("README.md"), "README.md")
    results, matrices = [], {}
    state_path = out / "state.npz"
    with (out / "replay.jsonl").open("x") as replay:
        for step, (uid, label, role, names) in enumerate(cases):
            batch_events, recordings, remaining = make_batch(uid, names)
            assert remaining == 0 and len(batch_events) == 2500
            payload = dict(participant_id="demo:control_to_pd", batch_id=f"{role}:{uid}",
                           timestamp_ms=step, events=batch_events)
            replay.write(json.dumps(payload, separators=(",", ":"), allow_nan=False)+"\n")
            if step:
                monitor = PersonalMonitor(baseline, payload["participant_id"])
                monitor.restore(state_path)
            result = monitor.observe(payload)
            result.update(source_participant=uid, source_label=label, scenario_role=role, recordings=recordings,
                          source_split=next(r for r in ROLES if uid in reg["splits"][r]),
                          same_person_as_baseline=uid == control)
            if not step:
                reference = monitor.baseline.copy()
            assert np.array_equal(reference, monitor.baseline)
            monitor.save(state_path)
            matrices[f"batch{step:02d}_embeddings"] = monitor.last_features[:, :128]
            matrices[f"batch{step:02d}_features"] = monitor.last_features
            results.append(result)
            print(json.dumps({k:v for k,v in result.items() if k != "recordings"}), flush=True)
    np.savez_compressed(out / "matrices.npz", **matrices)
    pd_rows = [r for r in results if r["scenario_role"] == "pd_followup"]
    report = dict(baseline_batches=1, control_followup_batches=1, pd_followup_batches=len(pd_rows),
        pd_classified_pd=sum(r["predicted_class"] for r in pd_rows),
        pd_classified_control=sum(1-r["predicted_class"] for r in pd_rows), synthetic_keystrokes=0,
        interpretation="Cross-person functional demonstration, not observed within-person onset or independent clinical accuracy",
        results=results, matrices_sha256=sha256(out / "matrices.npz"), replay_sha256=sha256(out / "replay.jsonl"))
    assert sha256(baseline / "freeze.json") == registration["checkpoint_freeze_sha256"]
    assert sha256(ROOT / reg["encoder"]) == registration["encoder_sha256"]
    write_json(out / "results.json", report)


def personal_features(current, reference):
    current, reference = np.asarray(current), np.asarray(reference)
    if current.ndim != 2 or reference.ndim != 2 or current.shape[1] != 135 or reference.shape[1] != 135:
        raise ValueError("Expected sequence feature matrices with 135 columns")
    mean, scale = reference.mean(0), reference.std(0)
    normalized = (current-mean) / np.where(scale > 1e-8, scale, 1.)
    return np.concatenate([current, np.broadcast_to(mean, current.shape),
                           np.broadcast_to(scale, current.shape), normalized], axis=1).astype(np.float32)


def prepare_personal(out, baseline, epochs):
    from paper_typenet.nn import TypeNetEncoder, events_to_features
    out.mkdir(parents=True, exist_ok=False)
    baseline = Path(baseline).resolve()
    original = json.loads((baseline / "registration.json").read_text())
    encoder = TypeNetEncoder()
    checkpoint = ROOT / original["encoder"]
    assert sha256(checkpoint) == original["encoder_sha256"]
    encoder.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    encoder.eval().requires_grad_(False)
    before = {k: v.clone() for k, v in encoder.state_dict().items()}
    neutral = float(encoder.input_batch_norm.running_mean[4])*255
    clinical, clinical_audit = typenet_sequences(ROOT / "data/clinical", neutral, synthetic=0)
    events, recordings, labels = [], {}, {}
    for uid in np.unique(clinical["owner"]):
        labels[str(uid)] = int(clinical["y"][np.flatnonzero(clinical["owner"] == uid)[0]])
        rows = clinical["events"][clinical["events"][:, 0] == uid]
        recordings[str(uid)] = []
        for name in sorted(set(rows[:, 1])):
            group = []
            for row in rows[rows[:, 1] == name]:
                press, release = float(row[3])*1000, (float(row[3])+float(row[4]))*1000
                group.append((press, release, float(row[5]), len(events)))
                events.append((str(uid), str(name), str(row[2]), str(press), str(release), str(row[5])))
            recordings[str(uid)].append((str(name), group))
    columns = ["key", "participant_id", "response_id", "diagnosis", "keydown", "keyup"]
    online = pd.read_csv(ROOT / "data/clinical/conll/typing.csv", usecols=columns, keep_default_na=False).drop_duplicates(columns)
    online_audit = Counter()
    for uid, person in online.groupby("participant_id", sort=True):
        owner = f"conll:{uid}"
        assert person["diagnosis"].nunique() == 1 and person["diagnosis"].iloc[0] in (0, 1)
        labels[owner] = int(person["diagnosis"].iloc[0])
        recordings[owner] = []
        previous = None
        for response_id, response in person.groupby("response_id", sort=True):
            group = []
            for row in response.itertuples():
                key = " " if row.key == "spacebar" else row.key
                if not (len(key) == 1 and key.isascii() and key.isprintable()):
                    continue
                try:
                    press, release = float(row.keydown), float(row.keyup)
                    if not (math.isfinite(press) and math.isfinite(release) and release > press >= 0):
                        continue
                except (ValueError, TypeError):
                    continue
                code = ord(key.upper()) if key.isalnum() else 32 if key == " " else neutral
                name = f"{owner}/response:{response_id}"
                group.append((press, release, code, len(events)))
                events.append((owner, name, str(row.Index), str(press), str(release), str(code)))
            group.sort(key=lambda row: (row[0], row[3]))
            if group:
                online_audit["clock_reset_or_response_overlap"] += previous is not None and group[0][0] < previous
                previous = group[-1][0]
                online_audit["retained_events"] += len(group)
                recordings[owner].append((name, group))
    events = np.asarray(events)
    np.savez_compressed(out / "events.npz", events=events)
    registration = dict(seed=SEED, repeats=5, epochs=epochs, families=["xgboost", "node"],
        arms=["population", "personal"], threshold=.5, encoder=original["encoder"],
        encoder_sha256=original["encoder_sha256"], baseline_run=str(baseline),
        baseline_registration_sha256=sha256(baseline / "registration.json"), events_sha256=sha256(out / "events.npz"),
        source_hashes={**source_hashes(), "paper_typenet/nn.py": sha256(ROOT / "paper_typenet/nn.py")},
        input_hashes={"data/clinical/neuroqwerty/neuroQWERTY.zip": sha256(ROOT / "data/clinical/neuroqwerty/neuroQWERTY.zip"),
                      "data/clinical/conll/typing.csv": sha256(ROOT / "data/clinical/conll/typing.csv")},
        audit=dict(neuroqwerty=clinical_audit, online=dict(online_audit)), scenarios={})
    for keys, count in ((300, 6), (400, 8)):
        folder = out / str(keys)
        folder.mkdir()
        raw, lineage, owners, batches, reference, records = [], [], [], [], {}, []
        for uid, sessions in recordings.items():
            chunks = [[session[start:start+50] for start in range(0, len(session)-49, 50)] for _, session in sessions]
            if keys == 300:
                windows = [window for group in chunks for window in group]
                if len(windows) < count*2:
                    continue
                base, follow = windows[:count], windows[count:]
            else:
                if not uid.startswith("nq:") or len(chunks) < 2 or len(chunks[0]) < count:
                    continue
                base, follow = chunks[0][:count], [window for group in chunks[1:] for window in group]
                if len(follow) < count:
                    continue
            following = [follow[i:i+count] for i in range(0, len(follow)-count+1, count)]
            for number, batch in enumerate([base] + following):
                grouped = {}
                for window in batch:
                    name = events[window[0][3], 1]
                    grouped.setdefault(name, []).extend(window)
                index = []
                for name, rows in grouped.items():
                    for start in range(0, len(rows), 50):
                        seq, length = events_to_features(rows[start:start+51], 50)
                        assert length == 50
                        index.append(len(raw))
                        raw.append(seq)
                        lineage.append([int(row[3]) for row in rows[start:start+50]])
                        owners.append(uid)
                        batches.append(number)
                assert len(index) == count
                if number == 0:
                    reference[uid] = index
                else:
                    records.append(dict(owner=uid, batch=number, index=index))
        raw, lineage, owners, batches = np.asarray(raw), np.asarray(lineage), np.asarray(owners), np.asarray(batches)
        assert len(np.unique(lineage)) == lineage.size
        assert np.all(events[lineage, 0] == owners[:, None])
        embedding = []
        with torch.inference_mode():
            for start in range(0, len(raw), 256):
                piece = torch.from_numpy(raw[start:start+256])
                embedding.append(encoder(piece, torch.full((len(piece),), 50, dtype=torch.long)).numpy())
        features = np.concatenate([np.concatenate(embedding), np.stack([summary(seq[:, 0]) for seq in raw])], axis=1).astype(np.float32)
        personal = np.zeros((len(features), 540), dtype=np.float32)
        for row in records:
            index = row["index"]
            personal[index] = personal_features(features[index], features[reference[row["owner"]]])
        people = sorted(reference)
        splits = {role: [uid for uid in original["splits"][role] if uid in reference] for role in ROLES}
        roles = np.array([next(role for role in ROLES if uid in splits[role]) for uid in owners])
        y = np.array([labels[uid] for uid in owners])
        fit_people = np.asarray(splits["train"])
        fit_labels = np.array([labels[uid] for uid in fit_people])
        folds = []
        for repeat in range(5):
            for fold, (fit, score) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED+repeat).split(fit_people, fit_labels)):
                folds.append(dict(repeat=repeat+1, fold=fold+1, fit=fit_people[fit].tolist(), score=fit_people[score].tolist()))
        counts = {role: dict(people=len(splits[role]), pd=sum(labels[uid] for uid in splits[role]),
                    controls=sum(1-labels[uid] for uid in splits[role]),
                    baseline_windows=int(np.sum((roles == role) & (batches == 0))),
                    followup_windows=int(np.sum((roles == role) & (batches > 0))),
                    followup_batches=int(np.sum((roles == role) & (batches > 0)))//count) for role in ROLES}
        np.savez_compressed(folder / "dataset.npz", x=features, personal=personal, raw=raw, lineage=lineage,
                            owner=owners, y=y, split=roles, batch=batches)
        registration["scenarios"][str(keys)] = dict(keys_per_batch=keys, windows_per_batch=count, people=len(people),
            selection="first baseline, all subsequent complete batches" if keys == 300 else "first-visit baseline, all complete later-visit batches",
            splits=splits, folds=folds, counts=counts, dataset_sha256=sha256(folder / "dataset.npz"))
        print(json.dumps(dict(keys=keys, counts=counts)), flush=True)
    assert all(torch.equal(v, before[k]) for k, v in encoder.state_dict().items())
    assert sha256(checkpoint) == original["encoder_sha256"]
    write_json(out / "registration.json", registration)
    with zipfile.ZipFile(out / "source.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in registration["source_hashes"]:
            archive.write(ROOT / path, path)
        archive.write(Path(__file__).with_name("README.md"), "README.md")


def prepare_personal_embeddings(out, baseline):
    baseline = baseline.resolve()
    original = json.loads((baseline / "registration.json").read_text())
    assert original["seed"] == SEED and original["arms"] == ["population", "personal"]
    assert sha256(ROOT / original["encoder"]) == original["encoder_sha256"]
    assert sha256(baseline / "events.npz") == original["events_sha256"]
    for name, digest in original["input_hashes"].items():
        assert sha256(ROOT / name) == digest
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(baseline / "events.npz", out / "events.npz")
    registration = dict(original, families=list(MODELS), scenarios={},
        features=dict(population=128, personal={size: 128*(value["windows_per_batch"]+1)
                                              for size, value in original["scenarios"].items()}),
        comparison_run=str(baseline.relative_to(ROOT)),
        comparison_registration_sha256=sha256(baseline / "registration.json"),
        comparison_source_sha256=sha256(baseline / "source.zip"),
        representation="Current 128-coordinate TypeNet embedding followed by the full chronologically ordered baseline embedding matrix, flattened row-major.",
        source_hashes={**source_hashes(), "paper_typenet/nn.py": sha256(ROOT / "paper_typenet/nn.py")})
    for size, scenario in original["scenarios"].items():
        source = baseline / size / "dataset.npz"
        assert sha256(source) == scenario["dataset_sha256"]
        with np.load(source, allow_pickle=False) as cache:
            data = {key: cache[key] for key in cache.files}
        assert data["x"].shape[1] == 135 and data["personal"].shape[1] == 540
        data["x"] = data["x"][:, :128].copy()
        data["personal"] = np.zeros((len(data["x"]), registration["features"]["personal"][size]), dtype=np.float32)
        for uid in np.unique(data["owner"]):
            reference = data["x"][(data["owner"] == uid) & (data["batch"] == 0)]
            assert reference.shape == (scenario["windows_per_batch"], 128)
            post = (data["owner"] == uid) & (data["batch"] > 0)
            data["personal"][post] = np.concatenate([data["x"][post],
                np.broadcast_to(reference.reshape(-1), (int(post.sum()), reference.size))], axis=1)
        assert np.isfinite(data["x"]).all() and np.isfinite(data["personal"]).all()
        folder = out / size
        folder.mkdir()
        np.savez_compressed(folder / "dataset.npz", **data)
        registration["scenarios"][size] = dict(scenario, dataset_sha256=sha256(folder / "dataset.npz"),
                                              source_dataset_sha256=scenario["dataset_sha256"])
    write_json(out / "registration.json", registration)
    with zipfile.ZipFile(out / "source.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in registration["source_hashes"]:
            archive.write(ROOT / name, name)
        archive.write(Path(__file__).with_name("README.md"), "README.md")
    print(json.dumps(dict(fits=400, families=registration["families"], features=registration["features"],
                          counts={size: value["counts"] for size, value in registration["scenarios"].items()})), flush=True)


def personal_verify(out):
    reg = json.loads((out / "registration.json").read_text())
    for name, digest in reg["source_hashes"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f"Source changed: {name}")
    assert sha256(ROOT / reg["encoder"]) == reg["encoder_sha256"]
    for size, scenario in reg["scenarios"].items():
        assert sha256(out / size / "dataset.npz") == scenario["dataset_sha256"]
    return reg


def personal_scores(data, mask, prediction):
    owners, labels = data["owner"][mask], data["y"][mask]
    people = np.unique(owners)
    y = np.array([labels[np.flatnonzero(owners == uid)[0]] for uid in people])
    probability = np.array([prediction[owners == uid].mean() for uid in people])
    return metrics(y, probability, "classification")


def personal_fold(args):
    out, size, index = args
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        reg = json.loads((out / "registration.json").read_text())
        fold = reg["scenarios"][size]["folds"][index]
        with np.load(out / size / "dataset.npz", allow_pickle=False) as cache:
            mask = (cache["split"] == "train") & (cache["batch"] > 0)
            data = {key: cache[key][mask] for key in ("x", "personal", "owner", "y")}
        fit, score = np.isin(data["owner"], fold["fit"]), np.isin(data["owner"], fold["score"])
        assert not np.any(fit & score) and np.all(fit | score)
        models, results = {}, []
        for family in reg["families"]:
            for arm in reg["arms"]:
                x = data["personal"] if arm == "personal" else data["x"]
                model = fit_model(family, x[fit], data["y"][fit], data["owner"][fit], "classification", "none", reg["epochs"])
                name = f"{family}_{arm}"
                models[name] = model
                result = personal_scores(data, score, predict(model, x[score]))
                results.append(dict(family=family, arm=arm, metrics=result, train_people=model["real_people"], train_rows=model["fit_rows"]))
        path = out / size / f"fold{index+1:02d}.joblib"
        if path.exists():
            raise FileExistsError(path)
        joblib.dump(dict(models=models, results=results, fold=fold, registration_sha256=sha256(out / "registration.json")), path, compress=3)
        return dict(size=size, index=index+1, sha256=sha256(path))


def train_personal(out):
    from concurrent.futures import as_completed
    reg = personal_verify(out)
    if (out / "freeze.json").exists() or list(out.glob("*/fold*.joblib")):
        raise FileExistsError("Training artifacts already exist")
    records = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(personal_fold, (out, size, i)) for size in reg["scenarios"] for i in range(25)]
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(json.dumps(record), flush=True)
    write_json(out / "freeze.json", dict(registration_sha256=sha256(out / "registration.json"),
                                        models=sorted(records, key=lambda r: (r["size"], r["index"]))))


def evaluate_personal(out):
    reg = personal_verify(out)
    freeze = json.loads((out / "freeze.json").read_text())
    assert freeze["registration_sha256"] == sha256(out / "registration.json")
    if (out / "results.json").exists():
        raise FileExistsError("Evaluation already completed")
    reports = {}
    for size, scenario in reg["scenarios"].items():
        with np.load(out / size / "dataset.npz", allow_pickle=False) as cache:
            keep = cache["batch"] > 0
            data = {key: cache[key][keep] for key in ("x", "personal", "owner", "y", "split", "batch")}
        ids = np.unique(data["owner"])
        first = np.array([np.flatnonzero(data["owner"] == uid)[0] for uid in ids])
        y, roles = data["y"][first], data["split"][first]
        predictions, rows, cv = defaultdict(list), [], []
        for record in [r for r in freeze["models"] if r["size"] == size]:
            path = out / size / f"fold{record['index']:02d}.joblib"
            assert sha256(path) == record["sha256"]
            bundle = joblib.load(path)
            assert bundle["registration_sha256"] == freeze["registration_sha256"]
            cv.extend(dict(index=record["index"], **r) for r in bundle["results"])
            for family in reg["families"]:
                for arm in reg["arms"]:
                    name = f"{family}_{arm}"
                    model = bundle["models"][name]
                    assert not model["parents"]
                    x = data["personal"] if arm == "personal" else data["x"]
                    probability = predict(model, x)
                    grouped = np.array([probability[data["owner"] == uid].mean() for uid in ids])
                    predictions[name].append(grouped)
                    for role in ("validation", "test"):
                        mask = roles == role
                        rows.append(dict(index=record["index"], family=family, arm=arm, role=role,
                                         **metrics(y[mask], grouped[mask], "classification")))
        aggregates = []
        for family in reg["families"]:
            for arm in reg["arms"]:
                name = f"{family}_{arm}"
                predictions[name] = np.stack(predictions[name])
                for role in ("validation", "test"):
                    group = [r for r in rows if r["family"] == family and r["arm"] == arm and r["role"] == role]
                    mask = roles == role
                    aggregate = {metric: dict(mean=float(np.mean([r[metric] for r in group])),
                        sd=float(np.std([r[metric] for r in group], ddof=1)))
                        for metric in ("accuracy", "balanced_accuracy", "auroc", "sensitivity", "specificity")}
                    aggregates.append(dict(family=family, arm=arm, role=role, people=int(mask.sum()), models=len(group),
                        **aggregate, ensemble=metrics(y[mask], predictions[name].mean(0)[mask], "classification")))
        np.savez_compressed(out / size / "predictions.npz", ids=ids, y=y, split=roles, **predictions)
        reports[size] = dict(counts=scenario["counts"], rows=rows, cv=cv, aggregates=aggregates,
                            predictions_sha256=sha256(out / size / "predictions.npz"))
    write_json(out / "results.json", dict(scenarios=reports, freeze_sha256=sha256(out / "freeze.json"),
        sd_definition="Sample SD across 25 fold models on the same fixed participants",
        target="PD versus control; baseline-aware input conditioning, no personal diagnosis labels used for adaptation"))
    print(json.dumps({size: r["aggregates"] for size, r in reports.items()}, indent=2), flush=True)


class ResidualNODE(nn.Module):
    def __init__(self, mode):
        super().__init__()
        if mode not in ("residual", "late"):
            raise ValueError(mode)
        self.mode = mode
        self.head = NODE(128 if mode == "residual" else 256, 2)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(SEED+1)
            self.query = nn.Linear(128, 16, bias=False)
            self.key = nn.Linear(128, 16, bias=False)
            self.correction = nn.Sequential(nn.Linear(256, 32), nn.GELU(), nn.Linear(32, 128))
        if mode == "residual":
            self.gate = nn.Parameter(torch.zeros(128))

    def representation(self, x, baseline):
        attention = (self.key(baseline) * self.query(x)[:, None]).sum(-1).div(4).softmax(-1)
        context = (attention[:, :, None] * baseline).sum(1)
        z = self.correction(torch.cat([x, context-x], dim=1))
        return x + self.gate.tanh()*z if self.mode == "residual" else torch.cat([x, z], dim=1)

    @torch.no_grad()
    def initialize(self, x, baseline):
        self.head.initialize(self.representation(x, baseline))

    def forward(self, x, baseline):
        return self.head(self.representation(x, baseline))[0]


def prepare_residual(out, baseline):
    baseline = baseline.resolve()
    original = json.loads((baseline / "registration.json").read_text())
    source = baseline / "300/dataset.npz"
    assert sha256(source) == original["scenarios"]["300"]["dataset_sha256"]
    assert sha256(ROOT / original["encoder"]) == original["encoder_sha256"]
    with np.load(source, allow_pickle=False) as cache:
        keep = cache["batch"] > 0
        data = {key: cache[key][keep] for key in ("x", "y", "owner", "split", "batch", "lineage")}
        assert data["x"].shape[1] == 128
        data["baseline"] = cache["personal"][keep, 128:].reshape(-1, 6, 128).copy()
        for uid in np.unique(data["owner"]):
            reference = cache["x"][(cache["owner"] == uid) & (cache["batch"] == 0)]
            assert reference.shape == (6, 128)
            assert np.all(data["baseline"][data["owner"] == uid] == reference)
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / "dataset.npz", **data)
    scenario = original["scenarios"]["300"]
    reg = dict(seed=SEED, modes=["population", "residual", "late"], epochs=original["epochs"],
        folds=scenario["folds"], splits=scenario["splits"], counts=scenario["counts"], threshold=.5,
        encoder=original["encoder"], encoder_sha256=original["encoder_sha256"],
        dataset_sha256=sha256(out / "dataset.npz"), source_dataset=str(source.relative_to(ROOT)),
        source_dataset_sha256=sha256(source), source_registration_sha256=sha256(baseline / "registration.json"),
        source_archive_sha256=sha256(baseline / "source.zip"),
        source_hashes={**source_hashes(), "paper_typenet/nn.py": sha256(ROOT / "paper_typenet/nn.py")},
        adapter=dict(query_key_width=16, hidden_width=32, output_width=128, seed=SEED+1,
                     context="Current-query attention over all six baseline embeddings, no positional encoding",
                     residual="x + tanh(gate) * correction; gate starts at zero",
                     late="concatenate current x and learned correction",
                     preprocessing="Fit on CV training follow-up embeddings only; same transform for baseline",
                     optimizer="AdamW", learning_rate=.003, weight_decay=.0001, gradient_clip=5),
        selection="All three prespecified configurations; no tuning or early stopping; freeze before evaluation")
    write_json(out / "registration.json", reg)
    with zipfile.ZipFile(out / "source.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in reg["source_hashes"]:
            archive.write(ROOT / name, name)
        archive.write(Path(__file__).with_name("README.md"), "README.md")
    print(json.dumps(dict(fits=75, counts=reg["counts"], modes=reg["modes"])), flush=True)


def fit_residual(x, baseline, y, owners, mode, epochs):
    if mode == "population":
        return fit_model("node", x, y, owners, "classification", "none", epochs)
    transformed, state = preprocess(x)
    reference, _ = preprocess(baseline.reshape(-1, 128), state)
    current = torch.from_numpy(transformed)
    reference = torch.from_numpy(reference.reshape(baseline.shape))
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    model = ResidualNODE(mode)
    model.initialize(current, reference)
    initial = {name: value.detach().clone() for name, value in model.named_parameters()}
    counts = Counter(owners)
    weights = np.asarray([1/counts[uid] for uid in owners], dtype=np.float32)
    weights *= len(weights)/weights.sum()
    weights, target = torch.from_numpy(weights), torch.tensor(y, dtype=torch.long)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=1e-4)
    rng = np.random.default_rng(SEED)
    updates = 0
    model.train()
    for _ in range(epochs):
        for index in np.array_split(rng.permutation(len(y)), max(1, math.ceil(len(y)/512))):
            optimizer.zero_grad()
            logits = model(current[index], reference[index])
            loss = (F.cross_entropy(logits, target[index], reduction="none") * weights[index]).mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite adapter loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            updates += 1
    changes = {name: float((value.detach()-initial[name]).norm()) for name, value in model.named_parameters()}
    assert changes["query.weight"] > 0 and changes["key.weight"] > 0
    return dict(mode=mode, preprocess=state, state=model.state_dict(), epochs=epochs, batch_size=512,
                optimizer_updates=updates, real_people=len(counts), real_rows=len(y), fit_rows=len(y), parents=[],
                parameter_changes=changes, parameters=sum(p.numel() for p in model.parameters()))


def predict_residual(model, x, baseline):
    if "mode" not in model:
        return predict(model, x)
    x, _ = preprocess(x, model["preprocess"])
    reference, _ = preprocess(baseline.reshape(-1, 128), model["preprocess"])
    reference = reference.reshape(baseline.shape)
    network = ResidualNODE(model["mode"])
    network.load_state_dict(model["state"])
    network.eval()
    scores = []
    with torch.inference_mode():
        for start in range(0, len(x), 512):
            output = network(torch.from_numpy(x[start:start+512]), torch.from_numpy(reference[start:start+512]))
            scores.append(output.softmax(-1)[:, 1].numpy())
    return np.concatenate(scores)


def residual_fold(args):
    out, index = args
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        reg = json.loads((out / "registration.json").read_text())
        fold = reg["folds"][index]
        with np.load(out / "dataset.npz", allow_pickle=False) as cache:
            keep = cache["split"] == "train"
            data = {key: cache[key][keep] for key in ("x", "baseline", "y", "owner")}
        fit, score = np.isin(data["owner"], fold["fit"]), np.isin(data["owner"], fold["score"])
        assert not np.any(fit & score) and np.all(fit | score)
        models, results = {}, []
        for mode in reg["modes"]:
            model = fit_residual(data["x"][fit], data["baseline"][fit], data["y"][fit],
                                 data["owner"][fit], mode, reg["epochs"])
            models[mode] = model
            prediction = predict_residual(model, data["x"][score], data["baseline"][score])
            results.append(dict(mode=mode, metrics=personal_scores(data, score, prediction)))
        path = out / f"fold{index+1:02d}.joblib"
        if path.exists():
            raise FileExistsError(path)
        joblib.dump(dict(models=models, results=results, fold=fold,
                         registration_sha256=sha256(out / "registration.json")), path, compress=3)
        return dict(index=index+1, sha256=sha256(path))


def train_residual(out):
    from concurrent.futures import as_completed
    reg = typenet_verify(out)
    if (out / "freeze.json").exists() or list(out.glob("fold*.joblib")):
        raise FileExistsError("Training artifacts already exist")
    records = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(residual_fold, (out, i)) for i in range(len(reg["folds"]))]
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(json.dumps(record), flush=True)
    write_json(out / "freeze.json", dict(registration_sha256=sha256(out / "registration.json"),
                                        models=sorted(records, key=lambda r: r["index"])))


def evaluate_residual(out):
    reg = typenet_verify(out)
    freeze = json.loads((out / "freeze.json").read_text())
    assert freeze["registration_sha256"] == sha256(out / "registration.json")
    assert len(freeze["models"]) == len(reg["folds"]) == 25
    if (out / "results.json").exists():
        raise FileExistsError("Evaluation already completed")
    with np.load(out / "dataset.npz", allow_pickle=False) as cache:
        data = {key: cache[key] for key in cache.files}
    ids = np.unique(data["owner"])
    first = np.array([np.flatnonzero(data["owner"] == uid)[0] for uid in ids])
    labels, roles = data["y"][first], data["split"][first]
    scores, rows, cv = defaultdict(list), [], []
    for record in freeze["models"]:
        path = out / f"fold{record['index']:02d}.joblib"
        assert sha256(path) == record["sha256"]
        bundle = joblib.load(path)
        assert bundle["registration_sha256"] == freeze["registration_sha256"]
        cv.extend(dict(index=record["index"], **row) for row in bundle["results"])
        for mode in reg["modes"]:
            probability = predict_residual(bundle["models"][mode], data["x"], data["baseline"])
            grouped = np.array([probability[data["owner"] == uid].mean() for uid in ids])
            scores[mode].append(grouped)
            for role in ("validation", "test"):
                mask = roles == role
                rows.append(dict(index=record["index"], mode=mode, role=role,
                                 **metrics(labels[mask], grouped[mask], "classification")))
    aggregates = []
    for mode in reg["modes"]:
        scores[mode] = np.stack(scores[mode])
        for role in ("validation", "test"):
            group = [r for r in rows if r["mode"] == mode and r["role"] == role]
            mask = roles == role
            aggregate = {key: dict(mean=float(np.mean([r[key] for r in group])),
                                  sd=float(np.std([r[key] for r in group], ddof=1)))
                         for key in ("accuracy", "balanced_accuracy", "auroc", "sensitivity", "specificity")}
            aggregates.append(dict(mode=mode, role=role, people=int(mask.sum()), models=len(group), **aggregate,
                ensemble=metrics(labels[mask], scores[mode].mean(0)[mask], "classification")))
    np.savez_compressed(out / "predictions.npz", ids=ids, y=labels, split=roles, **scores)
    write_json(out / "results.json", dict(counts=reg["counts"], rows=rows, cv=cv, aggregates=aggregates,
        freeze_sha256=sha256(out / "freeze.json"), predictions_sha256=sha256(out / "predictions.npz"),
        sd_definition="Sample SD across 25 models evaluated on the same fixed participants"))
    print(json.dumps(aggregates, indent=2), flush=True)


class ResidualTabNet(ResidualNODE):
    def __init__(self, mode):
        with torch.random.fork_rng(devices=[]):
            super().__init__(mode)
        self.head = neural_model("tabnet", 128 if mode == "residual" else 256, 2)


def fit_tabnet_adapter(x, baseline, y, owners, mode, epochs):
    transformed, state = preprocess(x)
    reference, _ = preprocess(baseline.reshape(-1, 128), state)
    current = torch.from_numpy(transformed)
    reference = torch.from_numpy(reference.reshape(baseline.shape))
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    model = ResidualTabNet(mode)
    model.initialize(current, reference)
    initial = {name: value.detach().clone() for name, value in model.named_parameters()}
    counts = Counter(owners)
    weights = np.asarray([1/counts[uid] for uid in owners], dtype=np.float32)
    weights *= len(weights)/weights.sum()
    weights, target = torch.from_numpy(weights), torch.tensor(y, dtype=torch.long)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=1e-4)
    rng = np.random.default_rng(SEED)
    model.train()
    updates = 0
    for _ in range(epochs):
        for index in np.array_split(rng.permutation(len(y)), max(1, math.ceil(len(y)/512))):
            optimizer.zero_grad()
            output, penalty = model.head(model.representation(current[index], reference[index]))
            loss = (F.cross_entropy(output, target[index], reduction="none")*weights[index]).mean() + .001*penalty
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite TabNet adapter loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            updates += 1
    return dict(family="tabnet", mode=mode, preprocess=state, state=model.state_dict(), epochs=epochs,
                optimizer_updates=updates, batch_size=512, real_people=len(counts), fit_rows=len(y), parents=[],
                parameter_changes={name: float((value.detach()-initial[name]).norm()) for name, value in model.named_parameters()})


def adapter_features(adapter, x, baseline):
    transformed, _ = preprocess(x, adapter["preprocess"])
    reference, _ = preprocess(baseline.reshape(-1, 128), adapter["preprocess"])
    reference = reference.reshape(baseline.shape)
    network = ResidualNODE(adapter["mode"])
    network.load_state_dict(adapter["state"])
    network.eval().requires_grad_(False)
    rows = []
    with torch.inference_mode():
        for start in range(0, len(x), 512):
            rows.append(network.representation(torch.from_numpy(transformed[start:start+512]),
                                               torch.from_numpy(reference[start:start+512])).numpy())
    return np.concatenate(rows)


def predict_adapters(model, x, baseline):
    if "adapter" in model:
        return predict(model["classifier"], adapter_features(model["adapter"], x, baseline))
    if model.get("family") != "tabnet":
        return predict(model, x)
    transformed, _ = preprocess(x, model["preprocess"])
    reference, _ = preprocess(baseline.reshape(-1, 128), model["preprocess"])
    reference = reference.reshape(baseline.shape)
    network = ResidualTabNet(model["mode"])
    network.load_state_dict(model["state"])
    network.eval()
    scores = []
    with torch.inference_mode():
        for start in range(0, len(x), 512):
            output = network(torch.from_numpy(transformed[start:start+512]), torch.from_numpy(reference[start:start+512]))
            scores.append(output.softmax(-1)[:, 1].numpy())
    return np.concatenate(scores)


def prepare_adapters(out, baseline):
    baseline = baseline.resolve()
    original = json.loads((baseline / "registration.json").read_text())
    assert sha256(baseline / "dataset.npz") == original["dataset_sha256"]
    assert sha256(ROOT / original["encoder"]) == original["encoder_sha256"]
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(baseline / "dataset.npz", out / "dataset.npz")
    reg = dict(original, families=["xgboost", "random_forest", "tabnet"],
        data_run=str(baseline.relative_to(ROOT)), data_registration_sha256=sha256(baseline / "registration.json"),
        data_source_sha256=sha256(baseline / "source.zip"),
        tree_adapter="Fresh separate adapter per tree family, mode and fold; train with temporary NODE head, freeze representation, fit trees",
        tabnet_adapter="Fresh jointly trained attention/correction and TabNet; original .001 attention penalty",
        fits=dict(classifiers=225, additional_adapter_training=100),
        source_hashes={**source_hashes(), "paper_typenet/nn.py": sha256(ROOT / "paper_typenet/nn.py")})
    write_json(out / "registration.json", reg)
    with zipfile.ZipFile(out / "source.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in reg["source_hashes"]:
            archive.write(ROOT / name, name)
        archive.write(Path(__file__).with_name("README.md"), "README.md")
    print(json.dumps(dict(fits=reg["fits"], families=reg["families"], modes=reg["modes"], counts=reg["counts"])), flush=True)


def adapter_fold(args):
    out, index = args
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        reg = json.loads((out / "registration.json").read_text())
        fold = reg["folds"][index]
        with np.load(out / "dataset.npz", allow_pickle=False) as cache:
            keep = cache["split"] == "train"
            data = {key: cache[key][keep] for key in ("x", "baseline", "y", "owner")}
        fit, score = np.isin(data["owner"], fold["fit"]), np.isin(data["owner"], fold["score"])
        assert not np.any(fit & score) and np.all(fit | score)
        models, results = {}, []
        for family in reg["families"]:
            for mode in reg["modes"]:
                if mode == "population":
                    model = fit_model(family, data["x"][fit], data["y"][fit], data["owner"][fit], "classification", "none", reg["epochs"])
                elif family == "tabnet":
                    model = fit_tabnet_adapter(data["x"][fit], data["baseline"][fit], data["y"][fit], data["owner"][fit], mode, reg["epochs"])
                else:
                    adapter = fit_residual(data["x"][fit], data["baseline"][fit], data["y"][fit], data["owner"][fit], mode, reg["epochs"])
                    features = adapter_features(adapter, data["x"][fit], data["baseline"][fit])
                    classifier = fit_model(family, features, data["y"][fit], data["owner"][fit], "classification", "none", reg["epochs"])
                    model = dict(classifier=classifier, adapter=adapter, adapter_origin="fresh training in this family/mode/fold")
                name = f"{family}_{mode}"
                models[name] = model
                p = predict_adapters(model, data["x"][score], data["baseline"][score])
                results.append(dict(family=family, mode=mode, metrics=personal_scores(data, score, p)))
        path = out / f"fold{index+1:02d}.joblib"
        if path.exists():
            raise FileExistsError(path)
        joblib.dump(dict(models=models, results=results, fold=fold,
                         registration_sha256=sha256(out / "registration.json")), path, compress=3)
        return dict(index=index+1, sha256=sha256(path))


def train_adapters(out):
    from concurrent.futures import as_completed
    reg = typenet_verify(out)
    if (out / "freeze.json").exists() or list(out.glob("fold*.joblib")):
        raise FileExistsError("Training artifacts already exist")
    records = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(adapter_fold, (out, i)) for i in range(len(reg["folds"]))]
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(json.dumps(record), flush=True)
    write_json(out / "freeze.json", dict(registration_sha256=sha256(out / "registration.json"),
                                        models=sorted(records, key=lambda r: r["index"])))


def evaluate_adapters(out):
    reg = typenet_verify(out)
    freeze = json.loads((out / "freeze.json").read_text())
    assert freeze["registration_sha256"] == sha256(out / "registration.json")
    assert len(freeze["models"]) == 25
    if (out / "results.json").exists():
        raise FileExistsError("Evaluation already completed")
    with np.load(out / "dataset.npz", allow_pickle=False) as cache:
        data = {key: cache[key] for key in cache.files}
    ids = np.unique(data["owner"])
    first = np.array([np.flatnonzero(data["owner"] == uid)[0] for uid in ids])
    labels, roles = data["y"][first], data["split"][first]
    scores, rows, cv = defaultdict(list), [], []
    for record in freeze["models"]:
        path = out / f"fold{record['index']:02d}.joblib"
        assert sha256(path) == record["sha256"]
        bundle = joblib.load(path)
        assert bundle["registration_sha256"] == freeze["registration_sha256"]
        cv.extend(dict(index=record["index"], **r) for r in bundle["results"])
        for name, model in bundle["models"].items():
            probability = predict_adapters(model, data["x"], data["baseline"])
            grouped = np.array([probability[data["owner"] == uid].mean() for uid in ids])
            scores[name].append(grouped)
            for role in ("validation", "test"):
                mask = roles == role
                rows.append(dict(index=record["index"], candidate=name, role=role,
                                 **metrics(labels[mask], grouped[mask], "classification")))
    aggregates = []
    for name in scores:
        scores[name] = np.stack(scores[name])
        for role in ("validation", "test"):
            group = [r for r in rows if r["candidate"] == name and r["role"] == role]
            mask = roles == role
            aggregate = {key: dict(mean=float(np.mean([r[key] for r in group])),
                                  sd=float(np.std([r[key] for r in group], ddof=1)))
                         for key in ("accuracy", "balanced_accuracy", "auroc", "sensitivity", "specificity")}
            aggregates.append(dict(candidate=name, role=role, people=int(mask.sum()), models=len(group), **aggregate,
                ensemble=metrics(labels[mask], scores[name].mean(0)[mask], "classification")))
    np.savez_compressed(out / "predictions.npz", ids=ids, y=labels, split=roles, **scores)
    write_json(out / "results.json", dict(counts=reg["counts"], rows=rows, cv=cv, aggregates=aggregates,
        freeze_sha256=sha256(out / "freeze.json"), predictions_sha256=sha256(out / "predictions.npz"),
        sd_definition="Sample SD across 25 models evaluated on the same fixed participants"))
    print(json.dumps(aggregates, indent=2), flush=True)


def check():
    """Synthetic-only checks; never opens clinical validation or test files."""
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=(60, 7))
    y = np.tile([0, 1], 30)
    ids = np.array([f"person{i}" for i in range(60)])
    roles = split_indices(ids, y)
    assert [len(roles[r]) for r in ROLES] == [42, 6, 12]
    assert all(np.array_equal(roles[r], split_indices(ids, y)[r]) for r in ROLES)
    fit, score = roles["train"], roles["test"]
    transformed, state = preprocess(x[fit])
    shifted, _ = preprocess(x[score] + 1e6, state)
    assert abs(shifted.mean()) > 1e5 and np.allclose(transformed.mean(0), 0, atol=1e-6)
    sx, sy, parents = augment(transformed, y[fit], ids[fit])
    assert np.bincount(sy).tolist() == [42, 42]
    for row, (left, right, weight) in zip(sx[len(fit):], parents):
        assert left in ids[fit] and right in ids[fit] and left != right
        a, b = transformed[np.flatnonzero(ids[fit] == left)[0]], transformed[np.flatnonzero(ids[fit] == right)[0]]
        assert np.allclose(row, a + weight*(b-a), atol=1e-6)
        assert y[np.flatnonzero(ids == left)[0]] == y[np.flatnonzero(ids == right)[0]]
    for kind in ("classification", "regression"):
        for name in MODELS:
            encoder = dict(preprocess=state, state=neural_model("tabnet", 7, 7).state_dict())
            model = fit_model(name, x[fit], y[fit], ids[fit], kind, "none", 2, encoder)
            prediction = predict(model, x[score])
            metrics(y[score], prediction, kind)
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "model.joblib"
                joblib.dump(model, path)
                assert np.array_equal(prediction, predict(joblib.load(path), x[score]))
    assert permutation_p(np.array([0, 0, 0, 1, 1, 1]), np.arange(6)) == .05
    grouped = dict(ids=np.array(["a", "b"]), views=np.zeros((9, 7)), owners=np.array(["a"] * 2 + ["b"] * 7))
    constant = predict_people(dict(constant=.6), grouped)
    assert np.array_equal(constant, np.full(2, .6))
    assert metrics(np.array([0, 1]), constant, "classification")["auroc"] == .5
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "sealed.json"
        write_json(path, {})
        try:
            write_json(path, {})
        except FileExistsError:
            pass
        else:
            raise AssertionError("Seal was overwritten")
    print("Passed: deterministic splits, train-only preprocessing, SMOTE lineage, all model heads, checkpoint round trips, exact permutation, overwrite refusal.")


def __getattr__(name):
    if name in {"PersonalMonitor", "BaselineMonitor", "baseline_distance", "replay_baselines"}:
        from prototype_net.exp4 import baseline
        return getattr(baseline, name)
    raise AttributeError(name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("classify", "export", "fetch", "prepare", "prepare-aalto", "expand", "distribution", "train", "evaluate", "report", "combined", "typenet-prepare", "typenet-train", "typenet-evaluate", "external", "real-prepare", "embeddings-prepare", "real-evaluate", "monitor", "baseline-monitor", "baseline-replay", "simulate", "simulate-impaired", "personal-prepare", "personal-embeddings-prepare", "personal-train", "personal-evaluate", "residual-prepare", "residual-train", "residual-evaluate", "adapters-prepare", "adapters-train", "adapters-evaluate", "check"), required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "runs")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--baseline", type=Path, default=Path(__file__).parent / "runs")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--run", type=Path, default=Path(__file__).parent / "runs/embeddings")
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--baseline-batches", type=int, default=1)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        if args.phase in ("classify", "export"):
            if args.output.exists():
                raise FileExistsError(args.output)
            engine = TypingClassifier.load(args.artifact) if args.artifact else TypingClassifier(args.run)
            if args.phase == "export":
                engine.export(args.output)
            else:
                if args.input is None:
                    parser.error("classify requires --input")
                write_json(args.output, engine.classify(json.loads(args.input.read_text())))
        elif args.phase == "fetch":
            fetch_tappy()
        elif args.phase == "prepare":
            prepare(args.output)
        elif args.phase == "prepare-aalto":
            prepare_aalto(args.output)
        elif args.phase == "expand":
            expand(args.output, args.baseline)
        elif args.phase == "distribution":
            prepare_distribution(args.output, args.baseline)
        elif args.phase == "train":
            train(args.output, args.epochs)
        elif args.phase == "evaluate":
            evaluate(args.output)
        elif args.phase == "report":
            comparison(args.output)
        elif args.phase == "combined":
            combined(args.output)
        elif args.phase == "typenet-prepare":
            prepare_typenet(args.output, args.baseline, args.epochs)
        elif args.phase == "typenet-train":
            train_typenet(args.output)
        elif args.phase == "typenet-evaluate":
            evaluate_typenet(args.output)
        elif args.phase == "real-prepare":
            prepare_real(args.output, args.baseline, args.epochs)
        elif args.phase == "embeddings-prepare":
            prepare_embeddings(args.output, args.baseline)
        elif args.phase == "real-evaluate":
            evaluate_real(args.output)
        elif args.phase == "personal-prepare":
            prepare_personal(args.output, args.baseline, args.epochs)
        elif args.phase == "personal-embeddings-prepare":
            prepare_personal_embeddings(args.output, args.baseline)
        elif args.phase == "personal-train":
            train_personal(args.output)
        elif args.phase == "personal-evaluate":
            evaluate_personal(args.output)
        elif args.phase == "residual-prepare":
            prepare_residual(args.output, args.baseline)
        elif args.phase == "residual-train":
            train_residual(args.output)
        elif args.phase == "residual-evaluate":
            evaluate_residual(args.output)
        elif args.phase == "adapters-prepare":
            prepare_adapters(args.output, args.baseline)
        elif args.phase == "adapters-train":
            train_adapters(args.output)
        elif args.phase == "adapters-evaluate":
            evaluate_adapters(args.output)
        elif args.phase == "simulate-impaired":
            simulate_impaired(args.output, args.baseline)
        elif args.phase == "simulate":
            simulate_monitor(args.output, args.baseline)
        elif args.phase == "baseline-replay":
            from prototype_net.exp4.baseline import replay_baselines
            replay_baselines(args.output, args.baseline)
        elif args.phase == "baseline-monitor":
            from prototype_net.exp4.baseline import BaselineMonitor
            if args.input is None or args.state is None:
                parser.error("baseline-monitor requires --input and --state")
            if args.output.exists():
                raise FileExistsError(args.output)
            if len({p.resolve() for p in (args.input, args.state, args.output)}) != 3:
                parser.error("Input, state and output paths must differ")
            payload = json.loads(args.input.read_text())
            engine = TypingClassifier(args.baseline)
            monitor = BaselineMonitor(engine, payload["participant_id"], payload["context_id"], args.baseline_batches)
            if args.state.exists():
                monitor.restore(args.state)
            result = monitor.observe(payload)
            monitor.save(args.state)
            write_json(args.output, result)
        elif args.phase == "monitor":
            from prototype_net.exp4.baseline import PersonalMonitor
            if args.input is None or args.state is None:
                parser.error("monitor requires --input and --state")
            if args.output.exists():
                raise FileExistsError(args.output)
            payload = json.loads(args.input.read_text())
            monitor = PersonalMonitor(args.baseline, payload["participant_id"])
            if args.state.exists():
                monitor.restore(args.state)
            result = monitor.observe(payload)
            if result["status"] != "insufficient_data":
                monitor.save(args.state)
            write_json(args.output, result)
        elif args.phase == "external":
            validate_external(args.output, args.baseline)
        else:
            check()


if __name__ == "__main__":
    main()
