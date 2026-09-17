# Exp3: personal-baseline deviation detection

The current binary task requires personal enrollment: **ten untouched Aalto
gallery windows**, followed by ten separate query windows from the same person.
Class 0 queries are untouched. Class 1 queries receive added-pause,
timing-change, or combined perturbation patterns, with equal positive-pattern weight.
Untouched queries carry half the total training weight. The label means
**injected deviation**, not clinical diagnosis or severity.
These timing patterns are components of the same deviation-detection task,
not separate clinical diagnoses. Historical generator API names remain unchanged.

## Binary commands

```bash
python3 -m prototype_net.exp3.model --binary --phase study --workers 8
python3 -m prototype_net.exp3.model --binary --phase predict --input observed.npz
```

The study accesses **only the original training participants**. A deterministic
participant split assigns 39,012 to model fitting, 8,360 to development checks,
and 8,360 to threshold calibration. Feature normalization is fitted on model-fit
galleries only. The threshold targets at most 5% empirical false positives on
training-calibration negatives; it is never chosen from validation or test.
The same 1,485 features and tree hyperparameters are retained.

Every study automatically validates generation on training-development people:
three independent draws, weaker effect sizes, alternate pause shapes, timing
invariants, per-mechanism sensitivity, paired sensitivity-loss intervals,
variance measurements, and a simple timing-statistic stump as a shortcut probe.
A statistically supported loss exceeding ten percentage points is recorded as
insufficient weak-change coverage. This is a diagnostic rule, not a clinical
acceptance standard. Weak changes are not strengthened to improve accuracy.

Validation/test evaluation is a separate read-only command, to run only after
freezing development choices:

```bash
python3 -m prototype_net.exp3.model --binary --phase evaluate --workers 8
```

It writes a separate evaluation report without changing the model or threshold,
and refuses to overwrite a prior evaluation. Historical exposure means these
cohorts still cannot provide pristine confirmation. No selection-score gate is
used. The score is not a calibrated probability of cognitive impairment.

`observed.npz` must contain gallery/query arrays and lengths, as described below.
Prediction requires enrollment and never synthesizes or modifies inference
inputs. The current artifact is `runs_med/binary_deviation_trainonly.joblib`, with
JSON results and a protocol written before fitting. Feature caches default to
`exp3_binary_trainonly` under the system temporary directory (`--cache` overrides
this location). Existing output checkpoints are not overwritten.

The former `binary_deviation.joblib` used validation to choose its threshold and
is **superseded**. Its historical results remain documented in `RESULTS.md`; they
must not be presented as results for the corrected model. Its source hash also
predates this revision, so the current predictor deliberately rejects it rather
than bypassing the provenance check. The original four-class winner is unchanged.

`ClinicalDeviationGenerator` in the existing generator package owns the positive
perturbations. It adds pauses relative to the personal gallery; it does not
resynthesize controls or remove their natural variation. See `RESULTS.md` for
clinical-transfer assumptions and the distinction from the earlier 77.17% assay.

## Training-only representation comparison

A matched four-way experiment compares the existing 1,485 features with 426
additional distribution/temporal features, under original and broader training
strengths. All fitting, calibration and development participants come from the
original training pool. Weak-change sensitivity improves from 15.44% to 31.07%
with broad training plus richer features; the gain is concentrated in motor and
combined changes, while cognitive-only detection remains weak and has a trade-off
at original strength. See the detailed comparison in `RESULTS.md`.

Research models and exact experiment source are archived together in
`runs_med/binary_representation_probe.joblib`, with JSON results and protocol.
The default predictor remains the preceding training-only model; no candidate
was automatically promoted and no validation/test observations were consulted.

## Severity assessment and final binary evaluation

**The percentages in this historical subsection are superseded as headline
results.** They excluded the additional attenuation used in training. Equal
counts of the original mild/moderate/severe settings did not make their strength
distribution match training. See the corrected full-distribution assessment in
`RESULTS.md`; do not compare these figures directly with broad-training accuracy.

The subsequent six-step study tested the richer/broad candidate by perturbation
level, attempted loss reweighting and a fixed model blend, selected using only
training-development people, froze the checkpoint and threshold, then evaluated
validation/test. Neither correction improved the weakest-group selection
criterion; the richer/broad candidate was retained unchanged.

On a randomized, equally represented mixture of the original mild/moderate/severe
generator levels versus untouched queries, accuracy was **85.61% validation /
86.60% test**. Test positive sensitivity was **51.94% mild, 82.80% moderate,
97.41% severe**. Validation false positives were **6.57%**, exceeding the 5%
training-calibration target; test false positives were **4.64%**. These are
synthetic-deviation results, not clinical diagnostic accuracy. The weak-change
gap remains unresolved.

The separately assessed current clinical-generator mixture achieved **87.19%
validation / 87.25% test** randomized-mixture accuracy. Its cognitive mechanism
has no severity label; only motor components are stratified. These two benchmark
definitions must not be conflated. Full subgroup results, training metrics,
uncertainty and safeguards are in the final section of `RESULTS.md`.

The frozen research artifact is `runs_med/binary_severity_final.joblib`, with
`binary_severity_*.json` design, development, freeze, evaluation and verification
records in the same directory. The default predictor remains the preceding
1,485-feature training-only model; this research artifact has not been integrated
into that prediction command. No production model/generator source was changed
for this study.

## Frozen-source compatibility

The original model source is preserved verbatim in a readable `_LEGACY_SOURCE`
archive inside `model.py`. Its original protocol hash is verified before it is
executed in an isolated namespace. Legacy commands continue using that code and
unchanged checkpoints. NODE, TabNet, and gcForest source checks recognize the
verified archive; their own historical source hashes exclude only the explicitly
marked compatibility adapter. All architecture computation remains checked.
Binary artifacts instead record hashes of the entire current source files.

## Historical four-class experiment

This experiment classifies four **synthetic perturbation classes** (normal, mild,
moderate, severe). Chance accuracy is 25%. The 70–80% target is a success
criterion, not a reason to change labels or the generator.

The frozen winner is **augmented_pause**: the 1,443-feature parent plus 42
person-specific raw press-to-press pause descriptors. Its recorded accuracy is
69.06% train, 67.40% selection, 65.45% validation, and 66.66% test.
The latest completed comparison tested MED additions and retained this winner.
Hold results are in `HOLD_RESULTS.md`; pause results are in `PAUSE_RESULTS.md`.
See `RESULTS.md` for the comparison, subsequent architecture research, and the
generator/recoverability audit. The audit identifies a clinical measurement
mismatch and preserves the current generator and frozen benchmark.

## Model

Enroll ten clean typing windows from the first chronological half of a person's
sentences. Observe ten query windows from the second half. A single synthetic
profile is shared by every query window in a class. Baseline is always clean.

Each window is summarized with hold times, press-to-press intervals, and
key-adjusted hold residuals (57 statistics). The original period representation
is the mean and variability of the ten query-window summaries after comparing
each query with the ten individual baselines (**618 features**).

The kept model also compares every query window with every baseline window
(100 dependent pairs, not 100 independent observations). Those pairs add **825
descriptors**: signed-comparison quantiles, direction, baseline-calibrated tail
frequency and persistence, fixed-bin histogram entropy change, Jensen–Shannon
divergence, and seven-quantile absolute discrepancies for hold, press-to-press,
and key-adjusted hold. The selected representation concatenates both
(**1,443 features**).

Pause features are 42 descriptors of observed press-to-press intervals versus
that person's clean baseline. Hold features use the same 42-d recipe on raw
hold times (every key, not dropping the last event). Both are gated on all
keys, CR/space, and backspace/delete. The generator pause mask is not an
input. This run selects among med, augmented, augmented+med, augmented+pause, and
augmented+pause+med by selection accuracy, then log loss. The 618-feature
summary is fitted as a comparator only.

A gradient-boosted tree uses the regularized ten-window hyperparameters
(15 leaves, depth 4, 200 samples per leaf, L2 50, learning rate 0.04, 600
iterations). Temperature calibration on separate participants preserves
predicted classes and only rescales confidence.

No TypeNet weights, generator profile, pause mask, clean query twin, participant
ID, or label enters the classifier. All classes share millisecond rounding and
terminal-transition exclusion. Entropy features are empirical histogram
summaries, not minimum-entropy deconvolution or a fitted maximum-entropy density.

## Cohort and controls

Windows are non-overlapping, at most 50 keys, remainders kept only with at least
six keys. Sentences never cross the baseline/query boundary. People whose two
halves overlap in time are excluded, as are repeated source windows.

Target sizes are not capped. Every eligible Aalto identity is used: v4 train,
historical quarantine (TypeNet-unseen leftovers, 83,000 IDs), and the original
exp3 selection/calibration/validation identities for **training**; remaining
v4 selection identities for model selection; remaining v4 final-test identities
minus the original exp3 test subset for calibration, validation and test.
Eligibility still requires ten usable windows on each side of the sentence
boundary. Roles remain identity-disjoint.

The earlier 12,000-train matched comparison is recorded in `RESULTS.md`. This
run is a larger-cohort experiment on the same model family, not a replay of
that 16,200-person freeze.

These are fresh synthetic draws on the historical v4 population. Aggregate
metrics from that population were already discussed in earlier experiments.
Validation and test for this comparison were previously inspected with the
618-feature baseline; they are a matched benchmark, not an untouched external
set.

## Run

The frozen winner's archived code lives in `model.py`. Its original project imports are the
Aalto sequence loader and the synthetic perturbation generator. New architecture
experiments each occupy one self-contained source file: `node.py`, `tabnet.py`,
and `gcforest.py`. Each contains its own model, training, checkpoint handling,
and evaluation; raw-window inference reuses the existing frozen feature encoder.

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 LOKY_MAX_CPU_COUNT=4 python3 -m prototype_net.exp3.model --phase develop
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 LOKY_MAX_CPU_COUNT=4 python3 -m prototype_net.exp3.model --phase compare
python3 -m prototype_net.exp3.model observed.npz
```

`observed.npz` must contain `gallery`, `gallery_lengths`, `query`, and
`query_lengths`. Both sides are exactly ten windows, shaped `(10, 50, 5)` in
the repository `[hold, inter-key, press, release, keycode/255]` format, with
integer lengths in `[6, 50]`. Provide distinct windows from separate
chronological periods.

`develop` writes artifacts under `runs_med/`, reusing people and draws from
the parent freeze in `runs/` and pause features from `runs_pause/`. Develop
must not be repeated after freeze. `compare` is single-use on validation/test.
Prediction refuses modified frozen artifacts or a changed `model.py` / generator.

## Architecture experiments

The new experiments use the same 1,485 features and full training cohort.
**Training and validation drive development.** Validation accuracy chooses the
checkpoint/cascade depth, with log loss breaking ties; the historical selection
split is diagnostic only. Each fixed candidate is calibrated separately and
evaluated on test once, regardless of its selection accuracy. Validation is
therefore a development metric in this study, not an untouched held-out estimate.

```bash
python3 -m prototype_net.exp3.node --phase train
python3 -m prototype_net.exp3.node --phase compare
python3 -m prototype_net.exp3.tabnet --phase train
python3 -m prototype_net.exp3.tabnet --phase compare
python3 -m prototype_net.exp3.gcforest --phase train
python3 -m prototype_net.exp3.gcforest --phase compare
```

Candidates write architecture-named checkpoints inside the existing `runs_med/`
directory. They never overwrite `model.joblib` or the frozen result files.
Training refuses an existing candidate checkpoint, and comparison is single-use.
Use `--output` to select another checkpoint in an existing directory for an
explicitly different experiment. All architecture-specific changes belong in
that architecture's existing source file, with no extra helper modules.

NODE and TabNet support `--epochs`, `--patience`, `--batch-size`, and `--device`.
gcForest supports `--trees` and `--layers`. Defaults are bounded research
configurations rather than reproductions of every hyperparameter in the papers.
After comparison/calibration, `--phase predict --input observed.npz` uses the same
raw-window input contract as the frozen winner. See `RESULTS.md` for findings.


## Clinical duration generator (opt-in)

The generator package now exposes a separate **control-versus-impaired** mode
calibrated to the local clinical pause/burst measurements. Its implementation
lives in the existing `prototype_net/perturbation_generator/__init__.py`.
Legacy `perturb.py`, four-class labels, and saved exp3 models remain unchanged;
the compatibility checks are described above. This new binary mode is not automatically used by the commands above.

```python
import numpy as np
from prototype_net.perturbation_generator import (
    ClinicalCalibration, ClinicalTimingGenerator, measure_clinical_session,
)

calibration = ClinicalCalibration.from_csv(
    "data/clinical/ad_mci_writing/Participant_level.csv",
    participant_ids=clinical_training_ids,  # exclude clinical evaluation people
)
rng = np.random.default_rng(9192026)
profile = calibration.sample_profile("impaired", rng)  # or "control"
generator = ClinicalTimingGenerator(calibration)
observed = generator.generate_session(source_session, profile, rng)
measurements = measure_clinical_session(observed)
```

`source_session` is a finite, unpadded `[events, 5]` continuous timeline with
nonnegative hold and press-to-press durations. Generate the whole session before
windowing; do not substitute concatenated cached windows for missing boundary
intervals. Reuse a profile for repeated sessions from one synthetic person.
Both groups receive planning pauses; hold timing and key identity remain intact.
Only observable measurements should enter a classifier. The two-second pause
threshold and missing/censored burst handling are explicit in the API.

Calibration fits participant and task variability in mean durations. Default
individual dwell shapes are maximum-entropy assumptions; `burst_shape` and
`pause_shape` exist for sensitivity testing. Clinical severity tiers are not
inferred from this small cohort. See `RESULTS.md` for behavioral fit, binary
train/validation/test results, and exploratory clinical held-out evaluation.
