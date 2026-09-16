# Exp3: measurable timing changes with independent calibration

This experiment targets four **synthetic perturbation classes**, not a clinical
diagnosis. The desired 70–80% accuracy is a success criterion, not a reason to
change labels, discard hard examples, or alter the generator.

## Model, in plain English

Enroll ten clean typing sessions from a person. Summarize each new session's
hold times, press intervals, variability, long pauses and temporal correlations.
Compare these summaries with the individual clean sessions. Also measure hold
times relative to that person's usual timing for each key, shrinking estimates
for rarely observed keys toward their overall average.

A small gradient-boosted tree classifier learns combinations of those changes
that distinguish normal, mild, moderate and severe. Unlike a biometric encoder,
it is directly exposed to timing variability rather than trained to suppress
within-person differences. No TypeNet weights are required. Training uses four
CPU threads, leaving MPS available.

Two observation budgets are fixed before evaluation:

- **Single:** one session of up to 50 keys, plus ten clean enrollment sessions.
- **Bundle:** five query sessions under a shared perturbation profile, plus the
  same ten enrollment sessions. Mean and variability of their observed timing
  summaries form one prediction. This is a different task budget; its accuracy
  must never be presented as single-session accuracy.

There are 309 features per session and 618 per five-session bundle. They contain
only observed timings and enrollment comparisons, not generator parameters,
unperturbed versions of queries, participant IDs or labels. Quantile differences
are computed against each enrollment session before summarization; no TypeNet
gallery centroid is involved.

## Controls that matter

The existing generator preserves the clean query's content while modifying its
timings. Clean and perturbed versions therefore remain within one participant
split. Enrollment is sessions 0–9; queries are sessions 10–14. The clean query
is never supplied to the classifier when classifying its perturbed version.

The generator changes numerical precision and zeros its final transition.
These are accidental synthetic cues. All model inputs are rounded to the same
millisecond resolution, and every final transition is ignored for every class.
Only hold and press timings are consumed; reconstructed release/inter-key
identities cannot act as a class marker. These controls mitigate known artifacts
but do not establish realism of the synthetic distribution.

Participant IDs come from the existing locked v4 protocol. A stable hash chooses
6,000 training participants and 4,000 test participants. The old selection pool
is split into 1,200 model-selection, 1,000 calibration and 1,200 final-validation
participants. Missing/incomplete participants are logged and excluded using the
same label-independent rule: at least fifteen sessions, each with at least six
keys. Actual retained counts are recorded. Duplicate source sequences across
participants are rejected.

**Historical caveat:** aggregate test results from the old final-test population
have already been discussed. Exp3's final test is independent of its fitting and
selection, but it is not a never-before-studied external dataset. Fresh synthetic
draws use role-, participant- and class-specific seeds. The generator's severity
factors and overlapping profiles are unchanged.

Three fixed tree configurations are compared on selection participants for each
observation budget. Row-random early stopping is disabled. Calibration then fits
one temperature on separate participants. This preserves predicted classes;
it adjusts confidence, not accuracy. No fit uses final validation or test data.

Models, temperatures, source hashes, participant lists and protocol hashes are
frozen before evaluation. `seal` is single-use and writes predictions and an
audit trail. It does not choose a model based on final-validation/test scores.
Metrics include 4-way accuracy, participant-bootstrap confidence intervals,
class recalls, confusion counts, negative log likelihood, multiclass Brier,
15-bin confidence calibration and pooled normal-versus-perturbed detection EER.
That pooled EER differs from earlier per-person EER averages.

## Run and use

```bash
python3 -m unittest prototype_net.exp3.test_controls
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python3 -m prototype_net.exp3.run --phase develop
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python3 -m prototype_net.exp3.run --phase seal
python3 -m prototype_net.exp3.predict observed_sessions.npz --budget single
```

`observed_sessions.npz` must contain `gallery`, `gallery_lengths`, `query`, and
`query_lengths` as described in `predict.py`. Inputs use the repository's
`[hold, inter-key, press, release, keycode/255]` representation in seconds.
The bundle model assumes all five query sessions share one severity condition.
Do not treat its output as calibrated for a mixture of conditions.

Outputs live in `runs/timing_v1/`. `protocol.json` is the pre-evaluation record;
`search.json` is the complete selection history; `frozen.joblib` contains both
classifiers and temperatures. `sealed_results.json` contains final metrics.
Calibration applies to this balanced synthetic class distribution. A different
class prevalence or real-world population requires new independent validation.

## Extended sequence experiment

The summary-model search reached 54.4% selection accuracy for individual
sessions and 61.8% for five sessions. Two MPS MLP configurations did not beat
the trees. Final validation and test were kept closed. The complete search
and a recorded recovery from a tree-model loading race are preserved under
`runs/combined_v1/`; no score from the failed invocation was used as a new trial.

`sequence_train.py` expands training to all 61,627 identities assigned to the
original training role (60,060 pass the same completeness rule). It samples
fresh perturbations every training step, including a shared profile over the
five queries. The validation/selection identities remain separate.

The 260,296-parameter model learns from **ordered timing observations**. Each
keypress is described by its hold and press timing, its deviation from that
person's usual timing, and its deviation from that person's typical timing for
that key. Rare-key estimates are stabilized with eight pseudo-observations
from the person's overall baseline. These are statistical reference values
calculated only from clean enrollment, not a mean TypeNet embedding.

Four temporal convolution blocks inspect progressively longer stretches of
typing. Mean, variability and peaks of their responses describe each session.
One classifier uses one session; another combines five sessions. Both are
trained jointly with ordinary cross-entropy. No generator profile, pause mask,
original clean query, participant identifier or reconstructed timing channel
is a model input. All timings are rounded to milliseconds and final transitions
are excluded. A learned key embedding represents which key was pressed.

The registered schedule permits 100 epochs of 300 batches of 64 participants,
with AdamW, warmup, cosine learning-rate decay and a 20-epoch patience rule.
Checkpoints are selected separately for the two observation budgets by
selection accuracy, with log loss as the tie-breaker. Each epoch saves a
resumable optimizer/RNG checkpoint. These online augmented training accuracies
are distinct from evaluation on a fixed training set.

Before reading sequence outcomes, `finish_plan.json` fixes five probability
mixtures (0%, 25%, 50%, 75%, 100% sequence model, the rest tree). The selection
cohort chooses the weight; separate calibration participants fit a scalar
temperature. This permits complementary models without choosing an ensemble
on final-validation or test results. The tree-only candidate remains eligible.

```bash
python3 -m unittest prototype_net.exp3.test_controls prototype_net.exp3.test_sequence
python3 -m prototype_net.exp3.sequence_train --device mps
# Only to continue an interrupted identical run:
python3 -m prototype_net.exp3.sequence_train --device mps --resume
python3 -m prototype_net.exp3.sequence_finish --phase select --device mps
# Once all modeling choices are finished:
python3 -m prototype_net.exp3.sequence_finish --phase seal --device mps
python3 -m prototype_net.exp3.sequence_predict observed_sessions.npz --budget bundle
```

Training artifacts live in `runs/sequence_v1/`. Final calibrated inference
requires `final_freeze.json`, the selected sequence checkpoints and the frozen
tree models. A completed training run by itself is not a calibrated model.

## Full-gallery extension and final pipeline

The sequence-only run completed 100 epochs. Its best selection accuracy was
54.31% for individual sessions (epoch 82) and 64.08% for five sessions (epoch 96).
Those checkpoints are preserved. On a selection-only stress test that coarsens
both enrollment and every query class from 1-ms to 5-ms precision, accuracy was
54.06% and 63.91%, respectively. This checks one numerical shortcut; it does
not prove the entire simulator is realistic.

`relational_train.py` warm-starts the temporal encoder from that completed run.
It also encodes all ten individual clean enrollment sessions. Each query is
compared with each enrollment representation; a small network processes each
pair, and the mean, variability and strongest responses of those comparisons
form a baseline-aware query representation. Its 391,752 parameters are jointly trained.
The gallery is never replaced by a mean embedding before comparison. The
registered run uses 100 epochs, 300 steps, batches of 32 participants, and a
20-epoch patience rule. Its input validity and gallery-order invariance are
covered by tests.

The final route is **`finalize.py` and `final_predict.py`**. The earlier
`sequence_finish.py` and `sequence_predict.py` preserve the two-family plan but
are superseded by this explicitly registered three-family comparison. The
final plan considers fifteen fixed probability mixtures of the summary tree,
sequence-only model and full-gallery model, in quarter-weight increments.
All choices use the selection cohort; temperature fitting uses calibration
participants; final validation/test remain separate. The plan was recorded
before full-gallery training results were available.

```bash
python3 -m prototype_net.exp3.relational_train --device mps --wait-for-initialization
python3 -m prototype_net.exp3.finalize --phase select --device mps
python3 -m prototype_net.exp3.finalize --phase seal --device mps
python3 -m prototype_net.exp3.final_predict observed_sessions.npz --budget bundle
```

`runs/final_v1/freeze.json` records selected weights, temperatures and artifact
hashes. `runs/final_v1/results.json` is created only after the one-time final
evaluation. Prediction refuses modified frozen model artifacts or preprocessing
sources. Final reported confidence must come from this calibrated pipeline,
not an intermediate training checkpoint. Generated data caches, model binaries
and logs remain local and are excluded from Git by this directory's `.gitignore`.
