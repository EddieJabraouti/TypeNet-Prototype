# Accumulated-window experiment

This extends exp3 with equal baseline/query observation counts. It uses newly
trained timing-summary trees and no TypeNet weights. Existing exp3 models and
sealed results are preserved.

## Available data determines the honest experiment

A complete outcome-blind census found 168,593 participants in the main Aalto
folder, each with 15 sentence sessions. The second folder contains 28,208 of
those IDs: 28,207 byte-identical files and one empty file. It adds no histories.

We recover keystrokes previously discarded by the 50-key loader, dividing each
sentence into **non-overlapping windows of at most 50 keys**, keeping remainders
only when they have at least six keys. Counting all windows before enforcing
the chronological split, 107,842 people have at least 20 windows, three have
at least 40, and zero have 100 or more. Thus 20, 50, 100, and 200 windows **per
side** cannot be evaluated at cohort scale with these recordings. Repeating
recordings or drawing new perturbations of the same recording does not fix this.

The current registered experiment compares **5 vs 10 windows per side** on the
same people. These are variable-length windows, not necessarily 50 keys each.
It does not simulate months of longitudinal data or justify extrapolation to
hundreds of windows.

## Chronology and synthetic conditions

Events are deduplicated within sentences; sentences are sorted by actual press
timestamps. The first chronological half of the sentences supplies baseline
windows and the second supplies query windows. No sentence crosses the baseline/
query boundary. Participants whose periods overlap in time are excluded.

The 10-window baseline uses the last ten windows before the boundary; the
5-window baseline uses its last five. Queries use the first ten or five windows
after the boundary. Each person needs ten usable windows on both sides. This
same eligibility rule applies to every role and both observation budgets.
Identical source-window hashes within/across selected participants are excluded
before synthetic outcomes are generated. Counts and reasons are logged.

A single synthetic profile is shared by all query windows in each class. The
five query windows are an exact subset of the ten-window realization, with the
same profile and perturbation draws. Generator configuration is unchanged.
Baseline is clean. No clean query twin, generator profile, ID, trace, class label,
or absolute timestamp is supplied to the classifier. Every class has the same
1-ms rounding and terminal-transition exclusions as original exp3.

Each observed query is compared against individual baseline-session summaries;
mean and variability across its query-period feature vectors produce one
618-feature period representation. This handles correlation by learning a
period classifier; it does **not** multiply window probabilities as if the
windows were independent. It does not model changing severity within the period.

## Roles, fitting, and calibration

The fixed target cohort sizes are 12,000 train, 800 selection, 600 calibration,
800 final-validation, and 2,000 test participants. All are identity-disjoint.
Train identities remain in the old training role. Selection comes from the old
selection pool, excluding every identity assigned to the original exp3 selection,
calibration, or validation subsets. Calibration/final-validation/test use distinct
parts of the old final-test pool, excluding every identity assigned to the
original exp3 test subset. No model weights are reused.

**Historical caveat:** these are fresh exp3 evaluation identities, but the old v4
population has already contributed aggregate metrics to earlier experiments.
They are not a pristine external cohort. Membership and source availability can
be inspected before training; synthetic final outcomes cannot.

For each budget, three predeclared gradient-boosted tree configurations are fit
with 300 iterations, no row-random early stopping, and selection by four-way
accuracy (log loss breaks ties). Separate scalar temperatures are then fitted
on calibration participants. All choices and temperatures are frozen before the
single-use final evaluation. Training runs on CPU because these scikit-learn
trees do not use MPS.

Metrics include four-way accuracy, participant-bootstrap confidence intervals,
pooled baseline-vs-perturbed EER, log loss, Brier score, 15-bin calibration error,
class recalls, confusion matrices, and the paired 10-minus-5 accuracy difference.
Bootstrap resamples people, keeping all four synthetic classes of one person
together. Calibration applies only to this balanced synthetic distribution.

## Reproduce and predict

```bash
python3 -m prototype_net.exp3.accumulated.audit
python3 -m unittest prototype_net.exp3.accumulated.test_accumulated prototype_net.exp3.test_controls prototype_net.exp3.test_sequence
python3 -m prototype_net.exp3.accumulated.experiment --phase develop
python3 -m prototype_net.exp3.accumulated.experiment --phase seal
python3 -m prototype_net.exp3.accumulated.predict observed.npz
```

Do not repeat the seal or retune from final outcomes. `--smoke` uses only old
training identities in five small, disjoint roles and verifies the workflow;
its scores are not experimental results.

Prediction NPZ keys are `gallery`, `gallery_lengths`, `query`, `query_lengths`.
Both sets must have exactly 5 or exactly 10 windows, shaped `(N,50,5)` in the
repository timing format, with integer lengths in `[6,50]`. The caller must
provide distinct windows from separate chronological periods. Unsupported
observation counts are rejected rather than assigned an unvalidated confidence.

Audit inventories, protocol, source hashes, models, caches, selection history,
calibration metrics, predictions, and evaluation records are under `runs/`.
