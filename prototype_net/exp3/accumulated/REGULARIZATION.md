# Regularization on an unchanged benchmark

This study holds the proposed new model families and synthetic-normal generation
experiments aside. It changes only regularization and training duration in the
current scikit-learn timing-summary tree classifier.

All candidates consume the **exact existing `runs/v1/*.npz` arrays**. Training,
selection, calibration, validation and test people, row order, synthetic draws,
labels, features and observation budgets remain unchanged. Hashes of every
cache, source and original model are registered before training. No new data,
perturbations, cohort filtering or feature selection is introduced.

The comparator is the frozen original model for the same observation budget.
Five-window and ten-window results remain separate tasks. Older prototypes
trained on different cohorts or observation budgets are not directly ranked
against this experiment. Future attempts intended for a fair comparison should
use this same registered data contract (and original raw-window definitions if
the model consumes sequences), rather than selecting easier participants.

## Predeclared search

Four tree configurations vary depth (3–5), maximum leaves (7–31), minimum leaf
size (200–500 versus the original 40), L2 penalties (20–100), and feature
subsampling (one configuration uses 70%). Learning rate is 0.04. Each fit is
examined at 75, 150, 300 and 600 boosting iterations. Warm starting avoids
repeating earlier iterations; a small deterministic check verified that it
produces identical predictions to an uninterrupted fit, including feature
subsampling. Automatic row-random early stopping is disabled.

A candidate must have selection accuracy no more than **0.5 percentage points
below its original model**. Among eligible candidates, select the lowest raw
selection log loss, then highest accuracy. The original model is eligible.
The guard prevents reducing the train–selection gap by accepting substantial
underfitting; the gap itself is reported, not optimized in isolation.

Model selection uses selection data only. Each selected model receives a scalar
temperature fitted on the same separate calibration participants as before.
The model, temperature and protocol are frozen before comparison. No model is
refit on train plus selection. All examined checkpoints remain in `search.json`.

## Evaluation status

The user explicitly requested the same validation/test data. Those outcomes
have already been inspected, and the choice to investigate regularization was
motivated by prior results. This is an honest **matched comparative benchmark**,
not a fresh untouched test. The fixed search is not revised after its comparison.
No validation/test score is used to select a candidate.

`compare` first reproduces the old saved probability arrays to within 1e-12 and
verifies identical labels and row identities. It reports four-way accuracy,
paired participant-bootstrap intervals, EER, raw/calibrated log loss, Brier,
ECE, confusion and class recalls. A smaller fitting gap without improved
held-out performance is not described as an accuracy breakthrough.

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 LOKY_MAX_CPU_COUNT=4 python3 -m prototype_net.exp3.accumulated.regularize --phase develop
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 LOKY_MAX_CPU_COUNT=4 python3 -m prototype_net.exp3.accumulated.regularize --phase compare
python3 -m prototype_net.exp3.accumulated.predict observed.npz --model-dir prototype_net/exp3/accumulated/runs/regularization_v1
```

These commands have single-use guards; do not rerun a completed study. New
artifacts live in `runs/regularization_v1/`; original v1 artifacts are unchanged.
The existing 17 preprocessing/model controls passed before this study started.
