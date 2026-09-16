# Matched-data regularization results

Every candidate used the exact cached v1 training/selection/calibration/validation/test arrays. Features, participant membership, labels, synthetic draws, and observation budgets are unchanged. Existing baseline artifacts are preserved.

Validation/test were previously inspected and reused at the user’s explicit request. This is a paired comparative benchmark, not new untouched external evidence. No candidate was selected using these outcomes.

Selection minimizes raw selection log loss while allowing no more than a 0.5-percentage-point selection-accuracy loss versus the original model. The original model remains eligible. Four regularization configurations and four iteration checkpoints were fixed before training; each budget has independent temperature calibration.

## 5 windows per side

Selected: config 3, 600 iterations.

Original training / selection accuracy: 83.6708% / 59.2500%.
New training / selection accuracy: 71.3042% / 58.7812%.

- Validation four-way accuracy: 58.38% → 58.44%. Paired change +0.06 pp (95% participant-bootstrap CI -0.81, +1.03).
  EER: 20.67% → 20.40%; calibrated ECE: 2.31 → 1.97 pp; log loss: 0.8980 → 0.8931.
- Test four-way accuracy: 59.54% → 59.34%. Paired change -0.20 pp (95% participant-bootstrap CI -0.86, +0.49).
  EER: 19.20% → 19.32%; calibrated ECE: 1.16 → 0.89 pp; log loss: 0.8847 → 0.8844.

## 10 windows per side

Selected: config 1, 600 iterations.

Original training / selection accuracy: 96.5292% / 64.2500%.
New training / selection accuracy: 72.5375% / 64.1562%.

- Validation four-way accuracy: 61.75% → 62.75%. Paired change +1.00 pp (95% participant-bootstrap CI -0.06, +2.03).
  EER: 16.40% → 16.69%; calibrated ECE: 3.41 → 2.47 pp; log loss: 0.8162 → 0.8174.
- Test four-way accuracy: 62.80% → 62.90%. Paired change +0.10 pp (95% participant-bootstrap CI -0.48, +0.80).
  EER: 16.18% → 16.14%; calibrated ECE: 2.25 → 1.69 pp; log loss: 0.8055 → 0.8030.

A smaller training–selection gap alone is not success: reduced capacity can also underfit. Generalization metrics and paired uncertainty determine whether the change helped. This experiment makes no claim about clinical cognitive severity.

Predict with the existing accumulated.predict CLI and --model-dir pointing to this run. No other experiment family, extra observations, new generator, or neuroQWERTY evaluation was introduced.
