# Pause-rate experiment on the full Aalto freeze

Same people and synthetic draws as the full-Aalto parent freeze. Pause features use observed press-to-press versus each person's clean baseline; they do not use the generator pause mask. Selection is among pause, augmented, and augmented+pause by 4-way accuracy, then log loss. Chance accuracy is 25%.

Cohort: train 55,732; selection 1,306; calibration 1,034; validation 1,202; test 2,738.

Selection candidates:
- summary (618 features): selection 64.34%, log loss 0.7894; train 66.41% (4-way); parent 64.34%
- pause (42 features): selection 42.96%, log loss 1.1852; train 45.73% (4-way)
- augmented (1443 features): selection 66.21%, log loss 0.7527; train 68.64% (4-way); parent 66.21%
- augmented_pause (1485 features): selection 67.40%, log loss 0.7449; train 69.06% (4-way)

Selected: **augmented_pause**. Train accuracy 69.06%; selection accuracy 67.40% (4-way, chance 0.25).

## Parent freeze (augmented, no pause features)

- Validation: 64.70%; EER 14.81%. Recall normal 78.5%, mild 55.7%, moderate 50.7%, severe 74.0%.
- Test: 65.75%; EER 13.93%. Recall normal 79.8%, mild 57.6%, moderate 51.5%, severe 74.1%.

## Four-class accuracy versus the 618-feature summary

- **Validation:** 63.14% → **65.45%**; change +2.31 pp (95% participant-bootstrap CI +1.37, +3.31).
  EER 15.86% → 14.57%; ECE 1.85 → 1.37 pp; log loss 0.7982 → 0.7573.
  Recall: normal 78.4%, mild 55.4%, moderate 51.5%, severe 76.5%.
  Versus previous augmented: 64.70% → 65.45%; change +0.75 pp (95% CI +0.06, +1.50).
- **Test:** 64.11% → **66.66%**; change +2.56 pp (95% participant-bootstrap CI +1.90, +3.19).
  EER 15.75% → 13.99%; ECE 1.06 → 1.50 pp; log loss 0.7817 → 0.7348.
  Recall: normal 80.1%, mild 58.7%, moderate 53.1%, severe 74.8%.
  Versus previous augmented: 65.75% → 66.66%; change +0.91 pp (95% CI +0.43, +1.40).

Pause descriptors are observed press-to-press rates and excess mass relative to the enrolled person, including CR/space and backspace/delete gates. They are not minimum-entropy deconvolution. These synthetic classes are not clinically validated severity labels.
