# Additional keystroke data and synthetic-generation recommendation

Checked 2026-09-16. This was a source/schema investigation and local availability
audit, not a new model evaluation. No new dataset was ingested into training.

Start with [emg2qwerty](https://github.com/facebookresearch/emg2qwerty) keyboard
metadata: the official loader exposes individual key presses and releases.
Retain Aalto for broad participant variation. Pursue
[Clarkson II](https://citer.clarkson.edu/clarkson-university-keystroke-dataset-ii/)
for naturalistic long histories; the institutional resource page specifies
[access by request](https://citer.clarkson.edu/research-resources/biometric-dataset-collections-2/).
Neither candidate's per-person eligibility for 100/200 windows has been audited.

Further candidates are the [LSIA human-written release](https://data.mendeley.com/datasets/mzm86rcxxd/2),
[expanded Tappy](https://data.mendeley.com/datasets/z39mhdsynx/3),
[KUPA-KEYS](https://huggingface.co/datasets/ALTACambridge/KUPA-KEYS),
and [Buffalo](https://www.buffalo.edu/cubs/research/datasets.html).
The side-by-side comparison is saved in the workspace's managed canvases
folder as `keystroke-dataset-options.canvas.tsx`.

## Compatibility is mathematical, not a matching column name

For adjacent keypresses ordered by press time p, with release time r:

- HL[i] = r[i] - p[i]
- PL[i] = p[i+1] - p[i]
- IL[i] = PL[i] - HL[i]
- RL[i] = PL[i] + HL[i+1] - HL[i]

Use seconds and a declared key mapping. Match down/up events for overlapping
keys; handle auto-repeat and missing keyups explicitly. Retain genuine negative
IL. Never compute transitions across a removed event or unrelated session.
If a dataset stores previous-to-current press latency at row i, it corresponds
to PL[i-1], not PL[i].

The [LSIA publication](https://pmc.ncbi.nlm.nih.gov/articles/PMC10139888/)
defines FT as press-to-press, and censors timings above 1500 ms to -1.
Those values must be missing/boundaries, not clamped or reconstructed as actual
pauses. Use HUMAN files only, with source-family deduplication. Its older linked
[latency-only release](https://www.lsia.fi.uba.ar/pub/papers/kd-dataset)
contains shuffled latency arrays, so is not a substitute for ordered sequences.

Tappy has hand/space categories instead of exact key identities. It requires
retraining a timing/hand representation and adapting any key-conditioned
perturbation logic. It cannot directly supply our exact-key features. Already
local data are audited by `audit_tappy.py`; results are in
`local_tappy_availability.json`. Counts are upper bounds before continuity,
duplication and chronological splitting checks. Non-PD status is self-reported,
not evidence of cognitive health. Existing generator parameters were informed
by Tappy/neuroQWERTY aggregate statistics, so these sources cannot be described
as untouched external confirmation of the generator.

## Proposed Monte Carlo experiment

Monte Carlo supplies repeated samples from a specified generative model. The
scientific question is whether that model represents plausible typing histories.
A useful next simulator would generate new clean typing histories, extending
our current method of perturbing existing recordings.

1. Reserve real participants and chronological evaluation blocks first. Fit
   all population, day/session, key/digraph and noise distributions on training
   identities only. A held-out person's clean enrollment may condition that
   person's baseline model, but their future query must not fit it.
2. Fit a hierarchical model: a persistent person profile; key-dependent log
   hold/press timing; slowly changing day/session effects; a burst/pause state;
   correlated within-burst residuals. Empirical block resampling is a simpler
   first comparator. Blocks remain synthetic reuse, not extra empirical data.
3. Sample independent baseline/query periods under the same virtual person.
   Preserve realistic persistent effects across many windows; do not redraw
   them independently each time. Apply predeclared perturbations to the query
   condition, using the same clean generator for all four classes.
4. Generate timestamps from positive hold and press intervals, deriving IL/RL
   consistently. Condition on text/key sequences from training sources, or use
   a separately fitted key-sequence model. Do not expose generator parameters.
5. Keep source descendants in their original role. Compare distributions,
   pauses, correlations, key effects and day drift with held-out real typing.
   Run cross-generator sensitivity checks. Independent random seeds are not
   independent evidence that the simulator is realistic.
6. Report separately: classification on distinct real source windows with
   synthetic perturbations; classification on fully generated histories; and
   any later clinical assessment. Calibrate on the distribution appropriate
   to each claim. Real clean recordings can test false alarms, but do not by
   themselves validate synthetic severity labels.

An interpretable simulator could efficiently explore 100/200-window budgets.
Its performance would remain conditional on its assumptions. It cannot create
new independent participants, guarantee a target accuracy, or establish
real-world cognitive-health calibration. Published keystroke synthesis methods
already exist (e.g. [KSDSLD](https://www.sciencedirect.com/science/article/pii/S2665963822001385));
that establishes feasibility of synthesis, not validity of health labels.
