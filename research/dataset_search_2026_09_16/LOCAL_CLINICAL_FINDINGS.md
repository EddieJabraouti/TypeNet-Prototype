# What we can use from the existing clinical directory

Local files audited directly; no model training or clinical classification was
performed. The JSON audits and archive hashes in this directory make the counts
reproducible. Existing exp3 checkpoints and sealed evaluations were unchanged.

## neuroQWERTY: closest feature match, shorter histories

`data/clinical/neuroqwerty/neuroQWERTY.zip` contains 85 participants: 43 controls
and 42 Parkinson's cases, with 116 recordings. MIT-CS1PD has 31 people with two
recordings each; MIT-CS2PD has 54 with one each. The extracted folder is another
copy of the archive contents, not additional participants.

The audit retained 175,533 timed keyboard records after basic validity checks,
excluding mouse records and deduplicating exact event tuples. A recording has
299–3,553 such records (median 1,359.5). Keys include characters, space,
backspace, modifiers and named special keys; they require an explicit mapping.

Raw columns are key, hold, release timestamp, press timestamp. All retained
holds agree with release minus press to within 1 ms. We can derive Aalto's HL,
PL, IL and RL from these timestamps. The supplied loader is Python 2 and has
inconsistent column ordering in different helper paths: port the verified raw
schema rather than running it unchanged. Mapping unavailable keys and preserving
boundaries requires a new adapter; this audit is not that finished adapter.

Counting full, nonoverlapping 50-key windows within recordings, with breaks at
nonpositive press differences or gaps longer than 30 seconds:

- 13 controls have two separate recordings with at least 10 windows each.
- 11 controls have two recordings with at least 20 windows each.
- Two controls have two recordings with at least 50 windows each.
- No control has enough total windows for 100 baseline plus 100 query windows.

Use it for a small, compatible synthetic-perturbation transfer experiment and
for motor-related timing checks. It cannot supply a large longitudinal
100/200-window benchmark or four cognitive-severity classes.

## Tappy: substantial temporal coverage, incomplete event sequences

The local archives contain records for 266 identifiers. Of these, 162 have
metadata reporting Parkinson's, 55 report no Parkinson's, and 49 have no matching
Parkinson's value. There are 227 metadata files overall; not all correspond to
recordings. Self-reported non-PD status does not establish cognitive health.

Among 9,316,858 rows, 14,979 fail basic structural/timing validation. Removing
178,469 exact repeated records leaves 9,123,410 unique valid-format records.
These counts do not imply physiologically curated data.

Available signals: date/time, hand or space category, hold, press-to-press
latency, release-to-press flight interval, and hand-transition direction.
Exact keys are absent. We cannot reconstruct per-key baselines or reliably
recover correction-key behavior. It is not a direct input replacement for our
existing exact-key sequence models.

I split each person's observed dates into earlier/later halves, never counted
windows across dates, and used full nonoverlapping 50-record blocks:

- By record counts alone: 22 non-PD participants support 50 blocks per side,
  19 support 100, and 16 support 200.
- A conservative continuity diagnostic leaves one non-PD participant at
  50/100 windows per side and none at 200. Five and ten windows per side
  retain four and three respectively; twenty retains two.

The diagnostic requires previous hold = current latency minus flight within
1 ms, consistent hand transitions, timestamp alignment, and positive inferred
press gaps no longer than 30 seconds. Timestamp differences align better with
release-event than press-event semantics. I tested both recorded-time and
inferred-press order, and 5/20/50-ms tolerances; eligibility counts above were
unchanged. This is evidence against assuming a complete stream, not proof that
every rejected record is bad. The timestamp interpretation remains an empirical
hypothesis, and filtering can itself select atypical typing.

**Most defensible use:** a newly trained model comparing earlier versus later
periods using distributions of valid observed hold/latency/flight records,
stratified by hand transition and day. Explicitly model missingness/coverage and
exclude identity/date/diagnosis from severity predictors. Periods must be defined
as observed-record collections, not represented as 50 consecutive keypresses.
Use participant-level splits and uncertainty; thousands of rows do not create
thousands of independent people. Baseline-only false-alarm checks across real
days are especially useful before adding synthetic perturbations.

We may use patient recordings as unperturbed source behavior in a separately
specified synthetic-change experiment, but must not call them healthy baselines
or interpret perturbation classes as clinical disease grades.

## AD/MCI writing: summary data, not raw key events

`Participant_level.csv` has 59 participant-task rows for 30 people: 15 controls,
10 MCI and five Alzheimer's participants. `Linguistic_analysis_word_level.csv`
has 2,648 word-level rows, also covering 30 identifiers and 59 participant-task
combinations. The files are semicolon-separated and require a non-UTF-8 encoding
(the audit uses Latin-1).

Useful measurements include pause fraction, burst duration/frequency,
characters per minute, between-word intervals, word start/end times and median
character intervals. These can inform pause/burst hypotheses and a separate
small summary-feature analysis. They cannot reconstruct individual key holds,
releases, keycodes or a full timing sequence.

Quality caveats found locally: only 29 IDs match exactly across the two files;
the unmatched pair appears to differ by a character, but was not silently joined.
Seven shared control IDs disagree in age-band labels. These require source
reconciliation before merging metadata or using age-adjusted comparisons.

## Recommended local-only work

1. Build the neuroQWERTY adapter and check cross-dataset behavior at 10/20-window
   budgets on the small eligible control cohort.
2. Build a separate Tappy period-distribution representation, with explicit
   coverage and missing-event handling, to study longer-term baseline variation.
   Audit retention after final quality rules before registering sample sizes.
3. Use AD/MCI summaries to constrain pause/burst scenarios, not to manufacture
   raw event data or label Parkinson's recordings with cognitive severity.

The current perturbation generator was already informed by aggregate Tappy,
neuroQWERTY and AD/MCI statistics. None can be described as untouched external
confirmation of that generator. We need a documented exposure history and a
new registered evaluation for any added model. These sources support useful
experiments now, but do not provide the requested four-class clinical ground
truth or a large complete-event longitudinal cohort.
