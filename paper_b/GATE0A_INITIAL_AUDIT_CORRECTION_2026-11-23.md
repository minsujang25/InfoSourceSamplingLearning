# Paper B Gate 0A — Initial Audit Correction

Date: 2026-11-23
Branch: `paper-b-gate0a-initial-audit`
Base: `paper-b-theory-closure` (f61d5e6)

## Scientific decision

**Period 0 is a non-disruptive credibility audit, not an attack.**
For each requester `i`, the Jammer supplies independent audit messages
from `Normal(centroid[initial_segment(i)], 1)`. The audience segmentation
uses the current period-0 (initial) citizen beliefs and the specified `K`.
This reproduces the original `init_message_param()` rule on the repository
`main` / legacy pipeline, which recorded
`initial_msg_param[cluster] = {"avg": centroids[cluster], "std": 1}`.

The centroid is **belief-consistent but not necessarily true**: this audit
removes *strategic disruption during initialization*, not all possible
distance from the state of the world.

All citizens undergo credibility (delta) audits in period 0; substantive
theta beliefs and uncertainty must remain unchanged. The same non-disruptive
audit message rule applies to the `adaptive`, `fixed_biased`, `truth_clone`,
and `null` Jammer regimes, to hold initial evaluation comparable across
sender counterfactuals. In the `null` regime, the source remains inactive
and returns no messages **after** the initial audit.

In `initial_audit_mode="centroid"` (the new default):

- Period **0**: segment centroids are computed; audit messages have mean
  equal to requester's initial segment centroid; no strategic optimization.
- Period **1**: first genuinely adaptive Jammer optimization, using the
  pre-update receiver states; the first attack messages are issued in
  this period (in substantive learning or audits, as randomly chosen).
- Subsequent strategic surveillance: periods 1, 6, 11, 16, ... for
  `surveillance_interval=5`. Messages remain unchanged between refreshes.
- `fixed_biased` returns to its fixed biased messages from period 1,
  `truth_clone` returns to truthful messages, and `null` returns to
  no substantive messages.

## Historical reproducibility

The `paper-b-theory-closure` branch and its frozen outputs have not been
altered. The uncorrected behavior can still be reproduced explicitly on
the new branch using `initial_audit_mode="legacy"`: previous strategic
Jammer optimization at period 0, previous fixed-biased initialization, and
period-0,5,10,... surveillance cadence.

The new default **changes the causal experiment** (including initial
credibility assessments and potentially all downstream learning and MSE).
Do **not** attach old 500-seed results to runs generated with corrected
initialization, and do **not** overwrite archived production outputs.
All affected sender experiments must be rerun under a new design ID,
with seed-level comparisons and a disclosed code fingerprint.

## Frozen receiver comparability

The existing model supports two distinct frozen ranking conventions:

1. `frozen_ranking_mode="first_audit"`: freeze the credibility rank that
   arises after the common non-disruptive period-0 audit. This is preferred
   for a prospective matched comparison of adaptive vs. frozen receiver
   **changes after initial evaluation**.
2. `frozen_ranking_mode="pre_disruption"`: freeze a deterministic ranking
   from initial source states *before* the audit. This was explicitly used
   in historical canonical production and is **not identical** to
   `first_audit`.

This commit does not silently change this ranking option. Any new Gate 1
experiment must freeze and report the chosen convention before inspecting
results. Sensitivity to the other convention can be assessed separately.

## Code and tests

Modified model: `model/InfoSourceSamplingLearning.py`

- Adds `initial_audit_mode` (`centroid` default; `legacy` alternative).
- Separates `initial_audit` and `adaptive_disruption` strategy phases
  and records strategy metadata in `jammer_strategy_history`.
- Makes `DisruptiveJammer.mu_out()` use the requester's initial cluster
  centroid during period 0, irrespective of Jammer regime.
- Offsets strategic surveillance refresh from period 0 to period 1.
- Does not change the citizen substantive/acquisition/posterior equations.

The matched pilot now records `initial_audit_mode` per run; its validator
recognizes both the corrected schedule and archived legacy schedule.

Regression tests:

```bash
python -m unittest paper_b.test_gate0a_initial_audit -v
```

Tests cover: period-0 K=1/2/4 centroid means and Gaussian messages;
all sender regimes' non-disruptive t0 credibility audits;
t1 and t6 strategic refreshes; identical after-first-audit ranking
with matched seeds for adaptive/frozen receivers; legacy mode; invalid
configuration rejection.

## Gate status

**Code correction committed. Behavioral Gate 0A PASS is conditional on
executing the regression suite in an installed Mesa 3.5.1 environment.**
No new full production run was performed with this change.

Next after regression PASS: run a matched small-seed audit through at
least period 25, validate that period 0 produces no theta update or
strategic displacement, compare `initial_audit_mode` values on identical
initial states and network maps, then pre-register the revised Gate 1
adversary × peer experimental design. Do not use the two-seed diagnostic
as a manuscript result.
