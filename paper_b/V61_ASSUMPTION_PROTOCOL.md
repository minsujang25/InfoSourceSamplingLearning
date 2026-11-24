# V6.1 assumption diagnostics — frozen prospective experiment protocol

Status: implementation candidate, not a passed scientific/technical gate.
Base commit: 53fc316326186db57a69873f2a32859aa7622afb.
Only new files on the diagnostics branch. No changes to the archived model,
production runners, archived outputs, or their fingerprints.

## Questions and fixed design

These are targeted, one-factor sensitivity experiments, not a new search over
parameter combinations or independent confirmatory data. The 20 seeds are the
existing matched sensitivity seeds; do not pool them with the 500-seed H/G
production or 100-seed original C. No effect-based stopping, extra seeds, or
parameter adjustment. Preserve all cells, outcomes and failed gates.

| Variant | Intervention | Grid | Seeds | New grid runs |
|---|---|---|---|---:|
| RA | Fixed-biased sender's period-0 emitted mean = 4, SD=1; native clustering, audit allocation, null and later sender unchanged | H high S: low/high H x null/fixed x adaptive/frozen; G low/high x null/fixed x adaptive/frozen | 6001–6020 | 320 |
| RB | Assigned external state precision per delivered source/period = 1 (unit source variance); average ALL delivered messages, no change to audits or peer likelihood | same H/G grid | 6001–6020 | 320 |
| RC | Constrain both strategic optima to [-5,5], K=1, refresh 5, original centroid audit unchanged | E/M/S/L x myopic (adaptive)/credibility_aware sender x adaptive/frozen receiver | 7501–7520 | 320 |

All other settings are inherited from the archived scientific design: N100,
T400, C20, epsilon=.05, tau_social=1, source_posterior, delta_comparison,
first_audit frozen mode, MIN_SD=1e-8, synchronous updates.
Preflight T26 uses the first seed, all 16 cells and one logger parity run per
suite; separate T400 archived replays and short initial-audit comparison runs
are technical, not scientific observations. The full grid also has one
logger-off representative per seed/suite. No new 500-seed production.

RA retains the original centroid-initialized null, so sender-minus-null damage
includes the early audit intervention. It is not a pure post-audit damage
estimand; primary RA estimands are absolute fixed-biased high-minus-low H/G
and changes relative to matched baseline; common-null damage is also reported.
RA null replays are checked against original terminal citizens.
A separately initialized null would define a different experiment, not part of
this grid.

RB is a receiver-assigned external precision sensitivity, not full peer/elite
evidence symmetry. The mean of repeated draws still has lower sampling
variance. Both sender and null are run with the same RB rule.

RC: both archived objectives have the form U(m)=U(m*)-d(m-m*)^2 with d>0.
Projection of m* onto the closed interval is therefore the exact constrained
optimum. Correct objective_value where recorded; retain unconstrained optimum,
bound flag and curvature. The bound applies to the chosen mean, not individual
Gaussian draws, not period-0 centroid audit. Nulls do not optimize and are
reused from the exact archived C; preflight checks all eight nulls at seed
7501 and two preselected active baseline replays, plus a patched-null replay.
E/M/S/L are distinct bundled interventions; no causal provenance/mediation.

## Prespecified reporting

Primary H: aligned high-S adaptive high-minus-low absolute MSE, absolute Q_J,
and Jammer-side/near-Jammer share. Report opposite group, population and frozen
conditions as well; do not infer a population effect from aligned group alone.
Primary G: population adaptive high-minus-low incremental damage and absolute
Q_J and MSE; report roles and frozen. Positive Q / negative MSE is G's inverse
divergence, not the H joint-sign direction.
Primary C: adaptive and frozen D_L-D_E and D_L-D_S for both strategic regimes,
bounded minus archived unbounded; retain all four conditions and all channel
contrasts. Report absolute MSE separately from D. First strategic message,
constraint activation and early belief shock are diagnostic outcomes.
Compare matched seed contrasts and their paired differences, not separate
interval overlaps. 10,000 whole-seed bootstrap draws, RNG 20261124, MCSE and
sign frequencies. Treat intervals as Monte Carlo uncertainty within model.
Capture threshold mu>2, near-Jammer abs(mu-4)<=.5, near-truth abs(mu)<=.5.
For RC also report abs(mu)>16 and mu>40; Jammer-side is not a capture measure
for extreme messages. The compact capture table reports run/population and
citizen-group summaries; role capture can be derived from raw shards.

Early passive observer records all source slots for periods 0–10, immediate
delivered sample means, second moments, requests, rank before update and
assigned precision, pending theta states. These are direct local evidence,
not upstream provenance. Aggregate early source summaries are retained locally.
Logger-off parity checks the same variant's belief/strategy/reliance records.
Full original baseline per-source message logs may be unavailable; do not
assert matched mediator effects without them.

Decision rule: all results reported regardless of sign. Persistence supports
robustness within the tested intervention; attenuation/reversal/null result
defines a scope condition. H sign reversal already exists under behavioral
variants. RC extreme-shock attenuation supports restricting interpretation of
unbounded C; if strategic comparisons vanish, narrow C to a diagnostic extension.
No additional topology family, parameter grid or model redesign is authorized
by this protocol.

## Local execution (macOS, from repository root)

Use the archived paper-b conda environment, including exact software versions.
The runner checks original archive SHA-256, manifest/gate and model fingerprint.

    conda activate paper-b
    python -m py_compile paper_b/experiments/v61_assumption_interventions.py paper_b/experiments/run_v61_assumption_robustness.py paper_b/test_v61_assumption_robustness.py
    python -m unittest paper_b.test_v61_assumption_robustness -v

Set paths to the exact unmodified archives:

    RESULTS="$HOME/Research/paperb_v6_results"
    HG="$RESULTS/v6_scientific_85faac503d4a/paper_b_v6_scientific_85faac503d4a_shareable.zip"
    C="/actual/path/paper_b_pilot_b7a571b4818a_shareable.zip"
    OUT="$RESULTS/v61_assumption_diagnostics"

Run RA first; share its preflight ZIP before proceeding if gate fails or
anything is unexpected:

    python -m paper_b.experiments.run_v61_assumption_robustness --experiment RA --phase preflight --baseline-zip "$HG" --output-dir "$OUT" --workers 4

After PASS, the same command with --phase robustness and --workers 4 (or 8 if
local RAM/CPU permit) runs the fixed 320-cell grid. RB uses the same HG archive
and --experiment RB. RC uses the original C archive and --experiment RC.
Use --dry-run to validate archive and print design without running the model.
Use --resume only in the same phase, design, software, and output directory;
validated complete shards are reused. Never delete old shards or reuse a
different archive. A missing original C ZIP must be recovered exactly; do not
silently substitute a re-zipped extracted directory.

A matching passing preflight is mandatory for robustness. PASS indicates
technical checks only, not scientific success. If local unit tests or gate
fail, send exact traceback/log and do not bypass checks. Runner and tests
were prepared for local verification; no claim of local execution here.

Output: v61_<variant>_<phase>_<design_id>/ under OUT, with raw gzipped JSON
shards, full group/checkpoint/early summary CSVs, capture table, paired
seed_contrasts and contrast_summary, manifest and gate. First share only
paper_b_v61_<variant>_<phase>_<design_id>_shareable.zip; this contains manifest,
gate, runs, capture and contrasts (if T400), not full raw early records.
Do not upload private manuscript, review, or result data to this public repo.
