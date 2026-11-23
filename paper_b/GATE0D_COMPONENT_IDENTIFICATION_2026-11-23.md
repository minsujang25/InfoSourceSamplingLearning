# Gate 0D — Prospective mechanism-decomposition pilot (frozen before outputs)

Date: 2026-11-23. Base: Gate 0C branch. Preserve all legacy frozen experiments,
Gate 0A/B/C inference restrictions and Paper B v5.2 manuscript.

## Scientific question

Gate 0C found a large drop in t=1 sender mean 53→5.4, but lambda=0
also produces ~5.8. This means the gain cannot be credited mainly
to the credibility-loss penalty. This study disentangles **timing**,
**average source access**, **rank-specific access**, and **explicit
credibility penalty**, without selecting a favored outcome after seeing data.

## Frozen staged sender regimes (all have the same t=0 centroid audit)

Model regime | one-message loss term | theta-learning weight | Jammer inclusion | perceived-delta audit cost
- `adaptive` (archived myopic) | yes | 1 | 1 for every observed citizen | 0
- `timing_only` | yes | 1-p_t | 1 | 0
- `exposure_uniform` | yes | 1-p_t | 1-(1-1/d_i)^C (rank-neutral) | 0
- `exposure_ranked` | yes | 1-p_t | 1-(1-pi_{Di})^C (actual rank policy) | 0
- `credibility_aware` | yes | 1-p_t | actual policy | lambda p_t E[a_i delta_hat_i(m)^2], lambda=1
- `fixed_biased` | constant source location mu_D | N/A | N/A | N/A
- `null` | no messages after t=0 | N/A | N/A | N/A

These regimes are **nested changes in objective weights and/or ranking
information**, not a randomized estimate of the value of information.
The difference `uniform→ranked` isolates rank-specific acquisition
weights conditional on source availability and a common timing discount.
The difference `ranked→aware` isolates the cost term at lambda=1 with
the same observed pre-update credibility rankings and beliefs.
The difference `myopic→timing_only` is timing discount, and
`timing_only→uniform` is rank-neutral source-exposure discount.
All use the same one-message posterior response-gain approximation and
fixed quadratic deviation cost. The model uses Gaussian messages,
not bounded or clipped messages.

## Diagnostic design

- N=100, T=26, seeds 6901 and 6902, initial regime flat.
- elite_only vs extended; same direct Expert/Jammer access.
- adaptive receiver and first_audit-frozen receiver.
- `source_posterior`, `tau_social=1`, `K=1`, `epsilon=.05`,
  `credit=20`, `surveillance_interval=5`, `initial_audit_mode=centroid`.
- 7 sender regimes × 2 receivers × 2 networks × 2 seeds = **56 runs**.
- Separate sensitivity to pre_disruption if primary diagnostics pass;
  T=200/400 and many seeds only after model checks.

## Outcomes and interpretation

Primary: paired terminal MSE, null-relative D, sender t=1 and refresh
message means, and per-stage changes (timing, neutral exposure, rank
exposure, penalty). Report total MSE and null MSE alongside D.

Mechanism trace at every t≤25, each citizen: whether audit vs theta
learned, selected Jammer request count versus *actual delivered message
count*, pre-update Jammer rank and substantive sampling probability,
pre and post source displacement/credibility estimates, pre and post
state belief, source-count, and evidence gain. Audit and theta learning
must be distinguished, and the event recorder must pass
trace-on/off model outcome parity checks.

Not direct mediation identification: peer addition changes feasible
source counts, allocation competition and possibly source rerouting.
Same seed can diverge in Gaussian message draws after altered actions.
Historical IIIa and IIIb output untouched; this is mechanism
diagnostic rather than a third confirmatory result.

## Gate and stop rules

1. Verify old myopic/fixed/null baseline MSE and sender history
   unchanged when optional event instrumentation is disabled.
2. Event on/off deterministic final-state and strategy parity.
3. Initial centroid audits remain exactly as Gate 0A and do not change theta.
4. Correct 56-run grid, fingerprints, no missing sender regimes.
5. Estimand differences computed within the same seed/network/receiver,
   with trace summaries reconciled to actual allocated requests.
6. Regardless of sign/noise, document all five staged sender outcomes,
   not only the most dramatic contrast.
7. Do not rewrite Abstract/Theory or overwrite the frozen production.
