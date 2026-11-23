# Gate 0D — Staged sender ablations, actual exposure and horizon check

Date: 2026-11-23; branch `paper-b-gate0d-mechanism-identification`.
Confirmed GitHub Actions run:
https://github.com/minsujang25/InfoSourceSamplingLearning/actions/runs/45267426764

**Status:** implementation and measurement gate **PASS**, scientific
mechanism interpretation **provisional**. This is an additional
diagnostic, not manuscript production or a replacement for Exp IIIa.

## Reproducible design

- N=100, K=1, C=20, epsilon=.05, source_posterior, tau_social=1,
  initial centroid audit, first_audit frozen ranking and t=1,6,11,...
  surveillance cadence.
- Matched networks `elite_only` vs `extended` (2 peers added while
  Expert/Jammer access stays fixed).
- Senders: null, fixed_biased, adaptive (archived myopic), timing_only,
  exposure_uniform, exposure_ranked, credibility_aware (lambda=1).
  All receivers adaptive/frozen.
- T=26, seeds 6901–6902: 56 complete runs, 145,600 citizen-period
  message/audit events captured. T=200, seeds 6901–6904:
  112 complete runs, capture disabled. 168 total runs.
- Within-block source-map and initial-state fingerprints, finite
  posterior tests, fixed horizon and all sender cells PASS.
- First 26 periods match *exactly* for overlapping 48 non-null
  matched sender contrasts in T26 vs T200 outputs: max absolute MSE
  and null-referenced D difference both zero.

## First strategic message, period t=1

| Regime | T26 mean m | T200 mean m |
|---|---:|---:|
| Myopic adaptive | 53.015 | 52.996 |
| timing_only | 7.439 | 7.437 |
| exposure_uniform | 7.428 | 7.427 |
| exposure_ranked | 5.833 | 5.859 |
| credibility_aware lambda=1 | 5.392 | 5.411 |

The vast majority of 53 -> ~5.4 attenuation appears from timing
discount, which removes the assumption that every citizen receives
a substantively processed attack at t=1. The next largest change is
rank-specific acquisition probability; uniform exposure contributes
almost nothing because with C=20, the probability of >=1 message
under equal sampling in degree-2/degree-4 networks is near one.
The explicit lambda=1 credibility penalty has a further, small effect.
This is a **nested modeling ablation**, not a causal information-value
experiment, and these are two/four-seed diagnostic means.

## Additional MSE versus structural null: D = MSE(sender)-MSE(null)

T=26 (n=2 matched seeds per condition):

| Network | Receiver | fixed | myopic | timing | uniform | ranked | aware |
|---|---|---:|---:|---:|---:|---:|---:|
| elite_only | adaptive | 6.77 | 42.14 | 8.07 | 8.07 | 7.69 | 7.23 |
| elite_only | frozen | 7.71 | 70.49 | 10.06 | 10.06 | 9.39 | 8.79 |
| extended | adaptive | 6.50 | 97.62 | 9.33 | 9.31 | 7.38 | 6.92 |
| extended | frozen | 7.58 | 173.25 | 12.22 | 12.19 | 8.97 | 8.34 |

T=200 (n=4 matched seeds per condition):

| Network | Receiver | fixed | myopic | timing | uniform | ranked | aware |
|---|---|---:|---:|---:|---:|---:|---:|
| elite_only | adaptive | 6.52 | 6.98 | 6.01 | 6.01 | 6.18 | 6.17 |
| elite_only | frozen | 7.72 | 11.72 | 7.98 | 7.98 | 7.91 | 7.82 |
| extended | adaptive | 6.74 | 21.15 | 7.81 | 7.80 | 7.03 | 6.92 |
| extended | frozen | 7.42 | 26.26 | 8.34 | 8.34 | 7.73 | 7.58 |

**At T200, ranked exposure is not uniformly associated with lower
MSE:** elite-only/adaptive receiver ranked D=6.175 > uniform D=6.009.
Do not report a universal monotone ordering among nonmyopic senders.
The myopic excess is much smaller at T200 than at T26, but still
larger under `extended` than `elite_only` in these four seeds.

Under extended/adaptive receiver, myopic D is ~458.7 after the
first attacked period (horizon step 2), ~99.2 at step 26, and
~21.15 at step 200. Aware D is ~3.78, ~7.13, and ~6.92 respectively
in the four-seed T200 diagnostic.

## Event-level integrity and mechanism observations

- Entire 145,600 event rows were validated for unique
  condition/seed/network/citizen/period keys and
  `jammer_delivered_count <= jammer_request_count`.
- Every citizen's t=0 learning target is audit and theta belief
  is unchanged (max numerical difference 0).
- The `null` sender supplied 6,000 messages during the shared
  t=0 centroid audit across 8 condition/network/seed blocks.
  For t>=1 it was **requested 25,658 times**, yet delivered **zero**
  substantive messages. This is why requested versus delivered
  must not be conflated.
- Actual audit share among all citizen conditions at t=1 is 0.4925,
  matching the intended per-citizen 0.5 probability.
- Under extended/adaptive receivers, t=1 Jammer audit exposure
  yields a mean increase in absolute perceived source displacement
  of ~16.39 in myopic, compared with ~0.80 in credibility_aware.
  This is associated with a sharp early loss of behavioral rank and
  lower subsequent reliance for myopic (but not identified mediation).
- Example t=2 mean pre-acquisition Jammer probability in this
  environment: myopic ~.127 versus credibility_aware ~.260.
  At t=25: myopic ~.140 versus aware ~.252.
- Actual Jammer message counts and the Gaussian means for both
  sender modes were recorded *before* citizen
  `commit_pending_update()` cleared sampled messages. Optional
  recording passes off/on outcome and strategy parity tests.

## Interpretation / restrictions

1. `adaptive` vs `timing_only` is not an optimal-control comparison.
   It is a change in how the sender weights a hypothetical single
   message by the receiver learning schedule. The attacker still
   holds a strategy for five periods, while the audit probability
   declines within that window; the sender discounts using p at the
   refresh period only.
2. `exposure_uniform` vs `exposure_ranked` adjusts use of sender
   knowledge about current rankings under otherwise identical
   objective terms. Rank-weighted source selection also changes
   later endogenous state trajectories, so outcomes are total
   adaptive paths rather than a purely controlled short-run effect.
3. `credibility_aware` penalizes squared expected source-displacement
   beliefs, not exact expected future rank loss or a Bellman solution.
4. Addition of peers changes degree, competition, effective allocation
   and indirect evidence simultaneously. Elevated myopic D under peers
   does **not** by itself identify propagation through peers.
5. Two-seed T26 and four-seed T200 results cannot establish stable
   parameter effects, uncertainty bounds, long-horizon resilience or
   a main-text finding.
6. Keep v5.2 manuscript backbone, frozen IIIa output, separate IIIb
   inclusion gate and historical myopic objective intact.

## Next gate

Prespecify a larger matched-seed T400 sender × network confirmation
(minimum myopic, timing_only, exposure_ranked, credibility_aware,
fixed, null) with seed-block confidence intervals. Also isolate
source-competition vs peer transmission with a network/cost
ablation and constrain any claim to actual observables.
