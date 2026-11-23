# Paper B Gate 0C — diagnostic evidence (not manuscript production)

Date: 2026-11-23. Code branch: `paper-b-gate0c-credibility-aware`.
[CI run and reproducible artifacts](https://github.com/minsujang25/InfoSourceSamplingLearning/actions/runs/45264863819).

## Gates

PASS: Gate 0A, 0B, 0C regression tests, canonical N=100, T=26 matched diagnostics, seed-level design/initial-state consistency, and first-audit/pre-disruption separation. 96 total diagnostic runs: 32 each for primary lambda=1 first-audit; sensitivity lambda=1 pre-disruption; predeclared lambda=0 first-audit component check. Seeds 6901 and 6902 only. No inferential or manuscript effect claims.

Historical reproducibility: for the 24 overlapping null/fixed-biased/myopic first-audit conditions, Gate 0C results match Gate 0B results *exactly* for MSE, jammer reliance, W_jammer_precision_share and mean_belief.

## First strategic message mean, t=1

- Myopic `adaptive`: 53.0154 (both primary and component diagnostic).
- Aware lambda=0: 5.8325.
- Aware lambda=1: 5.3917.

**The majority of the initial attenuation is due to learning probability and message-inclusion weighting, not the explicit credibility penalty.** It would be misleading to attribute 53 -> 5.39 exclusively to the lambda term.

## T=26 excess damage D = MSE(sender)-MSE(null), two-seed means

Primary: source_posterior, tau_social=1, first_audit, centroid t=0.

| Network | Receiver | Fixed-biased | Myopic adaptive | Aware lambda=0 | Aware lambda=1 |
|---|---|---:|---:|---:|---:|
| Elite-only | Adaptive | 6.7734 | 42.1424 | 7.6927 | 7.2288 |
| Elite-only | Frozen | 7.7098 | 70.4944 | 9.3944 | 8.7945 |
| Extended | Adaptive | 6.5000 | 97.6150 | 7.3787 | 6.9217 |
| Extended | Frozen | 7.5831 | 173.2497 | 8.9650 | 8.3435 |

A significantly weaker initial attack is *not* evidence of a more effective strategic attacker. Aware damage is comparable to the fixed-biased sender and much lower than myopic at T=26. These numbers do not identify social amplification.

Peer-minus-elite damage for aware lambda=1 is -0.31 (adaptive receiver) and -0.45 (frozen receiver), whereas myopic gives +55.47 and +102.76. These are total effects of adding peer sources, including rank competition.

## Credibility and acquisition

At t=25 with adaptive receivers, mean Jammer reliance:
elite-only myopic .3425 vs aware .4415; extended myopic .1398 vs aware .2516. The aware sender remains more frequently selected but achieves less total MSE. In frozen receivers, rankings are held fixed so this acquisition-selection mechanism cannot explain variation across senders.

The initial ranking convention matters: frozen receiver aware lambda=1 D changes from 8.7945 to 5.1030 in elite-only and from 8.3435 to 4.5594 in extended, when first_audit is replaced by pre_disruption. Keep these as separate designs.

## Important limitations and follow-up

The aware objective uses current Jammer-specific estimated source displacement and ranking information not used by the myopic objective, so it combines a changed utility approximation with additional attacker information. It is not a solved dynamic game or evidence that the sender optimizes actual finite-horizon MSE. The benefit term uses the archived one-message update proxy, not the full joint multi-source Bayesian update. The sender holds its message choice for each five-period surveillance window and computes audit probability at the refresh period. An information-set ablation, event-level source acquisition / credibility audit diagnostics, and T=200/400 with more matched seeds are required before considering a stronger theoretical conclusion.

Preserve archived Exp IIIa production, separate IIIb diagnostic inclusion rule, and v5.2 Abstract/Theory unchanged. Do not choose a preferred lambda based on these two seeds.
