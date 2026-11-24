# V6 robustness findings — 2026-11-24

Source: paper_b_v6_robustness_be334c1a5305_robustness_shareable.zip, SHA256 3abbfd40ea0b2f40d8706f2d4a14bcee6b3cce003f9cd29110a6bf4c84ac1df6. Design be334c1a5305, baseline 85faac503d4a, upstream fingerprint 754d7c373ff4b9f6eb24e1b0044917d559f69397531a053d5ab64d314cd67774, robustness fingerprint 1f1ac0e46587044efd87c39a6c2e2b641dd4fc18f51d298d34508f8df12cec88. Gate PASS, 640 grid, 80 parity reported, 20 seeds 6001–6020, N100 T400. 8000 paired seed rows. Archive and model unchanged. Preflight records are separate, not included in this ZIP; parity is based on gate.

## Primary H (high-S, +1 group, fixed-biased adaptive, high-minus-low)
| Variant | Delta X requests | Delta Q_jammer | Delta W_jammer | Delta MSE (95% paired MC interval) | joint Q decrease/MSE increase |
|---|---:|---:|---:|---|---:|
| Baseline | -45463.75 | -909.275 | -.078742 | .757163 [.332589,1.172579] |15/20|
| Cap5 | -50631.20 | -257.603 | -.094332 |1.070358 [.680060,1.466603]|17/20|
| Epsilon .20 | -32841.80 | -656.836 | -.045066 |.535732 [.322677,.737521]|17/20|

Cap vs baseline delta MSE effect +.313195 [.127166,.478665]; epsilon vs baseline -.221430 [-.691175,.292672]. Frozen aligned group's high-H MSE effect is negative across all three specifications, and opposite group's adaptive effect is negative. Population adaptive effect is not distinguished from zero. These are subgroup conditional patterns, not population deterioration. Increased squared mean bias, decreased within-group variance. X, Q, W describe direct incoming requests and evidence, not causal effects or provenance.

## Primary G (population, high-low damage)
| Variant | Adaptive delta D (95% interval) | Frozen delta D (95% interval) |
|---|---|---|
| Baseline | -.947018 [-1.249722,-.640312] | -1.186120 [-1.367428,-1.001385] |
| Cap5 | -1.222549 [-1.501038,-.932994] | -1.394459 [-1.559020,-1.229197] |
| Epsilon .20 | -.894384 [-1.045298,-.752554] | -1.182325 [-1.295450,-1.076036] |

Higher gateway composition improves both null and fixed MSE; adaptive fixed-biased population X, Q and W_jammer rise despite reduced MSE. This complements H in the opposite direction, at a different aggregation level and with a different intervention; not proof of a mathematical nonmonotone dose-response. Do not interpret as pure independent path effect.

## Decision
The central qualitative results survive these two individual changes. Do not extend to correlated message generation, intertemporal correlations, low-S robustness, joint variants, alternative acquisition function, audit timing, or general topology. Cap5 is a receiver-side within-period discount. All existing data preserved; no additional search over parameters. Next frozen task: fixed-biased C bridge; full matched null regeneration with updated measurement and exact archive checks, not a replacement of original Gate 0E2 or Gate1. Scientific sample still pilot scale; final production size not frozen.
Key numeric contrasts rechecked against uploaded contrast_summary.csv.gz in this step; no ABM run or independent parity execution.
