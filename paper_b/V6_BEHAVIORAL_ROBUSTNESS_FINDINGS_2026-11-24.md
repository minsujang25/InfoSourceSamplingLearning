# V6 behavioral robustness findings — 2026-11-24

Exact uploaded archive paper_b_v6_robustness_7247f5ad0d7a_behavioral_shareable.zip, SHA256 271268dd0b29faa1ce0563d4110de8cc1b12d863437a8405f6b1fc838fce9e84. Design 7247f5ad0d7a, behavioral fingerprint 8751c25837c9b08198fcb0a721056cd109596061098431fc63091ff45fa9ea24. Gate PASS: 640 grid, 80 parity reported, matched t0 audit true, seeds 6001–6020, T400. Key effects below directly checked against archive contrast_summary.csv.gz; no independent ABM rerun or parity replay in this step.

| H aligned adaptive, high-S fixed, high-low | Baseline | Uniform-tail | Constant audit |
|---|---:|---:|---:|
| Delta Q_jammer | -909.275 | -666.387 | -361.521 |
| Delta W_jammer | -.078742 | -.066097 | -.078381 |
| Delta MSE | .757163 | -.368648 | -.769346 |
| MSE 95% MC bootstrap | [.332589,1.172579] | [-.909811,.152087] | [-1.421790,-.134214] |
| Q falls/MSE rises |15/20|6/20|4/20|

Uniform-tail minus baseline MSE contrast -1.125811 [-1.609107,-.668399]; constant-audit minus baseline -1.526509 [-2.366793,-.732190]. Direct displacement survives but accuracy effect does not. Constant audit has same expected total at T400, not the same realized counts or timing of substantive learning; uniform tail also changes null allocation. No causal mediation claim. H divergence is a conditional existence result, not universal.

| G population high-low damage | Adaptive | Frozen |
|---|---|---|
| Baseline | -.947018 [-1.249722,-.640312] | -1.186120 [-1.367428,-1.001385] |
| Uniform-tail | -1.284891 [-1.542896,-1.016605] | -1.355798 [-1.593298,-1.108841] |
| Constant audit | -1.055050 [-1.287236,-.821418] | -1.199983 [-1.378227,-1.018062] |

The G protection persists under all four single-factor sensitivities, and direct displacement is not sufficient to infer accuracy. Not a general proof for arbitrary network topologies. Behavioral sensitivities are main-text boundary conditions, not hidden negative results. Close general behavioral parameter exploration; retain all results and do not search new settings to rescue H. No low-S, combined or strategic sensitivity inferred. Next: bounded strategic scope diagnostic only, then production freeze.
