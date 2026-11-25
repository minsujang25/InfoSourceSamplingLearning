# V6.3 joint H/G passive closure

Executed against design 7e2e4f6724e3, seeds 31001–31100, frozen simulation commit 55264824e35c02284174e206d86955269d2b0bbc. Extractor checkout 4e539a1a874ab30ba9f99a967d8cabf8fe5ab94a; transfer manifest SHA-256 4504342a9efaf51313f7d771dc456e89c2f71b8369f37beec818bf9e45d449ac. All 9 data files match their recorded checksums and sizes. No new simulations. 140,000 terminal records (100,000 new and 40,000 previously transferred baseline H) from 1,400 runs, 226,000 first-audit rows, 113,000 matched external-source pairs across repeated policies. Maximum midpoint MSE residual 3.33e-13. New terminal-derived run means match stored run summaries to 7.11e-15. Source shard hashes were checked locally by the extractor; raw shards were not received for this independent analysis.

## Executed scripts

- analyze_v63_passive.py: commit 4f3c58633a9b50932a5c6f486d270891d64414c8, blob c3bb833f9ad891db0161b032f346ec2d838b8ac2.
- verify_v63_transfer.py: commit 72affddcfc94932057f864eb9732b5975dfb5bd9, blob fffc539752036dc8c502b7ec7e14674ec23c6370.

Both GitHub blob hashes match the actually executed local files. Use NumPy and pandas with the 10 transfer files and the earlier baseline_H_individuals.json.gz, validation_all_runs.csv.gz, v62_7e2e4f6724e3_primary_001_of_008.zip, V63_FINAL_REVISION_PROTOCOL.md staged in ROOT; OUT is a fresh output directory. From repository root:

```bash
OPENBLAS_NUM_THREADS=1 python -m paper_b.experiments.verify_v63_transfer --root "$ROOT" --out "$OUT"
OPENBLAS_NUM_THREADS=1 python -m paper_b.experiments.analyze_v63_passive --root "$ROOT" --out "$OUT" --extra "$ROOT"/*_terminal.json.gz
```

## Key findings (rounded)

Symmetric terminal MSE *change* decomposition, not earlier signed level allocation or causal attribution:

| Focal contrast | Peer content | Peer weight | Jammer total | Net MSE |
|---|---:|---:|---:|---:|
| H aligned adaptive | 1.464159 | 2.905387 | -3.363095 | 1.006565 |
| H aligned uniform tail | .567758 | .887623 | -1.941405 | -.487 |
| G population adaptive | -1.423670 | -1.422223 | 1.708333 | -1.137603 |
| G population neutral | -1.192808 | .001004 | -.001517 | -1.193321 |

Full accounting includes prior, Expert and entry/exit terms. H uniform-tail reversal involves a smaller adverse peer change than the offset from Jammer weight. G neutral improvement is predominantly peer content in this arithmetic; peer posterior-normalized weight remains about .2236. Uniform allocation is a total policy contrast, not isolated credibility selection.

Joint H/G re-pairing uses canonical seed/suite/group-or-role/block order, 100 permutations, PCG64 seed 20261125. This supersedes the provisional H-only permutation stream. Focal category original / median permutation: H prevalence .3168/.2829, Q net share .84848/.87209, MSE net share .24902/.29554; G .2664/.2657, 1.07572/1.19985, .17135/.26593. Conventional categories' MSE share median .77315 H and .79928 G. All seed marginals preserved; qualitative split persists for this stress test only, not coupling-invariant individual effects. Mean initial belief mismatch about .155 H and .305 G; tail pairs can differ substantially. Exact-zero adaptive H/G categories arise from unchanged Q rather than unchanged error.

All external first-audit request/delivery counts, message means/variances, displacement means/SDs match between low/high; rank equality H 10222/20000 and G 7313/11000 per policy. This supports relational rank changes at identical initial external scores.

G measured W HHI structural contrast adaptive .073039, neutral .060416; difference .012622 [ .011151,.014088 ]. Top-five contrast adaptive .335664, neutral .250314; difference .085350 [.076605,.094013]. Same 100 seeds, actual W not U^A; total policy interaction, not pure behavioral centralization. MC intervals from 10,000 seed bootstraps; permutation percentiles descriptive only.

## Decision

Passive phase complete. Preserve full-precision outputs and hashes outside Git. W2 is the only remaining conditional scientific sensitivity, subject to the frozen protocol and inventory for an existing comparable run. No additional experiments chosen on the basis of signs. W2 has not been implemented or executed in this step. Manuscript and anonymous replication release remain pending. Prior Phase 1 auxiliary cohort comparison and peer initial-rank outputs remain in its separate archive.
