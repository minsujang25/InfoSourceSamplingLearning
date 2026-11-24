# V6 native-K strategic scope findings — 2026-11-24

Exact source: paper_b_v6_strategic_strategic_d6de4811876c_shareable.zip; SHA256 5a3866a9ac35c3bdf6314d599a2264590a0ff05f87d2badff1a99480cab6ee0b. Design d6de4811876c, upstream fingerprint 754d7c373ff4b9f6eb24e1b0044917d559f69397531a053d5ab64d314cd67774, scientific runner 3e217225e6552887d611478c7b3e44457628aea152bbf275ffcbed8d28202e89, strategic runner 2a518e539c127468f39d2c8107a1be50a6bfc691ea93694fedd8dcb9baa7973e. Gate PASS, 480 grid, 40 reported parity, 80 archived K1 null matches, 80 new K4 nulls, same 20 seeds 7501–7520, T400. Native K4 at t0 AND t>=1. No independent model replay in this record.

Initial active-sender top-ranked share K1 -> K4: E 51.90% -> 86.55%; L 22.85% -> 68.85%. Paired changes +34.65/+46.00 percentage points, respectively. Use one active-sender/receiver representative per seed and environment, not duplicated audit records as independent cases. Null ranks forced last. K1 t0 from refresh1 (same audit by design and preflight); initial centroid is belief-congruent, not necessarily accurate.

B = D_L - D_E, D = sender MSE minus corresponding null MSE. Matched 20-seed archive subset, not full archived 100-seed average:

| Sender / receiver | K1 interval5 | K4 interval5 | K1 interval1 | K4 minus baseline [95% MC interval] | Interval1 minus baseline [95% MC interval] |
|---|---:|---:|---:|---|---|
| Myopic / adaptive | 5.485933 | 41.904775 | 2.556106 | 36.418842 [28.088415,45.788920] | -2.929827 [-4.319831,-1.570481] |
| Myopic / frozen | 5.852631 | 3.562663 | .480575 | -2.289967 [-3.492012,-1.151716] | -5.372056 [-6.303614,-4.458028] |
| Aware / adaptive | .670503 | 2.521852 | .515213 | 1.851349 [1.340771,2.345545] | -.155290 [-.279286,-.050824] |
| Aware / frozen | -.978017 | .601880 | -.987608 | 1.579897 [1.102038,2.039514] | -.009591 [-.016539,-.003245] |

K4 raises absolute and incremental damage in all eight sender/receiver/environment mean comparisons. Myopic adaptive L damage 11.024 -> 50.479; all 20 K4 L-E damage contrasts positive. Report finite horizon, not permanent error. K4 raises initial credibility and changes subsequent message generation; not isolated mediation. More frequent surveillance/optimization reduces myopic damage under this particular short-horizon objective; not a general result about attacker sophistication. Myopic and aware differ in information set and objective. K4 and interval1 are separate single-factor interventions, no combined or monotonic K claim. E/L is total environment contrast; L-S mechanism evidence belongs to C, not this grid. H/G fixed-biased only. No causal provenance or independent-path claim. Do not search further parameters based on these results.

Decision: close new mechanism and behavioral exploration. Include core strategic scope results in C extension subsection, detailed traces in supplement. Preserve all conditions, including reduced B for myopic frozen and interval1. Existing 100-seed C results remain separate from 20-seed matched scope diagnostic. Proceed to independently seeded H/G final production only; no changes to common model. Exact summary B means/intervals verified against the uploaded archive in this step; initial audit and other findings carry forward the previous raw-data audit.
