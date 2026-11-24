# Fixed-biased channel bridge findings — 2026-11-24

Exact uploaded ZIP paper_b_v6_channel_bridge_4a9dbeb69ff4_shareable.zip, SHA256 590f7a1ec26f8d7d8f85a258187b551b9d5dda45cea9cc0dc2e9c1e7e8e2f80d. Manifest design hash recomputed, 4a9dbeb69ff4. Gate PASS: 1600 grid runs, 200 parity, 800 archived null matches, 100 seeds 7501–7600, T400. These are reported gate checks; no independent model replay in this step. 7800 channel seed contrasts; T400 fixed-biased means independently recalculated from the seed CSV; summary intervals read from archive.

| Sender / receiver | D L-E | D L-S |
|---|---:|---:|
| Fixed / adaptive | .150362 [-.028939,.328037] | .098597 [.004063,.191446] |
| Fixed / frozen | -.976072 [-1.186670,-.765746] | .026077 [-.085676,.137136] |
| Myopic / adaptive (archive) | 7.069724 | 4.686807 |
| Myopic / frozen (archive) | 5.716763 | 6.671848 |
| Aware / adaptive (archive) | .595318 | .423804 |
| Aware / frozen (archive) | -.956451 | .072688 |

Fixed L-S absolute sender MSE: -.025976 adaptive, -.294086 frozen (frozen interval [-.403711,-.184646]); null MSE -.124573, -.320163. Distinguish absolute loss from incremental damage. Myopic increases are sender-regime-dependent, not generic peer feedback. Objective and information set differ; no attribution to credibility penalty alone. L-S changes outgoing peer mean AND SD; frozen ranking does not freeze peer signal learning. E has fewer opportunities, M wastes credits, comparisons telescope arithmetically, not causal mediation.

H/G/C now connected by fixed-biased sender and common behavioral specification, not identical prior or topology. Do not replace historical Gate0E2. Preserve all results, including small and opposite contrasts. Next frozen task: two separate receiver-rule sensitivities on H high-S/G using original 20 seeds, then strategic-sender scope decision and final production freeze. No full production authorized.
