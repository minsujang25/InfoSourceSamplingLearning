# V6.4.1 passive analytical closure — 2026-11-25

Scientific freeze remains at b4118378eb6f5b0881bcf7636bd82ed40f9ca5d5 (22,120 scientific runs). No new model runs or scientific conditions; V6.3 remains the manuscript baseline. The reviewer-informed passive analysis is closed for the planned scope; manuscript revision comes next.

## Scope and provenance
- Primary analysis: 140,000 terminal records, 1,400 existing runs. RA/RB H adaptive/frozen extension: 80,000 terminal records, 800 existing runs. Both reuse Study 2 design 7e2e4f6724e3, seeds 31001–31100; no pooling with Study 1/pilot/W2.
- Primary analysis commit 4038128954d5883a5f08e7b02a08e39de40fa3c7; read-only RA/RB extractor 9585ba13cfb565566d1daedb510eba489dea5018; paired differences 19849af3eda7d47dea8fd665fe28a44689bee138; RA/RB analysis 35b8ea857d4dcf22a4ce3ef42aba8e88eca3afaa.
- RA/RB script blob ed2b4e60b0b7cf972c517e82cd3f4376c67c70bd matches the executed file. Transfer manifest and four extract hashes verified; 800 expected shard names present, but raw shard bytes not available for independent re-hashing here. 66 frozen point estimates and corresponding MC intervals match (max absolute residual 4.55e-13 across all three values). Reference-shift combined allocation max residual 3.42e-13.
- Use 100 matched seeds and whole-seed bootstrap (10,000 resamples, seed 20261125). All new analyses are retrospective reviewer-informed diagnostics on existing outputs.

## Focal RA/RB results
H assigned +1 group; high minus low. Numbers are mean contributions to MSE contrast; brackets are pointwise 95% Monte Carlo intervals.

| Condition | All-source content | Weight + entry/exit | Net MSE |
|---|---:|---:|---:|
| RA adaptive | 1.877417 [1.752388,2.004444] | -1.226963 [-1.331258,-1.120731] | 0.650454 |
| RA frozen | 1.002858 [0.914197,1.089408] | -0.488745 [-0.575300,-0.402640] | 0.514113 |
| RB adaptive | 3.233546 [2.982210,3.486656] | -0.049328 [-0.114659,0.017615] | 3.184218 |
| RB frozen | 0.143382 [-0.011533,0.307171] | 0.006279 [-0.059295,0.074930] | 0.149661 |

Adaptive-minus-frozen structural contrast decomposition: RA content +0.874559, combined weight -0.738218, net +0.136341; RB content +3.090164, combined weight -0.055606 (interval includes zero), net +3.034557. RA modifies initial audit, RB caps external assigned precision. These are separate rule variants, not a factorial or independent causal intervention on content/weight.

## Manuscript decisions
- P1: unchanged external scores can produce different ranks as competing alters change; connect to actual acquisition and heterogeneous accuracy, not novel ordinal arithmetic.
- P2: in focal baseline adaptive H/G, content contributions dominate opposite-signed combined weight/entry-exit allocations in implemented and all 100 tested alternative pairings; uniform-tail H reverses with larger favorable combined allocation. Use all-source totals first; source-class allocations depend on reference and pairing. Reference invariance of the combined allocation holds for common recentering with the truth-centered multiplier fixed, not arbitrary alternative coupling. Endpoint allocations and marginal contrasts are invariant to within-stratum re-pairing. No causal mediation or ultimate provenance.
- RA frozen retains adverse content and adverse net error, so reassessment is not necessary under that specification. RB adaptive has large adverse content, combined weight unresolved; RB frozen both aggregate terms unresolved. Do not generalize opposition to all conditions. Retain all negative/non-confirming results; RA/RB extension H only, in Supplement.
- Gateway role-specific policy levels; Q/W distinction; signed net vs gross contributions; descriptive promotion partition and eligibility/completion explanation. No identified spillover, pure rank effect or universal learner types.
- Preserve peer-degree/topology, random-frozen, partial saturation and fixed variance extensions for possible future work, no new simulations.
- A00–A08 completed within planned passive scope; A09 event-history chronology/drift reconstruction explicitly deferred to replication; A10 closure. No need for new transfers before theory/methods revision. Raw archive validation, executable anonymous package, reviewer access, and journal audit remain pending. No LaTeX manuscript changes yet.

The author-side closure package holds primary and RA/RB CSVs, seed components, scripts, manifests, checksums and updated checklist. These are not the final reviewer-facing replication release.
