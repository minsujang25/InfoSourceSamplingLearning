# V6 staged experiment matrix — 2026-11-24

Status: technical pilot frozen; scientific-pilot candidate specified; final production NOT frozen or approved. New branch from v5.3 documentation HEAD; preserved historical IIIa/IIIb and Gate 0E1/2, original Gate 1. Three suites use V6_MODEL_SPECIFICATION.md, V6_MEASUREMENT_PROTOCOL.md.

| Suite | Cells per seed | Grid |
|---|---:|---|
| H (homophily/segregation) |16|2 H x 2 S x null/fixed_biased x adaptive/frozen|
| G (gateway composition) |8|2 gateway levels x null/fixed_biased x adaptive/frozen|
| C (peer feedback) |32|E/M/S/L x null/fixed_biased/adaptive/credibility_aware x adaptive/frozen|
| Total |56|N100, C20, K1, epsilon .05, surveillance5, source_posterior tau1, centroid / first_audit|

Technical: seeds 8001,8002, T26, all suites, 112 grid runs plus 6 representative uninstrumented parity runs. These seeds and horizon are diagnostic only. Record both t0 and t1 individual events. Technical PASS is mandatory before scientific pilot.
Scientific-pilot candidate: seeds 6001–6020 (legacy overlap intentional, paired version comparison; not independent confirmation), T400, H/G only, 480 runs. Keep results separate from archive. Same H factorial now includes fixed-biased under low-S as well as high-S (not in old primary design). No selected old myopic structural arms; C already tests strategic senders. Scientific pilot activation requires reviewed technical PASS, including measurement coverage and runtime. Do not choose new parameters by outcome direction.
Fixed-biased C bridge: proposed existing seeds 7501–7600, T400, four channels x null/fixed_biased x two receivers =1600 runs if null reproduced; 800 treatment-only only if matched old null manifest/state/graph/scientific code and software identical, verified before pooling. Not enabled in new runner yet. Existing 2400 Gate 0E2 may be reused if common spec and parity validated; do not mix differently instrumented runs without parity evidence.
Final structural production candidate 500 matched seeds; 24 cells/seed =12000 H/G runs, distinct from frozen original Gate 1 design. Size not approved: budget, pilot MCSE and exact scope to be decided before execution, irrespective of outcome direction. Consider independent seed sample vs archived matched seeds explicitly. Do not use phase scientific as final production.

## Estimands and decisions
H: HxS interaction on null MSE, high-S fixed-biased H contrast in direct jammer W/Q/I and MSE by group/reliance, divergence sign proportion and group gaps, rank/accessibility at t1 and later. Primary comparison aligned group +1 adaptive, report all. G: D_high-D_low and both absolute MSE contrasts, matched role composition; A/Lambda/W gateway peer-conditional shares. C: L-E and L-S for each sender/receiver, D and absolute MSE, M-E/S-M descriptive contrast accounting, no causal mediation. Report T26,200,400 when present; no asymptotic claims.

Targeted robustness after technical pilot: first external per-source within-period precision cap (e.g. cap effective n at 5, receiver-side discount not correlated-generation model), epsilon=.20 (same recursive rule). Proposed additional rank rule: first .95, remaining .05 split uniformly; alternative audit schedule: t0 mandatory, later probability (H_400-1)/399 at t1..399, preserves expected count but not temporal allocation. All are prospective candidates and require code changes, tests and separate fingerprint, not executable in current runner. Evaluate H high-S and G, fixed/null, adaptive, 20–50 matched seeds candidate; later decision on frozen coverage and precision. Freeze levels, sample size, and estimands before inspecting corresponding results. No sign-based expansion.

Strategic continuity: review archived segmentation/surveillance runs, if unsuitable consider small predeclared K=1 vs 2/4 and interval 5 vs alternate, E/L (not full factorial), with comparable audit and senders. No production command until scope fixed. Gateway input blocking conditional on stronger path claim; provenance, alternative topologies and recovery reserve. Keep original IIIb inclusion rule. 

## Local commands
From repository root, clean checkout and environment paper-b. Run regression tests (Gate 0A–0E) and v6 dry-run, then technical phase only. See V6_LOCAL_RUN_GUIDE_KO.md. Do not overwrite old results. Pilot output under a design-ID subfolder. Share ZIP and stdout gate; if failure do not start scientific pilot. No production was run remotely.
