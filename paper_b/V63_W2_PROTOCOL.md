# W2 — peer evidence weighting (V6.3)

The final revision permits exactly one remaining sensitivity if no scientifically comparable existing results are available. This is not a new model or an attempt to obtain favorable signs.

## Fixed design

Reference design 7e2e4f6724e3; source commit 55264824e35c02284174e206d86955269d2b0bbc; reference MANIFEST.json SHA-256 3a83e3b7ff4af3ceeeaeda34bbd47633c9df07228089e25f6d57a56394cfff61. Match seeds 31001–31100, N=100, T=400, budget 20, active fixed-biased sender, H high segregation and G flat, low/high structure, baseline adaptive and neutral acquisition. Eight cells per seed, 800 new scientific runs. Existing tau=1 reference cells are not counted as new runs.

Change only receiver-assigned residual peer SD from tau_social=1 to 2. The model's variance term changes from sender_variance+1 to sender_variance+4. At fixed sender uncertainty the assigned per-peer-period precision ceiling changes from 1 to .25; realized total evidence, content, rankings and outcomes may change in either direction. No direct modification to message generation, audit rules or initialization. Not a correlated-evidence correction. No further tau levels, frozen policies, nulls, C experiments or alternative networks.

The native pilot runner explicitly supplies tau_social=1 and its validator hardcodes that value. W2 uses a process-local wrapper to replace only the explicit argument. It checks actual tau and retains the original validation checks through a temporary validation-only metadata view; stored and model metadata always retain their actual tau. The observer independently validates every delivered source precision. All previous source files remain unchanged.

## Gates and execution

First inventory local run manifests (including manifest members in ZIP archives) and inspect any nonbaseline tau values or scan errors. This is metadata discovery, not proof that every uninspected run lacks W2. If comparable results already exist, do not run duplicates. Only document an inventory resolution after inspection; do not bypass to obtain a preferred result.

From repository root, with the original paper-b environment:
```bash
git pull --ff-only
conda activate paper-b
python -m paper_b.experiments.run_v63_w2 --inventory-only
python -m paper_b.experiments.run_v63_w2 --workers 4
```

The second command automatically checks the inventory, frozen source SHA-256s and software versions, runs unit tests, 8 short-horizon tau2 preflight cells (with observer parity), 8 full-horizon tau1 archive replays at seed 31001, and all 8 full-horizon tau2 cells at seed 31001 with observer-on/off parity. Only then are the remaining 792 scientific runs launched. The eight full tau2 gate runs are part of the 800, not extra scientific cells. For a preflight-only run, append --preflight-only (short preflight and tau1 replay only). Technical replays must match archived terminal, window, checkpoint, first-audit and validation records within the documented tolerance (numeric 1e-12 relative/absolute; exact keys and nonnumeric values); initial-audit signatures and initial-state/structural fingerprints must match across tau in every production cell. The runner writes error logs per failed cell and refuses consolidation unless all gates and all 800 shards pass. No scientific sign is a gate.

Default reference is ~/Research/paperb_v6_results/v62_mechanism_closure/v62_7e2e4f6724e3; W2 output is ~/Research/paperb_v6_results/v63_w2/w2_<design-id>. Use --reference, --output-dir or repeated --archive-root if needed. Resume the same command in the same checkout/environment; checksummed valid shards are reused. Incomplete artifacts are retained. Do not modify archived V6.2 outputs or rerun the V6.2 master from this branch. Changes to the runner or checkout create a new W2 design identity, not an overwrite.

## Analysis and transfer

Paired whole-seed bootstrap, 10,000 draws, seed 20261125. Report both tau levels, within-tau high-minus-low contrasts, adaptive-minus-neutral interaction, and tau2-minus-tau1 changes in contrasts and interactions. Primary H aligned (group_1) and G population (all), MSE and Q_jammer. Secondary all groups/roles, absolute outcomes, W_jammer, cumulative Q and raw precision shares, delivered peer content and snapshots in predefined windows, and floor events. Raw prior-inclusive precision shares are Q_k/(initial_phi+sum Q); these are posterior-normalized weights when the floor is inactive. W_jammer retains the existing evidence-channel definition. Missing content denominators are undefined, not zero; valid counts reported. No damage contrast or causal mediation. All valid effects, including reversal or null, are reported.

Upload all ZIP files from shareable/w2_primary and shareable/TRANSFER_INDEX.json when COMPLETE is printed. The inherited validated packaging helper names parts v62_<W2-design-id>_w2_primary_...zip; the W2 identity and category distinguish them from V6.2. The packages include run validation/checksums, manifest, inventory, execution status, all seed-level statistics and summaries; retain the complete local W2 shards for final replication. Do not commit data to Git. If a gate fails, send its error log and EXECUTION_SUMMARY.json instead of changing the scientific design.

Scientific closure: all 800 valid runs or an existing valid equivalent comparison, no outcome-dependent expansion. Then revise manuscript, freeze submission candidate and build anonymized replication materials.
