# V6.2 mechanism closure — local execution protocol

Status: implementation candidate. The new grid and preflight checks are fixed
before any new scientific outcomes are inspected. No simulation has been run
by the author of this code here. The local unit tests and execution gates
must pass. Never bypass a failed gate.

## Frozen design

Base V6.1 commit: e8aeae6b6650461ac0e3cdba0b78a14b010f6013.
Archived model and V6.0/V6.1 runners remain unchanged. All new source lives
in v62_* modules, run_v62_mechanism_closure.py, and the test module.

N=100, T=400, C=20, K=1, epsilon=.05, tau_social=1,
MIN_SD=1e-8, original source-posterior update, delta comparison,
first-audit frozen ranking, synchronous updates. H high prior segregation;
G flat priors and the original gateway-composition/topology bundle.
All scientific cells are active fixed-biased sender; baseline first
audit is centroid-centered, subsequent mean 4. No new null cells.
Do not combine these new seeds with old nulls.

| Variant | H cells | G cells | New runs |
| --- | ---: | ---: | ---: |
| baseline | 4 | 4 | 800 |
| neutral | 2 | 2 | 400 |
| RA | 4 | 4 | 800 |
| RB | 4 | 4 | 800 |
| uniform_tail | 2 | 0 | 200 |
| constant_audit | 2 | 0 | 200 |
| epsilon35 | 2 | 0 | 200 |
| Total | 20 | 14 | 3400 |

All variants use common seeds 31001..31100. Neutral means IID uniform
substantive requests with canonical source-ID order, retaining audits and
latent credibility learning. RA and RB reuse the V6.1 interventions.
Uniform-tail and constant-audit reuse V6 rules (constant probability
calculated for T400 even during short preflight). Intermediate epsilon=.35.
No extra seeds, topology, C experiments, factorial or effect-driven reruns.

## One-command execution

From repository root, in the exact archived paper-b conda environment:

    conda activate paper-b
    python -m paper_b.experiments.run_v62_mechanism_closure --workers 4

The master first resolves and verifies exact original H/G production
shareable and H/G scientific-pilot ZIP archives. It searches
~/Research/paperb_v6_results and repo production_results/ and local_results/.
Optional RA/RB and behavioral archives are replayed when present, and
missing optional reference archives are explicitly recorded. Required
archive absence, hash mismatch, environment/fingerprint mismatch stops
execution. Existing compact production ZIP supports archived group/run
replay; pilot ZIP supports individual replay. Full old raw production
archive is not required or independently replayed at citizen level here.

For different archive locations use repeated --archive-root or explicit
--production-zip, --pilot-zip, --ra-zip, --rb-zip, --behavioral-zip.
--dry-run verifies archive inventory and prints design without learning.
--preflight-only executes technical gates only and exits nonzero if failed.
Default invocation does both stages automatically. No need to invoke
seven separate runners. Reinvoke the same command after interruption:
resume is automatic, with design ID, software/source hashes and shard
checksums checked. Incomplete uncommitted shard files are quarantined;
committed hash mismatch is an error. No existing outputs are deleted.

## Gate schedule

- New and inherited unit tests, archived code/software fingerprints, seed
  disjointness against known old ranges and accessible archive manifests.
- Exact old production seed 20001 and pilot seed 6001 active H/G baseline
  replays, plus old RA/RB/behavioral replays when the reference ZIPs exist.
- Reconstruct 500 paired G opportunity graphs without learning, compare
  structural fingerprints and topology to original production, compute
  all-slot uniform-use U^A incoming peer HHI/top5 alongside archived W.
- Short T26 preflight for all 34 cells, added observer on/off exact
  trajectory comparison, matched initial-state/graph and first-audit
  comparisons. RA changes the initial Jammer message intentionally.
- T400 scientific grid only for passing variants. A failed baseline gate
  blocks dependent variants. Independent passing variants can complete;
  no combined scientific summary is published unless all 3400 pass.
- Full observer parity at 31001,31025,31050,31075,31100 in low/adaptive
  representative per variant/suite =55 technical runs.
- Per-period request/delivery, actual assigned precision, finite state,
  audit-no-state-change, posterior accounting and terminal reconciliation.
- Whole-seed bootstrap 10,000 draws, RNG 20261125, explicit missing
  denominators. Structural summaries use the 500 original graph seeds.

## Output and interpretation

Output default: ~/Research/paperb_v6_results/v62_mechanism_closure/v62_<id>/.
Each run is independently compressed and checksum-validated. Full compact
period histories are streamed to local files for 400 baseline H runs
(16 million citizen-periods). Other runs retain individual terminal
accounting, initial audits, window aggregates and checkpoint/event records.
All learning records are immediate-source evidence, not provenance.

The observer uses actual pending source precision and actual delivered
sample mean; for peers also records the snapshotted transmitted posterior
mean. Windows: 1..10, 11..100, 101..399. The prior-precision field within
source windows is summed over source opportunities, not a unique count of
receiver-periods. Per-period gain is q/(phi_pre + sum q).
The adjusted signed accounting rescales at SD floor activation, retaining
unscaled Q/B totals. Tolerance: absolute 1e-9 plus 1e-10 times
max(1,abs(a),abs(b)); old archived replay numeric tolerance is 1e-12
absolute/relative. Signed MSE contributions are arithmetic, not causal.

Seed tables retain levels, high-minus-low contrasts, adaptive-neutral,
adaptive-frozen, variant-minus-baseline and interaction changes, group
and role metrics, window content ratios, conditional sign categories,
checkpoint precision paths and delayed-promotion incidence. Initial
rank3/4, prior peer-associated drift without direct Jammer evidence in
that update, later behavioral rank1 promotion, next five substantive
opportunities for uptake. Right censoring and strict zero prior Jammer
precision indicator retained. Event order is descriptive, not mediation.

Primary results are absolute MSE and assigned immediate-source Q_J.
H initially aligned group; report opposite and all. G population and
roles. No new inactive-slot excess-loss estimates. Scope results and
null/contrary directions are reported regardless of sign. No new
simulation after technically valid complete 3400 grid. E1/E2/E3/E4/E5
interpretation follows the prospective analytical protocol.

Automatically generated shareable/primary/ ZIP files contain execution
summary, manifests, seed-level tables and statistical summaries, G
structural benchmarks; share these first, together with its INDEX.json.
shareable/individual_detail/ has per-run individual accounts and windows.
shareable/histories/ has full baseline H compressed histories. Keep
these locally unless specifically requested. Each category has an
INDEX.json and independent ZIPs with member hashes and row/seed coverage;
all ZIPs must be <=20 MiB. Data members are already partitioned per
scientific run or seed, with summary rows chunked. If an unexpectedly
oversize individual logical member prevents transfer, preserve the local
files and report the error rather than truncate any data.

Verify a received complete category:

    python -m paper_b.experiments.run_v62_mechanism_closure --verify-package /path/to/primary

The master prints COMPLETE only after all three categories have been
packaged and checked. Technical gate success does not establish any
theoretical hypothesis. Do not commit results, private reviews or
manuscripts to the public source repository.
