# V6.3 passive refinement: terminal transfer

## Provenance and scope

- Base simulation commit: `55264824e35c02284174e206d86955269d2b0bbc` (`paper-b-v6-2-mechanism-closure`).
- Frozen source design: `7e2e4f6724e3`, 34 cells x 100 seeds (31001–31100), T=400, N=100.
- This branch adds read-only post-processing. It is not a new simulation release. Do not rerun the V6.2 master from this branch: its design hash includes source files.
- Preserve committed source shards and earlier branches. Raw outputs, transfer archives, local paths containing identifying information, manuscripts and review reports are not repository content.

## Run from repository root

```bash
conda activate paper-b
git fetch origin
git switch paper-b-v6-3-final-revision
git pull --ff-only
python -m paper_b.experiments.extract_v63_terminal --include-frozen-G
```

The optional frozen-G cohort provides the existing policy comparison at no simulation cost. Omit the flag for the required minimum only. The extractor uses the standard library, reads the existing completed run root by default, and supports `--root PATH` and `--out PATH`. It refuses to overwrite a nonempty transfer directory. If restarting an interrupted extraction, preserve that directory and choose a fresh `--out`.

Default output: the `v63_passive_transfer` subdirectory of the V6.2 run root. Transfer the entire contents of this directory, including `TRANSFER_V63.json`. Do not commit these generated data files.

## Data contract and verification

Required terminal files (20,000 citizen rows each):
- baseline_G_adaptive_terminal.json.gz
- neutral_H_adaptive_terminal.json.gz
- neutral_G_adaptive_terminal.json.gz
- uniform_tail_H_adaptive_terminal.json.gz

Optional: baseline_G_frozen_terminal.json.gz (20,000 more rows).

The extractor also reopens the existing baseline H adaptive/frozen shards for external first-audit observations and run-level provenance; their terminal records need not be exported again. The transfer includes run_metadata.csv.gz, first_audit_external.csv.gz, first_audit_pair_checks.csv.gz, shard_inventory.csv.gz, and TRANSFER_V63.json. The index records source manifest digest, extractor digest, checkout commit (when available), file sizes and SHA-256. Minimum 1,200 inspected shards; 1,400 with optional frozen G. Required terminal rows 80,000; 100,000 with optional frozen G.

Checks include committed shard checksums, design/cell/seed identity, complete citizen IDs, zero floor activation for this specific archived cohort, terminal precision/signed accounting, paired initial-state fingerprints and attributes, first-audit coverage and equality counts, and expected neutral external precision equality. On failure retain the source and error message; diagnose rather than silently skip records or change a scientific specification. The transfer index is written only after the complete extraction. These checks do not constitute a new simulation replay.

## Later final replication release

Use versioned scripts, explicit scientific configuration and seed manifests, source and output checksums, and documented commands. Keep simulation, passive extraction/analysis, exhibit generation, and technical replay distinguishable. Continue to commit analysis code and scientific decisions on this branch as they are implemented, keeping large derived/raw data outside Git with verified manifests. Any subsequent scientific sensitivity requires a distinct design and explicit gate; the current extraction requires no new runs. Final repository cleaning and anonymized reviewer-facing release occur after manuscript freeze, without deleting the immutable development record. Do not include internal review history or author-identifying metadata in the anonymous release.
