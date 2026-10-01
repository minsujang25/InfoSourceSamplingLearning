# Experiment IIIb Extension Plan — Additional 50 Seeds and Cumulative N=100

## Status and provenance

The original Experiment IIIb diagnostic is complete and remains frozen at
50 matched seeds (4001-4050). Its result was inspected before this extension
was proposed. The original 50-seed classification must therefore remain
visible and must not be rewritten as if the 100-seed exercise had been
pre-specified from the start.

The extension is a one-time precision check motivated by the original
Case-B result: the independent-path contrast was directionally negative and
not baseline-driven, but Monte Carlo-noisy.

This document is frozen **before** running the additional seeds.

## Extension design

Run only new matched seeds:

```text
4051-4100
```

with the same scientific specification as the original diagnostic:

```text
50 new matched seeds
x 2 path structures
x 2 reliance modes
x 2 Jammer states
= 400 additional runs

N citizens   = 100
T            = 200
K            = 1
epsilon      = 0.05
credit       = 20
surveillance = 5
MIN_SD       = 1e-8
prior        = flat
Jammer        = posterior-uncertainty formulation
```

The original seeds 4001-4050 are not rerun by the extension wrapper.

## Reporting rule

The cumulative analysis must preserve three separate views:

1. original 50 seeds (4001-4050);
2. extension 50 seeds (4051-4100);
3. cumulative 100 seeds (4001-4100).

The extension must not overwrite the original Case-B classification. The
cumulative analysis is used only to determine whether the directional pattern
is sufficiently stable and precise to justify considering a larger
confirmation run.

At minimum report, for each cohort and for the cumulative sample:

- adaptive population path-effect mean;
- MCSE and approximate 95% Monte Carlo precision interval;
- median and 5% trimmed mean;
- share of seed-level contrasts below zero;
- adaptive J=1 and J=0 absolute-loss differences;
- focal-citizen path effect;
- frozen path effect;
- adaptive-minus-frozen path interaction.

The original and extension seed sets must be disjoint, all structural/scientific
gates must pass, and the two bundles must have identical scientific-code and
decision-rule fingerprints.

## Interpretation

A cumulative result that remains directionally negative but Monte Carlo-noisy
supports retaining IIIb as a Supplementary Materials mechanism check.

A cumulative result that becomes clearly negative, with the extension cohort
pointing in the same direction and without a material baseline-driven pattern,
can justify considering a later 500-seed confirmation. It does not
retroactively convert the original 50-seed diagnostic into a precommitted
main-text result.

If the extension cohort reverses the pattern or the cumulative result weakens
substantially, no larger IIIb run is warranted.

## Commands

Run the additional seeds only:

```bash
PAPER_B_WORKERS=12 bash scripts/run_paper_b_exp3b_extension50.sh
```

Then combine the frozen original bundle with the extension bundle:

```bash
bash scripts/run_paper_b_exp3b_cumulative100.sh
```

The combiner never reruns simulations.
