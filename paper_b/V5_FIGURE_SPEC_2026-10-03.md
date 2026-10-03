# Paper B v5 figure specification
## Frozen after figure-data gate PASS — 2026-10-03

The uploaded v5 figure-data bundle passed all row-count gates. The plotting stage
must remain passive: it reads only the frozen CSVs in
`production_results/paper_b_v5_figure_data` and creates publication draft
panels. No simulation or recomputation of model outcomes is allowed.

## Figure 2 — Experiment 1 phenomenon

**Panel A: terminal learning loss.** Plot the eight canonical Experiment-1 null
cells on a log MSE axis. Order structural cells as low-H/low-S,
high-H/low-S, low-H/high-S, high-H/high-S. Distinguish adaptive and frozen
reliance with slightly offset markers and MCSE error bars, but do **not** connect
the four categorical cells with lines. The visual target is the high-H/high-S
departure, especially under frozen reliance.

**Panel B: high-H/high-S trajectory.** Plot MSE at T=100,200,300,400 for
adaptive and frozen reliance. Use a log MSE axis. The target interpretation is
slower convergence, not permanent lock-in.

The terminal group-gap result remains supplementary/supporting text rather than
a third main panel.

## Figure 3 — Scope-condition phase diagram

Create one heatmap panel for adaptive and one for frozen reliance.

- x: epsilon = .02,.05,.10,.20,.30
- y: peer degree d = 2,3,4
- fill: log10 H-by-S terminal-MSE interaction
- annotate each cell with the raw interaction value

Both panels must use the same color limits so magnitudes are directly
comparable. The manuscript text, not the graphic, states the monotonicity:
every row declines with epsilon and every column rises with d, with no
reversals.

## Figure 4 — Corrective Expert accessibility

**Panel A:** mean initial Expert rank by peer degree.

**Panel B:** Expert inclusion-probability heatmap over the same epsilon-by-d
grid.

**Panel C:** initial Expert inclusion probability versus terminal MSE for all
15 high-H/high-S specifications, with adaptive and frozen reliance
distinguished. Use log terminal MSE. Report in the caption/text the descriptive
Spearman relationships (adaptive -1.000; frozen about -.982), but do not call
this mediation.

Do not use terminal W-Expert share as a welfare criterion. The adaptive d=4
non-monotonicity in terminal W-Expert is substantively compatible with early
corrective access followed by accurate peer transmission.

## Figure 5 — Experiment 2 structural rank protection

The initial plotting pass creates empirical panels only; the stylized
low/high-multiplicity network schematic can be composed separately as Panel A
at manuscript-layout stage.

**Accessibility panel:** compare low versus high multiplicity in top-gateway
share, gateway acquisition mass, and gateway inclusion probability.

**Best-rank diagnostic:** compare mean best gateway rank under low/high
multiplicity, but retain this as a Supplement diagnostic rather than a required
main-text panel because it is largely redundant with the top-ranked-gateway
share in the accessibility panel.

**Null-loss panel:** terminal MSE under low/high multiplicity by reliance mode.

**Stress panel:** fixed-biased excess MSE (fixed-biased MSE minus null MSE)
under low/high multiplicity by reliance mode, with MCSE computed from the paired
seed-level fixed-minus-null contrast.

Core interpretation: Experiment 2 protects corrective sources from local rank
competition; it is not merely a generic "redundancy" result.

## Formatting rules

- Use matplotlib only.
- Use matplotlib defaults rather than a custom style or manually fixed colors.
- Each panel is generated as its own PDF and PNG. Multi-panel composition will
  occur in LaTeX after panel review.
- Use readable axis titles and compact captions; do not embed manuscript
  interpretation as large text inside the plotting area.
- Preserve "null" as a literal sender-regime label when reading CSVs; pandas
  default NA parsing can otherwise convert it to NaN.
