# Social Networks Paper B — v5 Restructuring Blueprint
## Frozen after analysis closure — October 3, 2026

This blueprint governs the next manuscript round. It starts from
`SocialNetworks_PaperB_LaTeX_v4_2026-10-02` but incorporates all post-v4
results: W-channel decomposition, the complete epsilon-by-degree phase diagram,
rank-unification diagnostics, and the final passive analysis-closure audit.

The analysis program is frozen. v5 is a manuscript/visual restructuring round,
not an invitation to search for new robustness results.

---

## 1. v5 narrative spine

The paper should now read as one rank-competition argument:

[
A
ightarrow
R_t / Pi_t
ightarrow
Lambda_t
ightarrow
W_t
ightarrow
	ext{collective learning}.
]

The substantive mechanism is:

[
	ext{opportunity composition}
ightarrow
	ext{relative rank}
ightarrow
	ext{behavioral accessibility}
ightarrow
	ext{evidence allocation}
ightarrow
	ext{learning}.
]

The central scope condition is:

[
	ext{segregated priors}
+
	ext{many congenial alternatives}
+
	ext{steep attention concentration}
Rightarrow
	ext{corrective crowding-out}.
]

Experiment 1 shows crowding-out. The epsilon-by-degree phase diagram defines its
scope. Expert-access reconstruction identifies the mechanism. Experiment 2
shows structural rank protection. Persistent biased information is an
asymmetric stress application, not a simple mirror image of Expert
crowding-out.

---

## 2. Main-text visual architecture

### Figure 1 — From structural opportunity to evidence dependence

Purpose: establish the Social Networks contribution before presenting outcomes.

Preferred format: one common node layout across three panels.

- Panel A: binary structural opportunity network (A);
- Panel B: acquisition network (Lambda), same feasible ties but weighted by
  expected sampling;
- Panel C: evidence-precision network (W), same sources weighted by cumulative
  precision contribution.

Annotate the intervening ranking/acquisition process (R_t,Pi_t). Use the
same nodes in every panel so the visual claim is literally (A
eqLambda
eq W).
Do not label (W) as causal influence.

Status relative to v4: REPLACE placeholder Figure 1.

### Figure 2 — Experiment 1: canonical learning consequence

Purpose: show the phenomenon before explaining its scope and mechanism.

Recommended panels:

- A: terminal null MSE across the full (H	imes S) factorial, with adaptive
  and frozen reliance distinguished;
- B: high-H/high-S learning trajectory through (T=400), showing slower
  convergence rather than permanent lock-in.

Optional: show the terminal group gap only in the Supplement unless it is needed
as a small secondary annotation.

Status relative to v4:
- REPLACE main Table `tab:exp1_null` with this figure;
- move exact cell values to a Supplement table.

### Figure 3 — Scope-condition phase diagram

Purpose: become the central mechanism/scope figure.

Two heatmaps, one adaptive and one frozen:

- rows: (din{2,3,4});
- columns: (epsilonin{.02,.05,.10,.20,.30});
- cell: (H	imes S) terminal-MSE interaction.

Because the surface spans several orders of magnitude, use a common log-scaled
color mapping while annotating cells with raw interaction values.

The caption/text must emphasize:

- every row is monotonically decreasing in epsilon;
- every column is monotonically increasing in d;
- there are no reversals;
- the canonical (d=2,epsilon=.05) case is one point on a continuous
  scope-condition surface, not a privileged representative value.

Status relative to v4:
- REPLACE `tab:mechanism_robustness`;
- retire the separate epsilon and degree robustness framing in the main text.

### Figure 4 — Why the phase surface emerges: corrective Expert accessibility

Purpose: connect rank depth and lower-rank accessibility directly to learning.

Recommended panels:

- A: mean initial Expert rank in high-H/high-S as d increases
  (approximately 2.63, 3.45, 4.26);
- B: initial Expert inclusion-probability heatmap over the same
  ((epsilon,d)) grid;
- C: terminal MSE versus initial Expert inclusion across the 15 high-H/high-S
  specifications, distinguishing adaptive and frozen reliance;
- D (only if visually clean): canonical high-H/high-S Expert-acquisition path,
  showing early adaptive restoration relative to frozen reliance.

Use the observed near-mirror relationship between accessibility and learning:
Spearman rho is approximately -.982 frozen and -1.000 adaptive across the
15 specifications. Present this as descriptive mechanism alignment, not causal
mediation.

Important nuance: terminal (W^E) is not a welfare criterion. Adaptive d=4
can continue improving after terminal direct Expert dependence stops rising,
because early Expert access can correct peers that later transmit accurate
information.

Status relative to v4:
- REPLACE `tab:expert_rank` in the main text;
- retain detailed rank distributions and W-channel decomposition in the
  Supplement.

### Figure 5 — Experiment 2: structural rank protection

Purpose: unify Experiment 2 with the same rank-competition framework.

Recommended panels:

- A: stylized low- versus high-multiplicity opportunity structure for a
  non-gateway citizen;
- B: gateway rank/accessibility:
  low multiplicity versus high multiplicity (top-ranked gateway share,
  best rank, acquisition/inclusion);
- C: null terminal MSE by multiplicity and reliance mode;
- D (optional but useful): fixed-biased damage by multiplicity and reliance,
  showing that structural rank protection also helps under persistent stress.

Core mechanism statement:
Experiment 1 allows congenial alternatives to crowd corrective access out;
Experiment 2 structurally protects corrective access from local rank
competition by replacing the ordinary competitor with another corrective
gateway.

Status relative to v4:
- REPLACE `tab:exp2_gateway` and placeholder Figure 3;
- move detailed A/Lambda/W gateway-share contrasts to the Supplement.

---

## 3. Main-text table triage

### Canonical production design table
Status: KEEP / REVISE.

Retain one compact design table in the main text because the paper has multiple
simulation modules. Revise the caption and/or rows so readers can distinguish
canonical production from the later phase/scope and passive mechanism audits.
Do not let the table become a chronological version history.

### `tab:exp1_null`
Status: REPLACE in main text; MOVE exact values to Supplement.

Figure 2 should carry the phenomenon.

### `tab:expert_rank`
Status: REPLACE in main text; KEEP detailed values in Supplement.

Figure 4 should carry rank depth and accessibility.

### `tab:mechanism_robustness`
Status: DROP as a main-text object.

It is superseded by the complete 15-cell epsilon-by-degree phase diagram.

### `tab:exp1_layers`
Status: MOVE TO SUPPLEMENT.

Useful for documenting (A
eqLambda
eq W), but it no longer organizes the
main mechanism argument.

### `tab:exp2_gateway`
Status: REPLACE in main text; KEEP expanded version in Supplement.

Figure 5 should carry structural rank protection.

### `tab:total_loss`
Status: MOVE TO SUPPLEMENT.

The fixed-biased environment remains substantively important, but the full
8-row total-loss table interrupts the main rank-competition narrative.
Main text should report only the comparisons needed for the threat-dependent
reconfigurability claim.

Net target: roughly five main figures and one compact main table.

---

## 4. Theory restructuring

### Keep and sharpen: From observed ties to consequential dependence

Retain the literature bridge from binary opportunity to weighted consequential
relations and the distinction (A
eqLambda
eq W).

### Rewrite: Credibility ranking and congenial crowding-out

Make this the central micro-to-network mechanism section.

Define:
- rank competition;
- corrective-source rank depth;
- lower-rank accessibility;
- Expert inclusion probability.

The conjectures on congenial crowding-out and lower-rank accessibility remain
useful but should be rewritten around the full phase surface rather than the
old canonical/robustness split.

### Demote/absorb: Re-ranking, acquisition homophily, and closed dependence

Do not retain "closed dependence" as a co-equal primitive theory subsection.

Keep:
- adaptive versus frozen re-ranking;
- (H^A,H^Lambda,H^W) distinctions.

Move dominant cycles / Expert reach into a downstream network-signature
paragraph. State explicitly that dominant-edge skeletons can suppress weak but
behaviorally consequential access paths.

### Rewrite: Corrective-route multiplicity and gateway concentration

Reframe as structural rank protection:
more corrective routes do not merely increase "redundancy"; they reduce the
chance that an ordinary competitor occupies the scarce high-accessibility
position.

### Rewrite: Persistent biased sources

Keep the threat-environment logic but remove any suggestion of a simple
homophily-induced Jammer crowding-in mirror image.

Supported interpretation:
- source/prior alignment places the biased source at favorable ranks for the
  aligned +1 subgroup;
- high homophily does not further improve Jammer rank;
- additional high-H damage combines existing biased exposure with reduced
  corrective access and greater same-group peer dependence.

### Synthesis

End with the common rank-competition framework rather than separate
"homophily", "redundancy", and "sender" stories.

---

## 5. Results restructuring

Recommended order:

### 5.1 Relational layers are distinct
Very brief behavioral-identity gate + one or two illustrative A/Lambda/W
differences. Do not front-load detailed network statistics.

### 5.2 Experiment 1: the canonical crowding-out phenomenon
Figure 2. Establish the H-by-S learning consequence and slower convergence.

Add the new terminal group-gap diagnostic as supporting interpretation:
high-H/high-S residual error includes persistent subgroup separation, especially
under frozen reliance. Do not relabel this as a new polarization outcome.

### 5.3 Scope conditions: attention concentration x peer multiplicity
Figure 3. Replace the old "pre-specified mechanism robustness" narrative with
the complete phase surface.

### 5.4 Mechanism: rank depth x lower-rank accessibility
Figure 4. Reconstruct initial Expert rank and inclusion; connect the surface to
learning; explain early adaptive restoration.

Include the W-channel result:
increasing epsilon also increases cross-group peer exposure, but restored Expert
access is the dominant evidence channel in the present design.

### 5.5 Downstream network signatures
Short subsection or closing paragraphs of 5.4:
A/Lambda/W homophily, evidence enclosure, dominant Expert reach.
Use them as consequences/signatures, not mediation.

### 5.6 Experiment 2: structural rank protection
Figure 5. Show gateway rank/accessibility and null learning, then the
fixed-biased stress result if Panel D is retained.

### 5.7 Persistent biased information: asymmetric stress application
Condense the current total-loss and subgroup sections into one subsection.

Key supported facts:
- fixed-biased damage is concentrated in the aligned +1 group;
- alignment drives favorable Jammer rank/access;
- high H does not further crowd the Jammer in;
- high-H incremental damage is better described as biased exposure combined
  with reduced corrective access and increased same-group peer dependence.

### 5.8 Myopic adaptive sender
Reduce to a short boundary-condition paragraph or move the full analysis to the
Supplement. It should not organize the paper.

---

## 6. Supplement restructuring

Keep:
- receiver specification checks;
- horizon calibration;
- behavioral identity of the W rerun;
- detailed A/Lambda/W layer tables;
- exact Expert-rank reconstruction;
- adaptive Expert-acquisition checkpoints;
- myopic adaptive-sender boundary condition;
- numerical/control semantics.

Merge/replace:
- merge the old epsilon robustness and degree-four robustness subsections into
  one full phase-diagram appendix with the 15-cell numeric surface;
- replace the old Experiment-2 gateway baseline subsection with an expanded
  gateway-rank/accessibility audit.

Add:
- W-channel decomposition (Expert / same-group / other-group channels);
- terminal group-gap closure table;
- canonical A-peer SNA descriptives:
  outdegree, indegree SD, reciprocity, transitivity, same-group share, E-I;
- fixed-biased Jammer rank/accessibility by citizen group and homophily;
- detailed total-loss table moved from main text.

Do not use Supplement headings that encode project/version history such as
"historical", "post-v4", or "rank-unification audit".

---

## 7. Analysis-closure facts to integrate

### Group-gap diagnostic

High-segregation terminal group gaps:
- adaptive low H: about .0062;
- adaptive high H: about .0998;
- frozen low H: about .0164;
- frozen high H: about .5942.

H-by-S group-gap interaction:
- adaptive: about .0887 (MCSE .0034);
- frozen: about .5757 (MCSE .0137).

Use absolute terminal gap as the primary descriptive diagnostic. Do not use
low-segregation terminal/initial gap ratios in the manuscript because the
near-zero denominator makes those ratios unstable.

### A-peer opportunity-network descriptives

Low H versus high H:
- peer outdegree: 2.000 vs 2.000;
- mean indegree: 2.000 vs 2.000;
- indegree SD: about 1.391 vs 1.393;
- reciprocity: about .0209 vs .0345;
- undirected transitivity: about .0283 vs .0435;
- same-group peer-edge share: about .501 vs .900;
- E-I index: about -.003 vs -.801.

Interpretation:
the manipulation primarily changes mixing composition while degree and overall
indegree dispersion remain essentially unchanged. The modest changes in
reciprocity/transitivity are small in absolute terms and do not indicate a
strongly clustered-community artifact.

---

## 8. Immediate execution order for v5

1. Build figure-data tables from frozen outputs.
2. Produce draft Figures 2--5 with publication-neutral formatting.
3. Produce Figure 1 conceptual schematic.
4. Rewrite Results to match Figures 2--5.
5. Rewrite Theory to match the rank-competition framework.
6. Revise Model/Design only where needed for consistency.
7. Rewrite Introduction and Discussion after the core is stable.
8. Rebuild Supplement and move detailed tables out of the main text.
9. Compile and perform a manuscript-wide terminology/version-history audit.

Do not start new model experiments unless a concrete implementation error is
found.
