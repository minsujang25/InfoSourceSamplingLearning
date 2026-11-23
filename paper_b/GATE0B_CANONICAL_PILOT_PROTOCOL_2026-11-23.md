# Paper B Gate 0B — Corrected canonical sender × peer pilot

Date: 2026-11-23
Branch: `paper-b-gate0b-canonical-pilot`
Parent: `paper-b-gate0a-initial-audit`

## Scientific status

Gate 0A restored non-disruptive `t=0` centroid audits.  Gate 0B is
**not** a replacement for previously frozen Experiment IIIa, the
separate Experiment IIIb diagnostic, or the v5.2 manuscript.

Gate 0B asks:

1. Does the current myopic K=1 adversary still choose extreme first-attack
   messages after the corrected centroid audit, under the canonical
   `source_posterior` evidence assumption?
2. Does adding two peer sources change **total MSE and null-relative
   excess damage**, under the same elite access and initial beliefs?
3. Is the peer contrast different for adaptive versus fixed-biased senders,
   and for adaptive versus frozen receiver reliance?

## Principal pilot design (explicit flags)

- Structural environments: `elite_only` and `extended`. Both give
  every citizen direct Expert and Jammer access. `extended` adds two
  random citizen peers; it is a total social-opportunity intervention,
  **not** a pure peer-message or fixed source-count effect.
- Sender regimes: `null`, `fixed_biased`, `adaptive`. The
  `null` sender retains its structural slot and, in centroid mode,
  participates in the same initial `t=0` credibility audit, but
  provides no substantive messages thereafter. It is **not** a
  truth-clone source. All three sender regimes have their own
  `jammer_regime` labels and must never be collapsed by the boolean
  `jammer_active`.
- Receiver reliance: `adaptive`, `frozen`. Primary freeze rule:
  `first_audit`: both regimes use the `t=0` credibility audit
  before frozen rankings are fixed.
- Peer evidence: `source_posterior`; `tau_social=1`.
  Citizen peer messages are grouped per source within each period.
- Initial audit: `centroid` for **all** sender regimes.
- Prior: `flat` for the first mechanism pilot. `polarized` can be
  added as a prespecified boundary-condition run, but not post-hoc to
  hunt for preferred results.
- N=100 citizens, K=1, `epsilon=.05`, `credit=20`,
  5-period surveillance, fixed horizon T=26 (diagnostic), matching
  seeds 6901 and 6902. Use a separate design ID to distinguish this
  from production.
- Conditions per seed/prior/network = 3 senders × 2 receivers = 6.
  Two networks × 2 seeds × 6 = **24 runs** for the first pilot.
- A full T=400 and larger matched-seed production should be considered
  **only after** the 26-period gate, not treated as already run.

### Command: corrected, primary first-audit pilot

```bash
python -m paper_b.experiments.run_matched_pilot \
  --seeds 6901,6902 --initial-regimes flat \
  --network-environments elite_only,extended \
  --n-citizens 100 --horizon 26 --k 1 \
  --epsilon 0.05 --credit 20 \
  --sender-regimes null,fixed_biased,adaptive \
  --peer-evidence-mode source_posterior \
  --tau-social 1 --frozen-ranking-mode first_audit \
  --initial-audit-mode centroid --workers 1 \
  --output-dir "$HOME/Downloads/paper_b_gate0b_primary"
```

### Command: separate frozen-rank sensitivity

Same inputs/seeds, except `--frozen-ranking-mode pre_disruption`
and `--output-dir "$HOME/Downloads/paper_b_gate0b_pre_disruption"`.
This is intentionally a **different treatment design**, not another
draw from the same frozen ranking treatment. The adaptive receiver
has the same substantive rule, but stochastic draws/initial ranking
must still be inspected rather than presumed identical.

### Baseline compatibility

Omitting `--sender-regimes` preserves the old 4-way
`adaptive__J1,adaptive__J0,frozen__J1,frozen__J0` design, where
J0 is `truth_clone`. The default peer-evidence and tau flags also
remain `legacy_batch` / 0 so previously used pilot commands retain
their interpretation. **Never silently treat legacy defaults as
canonical**; Gate 0B requires every scientific option explicitly.

## Prespecified estimands

For seed s, network g, receiver r, and sender a:

- Total loss: `MSE(s,g,r,a)`.
- Null-relative excess damage:
  `D(s,g,r,a)=MSE(s,g,r,a)-MSE(s,g,r,null)`.
- Peer addition effect on damage for each sender:
  `Delta_peer(s,r,a)=D(s,extended,r,a)-D(s,elite_only,r,a)`.
- Sender adaptation contrast within a network:
  `D(s,g,r,adaptive)-D(s,g,r,fixed_biased)`.
- Peer × sender adaptation interaction:
  `[D(extended,adaptive)-D(extended,fixed_biased)] -
   [D(elite_only,adaptive)-D(elite_only,fixed_biased)]`
  for each s and r (network indices suppressed).
- Receiver effect on damage: `D(adaptive receiver)-D(frozen receiver)`
  within s, g and sender.

Each contrast is paired at **seed/initial regime**. Same seed
guarantees matched prescribed initial states and structural maps
*within* counterfactual blocks, but not identical realized noise
streams after different graph/source/sender choices. Compare
init-state hashes; do not overstate per-message common randomness.

Report null MSE and total sender MSE as well as D; a low D can coexist
with poor baseline learning. Treat the two-seed results as diagnostics,
without SE-based substantive claims or extending sample results
to the full canonical production.

## Gate conditions

PASS requires:

1. No missing or duplicate six-cell blocks and correct manifest labels.
2. All runs finite, full horizon, matched within-block source maps/initial
   states; initial states also match across elite_only/extended for a seed.
3. Exact `source_posterior`, `tau_social=1`, `first_audit`,
   `centroid` values recorded per run.
4. Period-0 message means exactly equal cluster centroids for
   null/fixed/adaptive, no actual strategic objective at t=0.
5. Period-1 first adaptive strategic optimization; refresh at
   t=1,6,11,16,21 for interval 5.
6. Credibility audit probability remains `1/(t+1)` (100% at t=0;
   50% per citizen at t=1); it is **not** changed by this experiment.
7. Null-relative contrasts are calculated from the **null** regime,
   never the old truth-clone J0.
8. First-audit and pre-disruption results have different design IDs
   and are not pooled as if the same ranking rule.

If Gate 0B reveals extreme t=1 means again, preserve the myopic
sender as a documented baseline. Designing a credibility-aware
sender becomes a **separate prospective model regime**, not an
undocumented adjustment to the historical one.

## Output files

`pilot_manifest.json` (exact configuration + code fingerprint);
`pilot_gate.json` (validation result);
`runs.csv`, `belief_checkpoints.csv` (total MSE and time path);
`gate0b_sender_damage.csv` (null-relative D);
`gate0b_peer_damage.csv` (peer minus elite);
`gate0b_adaptation_network_interaction.csv` (sender × peers);
`gate0b_receiver_differences.csv` (receiver contrast);
`gate0b_checkpoint_sender_damage.csv` and
`gate0b_checkpoint_peer_damage.csv` (time-varying contrasts).
The consolidated shareable ZIP contains these files.

## Manuscript firewall

Do **not** update v5.2 Abstract, Theory, or figures based on this
two-seed 26-period smoke/diagnostic. Preserve all archived 500-seed
Exp IIIa production. IIIb remains separate and follows its own
pre-frozen inclusion rule. Any future production must specify
corrected model version and rerun affected adversarial claims.
