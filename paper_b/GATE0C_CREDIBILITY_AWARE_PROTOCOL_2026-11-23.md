# Gate 0C — Prospective credibility-aware sender diagnostic (FROZEN BEFORE RESULTS)

Date: 2026-11-23. Branch: `paper-b-gate0c-credibility-aware`.
Reference: `paper-b-gate0b-canonical-pilot` (legacy myopic sender preserved).

## Scientific hypothesis, not a guaranteed outcome

The existing posterior-gain myopic K=1 attacker maximizes hypothetical
one-message belief damage, with a quadratic message-deviation cost, and
does not discount the action by probability of theta-learning, message
acquisition, or subsequent credibility audit. In Gate 0B its first
strategic message at t=1 was ~53 despite the corrected t=0 audit.

Gate 0C tests whether including **the receiver's existing behavior rules**
in a transparent one-step proxy changes the attack mean and consequences.
It does *not* assume that more credible messages are necessarily better
at maximizing finite-horizon MSE, nor treat favorable outcomes as a
criterion for accepting the new regime.

## Prespecified strategic rule

Let `p_t = 1/(t+1)` be each citizen's audit probability.
The attacker observes beliefs, posterior SDs, current Jammer-specific
credibility-displacement estimates and rankings at surveillance refresh.
For citizen i in current cluster g, define:

- `mu_i`, `sigma_i`: pre-update theta belief mean and SD;
- `b_i`, `v_i`: prior mean and SD for perceived Jammer displacement;
- `q_i = 1-(1-pi_i)^C`, when the Jammer is structurally available:
  probability of >=1 Jammer request in a theta-learning period, from
  citizen's rank `pi_i` and credit budget C; otherwise q_i=0;
- `a_i = min(1,C/d_i)` with `d_i` source count: probability of >=1
  Jammer request in an equal-request audit round. At baseline C=20,
  d_i is 2 or 4, hence a_i=1;
- `k_i = sigma_i^2/(sigma_i^2+1)`: same single-message response gain
  proxy used in the archived myopic objective (variance 1);
- `eta_i = v_i^2/(v_i^2+sigma_i^2+1/max(1,C/d_i))`:
  expectation-based displacement update gain under `learn_delta()`.

With true state theta, attacker's underlying position mu_D and one
common cluster message mean m, define

```
U_g(m) =
 (1-p_t) mean_i{q_i [ (1-k_i) mu_i + k_i m - theta ]^2}
 - (m-mu_D)^2
 - lambda p_t mean_i{a_i [ (1-eta_i)b_i + eta_i(m-mu_i) ]^2}.
```

All means run over **the same full observed cluster**, with zero
exposure terms for inaccessible citizens. The cost term is the archived
quadratic sender deviation cost and is not rescaled. The final term
penalizes the **expected squared posterior perceived displacement** if
an audit occurs; it is NOT a mechanically specified future rank-change
probability. Benefits are discounted by both learning probability and
actual expected source inclusion. The approximate k_i does **not**
model simultaneous multi-source posterior gain exactly.

**Frozen primary lambda = 1.0**, giving equal units to squared
truth loss and squared perceived credibility displacement. The
objective is a strictly concave quadratic with a closed-form maximizer;
no message clipping, arbitrary bounds or results-driven penalty
fitting. A prospective lambda=0 diagnostic separates probabilistic
acquisition/audit timing from the explicit credibility penalty; it is
not an alternative selected after inspecting outcome MSE.

For all regimes, t=0 stays the same non-disruptive
`N(initial-cluster-centroid,1)` audit. In `credibility_aware`,
optimization begins t=1; refreshes at 1,6,11,... for interval 5.
The existing `adaptive` attack strategy and `fixed_biased`,
`null` regimes are retained unchanged.

## Explicit information-set caveat

The new sender reads current recipient Jammer-specific credibility
posteriors and policy rankings at each refresh. The old myopic
attacker only used current theta belief mean/uncertainty.
Thus a comparison is a **joint change in attacker decision rule and
information used**. It is not an isolated causal effect of forecast
horizon, perfect stealth, or a fully solved dynamic game. Call this
strategy a *one-step credibility-aware proxy*.

The attacker uses current true theta in its objective, as the
archived myopic sender does. No extra K segmentation, no surveillance
accuracy manipulation, no behavioral changes on the receiver side.

## First diagnostic: before seeing outcomes

N=100 citizens, T=26, matched seeds 6901 and 6902;
networks `elite_only`, `extended`; initial prior `flat`;
senders `null,fixed_biased,adaptive,credibility_aware`;
receivers adaptive/frozen; K=1; epsilon=.05; budget C=20;
peer_evidence_mode=`source_posterior`; tau_social=1;
initial_audit_mode=`centroid`; surveillance_interval=5.

Primary frozen ranking: `first_audit`; second, separately
fingerprinted sensitivity: `pre_disruption`.

32 runs per ranking specification; 64 total diagnostic runs.
No production hypothesis testing with only 2 seeds.

Primary outcome: terminal total MSE and D_sender =
MSE(sender)-MSE(null). Compare paired credibility-aware minus myopic
D, and interaction of this contrast with peer addition. Secondary:
sender message mean at t=1 and later refreshes; source acquisition
reliance, audit/rank trajectories where measurements are recorded.
Always examine null total MSE; damage reductions from a worse
null baseline are not a resilience proof.

## Execution

```bash
python -m unittest paper_b.test_gate0a_initial_audit \
  paper_b.test_gate0b_canonical_pilot \
  paper_b.test_gate0c_credibility_aware -v

python -m paper_b.experiments.run_matched_pilot \
  --seeds 6901,6902 --initial-regimes flat \
  --network-environments elite_only,extended \
  --n-citizens 100 --horizon 26 --k 1 \
  --epsilon .05 --credit 20 --surveillance-interval 5 \
  --sender-regimes null,fixed_biased,adaptive,credibility_aware \
  --peer-evidence-mode source_posterior --tau-social 1 \
  --jammer-credibility-penalty 1 \
  --frozen-ranking-mode first_audit --initial-audit-mode centroid \
  --workers 1 --no-edge-log \
  --output-dir output/gate0c_primary
```

Run sensitivity with identical arguments except
`--frozen-ranking-mode pre_disruption` and
`--output-dir output/gate0c_pre_disruption`.

## Gate criteria

- Mathematical optimizer matches direct numerical utility evaluation.
- Source and citizen t=0 audit unchanged across new and myopic arms.
- All 8 sender × receiver cells present per seed/network.
- Source origin and initial state fingerprints match within each block;
  initial states also match across environments for a seed.
- `first_audit` and `pre_disruption` have distinct design IDs;
  their outcomes must not be pooled.
- Model policy `p_t=1/(t+1)` unchanged, finite messages/states.
- Gate 0C must not change archived Experiment IIIa, the separate IIIb
  inclusion decision, or the v5.2 main manuscript.
- Sensitivity/large-seed production is planned only after diagnostics,
  regardless of whether aware attacker has higher or lower MSE.

## Interpretation firewall

Peer addition changes source count and therefore rank competition and
allocation probabilities. A peer-minus-elite result is a **total
structural addition** effect, not proof of information recirculation.
W_T is attributed incoming precision, not message-origin provenance.
`credibility_aware` explicitly differs from `adaptive` in information
access; any separate value-of-information claim would require matched
information-access ablation. Any dramatic MSE difference should be
traced to initial t=1 selection, source ranking and subsequent audit.
