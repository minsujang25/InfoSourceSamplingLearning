# Paper B Pipeline — Theory-Aligned Reconstruction

This branch reconstructs the migrated Mesa model around the frozen Social Networks Paper B theory rather than preserving legacy scientific output.

## Frozen model architecture

```text
structural opportunity A
    -> credibility ranking R_t
    -> acquisition policy Pi_t
    -> effective reliance Lambda_t
    -> realized information flow X_t
    -> jammer-induced Delta MSE
```

The shared substantive model remains under `model/`. Paper B configuration, measurement, and experiment code live under `paper_b/`.

## Reconstructed behavioral rules

- Ordinary citizens communicate sincerely from their pre-update posterior:
  `M_ji,t ~ Normal(mu_j,t, sigma_j,t^2)`.
- Periods are synchronous: all period-t messages use the sender state at the beginning of period t; posterior states are exposed as message parameters only at t+1.
- Theta and source-displacement/credibility learning remain separate learning targets.
- Multi-source theta acquisition uses recursive rank-based exploration. For two sources it is exactly `(1-epsilon, epsilon)`.
- `reliance_mode="adaptive"` lets later credibility audits change the behavioral ranking.
- `reliance_mode="frozen"` freezes the ranking after the first credibility audit while substantive and credibility beliefs continue to update.
- Periodic Jammer re-surveillance observes current pre-update citizen beliefs.
- `jammer_active=False` neutralizes adversarial content while retaining the same structural Jammer source slot for matched J=0 counterfactuals.

## Correctness fixes

The reconstruction intentionally breaks strict legacy-output equivalence where the old implementation contradicted the theory:

1. theta posterior variance is no longer stored as a standard deviation;
2. source-displacement uncertainty is updated as a standard deviation consistently;
3. sample-mean precision uses an observation-mean variance rather than treating the raw message SD as the likelihood SD;
4. citizen message states are staged synchronously;
5. Jammer re-surveillance uses current beliefs rather than `mu_theta_beliefs[0]`.

No arbitrary clipping is used.

## Explicit communication environments

Theory-aligned runs should set `network_environment` explicitly:

- `elite_only`: direct Expert + Jammer access, no citizen peers;
- `random_2`: exactly two local sources sampled from the full citizen+elite pool; elite access is localized by the sparse graph itself;
- `group_id`: exactly two local sources with a 0.9 within-group / 0.1 cross-group preference; elite nodes are embedded in the same group-based pool;
- `extended`: direct Expert + Jammer access plus exactly two random citizen peers.

Legacy `mode` labels remain as compatibility aliases but should not be used to define the final Paper B design.

The baseline topology is now frozen: Random 2 local degree = 2, Group ID local degree = 2 with 0.9/0.1 within/cross-group mixing, and Extended peer degree = 2. Degree/bridging variations are robustness treatments, not part of the primary environment definitions.

See `paper_b/NETWORK_ENVIRONMENT_AUDIT.md` for the evidence and the legacy-code mismatch discovered during the audit.

## Main metrics

Main-text mechanism metrics are intentionally parsimonious:

1. Expert/Jammer/peer expected reliance shares;
2. structural-to-effective total-variation divergence;
3. effective same-group peer reliance and its divergence from structural homophily.

Primary outcome:

```text
D_g(K) = E[MSE_T | J=1, g, K] - E[MSE_T | J=0, g]
```

MSE is the theoretical primary loss; RMSE is an interpretable presentation metric; MAE is a sensitivity check. MSE is decomposed into squared population displacement and belief variance.

HHI, realized X_t shares, weighted assortativity, and path-count diagnostics are secondary/supplementary unless later results make them necessary.

## Scientific checks

Run:

```bash
python -m scripts.smoke_test
python -m paper_b.experiments.smoke_pipeline
python -m paper_b.experiments.check_reconstruction
```

The reconstruction checks verify:

- exact two-source nesting and recursive rank probabilities;
- posterior-SD contraction;
- synchronous sender-state snapshots;
- first-audit-frozen reliance;
- current-belief Jammer re-surveillance;
- finite multi-period dynamics and coherent reliance shares.

## Next order

1. Make the reconstruction checks green in CI.
2. Audit remaining model-equation choices against the frozen theory and baseline-paper equations.
3. Run small paired J=1/J=0 and adaptive/frozen diagnostics under the frozen four-environment map.
4. Quantify how corrected topology and learning rules change archived qualitative results.
5. Only then launch the full Paper B simulation grid.
