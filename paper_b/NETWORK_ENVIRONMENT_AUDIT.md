# Paper B Network Environment Design Audit

## Purpose

This audit maps the four communication environments used in the Social Networks manuscript to explicit structural-network generators in the reconstructed Mesa model.

The reconstruction uses the current manuscript/accepted-design descriptions as the target specification and treats the legacy implementation as historical evidence rather than as authoritative when the two conflict.

## Source-supported design

### 1. Isolated elite exposure

**Manuscript specification**

Every citizen can sample the Expert and the Jammer; citizens cannot communicate with one another.

**Reconstructed generator**

For every citizen:

```text
A_i = {Expert, Jammer}
```

Structural source count: exactly 2.

---

### 2. Random 2 matching

**Manuscript specification**

Citizens receive two randomly assigned local connections in a sparse network that includes citizens and the two elite nodes. Direct elite access is localized rather than guaranteed.

**Reconstructed generator**

For every citizen, sample exactly two distinct sources without replacement from all other nodes:

```text
candidate pool = all citizens except ego + Expert + Jammer
degree = 2
```

This directly implements the manuscript phrase “two randomly assigned local connections” and makes elite access an endogenous sparse-network event rather than a universal attachment.

---

### 3. Group ID matching

**Manuscript specification**

Connections are organized primarily within groups defined by similar initial beliefs, with limited cross-group contact. Elite nodes are embedded in the group-based network.

**Legacy-supported mixing rule**

The legacy `group_id_matching` code used a 0.9 within-group / 0.1 out-group draw for citizen contacts.

**Reconstructed generator**

For every citizen, sample exactly two distinct local sources. At each source draw:

```text
P(prefer same initial-belief group) = 0.9
P(prefer other group) = 0.1
```

The candidate pool includes citizens, Expert, and Jammer. All source nodes have fixed group IDs determined by their initial/source position sign, so elite nodes are embedded in the same group-based opportunity pool rather than attached universally.

Under the paper's standard elite positions:

- Expert: mu = 0 -> non-positive group;
- Jammer: positive underlying position -> positive group.

Fallback to the opposite pool is allowed only if the preferred pool has no remaining eligible source.

Structural source count: exactly 2.

---

### 4. Extended network

**Manuscript specification**

Every citizen has direct access to both elite providers and also two random citizen peers.

**Reconstructed generator**

For every citizen:

```text
A_i = {Expert, Jammer} + 2 random citizen peers
```

Structural source count: exactly 4.

This is the environment with universal top-down access plus horizontal peer communication and therefore the greatest direct structural corrective redundancy among the four baseline regimes.

---

## Critical legacy-code finding

The legacy Mesa implementation does **not** implement Random 2 and Group ID as described in the current manuscript.

In the old `Citizen.pick_counterpart()` logic:

1. `baseline_neighbors` always contains both elite sources;
2. `mode == "random"` then adds a random number of citizen neighbors between 0 and `num_max_citizen_neighbor`;
3. `mode == "group_id_matching"` likewise keeps both elites universally and adds 0..cap citizen neighbors with a 0.9/0.1 within/out-group rule.

Therefore the historical `random` and `group_id_matching` code paths are closer to variable-degree versions of the current **Extended** environment than to the manuscript's sparse Random 2 and Group ID regimes.

This mismatch means archived results cannot be assumed to correspond exactly to the four environments now described in the paper. The corrected model must regenerate the full four-environment evidence.

The reconstruction preserves the old modes only under explicit `legacy_extended_*` aliases so they remain auditable without contaminating the production design.

## Canonical environment names

Production Paper B code now uses:

```text
elite_only
random_2
group_id
extended
```

Older aliases such as `random_peer` and `homophilous_peer` are accepted only for compatibility and normalize to the canonical names.

## Parameters frozen for the baseline design

```text
Random 2 local degree = 2
Group ID local degree = 2
Group ID same-group preference = 0.9
Extended peer degree = 2
Isolated direct elite access = Expert + Jammer
Extended direct elite access = Expert + Jammer
```

Degree or bridging-probability robustness analyses should be implemented as explicit deviations from this frozen baseline rather than silently changing the primary environment definitions.

## Remaining ambiguity

The current manuscript does not provide a more detailed generative rule for Group ID beyond predominantly within-group connections, limited cross-group contact, and embedded elite nodes. The 0.9/0.1 mixing probability is therefore retained from the legacy implementation because it is the only explicit historical quantitative rule for this feature.

No separate `elite_access_probability` is used in the primary Random 2 or Group ID generators.

## Validation targets

CI should verify:

- **elite_only:** each citizen has exactly two sources and both are elites;
- **random_2:** each citizen has exactly two total sources from the full non-ego node pool; elite access is not universal;
- **group_id:** each citizen has exactly two total sources, with aggregate same-group opportunity share close to the 0.9 design target and nonzero cross-group contact;
- **extended:** each citizen has exactly four sources: both elites plus exactly two citizen peers;
- all four source sets are fixed within a simulation run.
