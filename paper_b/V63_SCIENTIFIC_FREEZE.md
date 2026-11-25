# V6.3 scientific analysis freeze

W2 design b8f139d546be, run checkout b1f62298a63f8c3a221ecaa807e400b5fbf52387; source design 7e2e4f6724e3. W2 primary ZIP SHA-256 b3be02e7cb942c5f20fa71fc4239cd08f33914fa47339b9b411824a8ff77a14e. Inventory: 48 numeric tau metadata records, zero candidates or scan errors (manifest scan only). All 800 scientific runs completed; archived local technical checks report 8 short preflight cells, 8 full tau1 replays, 8 full W2 parity cells, zero floor events.

Independent post-processing: verify_v63_w2.py, commit b4e694565cd1d6f258d646f1cebb5e6626d0e9a8, blob 0b7f41ff85f120d4e767b576d7597cff2ca2f97c, SHA-256 8b2200b8e4abe174f192cd72bb1431924414b3e5e5bcc6176c1cb225d400b3ea (matches executed file). Stage both W2 indices as INDEX.json and TRANSFER_INDEX(1).json, its ZIP, the eight V6.2 primary ZIPs and the V6.3 extraction shard_inventory.csv.gz in ROOT. Run from repo root:

```bash
OPENBLAS_NUM_THREADS=2 python -m paper_b.experiments.verify_v63_w2 --root "$ROOT" --out "$OUT"
```

Verified all 105 ZIP members and 800 reference shard hash links against earlier extraction; 105,600 overlapping tau1 seed-level statistics and 280,800 derived seed identities exactly match. 530,400 seed statistics, 5,304 independently recomputed means and 10,000-draw whole-seed bootstrap intervals, maximum absolute discrepancy 9.095e-13. Run-validation maxima: precision 4.9113e-11, state 6.3061e-14. VALIDATION.json SHA-256 87a0792c8cfa167de2391e7c731abc8a41d9e0d32a35a586bc84a027d91abf8d; recomputed_summary.csv.gz SHA-256 b864a29b09b10884ed500080b08734f7403ad0a69b036227b9d7d42be500ce9f. These checks do not independently replay W2 trajectories or inspect raw W2 shards. Preserve those locally for replication.

## Frozen focal results (high minus low; 95% whole-seed MC intervals)

| Scope | Policy | tau1 MSE | tau2 MSE | Paired tau2 minus tau1 |
|---|---|---:|---:|---:|
| H aligned | Adaptive | 1.006565 [.825598,1.191388] | .876016 [.729729,1.029905] | -.130549 [-.245804,-.019018] |
| H aligned | Neutral | .000260 [-.000512,.001064] | .000145 [-.000070,.000369] | -.000115 [-.000704,.000446] |
| G all | Adaptive | -1.137603 [-1.269447,-1.013823] | -.939977 [-1.061871,-.822296] | .197626 [.104315,.289272] |
| G all | Neutral | -1.193321 [-1.195450,-1.191199] | -.452138 [-.452948,-.451329] | .741182 [.739850,.742528] |

Adaptive-minus-neutral MSE interaction: H tau1 1.006305, tau2 .875871 [.729637,1.029660]; change -.130433 [-.245602,-.019044]. G tau1 .055718 [-.076048,.178724], tau2 -.487839 [-.609868,-.370103]; change -.543557 [-.637240,-.451916]. H adaptive Q contrast -1007.5868 -> -975.5716; G +595.4363 -> +818.5303. Neutral direct Q contrast exactly zero under the implemented pairing at both weights. H population tau2 +.060675 [-.044739,.167069], not a population-wide harm claim. G adaptive absolute MSE rises under tau2 in both networks (13.627818 -> 14.363432 low; 12.490216 -> 13.423455 high). G neutral also worsens in both. Thus relative benefits and absolute accuracy must remain separate.

Neutral role-mean approximation generalizes to weights w_L = [1-(1-1/L)^20]/tau^2. Solve (10+2w_4)g-2w_4 n=20; -k w_3 g+(20/3+k w_3)n=80/3, k=1 low or 2 high. Population MSE approximation .1g^2+.9n^2. At tau2: low 14.3453106034, high 13.8923635787, contrast -.4529470248; simulated 14.345338,13.893200, contrast -.452138. H homogeneous midpoint remains 2. This is a late-state approximation, not a convergence proof. Existing frozen neutral model was not refitted.

## Scientific decision

The H aligned adverse contrast survives reduced assigned peer precision, with attenuation. G's adaptive and neutral accuracy contrasts both remain negative but the neutral benefit attenuates more, making the total policy interaction negative at tau2. The conclusion that G can improve under neutral use survives; any stronger claim of negligible rank-policy contribution in G must be restricted to tau1. Neither interaction isolates credibility from concentration. The tau intervention is a weight sensitivity, not a provenance or dependence correction. No new parameter levels or networks.

Scientific grid complete: 21,320 previously listed distinct runs + 800 W2 = 22,120; do not double-count references or technical replays. Passive closure remains as documented in V63_PASSIVE_CLOSURE.md and prior auxiliary archive. No further scientific simulations or outcome-dependent sensitivity search. Manuscript revision and final evidence integration next; executable replication and anonymous release only after submission-ready manuscript freeze. Retain full-precision summaries, audit outputs, raw shards, and immutable commits; do not publish private development records or identifying archive paths.
