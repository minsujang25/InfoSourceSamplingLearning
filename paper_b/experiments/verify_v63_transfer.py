"""Validate V6.3 transfer and report same-cohort audit/concentration comparisons."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

KEY = ['seed', 'variant', 'suite', 'mode', 'level']
AUDIT = ['requested', 'delivered', 'message_mean', 'sample_mean_variance',
         'delta_post', 'delta_sd_post', 'rank_post']
SEEDS = list(range(31001, 31101))


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def check(x, y):
    np.testing.assert_allclose(x, y, rtol=1e-10, atol=1e-9)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    root, out = a.root, a.out
    out.mkdir(parents=True, exist_ok=True)
    m = json.loads((root / 'TRANSFER_V63.json').read_text())
    assert m['design_id'] == '7e2e4f6724e3'
    assert m['source_commit'] == '55264824e35c02284174e206d86955269d2b0bbc'
    assert m['required_terminal_records'] == 80000
    for f in m['files']:
        path = root / f['name']
        assert path.stat().st_size == f['bytes'] and digest(path) == f['sha256'], path
    r = pd.read_csv(root / 'run_metadata.csv.gz')
    inv = pd.read_csv(root / 'shard_inventory.csv.gz')
    assert len(r) == len(inv) == (1400 if m['optional_frozen_G'] else 1200)
    assert not r.duplicated(KEY).any() and not inv.duplicated(KEY).any()
    assert set(map(tuple, r[KEY].values)) == set(map(tuple, inv[KEY].values))
    assert inv.sha256.str.fullmatch('[0-9a-f]{64}').all()
    assert r.floor_events.eq(0).all() and r.periods.eq(40000).all()
    assert r.groupby(['seed','suite']).initial_state_fingerprint.nunique().eq(1).all()
    assert r.groupby(['seed','suite','level']).structural_fingerprint.nunique().eq(1).all()
    d = pd.read_csv(root / 'first_audit_external.csv.gz')
    c = pd.read_csv(root / 'first_audit_pair_checks.csv.gz')
    ak = KEY + ['citizen_pos', 'source_pos']
    assert not d.duplicated(ak).any() and not c.duplicated(KEY[:-1]).any()
    assert set(d.source_class) == {'expert','jammer'}
    counts = d.groupby(KEY).size().reset_index(name='count')
    assert counts['count'].eq(counts.suite.map({'H':200,'G':110})).all()
    assert set(map(tuple, counts[KEY].values)) == set(map(tuple, r[KEY].values))
    pk = KEY[:-1] + ['citizen_pos', 'source_pos']
    l = d[d.level.eq('low')].set_index(pk).sort_index()
    h = d[d.level.eq('high')].set_index(pk).sort_index()
    assert l.index.equals(h.index)
    eq = pd.DataFrame({f'equal_{k}': l[k].eq(h[k]) | (l[k].isna() & h[k].isna()) for k in AUDIT})
    rec = eq.groupby(level=KEY[:-1]).sum()
    rec['n_external'] = eq.groupby(level=KEY[:-1]).size()
    pd.testing.assert_frame_equal(rec[c.columns.difference(KEY[:-1])].sort_index(), c.set_index(KEY[:-1])[c.columns.difference(KEY[:-1])].sort_index(), check_dtype=False)
    assert rec[[f'equal_{k}' for k in AUDIT[:-1]]].eq(rec.n_external, axis=0).all().all()
    rec.reset_index().groupby(['suite','variant','mode']).sum(numeric_only=True).drop(columns='seed').to_csv(out/'audit_equality.csv')
    n = 0
    maxerr = 0.
    for f in m['files']:
        if not f['name'].endswith('_terminal.json.gz'):
            continue
        with gzip.open(root/f['name'], 'rt') as stream:
            t = pd.DataFrame(json.load(stream))
        assert len(t) == 20000 and not t.duplicated(KEY+['citizen_pos']).any()
        assert t.groupby(KEY).citizen_pos.apply(lambda x: set(x)==set(range(2,102))).all()
        assert t.floor_events.eq(0).all()
        for name in ['expert','jammer']:
            t['Q_'+name+'_channel'] = t.Q.map(lambda q:q[name])
        t['mse_truth'] = t.mu**2
        t['mean_belief'] = t.mu
        t['W_jammer_channel'] = t.W_jammer
        metrics = ['mse_truth','mean_belief','Q_jammer_channel','Q_expert_channel','W_jammer_channel']
        g = t.groupby(KEY)[metrics].mean()
        ref = r.set_index(KEY).loc[g.index, metrics]
        check(g, ref)
        maxerr = max(maxerr, float(np.max(np.abs(g.to_numpy()-ref.to_numpy()))))
        n += len(t)
    assert n == (100000 if m['optional_frozen_G'] else 80000)
    # Same 100 seeds: compare measured W, not uniform opportunity U^A.
    rng = np.random.default_rng(20261125)
    ix = rng.integers(0,100,(10000,100))
    def summarize(x):
        y = x[ix].mean(axis=1)
        return dict(mean=float(x.mean()), lo=float(np.quantile(y,.025)), hi=float(np.quantile(y,.975)))
    rows = []
    for suite in ['G','H']:
        for metric in ['W_peer_incoming_hhi','W_peer_incoming_top5_share']:
            contrasts = {}
            for (variant,mode), b in r[r.suite.eq(suite)].groupby(['variant','mode']):
                z = b.pivot(index='seed', columns='level', values=metric).reindex(SEEDS)
                assert z.notna().all().all()
                k = variant+'_'+mode
                for kind,x in [('low',z.low),('high',z.high),('contrast',z.high-z.low)]:
                    rows.append(dict(suite=suite, metric=metric, policy=k, kind=kind, **summarize(x.to_numpy())))
                contrasts[k] = (z.high-z.low).to_numpy()
            for k in ['neutral_adaptive','baseline_frozen']:
                if k in contrasts:
                    rows.append(dict(suite=suite,metric=metric,policy='baseline_adaptive_minus_'+k,kind='contrast_interaction',**summarize(contrasts['baseline_adaptive']-contrasts[k])))
    pd.DataFrame(rows).to_csv(out/'actual_W_concentration.csv',index=False)
    (out/'transfer_validation.json').write_text(json.dumps(dict(files_verified=len(m['files']), runs=len(r), terminal_records=n, first_audit_rows=len(d), first_audit_matched_rows=len(l), first_audit_pairs=len(c), max_run_mean_difference=maxerr, manifest_sha256=digest(root/'TRANSFER_V63.json')), indent=2)+'\n')
    print((out/'transfer_validation.json').read_text())


if __name__ == '__main__':
    main()
