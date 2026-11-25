"""Read-only V6.5.2B extraction from frozen V62 baseline adaptive shards."""
import argparse
import csv
import gzip
import hashlib
import itertools
import json
import math
import tempfile
import zipfile
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

DESIGN = '7e2e4f6724e3'
MANIFEST_SHA = '3a83e3b7ff4af3ceeeaeda34bbd47633c9df07228089e25f6d57a56394cfff61'
SOURCE_COMMIT = '55264824e35c02284174e206d86955269d2b0bbc'
SEEDS = list(range(31001, 31101))
CELLS = [('baseline', s, l, 'adaptive') for s in ('H', 'G') for l in ('low', 'high')]
CHECKPOINTS = (0, 1, 2, 3, 6, 11, 26, 51, 100, 200, 400)
KEY = ['seed', 'variant', 'suite', 'level', 'mode']
WINDOW = ['citizen_pos', 'window', 'source_class', 'opportunities', 'expected_requests', 'requests', 'delivered', 'Q', 'B', 'Lnum', 'snapshot_B', 'snapshot_Lnum', 'snapshot_Q', 'observations', 'gain', 'signed_update', 'raw_draw_Lnum', 'prior_precision']
AUDIT = ['citizen_pos', 'source_pos', 'source_class', 'requested', 'delivered', 'message_mean', 'sample_mean_variance', 'snapshot_mu', 'delta_pre', 'delta_sd_pre', 'delta_post', 'delta_sd_post', 'rank_pre', 'rank_post']
META = ['initial_mu', 'initial_sd', 'group', 'role']
CHECK = ['citizen_pos', 'h', 'mu', 'sd', 'rank', 'latent_rank']
NAMES = ('baseline_HG_peer_type_windows.csv.gz', 'baseline_HG_initial_audit_all_sources.csv.gz', 'baseline_HG_checkpoints.csv.gz', 'shard_inventory.csv.gz')
FIELDS = (KEY + WINDOW, KEY + AUDIT + META, KEY + CHECK, KEY + ['sha256', 'file', 'bytes', 'windows', 'initial_audit', 'checkpoints', 'floor_events', 'history_sha256'])


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def close(a, b):
    return math.isfinite(a) and math.isfinite(b) and abs(a-b) <= 1e-8 + 1e-10*max(abs(a), abs(b))


def unique(rows, fields):
    return len({tuple(r[k] for k in fields) for r in rows}) == len(rows)


def check_shard(o, seed, cell):
    v, suite, level, mode = cell
    require((o['design_id'], o['phase'], o['seed'], o['variant'], o['suite'], o['level'], o['mode'], o['horizon']) == (DESIGN, 'production', seed, v, suite, level, mode, 400), 'Shard identity')
    require(o['validation']['periods'] == 40000 and o['validation']['floor_events'] == 0, 'Coverage/floor')
    t = {r['citizen_pos']: r for r in o['terminal']}
    require(len(o['terminal']) == len(t) == 100 and set(t) == set(range(2, 102)), 'Citizen coverage')
    if suite == 'H':
        require(sum(r['group'] == 1 for r in t.values()) == sum(r['group'] == -1 for r in t.values()) == 50, 'H groups')
    else:
        require(sum(r['role'] == 'gateway' for r in t.values()) == 10 and sum(r['role'] == 'non_gateway' for r in t.values()) == 90, 'G roles')
    w, a, c = (o[k] for k in ('windows', 'initial_audit', 'checkpoints'))
    require(unique(w, ['citizen_pos', 'window', 'source_class']) and unique(a, ['citizen_pos', 'source_pos']) and unique(c, ['citizen_pos', 'h']), 'Duplicate keys')
    require(len(a) == (400 if suite == 'H' else 310) and len(c) == 1100, 'Audit/checkpoint counts')
    by_w, by_a, by_c = defaultdict(list), defaultdict(list), defaultdict(dict)
    for r in w:
        require(r['citizen_pos'] in t and r['window'] in ('early', 'middle', 'late'), 'Window key')
        require(all(math.isfinite(r[k]) for k in WINDOW[3:]) and all(r[k] >= 0 for k in ('Q', 'Lnum', 'snapshot_Q', 'opportunities', 'requests', 'delivered', 'observations')), 'Window values')
        by_w[r['citizen_pos']].append(r)
    for r in a:
        require(r['citizen_pos'] in t and r['requested'] == r['delivered'] > 0 and r['sample_mean_variance'] > 0, 'Audit delivery')
        require(all(math.isfinite(r[k]) for k in ('message_mean', 'sample_mean_variance', 'delta_pre', 'delta_sd_pre', 'delta_post', 'delta_sd_post')), 'Audit values')
        by_a[r['citizen_pos']].append(r)
    for r in c:
        require(r['citizen_pos'] in t and math.isfinite(r['mu']) and math.isfinite(r['sd']) and r['sd'] > 0, 'Checkpoint values')
        by_c[r['citizen_pos']][r['h']] = r
    max_q, max_b = 0., 0.
    for pos, r in t.items():
        peers = r['peer_ids']
        require(len(set(peers)) == len(peers) == 2 and pos not in peers and all(p in t for p in peers), 'Peer IDs')
        source = {1: 'jammer'}
        if suite == 'H' or r['role'] == 'gateway':
            source[0] = 'expert'
        for p in peers:
            source[p] = ('same_peer' if t[p]['group'] == r['group'] else 'other_peer') if suite == 'H' else ('gateway_peer' if t[p]['role'] == 'gateway' else 'non_gateway_peer')
        require(r['floor_events'] == 0 and r['audit_count'] + r['theta_count'] == 400, 'Terminal counters')
        aa, ww, cc = by_a[pos], by_w[pos], by_c[pos]
        require({x['source_pos']: x['source_class'] for x in aa} == source, 'Audit source map')
        d = len(source)
        require(sum(x['requested'] for x in aa) == 20 and sorted(x['rank_post'] for x in aa) == list(range(1, d+1)), 'Audit ranks/budget')
        for x in aa:
            var = r['initial_sd']**2 + x['sample_mean_variance']; pv = x['delta_sd_pre']**2
            post = (var*x['delta_pre'] + pv*(x['message_mean']-r['initial_mu']))/(var+pv)
            require(close(post, x['delta_post']), 'Audit score identity')
        require(set(cc) == set(CHECKPOINTS), 'Checkpoint grid')
        require(all(1 <= x['rank'] <= d and 1 <= x['latent_rank'] <= d for x in cc.values()), 'Checkpoint ranks')
        require(all(close(cc[h]['mu'], r['initial_mu']) and close(cc[h]['sd'], r['initial_sd']) for h in (0, 1)), 'Initial states')
        require(cc[1]['rank'] == r['initial_rank'] and cc[400]['rank'] == r['rank'] and close(cc[400]['mu'], r['mu']) and close(cc[400]['sd'], r['sd']), 'Endpoint states')
        require(all(x['source_class'] in source.values() for x in ww), 'Window source class')
        require(sum(x['opportunities'] for x in ww) == d*r['theta_count'] and sum(x['requests'] for x in ww) == 20*r['theta_count'] and all(x['requests'] == x['delivered'] for x in ww), 'Substantive exposure')
        for k in ('expert', 'jammer', 'peer'):
            selected = [x for x in ww if x['source_class'] == k or (k == 'peer' and x['source_class'].endswith('_peer'))]
            q, b = (math.fsum(x[f] for x in selected) for f in ('Q', 'B'))
            require(close(q, r['Q'][k]) and close(b, r['B_raw'][k]), 'Window Q/B reconciliation')
            max_q = max(max_q, abs(q-r['Q'][k])); max_b = max(max_b, abs(b-r['B_raw'][k]))
        require(close(r['initial_phi']+math.fsum(r['Q'].values()), r['phi']) and close(math.fsum(r['B_raw'].values())/r['phi'], r['mu']), 'Posterior reconciliation')
    meta = dict(zip(KEY, (seed, *cell)))
    return ([dict(meta, **r) for r in w], [dict(meta, **r, **{k: t[r['citizen_pos']][k] for k in META}) for r in a], [dict(meta, **r) for r in c], max_q, max_b)


def extract(root, out):
    root, out = root.resolve(), out.resolve()
    require(out.suffix == '.zip' and not out.exists() and not out.is_relative_to(root), 'Use a new ZIP outside source root')
    manifest = root/'MANIFEST.json'
    require(digest(manifest) == MANIFEST_SHA, 'Wrong manifest hash')
    m = json.loads(manifest.read_text())
    require(m['design_id'] == DESIGN and m['seeds'] == SEEDS and m['horizon'] == 400 and all(list(c) in m['grid'] for c in CELLS), 'Manifest design')
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent, prefix='.v652b-') as td:
        dest = Path(td); counts = [0]*4; max_q = max_b = 0.
        with ExitStack() as stack:
            writers = [csv.DictWriter(stack.enter_context(gzip.open(dest/n, 'wt', encoding='utf-8', newline='')), fieldnames=f) for n, f in zip(NAMES, FIELDS)]
            for writer in writers: writer.writeheader()
            for seed, cell in itertools.product(SEEDS, CELLS):
                stem = '__'.join([str(seed), *cell]); p = root/'production'/(stem+'.json.gz'); sidecar = p.with_name(stem+'.sha256')
                sha = digest(p); require(sha == sidecar.read_text().strip(), 'Shard checksum: '+stem)
                with gzip.open(p, 'rt', encoding='utf-8') as f: o = json.load(f)
                w, a, c, q, b = check_shard(o, seed, cell)
                meta = dict(zip(KEY, (seed, *cell)))
                inv = dict(meta, sha256=sha, file=p.name, bytes=p.stat().st_size, windows=len(w), initial_audit=len(a), checkpoints=len(c), floor_events=0, history_sha256=o.get('history_sha256'))
                for j, data in enumerate((w, a, c, [inv])):
                    writers[j].writerows(data); counts[j] += len(data)
                max_q, max_b = max(max_q, q), max(max_b, b)
                if counts[3] % 20 == 0: print(f'{counts[3]}/400 validated', flush=True)
        require(counts[1:] == [142000, 440000, 400], 'Incomplete extraction')
        files = [dict(name=n, rows=k, bytes=(dest/n).stat().st_size, sha256=digest(dest/n), fields=f) for n, k, f in zip(NAMES, counts, FIELDS)]
        transfer = dict(design_id=DESIGN, source_commit=SOURCE_COMMIT, source_manifest_sha256=MANIFEST_SHA, extractor_sha256=digest(Path(__file__)), grid=CELLS, seeds=SEEDS, runs=400, citizens=40000, floor_events=0, max_Q_residual=max_q, max_B_residual=max_b, new_simulations=0, files=files)
        (dest/'TRANSFER_V652B.json').write_text(json.dumps(transfer, indent=2)+'\n', encoding='utf-8')
        temp_zip = dest/'transfer.zip'
        with zipfile.ZipFile(temp_zip, 'w', compression=zipfile.ZIP_STORED) as z:
            for n in (*NAMES, 'TRANSFER_V652B.json'): z.write(dest/n, n)
        with zipfile.ZipFile(temp_zip) as z: require(z.testzip() is None, 'ZIP integrity')
        require(not out.exists(), 'Output exists'); temp_zip.rename(out)
    print(f'{out}\nSHA256: {digest(out)}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    args = p.parse_args(); extract(args.root, args.out)
