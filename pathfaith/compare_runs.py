"""Check that a fresh run reproduces the recorded benchmark, query by query."""
from __future__ import annotations

import argparse
import json
import sys

SCALARS = ['base_prob', 'comp_topk', 'comp_random', 'suff_topk', 'bottleneck_drop',
           'rewire_follow', 'rewire_base_drop']
EXACT = ['base_rank', 'n_paths', 'n_distinct_edges', 'comp_rank_after']


def rows(path):
    out = {}
    for line in open(path):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get('kind') == 'query':
            out[(d['h'], d['r'], d['t'])] = d
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--new', required=True)
    ap.add_argument('--ref', required=True)
    ap.add_argument('--tol', type=float, default=1e-4)
    args = ap.parse_args()

    new, ref = rows(args.new), rows(args.ref)
    shared = [k for k in new if k in ref]
    if not shared:
        print('no queries in common: the new run drew different queries than the record')
        return 1
    ok, worst = 0, 0.0
    for k in shared:
        a, b = new[k], ref[k]
        if 'channels' in a:          # a neuromodulation record
            diffs = [abs(a['base_prob'] - b['base_prob'])]
            diffs += [abs(x - y) for x, y in zip(a['grad_all'], b['grad_all'])]
            ca = {(c['c'], c['role']): c for c in a['channels']}
            cb = {(c['c'], c['role']): c for c in b['channels']}
            same = ca.keys() == cb.keys() and a['base_rank'] == b['base_rank']
            for key in ca.keys() & cb.keys():
                for dose, (pa, _) in ca[key]['doses'].items():
                    diffs.append(abs(pa - cb[key]['doses'][dose][0]))
        else:
            diffs = [abs(a[f] - b[f]) for f in SCALARS if f in a and f in b]
            n = min(len(a['true_drops']), len(b['true_drops']))
            diffs += [abs(x - y) for x, y in zip(a['true_drops'][:n], b['true_drops'][:n])]
            same = all(a.get(f) == b.get(f) for f in EXACT) and len(a['true_drops']) == len(b['true_drops'])
        d = max(diffs) if diffs else 0.0
        worst = max(worst, d)
        if same and d <= args.tol:
            ok += 1
        else:
            print(f'  differs: h={k[0]} r={k[1]} t={k[2]}  largest gap {d:.2e}'
                  + ('' if same else '  (counts or ranks differ)'))
    print(f'{len(new)} queries re-run; {len(shared)} are in the record; '
          f'{ok} of {len(shared)} reproduce every recorded number to within {args.tol:g} '
          f'(largest gap {worst:.1e})')
    return 0 if ok == len(shared) == len(new) else 1


if __name__ == '__main__':
    sys.exit(main())
