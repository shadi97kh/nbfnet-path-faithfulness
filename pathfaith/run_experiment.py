"""Sweep queries and record every condition, one JSON line per query."""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time

import torch

from pathfaith.loader import build
from pathfaith.experiment import prob_and_rank, run_query


def pick_queries(bundle, n: int, seed: int, max_rank: int = 10):
    """Test triples the frozen model ranks within `max_rank`, in a fixed order."""
    d = bundle.test_data
    tri = torch.cat([d.target_edge_index, d.target_edge_type.unsqueeze(0)]).t()
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(tri), generator=g)

    picked, scanned = [], 0
    for i in order.tolist():
        h, t, r = (int(x) for x in tri[i])
        _, rank, _ = prob_and_rank(bundle, d, h, r, t)
        scanned += 1
        if rank <= max_rank:
            picked.append((h, r, t, rank))
        if len(picked) >= n:
            break
    return picked, scanned


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='WN18RR')
    ap.add_argument('--ckpt', default='ultra_4g.pth')
    ap.add_argument('--n', type=int, default=60)
    ap.add_argument('--pool', type=int, default=20)
    ap.add_argument('--topk', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max-rank', type=int, default=10)
    ap.add_argument('--out', default=None)
    ap.add_argument('--corrected', action='store_true',
                    help='explain with the direction-corrected gradient (see attribution.flipped)')
    ap.add_argument('--path-controls', type=int, default=5,
                    help='per-path random-deletion nulls; not used by any reported number')
    args = ap.parse_args()

    bundle = build(args.dataset, args.ckpt)
    out = args.out or f'results/paths_{args.dataset.lower()}_seed{args.seed}.jsonl'
    os.makedirs(os.path.dirname(out), exist_ok=True)

    print(f'{bundle.name} | {bundle.num_nodes} entities | {bundle.num_edges} edges '
          f'| ckpt {bundle.ckpt}', flush=True)
    t0 = time.time()
    queries, scanned = pick_queries(bundle, args.n, args.seed, args.max_rank)
    print(f'selected {len(queries)} queries ranked <= {args.max_rank} '
          f'out of {scanned} scanned, in {time.time() - t0:.0f}s', flush=True)

    done, kept, meta = set(), [], None
    if os.path.exists(out):
        for line in open(out):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get('kind') == 'meta':
                meta = d
            elif d.get('kind') == 'query':
                key = (d['h'], d['r'], d['t'])
                if key not in done:
                    done.add(key)
                    kept.append(d)
        print(f'resuming: {len(kept)} queries already on disk', flush=True)

    new_meta = dict(kind='meta', dataset=bundle.name, ckpt=bundle.ckpt,
                    pool=args.pool, topk=args.topk, seed=args.seed,
                    n_requested=args.n, n_selected=len(queries),
                    n_scanned=scanned, max_rank=args.max_rank,
                    num_nodes=bundle.num_nodes, num_edges=bundle.num_edges,
                    num_relations=bundle.num_relations,
                    explainer='corrected' if args.corrected else 'released')
    if meta and meta.get('explainer', 'released') != new_meta['explainer']:
        raise SystemExit(f'{out} holds {meta.get("explainer", "released")} explanations; '
                         'refusing to mix explainers in one file')
    if meta and (meta.get('pool'), meta.get('topk')) != (args.pool, args.topk):
        raise SystemExit(f'{out} was made with pool={meta.get("pool")} '
                         f'topk={meta.get("topk")}; refusing to mix protocols')

    with open(out, 'w') as f:
        f.write(json.dumps(new_meta) + '\n')
        for d in kept:
            f.write(json.dumps(d) + '\n')
        f.flush()

        todo = [q for q in queries if (q[0], q[1], q[2]) not in done]
        for i, (h, r, t, _) in enumerate(todo):
            res = run_query(bundle, bundle.test_data, h, r, t,
                            pool=args.pool, topk=args.topk, seed=args.seed,
                            path_controls=args.path_controls, corrected=args.corrected)
            row = dataclasses.asdict(res)
            row['kind'] = 'query'
            f.write(json.dumps(row) + '\n')
            f.flush()
            print(f'[{len(kept) + i + 1}/{len(queries)}] h={h} r={r} t={t} '
                  f'base_rank={res.base_rank} comp={res.comp_topk:.4f} '
                  f'bottleneck={res.bottleneck_frac:.2f} '
                  f'{res.seconds:.0f}s', flush=True)

    print(f'wrote {out} in {time.time() - t0:.0f}s total', flush=True)


if __name__ == '__main__':
    main()
