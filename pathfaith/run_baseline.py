"""Reproduce the frozen checkpoint's link-prediction numbers.

Nothing downstream is worth reading unless the model being explained is the
model the authors published, so this runs first. The protocol is ULTRA's own
`script/run.py::test`: score every entity as the missing side, filter out the
other true triples, and rank the gold one. Head and tail directions are both
scored, as in the paper.

Usage:
    python -m pathfaith.run_baseline --dataset FB15k237 --n 200 --seed 0
"""
from __future__ import annotations

import argparse
import json
import time

import torch
from torch_geometric.data import Data

from pathfaith.loader import build, n_params


def filtering_graph(bundle) -> Data:
    """All known true triples, used to filter false negatives out of the ranking."""
    tr, va, te = bundle.train_data, bundle.valid_data, bundle.test_data
    idx = torch.cat([tr.target_edge_index, va.target_edge_index, te.target_edge_index], dim=1)
    typ = torch.cat([tr.target_edge_type, va.target_edge_type, te.target_edge_type])
    return Data(edge_index=idx, edge_type=typ, num_nodes=tr.num_nodes)


@torch.no_grad()
def evaluate(bundle, n: int = 200, seed: int = 0, batch_size: int = 4,
             verbose: bool = True) -> dict:
    from ultra import tasks

    data = bundle.test_data
    filt = filtering_graph(bundle)
    triples = torch.cat([data.target_edge_index,
                         data.target_edge_type.unsqueeze(0)]).t()

    g = torch.Generator().manual_seed(seed)
    if n and n < len(triples):
        sel = torch.randperm(len(triples), generator=g)[:n]
        triples = triples[sel]

    ranks = []
    t0 = time.time()
    for i in range(0, len(triples), batch_size):
        batch = triples[i:i + batch_size]
        t_batch, h_batch = tasks.all_negative(data, batch)
        t_pred = bundle.model(data, t_batch)
        h_pred = bundle.model(data, h_batch)
        t_mask, h_mask = tasks.strict_negative_mask(filt, batch)
        pos_h, pos_t, _ = batch.t()
        ranks.append(tasks.compute_ranking(t_pred, pos_t, t_mask))
        ranks.append(tasks.compute_ranking(h_pred, pos_h, h_mask))
        if verbose and (i // batch_size) % 5 == 0:
            done = min(i + batch_size, len(triples))
            print(f'  {done}/{len(triples)} triples, {time.time() - t0:.0f}s',
                  flush=True)

    rank = torch.cat(ranks).float()
    # ranks alternate tail batch, head batch; keep the tail-only view too, since
    # ULTRA's README lists a few graphs (NELL23k among them) as tail-only even
    # though its paper scores NELL23k in both directions
    tail = torch.cat(ranks[0::2]).float()
    out = {
        'dataset': bundle.name,
        'checkpoint': bundle.ckpt,
        'n_triples': int(len(triples)),
        'n_rankings': int(len(rank)),
        'seed': seed,
        'mr': float(rank.mean()),
        'mrr': float((1.0 / rank).mean()),
        'hits@1': float((rank <= 1).float().mean()),
        'hits@3': float((rank <= 3).float().mean()),
        'hits@10': float((rank <= 10).float().mean()),
        'mrr_tail_only': float((1.0 / tail).mean()),
        'hits@10_tail_only': float((tail <= 10).float().mean()),
        'seconds': round(time.time() - t0, 1),
        'sec_per_query': round((time.time() - t0) / max(1, len(rank)), 3),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='FB15k237')
    ap.add_argument('--ckpt', default='ultra_4g.pth')
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    bundle = build(args.dataset, args.ckpt)
    print(f'{bundle.name}: {bundle.num_nodes} entities, '
          f'{bundle.num_relations} relations, {bundle.num_edges} message edges')
    print(f'checkpoint {bundle.ckpt}: {n_params(bundle.model):,} parameters, frozen')

    res = evaluate(bundle, n=args.n, seed=args.seed, batch_size=args.batch_size)
    print(json.dumps(res, indent=2))
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(res, f, indent=2)


if __name__ == '__main__':
    main()
