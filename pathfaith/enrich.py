"""Add graph-structure controls to a finished results file.

The central finding needs one obvious objection ruled out. If deleting a single
edge destroys the prediction, that could mean the model leans on one fact, or
it could mean the tail entity simply has one edge and deleting it strands the
node. This computes the tail's degree for every query, so the two can be told
apart, and re-writes the rows with those fields added.

Usage:
    python -m pathfaith.enrich --dataset WN18RR --file results/paths_wn18rr_seed0.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from pathfaith.loader import build
from pathfaith import attribution as A


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='WN18RR')
    ap.add_argument('--ckpt', default='ultra_4g.pth')
    ap.add_argument('--file', required=True)
    args = ap.parse_args()

    # Rewriting a results file while run_experiment still has it open corrupts
    # whatever that process writes next, which is exactly what happened once.
    # Refuse if the file is still growing.
    size = os.path.getsize(args.file)
    time.sleep(3)
    if os.path.getsize(args.file) != size:
        print(f'{args.file} is still being written; skipping enrichment')
        return

    bundle = build(args.dataset, args.ckpt)
    data = bundle.test_data
    src, dst = data.edge_index[0], data.edge_index[1]
    in_deg = torch.bincount(dst, minlength=int(data.num_nodes))
    out_deg = torch.bincount(src, minlength=int(data.num_nodes))

    rows, meta, skipped = [], None, 0
    with open(args.file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1      # a torn line from an interrupted write
                continue
            if d.get('kind') == 'meta':
                meta = d
            else:
                rows.append(d)
    if skipped:
        print(f'  dropped {skipped} unreadable line(s)')

    # WN18RR keeps several symmetric relations, so for many test triples the
    # reverse fact is sitting in the training graph and a one-hop lookup
    # answers the query. That is not leakage of the test edge itself, but it
    # changes what an explanation is explaining, so each query is flagged and
    # the central results are reported split on it.
    half = int(data.num_relations) // 2
    nn, nr = int(data.num_nodes), int(data.num_relations)
    key = set(((data.edge_index[0] * nn + data.edge_index[1]) * nr
               + data.edge_type).tolist())

    for d in rows:
        t, h = d['t'], d['h']
        r = d['r']
        d['forward_in_graph'] = bool(((h * nn + t) * nr + r) in key)
        r_inv = r + half if r < half else r - half
        d['reverse_in_graph'] = bool(((h * nn + t) * nr + r_inv) in key)
        d['tail_in_degree'] = int(in_deg[t])
        d['tail_out_degree'] = int(out_deg[t])
        d['head_out_degree'] = int(out_deg[h])
        # a tail reachable by only one edge would break under any deletion
        d['tail_is_leaf'] = bool(int(in_deg[t]) <= 1)
        # how many edges in the graph could even reach t in one hop
        d['tail_degree'] = int(in_deg[t]) + int(out_deg[t])

    tmp = args.file + '.tmp'
    with open(tmp, 'w') as f:
        if meta:
            meta['enriched'] = True
            # older runs did not record this; the dataset is loaded here anyway
            meta.setdefault('num_relations', int(data.num_relations))
            f.write(json.dumps(meta) + '\n')
        for d in rows:
            f.write(json.dumps(d) + '\n')
    os.replace(tmp, args.file)

    deg = [d['tail_degree'] for d in rows]
    leaf = sum(d['tail_is_leaf'] for d in rows)
    rev = sum(d['reverse_in_graph'] for d in rows)
    fwd = sum(d['forward_in_graph'] for d in rows)
    print(f'{args.dataset}: {len(rows)} queries enriched; '
          f'median tail degree {sorted(deg)[len(deg) // 2]}, '
          f'min {min(deg)}, leaf tails {leaf}; '
          f'reverse fact present {rev}, forward edge present {fwd}')


if __name__ == '__main__':
    main()
