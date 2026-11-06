"""Integrated gradients on the same queries: the standard fix, tested.

NBFNet's explanation reads the gradient of the score at the model's operating
point, every edge weight at one. Integrated gradients (Sundararajan et al.,
ICML 2017) average that gradient along a straight path from a baseline to the
input, which is the textbook remedy when a response is flat or saturated near
the operating point, the shape the neuromodulation experiment finds. Here the
baseline is the graph with every message switched off and the path scales all
edge weights together from 0 to 1:

    IG_e = integral over a in [0, 1] of dp/dw_e at w = a * 1

approximated with 16 midpoint steps, so the attributions sum to
p(full graph) - p(no messages), which each query checks. The gradients are the
deployed model's (taken on the direction-corrected path), and the candidate
paths are the corrected explanation's, so IG is compared with the gradient it
is meant to improve on rather than with the released code's.

From one run per query this gives two things:
  paths     each of the explanation's 20 candidate paths scored by the sum of
            IG over its edges, to rank against the measured deletion drops
  channels  the IG of each relation channel, the sum over its edges (scaling
            every channel together is the same path), to set against the
            silencing effects measured by pathfaith.neuromod

Usage:
    python -m pathfaith.ig --dataset CoDExSmall --n 200
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import time
import warnings

import torch

from pathfaith.loader import build
from pathfaith import attribution as A
from pathfaith.experiment import path_edges
from pathfaith import neuromod as NM

warnings.filterwarnings('ignore')
STEPS = 16


def run_query(bundle, data, rec, steps=STEPS, released=False):
    h, r, t = rec['h'], rec['r'], rec['t']
    E = int(data.num_edges)
    ch, R = NM.channel_of(data)
    t0 = time.time()

    # the deployed model's own gradient path; --released differentiates the
    # code path the released explanation uses, to show it fails completeness
    gdata = data if released else A.flipped(data)
    p1 = float(NM.prob(bundle, data, h, r, t)[0])
    p0 = float(NM.prob(bundle, data, h, r, t, torch.zeros(E))[0])

    total = torch.zeros(E)
    for k in range(steps):
        a = (k + 0.5) / steps
        w = torch.full((E,), a, requires_grad=True)
        p, _ = NM.prob(bundle, gdata, h, r, t, w, grad=True)
        p.backward()
        total += w.grad.detach()
    ig = total / steps

    with contextlib.redirect_stdout(io.StringIO()):
        exp = A.explain_gradient(bundle, data, h, r, t, topk=20, num_beam=20, corrected=True)
    paths = exp.paths[:len(rec['true_drops'])]
    path_ig = [float(ig[path_edges(data, p)].sum()) if p else 0.0 for p in paths]
    w_gap = max((abs(a - b) for a, b in zip(exp.weights, rec['grad_weights'])), default=0.0)
    chan_ig = torch.zeros(R).index_add_(0, ch, ig)

    return dict(kind='query', h=h, r=r, t=t, base_prob=p1, p_empty=p0,
                ig_sum=float(ig.sum()), completeness_gap=abs(float(ig.sum()) - (p1 - p0)),
                path_ig=path_ig, grad_weights=rec['grad_weights'][:len(paths)],
                true_drops=rec['true_drops'][:len(paths)], weights_gap=w_gap,
                n_paths_match=len(paths) == len(rec['true_drops']),
                channel_ig=[round(float(x), 8) for x in chan_ig],
                steps=steps, seconds=round(time.time() - t0, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='CoDExSmall')
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--steps', type=int, default=STEPS)
    ap.add_argument('--out', default=None)
    ap.add_argument('--released', action='store_true',
                    help="integrate the released code path's gradient instead (a completeness check)")
    args = ap.parse_args()

    NM._install()
    from pathfaith.analyze import load
    lc = args.dataset.lower()
    _, recs = load(f'results/paths_{lc}_fixed_seed0.jsonl')   # corrected explanations
    recs = recs[:args.n]
    out = args.out or (f'results/ig_{lc}_released_seed0.jsonl' if args.released
                       else f'results/ig_{lc}_seed0.jsonl')
    done, kept = set(), []
    if os.path.exists(out):
        for line in open(out):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get('kind') == 'query' and (d['h'], d['r'], d['t']) not in done:
                done.add((d['h'], d['r'], d['t'])); kept.append(d)
    b = build(args.dataset, 'ultra_4g.pth')
    data = b.test_data
    print(f'{b.name}: {len(recs)} benchmark queries, {len(kept)} already done', flush=True)
    t0 = time.time()
    with open(out, 'w') as f:
        f.write(json.dumps(dict(kind='meta', dataset=b.name, ckpt=b.ckpt, steps=args.steps,
                                gradient='released' if args.released else 'corrected',
                                baseline='all edge weights zero')) + '\n')
        for d in kept:
            f.write(json.dumps(d) + '\n')
        f.flush()
        for i, rec in enumerate(x for x in recs if (x['h'], x['r'], x['t']) not in done):
            res = run_query(b, data, rec, steps=args.steps, released=args.released)
            f.write(json.dumps(res) + '\n'); f.flush()
            print(f'[{len(kept) + i + 1}/{len(recs)}] h={rec["h"]} r={rec["r"]} t={rec["t"]} '
                  f'completeness gap {res["completeness_gap"]:.1e}  weights gap '
                  f'{res["weights_gap"]:.1e}  {res["seconds"]:.0f}s', flush=True)
    print(f'wrote {out} in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
