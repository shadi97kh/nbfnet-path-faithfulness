"""The message-direction check, measured over many queries for the figure."""
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
from pathfaith import neuromod as NM

warnings.filterwarnings('ignore')
EPS, EPS_WIDE = 1e-4, 1e-3
CURVE = [round(0.1 * i, 1) for i in range(13)]           # gain 0 to 1.2


def scores(bundle, data, h, r, gain=None, grad=False):
    tails = torch.arange(int(data.num_nodes))
    batch = torch.stack([torch.full_like(tails, h), tails, torch.full_like(tails, r)],
                        dim=-1).unsqueeze(0)
    with torch.set_grad_enabled(grad), NM.edge_gain(bundle, gain):
        return bundle.model(data, batch).view(-1)


def grad_of_p(bundle, data, h, r, t, w):
    w = w.clone().requires_grad_()
    p, _ = NM.prob(bundle, data, h, r, t, w, grad=True)
    p.backward()
    return w.grad.detach()


def run_query(bundle, data, rec, seed, keep_scores):
    h, r, t = rec['h'], rec['r'], rec['t']
    E = int(data.num_edges)
    fl = A.flipped(data)
    t0 = time.time()
    out = dict(kind='query', h=h, r=r, t=t)

    s_dep = scores(bundle, data, h, r)
    ones = torch.ones(E, requires_grad=True)
    s_rel = scores(bundle, data, h, r, ones, grad=True).detach()
    s_cor = scores(bundle, fl, h, r, torch.ones(E, requires_grad=True), grad=True).detach()
    out['gap_released'] = float((s_rel - s_dep).abs().max())
    out['gap_corrected'] = float((s_cor - s_dep).abs().max())
    if keep_scores:
        out['scores'] = dict(deployed=[round(float(x), 4) for x in s_dep],
                             released=[round(float(x), 4) for x in s_rel],
                             corrected=[round(float(x), 4) for x in s_cor])

    g = torch.Generator().manual_seed(seed * 7919 + h * 31 + r)
    v = (torch.randint(0, 2, (E,), generator=g) * 2 - 1).float()
    p_plus = float(NM.prob(bundle, data, h, r, t, 1 + EPS * v)[0])
    p_minus = float(NM.prob(bundle, data, h, r, t, 1 - EPS * v)[0])
    out['fd'] = (p_plus - p_minus) / (2 * EPS)
    p_plus = float(NM.prob(bundle, data, h, r, t, 1 + EPS_WIDE * v)[0])
    p_minus = float(NM.prob(bundle, data, h, r, t, 1 - EPS_WIDE * v)[0])
    out['fd_wide'] = (p_plus - p_minus) / (2 * EPS_WIDE)
    g_rel = grad_of_p(bundle, data, h, r, t, torch.ones(E))
    g_cor = grad_of_p(bundle, fl, h, r, t, torch.ones(E))
    out['dir_released'] = float(g_rel @ v)
    out['dir_corrected'] = float(g_cor @ v)

    ch, R = NM.channel_of(data)
    cr = torch.zeros(R).index_add_(0, ch, g_rel)
    cc = torch.zeros(R).index_add_(0, ch, g_cor)
    with contextlib.redirect_stdout(io.StringIO()):
        exp = A.explain_gradient(bundle, data, h, r, t, topk=5, num_beam=10, corrected=True)
    explained = sorted({hop[2] % R for p in exp.paths for hop in p})
    if explained:
        c = max(explained, key=lambda k: abs(float(cr[k] - cc[k])))
        curve = []
        for a in CURVE:
            gch = torch.ones(R); gch[c] = a
            curve.append(float(NM.prob(bundle, data, h, r, t, gch[ch])[0]))
        out['channel'] = dict(c=int(c), grad_released=float(cr[c]), grad_corrected=float(cc[c]),
                              gains=CURVE, p=curve, p1=float(NM.prob(bundle, data, h, r, t)[0]))
    out['seconds'] = round(time.time() - t0, 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='CoDExSmall')
    ap.add_argument('--n', type=int, default=40)
    ap.add_argument('--keep-scores', type=int, default=3)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    NM._install()
    from pathfaith.analyze import load
    lc = args.dataset.lower()
    _, recs = load(f'results/paths_{lc}_seed0.jsonl')
    recs = recs[:args.n]
    out = args.out or f'results/direction_{lc}.jsonl'
    b = build(args.dataset, 'ultra_4g.pth')
    data = b.test_data
    with open(out, 'w') as f:
        f.write(json.dumps(dict(kind='meta', dataset=b.name, ckpt=b.ckpt, eps=EPS, eps_wide=EPS_WIDE,
                                n=len(recs))) + '\n')
        for i, rec in enumerate(recs):
            res = run_query(b, data, rec, seed=0, keep_scores=i < args.keep_scores)
            f.write(json.dumps(res) + '\n'); f.flush()
            print(f'[{i + 1}/{len(recs)}] score gap released {res["gap_released"]:.2f} '
                  f'corrected {res["gap_corrected"]:.1e}  fd {res["fd"]:+.2e} '
                  f'released {res["dir_released"]:+.2e} corrected {res["dir_corrected"]:+.2e} '
                  f'{res["seconds"]:.1f}s', flush=True)
    print('wrote', out)


if __name__ == '__main__':
    main()
