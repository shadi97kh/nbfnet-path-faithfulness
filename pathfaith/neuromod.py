"""Relation channels as neuromodulators: a dose-response test of the explanation."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import random
import time
import warnings

import torch

from pathfaith.loader import build
from pathfaith import attribution as A
from pathfaith.experiment import path_edges

warnings.filterwarnings('ignore')

DOSES = (0.75, 0.5, 0.25, 0.0)        # for explained and gradient channels
CONTROL_DOSES = (0.5, 0.0)            # for random channels
N_GRAD, N_RANDOM = 3, 3


def _install():
    from ultra import models
    cls = models.EntityNBFNet
    if getattr(cls, '_pf_installed', False):
        return

    def bellmanford(self, data, h_index, r_index, separate_grad=False):
        batch_size = len(r_index)
        query = self.query[torch.arange(batch_size, device=r_index.device), r_index]
        index = h_index.unsqueeze(-1).expand_as(query)
        boundary = torch.zeros(batch_size, data.num_nodes, self.dims[0], device=h_index.device)
        boundary.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1))
        size = (data.num_nodes, data.num_nodes)
        gain = getattr(self, '_pf_gain', None)
        edge_weight = (torch.ones(data.num_edges, device=h_index.device)
                       if gain is None else gain)
        hiddens, edge_weights, layer_input = [], [], boundary
        for layer in self.layers:
            if separate_grad:
                edge_weight = edge_weight.clone().requires_grad_()
            hidden = layer(layer_input, query, boundary, data.edge_index, data.edge_type,
                           size, edge_weight)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            edge_weights.append(edge_weight)
            layer_input = hidden
        node_query = query.unsqueeze(1).expand(-1, data.num_nodes, -1)
        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
        else:
            output = torch.cat([hiddens[-1], node_query], dim=-1)
        return {"node_feature": output, "edge_weights": edge_weights}

    cls.bellmanford = bellmanford
    cls._pf_installed = True


@contextlib.contextmanager
def edge_gain(bundle, gain):
    """Run the model with `gain` (one value per edge column) on its messages."""
    ent = bundle.model.entity_model
    ent._pf_gain = gain
    try:
        yield
    finally:
        ent._pf_gain = None


def channel_of(data):
    """Channel of every edge column: a relation and its inverse share one."""
    half = int(data.num_relations) // 2
    return data.edge_type % half, half


def prob(bundle, data, h, r, t, gain=None, grad=False):
    tails = torch.arange(int(data.num_nodes))
    batch = torch.stack([torch.full_like(tails, h), tails, torch.full_like(tails, r)],
                        dim=-1).unsqueeze(0)
    with torch.set_grad_enabled(grad), edge_gain(bundle, gain):
        s = bundle.model(data, batch).view(-1)
    p = torch.softmax(s, dim=0)
    rank = int((s.detach() > s.detach()[t]).sum()) + 1
    return p[t], rank


def run_query(bundle, data, rec, seed=0, corrected=True):
    h, r, t = rec['h'], rec['r'], rec['t']
    gdata = A.flipped(data) if corrected else data
    ch, R = channel_of(data)
    E = int(data.num_edges)
    t0, nf = time.time(), 0
    rng = random.Random(seed * 1000003 + h * 31 + r + 7)

    p0, rank0 = prob(bundle, data, h, r, t)
    p0 = float(p0); nf += 1

    with contextlib.redirect_stdout(io.StringIO()):
        exp = A.explain_gradient(bundle, data, h, r, t, topk=20, num_beam=20,
                                 corrected=corrected)
    top = exp.paths[:5]
    joint = torch.unique(torch.cat([path_edges(data, p) for p in top])) if top else \
        torch.empty(0, dtype=torch.long)

    g = torch.ones(E); g[joint] = 0.0
    p_val, _ = prob(bundle, data, h, r, t, g); nf += 1
    val_gap = abs((p0 - float(p_val)) - rec['comp_topk'])

    gc = torch.ones(R, requires_grad=True)
    pg, _ = prob(bundle, gdata, h, r, t, gc[ch], grad=True)
    pg.backward()
    grad = gc.grad.detach(); nf += 1

    n_edges = torch.bincount(ch, minlength=R)
    hop_types = [hop[2] % R for p in top for hop in p]
    explained = sorted(set(hop_types), key=lambda c: -hop_types.count(c))
    present = [c for c in range(R) if n_edges[c] > 0]
    by_grad = sorted(present, key=lambda c: -abs(float(grad[c])))
    gradient = [c for c in by_grad if c not in explained][:N_GRAD]
    pool = [c for c in present if c not in explained and c not in gradient]
    randoms = rng.sample(pool, min(N_RANDOM, len(pool)))

    def dose(cs, levels):
        out = {}
        for a in levels:
            gch = torch.ones(R)
            gch[cs] = a
            p, rk = prob(bundle, data, h, r, t, gch[ch])
            out[str(a)] = [float(p), rk]
        return out

    channels = []
    for role, cs, levels in (('explained', explained, DOSES), ('gradient', gradient, DOSES),
                             ('random', randoms, CONTROL_DOSES)):
        for c in cs:
            d = dose([c], levels); nf += len(levels)
            channels.append(dict(c=int(c), role=role, grad=float(grad[c]),
                                 n_edges=int(n_edges[c]), is_query_rel=int(c) == r % R,
                                 hops=hop_types.count(c), doses=d))
    joint_dose = dose(explained, (0.5, 0.0)) if explained else {}
    nf += len(joint_dose)

    return dict(kind='query', h=h, r=r, t=t, base_prob=p0, base_rank=rank0,
                comp_topk=rec['comp_topk'], val_gap=val_gap,
                n_channels=len(present), grad_all=[round(float(x), 8) for x in grad],
                channels=channels, joint=joint_dose, n_forward=nf,
                seconds=round(time.time() - t0, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='CoDExSmall')
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=None)
    ap.add_argument('--released', action='store_true',
                    help="use the released code's explanation and gradient instead of the corrected ones")
    args = ap.parse_args()

    _install()
    from pathfaith.analyze import load
    lc = args.dataset.lower()
    corrected = not args.released
    src = f'results/paths_{lc}_fixed_seed0.jsonl' if corrected else f'results/paths_{lc}_seed0.jsonl'
    _, recs = load(src)
    recs = recs[:args.n]
    out = args.out or (f'results/neuromod_{lc}_seed{args.seed}.jsonl' if corrected
                       else f'results/neuromod_{lc}_released_seed{args.seed}.jsonl')
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)

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
    meta = dict(kind='meta', dataset=b.name, ckpt=b.ckpt, seed=args.seed, doses=DOSES,
                explainer='corrected' if corrected else 'released', queries_from=src,
                control_doses=CONTROL_DOSES, n_grad=N_GRAD, n_random=N_RANDOM,
                num_relations=int(data.num_relations), num_edges=int(data.num_edges))
    print(f'{b.name}: {len(recs)} benchmark queries, {len(kept)} already done', flush=True)
    t0 = time.time()
    with open(out, 'w') as f:
        f.write(json.dumps(meta) + '\n')
        for d in kept:
            f.write(json.dumps(d) + '\n')
        f.flush()
        todo = [x for x in recs if (x['h'], x['r'], x['t']) not in done]
        for i, rec in enumerate(todo):
            res = run_query(b, data, rec, seed=args.seed, corrected=corrected)
            f.write(json.dumps(res) + '\n'); f.flush()
            ex = [c for c in res['channels'] if c['role'] == 'explained']
            best = max((1 - c['doses']['0.0'][0] / res['base_prob'] for c in ex), default=0)
            print(f'[{len(kept) + i + 1}/{len(recs)}] h={rec["h"]} r={rec["r"]} t={rec["t"]} '
                  f'explained channels {len(ex)}  best silencing {best:.2f}  '
                  f'check gap {res["val_gap"]:.1e}  {res["seconds"]:.0f}s', flush=True)
    print(f'wrote {out} in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
