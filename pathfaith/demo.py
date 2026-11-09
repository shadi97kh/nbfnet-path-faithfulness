"""Live demo: run the whole audit on a few queries, then show the benchmark.

This is the script to run in front of an audience. It loads the frozen
checkpoint, picks several queries the model gets right, and for each one shows
NBFNet's own explanation and then intervenes on the graph live: delete the
explanation, delete as many random edges, delete only the busiest edge, and
reroute the last hop. Every number on screen is computed while you watch.

It ends by reading the full benchmark off disk, so the audience sees that the
handful of live queries is one slice of a much larger, pre-registered sweep.

CoDEx-S is the default because it is small enough to run live on a laptop CPU
and its entities are Wikidata items, so the output is readable.

Usage:
    python -m pathfaith.demo                       # 5 queries on CoDEx-S
    python -m pathfaith.demo --n 3 --seed 7        # a different set of queries
    python -m pathfaith.demo --dataset WN18RR      # WordNet synset names
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import warnings
from collections import Counter

import torch

# library deprecation chatter is noise in front of an audience
warnings.filterwarnings('ignore')

from pathfaith.loader import build, n_params, DATA_ROOT, REPO
from pathfaith import attribution as A
from pathfaith.experiment import prob_and_rank, path_edges, random_real_edges

LABELS = os.environ.get('KG_LABEL_ROOT', os.path.join(REPO, 'kg-labels'))

# ------------------------------------------------------------------ display
USE_COLOR = sys.stdout.isatty()


def c(text, code):
    return f'\033[{code}m{text}\033[0m' if USE_COLOR else str(text)


def bold(t): return c(t, '1')
def dim(t): return c(t, '2')
def red(t): return c(t, '31')
def green(t): return c(t, '32')
def yellow(t): return c(t, '33')
def blue(t): return c(t, '34')


def rule(title=''):
    line = '=' * 72
    if title:
        title = f' {title} '
        line = '==' + title + '=' * max(0, 70 - len(title))
    print(bold(line))


# ------------------------------------------------------------------ names
class Names:
    """Turn ULTRA's integer ids back into something a person can read."""

    def __init__(self, dataset: str, num_relations: int):
        self.half = num_relations // 2
        self.ent, self.rel = {}, {}
        if dataset.lower().startswith('codex'):
            self._codex(dataset)
        elif dataset.lower() == 'wn18rr':
            self._wn18rr()

    def _codex(self, dataset):
        # ULTRA's TransductiveDataset numbers entities and relations by first
        # appearance, line by line, over train, valid, test. Replay that.
        size = {'codexsmall': 's', 'codexmedium': 'm', 'codexlarge': 'l'}[dataset.lower()]
        raw = os.path.join(DATA_ROOT, f'codex-{size}', 'raw')
        ent_id, rel_id = {}, {}
        for split in ('train', 'valid', 'test'):
            with open(os.path.join(raw, f'{split}.txt'), encoding='utf-8') as f:
                for line in f:
                    u, r, v = line.split()
                    for e in (u, v):
                        if e not in ent_id:
                            ent_id[e] = len(ent_id)
                    if r not in rel_id:
                        rel_id[r] = len(rel_id)
        try:
            el = json.load(open(os.path.join(LABELS, 'codex', 'entities.json')))
            rl = json.load(open(os.path.join(LABELS, 'codex', 'relations.json')))
        except FileNotFoundError:
            el, rl = {}, {}
        self.ent = {i: el.get(q, {}).get('label', q) for q, i in ent_id.items()}
        self.rel = {i: rl.get(p, {}).get('label', p) for p, i in rel_id.items()}

    def _wn18rr(self):
        # PyG numbers WN18RR by first appearance of the synset offsets, heads
        # then tails within each split. The text release of WN18RR is line for
        # line the same triples with readable synset names, so zipping the two
        # gives each offset its name.
        from itertools import chain
        root = os.path.join(DATA_ROOT, 'wn18rr', 'raw')
        text = os.environ.get('WN18RR_TEXT', os.path.join(LABELS, 'wn18rr'))
        # Relations are NOT numbered by first appearance: PyG fixes the order
        # in WordNet18RR.edge2id, so read it from there instead of guessing.
        from torch_geometric.datasets import WordNet18RR
        rel = dict(WordNet18RR.edge2id)
        node, name = {}, {}
        for split in ('train', 'valid', 'test'):
            orig = open(os.path.join(root, f'{split}.txt')).read().splitlines()
            src, _, dst = zip(*(l.split('\t') for l in orig))
            for o in chain(src, dst):
                node.setdefault(o, len(node))
            tp = os.path.join(text, f'{split}.txt')
            if os.path.exists(tp):
                for lo, lt in zip(orig, open(tp).read().splitlines()):
                    oh, _, ot = lo.split('\t')
                    th, _, tt = lt.split('\t')
                    name.setdefault(oh, set()).add(th)
                    name.setdefault(ot, set()).add(tt)
        # WordNet offsets repeat across parts of speech, so the offset-only
        # release merges a few unrelated synsets into one node. Show both
        # names rather than pretend the node is one thing.
        self.ent = {i: ' / '.join(sorted(name.get(o, {o}))) for o, i in node.items()}
        self.rel = {i: r.lstrip('_').replace('_', ' ') for r, i in rel.items()}

    def e(self, i):
        return self.ent.get(int(i), f'#{int(i)}')

    def r(self, i):
        i = int(i)
        return self.rel.get(i % self.half, f'rel{i % self.half}')

    def hop(self, h, t, r):
        """Draw the arrow the way the real fact points. A hop along a stored
        inverse edge is a fact pointing the other way, read backwards."""
        if int(r) >= self.half:
            return f'{self.e(h)} <-[{self.r(r)}]- {self.e(t)}'
        return f'{self.e(h)} -[{self.r(r)}]-> {self.e(t)}'


# ------------------------------------------------------------------ one query
def show_query(bundle, data, names, h, r, t, k, rng, idx, total):
    rule(f'query {idx} of {total}')
    t0 = time.time()
    base_p, base_rank, scores = prob_and_rank(bundle, data, h, r, t)
    print(f'  {bold(names.e(h))}  -[{names.r(r)}]->  ?')
    print(f'  model answers {bold(names.e(t))}   '
          f'rank {base_rank}, p = {base_p:.3f}')

    exp = A.explain_gradient(bundle, data, h, r, t, topk=k, num_beam=10)
    paths = exp.paths[:k]
    print()
    print(f'  {blue("NBFNet's own explanation")} (gradient, top {len(paths)} paths):')
    for j, (p, w) in enumerate(zip(paths, exp.weights), 1):
        print(f'    {j}. weight {w:.3f}   ' + dim(f'{len(p)} hop' + ('s' if len(p) > 1 else '')))
        for hop in p:
            print(f'         {names.hop(*hop)}')

    # the direction check: the released explanation takes its gradient through
    # PyG's generic message passing, which runs messages the opposite way from
    # the kernel the model was trained and is served with
    from pathfaith import neuromod as NM
    NM._install()
    p_gen = float(NM.prob(bundle, data, h, r, t,
                          torch.ones(int(data.num_edges), requires_grad=True), grad=True)[0])
    print(dim(f'    (that gradient is taken on a model that gives this answer p = {p_gen:.3f}; '
              f'the deployed model gives {base_p:.3f})'))

    cols = [path_edges(data, p) for p in paths]
    joint = torch.unique(torch.cat(cols)) if cols else torch.empty(0, dtype=torch.long)

    print()
    print(f'  {yellow("intervene on the graph")} (each line is a fresh forward pass):')

    p_del, rank_del, _ = prob_and_rank(bundle, A.drop_edges(data, joint), h, r, t)
    share = (base_p - p_del) / base_p if base_p > 0 else 0.0
    verdict = green('load-bearing') if rank_del > 10 or share > 0.5 else yellow('partly')
    print(f'    delete those {len(paths)} paths ({len(joint)} edges)'.ljust(44)
          + f'p {base_p:.3f} -> {p_del:.3f}   rank {base_rank} -> {rank_del:<5d} {verdict}')

    rnd = random_real_edges(data, len(joint), rng)
    p_rnd, rank_rnd, _ = prob_and_rank(bundle, A.drop_edges(data, rnd), h, r, t)
    print(f'    delete {len(joint)} random edges instead'.ljust(44)
          + f'p {base_p:.3f} -> {p_rnd:.3f}   rank {base_rank} -> {rank_rnd:<5d} '
          + dim('control'))

    counts = Counter(int(x) for col in cols for x in col)
    busiest, n_using = counts.most_common(1)[0] if counts else (None, 0)
    if busiest is not None:
        p_b, rank_b, _ = prob_and_rank(bundle, A.drop_edges(data, torch.tensor([busiest])), h, r, t)
        rec = (base_p - p_b) / (base_p - p_del) if base_p - p_del > 1e-6 else float('nan')
        if rec != rec:
            rec_txt = ''
        elif rec < 0.005:
            rec_txt = dim('none of the effect')
        else:
            rec_txt = bold(f'{100 * min(rec, 9.99):.0f}% of the effect')
        eh, et, er = (int(data.edge_index[0, busiest]), int(data.edge_index[1, busiest]),
                      int(data.edge_type[busiest]))
        print(f'    delete only the busiest edge'.ljust(44)
              + f'p {base_p:.3f} -> {p_b:.3f}   rank {base_rank} -> {rank_b:<5d} ' + rec_txt)
        print(dim(f'      that edge, in {n_using} of {len(paths)} paths: {names.hop(eh, et, er)}'))

    term = path_edges(data, [paths[0][-1]]) if paths else torch.empty(0)
    if len(term):
        order = scores.argsort(descending=True)
        alt = int(order[1]) if int(order[0]) == t else int(order[0])
        rew = A.rewire_edge(data, int(term[0]), alt)
        s_rew = A.score(bundle, rew, h, r, torch.arange(int(data.num_nodes)))
        p_rew = torch.softmax(s_rew, 0)
        p_alt0 = float(torch.softmax(scores, 0)[alt])
        lost, gained = base_p - float(p_rew[t]), float(p_rew[alt]) - p_alt0
        def short(x, n=16):
            x = str(x)
            return x if len(x) <= n else x[:n - 1] + '~'
        head = f'    reroute the last hop to {short(names.e(alt))}'
        head = head.ljust(44) if len(head) < 44 else head + ' '   # never let two names touch
        print(head + f'{short(names.e(t), 14)} {-lost:+.3f}   {short(names.e(alt), 14)} {gained:+.3f}')

    # the neuromodulation test: one relation type is one channel, broadcast to
    # every edge of that type; turn the explanation's main channel down
    nm = None
    if paths:
        from pathfaith import neuromod as NM
        NM._install()
        ch, R = NM.channel_of(data)
        hops = [hop[2] % R for p in paths for hop in p]
        c = max(set(hops), key=hops.count)
        gc = torch.ones(R, requires_grad=True)
        pg, _ = NM.prob(bundle, A.flipped(data), h, r, t, gc[ch], grad=True)
        pg.backward()
        slope = float(gc.grad[c])
        doses = {}
        for a in (0.5, 0.0):
            g = torch.ones(R); g[c] = a
            doses[a] = float(NM.prob(bundle, data, h, r, t, g[ch])[0])
        n_c = int((ch == c).sum())
        full = base_p - doses[0.0]
        agrees = slope * full > 0
        print(f'    {blue("neuromodulate")} the [{names.r(c)}] channel ({n_c:,} edges)')
        print(f'      gain 1 -> 0.5 -> 0'.ljust(44)
              + f'p {base_p:.3f} -> {doses[0.5]:.3f} -> {doses[0.0]:.3f}')
        verdict = green('right sign') if agrees else red('wrong sign')
        print(f'      gradient at gain 1 predicts'.ljust(44)
              + f'p -> {base_p - slope:.3f} at gain 0   ' + verdict)
        nm = dict(agrees=agrees, full=full / base_p if base_p > 0 else 0.0)

    print(dim(f'  ({time.time() - t0:.1f}s for this query)'))
    print()
    return dict(share=share, broken=rank_del > 10, random_share=(base_p - p_rnd) / base_p
                if base_p > 0 else 0.0, nm=nm)


# ------------------------------------------------------------------ benchmark
def show_benchmark(results_dir='results'):
    """Read the full sweep off disk and print it, with sample sizes and CIs."""
    from pathfaith.analyze import load, Summary
    rule('the same pipeline, at benchmark scale')
    sets = [('WN18RR', 'wn18rr', 'in pre-training'),
            ('FB15k-237', 'fb15k237', 'in pre-training'),
            ('CoDEx-S', 'codexsmall', 'held out'),
            ('NELL23k', 'nell23k', 'held out')]
    rows, total_q, total_del = [], 0, 0
    for label, tag, seen in sets:
        p = os.path.join(results_dir, f'paths_{tag}_seed0.jsonl')
        if not os.path.exists(p):
            continue
        _, R = load(p)
        if not R:
            continue
        rep = Summary(dataset=label, rows=R).report()
        total_q += len(R)
        total_del += sum(len(r['true_drops']) for r in R)
        rows.append((label, seen, len(R), rep))
    if not rows:
        print('  no results on disk yet; run section 4 of the notebook first')
        return

    def ci(rep, k):
        m, lo, hi = rep[k]
        return f'{m:+.3f} [{lo:+.2f},{hi:+.2f}]'

    print(f'  {"":24s}' + ''.join(f'{lab:>22s}' for lab, *_ in rows))
    print(f'  {"":24s}' + ''.join(f'{dim(s):>31s}' if USE_COLOR else f'{s:>22s}'
                                  for _, s, *_ in rows))
    print(f'  {"queries":24s}' + ''.join(f'{n:>22d}' for _, _, n, _ in rows))
    for name, key in [('delete shortlist', 'comp_frac_of_base'),
                      ('delete random edges', 'comp_random_frac_of_base'),
                      ('Kendall tau', 'kendall_tau_grad_vs_drop'),
                      ('top path is best', 'top1_is_best'),
                      ('  chance rate', 'top1_chance'),
                      ('paths share one edge', 'bottleneck_frac'),
                      ('one edge recovers', 'single_edge_recovers_frac')]:
        print(f'  {name:24s}' + ''.join(f'{ci(rep, key):>22s}' for *_, rep in rows))
    print()
    print(f'  {total_q} queries, {total_del:,} individual path deletions, '
          f'95% bootstrap intervals over queries.')


# ------------------------------------------------------------------ main
def main() -> None:
    global USE_COLOR
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='CoDExSmall')
    ap.add_argument('--ckpt', default='ultra_4g.pth')
    ap.add_argument('--n', type=int, default=5)
    ap.add_argument('--k', type=int, default=3, help='paths shown per query')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--no-benchmark', action='store_true')
    ap.add_argument('--color', choices=['auto', 'always', 'never'], default='auto')
    args = ap.parse_args()
    USE_COLOR = {'auto': sys.stdout.isatty(), 'always': True, 'never': False}[args.color]

    rule('Are NBFNet\'s path explanations actually faithful?  live run')
    t0 = time.time()
    bundle = build(args.dataset, args.ckpt)
    data = bundle.test_data
    names = Names(args.dataset, int(data.num_relations))
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):     # ULTRA prints on first use
        prob_and_rank(bundle, data, 0, 0, 0)
    print(f'  graph       {bundle.name}: {bundle.num_nodes:,} entities, '
          f'{bundle.num_relations // 2} relations, {bundle.num_edges:,} edges')
    print(f'  checkpoint  {bundle.ckpt}: {n_params(bundle.model):,} parameters, '
          f'all frozen, no training')
    print(f'  loaded in {time.time() - t0:.1f}s on CPU')
    print()

    tri = torch.cat([data.target_edge_index, data.target_edge_type.unsqueeze(0)]).t()
    order = torch.randperm(len(tri), generator=torch.Generator().manual_seed(args.seed))
    rng = random.Random(args.seed)
    picked = []
    for i in order.tolist():
        h, t, r = (int(x) for x in tri[i])
        _, rank, _ = prob_and_rank(bundle, data, h, r, t)
        if rank == 1:
            picked.append((h, r, t))
        if len(picked) >= args.n:
            break

    out, failed = [], 0
    for i, (h, r, t) in enumerate(picked, 1):
        try:
            out.append(show_query(bundle, data, names, h, r, t, args.k, rng, i, len(picked)))
        except Exception as exc:          # count it, show it, keep going
            failed += 1
            print(red(f'  query {i} failed: {type(exc).__name__}: {exc}'))
            print()

    rule('these live queries')
    broke = sum(o['broken'] for o in out)
    share = sum(o['share'] for o in out) / max(1, len(out))
    rshare = sum(o['random_share'] for o in out) / max(1, len(out))
    q = 'query' if len(picked) == 1 else 'queries'
    print(f'  {len(picked)} {q} attempted, {len(out)} completed every intervention, '
          f'{failed} failed.')
    pc = lambda x: f'{abs(100 * x):.0f}%' if abs(100 * x) < 0.5 else f'{100 * x:.0f}%'
    print(f'  deleting the explanation removed {pc(share)} of p(answer) on average; '
          f'random edges removed {pc(rshare)}.')
    print(f'  the answer fell out of the top 10 in {broke} of {len(out)}.')
    nms = [o['nm'] for o in out if o.get('nm')]
    if nms:
        print(f'  silencing the main channel removed {pc(sum(x["full"] for x in nms) / len(nms))} '
              f'of p on average; the gradient had the right sign in '
              f'{sum(x["agrees"] for x in nms)} of {len(nms)}.')
    print(dim(f'  total {time.time() - t0:.0f}s'))
    print()

    if not args.no_benchmark:
        show_benchmark()


if __name__ == '__main__':
    main()
