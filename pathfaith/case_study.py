"""Worked examples for the report, with readable names.

The examples are not hand-picked. Each one is the first query in the
benchmark's pre-registered order that meets a stated rule, so a reader can
check that the example was not chosen for being striking:

  WN18RR   the first benchmark query whose reverse fact is in the graph
  CoDEx-S  the first benchmark query

Each is re-run here under the benchmark's protocol, and the output is a LaTeX
table the report includes. Numbers are recomputed, not copied from the
results file, and the script checks that they agree with it.

Usage:
    python -m pathfaith.case_study --out report/case.tex
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
from collections import Counter

import torch

from pathfaith.loader import build
from pathfaith import attribution as A
from pathfaith.analyze import load
from pathfaith.experiment import prob_and_rank, path_edges
from pathfaith.demo import Names


def tex(s: str) -> str:
    s = str(s)
    for a, b in (('\\', r'\textbackslash{}'), ('&', r'\&'), ('%', r'\%'), ('$', r'\$'),
                 ('#', r'\#'), ('_', r'\_'), ('{', r'\{'), ('}', r'\}'), ('~', r'\~{}'),
                 ('^', r'\^{}')):
        s = s.replace(a, b)
    return s


def short(name: str, n: int = 26) -> str:
    """Synset names like reform.n.01 read better without the sense suffix."""
    parts = name.split(' / ')
    out = []
    for p in parts:
        bits = p.split('.')
        if len(bits) >= 3 and bits[-1].isdigit() and len(bits[-2]) == 1:
            p = '.'.join(bits[:-2]).replace('_', ' ')
        out.append(p)
    s = ' / '.join(out)
    return s if len(s) <= n else s[:n - 1] + '…'


def hop_tex(names: Names, h, t, r) -> str:
    rel = tex(names.r(r))
    if int(r) >= names.half:
        return rf'{tex(short(names.e(h)))} \(\leftarrow\)\,{{\scriptsize {rel}}}\,\(-\) {tex(short(names.e(t)))}'
    return rf'{tex(short(names.e(h)))} \(-\)\,{{\scriptsize {rel}}}\,\(\rightarrow\) {tex(short(names.e(t)))}'


def example(dataset: str, rule, label: str, k: int = 3):
    path = f'results/paths_{dataset.lower()}_seed0.jsonl'
    if not os.path.exists(path):
        return None
    _, rows = load(path)
    pick = next((r for r in rows if rule(r)), None)
    if pick is None:
        return None
    b = build(dataset, 'ultra_4g.pth')
    data = b.test_data
    names = Names(dataset, int(data.num_relations))
    h, r, t = pick['h'], pick['r'], pick['t']
    with contextlib.redirect_stdout(io.StringIO()):
        base_p, base_rank, scores = prob_and_rank(b, data, h, r, t)
        exp = A.explain_gradient(b, data, h, r, t, topk=20, num_beam=20)
    paths = exp.paths[:5]
    cols = [path_edges(data, p) for p in paths]
    joint = torch.unique(torch.cat(cols))
    p_del, rank_del, _ = prob_and_rank(b, A.drop_edges(data, joint), h, r, t)
    counts = Counter(int(x) for c in cols for x in c)
    busiest, n_using = counts.most_common(1)[0]
    p_b, rank_b, _ = prob_and_rank(b, A.drop_edges(data, torch.tensor([busiest])), h, r, t)
    eh, et, er = (int(data.edge_index[0, busiest]), int(data.edge_index[1, busiest]),
                  int(data.edge_type[busiest]))

    # the recomputation must agree with the benchmark record it came from
    assert abs(base_p - pick['base_prob']) < 1e-4, (base_p, pick['base_prob'])
    assert abs((base_p - p_del) - pick['comp_topk']) < 1e-4, (base_p - p_del, pick['comp_topk'])

    lines = [rf'\multicolumn{{2}}{{@{{}}p{{0.97\textwidth}}@{{}}}}{{\textbf{{{label}}}.\quad '
             rf'Query: {tex(short(names.e(h), 30))} \(-\)\,{tex(names.r(r))}\,\(\rightarrow\) ?\quad '
             rf'answer \textbf{{{tex(short(names.e(t), 30))}}}, rank {base_rank}, '
             rf'\(p = {base_p:.3f}\)}} \\']
    for j, (p, w) in enumerate(zip(paths[:k], exp.weights[:k]), 1):
        hops = r' \\ & '.join(hop_tex(names, *x) for x in p)
        lines.append(rf'path {j} ({len(p)} hop{"s" if len(p) > 1 else ""}) & {hops} \\')
    lines.append(r'\addlinespace[2pt]')
    lines.append(rf'delete top 5 ({len(joint)} edges) & \(p\) {base_p:.3f} \(\rightarrow\) '
                 rf'{p_del:.3f}, rank {base_rank} \(\rightarrow\) {rank_del} \\')
    rec = (base_p - p_b) / (base_p - p_del) if base_p - p_del > 1e-6 else float('nan')
    rec_s = f'{100 * rec:.0f}\\% of that effect' if rec == rec and rec > 0.005 else 'none of that effect'
    lines.append(rf'delete busiest edge only & \(p\) {base_p:.3f} \(\rightarrow\) {p_b:.3f}, '
                 rf'rank {base_rank} \(\rightarrow\) {rank_b}: {rec_s} \\')
    lines.append(rf' & {{\footnotesize that edge, in {n_using} of 5 paths: {hop_tex(names, eh, et, er)}}} \\')
    return '\n'.join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='report/case.tex')
    ap.add_argument('--codex', action='store_true',
                    help='add the first CoDEx-S query as a second example (the report has room for one)')
    args = ap.parse_args()

    blocks = []
    wn = example('WN18RR', lambda r: r.get('reverse_in_graph') is True,
                 'WN18RR, first query with its reverse fact in the graph')
    if wn:
        blocks.append(wn)
    cx = example('CoDExSmall', lambda r: True, 'CoDEx-S, first benchmark query') \
        if args.codex else None
    if cx:
        blocks.append(cx)

    body = '\n\\midrule\n'.join(blocks) if blocks else r'\multicolumn{2}{l}{(examples pending)} \\'
    what = ('A worked example, the first query' if len(blocks) == 1
            else f'{["No", "One", "Two", "Three"][len(blocks)]} worked examples, each the first query')
    out = (r'\begin{table*}[t]' '\n' r'\centering\small' '\n'
           r'\caption{' + what + r" in the benchmark's fixed order "
           r'that meets the rule in its heading, re-run under the benchmark protocol. Arrows '
           r'show the direction of the stored fact, so a leftward arrow is a hop taken '
           r'backwards along a reverse edge. Names are from '
           + (r'the CoDEx labels and ' if cx else '') + r'the WN18RR text release.}' '\n'
           r'\label{tab:case}' '\n'
           r'\begin{tabular}{@{}p{0.2\textwidth}p{0.76\textwidth}@{}}' '\n'
           r'\toprule' '\n' + body + '\n' r'\bottomrule' '\n' r'\end{tabular}' '\n'
           r'\end{table*}' '\n')
    with open(args.out, 'w') as f:
        f.write('% generated by pathfaith.case_study\n' + out)
    print(f'wrote {args.out} with {len(blocks)} example(s)')


if __name__ == '__main__':
    main()
