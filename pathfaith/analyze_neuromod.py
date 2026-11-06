"""Summaries for the neuromodulatory channel experiment, and the ranking
metrics from the retrieval lectures applied to the path benchmark.

Every number is a mean over queries with a 95% bootstrap interval, the same
convention as the rest of the benchmark: the model is deterministic, so the
queries are the only source of variance.

Channel metrics (one value per query, then averaged):
  silence_best      share of p removed by silencing the strongest of the
                    channels the explanation's paths use
  silence_explained mean share removed per explained channel
  silence_random    mean share removed per random channel (the control)
  joint             share removed by silencing every explained channel at once
  joint_half        the same at half gain
  content           share removed by deleting only the explanation's own edges
  half_dose         for channels whose full silencing moves p by at least 5%,
                    the fraction of that effect already present at half gain;
                    a linear dose response gives 0.5
  switch_like       share of those channels with less than a quarter of the
                    effect at half gain
  inverted_u        share of those channels whose response changes direction
                    along the dose, the shape neuromodulators are known for
  sign_agree        share of those channels where the gradient at gain one has
                    the sign of the full-silencing effect; chance is about 0.5
  grad_within_2x    share where the gradient's linear prediction of full
                    silencing is within a factor of two of what happened
  tau               Kendall's tau, gradient against silencing effect, over the
                    channels tested in a query
  top_explained     share of queries where the channel whose silencing does
                    most, among those tested, is one the explanation names
  top_is_query_rel  share where that channel is the query's own relation

Ranking metrics for the path benchmark (relevance = share of p a path's
deletion removes, negatives counted as zero):
  ndcg5, ndcg5_chance    nDCG@5 of the method's order, and its expectation
                         under a random order
  mrr_best               reciprocal rank, in the method's order, of the path
                         whose deletion does most
"""
from __future__ import annotations

import json
import math

import numpy as np

from pathfaith.analyze import TIE, boot, kendall_tau, load

MEANINGFUL = 0.05      # a channel "matters" if full silencing moves p by 5%
STEP = 0.01            # a change of direction must be at least 1% of p


def rows_of(path):
    meta, rows = None, []
    for line in open(path):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get('kind') == 'meta':
            meta = d
        elif d.get('kind') == 'query':
            rows.append(d)
    return meta, rows


def _drop(c, base, a):
    return (base - c['doses'][a][0]) / base


def per_query(rows):
    """One value per query and metric. None where a query has no instance."""
    out = {k: [] for k in ('silence_best', 'silence_explained', 'silence_random', 'joint',
                           'joint_half', 'content', 'context_minus_content', 'half_dose',
                           'switch_like', 'inverted_u', 'sign_agree', 'grad_within_2x',
                           'tau', 'top_explained', 'top_is_query_rel', 'val_gap')}
    curves = {'explained': [], 'gradient': [], 'random': [], 'pred_explained': []}
    for d in rows:
        base = d['base_prob']
        if base <= 0:
            continue
        out['val_gap'].append(d['val_gap'])
        ex = [c for c in d['channels'] if c['role'] == 'explained']
        rn = [c for c in d['channels'] if c['role'] == 'random']
        dosed = [c for c in d['channels'] if c['role'] in ('explained', 'gradient')]
        if ex:
            full = [_drop(c, base, '0.0') for c in ex]
            out['silence_best'].append(max(full))
            out['silence_explained'].append(float(np.mean(full)))
        if rn:
            out['silence_random'].append(float(np.mean([_drop(c, base, '0.0') for c in rn])))
        if d.get('joint'):
            j = _drop({'doses': d['joint']}, base, '0.0')
            out['joint'].append(j)
            out['joint_half'].append(_drop({'doses': d['joint']}, base, '0.5'))
            out['content'].append(d['comp_topk'] / base)
            out['context_minus_content'].append(j - d['comp_topk'] / base)

        # shape and gradient, over channels whose full silencing matters
        half, sw, inv, sg, w2 = [], [], [], [], []
        for c in dosed:
            full = _drop(c, base, '0.0')
            if abs(full) < MEANINGFUL:
                continue
            path = [0.0] + [_drop(c, base, a) for a in ('0.75', '0.5', '0.25', '0.0')]
            half.append(path[2] / full)
            sw.append(1.0 if path[2] / full < 0.25 else 0.0)
            # a change of direction: some intermediate dose moves p the other way
            inv.append(1.0 if any(x * full < 0 and abs(x) >= STEP for x in path[1:4]) else 0.0)
            pred = c['grad'] / base          # linear: drop(0) ~ dp/dg * 1
            sg.append(1.0 if pred * full > 0 else 0.0)
            w2.append(1.0 if pred * full > 0 and 0.5 <= pred / full <= 2.0 else 0.0)
        for k, v in (('half_dose', half), ('switch_like', sw), ('inverted_u', inv),
                     ('sign_agree', sg), ('grad_within_2x', w2)):
            if v:
                out[k].append(float(np.mean(v)))

        if len(dosed) >= 3:
            g = [c['grad'] for c in dosed]
            e = [_drop(c, base, '0.0') for c in dosed]
            tau = kendall_tau(g, e)
            if tau == tau:
                out['tau'].append(tau)
        tested = dosed + rn
        if tested:
            best = max(tested, key=lambda c: _drop(c, base, '0.0'))
            if _drop(best, base, '0.0') >= MEANINGFUL:
                out['top_explained'].append(1.0 if best['role'] == 'explained' else 0.0)
                out['top_is_query_rel'].append(1.0 if best['is_query_rel'] else 0.0)

        # dose-response curves, as share of p removed, from gain 1 down to 0
        for role in ('explained', 'gradient', 'random'):
            cs = [c for c in d['channels'] if c['role'] == role]
            if not cs:
                continue
            levels = ('0.75', '0.5', '0.25', '0.0') if role != 'random' else ('0.5', '0.0')
            curves[role].append([0.0] + [float(np.mean([_drop(c, base, a) for c in cs]))
                                         for a in levels])
        if ex:
            curves['pred_explained'].append(
                [float(np.mean([(1 - a) * c['grad'] / base for c in ex]))
                 for a in (1.0, 0.75, 0.5, 0.25, 0.0)])
    return out, curves


def summarize(rows):
    vals, curves = per_query(rows)
    s = {k: boot(v) + (len(v),) for k, v in vals.items()}
    s['val_gap_max'] = (max(vals['val_gap']) if vals['val_gap'] else float('nan'),)
    return s, curves, vals


# ------------------------------------------------------------------ ranking metrics
def _dcg(rel, k):
    return sum(r / math.log2(i + 2) for i, r in enumerate(rel[:k]))


def ranking_metrics(rows, k=5):
    nd, ndc, mrr, mrc = [], [], [], []
    for d in rows:
        b = d['base_prob']
        drops = d['true_drops']
        if b <= 0 or len(drops) < 2:
            continue
        rel = [max(x, 0.0) / b for x in drops]       # in the method's own order
        ideal = _dcg(sorted(rel, reverse=True), k)
        if ideal <= 0:
            continue
        nd.append(_dcg(rel, k) / ideal)
        disc = sum(1 / math.log2(i + 2) for i in range(min(k, len(rel))))
        ndc.append((sum(rel) / len(rel)) * disc / ideal)
        # ties count: the list reaches a best path at the first position whose
        # drop is within TIE of the largest
        mx = max(drops)
        hits = [i for i, x in enumerate(drops) if x >= mx - TIE]
        mrr.append(1.0 / (hits[0] + 1))
        mrc.append(_first_hit_rr(len(drops), len(hits)))
    return {'ndcg5': boot(nd) + (len(nd),), 'ndcg5_chance': boot(ndc) + (len(ndc),),
            'mrr_best': boot(mrr) + (len(mrr),), 'mrr_best_chance': boot(mrc) + (len(mrc),)}


def _first_hit_rr(n, m):
    """Expected 1/rank of the first of m best items in a random order of n."""
    total, rest = 0.0, math.comb(n, m)
    for j in range(1, n - m + 2):
        total += math.comb(n - j, m - 1) / rest / j
    return total


def main():
    import sys
    for p in sys.argv[1:]:
        meta, rows = rows_of(p)
        s, _, _ = summarize(rows)
        print(p, len(rows), 'queries')
        for k, v in s.items():
            if len(v) == 4:
                print(f'  {k:22s} {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]  n={v[3]}')
            else:
                print(f'  {k:22s} {v[0]:.2e}')


if __name__ == '__main__':
    main()
