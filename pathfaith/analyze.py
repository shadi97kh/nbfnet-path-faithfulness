"""Turn the per-query records into the numbers and figures in the report.

Every headline number carries a bootstrap interval, because the sample of
queries is the only source of randomness here: the model is frozen and the
interventions are deterministic, so a spread across seeds would be meaningless
and a spread across queries is the honest one.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np

TIE = 1e-6   # two drops this close are the same number, not a ranking


def load(path: str):
    """Read one results file: the meta line, then one row per query."""
    meta, rows = None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue          # a torn line from an interrupted write
            if d.get('kind') == 'meta':
                meta = d
            elif d.get('kind') == 'query':
                rows.append(d)
    return meta, rows


def boot(x, fn=np.mean, n: int = 10000, seed: int = 0):
    """Point estimate and a 95% bootstrap interval."""
    x = np.asarray([v for v in x if v is not None and not math.isnan(v)], dtype=float)
    if len(x) == 0:
        return float('nan'), float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    stats = fn(x[idx], axis=1)
    return float(fn(x)), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def kendall_tau(a, b):
    """Kendall's tau-b, written out so ties are handled explicitly rather than
    hidden inside a library default."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    if n < 2:
        return float('nan')
    conc = disc = ta = tb = 0
    for i in range(n):
        for j in range(i + 1, n):
            da, db = a[i] - a[j], b[i] - b[j]
            if abs(da) < TIE and abs(db) < TIE:
                continue
            if abs(da) < TIE:
                ta += 1
            elif abs(db) < TIE:
                tb += 1
            elif da * db > 0:
                conc += 1
            else:
                disc += 1
    denom = math.sqrt((conc + disc + ta) * (conc + disc + tb))
    return (conc - disc) / denom if denom > 0 else float('nan')


def order_tau(drops, score=None):
    """Kendall's tau between a ranking of the candidate paths and their
    measured drops. With no score it is the method's own order (rank i scores
    -i), the definition Summary uses; otherwise the given scores, for example
    integrated gradients re-ranking the same candidates."""
    if len(drops) < 2:
        return float('nan')
    s = [-i for i in range(len(drops))] if score is None else list(score)[:len(drops)]
    return kendall_tau(s, drops)


def order_top_gap(drops, score=None):
    """1 if the path ranked first is the most damaging (ties count), minus the
    rate a random pick would get. Same conventions as order_tau."""
    if len(drops) < 2:
        return float('nan')
    i = 0 if score is None else int(np.argmax(list(score)[:len(drops)]))
    mx = max(drops)
    return (1.0 if drops[i] >= mx - TIE else 0.0) - sum(1 for x in drops if x >= mx - TIE) / len(drops)


@dataclass
class Summary:
    dataset: str = ''
    n: int = 0
    rows: list = None

    def report(self) -> dict:
        R = self.rows
        out = {'dataset': self.dataset, 'n_queries': len(R)}

        # --- does the gradient ranking resolve anything inside its own pool
        taus, all_tied, spread = [], 0, []
        for r in R:
            d, w = r['true_drops'], r['grad_weights'][:len(r['true_drops'])]
            if len(d) < 2:
                continue
            if max(d) - min(d) < TIE:
                all_tied += 1
            spread.append(max(d) - min(d))
            # gradient ranks the pool in order, so rank i is simply -i
            taus.append(kendall_tau([-i for i in range(len(d))], d))
        out['frac_pool_fully_tied'] = all_tied / max(1, len(R))
        out['drop_spread_within_pool'] = boot(spread)
        out['kendall_tau_grad_vs_drop'] = boot([t for t in taus if not math.isnan(t)])

        # The most interpretable version of the same question: how often is
        # the path the method puts first actually the most damaging one to
        # delete? Scored against the rate a coin would get, since many
        # candidates tie at the maximum and a tie counts as a hit.
        top1, chance = [], []
        for r in R:
            d = r['true_drops']
            if len(d) < 2:
                continue
            mx = max(d)
            top1.append(1.0 if d[0] >= mx - TIE else 0.0)
            chance.append(sum(1 for x in d if x >= mx - TIE) / len(d))
        out['top1_is_best'] = boot(top1)
        out['top1_chance'] = boot(chance)
        out['top1_over_chance'] = boot([a - b for a, b in zip(top1, chance)])

        # --- top of the pool against the bottom of the same pool
        tops, bots = [], []
        for r in R:
            d = r['true_drops']
            if len(d) >= 6:
                k = max(1, len(d) // 4)
                tops.append(float(np.mean(d[:k])))
                bots.append(float(np.mean(d[-k:])))
        out['mean_drop_top_quartile'] = boot(tops)
        out['mean_drop_bottom_quartile'] = boot(bots)
        out['top_minus_bottom'] = boot([a - b for a, b in zip(tops, bots)])

        # --- set level: is the explanation load bearing at all
        out['base_prob'] = boot([r['base_prob'] for r in R])
        out['comprehensiveness'] = boot([r['comp_topk'] for r in R])
        out['comprehensiveness_random'] = boot([r['comp_random'] for r in R])
        out['comp_minus_random'] = boot([r['comp_topk'] - r['comp_random'] for r in R])
        # as a share of what the query started with, so queries with very
        # different base probabilities can be averaged honestly
        out['comp_frac_of_base'] = boot([r['comp_topk'] / r['base_prob']
                                         for r in R if r['base_prob'] > 0])
        out['comp_random_frac_of_base'] = boot([r['comp_random'] / r['base_prob']
                                                for r in R if r['base_prob'] > 0])
        out['sufficiency'] = boot([r['suff_topk'] for r in R])
        # Keeping only a handful of edges also strips the competitors out of
        # the softmax, so this ratio can exceed 1 without the explanation being
        # sufficient in any useful sense. It is reported, and read with that
        # in mind, alongside the rank-based measures that do not renormalise.
        out['suff_frac_of_base'] = boot([r['suff_topk'] / r['base_prob']
                                         for r in R if r['base_prob'] > 0])
        out['frac_suff_above_one'] = boot(
            [1.0 if (r['base_prob'] > 0 and r['suff_topk'] > r['base_prob']) else 0.0
             for r in R])
        out['rank_after_deletion'] = boot([r['comp_rank_after'] for r in R], np.median)
        out['frac_rank_broken'] = boot([1.0 if r['comp_rank_after'] > 10 else 0.0
                                        for r in R])
        # deleting the explanation occasionally helps the gold tail, which no
        # account of the shortlist as supporting evidence predicts
        out['frac_comp_negative'] = boot([1.0 if r['comp_topk'] < -TIE else 0.0
                                          for r in R])

        # --- is the shortlist k explanations or one
        out['bottleneck_frac'] = boot([r['bottleneck_frac'] for r in R])
        out['distinct_edge_ratio'] = boot(
            [r['n_distinct_edges'] / max(1, sum(r['path_lengths'][:r['n_paths']]) * 2)
             for r in R if r['n_paths']])
        out['single_edge_drop'] = boot([r['bottleneck_drop'] for r in R])
        # The ratio is only meaningful where the shortlist deletion actually
        # did something; dividing by a near-zero denominator produces numbers
        # that swing between large positive and large negative for no reason.
        meaningful = [r for r in R
                      if r['base_prob'] > 0 and r['comp_topk'] / r['base_prob'] > 0.05]
        out['single_edge_recovers_frac'] = boot(
            [r['bottleneck_drop'] / r['comp_topk'] for r in meaningful])
        out['frac_comp_meaningful'] = boot(
            [1.0 if (r['base_prob'] > 0 and r['comp_topk'] / r['base_prob'] > 0.05)
             else 0.0 for r in R])
        out['frac_single_edge_breaks'] = boot(
            [1.0 if r['bottleneck_rank_after'] > 10 else 0.0 for r in R])

        # --- portability: does the answer follow an edited path
        out['rewire_drop_on_gold'] = boot([r['rewire_base_drop'] for r in R])
        out['rewire_gain_on_target'] = boot([r['rewire_follow'] for r in R])
        out['frac_rewire_followed'] = boot([1.0 if r['rewire_follow'] > 0 else 0.0
                                            for r in R])

        # --- cost
        out['forwards_per_query'] = boot([r['n_forward'] for r in R])
        out['seconds_per_query'] = boot([r['seconds'] for r in R])
        return out
