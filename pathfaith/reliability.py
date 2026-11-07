"""Is the benchmark big enough? Show it rather than assert it.

Two checks, both run on the per-query records already on disk.

Convergence. Queries were drawn in a fixed random order before anything was
measured, so the first n rows of a results file are an honest sample of size
n. For each headline metric, the running estimate and its 95% bootstrap
interval are computed at n = 5, 10, 20, ... up to everything. If the benchmark
is large enough, the estimate settles and the interval narrows roughly as one
over the square root of n. The same curve shows how badly a five-query
benchmark would have misled.

Split-half agreement. The queries are split into two random halves many
times, and the metric is computed on each half. A reliable benchmark gives the
same answer on both halves; the spread of the difference is reported.

Usage:
    python -m pathfaith.reliability
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from pathfaith.analyze import load, TIE

DATASETS = [('WN18RR', 'wn18rr'), ('FB15k-237', 'fb15k237'),
            ('CoDEx-S', 'codexsmall'), ('NELL23k', 'nell23k')]


# ------------------------------------------------------------------ per-query values
def per_query(rows):
    """One number per query for each headline metric, so that means, intervals
    and splits are all computed the same way."""
    comp, rec, bott, top1gap = [], [], [], []
    for r in rows:
        b = r['base_prob']
        if b <= 0:
            continue
        comp.append(r['comp_topk'] / b)
        bott.append(r['bottleneck_frac'])
        if r['comp_topk'] / b > 0.05:
            rec.append(r['bottleneck_drop'] / r['comp_topk'])
        d = r['true_drops']
        if len(d) >= 2:
            mx = max(d)
            hit = 1.0 if d[0] >= mx - TIE else 0.0
            chance = sum(1 for x in d if x >= mx - TIE) / len(d)
            top1gap.append(hit - chance)
    return {'delete shortlist (share of p removed)': np.array(comp),
            'one edge recovers (share of effect)': np.array(rec),
            'paths sharing one edge': np.array(bott),
            'top path minus chance': np.array(top1gap)}


def drawn_index(rows):
    """For each headline metric, the position in the draw order of every
    query that has a value for it, so a curve can be indexed by how many
    queries were drawn, not by how many happened to have that metric."""
    out = {k: [] for k in per_query([])}
    for i, r in enumerate(rows):
        for k, v in per_query([r]).items():
            if len(v):
                out[k].append(i)
    return {k: np.array(v) for k, v in out.items()}


def convergence_drawn(values, drawn, n_total):
    """Running estimate against the number of queries drawn, 5 to n_total;
    each point uses the queries among the first n drawn that have a value."""
    out = []
    for n in grid(n_total):
        k = int((drawn < n).sum())
        if k >= 3:
            m, lo, hi = boot_ci(values[:k])
            out.append((n, m, lo, hi))
    return out


def boot_ci(x, n_boot=2000, seed=0):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float(x.mean()) if len(x) else float('nan'), float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def grid(n_max):
    """5, 10, 20, 30, ... then every 25, then the full sample."""
    pts = [n for n in (5, 10, 20, 30, 40, 50) if n < n_max]
    pts += list(range(75, n_max, 25))
    pts.append(n_max)
    return sorted(set(pts))


def convergence(values):
    ns = grid(len(values))
    out = []
    for n in ns:
        m, lo, hi = boot_ci(values[:n])
        out.append((n, m, lo, hi))
    return out


def split_half(values, reps=1000, seed=0):
    """Distribution of |mean(half A) - mean(half B)| over random splits."""
    rng = np.random.default_rng(seed)
    n = len(values)
    if n < 10:
        return float('nan'), float('nan')
    diffs = []
    for _ in range(reps):
        idx = rng.permutation(n)
        a, b = values[idx[:n // 2]], values[idx[n // 2:]]
        diffs.append(abs(a.mean() - b.mean()))
    diffs = np.array(diffs)
    return float(np.median(diffs)), float(np.percentile(diffs, 95))


# ------------------------------------------------------------------ report
def analyse(results_dir='results'):
    report = {}
    for label, tag in DATASETS:
        p = os.path.join(results_dir, f'paths_{tag}_seed0.jsonl')
        if not os.path.exists(p):
            continue
        _, rows = load(p)
        if len(rows) < 10:
            continue
        vals = per_query(rows)
        report[label] = {}
        for metric, v in vals.items():
            curve = convergence(v)
            n_full, m_full, lo_full, hi_full = curve[-1]
            # how wrong could a five-query benchmark have been?
            five = boot_ci(v[:5]) if len(v) >= 5 else (float('nan'),) * 3
            # every disjoint block of five, to show the spread directly
            blocks = [v[i:i + 5].mean() for i in range(0, len(v) - 4, 5)]
            med, p95 = split_half(v)
            report[label][metric] = dict(
                n=len(v), estimate=m_full, ci=[lo_full, hi_full],
                ci_halfwidth=(hi_full - lo_full) / 2,
                five_query_ci=[five[1], five[2]],
                five_query_block_range=[float(min(blocks)), float(max(blocks))] if blocks else None,
                split_half_median_gap=med, split_half_p95_gap=p95,
                curve=curve)
    return report


def main():
    rep = analyse()
    for ds, metrics in rep.items():
        print(f'\n{ds}')
        for m, r in metrics.items():
            lo, hi = r['ci']
            b = r['five_query_block_range']
            print(f'  {m:40s} n={r["n"]:4d}  {r["estimate"]:+.3f} [{lo:+.3f}, {hi:+.3f}]'
                  f'  +/-{r["ci_halfwidth"]:.3f}   split-half gap p95 {r["split_half_p95_gap"]:.3f}'
                  + (f'   5-query blocks span [{b[0]:+.2f}, {b[1]:+.2f}]' if b else ''))
    os.makedirs('results', exist_ok=True)
    with open('results/reliability.json', 'w') as f:
        json.dump(rep, f, indent=1, default=float)
    print('\nwrote results/reliability.json')


if __name__ == '__main__':
    main()
