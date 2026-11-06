"""Figures for the report.

Each curve is drawn as one continuous line across the whole range it covers,
with its uncertainty as a band rather than as separate error bars, so a reader
can see the shape before reading any number.
"""
from __future__ import annotations

import math
import os

import matplotlib
import matplotlib.ticker
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from pathfaith.analyze import load, boot

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8.5,
    'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 200, 'savefig.bbox': 'tight', 'lines.linewidth': 1.4,
})
BLUE, RED, GREY = '#2B6CB0', '#C0392B', '#7F8C8D'


def _band(ax, x, series, color, label):
    """Mean with a 95% bootstrap band, as one unbroken line over all of x."""
    mean, lo, hi = [], [], []
    for col in series:
        m, a, b = boot(col)
        mean.append(m); lo.append(a); hi.append(b)
    ax.plot(x, mean, color=color, label=label, zorder=3)
    ax.fill_between(x, lo, hi, color=color, alpha=0.18, lw=0, zorder=2)


def fig_rank_vs_drop(paths: dict, out: str, row: bool = None) -> None:
    """The headline: what the method ranks first is not what matters most."""
    n = len(paths)
    row = (n > 2) if row is None else row
    # one row of panels across the page once there are more than two graphs
    if row:
        fig, axes = plt.subplots(1, n, figsize=(6.9, 1.85), squeeze=False, sharey=True)
        panel_axes = axes[0, :]
    else:
        fig, axes = plt.subplots(n, 1, figsize=(3.3, 1.9 * n), squeeze=False, sharex=True)
        panel_axes = axes[:, 0]
    for ax, (name, p) in zip(panel_axes, paths.items()):
        rows = load(p)[1] if os.path.exists(p) else []
        rows = [r for r in rows if r['true_drops'] and r['base_prob'] > 0]
        if not rows:
            ax.set_title(f'{name}  (pending)', loc='left')
            continue
        # Every query that reached a given position contributes to it, so the
        # curve runs across the whole pool instead of stopping at the shortest
        # candidate list. Positions with too few queries to bootstrap are cut.
        kmax = max(len(r['true_drops']) for r in rows)
        drops, x = [], []
        for i in range(kmax):
            col = [r['true_drops'][i] / r['base_prob']
                   for r in rows if len(r['true_drops']) > i]
            if len(col) >= max(5, 0.2 * len(rows)):
                drops.append(col)
                x.append(i + 1)
        _band(ax, np.array(x), drops, BLUE, 'measured drop when deleted')

        ctrl = [r['comp_random'] / r['base_prob'] for r in rows]
        m, lo, hi = boot(ctrl)
        n = len(x)
        ax.plot(x, [m] * n, color=RED, ls='--', label='random edges, same count')
        ax.fill_between(x, [lo] * n, [hi] * n, color=RED, alpha=0.15, lw=0)

        ax.set_title(f'{name}  (n={len(rows)})', loc='left')
        ax.set_xlim(min(x), max(x))
        ax.set_ylim(-0.05, 1.05)
    if row:
        panel_axes[0].set_ylabel('share of p(tail)\nremoved')
        for ax in panel_axes:
            ax.set_xlabel('position in its ranking')
        handles, labels = panel_axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False,
                   bbox_to_anchor=(0.5, -0.04))
        fig.tight_layout(pad=0.3, rect=(0, 0.1, 1, 1))
    else:
        for ax in panel_axes:
            ax.set_ylabel('share of p(tail)\nremoved')
        panel_axes[-1].set_xlabel('path position in the method\'s own ranking')
        panel_axes[0].legend(loc='lower left', frameon=False)
        fig.subplots_adjust(hspace=0.3)
    fig.savefig(out)
    plt.close(fig)


def _cdf(ax, values, color, label, ls='-'):
    v = np.sort(np.asarray([x for x in values if np.isfinite(x)], float))
    if len(v) == 0:
        return
    y = np.arange(1, len(v) + 1) / len(v)
    ax.plot(np.concatenate([v[:1], v]), np.concatenate([[0.0], y]),
            color=color, ls=ls, label=label, zorder=3)


def fig_set_level(paths: dict, out: str) -> None:
    """Deleting the explanation does matter. Keeping only it does not suffice."""
    n = len(paths)
    fig, axes = plt.subplots(n, 2, figsize=(3.3, 1.9 * n), squeeze=False)
    axes = axes.reshape(-1)
    col = 0
    for name, p in paths.items():
        _, rows = load(p)
        base = np.array([r['base_prob'] for r in rows])

        ax = axes[col]; col += 1
        _cdf(ax, np.array([r['comp_topk'] for r in rows]) / base, BLUE,
             'top-5 deleted')
        _cdf(ax, np.array([r['comp_random'] for r in rows]) / base, RED,
             'random', ls='--')
        _cdf(ax, np.array([r['bottleneck_drop'] for r in rows]) / base, GREY,
             'busiest edge', ls=':')
        ax.set_title(f'{name}: removed', loc='left')
        # a few deletions raise the probability, so the axis starts slightly
        # below zero rather than stretching to the single worst outlier
        ax.set_xlim(-0.25, 1.05)
        ax.set_xlabel('share of p(tail) removed')
        ax.set_ylabel('fraction of queries')
        ax.legend(loc='lower right', frameon=False, fontsize=5.5)

        ax = axes[col]; col += 1
        _cdf(ax, np.array([r['suff_topk'] for r in rows]) / base, BLUE,
             'kept alone')
        ax.axvline(1.0, color=GREY, lw=0.8, ls=':')
        ax.set_title(f'{name}: retained', loc='left')
        ax.set_xlabel('p(tail) kept vs full graph')
        ax.set_xscale('log')
        ax.legend(loc='lower right', frameon=False, fontsize=5.5)
    fig.tight_layout(pad=0.3)
    fig.savefig(out)
    plt.close(fig)


def fig_redundancy(paths: dict, out: str) -> None:
    """The k paths are mostly one path shown k times."""
    fig, axes = plt.subplots(1, len(paths), figsize=(6.9, 2.3), squeeze=False)
    for ax, (name, p) in zip(axes[0], paths.items()):
        _, rows = load(p)
        _cdf(ax, [r['bottleneck_frac'] for r in rows], BLUE,
             'share of paths using one common edge')
        _cdf(ax, [r['n_distinct_edges'] / max(1, sum(r['path_lengths']) * 2)
                  for r in rows if r['path_lengths']], GREY,
             'distinct edges / edges listed', ls='--')
        ax.set_title(name)
        ax.set_xlabel('value')
        ax.set_xlim(0, 1)
        ax.legend(loc='upper left', frameon=False)
    axes[0][0].set_ylabel('fraction of queries')
    fig.savefig(out)
    plt.close(fig)


def make_all(paths: dict, outdir: str = 'report/figs') -> list:
    os.makedirs(outdir, exist_ok=True)
    made = []
    path = os.path.join(outdir, 'rank_vs_drop.pdf')
    fig_rank_vs_drop(paths, path, row=True)
    made.append(path)
    have = {k: v for k, v in paths.items() if os.path.exists(v)}
    for name, fn in (('set_level', fig_set_level), ('redundancy', fig_redundancy)):
        path = os.path.join(outdir, f'{name}.pdf')
        fn(have, path)
        made.append(path)
    path = os.path.join(outdir, 'convergence.pdf')
    # one row across the page: the report is held to eight pages
    fig_convergence(path, figsize=(6.9, 2.15), rows=1)
    made.append(path)
    return made


def make_slide_figs(paths: dict, outdir: str = 'video/figs') -> list:
    """Single-panel versions, sized to be read from the back of a room.

    The report's figures put both datasets side by side, which is right on a
    page and unreadable on a slide. These are the same data, one panel each,
    with larger type and thicker lines.
    """
    os.makedirs(outdir, exist_ok=True)
    old = dict(plt.rcParams)
    plt.rcParams.update({'font.size': 13, 'axes.labelsize': 13,
                         'axes.titlesize': 14, 'legend.fontsize': 12,
                         'xtick.labelsize': 12, 'ytick.labelsize': 12,
                         'lines.linewidth': 2.4})
    made = []
    for name, p in paths.items():
        tag = name.lower().replace('-', '').replace('_', '')
        _, rows = load(p)
        rows = [r for r in rows if r['true_drops'] and r['base_prob'] > 0]

        fig, ax = plt.subplots(figsize=(7.4, 3.6))
        kmax = max(len(r['true_drops']) for r in rows)
        drops, x = [], []
        for i in range(kmax):
            col = [r['true_drops'][i] / r['base_prob']
                   for r in rows if len(r['true_drops']) > i]
            if len(col) >= max(5, 0.2 * len(rows)):
                drops.append(col); x.append(i + 1)
        _band(ax, np.array(x), drops, BLUE, 'deleting this path')
        m, lo, hi = boot([r['comp_random'] / r['base_prob'] for r in rows])
        ax.plot(x, [m] * len(x), color=RED, ls='--', label='deleting random edges')
        ax.fill_between(x, [lo] * len(x), [hi] * len(x), color=RED, alpha=0.15, lw=0)
        ax.set_xlabel("position in the method's own ranking")
        ax.set_ylabel('share of p(tail) removed')
        ax.set_xlim(min(x), max(x)); ax.set_ylim(-0.05, 1.0)
        ax.legend(loc='center left', frameon=False)
        f = os.path.join(outdir, f'rank_vs_drop_{tag}.pdf')
        fig.savefig(f); plt.close(fig); made.append(f)

        fig, ax = plt.subplots(figsize=(7.4, 3.6))
        _cdf(ax, [r['bottleneck_frac'] for r in rows], BLUE,
             'paths sharing one common edge')
        _cdf(ax, [r['n_distinct_edges'] / max(1, sum(r['path_lengths']) * 2)
                  for r in rows if r['path_lengths']], GREY,
             'distinct edges / edges listed', ls='--')
        ax.set_xlabel('value'); ax.set_ylabel('fraction of queries')
        ax.set_xlim(0, 1); ax.legend(loc='upper left', frameon=False)
        f = os.path.join(outdir, f'redundancy_{tag}.pdf')
        fig.savefig(f); plt.close(fig); made.append(f)

    plt.rcParams.update(old)
    return made


PALETTE = {'WN18RR': '#2B6CB0', 'FB15k-237': '#C0392B',
           'CoDEx-S': '#27AE60', 'NELL23k': '#8E44AD'}


def fig_convergence(out: str, results_dir: str = 'results',
                    metrics=None, figsize=(6.9, 4.2), big=False, rows=None) -> None:
    """How each headline estimate settles as queries are added.

    The queries were drawn in a fixed random order before anything was
    measured, so the first n of them are an honest sample of size n. Each
    curve is the running estimate with its 95% bootstrap band, drawn as one
    unbroken line from n = 5 to the full sample. The grey dots at the left are
    every disjoint block of five queries, which is what a five-query benchmark
    could have reported.
    """
    from pathfaith.reliability import DATASETS, per_query, convergence
    metrics = metrics or ['delete shortlist (share of p removed)',
                          'one edge recovers (share of effect)',
                          'paths sharing one edge',
                          'top path minus chance']
    old = dict(plt.rcParams)
    if big:
        plt.rcParams.update({'font.size': 12, 'axes.labelsize': 12, 'axes.titlesize': 12.5,
                             'legend.fontsize': 11, 'xtick.labelsize': 11,
                             'ytick.labelsize': 11, 'lines.linewidth': 2.2})
    rows_n = rows or (2 if len(metrics) > 2 else 1)
    cols_n = math.ceil(len(metrics) / rows_n)
    fig, axes = plt.subplots(rows_n, cols_n, figsize=figsize, squeeze=False)
    axes = axes.reshape(-1)
    loaded = {}
    for label, tag in DATASETS:
        p = os.path.join(results_dir, f'paths_{tag}_seed0.jsonl')
        if os.path.exists(p):
            _, rows = load(p)
            if len(rows) >= 10:
                loaded[label] = per_query(rows)
    for ax, metric in zip(axes, metrics):
        for label, vals in loaded.items():
            v = vals[metric]
            if len(v) < 10:
                continue
            col = PALETTE.get(label, GREY)
            blocks = [v[i:i + 5].mean() for i in range(0, len(v) - 4, 5)]
            ax.scatter(np.full(len(blocks), 5.0) * np.exp(np.random.default_rng(1)
                       .uniform(-0.08, 0.08, len(blocks))),
                       blocks, s=6 if not big else 14, color=col, alpha=0.35, lw=0, zorder=1)
            curve = convergence(v)
            n = np.array([c[0] for c in curve], float)
            m = np.array([c[1] for c in curve])
            lo = np.array([c[2] for c in curve])
            hi = np.array([c[3] for c in curve])
            ax.fill_between(n, lo, hi, color=col, alpha=0.15, lw=0, zorder=2)
            ax.plot(n, m, color=col, label=f'{label} (n={len(v)})', zorder=3)
        ax.set_xscale('log')
        # plain counts on the axis; scientific notation hides the point
        ticks = [t for t in (5, 10, 20, 50, 100, 200, 400, 800)
                 if t <= max(len(v[metric]) for v in loaded.values()) * 1.05]
        ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter([str(t) for t in ticks]))
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        short = {'delete shortlist (share of p removed)': 'shortlist deleted',
                 'one edge recovers (share of effect)': 'one edge recovers',
                 'paths sharing one edge': 'paths sharing one edge',
                 'top path minus chance': 'top path minus chance'}
        ax.set_title(short.get(metric, metric) if rows_n == 1 and not big else metric, loc='left')
        ax.axvline(5, color=GREY, lw=0.6, ls=':')
        if 'chance' in metric or 'recovers' in metric:
            ax.axhline(0, color=GREY, lw=0.6)
    for ax in axes[len(metrics):]:
        ax.set_visible(False)
    for ax in axes[-cols_n:]:
        ax.set_xlabel('queries (log scale)' if rows_n == 1 else 'number of queries (log scale)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), frameon=False,
               fontsize=7 if not big else 11, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(pad=0.4, rect=(0, (0.05 if rows_n > 1 else 0.09) if not big else 0.06, 1, 1))
    fig.savefig(out)
    plt.close(fig)
    plt.rcParams.update(old)


# ------------------------------------------------------------------ neuromodulation
ORANGE = '#D68910'
NM_SETS = (('WN18RR', 'wn18rr'), ('FB15k-237', 'fb15k237'),
           ('CoDEx-S', 'codexsmall'), ('NELL23k', 'nell23k'))


def fig_dose_response(out: str, results_dir: str = 'results', figsize=(6.9, 1.8),
                      big: bool = False) -> None:
    """Turning a relation channel's gain down from 1 to 0, one panel per graph.

    Each curve is the share of p(answer) removed, averaged over queries, with a
    95% bootstrap band, drawn as one line from gain 1 to gain 0. The dashed
    line is what the gradient at gain 1 predicts for the explained channels if
    the response were linear.
    """
    from pathfaith.analyze_neuromod import rows_of, per_query
    n = len(NM_SETS)
    if big:
        fig, axes = plt.subplots(2, 2, figsize=figsize, squeeze=False)
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(1, n, figsize=figsize, squeeze=False, sharey=False)
        axes = axes[0]
    x5 = np.array([1.0, 0.75, 0.5, 0.25, 0.0])
    x3 = np.array([1.0, 0.5, 0.0])
    handles = None
    for ax, (label, tag) in zip(axes, NM_SETS):
        p = os.path.join(results_dir, f'neuromod_{tag}_seed0.jsonl')
        rows = rows_of(p)[1] if os.path.exists(p) else []
        if not rows:
            ax.set_title(f'{label}  (pending)', loc='left')
            ax.set_xlim(1, 0)
            continue
        _, curves = per_query(rows)
        for role, color, name, x in (('explained', BLUE, 'channels the explanation uses', x5),
                                     ('gradient', ORANGE, 'channels the gradient ranks first', x5),
                                     ('random', GREY, 'random channels', x3)):
            if curves[role]:
                cols = np.array(curves[role]).T
                _band(ax, x, cols, color, name)
        if curves['pred_explained']:
            pred = np.array(curves['pred_explained']).mean(axis=0)
            ax.plot(x5, pred, color=BLUE, ls='--', lw=1.0, label="gradient's linear prediction")
        ax.axhline(0, color='0.6', lw=0.6, zorder=1)
        ax.set_xlim(1, 0)
        ax.set_xticks([1, 0.75, 0.5, 0.25, 0])
        ax.set_xticklabels(['1', '.75', '.5', '.25', '0'])
        ax.set_title(f'{label} (n={len(rows)})', loc='left')
        ax.set_xlabel('gain on the channel')
        handles = ax.get_legend_handles_labels()
    axes[0].set_ylabel('share of p removed')
    if big:
        axes[2].set_ylabel('share of p removed')
    if handles:
        fig.legend(*handles, loc='lower center', ncol=4, frameon=False,
                   fontsize=7 if not big else 11, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(pad=0.4, rect=(0, 0.1 if not big else 0.07, 1, 1))
    fig.savefig(out)
    plt.close(fig)
