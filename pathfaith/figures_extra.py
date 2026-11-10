"""Figures for the direction fix, the second checkpoint, integrated gradients and the
relation channels."""
from __future__ import annotations

import contextlib
import json
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from pathfaith.analyze import boot, kendall_tau, TIE, order_tau, order_top_gap
from pathfaith.figures import BLUE, RED, GREY, ORANGE, PALETTE, _band

GREEN, PURPLE = '#1E8449', '#7D3C98'
GRAPHS = (('WN18RR', 'wn18rr'), ('FB15k-237', 'fb15k237'),
          ('CoDEx-S', 'codexsmall'), ('NELL23k', 'nell23k'))
SETTINGS = (('released explanation', 'paths_{}_seed0.jsonl', BLUE, 'o'),
            ('corrected explanation', 'paths_{}_fixed_seed0.jsonl', GREEN, 's'),
            ('corrected, ULTRA 3g', 'paths_{}_3g_fixed_seed0.jsonl', PURPLE, '^'))
IG_SETTING = ('integrated gradients', 'ig_{}_seed0.jsonl', ORANGE, 'D')
MEANINGFUL = 0.05


def _rows(path):
    if not os.path.exists(path):
        return []
    out = []
    for line in open(path):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get('kind') == 'query':
            out.append(d)
    return out


@contextlib.contextmanager
def _style(big):
    old = dict(plt.rcParams)
    if big:
        plt.rcParams.update({'font.size': 12, 'axes.labelsize': 12, 'axes.titlesize': 12.5,
                             'legend.fontsize': 11, 'xtick.labelsize': 11,
                             'ytick.labelsize': 11, 'lines.linewidth': 2.2})
    try:
        yield
    finally:
        plt.rcParams.update(old)


def _cdf_full(ax, values, color, label, lo, hi, ls='-'):
    v = np.sort(np.clip(np.asarray([x for x in values if np.isfinite(x)], float), lo, hi))
    if len(v) == 0:
        return
    y = np.arange(1, len(v) + 1) / len(v)
    xs = np.concatenate([[lo], np.repeat(v, 2), [hi]])
    ys = np.concatenate([[0.0, 0.0], np.repeat(y, 2)[:-1], [1.0]])
    ax.plot(xs, ys, color=color, ls=ls, label=label, zorder=3)


def _pending(ax, title):
    ax.set_title(f'{title}  (pending)', loc='left')
    ax.set_xticks([]); ax.set_yticks([])


def _symlog(ax, lin, which='both'):
    if which in ('both', 'x'):
        ax.set_xscale('symlog', linthresh=lin)
    if which in ('both', 'y'):
        ax.set_yscale('symlog', linthresh=lin)


def fig_direction(out, results_dir='results', figsize=(6.9, 1.95), big=False):
    """The released explanation differentiates a different function."""
    recs = {lab: _rows(os.path.join(results_dir, f'direction_{tag}.jsonl'))
            for lab, tag in GRAPHS}
    recs = {k: v for k, v in recs.items() if v}
    with _style(big):
        if big:
            fig, axes = plt.subplots(1, 3, figsize=figsize)
        else:
            fig, axes = plt.subplots(1, 3, figsize=figsize)
        a, b, c = axes
        if not recs:
            for ax, t in zip(axes, ('(a) scores', '(b) gradient check', '(c) one channel')):
                _pending(ax, t)
            fig.tight_layout(pad=0.4); fig.savefig(out); plt.close(fig)
            return

        first = next(iter(recs.values()))
        kept = [d for d in first if 'scores' in d]
        dep = np.concatenate([d['scores']['deployed'] for d in kept])
        rel = np.concatenate([d['scores']['released'] for d in kept])
        cor = np.concatenate([d['scores']['corrected'] for d in kept])
        s = 2 if not big else 5
        a.scatter(dep, rel, s=s, color=BLUE, alpha=0.25, lw=0, label='released code path',
                  rasterized=True)
        a.scatter(dep, cor, s=s, color=GREEN, alpha=0.6, lw=0, label='corrected code path',
                  rasterized=True)
        lo, hi = float(min(dep.min(), rel.min())), float(max(dep.max(), rel.max()))
        a.plot([lo, hi], [lo, hi], color='k', lw=0.7, zorder=0)
        a.set_xlabel('score, deployed model')
        a.set_ylabel('score on the code path\nthe explanation uses')
        gap = max(d['gap_released'] for d in kept)
        a.set_title(f'(a) {len(kept)} queries, {next(iter(recs))}', loc='left')
        a.text(0.03, 0.97, f'largest gap here {gap:.1f}', transform=a.transAxes, va='top',
               fontsize=plt.rcParams['legend.fontsize'])
        a.legend(loc='upper left', bbox_to_anchor=(0.0, 0.9), frameon=False,
                 markerscale=3 if not big else 2, handletextpad=0.2)

        fd = np.array([d['fd'] for v in recs.values() for d in v])
        dr = np.array([d['dir_released'] for v in recs.values() for d in v])
        dc = np.array([d['dir_corrected'] for v in recs.values() for d in v])
        lin = float(np.percentile(np.abs(fd), 10)) or 1e-4
        ms = 9 if not big else 28
        b.scatter(fd, dr, s=ms, color=BLUE, alpha=0.7, lw=0, label='released gradient')
        b.scatter(fd, dc, s=ms, color=GREEN, alpha=0.8, lw=0, marker='s',
                  label='corrected gradient')
        m = float(np.abs(np.concatenate([fd, dr, dc])).max()) * 1.3
        grid = np.concatenate([-np.geomspace(m, lin / 10, 60), [0], np.geomspace(lin / 10, m, 60)])
        b.plot(grid, grid, color='k', lw=0.7, zorder=0)
        _symlog(b, lin)
        b.set_xlim(-m, m); b.set_ylim(-m, m)
        b.set_xlabel('measured change in p, deployed model')
        b.set_ylabel('gradient $\\cdot$ direction')
        b.set_title(f'(b) {len(fd)} queries, random direction', loc='left')
        b.legend(loc='upper left', frameon=False, handletextpad=0.2)
        ex = [10.0 ** e for e in range(-8, 2) if 10 * lin <= 10.0 ** e <= m]
        ex = ex[::-1][::2][::-1] if len(ex) > 4 else ex
        ticks = [-x for x in ex[::-1]] + [0.0] + ex
        labels = ['0' if t == 0 else f'$-10^{{{int(np.log10(-t))}}}$' if t < 0
                  else f'$10^{{{int(np.log10(t))}}}$' for t in ticks]
        for axis in (b.xaxis, b.yaxis):
            axis.set_major_locator(mticker.FixedLocator(ticks))
            axis.set_major_formatter(mticker.FixedFormatter(labels))
            axis.set_minor_locator(mticker.NullLocator())

        chans = [(d, d['channel']) for v in recs.values() for d in v if 'channel' in d]
        if chans:
            def score(x):
                ch = x[1]
                wrong_sign = ch['grad_released'] * ch['grad_corrected'] < 0
                return (wrong_sign, min(abs(ch['grad_released']), abs(ch['grad_corrected']))
                        / ch['p1'])
            d, ch = max(chans, key=score)
            g = np.array(ch['gains']); p = np.array(ch['p'])
            c.plot(g, p, color='k', label='deployed model', zorder=3)
            c.plot(g, ch['p1'] + ch['grad_corrected'] * (g - 1), color=GREEN, ls='--',
                   label='corrected tangent')
            c.plot(g, ch['p1'] + ch['grad_released'] * (g - 1), color=BLUE, ls=':',
                   label='released tangent')
            c.scatter([1.0], [ch['p1']], color='k', s=12 if not big else 30, zorder=4)
            span = p.max() - p.min()
            c.set_ylim(p.min() - 0.25 * span, p.max() + 0.25 * span)
            c.set_xlim(g.min(), g.max())
            c.set_xlabel('gain on one relation channel')
            c.set_ylabel('p(answer)')
            c.set_title('(c) one query, one channel', loc='left')
            c.legend(loc='best', frameon=False, handletextpad=0.3)
        else:
            _pending(c, '(c) one channel')
        fig.tight_layout(pad=0.4)
        fig.savefig(out, dpi=300)
        plt.close(fig)


def _position_curves(rows):
    rows = [r for r in rows if r['true_drops'] and r['base_prob'] > 0]
    if not rows:
        return None, None
    kmax = max(len(r['true_drops']) for r in rows)
    cols, x = [], []
    for i in range(kmax):
        col = [r['true_drops'][i] / r['base_prob'] for r in rows if len(r['true_drops']) > i]
        if len(col) >= max(5, 0.2 * len(rows)):
            cols.append(col); x.append(i + 1)
    return np.array(x), cols


def fig_released_vs_corrected(out, results_dir='results', figsize=(6.9, 1.95), big=False,
                              graphs=GRAPHS):
    """The method's ranking, before and after the direction fix."""
    with _style(big):
        if big:
            fig, axes = plt.subplots(2, 2, figsize=figsize, squeeze=False)
            axes = axes.flatten()
        else:
            fig, axes = plt.subplots(1, len(graphs), figsize=figsize, squeeze=False, sharey=True)
            axes = axes[0]
        handles = None
        for ax, (lab, tag) in zip(axes, graphs):
            drew = False
            ns, xs = [], []
            for name, pat, color, _ in SETTINGS[:2]:
                rows = _rows(os.path.join(results_dir, pat.format(tag)))
                x, cols = _position_curves(rows)
                if x is None:
                    continue
                _band(ax, x, cols, color, name)
                ns.append(str(len(rows))); xs.append(x)
                drew = True
                if name.startswith('released'):
                    m, lo, hi = boot([r['comp_random'] / r['base_prob'] for r in rows
                                      if r['base_prob'] > 0])
                    ax.plot(x, [m] * len(x), color=RED, ls='--', label='random edges, same count')
                    ax.fill_between(x, [lo] * len(x), [hi] * len(x), color=RED, alpha=0.12, lw=0)
            if not drew:
                _pending(ax, lab)
                continue
            ax.set_title(f'{lab} (n={"/".join(ns)})', loc='left')
            ax.set_xlim(min(v.min() for v in xs), max(v.max() for v in xs))
            ax.set_ylim(-0.05, 1.0)
            ax.set_xlabel('position in its ranking')
            handles = ax.get_legend_handles_labels()
        axes[0].set_ylabel('share of p removed')
        if big:
            axes[2].set_ylabel('share of p removed')
        if handles:
            fig.legend(*handles, loc='lower center', ncol=3, frameon=False,
                       bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout(pad=0.4, rect=(0, 0.1 if not big else 0.07, 1, 1))
        fig.savefig(out)
        plt.close(fig)


def _rank_mass(drops):
    d = np.round(np.asarray(drops, float) / TIE) * TIE
    n = len(d)
    M = np.zeros((n, n))
    for i in range(n):
        above = int((d > d[i]).sum())
        tied = int((d == d[i]).sum())
        M[i, above:above + tied] = 1.0 / tied
    return M


def fig_rank_heatmap(out, results_dir='results', pattern='paths_{}_seed0.jsonl',
                     figsize=(6.9, 1.85), big=False, k=20):
    """Where the method's i-th path really ranks, pooled over queries."""
    with _style(big):
        if big:
            fig, axes = plt.subplots(1, 4, figsize=figsize, squeeze=False)
        else:
            fig, axes = plt.subplots(1, 4, figsize=figsize, squeeze=False)
        axes = axes[0]
        im = None
        for ax, (lab, tag) in zip(axes, GRAPHS):
            rows = [r for r in _rows(os.path.join(results_dir, pattern.format(tag)))
                    if len(r['true_drops']) == k]
            if not rows:
                _pending(ax, lab)
                continue
            H = sum(_rank_mass(r['true_drops']) for r in rows) / len(rows)
            im = ax.imshow(H, origin='upper', cmap='Blues', vmin=0, vmax=0.25,
                           extent=(0.5, k + 0.5, k + 0.5, 0.5), interpolation='nearest')
            ax.plot([0.5, k + 0.5], [0.5, k + 0.5], color=RED, lw=0.6, ls=':')
            ax.set_title(f'{lab} (n={len(rows)})', loc='left')
            ax.set_xticks([1, 5, 10, 15, 20]); ax.set_yticks([1, 5, 10, 15, 20])
            ax.set_xlabel('measured rank')
        axes[0].set_ylabel("method's rank")
        if im is not None:
            cb = fig.colorbar(im, ax=list(axes), fraction=0.02, pad=0.01)
            cb.set_label('share of queries')
            cb.ax.axhline(1 / k, color=RED, lw=0.8)      # what a random order gives
        fig.savefig(out, dpi=300)
        plt.close(fig)


def _channel_points(rows):
    pred, meas, half = [], [], []
    for d in rows:
        base = d['base_prob']
        if base <= 0:
            continue
        for c in d['channels']:
            if c['role'] not in ('explained', 'gradient'):
                continue
            full = (base - c['doses']['0.0'][0]) / base
            pred.append(c['grad'] / base)
            meas.append(full)
            if abs(full) >= MEANINGFUL and '0.5' in c['doses']:
                half.append(((base - c['doses']['0.5'][0]) / base) / full)
    return np.array(pred), np.array(meas), np.array(half)


def fig_channels(out, results_dir='results', figsize=(6.9, 2.1), big=False):
    """(a) the gradient's prediction of silencing a channel against the measured effect,
    every tested channel on every graph; (b) how much of the full effect is already
    there at half gain, where a linear response gives 0.5."""
    with _style(big):
        fig, (a, b) = plt.subplots(1, 2, figsize=figsize,
                                   gridspec_kw=dict(width_ratios=(1.15, 1)))
        any_rows = False
        allv = []
        for lab, tag in GRAPHS:
            rows = _rows(os.path.join(results_dir, f'neuromod_{tag}_seed0.jsonl'))
            if not rows:
                continue
            any_rows = True
            pred, meas, half = _channel_points(rows)
            col = PALETTE.get(lab, GREY)
            agree = float(np.mean(np.sign(pred[np.abs(meas) >= MEANINGFUL]) ==
                                  np.sign(meas[np.abs(meas) >= MEANINGFUL]))) \
                if np.any(np.abs(meas) >= MEANINGFUL) else float('nan')
            a.scatter(pred, meas, s=6 if not big else 18, color=col, alpha=0.45, lw=0,
                      label=f'{lab} (sign right {agree:.0%})', rasterized=True)
            allv.append(np.abs(np.concatenate([pred, meas])))
            if len(half):
                _cdf_full(b, half, col, f'{lab} ({len(half)} channels)', -0.5, 1.5)
        if not any_rows:
            _pending(a, '(a) gradient vs silencing'); _pending(b, '(b) half dose')
            fig.tight_layout(pad=0.4); fig.savefig(out); plt.close(fig)
            return
        v = np.concatenate(allv)
        lin = 1e-3
        m = float(max(v.max(), 1.0)) * 1.3
        grid = np.concatenate([-np.geomspace(m, lin / 10, 60), [0], np.geomspace(lin / 10, m, 60)])
        a.plot(grid, grid, color='k', lw=0.7, zorder=0, label='linear prediction exact')
        a.axhline(0, color='0.7', lw=0.5, zorder=0); a.axvline(0, color='0.7', lw=0.5, zorder=0)
        _symlog(a, lin)
        a.set_xlim(-m, m); a.set_ylim(-1.05, 1.05)
        a.set_yscale('symlog', linthresh=lin)
        a.set_xlabel('gradient at gain 1, as share of p')
        a.set_ylabel('share of p removed\nby silencing')
        a.set_title('(a) gradient against silencing', loc='left')
        a.legend(loc='upper left', frameon=False, fontsize=plt.rcParams['legend.fontsize'] - 0.5,
                 handletextpad=0.2, markerscale=1.5)

        b.axvline(0.5, color='k', lw=0.7, ls='--', label='linear dose response')
        b.set_xlim(-0.5, 1.5)
        b.set_xlabel('share of the full effect present at half gain')
        b.set_ylabel('fraction of channels')
        b.set_title('(b) switches, not dials', loc='left')
        b.legend(loc='lower right', frameon=False)
        fig.tight_layout(pad=0.4)
        fig.savefig(out, dpi=300)
        plt.close(fig)


def fig_ig(out, results_dir='results', figsize=(6.9, 1.95), big=False,
           graphs=(('CoDEx-S', 'codexsmall'), ('NELL23k', 'nell23k'))):
    """Does the textbook fix rank better?"""
    with _style(big):
        if big:
            fig, axes = plt.subplots(2, 2, figsize=figsize, squeeze=False)
            axes = axes.flatten()
        else:
            fig, axes = plt.subplots(1, 4, figsize=figsize, squeeze=False, sharey=True)
            axes = axes[0]
        handles = None
        for j, (lab, tag) in enumerate(graphs):
            ig = {(d['h'], d['r'], d['t']): d
                  for d in _rows(os.path.join(results_dir, f'ig_{tag}_seed0.jsonl'))}
            nm = {(d['h'], d['r'], d['t']): d
                  for d in _rows(os.path.join(results_dir, f'neuromod_{tag}_seed0.jsonl'))}
            ax = axes[j]
            tg, ti = [], []
            for d in ig.values():
                a_, b_ = order_tau(d['true_drops']), order_tau(d['true_drops'], d['path_ig'])
                if a_ == a_ and b_ == b_:
                    tg.append(a_); ti.append(b_)
            if tg:
                _cdf_full(ax, tg, GREEN, f'gradient (mean {np.mean(tg):.2f})', -1, 1)
                _cdf_full(ax, ti, ORANGE, f'integrated gradients ({np.mean(ti):.2f})', -1, 1, ls='--')
                ax.axvline(0, color='0.6', lw=0.6)
                ax.set_xlim(-1, 1)
                ax.set_title(f'{lab}, paths (n={len(tg)})', loc='left')
                ax.legend(loc='upper left', frameon=False,
                          fontsize=plt.rcParams['legend.fontsize'] - 1, handlelength=1.6)
            else:
                _pending(ax, f'{lab}, paths')
            ax = axes[j + len(graphs)]
            cg, ci = [], []
            for key, d in nm.items():
                if key not in ig:
                    continue
                cs = [c for c in d['channels'] if c['role'] in ('explained', 'gradient')]
                if len(cs) < 3 or d['base_prob'] <= 0:
                    continue
                eff = [(d['base_prob'] - c['doses']['0.0'][0]) / d['base_prob'] for c in cs]
                a_ = kendall_tau([c['grad'] for c in cs], eff)
                b_ = kendall_tau([ig[key]['channel_ig'][c['c']] for c in cs], eff)
                if a_ == a_ and b_ == b_:
                    cg.append(a_); ci.append(b_)
            if cg:
                _cdf_full(ax, cg, GREEN, f'gradient (mean {np.mean(cg):.2f})', -1, 1)
                _cdf_full(ax, ci, ORANGE, f'integrated gradients ({np.mean(ci):.2f})', -1, 1, ls='--')
                ax.axvline(0, color='0.6', lw=0.6)
                ax.set_xlim(-1, 1)
                ax.set_title(f'{lab}, channels (n={len(cg)})', loc='left')
                ax.legend(loc='upper left', frameon=False,
                          fontsize=plt.rcParams['legend.fontsize'] - 1, handlelength=1.6)
            else:
                _pending(ax, f'{lab}, channels')
        for ax in axes:
            if ax.get_title() and 'pending' not in ax.get_title():
                ax.set_xlabel("Kendall's $\\tau$ with measured effect")
        axes[0].set_ylabel('fraction of queries')
        if big:
            axes[2].set_ylabel('fraction of queries')
        fig.tight_layout(pad=0.4)
        fig.savefig(out)
        plt.close(fig)


def _per_query_metrics(rows, score='grad_weights'):
    comp, tau, top, nd = [], [], [], []
    from pathfaith.analyze_neuromod import _dcg
    for r in rows:
        b = r['base_prob']
        d = list(r['true_drops'])
        if b <= 0 or len(d) < 2:
            if b > 0 and 'comp_topk' in r and score == 'grad_weights':
                comp.append(r['comp_topk'] / b)
            continue
        if 'comp_topk' in r and score == 'grad_weights':
            comp.append(r['comp_topk'] / b)
        sc = None if score == 'grad_weights' else list(r[score])[:len(d)]
        t = order_tau(d, sc)
        if t == t:
            tau.append(t)
        top.append(order_top_gap(d, sc))
        order = list(range(len(d))) if sc is None else sorted(range(len(d)), key=lambda i: -sc[i])
        rel = [max(d[i], 0.0) / b for i in order]
        ideal = _dcg(sorted(rel, reverse=True), 5)
        if ideal > 0:
            disc = sum(1 / np.log2(i + 2) for i in range(min(5, len(rel))))
            nd.append(_dcg(rel, 5) / ideal - (sum(rel) / len(rel)) * disc / ideal)
    return {'share of p removed\nby the top 5 paths': comp,
            "Kendall's $\\tau$, ranking\nvs measured effect": tau,
            'top path is the most\ndamaging, minus chance': top,
            'nDCG@5 minus chance': nd}


def fig_robustness(out, results_dir='results', figsize=(6.9, 1.85), big=False):
    """Every headline number under the released explanation, the corrected one, the
    corrected one on a second checkpoint, and integrated gradients re-ranking the
    corrected candidates, with 95% intervals."""
    with _style(big):
        metrics = None
        table = {}
        settings = SETTINGS + (IG_SETTING,)
        for lab, tag in GRAPHS:
            for name, pat, color, mk in settings:
                rows = _rows(os.path.join(results_dir, pat.format(tag)))
                if len(rows) < 10:
                    continue
                score = 'path_ig' if name == IG_SETTING[0] else 'grad_weights'
                vals = _per_query_metrics(rows, score)
                metrics = metrics or list(vals)
                table[(lab, name)] = {k: boot(v) for k, v in vals.items() if v}
        if big:
            fig, axes = plt.subplots(2, 2, figsize=figsize, squeeze=False, sharey=True)
            axes = axes.flatten()
        else:
            fig, axes = plt.subplots(1, 4, figsize=figsize, squeeze=False, sharey=True)
            axes = axes[0]
        if not table:
            for ax in axes:
                _pending(ax, '')
            fig.savefig(out); plt.close(fig)
            return
        labels = [lab for lab, _ in GRAPHS]
        offs = dict(zip([x[0] for x in settings], (-0.27, -0.09, 0.09, 0.27)))
        for ax, m in zip(axes, metrics):
            for gi, lab in enumerate(labels):
                for name, _, color, mk in settings:
                    if (lab, name) not in table or m not in table[(lab, name)]:
                        continue
                    est, lo, hi = table[(lab, name)][m]
                    y = gi + offs[name]
                    ax.plot([lo, hi], [y, y], color=color, lw=1.7 if not big else 2.6)
                    ax.scatter([est], [y], color=color, marker=mk, s=24 if not big else 70,
                               zorder=3, label=name)
            if 'chance' in m or 'tau' in m:
                ax.axvline(0, color='0.6', lw=0.6, zorder=0)
            ax.set_title(m, loc='left')
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels)
            ax.set_ylim(len(labels) - 0.5, -0.5)
        handles = {}
        for ax in axes:
            for h, l in zip(*ax.get_legend_handles_labels()):
                handles.setdefault(l, h)
        order = [x[0] for x in settings if x[0] in handles]
        fig.legend([handles[k] for k in order], order, loc='lower center', ncol=4,
                   frameon=False, bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout(pad=0.4, rect=(0, 0.1 if not big else 0.07, 1, 1))
        fig.savefig(out)
        plt.close(fig)


def make_extra(results_dir='results', report='report/figs', ext='.pdf'):
    os.makedirs(report, exist_ok=True)
    jobs = [
        ('direction', fig_direction),
        ('released_vs_corrected', fig_released_vs_corrected),
        ('rank_heatmap', fig_rank_heatmap),
        ('rank_heatmap_corrected', fig_rank_heatmap),
        ('channels', fig_channels),
        ('ig', fig_ig),
        ('robustness', fig_robustness),
    ]
    made = []
    for name, fn in jobs:
        kw = dict(pattern='paths_{}_fixed_seed0.jsonl') if name == 'rank_heatmap_corrected' else {}
        p = os.path.join(report, name + ext)
        fn(p, results_dir=results_dir, **kw)
        made.append(p)
    return made

if __name__ == '__main__':
    for p in make_extra():
        print(p)
