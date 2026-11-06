"""Run the four conditions on one query and report what actually moved.

The measurement that decides the paper is not the size of any single drop, it
is whether deleting the paths the method ranks first hurts more than deleting
an equally long path picked at random. A drop that a random path reproduces is
not evidence that the explanation found the mechanism.

Scores are turned into a probability over all entities before anything is
compared, so a drop means the same thing from one query to the next, and the
gold tail's rank is reported alongside as a measure that does not depend on
the scale of the logits at all.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

import torch

from pathfaith import attribution as A


# --------------------------------------------------------------------------
def prob_and_rank(bundle, data, h: int, r: int, t: int):
    """p(t) over all entities, and t's rank. One forward pass."""
    s = A.score(bundle, data, h, r, torch.arange(int(data.num_nodes)))
    p = torch.softmax(s, dim=0)
    rank = int((s > s[t]).sum()) + 1
    return float(p[t]), rank, s


def path_edges(data, path) -> torch.Tensor:
    """Edge columns for every hop of a path."""
    if not path:
        return torch.empty(0, dtype=torch.long)
    tri = torch.tensor([[hh, tt, rr] for hh, tt, rr in path], dtype=torch.long)
    return A.find_edges(data, tri)


def random_path_like(data, path, rng: random.Random):
    """A path with the same hop count whose relations match, but whose
    intermediate entities are drawn at random. The control for condition A."""
    if not path:
        return []
    n = int(data.num_nodes)
    out, cur = [], path[0][0]
    for i, (_, _, rel) in enumerate(path):
        nxt = path[-1][1] if i == len(path) - 1 else rng.randrange(n)
        out.append((cur, nxt, rel))
        cur = nxt
    return out


def random_real_edges(data, k: int, rng: random.Random) -> torch.Tensor:
    """k edge columns drawn uniformly from the graph. The control for the
    joint deletion, matched on how many edges are removed."""
    m = data.edge_index.shape[1]
    return torch.tensor([rng.randrange(m) for _ in range(k)], dtype=torch.long)


# --------------------------------------------------------------------------
@dataclass
class QueryResult:
    h: int = 0
    r: int = 0
    t: int = 0
    base_prob: float = 0.0
    base_rank: int = 0
    n_forward: int = 0
    seconds: float = 0.0
    grad_rank_order: list = field(default_factory=list)   # gradient's ranking
    true_drops: list = field(default_factory=list)        # measured drop per path
    grad_weights: list = field(default_factory=list)
    path_lengths: list = field(default_factory=list)
    overlap: list = field(default_factory=list)           # shared edges w/ others
    control_drops: list = field(default_factory=list)     # matched random paths
    comp_topk: float = 0.0        # joint deletion of the top-k paths
    comp_random: float = 0.0      # joint deletion of as many random edges
    comp_rank_after: int = 0
    suff_topk: float = 0.0        # keep only the top-k paths
    rewire_follow: float = 0.0    # did the answer move with the edited path
    rewire_base_drop: float = 0.0
    # is the shortlist many independent explanations or one wearing k hats
    n_paths: int = 0
    n_distinct_edges: int = 0
    bottleneck_frac: float = 0.0     # share of paths using the most common edge
    bottleneck_drop: float = 0.0     # deleting that one edge, on its own
    bottleneck_rank_after: int = 0
    notes: str = ''


def run_query(bundle, data, h: int, r: int, t: int, *, pool: int = 20,
              topk: int = 5, seed: int = 0, path_controls: int = 5,
              corrected: bool = False) -> QueryResult:
    rng = random.Random(seed * 1000003 + h * 31 + r)
    res = QueryResult(h=h, r=r, t=t)
    t0 = time.time()

    base_p, base_rank, base_scores = prob_and_rank(bundle, data, h, r, t)
    res.base_prob, res.base_rank, res.n_forward = base_p, base_rank, 1

    # ---- condition A: the published explanation ------------------------
    exp = A.explain_gradient(bundle, data, h, r, t, topk=pool, num_beam=pool,
                             corrected=corrected)
    paths = exp.paths[:pool]
    res.grad_weights = exp.weights[:pool]
    res.path_lengths = [len(p) for p in paths]
    if not paths:
        res.notes = 'no paths returned'
        res.seconds = time.time() - t0
        return res

    cols = [path_edges(data, p) for p in paths]

    # how much does each path share with the rest of the shortlist
    for i, c in enumerate(cols):
        mine = set(c.tolist())
        others = set().union(*[set(x.tolist()) for j, x in enumerate(cols) if j != i]) \
            if len(cols) > 1 else set()
        res.overlap.append(len(mine & others) / max(1, len(mine)))

    # is this one explanation or many? count distinct edges and the busiest one
    all_cols = torch.cat(cols) if cols else torch.empty(0, dtype=torch.long)
    res.n_paths = len(paths)
    res.n_distinct_edges = int(torch.unique(all_cols).numel())
    if all_cols.numel():
        vals, counts = torch.unique(all_cols, return_counts=True)
        busiest = int(vals[int(counts.argmax())])
        res.bottleneck_frac = float(counts.max()) / max(1, len(paths))
        p_b, rank_b, _ = prob_and_rank(
            bundle, A.drop_edges(data, torch.tensor([busiest])), h, r, t)
        res.bottleneck_drop, res.bottleneck_rank_after = base_p - p_b, rank_b
        res.n_forward += 1

    # ---- conditions B and C: delete one path at a time ------------------
    # Every candidate in the pool is occluded, so the gradient ranking can be
    # compared against the drop that each path actually causes. The pool's own
    # bottom half is the control: if the paths the method ranks last hurt as
    # much as the ones it ranks first, the ranking carries no information.
    for i, c in enumerate(cols):
        p_i, _, _ = prob_and_rank(bundle, A.drop_edges(data, c), h, r, t)
        res.true_drops.append(base_p - p_i)
        res.n_forward += 1

        if i < path_controls:   # a small per-path null; optional, see above
            cc = random_real_edges(data, len(c), rng)
            p_c, _, _ = prob_and_rank(bundle, A.drop_edges(data, cc), h, r, t)
            res.control_drops.append(base_p - p_c)
            res.n_forward += 1

    res.grad_rank_order = list(range(len(paths)))  # already gradient-ordered

    # ---- comprehensiveness and sufficiency on the top-k ----------------
    joint = torch.unique(torch.cat(cols[:topk])) if cols[:topk] else torch.empty(0, dtype=torch.long)
    p_wo, rank_wo, _ = prob_and_rank(bundle, A.drop_edges(data, joint), h, r, t)
    res.comp_topk, res.comp_rank_after = base_p - p_wo, rank_wo
    res.n_forward += 1

    rnd = random_real_edges(data, len(joint), rng)
    p_rnd, _, _ = prob_and_rank(bundle, A.drop_edges(data, rnd), h, r, t)
    res.comp_random = base_p - p_rnd
    res.n_forward += 1

    p_only, _, _ = prob_and_rank(bundle, A.keep_only(data, joint), h, r, t)
    res.suff_topk = p_only
    res.n_forward += 1

    # ---- condition D: rewire the explanation's terminal edge -----------
    top = paths[0]
    term = path_edges(data, [top[-1]])
    if len(term):
        alt = int(base_scores.argsort(descending=True)[1])   # the runner-up tail
        if alt == t:
            alt = int(base_scores.argsort(descending=True)[2])
        col = int(term[0])
        if corrected:
            # a corrected hop's message arrives at edge_index[0]; rewiring its
            # stored inverse moves that end, and rewire_edge moves the pair
            col = A.inverse_column(data, col)
        rew = A.rewire_edge(data, col, alt) if col is not None else None
    if len(term) and rew is not None:
        s_rew = A.score(bundle, rew, h, r, torch.arange(int(data.num_nodes)))
        p_rew = torch.softmax(s_rew, dim=0)
        res.rewire_base_drop = base_p - float(p_rew[t])
        res.rewire_follow = float(p_rew[alt]) - float(torch.softmax(base_scores, 0)[alt])
        res.n_forward += 1

    res.seconds = time.time() - t0
    return res
