"""The four conditions compared in this project.

A  gradient      NBFNet's published explanation, unchanged: run the model with
                 per-edge weights marked for autograd, take d(score)/d(weight),
                 and beam-search the top paths under those gradients. This is
                 `BaseNBFNet.visualize`, called on the authors' own code.
B  hybrid        Take A's shortlist, then delete each path's edges from the
                 graph, re-score, and re-rank by the drop that actually happens.
C  occlusion     The same deletion test over a much wider candidate pool, with
                 no gradient shortlist. Too slow to deploy; it is the ceiling.
D  rewire        An edit rather than a deletion. Redirect the explanation's
                 final edge to a different tail and ask whether the prediction
                 follows it. A path the model truly reasons over should drag
                 the answer with it; a decorative one will not.

Interventions are made on the graph itself, not on a mask, so what the model
sees afterwards is a knowledge graph that genuinely lacks (or genuinely
re-routes) those facts.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch


# --------------------------------------------------------------------------
# graph surgery
# --------------------------------------------------------------------------
def _inverse_type(r: torch.Tensor, num_relations: int) -> torch.Tensor:
    """FB15k-237 and WN18RR are stored with both directions; relation r's
    inverse is r + R/2 (and vice versa)."""
    half = num_relations // 2
    return torch.where(r < half, r + half, r - half)


def find_edges(data, triples: torch.Tensor) -> torch.Tensor:
    """Column indices in data.edge_index for each (h, t, r) row of `triples`,
    including the stored inverse edge. Returns a flat LongTensor."""
    if triples.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    h, t, r = triples.t()
    r_inv = _inverse_type(r, int(data.num_relations))
    want = torch.cat([torch.stack([h, t, r], dim=1),
                      torch.stack([t, h, r_inv], dim=1)], dim=0)

    key_graph = (data.edge_index[0] * data.num_nodes + data.edge_index[1]) \
        * int(data.num_relations) + data.edge_type
    key_want = (want[:, 0] * data.num_nodes + want[:, 1]) \
        * int(data.num_relations) + want[:, 2]
    hit = (key_graph.unsqueeze(0) == key_want.unsqueeze(1)).any(dim=0)
    return hit.nonzero(as_tuple=True)[0]


def _variant(data, edge_index, edge_type):
    """A graph that differs from `data` only in its edges.

    Built field by field rather than with `Data.clone()`, which would copy the
    relation graph as well. That matters here: an occlusion study runs tens of
    thousands of these, and the relation graph is the largest object in the
    bundle.

    The relation graph is also deliberately carried over unchanged. Deleting a
    handful of edges out of hundreds of thousands should not be allowed to move
    the relation representations too, or the measured drop would mix the
    deletion with a second-order change to the conditioning.
    """
    from torch_geometric.data import Data
    out = Data(edge_index=edge_index, edge_type=edge_type,
               num_nodes=int(data.num_nodes))
    out.num_relations = data.num_relations
    out.relation_graph = data.relation_graph
    return out


def drop_edges(data, cols: torch.Tensor):
    """`data` with those edge columns deleted."""
    keep = torch.ones(data.edge_index.shape[1], dtype=torch.bool)
    keep[cols] = False
    return _variant(data, data.edge_index[:, keep], data.edge_type[keep])


def keep_only(data, cols: torch.Tensor):
    """`data` keeping only those edge columns. Used for sufficiency."""
    return _variant(data, data.edge_index[:, cols], data.edge_type[cols])


def inverse_column(data, col: int):
    """The column storing the same fact the other way round, or None."""
    ei, et = data.edge_index, data.edge_type
    h, t, r = int(ei[0, col]), int(ei[1, col]), int(et[col])
    r_inv = int(_inverse_type(torch.tensor(r), int(data.num_relations)))
    hit = ((ei[0] == t) & (ei[1] == h) & (et == r_inv)).nonzero(as_tuple=True)[0]
    return int(hit[0]) if len(hit) else None


def rewire_edge(data, col: int, new_tail: int):
    """`data` with edge `col` pointing at `new_tail` instead.

    The matching inverse edge is moved with it, so the graph stays a consistent
    pair of directions and the edit is a real change of fact rather than a
    half-applied one.
    """
    ei = data.edge_index.clone()
    et = data.edge_type.clone()
    h, t = int(ei[0, col]), int(ei[1, col])
    r = int(et[col])
    r_inv = int(_inverse_type(torch.tensor(r), int(data.num_relations)))

    inv = ((ei[0] == t) & (ei[1] == h) & (et == r_inv)).nonzero(as_tuple=True)[0]
    ei[1, col] = new_tail
    if len(inv):
        ei[0, inv[0]] = new_tail
    return _variant(data, ei, et)


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
@torch.no_grad()
def score(bundle, data, h: int, r: int, tails) -> torch.Tensor:
    """Score (h, r, tail) for every tail in `tails`, on the given graph."""
    tails = torch.as_tensor(tails).view(-1)
    batch = torch.stack([torch.full_like(tails, h), tails,
                         torch.full_like(tails, r)], dim=-1).unsqueeze(0)
    return bundle.model(data, batch).view(-1)


@torch.no_grad()
def rank_of(bundle, data, h: int, r: int, t: int) -> int:
    """Rank of the gold tail among all entities, unfiltered, on `data`."""
    s = score(bundle, data, h, r, torch.arange(data.num_nodes))
    return int((s > s[t]).sum()) + 1


# --------------------------------------------------------------------------
# condition A: the published explanation
# --------------------------------------------------------------------------
@dataclass
class Explanation:
    paths: list = field(default_factory=list)   # each: list of (h, t, r)
    weights: list = field(default_factory=list)  # the method's own score
    seconds: float = 0.0


def flipped(data):
    """The same edge columns with their direction reversed.

    The fused rspmm kernel that ULTRA (and NBFNet-PyG) use for training and
    inference sends each message from edge_index[1] to edge_index[0]. PyG's
    generic message passing, which those layers fall back to whenever edge
    weights need a gradient, sends it from edge_index[0] to edge_index[1].
    The released explanation therefore differentiates a model with every
    message reversed. Running the generic path on the flipped graph computes
    exactly the deployed model (scores agree to 3e-6), so its gradient is the
    real one.
    """
    return _variant(data, data.edge_index.flip(0), data.edge_type)


def explain_gradient(bundle, data, h: int, r: int, t: int,
                     topk: int = 10, num_beam: int = 10,
                     corrected: bool = False) -> Explanation:
    """NBFNet's path explanation for (h, r, t).

    corrected=False is the released code, unchanged. corrected=True runs the
    same code on the flipped graph, so the gradients and the beam search follow
    the direction the deployed model actually passes messages in. Its paths
    are returned as columns of the original graph, (edge_index[0],
    edge_index[1], type), listed in the order the message travels: each hop's
    message goes from its second entity to its first.
    """
    ent = bundle.model.entity_model
    ent.num_beam, ent.path_topk = num_beam, topk

    t0 = time.time()
    with torch.no_grad():
        rel_rep = bundle.model.relation_model(
            data.relation_graph, query=torch.tensor([r]))
    ent.query = rel_rep
    for layer in ent.layers:
        layer.relation = rel_rep

    batch = torch.tensor([[h, t, r]])
    paths, weights = ent.visualize(flipped(data) if corrected else data, batch)
    paths = [[tuple(int(x) for x in e) for e in p] for p in paths]
    if corrected:
        paths = [[(b, a, typ) for a, b, typ in p] for p in paths]
    return Explanation(paths=paths, weights=[float(w) for w in weights],
                       seconds=time.time() - t0)
