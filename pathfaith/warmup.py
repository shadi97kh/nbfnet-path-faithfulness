"""Load the frozen model on a graph once and run one forward pass."""
from __future__ import annotations

import argparse
import contextlib
import io
import time
import warnings

warnings.filterwarnings('ignore')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='CoDExSmall')
    args = ap.parse_args()
    t0 = time.time()
    from pathfaith.loader import build, n_params
    from pathfaith.experiment import prob_and_rank
    b = build(args.dataset)
    t1 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        prob_and_rank(b, b.test_data, 0, 0, 0)
    print(f'{b.name}: {b.num_nodes:,} entities, {b.num_edges:,} edges, '
          f'{b.num_relations // 2} relations')
    print(f'{b.ckpt}: {n_params(b.model):,} parameters, all frozen')
    print(f'graph ready in {t1 - t0:.0f}s, first forward pass (kernel compile) in {time.time() - t1:.0f}s')


if __name__ == '__main__':
    main()
