"""Load a frozen ULTRA checkpoint and a knowledge graph, on CPU.

ULTRA's entity-level reasoner is NBFNet (`ultra.models.EntityNBFNet`), and the
checkpoints shipped in the repository were pre-trained with FB15k-237 and
WN18RR in the mixture. Nothing here trains or fine-tunes anything: every
experiment in this project runs against the released weights exactly as
published, so no training noise enters the comparison.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import torch

# ULTRA's own dataset classes reload their processed cache with a bare
# torch.load, which PyTorch 2.6+ refuses unless legacy unpickling is allowed.
# The files are ones this machine just built from the public raw triples.
os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ULTRA_ROOT = os.environ.get('ULTRA_ROOT', os.path.join(REPO, 'ultra'))
DATA_ROOT = os.environ.get('KG_DATA_ROOT', os.path.join(REPO, 'kg-datasets'))
if ULTRA_ROOT not in sys.path:
    sys.path.insert(0, ULTRA_ROOT)

# the hyperparameters in ULTRA's config/transductive/inference.yaml, inlined so
# that a reader can see the whole model definition without chasing a yaml file
REL_MODEL = dict(class_='RelNBFNet', input_dim=64, hidden_dims=[64] * 6,
                 message_func='distmult', aggregate_func='sum',
                 short_cut=True, layer_norm=True)
ENT_MODEL = dict(class_='EntityNBFNet', input_dim=64, hidden_dims=[64] * 6,
                 message_func='distmult', aggregate_func='sum',
                 short_cut=True, layer_norm=True)


@dataclass
class Bundle:
    """Everything an experiment needs, in one object."""
    model: torch.nn.Module
    train_data: object   # message-passing graph plus the training targets
    valid_data: object
    test_data: object
    name: str
    ckpt: str

    @property
    def num_nodes(self) -> int:
        return int(self.train_data.num_nodes)

    @property
    def num_relations(self) -> int:
        return int(self.train_data.num_relations)

    @property
    def num_edges(self) -> int:
        return int(self.train_data.edge_index.shape[1])


def build(dataset: str = 'FB15k237', ckpt: str = 'ultra_4g.pth',
          seed: int = 0) -> Bundle:
    """Return a frozen model and the three data splits for `dataset`."""
    torch.manual_seed(seed)
    from ultra import datasets as ultra_datasets
    from ultra.models import Ultra

    builder = getattr(ultra_datasets, dataset)
    ds = builder(root=DATA_ROOT)
    train_data, valid_data, test_data = ds[0], ds[1], ds[2]

    rel_cfg = dict(REL_MODEL)
    ent_cfg = dict(ENT_MODEL)
    rel_cfg['class'] = rel_cfg.pop('class_')
    ent_cfg['class'] = ent_cfg.pop('class_')
    model = Ultra(rel_model_cfg=rel_cfg, entity_model_cfg=ent_cfg)

    path = ckpt if os.path.isabs(ckpt) else os.path.join(ULTRA_ROOT, 'ckpts', ckpt)
    state = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'] if 'model' in state else state)

    model = model.to('cpu').eval()
    for p in model.parameters():          # frozen, everywhere, for every condition
        p.requires_grad_(False)

    return Bundle(model=model, train_data=train_data, valid_data=valid_data,
                  test_data=test_data, name=dataset, ckpt=os.path.basename(path))


def n_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
