# Are NBFNet's Path Explanations Actually Faithful?

CAP 6938 AI Knowledge Systems, term project, research track.
Shadi Khodagholi, University of Central Florida.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shadi97kh/nbfnet-path-faithfulness/blob/main/NBFNet_path_faithfulness.ipynb)

## How to run it

Open [`NBFNet_path_faithfulness.ipynb`](NBFNet_path_faithfulness.ipynb) with the
badge above and choose **Runtime > Run all**. That is the only thing to run. The
notebook downloads this repository, installs what it needs and runs on Colab's
free CPU runtime. Setup takes about 3 minutes. Then:

1. **Section 2** is the demo: two runs on different queries.
2. **Section 3** re-runs part of the recorded benchmark and checks every number
   against the saved results.
3. **Section 4** is the neuromodulation experiment: it runs it live on a few
   queries, summarises the saved results on all four graphs and draws the
   dose-response curves.
4. **Section 5** redraws every figure in the report and the talk from the saved
   results, shows them, and zips them into `figures.zip`. It does not run the
   model.
5. **Section 6** holds the long runs behind the report. They are switched off,
   because they take hours and their output is already in `results/`.

The notebook is saved with the outputs of a complete run, so you can read
every result on GitHub without running anything. That run took about
7 minutes on a 2-core CPU.

## What the project does

NBFNet answers a knowledge graph query and explains the answer with a few
paths through the graph, read off the gradient of its score. I test whether
those paths are really what the model used. Instead of reading the
explanation, I change the graph and run the frozen model again:

| | What changes in the graph | The question |
|---|---|---|
| A | nothing; call NBFNet's `visualize()` | what does the method claim? |
| B | delete each of its 20 candidate paths, one at a time | does its ranking match the real effect? |
| C | delete the top 5 together, or keep only them | are they load-bearing? |
| D | point the last hop of the top path at another entity | does the answer follow the edit? |
| E | neuromodulation: turn one relation type's gain from 1 down to 0 | does the gradient predict the effect? |

Every result has a control: deleting the same number of random edges. Nothing
is trained. The model is ULTRA's released `ultra_4g` checkpoint, whose
entity-level reasoner is NBFNet, with every weight frozen.

## Where each part lives

| Part | Code | Saved results | Notebook | Report | Slides |
|---|---|---|---|---|---|
| Live demo | `pathfaith/demo.py` | | 2 | | 8 |
| Path deletions and rewires (A to D) | `experiment.py`, `run_experiment.py` | `results/paths_*` | 3 | 7.1 to 7.3, 7.6 to 7.8 | 9, 10, 14 |
| The message-direction bug | `direction_check.py`, `attribution.flipped` | `results/direction_*` | 3 | 5 | 11 |
| Integrated gradients and the step ablation | `ig.py` | `results/ig_*` | 6 | 7.4 | 12, 13 |
| Neuromodulation (E) | `neuromod.py`, `analyze_neuromod.py` | `results/neuromod_*` | 4 | 4 (E), 7.5 | 4, 15 |
| Link prediction | `run_baseline.py` | `results/baseline*` | 3 | Table 1 | 5 |
| How many queries are enough | `reliability.py` | `results/reliability.json` | 3 | 6 | 6 |
| Every figure | `figures.py`, `figures_extra.py` | `report/figs/` | 5 | all | |

## Results

800 queries (200 on each of four graphs) and 15,447 single-path deletions.
Shares are of the right answer's probability. Tau is Kendall's tau between the
explanation's ranking of its paths and what deleting each one does; 1 would be
a faithful ranking.

| | WN18RR | FB15k-237 | CoDEx-S | NELL23k |
|---|---|---|---|---|
| seen in the checkpoint's training | yes | yes | no | no |
| MRR here (ULTRA paper) | 0.484 (0.480) | 0.371 (0.368) | 0.464 (0.472) | 0.258 (0.239) |
| delete the top 5 paths: share of p removed | 0.858 | 0.259 | 0.427 | 0.368 |
| delete as many random edges | 0.000 | 0.000 | 0.000 | 0.007 |
| tau, explanation as released | -0.009 | 0.048 | 0.076 | 0.041 |
| tau, direction-corrected explanation | 0.082 | 0.153 | 0.185 | 0.103 |
| tau, integrated gradients on the same paths | 0.270 | 0.432 | 0.489 | 0.445 |
| one shared edge recovers this share of the effect | 0.650 | 0.025 | 0.143 | 0.086 |
| silence the strongest explained relation: share of p removed | 0.995 | 0.770 | 0.895 | 0.895 |

Integrated gradients was run on 100 WN18RR and 50 FB15k-237 queries, and the
relation experiment on 100 and 50. The corrected explanation on FB15k-237 has
100 queries. Everything else uses all 200 per graph.

In short: the paths point at edges the model depends on, but their order
means little, and fixing the bug below plus integrated gradients helps without
making the ranking faithful. Relation types act like switches: most of their
effect shows up only as the gain gets close to zero, which a gradient taken at
full gain cannot see. The ULTRA paper's own checkpoint reproduces its reported
MRR here (0.472 on CoDEx-S) and gives the same picture.

## The bug in the released explanation

ULTRA makes its predictions with a fast kernel that sends each message from
`edge_index[1]` to `edge_index[0]`. When a gradient is needed, which is when
`visualize()` runs, the code switches to PyTorch Geometric's generic message
passing, which sends it the other way. So the explanation is the gradient of a
different model: its scores are up to 10.5 logits off, and its gradient has
the wrong sign on 21 of 50 random directions. The fix, `attribution.flipped`,
runs `visualize()` on the reversed graph; the error then drops to 0.7%.
NBFNet's PyTorch Geometric release has the same code pattern, which I read but
did not run.

## Reproducibility

The model is frozen and gives the same output every time, so re-running a
recorded query gives back the recorded numbers. `results/colab_check.txt` is
the notebook run end to end on Colab from this repository, and
`results/reproduction_check.txt` is the same check from a fresh clone on
another machine. Every number matched, to within 0.00005.

## What is in the repository

```
NBFNet_path_faithfulness.ipynb   the notebook to run
pathfaith/                       the code the notebook calls
results/                         every recorded run, one JSON line per query
report/main.pdf                  the 8-page report, and its figures
presentation/                    the talk, NBFNet_Path_Faithfulness.pptx
kg-labels/                       entity and relation names for CoDEx-S and WN18RR
torch_scatter_shim/              used by the notebook only if Colab has no torch_scatter build
requirements.txt                 for running outside Colab
```

To run outside Colab, install `requirements.txt`, clone
[ULTRA](https://github.com/DeepGraphLearning/ULTRA) into `ultra/` at commit
`427966a` (it includes the checkpoints), and run the notebook's cells or
`python -m pathfaith.demo --n 5`.
