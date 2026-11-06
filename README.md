# Are NBFNet's Path Explanations Actually Faithful?

CAP 6938 AI Knowledge Systems, term project, research track.
Shadi Khodagholi, University of Central Florida.

NBFNet explains a link prediction by taking the gradient of its score with
respect to every edge and beam searching the highest-scoring paths. Those
paths are presented as the reason for the answer. This repository tests that
claim by changing the graph instead of inspecting the explanation: delete the
paths, keep only the paths, or point them somewhere else, rerun the model, and
measure what the prediction does. It also turns whole relation types down,
like a neuromodulator, to see whether the gradient predicts the effect.

The audit found a bug in the released explanation code, shared with
NBFNet-PyG: the explanation differentiates the model with every message
reversed (see "The message-direction bug" below). Every result is reported
for the explanation as released and for the corrected one.

Nothing here trains anything. Every number comes from ULTRA's released
`ultra_4g.pth` checkpoint, whose entity-level reasoner is NBFNet, with every
parameter frozen.

## What is in this repository

* [`report/main.pdf`](report/main.pdf): the 8-page report (CVPR 2026 template), built from `report/main.tex`
* [`presentation/talk.pdf`](presentation/talk.pdf): the slides; `video/narration.md` is the script read over them
* `pathfaith/`: all the code; `results/`: every recorded run, one JSON line per query
* `colab/pathfaith_colab.ipynb`: the whole project on Google Colab, CPU runtime

## Results at a glance

800 correctly answered queries over four graphs (200 each), 15,447 single-path
deletions. Shares are of the gold answer's probability; tau is Kendall's tau
between the explanation's ranking of its paths and what deleting each one does
(1 would be a faithful ranking).

| | WN18RR | FB15k-237 | CoDEx-S | NELL23k |
|---|---|---|---|---|
| MRR of the frozen checkpoint (ULTRA paper, 3-graph model) | 0.484 (0.480) | 0.371 (0.368) | 0.464 (0.472) | 0.258 (0.239) |
| delete the top 5 paths: share of p removed | 0.858 | 0.259 | 0.427 | 0.368 |
| delete as many random edges | 0.000 | 0.000 | 0.000 | 0.007 |
| tau, explanation as released | -0.009 | 0.048 | 0.076 | 0.041 |
| tau, direction-corrected explanation | 0.082 | 0.153 | 0.185 | 0.103 |
| tau, integrated gradients on the same paths | | | 0.489 | 0.445 |
| one shared edge recovers this share of the effect | 0.650 | 0.025 | 0.143 | 0.086 |
| silence the main relation channel: share of p removed | 0.995 | 0.770 | 0.895 | 0.895 |

* **A bug in the released explanation.** It differentiates the model with every
  message reversed: scores up to 10.5 logits off, gradient off by as much as
  the effect it predicts, wrong sign on 21 of 50 random directions. Fixed, the
  error is 0.7%.
* **The selected edges are load-bearing; the order is not.** Fixing the bug
  helps a little, integrated gradients help more, and neither makes the
  ranking faithful.
* **Relation channels act as switches.** Most of a channel's effect appears
  only as its gain nears zero, which a gradient taken at full gain cannot see.
* ULTRA's own 3-graph checkpoint reproduces the paper's MRR here, 0.472 on
  CoDEx-S and 0.241 on NELL23k, and gives the same picture.

## Quick start: the live demo

```bash
git clone https://github.com/shadi97kh/nbfnet-path-faithfulness.git
cd nbfnet-path-faithfulness
pip install -r requirements.txt
git clone https://github.com/DeepGraphLearning/ULTRA.git ultra
git -C ultra checkout 427966ad8ed60420eef034063d44f3153addff90
python -m pathfaith.demo --n 5
```

ULTRA's repository carries the frozen checkpoints (`ultra/ckpts/`). The graphs
download on first use.

The demo loads the frozen checkpoint on CPU, picks five test queries the model
answers correctly on CoDEx-S (a graph the checkpoint never saw, with Wikidata
names), and for each one prints NBFNet's own explanation and then runs every
intervention live, one forward pass per line:

* delete the shown paths together, and see the answer drop
* delete the same number of random edges, the control, and see it not move
* delete only the busiest edge, the one most paths share, and see how much of
  the effect it carries alone
* reroute the last hop to another entity, and see whether probability follows
* turn the main relation channel down to half and to zero, and check whether
  the gradient at full gain had the right sign

A line under the explanation shows the bug live: the released gradient is
taken on a model that gives the answer a different probability from the
deployed model.

It ends with a count of how many queries completed every intervention, then
reads the full benchmark off disk so the five live queries are visibly one
slice of it. `--seed 7` picks a different five, `--dataset WN18RR` runs on
WordNet with synset names. About 15 seconds per query on two CPU cores.

## Running on Google Colab

Open `colab/pathfaith_colab.ipynb` in Colab (File, Upload notebook) and run
all cells. The first cell asks for `CAP6938_pathfaith_code.zip`, or reads it
from Google Drive or a git URL if you set `SOURCE`. The default CPU runtime is
enough. The notebook then:

1. installs PyTorch Geometric, clones ULTRA at the pinned commit, and installs
   `torch_scatter` (a prebuilt wheel if one exists for Colab's PyTorch,
   otherwise `colab/torch_scatter_shim`, a pure-PyTorch package with the same
   functions that matches the compiled extension exactly),
2. runs the demo twice with different seeds, and optionally on WN18RR,
3. re-runs the first 10 recorded CoDEx-S queries and checks every number
   against `results/` with `pathfaith.compare_runs`, then does the same for
   the direction-corrected explanation,
4. shows the message-direction bug on three queries,
5. re-runs the neuromodulation experiment and compares it the same way,
6. reproduces link prediction on the full CoDEx-S test set,
7. redraws the reliability analysis and the dose-response figure.

Setup takes about three minutes and each demo run one to two.

`video/record_demo.py` records a real run under a pseudo-terminal and renders
it to video, with the true elapsed time in every frame; it refuses to render a
run that failed.

## What is measured

| | What changes in the graph | The question it answers |
|---|---|---|
| A | nothing; call NBFNet's `visualize()` | what does the method claim? |
| B | delete each of its 20 candidate paths, one at a time | does its ranking match the real effect? |
| C | delete the top 5 together, or keep only them | are they load-bearing? |
| D | point the last hop of the top path at another entity | does the answer follow the edit? |
| E | turn one relation channel's gain from 1 down to 0 | does the gradient predict the dose response? |

Conditions A to D are run with the explanation as released and with the
direction-corrected one, and the corrected one again on ULTRA's 3-graph
checkpoint (the ULTRA paper's model). Integrated gradients re-ranks the
corrected explanation's candidate paths, as the standard fix to test.

Every condition removes or rewires real edges and reruns the model; ULTRA's
relation graph is held fixed so the change is only to the entity graph.
Controls: the same number of random edges, the bottom of the method's own
candidate list, the tail's degree, and whether the query's reverse fact is
already in the graph.

## The benchmark

| Graph | In `ultra_4g`'s training mix | Released | Corrected | 3g, corrected | Integrated gradients | Channels |
|---|---|---|---|---|---|---|
| WN18RR | yes | 200 | 200 | | | 100 |
| FB15k-237 | yes | 200 | 100 | | | 50 |
| CoDEx-S | no | 200 | 200 | 200 | 200 | 200 |
| NELL23k | no | 200 | 200 | 200 | 200 | 200 |

On each graph the checkpoint first reproduces link prediction on the full
test set with ULTRA's own evaluation code (`run_baseline.py`). Explained
queries are test triples drawn in a random order fixed by the seed before
anything was measured, kept when the model ranks the gold tail in its top
ten. Every interval is a 95% bootstrap over queries; the model is
deterministic, so queries are the only source of variance.

`reliability.py` checks that the sample is large enough instead of assuming
it. Because the order was fixed in advance, the first *n* queries are an honest
sample of size *n*, so it plots each headline estimate as queries are added,
reports split-half agreement, and shows what every disjoint block of five
queries would have concluded.

Every stage resumes where it stopped, so it can be run in pieces:

```bash
./run_stage.sh sweep     CoDExSmall 200   # released explanation, conditions A to D
./run_stage.sh fixed     CoDExSmall 200   # the same with the corrected explanation
./run_stage.sh fixed3g   CoDExSmall 200   # corrected, on ULTRA's 3-graph checkpoint
./run_stage.sh neuromod  CoDExSmall 200   # condition E (needs the fixed stage first)
./run_stage.sh ig        CoDExSmall 200   # integrated gradients (needs the fixed stage first)
./run_stage.sh direction CoDExSmall 30    # the message-direction check
./run_stage.sh baseline  CoDExSmall 0 codexsmall   # MRR on the full test set
python -m pathfaith.reliability
```

On two CPU cores a CoDEx-S or NELL23k stage takes 10 to 30 minutes; WN18RR and
FB15k-237 are slower. The ultra_3g baselines and the full-test-set WN18RR and
NELL23k baselines were run on Colab with the same code.

Datasets download on first use into `kg-datasets/` (PyTorch Geometric for
WN18RR and FB15k-237, ULTRA's loaders for CoDEx-S and NELL23k).

## Rebuilding every deliverable

```bash
./finalize.sh             # numbers, figures, worked example, report, deck, script, video, audit, zip
./finalize.sh --record    # the same, recording a fresh demo run first
```

No number in the report, the slides or the narration is typed by hand.
`make_numbers.py` writes `report/numbers.tex`; the report and the deck use its
macros, and `make_narration.py` fills the spoken script from the same file.
Any macro the data does not yet define renders as **??**, and
`check_report.py` fails the build if one survives, if an em dash appears, if a
citation or figure is missing, if the body passes eight pages, if a quoted
query count disagrees with its results file, or if the recorded demo did not
complete every query.

## Layout

```
pathfaith/
  loader.py          build the frozen ULTRA/NBFNet model and a dataset on CPU
  attribution.py     graph surgery (delete, keep, rewire) and condition A
  experiment.py      one query through conditions A to D
  neuromod.py        condition E: relation channels as neuromodulators
  analyze_neuromod.py  summaries for condition E, and nDCG and MRR of the paths
  ig.py              integrated gradients on the same queries
  direction_check.py the message-direction check behind the bug
  run_baseline.py    filtered MRR and Hits@k on the full test set
  run_experiment.py  sweep queries, append to results/*.jsonl, resumable
  enrich.py          add degree and reverse-fact controls to a results file
  analyze.py         bootstrap summaries, Kendall's tau with explicit ties
  reliability.py     convergence curves and split-half agreement
  figures.py         the benchmark figures in the report and on the slides
  figures_extra.py   the bug, released vs corrected, integrated gradients, channels
  case_study.py      the worked example, re-run and checked against its record
  make_numbers.py    every quoted number as a LaTeX macro
  make_narration.py  the narration script, numbers filled from the same macros
  demo.py            the live demo
  warmup.py          load a graph and compile ULTRA's kernel before the demo
  compare_runs.py    check a fresh run against the recorded benchmark
  check_report.py    the pre-submission audit
colab/               pathfaith_colab.ipynb and the torch_scatter stand-in
report/              main.tex refs.bib numbers.tex case.tex figs/
presentation/        talk.tex, the slides for the live presentation
video/               narration_template.md, build_video.sh, record_demo.py
kg-labels/           CoDEx-S entity and relation labels, WN18RR synset names
results/             baselines (*.json) and one JSON line per query (*.jsonl)
```

## The message-direction bug

ULTRA (like NBFNet-PyG) runs its layers with a fused `rspmm` kernel that sends
the message on edge column `e` from `edge_index[1, e]` to `edge_index[0, e]`.
That is the model used for training and prediction. When edge weights need a
gradient, which is exactly when `visualize()` runs, the layers fall back to
PyTorch Geometric's generic message passing, which sends it from
`edge_index[0, e]` to `edge_index[1, e]`. So the explanation is the gradient of
a different model. The fix is `attribution.flipped`: run `visualize()` on the
graph with every edge reversed and map each hop back to its original column.
`direction_check.py` shows the two code paths give scores up to about 10 logits
apart, that the flipped graph reproduces the deployed model to about 1e-5, and
that the corrected gradient matches the deployed model's measured response
while the released one often has the wrong sign.

## Notes on the setup

**Why ULTRA.** Its entity-level reasoner is NBFNet unchanged, in a class the
repository calls `EntityNBFNet`, and unlike NBFNet it ships pre-trained
checkpoints. Training NBFNet would have put training noise into a comparison
that is about explanations, and the checkpoint lets the same frozen model be
tested on graphs it saw and graphs it did not.

**Why the relation graph is held fixed.** ULTRA conditions the entity-level
pass on relation representations computed over a graph of relations.
Rebuilding that after removing a handful of edges would mix the deletion with a
second change to the conditioning.

**Why keep-only is not read as sufficiency.** With a handful of edges left,
most rival entities are cut off as well, so the gold tail can gain probability
just from having fewer competitors. The retained share is reported, and no
conclusion about sufficiency is drawn from it.

**WN18RR node merges.** The offset-only release PyTorch Geometric loads maps
41,105 distinct synsets onto 40,943 nodes, because WordNet offsets repeat
across parts of speech. The demo shows both names for the 161 merged nodes.
