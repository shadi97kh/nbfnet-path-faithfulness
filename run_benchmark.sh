#!/usr/bin/env bash
# Every run behind the report, in the order they depend on each other, with
# the query counts that are in results/. Every stage resumes where it
# stopped, so this can be stopped and restarted at any point. On two CPU
# cores the whole thing takes most of a day, FB15k-237 and WN18RR most of it.
set -uo pipefail
cd "$(dirname "$0")"
S=./run_stage.sh

# 1. the explanation as released, conditions A to D, 200 queries per graph
for DS in WN18RR FB15k237 CoDExSmall NELL23k; do $S sweep "$DS" 200; done

# 2. the same with the direction-corrected explanation, and on ULTRA's
#    3-graph checkpoint for the held-out graphs
for DS in WN18RR CoDExSmall NELL23k; do $S fixed "$DS" 200; done
$S fixed FB15k237 100
for DS in CoDExSmall NELL23k; do $S fixed3g "$DS" 200; done

# 3. condition E, relation channels (reads the corrected queries)
$S neuromod CoDExSmall 200; $S neuromod NELL23k 200
$S neuromod WN18RR 100;     $S neuromod FB15k237 50

# 4. integrated gradients on the corrected candidates, the step-count
#    ablation, and the completeness check with the released gradient
$S ig CoDExSmall 200; $S ig NELL23k 200; $S ig WN18RR 100; $S ig FB15k237 100
$S ig CoDExSmall 50 32; $S ig CoDExSmall 50 64
$S ig CoDExSmall 30 released

# 5. the message-direction check
$S direction CoDExSmall 30; $S direction NELL23k 20

# 6. link prediction for the reproduction table. The ultra_3g runs, the full
#    WN18RR and NELL23k test sets and the FB15k-237 sample were run on Colab
#    with colab/baseline_jobs.py; this is the same thing locally
[ -f results/baseline_codexsmall.json ]    || $S baseline CoDExSmall 0 codexsmall
[ -f results/baseline_wn18rr_full.json ]   || $S baseline WN18RR 0 wn18rr_full
[ -f results/baseline_nell23k_full.json ]  || $S baseline NELL23k 0 nell23k_full
[ -f results/baseline_fb15k237_2000.json ] || $S baseline FB15k237 2000 fb15k237_2000

python3 -m pathfaith.reliability
python3 -m pathfaith.figures
