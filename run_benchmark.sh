#!/usr/bin/env bash
# The full benchmark. Every stage is resumable, so this can be stopped and
# restarted at any point without losing finished queries.
#
#   four graphs: two in ultra_4g's pre-training mixture (WN18RR, FB15k-237)
#   and two held out from it (CoDEx-S, NELL23k)
#   200 correctly answered queries per graph, 20 candidate paths each
#   link prediction on the test sets, for the reproduction table
#   the neuromodulatory channel experiment on the same queries
#
# About ten hours on two CPU cores, most of it FB15k-237.
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p results logs

stage() { echo "[$(date '+%H:%M:%S')] $*" | tee -a logs/benchmark.log; }

sweep() {  # dataset, target n
  local DS=$1 N=$2 lc
  lc=$(echo "$DS" | tr '[:upper:]' '[:lower:]')
  stage "sweep $DS to $N queries"
  python3 -m pathfaith.run_experiment --dataset "$DS" --n "$N" --pool 20 --topk 5 \
    --seed 0 --path-controls 0 --out "results/paths_${lc}_seed0.jsonl" \
    >> "logs/paths_${lc}.log" 2>&1
  stage "enrich $DS"
  python3 -m pathfaith.enrich --dataset "$DS" --file "results/paths_${lc}_seed0.jsonl" \
    >> logs/benchmark.log 2>&1
}

baseline() {  # dataset, number of test triples (0 = all), output tag
  stage "baseline $1 on $2 test triples (0 = all)"
  python3 -m pathfaith.run_baseline --dataset "$1" --n "$2" --batch-size 8 \
    --out "results/baseline_$3.json" > "logs/baseline_$3.log" 2>&1
}

# 1. the path benchmark: every graph to 200 queries, in the fixed order
for DS in WN18RR CoDExSmall NELL23k FB15k237; do sweep "$DS" 200; done

# 2. reproduction: full test sets, and a 2,000-triple sample of FB15k-237,
#    whose full test set alone takes over two hours on two cores
[ -f results/baseline_codexsmall.json ]      || baseline CoDExSmall 0 codexsmall
[ -f results/baseline_wn18rr_full.json ]     || baseline WN18RR 0 wn18rr_full
[ -f results/baseline_nell23k_full.json ]    || baseline NELL23k 0 nell23k_full
[ -f results/baseline_fb15k237_2000.json ]   || baseline FB15k237 2000 fb15k237_2000

# 3. neuromodulatory channels, on the same queries in the same order
for DS in CoDExSmall NELL23k WN18RR; do ./run_stage.sh neuromod "$DS" 200; done
./run_stage.sh neuromod FB15k237 100

stage "ALL DONE"
