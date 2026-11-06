#!/usr/bin/env bash
# One stage of the benchmark, for running in pieces:
#   ./run_stage.sh sweep|fixed|fixed3g|neuromod|ig|direction DS N   or   baseline DS N TAG
# Every stage resumes where it stopped.
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p results logs
stage() { echo "[$(date '+%H:%M:%S')] $*" | tee -a logs/benchmark.log; }
case "$1" in
  sweep)
    lc=$(echo "$2" | tr '[:upper:]' '[:lower:]')
    stage "sweep $2 to $3 queries"
    python3 -m pathfaith.run_experiment --dataset "$2" --n "$3" --pool 20 --topk 5 \
      --seed 0 --path-controls 0 --out "results/paths_${lc}_seed0.jsonl" >> "logs/paths_${lc}.log" 2>&1
    stage "enrich $2"
    python3 -m pathfaith.enrich --dataset "$2" --file "results/paths_${lc}_seed0.jsonl" >> logs/benchmark.log 2>&1 ;;
  fixed|fixed3g)
    # the benchmark again with the direction-corrected explanation; fixed3g
    # does it with ULTRA's 3-graph checkpoint, the ULTRA paper's own model
    lc=$(echo "$2" | tr '[:upper:]' '[:lower:]')
    if [ "$1" = fixed ]; then ck=ultra_4g.pth; f="results/paths_${lc}_fixed_seed0.jsonl"
    else ck=ultra_3g.pth; f="results/paths_${lc}_3g_fixed_seed0.jsonl"; fi
    stage "$1 $2 to $3 queries"
    python3 -m pathfaith.run_experiment --dataset "$2" --n "$3" --pool 20 --topk 5 \
      --seed 0 --path-controls 0 --corrected --ckpt "$ck" --out "$f" >> "logs/$1_${lc}.log" 2>&1
    python3 -m pathfaith.enrich --dataset "$2" --file "$f" >> logs/benchmark.log 2>&1 ;;
  ig)
    lc=$(echo "$2" | tr '[:upper:]' '[:lower:]')
    stage "integrated gradients $2 to $3 queries"
    python3 -m pathfaith.ig --dataset "$2" --n "$3" >> "logs/ig_${lc}.log" 2>&1 ;;
  neuromod)
    lc=$(echo "$2" | tr '[:upper:]' '[:lower:]')
    stage "neuromod $2 to $3 queries"
    python3 -m pathfaith.neuromod --dataset "$2" --n "$3" >> "logs/neuromod_${lc}.log" 2>&1 ;;
  direction)
    # the message-direction check behind the direction figure
    lc=$(echo "$2" | tr '[:upper:]' '[:lower:]')
    stage "direction check $2 on $3 queries"
    python3 -m pathfaith.direction_check --dataset "$2" --n "$3" > "logs/direction_${lc}.log" 2>&1 ;;
  baseline)
    stage "baseline $2 on ${3} test triples (0 = all)"
    python3 -m pathfaith.run_baseline --dataset "$2" --n "$3" --batch-size 8 \
      --out "results/baseline_$4.json" > "logs/baseline_$4.log" 2>&1 ;;
esac
stage "done: $*"
