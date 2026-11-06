#!/usr/bin/env bash
# Rebuild every deliverable from whatever is in results/, then audit and zip.
#
# Safe to run as often as you like, including while run_benchmark.sh is still
# going: everything here is derived from the results files, so re-running after
# more queries land just refreshes the numbers, figures, report, deck, script
# and video together.
#
#   ./finalize.sh            everything, using video/demo_run.mp4 if it exists
#   ./finalize.sh --record   also record a fresh demo run first (a few minutes)
set -euo pipefail
cd "$(dirname "$0")"

GRAPHS="WN18RR FB15k237 CoDExSmall NELL23k"
lc() { echo "$1" | tr '[:upper:]' '[:lower:]'; }

echo "== 1. structural controls =="
# enrich refuses on a file that is still growing, so a graph mid-sweep is left
# for the benchmark's own enrich step
for DS in $GRAPHS; do
  f="results/paths_$(lc "$DS")_seed0.jsonl"
  [ -f "$f" ] && { python3 -m pathfaith.enrich --dataset "$DS" --file "$f" || true; }
done

echo
echo "== 2. reliability, numbers, figures, worked example =="
python3 -m pathfaith.reliability > logs/reliability.txt && tail -1 logs/reliability.txt
python3 -m pathfaith.make_numbers --out report/numbers.tex
python3 - <<'PY'
import os
from pathfaith.figures import make_all, fig_convergence
have = {name: f'results/paths_{tag}_seed0.jsonl'
        for name, tag in (('WN18RR', 'wn18rr'), ('FB15k-237', 'fb15k237'),
                          ('CoDEx-S', 'codexsmall'), ('NELL23k', 'nell23k'))}
made = make_all(have)
os.makedirs('video/figs', exist_ok=True)
fig_convergence('video/figs/convergence_big.pdf', figsize=(11.07, 6.32), big=True)
from pathfaith.figures import fig_dose_response
fig_dose_response('report/figs/dose_response.pdf')
fig_dose_response('video/figs/dose_response_big.pdf', figsize=(11.07, 6.0), big=True)
from pathfaith.figures_extra import make_extra
made += make_extra()
print('figures:', ', '.join(os.path.basename(m) for m in made), '+ convergence_big')
print('graphs with data:', ', '.join(k for k, v in have.items() if os.path.exists(v)))
PY
python3 -m pathfaith.case_study --out report/case.tex

echo
echo "== 3. report =="
( cd report
  pdflatex -interaction=nonstopmode main.tex > /dev/null || true
  bibtex main > /dev/null 2>&1 || true
  pdflatex -interaction=nonstopmode main.tex > /dev/null || true
  pdflatex -interaction=nonstopmode main.tex > /dev/null || true )
echo "   report/main.pdf: $(pdfinfo report/main.pdf | awk '/Pages/{print $2}') pages"

echo
echo "== 4. presentation deck and narration =="
( cd presentation
  pdflatex -interaction=nonstopmode talk.tex > /dev/null || true
  pdflatex -interaction=nonstopmode talk.tex > /dev/null || true )
echo "   presentation/talk.pdf: $(pdfinfo presentation/talk.pdf | awk '/Pages/{print $2}') slides"
python3 -m pathfaith.make_narration || echo "   narration not written yet (numbers pending)"

echo
echo "== 5. demo recording and video =="
if [ "${1:-}" = "--record" ]; then
  python3 video/record_demo.py --out video/demo_run.mp4 --log video/demo_run.log \
    --run "--n 5 --seed 0" --run "--n 3 --seed 7 --no-benchmark"
fi
( cd video && ./build_video.sh ) | sed 's/^/   /'

echo
echo "== 6. audit =="
python3 -m pathfaith.check_report

echo
echo "== 7. package =="
./package.sh
