#!/usr/bin/env bash
# Assemble the submission files.
#
#   CAP6938_pathfaith_code.zip   code, README, results, report and deck sources
#   report/main.pdf              the report
#   video/talk_silent.mp4        the talk with the recorded demo, to narrate over
#   presentation/talk.pdf        the deck for the live presentation
#
# The vendored ULTRA checkout and the downloaded datasets are left out: the
# README fetches both in two commands, and shipping them would make the zip
# hundreds of megabytes of somebody else's code. The small name files the demo
# needs (CoDEx labels, the WN18RR text release) are included.
set -euo pipefail
cd "$(dirname "$0")"

OUT=CAP6938_pathfaith_code.zip
rm -f "$OUT"

# --no-audit is for building a test zip before the benchmark has finished
if [ "${1:-}" != "--no-audit" ]; then
  python3 -m pathfaith.check_report || {
    echo "audit failed; fix before packaging" >&2; exit 1; }
fi

FILES=()
for f in README.md requirements.txt run_benchmark.sh run_stage.sh finalize.sh package.sh pathfaith/*.py report/main.tex report/refs.bib report/numbers.tex report/case.tex report/figs/*.pdf report/main.pdf presentation/talk.tex presentation/talk.pdf video/narration_template.md video/narration.md video/build_video.sh video/record_demo.py video/figs/*.pdf video/demo_run.log video/timeline.txt kg-labels colab/pathfaith_colab.ipynb colab/torch_scatter_shim/pyproject.toml colab/torch_scatter_shim/torch_scatter/*.py results/*.json results/*.jsonl logs/benchmark.log logs/reliability.txt; do
  if [ -e "$f" ]; then FILES+=("$f")
  else echo "  (not built yet, left out: $f)"; fi
done
zip -q -r "$OUT" "${FILES[@]}" -x '*__pycache__*'

echo
echo "wrote $OUT"
unzip -l "$OUT" | tail -1
echo
echo "submit:"
echo "  $OUT"
echo "  report/main.pdf"
echo "  video/talk_silent.mp4  (narrate it from video/narration.md, or mux a wav with video/build_video.sh)"
echo "present:"
echo "  presentation/talk.pdf, then run: python -m pathfaith.demo --n 5"
