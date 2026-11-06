#!/usr/bin/env bash
# Build the submission video: the talk deck, with the recording of a real demo
# run spliced in right after the "Live demo" slide.
#
# Slide timings come from the "(NNs)" in each heading of narration_template.md,
# so the script and the film cannot drift apart. The demo segment lasts as long
# as the real recording. There is no text to speech here, so the film is silent:
# record your voice over it, or pass a wav and it is muxed in.
#
#   ./build_video.sh                 -> talk_silent.mp4
#   ./build_video.sh narration.wav   -> talk_narrated.mp4
set -euo pipefail
cd "$(dirname "$0")"

DECK=../presentation/talk.pdf
DEMO=${DEMO:-demo_run.mp4}
DEMO_AFTER=6
FPS=30

[ -f "$DECK" ] || { echo "build the deck first: $DECK" >&2; exit 1; }
mapfile -t DURATIONS < <(grep -E '^## Slide [0-9]+,' narration_template.md \
                         | sed -E 's/.*\(([0-9]+)s\)\s*$/\1/')
# QUICK=1 holds every slide for one second: a cheap check that the pipeline works
[ "${QUICK:-0}" = 1 ] && for k in "${!DURATIONS[@]}"; do DURATIONS[$k]=1; done
NSLIDES=$(pdfinfo "$DECK" | awk '/Pages/{print $2}')
[ "${#DURATIONS[@]}" -eq "$NSLIDES" ] || {
  echo "narration has ${#DURATIONS[@]} slide timings, deck has $NSLIDES slides" >&2; exit 1; }

rm -rf frames && mkdir -p frames/seg
pdftoppm -r 150 -png -scale-to-x 1920 -scale-to-y 1080 "$DECK" frames/slide

# every segment gets identical stream settings so they can be joined without
# re-encoding
enc() {  # output, then input arguments
  local out=$1; shift
  ffmpeg -y -loglevel error "$@" -vf "scale=1920:1080,fps=$FPS,format=yuv420p" \
    -c:v libx264 -preset veryfast -crf 20 -an "$out"
}

LIST=frames/seg/list.txt; : > "$LIST"
TIMELINE=timeline.txt; : > "$TIMELINE"
t=0
stamp() { printf '%d:%02d' $(( $1 / 60 )) $(( $1 % 60 )); }
i=0
for f in frames/slide-*.png; do
  d=${DURATIONS[$i]}
  n=$((i + 1))
  enc "frames/seg/s$(printf %02d $n).mp4" -loop 1 -t "$d" -i "$f"
  echo "file 's$(printf %02d $n).mp4'" >> "$LIST"
  echo "slide $n   $(stamp $t) to $(stamp $((t + d)))" >> "$TIMELINE"
  t=$((t + d))
  if [ "$n" -eq "$DEMO_AFTER" ]; then
    if [ -f "$DEMO" ]; then
      enc frames/seg/demo.mp4 -i "$DEMO"
      dd=$(ffprobe -v error -show_entries format=duration -of csv=p=0 frames/seg/demo.mp4 | cut -d. -f1)
      echo "file 'demo.mp4'" >> "$LIST"
      echo "demo      $(stamp $t) to $(stamp $((t + dd)))   (recorded run, $DEMO)" >> "$TIMELINE"
      t=$((t + dd))
    else
      echo "warning: $DEMO not found, building without the demo segment" >&2
      echo "demo      MISSING" >> "$TIMELINE"
    fi
  fi
  i=$((i + 1))
done

ffmpeg -y -loglevel error -f concat -safe 0 -i "$LIST" -c copy talk_silent.mp4
echo "total     $(stamp $t)" >> "$TIMELINE"
cat "$TIMELINE"

if [ $# -ge 1 ] && [ -f "$1" ]; then
  ffmpeg -y -loglevel error -i talk_silent.mp4 -i "$1" \
    -c:v copy -c:a aac -b:a 192k -shortest talk_narrated.mp4
  echo "wrote talk_narrated.mp4"
else
  echo "wrote talk_silent.mp4. Record over it with narration.md, or rerun with a .wav."
fi
