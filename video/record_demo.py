"""Record a real run of the demo and render it as a terminal video.

There is no screen recorder in this environment, so this does the honest
equivalent: run `python -m pathfaith.demo` under a pseudo-terminal, timestamp
every line as it is printed, and draw the session frame by frame. Long waits
are shortened so the video moves, and the true elapsed time of the run is
printed in the corner of every frame, so the recording cannot pass off a
compressed run as a fast one.

Several runs can be recorded back to back in one terminal session, each with
its own arguments, to show that the demo works on more than one draw of
queries and is not a single lucky run.

Usage:
    python video/record_demo.py --run "--n 5 --seed 0" --run "--n 3 --seed 7 --no-benchmark"
"""
from __future__ import annotations

import argparse
import os
import pty
import re
import select
import shutil
import subprocess
import sys
import tempfile
import time

from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'
FONT_BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf'
SIZE = 21
BG = (24, 26, 33)
FG = (220, 223, 228)
DIM = (120, 126, 138)
COLORS = {'31': (235, 105, 100), '32': (120, 205, 130), '33': (230, 190, 100),
          '34': (110, 165, 240)}
PROMPT = '$ '
FPS = 12
MAX_GAP = 1.2          # longest pause kept, in seconds, between two lines
TYPE_CPS = 28          # typing speed for the command line
HOLD = 2.5             # seconds to hold each finished query on screen

ANSI = re.compile(r'\x1b\[([0-9;]*)m')


def capture(cmd, cwd):
    """Run cmd under a pty and return [(seconds_since_start, line)]."""
    master, slave = pty.openpty()
    env = dict(os.environ, COLUMNS='110', LINES='60', PYTHONUNBUFFERED='1')
    proc = subprocess.Popen(cmd, stdout=slave, stderr=subprocess.DEVNULL,
                            stdin=subprocess.DEVNULL, cwd=cwd, env=env, close_fds=True)
    os.close(slave)
    t0, buf, lines = time.time(), b'', []
    while True:
        r, _, _ = select.select([master], [], [], 0.2)
        if r:
            try:
                chunk = os.read(master, 4096)
            except OSError:
                chunk = b''
            if not chunk:
                break
            buf += chunk
            while b'\n' in buf:
                raw, buf = buf.split(b'\n', 1)
                lines.append((time.time() - t0, raw.decode('utf-8', 'replace').rstrip('\r')))
        elif proc.poll() is not None:
            break
    if buf:
        lines.append((time.time() - t0, buf.decode('utf-8', 'replace')))
    proc.wait()
    return lines, proc.returncode


def spans(line):
    """Split a line with ANSI codes into (text, colour, bold) runs."""
    out, color, bold, pos = [], FG, False, 0
    for m in ANSI.finditer(line):
        if m.start() > pos:
            out.append((line[pos:m.start()], color, bold))
        for code in (m.group(1) or '0').split(';'):
            if code in ('0', ''):
                color, bold = FG, False
            elif code == '1':
                bold = True
            elif code == '2':
                color = DIM
            elif code in COLORS:
                color = COLORS[code]
        pos = m.end()
    if pos < len(line):
        out.append((line[pos:], color, bold))
    return out


class Screen:
    def __init__(self):
        self.font = ImageFont.truetype(FONT, SIZE)
        self.bold = ImageFont.truetype(FONT_BOLD, SIZE)
        self.small = ImageFont.truetype(FONT, 17)
        self.cw = self.font.getbbox('M')[2]
        self.lh = int(SIZE * 1.32)
        self.top, self.left = 64, 44
        self.rows = (H - self.top - 40) // self.lh

    def frame(self, lines, clock, note):
        img = Image.new('RGB', (W, H), BG)
        d = ImageDraw.Draw(img)
        # window chrome
        d.rectangle([0, 0, W, 40], fill=(36, 39, 48))
        for i, col in enumerate([(237, 106, 94), (245, 191, 79), (98, 197, 84)]):
            d.ellipse([18 + i * 26, 13, 32 + i * 26, 27], fill=col)
        d.text((W // 2 - 190, 10), 'pathfaith demo  ·  frozen ULTRA/NBFNet on CPU',
               font=self.small, fill=DIM)
        d.text((W - 330, 10), f'real elapsed {clock:6.1f}s{note}', font=self.small, fill=DIM)
        y = self.top
        for line in lines[-self.rows:]:
            x = self.left
            for text, color, bold in spans(line):
                d.text((x, y), text, font=self.bold if bold else self.font, fill=color)
                x += self.cw * len(text)
            y += self.lh
        return img


def render(runs, out):
    """runs: [(command shown, [(t, line)], real seconds before this run)]"""
    screen = Screen()
    tmp = tempfile.mkdtemp()
    frames, shown = 0, []

    def emit(n, clock, note=''):
        nonlocal frames
        img = screen.frame(shown, clock, note)
        for _ in range(max(1, n)):
            img.save(os.path.join(tmp, f'{frames:06d}.png'))
            frames += 1

    for command, events, offset in runs:
        typed = PROMPT
        shown.append(typed)
        emit(FPS // 2, offset)
        for ch in command:
            typed += ch
            shown[-1] = typed
            emit(max(1, FPS // TYPE_CPS), offset)
        emit(FPS // 2, offset)
        prev = 0.0
        for t, line in events:
            gap = t - prev
            kept = min(gap, MAX_GAP)
            note = '' if gap <= MAX_GAP + 1e-6 else '  (wait shortened)'
            emit(int(kept * FPS), offset + t, note)
            shown.append(line)
            prev = t
            # hold each finished query on screen long enough to read
            if '(' in line and 's for this query)' in line:
                emit(int(HOLD * FPS), offset + t)
        emit(FPS * 2, offset + prev)
    shown.append(PROMPT)
    emit(FPS * 3, runs[-1][2] + (runs[-1][1][-1][0] if runs[-1][1] else 0))

    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(FPS),
                    '-i', os.path.join(tmp, '%06d.png'), '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p', '-crf', '22', '-preset', 'veryfast', out],
                   check=True)
    shutil.rmtree(tmp)
    return frames / FPS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='video/demo_run.mp4')
    ap.add_argument('--log', default='video/demo_run.log')
    ap.add_argument('--run', action='append', default=None,
                    help='arguments for one demo run; repeat for several runs')
    args = ap.parse_args()
    run_args = args.run or ['--n 5 --seed 0']

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    runs, offset, log = [], 0.0, []
    for a in run_args:
        shown_cmd = ('python -m pathfaith.demo ' + a).strip()
        cmd = [sys.executable, '-m', 'pathfaith.demo', '--color', 'always'] + a.split()
        print('recording:', shown_cmd)
        events, rc = capture(cmd, root)
        real = events[-1][0] if events else 0.0
        print(f'  captured {len(events)} lines, exit code {rc}, real time {real:.1f}s')
        log.append(f'$ {shown_cmd}')
        log += [f'{offset + t:8.2f}  {ANSI.sub("", line)}' for t, line in events]
        if rc != 0:
            open(os.path.join(root, args.log), 'w').write('\n'.join(log) + '\n')
            raise SystemExit('demo failed; not rendering a recording of a failed run')
        runs.append((shown_cmd, events, offset))
        offset += real
    with open(os.path.join(root, args.log), 'w') as f:
        f.write('\n'.join(log) + '\n')
    dur = render(runs, os.path.join(root, args.out))
    print(f'wrote {args.out}: {dur:.0f}s of video for {offset:.0f}s of real runs')


if __name__ == '__main__':
    main()
