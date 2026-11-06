#!/usr/bin/env python3
r"""Audit the built report before it is submitted.

Checks, in the order they would embarrass me:
  1. no dash characters the author does not want
  2. every macro the manuscript uses is backed by real data, so no ?? survives
  3. every \cite key resolves to an entry in refs.bib, and every entry is cited
  4. every \includegraphics file exists
  5. the page count is within the eight-page limit
  6. the query counts quoted match the results files, on all four graphs
  7. the deck, the narration script, the demo recording and the video exist,
     carry no placeholder or em dash, and the recorded demo ran every query
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT = os.path.join(ROOT, 'report')
PDF = os.path.join(REPORT, 'main.pdf')

FAIL = []


def check(ok, msg):
    print(('  ok   ' if ok else '  FAIL ') + msg)
    if not ok:
        FAIL.append(msg)


def main() -> int:
    tex = open(os.path.join(REPORT, 'main.tex')).read()
    nums = open(os.path.join(REPORT, 'numbers.tex')).read()
    text = subprocess.run(['pdftotext', PDF, '-'], capture_output=True,
                          text=True).stdout

    print('=== 1. dash scan ===')
    # An em dash is banned outright. A minus sign set by math mode and an en
    # dash in a bibliography page range are correct typography, not the thing
    # being ruled out, so prose is checked at the source instead of the PDF.
    check(text.count('\u2014') == 0,
          f"no em dash in the rendered report ({text.count('\u2014')})")
    prose = re.sub(r'\\begin\{thebibliography\}.*', '', tex, flags=re.S)
    bad = re.findall(r'[^-]---?[^->]', prose)
    check(not bad, f'no literal -- or --- in the manuscript source ({len(bad)})')
    check('\u2014' not in tex, 'no em dash character in the manuscript source')

    print('\n=== 2. every quoted number comes from the data ===')
    defined = set(re.findall(r'\\newcommand\{\\([A-Za-z]+)\}', nums))
    stubbed = set(re.findall(r'\\providecommand\{\\([A-Za-z]+)\}', nums))
    used = set(re.findall(r'\\((?:wn|fb|cx|nl|all)[A-Z][A-Za-z]*)', tex))
    check(not (used & stubbed), 'no macro falls back to a placeholder'
          + (f'; stubbed: {sorted(used & stubbed)}' if used & stubbed else ''))
    check(not (used - defined - stubbed), 'every macro used is defined'
          + (f'; undefined: {sorted(used - defined - stubbed)}' if used - defined - stubbed else ''))
    check('??' not in text, 'no ?? placeholder survives into the PDF')

    print('\n=== 3. citations ===')
    bib = open(os.path.join(REPORT, 'refs.bib')).read()
    keys = set(re.findall(r'@\w+\{([^,]+),', bib))
    cited = set()
    for grp in re.findall(r'\\cite\{([^}]+)\}', tex):
        cited |= {k.strip() for k in grp.split(',')}
    check(not (cited - keys), 'every cite key is in refs.bib'
          + (f'; missing: {sorted(cited - keys)}' if cited - keys else ''))
    check(not (keys - cited), 'every bib entry is actually cited'
          + (f'; uncited: {sorted(keys - cited)}' if keys - cited else ''))

    print('\n=== 4. figures ===')
    missing = [f for f in re.findall(r'\\includegraphics\[[^\]]*\]\{([^}]+)\}', tex)
               if not os.path.exists(os.path.join(REPORT, f))
               and not os.path.exists(os.path.join(REPORT, f + '.pdf'))]
    check(not missing, 'every included figure file exists'
          + (f'; missing: {missing}' if missing else ''))

    print('\n=== 5. length ===')
    # CVPR's limit: eight pages of body, with references allowed after them
    info = subprocess.run(['pdfinfo', PDF], capture_output=True, text=True).stdout
    pages = int(re.search(r'Pages:\s+(\d+)', info).group(1))
    body_end = pages
    for pg in range(1, pages + 1):
        t = subprocess.run(['pdftotext', '-f', str(pg), '-l', str(pg), PDF, '-'],
                           capture_output=True, text=True).stdout
        m = re.search(r'^References\s*$', t, flags=re.M)
        if m:
            body_end = pg if len(t[:m.start()].strip()) > 5 else pg - 1
            break
    check(body_end <= 8, f'body is {body_end} pages, within the 8 page limit '
          f'({pages} with references)')

    print('\n=== 6. abstract figures match the results files ===')
    bad = []
    for ds, pre in [('wn18rr', 'wn'), ('fb15k237', 'fb'), ('codexsmall', 'cx'), ('nell23k', 'nl')]:
        p = os.path.join(ROOT, 'results', f'paths_{ds}_seed0.jsonl')
        if not os.path.exists(p):
            bad.append(f'{ds}: no results file')
            continue
        rows = []
        for l in open(p):
            try:
                d = json.loads(l)
            except json.JSONDecodeError:
                continue
            if d.get('kind') == 'query':
                rows.append(d)
        m = re.search(r'\\newcommand\{\\' + pre + r'Queries\}\{(\d+)\}', nums)
        if not m or int(m.group(1)) != len(rows):
            bad.append(f'{ds}: query count macro disagrees with the file')
    check(not bad, 'result files and macros agree' + ('; ' + '; '.join(bad) if bad else ''))

    print('\n=== 7. presentation, narration, demo, video ===')
    deck_tex = open(os.path.join(ROOT, 'presentation', 'talk.tex')).read()
    deck_pdf = os.path.join(ROOT, 'presentation', 'talk.pdf')
    deck = subprocess.run(['pdftotext', deck_pdf, '-'], capture_output=True, text=True).stdout \
        if os.path.exists(deck_pdf) else ''
    check(bool(deck), 'presentation/talk.pdf is built')
    check('??' not in deck, 'no ?? placeholder in the deck')
    check('\u2014' not in deck and '\u2014' not in deck_tex, 'no em dash in the deck')
    used = set(re.findall(r'\\((?:wn|fb|cx|nl|all)[A-Z][A-Za-z]*)', deck_tex))
    check(not (used & stubbed), 'every number on the slides comes from the data'
          + (f'; stubbed: {sorted(used & stubbed)}' if used & stubbed else ''))
    nar = os.path.join(ROOT, 'video', 'narration.md')
    ntext = open(nar).read() if os.path.exists(nar) else ''
    check(bool(ntext), 'video/narration.md is generated')
    check('\u2014' not in ntext and '??' not in ntext, 'no em dash or placeholder in the narration')
    log = os.path.join(ROOT, 'video', 'demo_run.log')
    ltext = open(log).read() if os.path.exists(log) else ''
    runs = re.findall(r'(\d+) quer(?:y|ies) attempted, (\d+) completed every intervention, (\d+) failed', ltext)
    done = sum(int(b) for a, b, c in runs)
    tried = sum(int(a) for a, b, c in runs)
    check(len(runs) >= 2 and all(a == b and c == '0' for a, b, c in runs),
          f'the recorded demo has {len(runs)} separate runs, {done} of {tried} queries '
          'completed every intervention')
    tl = os.path.join(ROOT, 'video', 'timeline.txt')
    ttext = open(tl).read() if os.path.exists(tl) else ''
    check(os.path.exists(os.path.join(ROOT, 'video', 'talk_silent.mp4')) and 'demo_run.mp4' in ttext,
          'video/talk_silent.mp4 is built with the recorded demo spliced in')

    print('\n' + ('ALL CHECKS PASS' if not FAIL else f'{len(FAIL)} CHECK(S) FAILED'))
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
