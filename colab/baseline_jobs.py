"""Link-prediction checks for the report, run on Colab while the main machine
runs the explanation experiments. Each result is a small JSON file in
CAP6938 pathfaith/colab_results/, and a job that has finished is skipped.

  CoDEx-S and NELL23k with ultra_3g, the ULTRA paper's own model, to compare
  with the published zero-shot MRR (0.472 and 0.239)
  NELL23k and WN18RR full test sets, and a 2,000-triple sample of FB15k-237,
  with ultra_4g, the checkpoint the benchmark explains

The printed RESULT lines were copied into results/baseline3g_*_full.json,
results/baseline_nell23k_full.json, results/baseline_wn18rr_full.json and
results/baseline_fb15k237_2000.json.
"""
import json
import os
import subprocess
import sys

OUT = '/content/drive/MyDrive/CAP6938 pathfaith/colab_results'
os.makedirs(OUT, exist_ok=True)
JOBS = [('CoDExSmall', 'ultra_3g.pth', 0), ('NELL23k', 'ultra_3g.pth', 0),
        ('NELL23k', 'ultra_4g.pth', 0), ('WN18RR', 'ultra_4g.pth', 0),
        ('FB15k237', 'ultra_4g.pth', 2000)]
KEYS = ('n_triples', 'mrr', 'hits@1', 'hits@3', 'hits@10', 'mrr_tail_only',
        'hits@10_tail_only', 'seconds')

for ds, ck, n in JOBS:
    tag = f"baseline_{ds.lower()}_{ck.split('_')[1].split('.')[0]}_{n if n else 'full'}"
    path = os.path.join(OUT, tag + '.json')
    if not os.path.exists(path):
        print('running', tag, flush=True)
        subprocess.run([sys.executable, '-m', 'pathfaith.run_baseline', '--dataset', ds,
                        '--ckpt', ck, '--n', str(n), '--batch-size', '8', '--out', path],
                       cwd='/content/pathfaith', stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, check=True)
    d = json.load(open(path))
    print('RESULT', tag, json.dumps({k: d.get(k) for k in KEYS}), flush=True)
print('ALL BASELINES DONE', flush=True)
