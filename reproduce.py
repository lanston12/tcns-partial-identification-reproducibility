"""One-command frozen reproduction; no downloads or implicit experiment changes."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys
import numpy as np
ROOT=Path(__file__).resolve().parent
def run(*args):
 subprocess.run([sys.executable,*args],cwd=str(ROOT),check=True)
def digest(path):
 if path.suffix=='.json':
  obj=json.loads(path.read_text(encoding='utf-8'))
  if isinstance(obj,dict):obj.pop('runtime_seconds',None)
  return dict(kind='json',sha256=hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest())
 with np.load(path,allow_pickle=False) as z:
  return dict(kind='npz',arrays={k:dict(shape=list(z[k].shape),dtype=str(z[k].dtype),sha256=hashlib.sha256(z[k].tobytes(order='C')).hexdigest()) for k in sorted(z.files)})
def main():
 parser=argparse.ArgumentParser(description=__doc__)
 group=parser.add_mutually_exclusive_group();group.add_argument('--full',action='store_true',help='Regenerate all fixed experiments, saved results, and six figures; verify exact scientific digests')
 group.add_argument('--check',action='store_true',help='Run 21 focused tests using small frozen references (default)')
 args=parser.parse_args()
 if args.full:
  for directory in ['results/raw','results/processed','paper/figures']:(ROOT/directory).mkdir(parents=True,exist_ok=True)
  run('scripts/run_experiments.py');run('scripts/additional_checks.py');run('scripts/make_figures.py')
  expected=json.loads((ROOT/'reference/manifest.json').read_text(encoding='utf-8'))
  rows=[dict(path=rel,identical=digest(ROOT/rel)==value) for rel,value in expected.items()]
  (ROOT/'results/verification.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
  failures=[r['path'] for r in rows if not r['identical']]
  if failures:raise RuntimeError('Scientific digest mismatch; inspect results/verification.json and reference environment: '+', '.join(failures))
  print('PASS: all {} scientific artifacts match frozen reference content.'.format(len(rows)),flush=True)
 run('-m','unittest','discover','-s','tests','-v')
if __name__=='__main__':main()
