"""Shanghai workflow entry: preprocess, generate load, build PV and validate."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def run(module,*args):
    subprocess.run([sys.executable,'-m',module,*args],cwd=ROOT,check=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--validate-only',action='store_true')
    mode.add_argument('--pv-only',action='store_true')
    mode.add_argument('--load-only',action='store_true')
    p.add_argument('--rebuild-load',action='store_true',help='Reconstruct load and weather from data_raw first')
    args=p.parse_args()
    if args.rebuild_load and (args.validate_only or args.pv_only): p.error('--rebuild-load conflicts with selected mode')
    if args.rebuild_load: run('model.preprocess.process')
    if not args.validate_only and not args.pv_only: run('model.generate_load.sample_load')
    if not args.pv_only: run('test.load.validate_fragments')
    if not args.load_only:
        if not args.validate_only: run('model.pv.run')
        run('test.pv.validate_pv')
    run('test.weather.validate_weather')
    run('test.integration.validate_layout')

if __name__=='__main__': main()
