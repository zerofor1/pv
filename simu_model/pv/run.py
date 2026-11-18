"""Prepare Shanghai reference/forecast PV only; no archived purchase dependency."""
import csv
import hashlib
import json
from pathlib import Path
from model.configuration import read_config,configuration_id
from model.pv.simulate import Site,simulate_rows,write_results

ROOT=Path(__file__).resolve().parents[2]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    cfg=read_config(); site=Site(**cfg['pv'])
    sources={
        'era5':ROOT/'data/weather/era5/hourly.csv',
        'gfs':ROOT/'data/weather/gfs/hourly.csv',
        'config':ROOT/'model/pv/config.json',
        'model':ROOT/'model/pv/simulate.py',
        'runner':ROOT/'model/pv/run.py'}
    target=ROOT/'test/reports/pv/input_manifest.json'; target.parent.mkdir(parents=True,exist_ok=True)
    hashes={k:sha(v) for k,v in sources.items()}
    if target.exists():
        old=json.loads(target.read_text(encoding='utf-8'))
        if old['input_sha256']==hashes and all((ROOT/name).exists() and sha(ROOT/name)==v['sha256'] for name,v in old['outputs'].items()):
            print('Reused verified PV outputs.'); return
    outputs={}
    for name in ['era5','gfs']:
        destination=ROOT/'data/weather'/name/'pv_hourly.csv'
        with sources[name].open(encoding='utf-8-sig',newline='') as f:
            write_results(destination,simulate_rows(csv.DictReader(f),site))
        outputs[destination.relative_to(ROOT).as_posix()]={'sha256':sha(destination),'role':'reference PV' if name=='era5' else 'forecast PV'}
    report=dict(input_paths={k:v.relative_to(ROOT).as_posix() for k,v in sources.items()},input_sha256=hashes,
        outputs=outputs,site_id=cfg['site_id'],pv_config_id=configuration_id(cfg['pv']),
        note='Reference engineering array, not measured station equipment. ERA5 actuals drive reference PV.')
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Prepared ERA5/GFS PV: 35064 hours each.')

if __name__=='__main__': main()
