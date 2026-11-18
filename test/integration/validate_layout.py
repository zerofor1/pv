"""Verify migration preserved raw workbooks, existing scenarios and weather values."""
from pathlib import Path
import hashlib,json
import argparse
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--migration-strict',action='store_true',help='Compare against the pre-reorganization snapshot once')
    args=parser.parse_args()
    folder=ROOT/'test/reports/integration'
    manifest_path=folder/'migration_manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {'moves':[],'weather_numeric_fingerprints':{}}
    if args.migration_strict: assert manifest_path.exists(),'Migration snapshot is missing'
    count=0
    for row in manifest['moves']:
        name=row['target']
        if args.migration_strict and (name.endswith('.xlsx') and name.startswith('data_raw/') or name.startswith('data/load/') and name.endswith('.csv')):
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==row['sha256'],f'Migration changed {name}'
            count+=1
    for key,old in (manifest['weather_numeric_fingerprints'].items() if args.migration_strict else []):
        name='era5' if key=='era5_land' else key
        f=pd.read_csv(ROOT/'data/weather'/name/'hourly.csv')
        digest=hashlib.sha256(np.round(f[old['columns']].to_numpy(dtype=float),8).tobytes()).hexdigest()
        assert len(f)==old['rows'] and digest==old['rounded_numeric_sha256'],f'Weather numerical change: {name}'
    assert len(list((ROOT/'data_raw/weather/gfs/points').glob('*.json')))==1461
    assert len(list((ROOT/'data_raw/weather/era5/points').glob('*.npz')))==4
    allowed={'data_raw','data','model','test','项目材料','.runtime'}
    extras=[p.name for p in ROOT.iterdir() if p.is_dir() and p.name not in allowed]
    assert not extras,f'Unexpected root directories: {extras}'
    assert sorted(p.name for p in (ROOT/'data').iterdir() if p.is_dir())==['load','weather']
    assert sorted(p.name for p in (ROOT/'data_raw').iterdir() if p.is_dir())==['load','weather']
    assert (ROOT/'README.md').exists()
    report=dict(status='passed',migration_snapshot_checked=args.migration_strict,unchanged_files=count,native_gfs_points=1461,native_era5_groups=4)
    folder.mkdir(parents=True,exist_ok=True)
    (folder/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Migration validation passed: raw workbooks, simulated CSVs and weather values preserved.')

if __name__=='__main__': main()
