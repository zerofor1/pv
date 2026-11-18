"""Download independent years concurrently, retaining verified point caches."""
from pathlib import Path
from datetime import date,timedelta
from concurrent.futures import ThreadPoolExecutor
import csv
from model.preprocess import fetch_gfs as gfs
import fetch_gfs_mirror as mirror
ROOT=Path(__file__).resolve().parents[2]
def year(y):
    target=ROOT/'data/weather/gfs'/f'gfs_{y}.csv'
    if target.exists():
        with target.open(encoding='utf-8-sig') as f: return list(csv.DictReader(f))
    start,end=date(y,1,1),date(y+1,1,1)
    days=[start+timedelta(days=i) for i in range((end-start).days)]
    with ThreadPoolExecutor(max_workers=8) as pool: runs=list(pool.map(gfs.select_run,days))
    rows=mirror.collect(runs,ROOT/'data_raw/weather/gfs/points',workers=64)
    gfs.write_csv(target,rows)
    return rows
if __name__=='__main__':
    existing=ROOT/'data/weather/gfs/hourly.csv'
    if existing.exists():
        with existing.open(encoding='utf-8-sig') as f: saved=list(csv.DictReader(f))
        if len(saved)==35064 and len({r['valid_time_utc'] for r in saved})==35064:
            from model.preprocess.process import weather
            weather()
            print('Reused complete GFS output; alignment refreshed.',flush=True)
            raise SystemExit(0)
    with ThreadPoolExecutor(max_workers=4) as pool: chunks=list(pool.map(year,range(2022,2026)))
    rows=sorted([r for chunk in chunks for r in chunk],key=lambda r:r['valid_time_utc'])
    assert len(rows)==35064 and len({r['valid_time_utc'] for r in rows})==35064
    gfs.write_csv(ROOT/'data/weather/gfs/hourly.csv',rows)
    from model.preprocess.process import weather
    weather()
    print('DONE: 35064 aligned hours',flush=True)
