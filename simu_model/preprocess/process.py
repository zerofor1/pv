"""Extract all MP-EVData stations and conservatively align hourly load/weather."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import openpyxl
from datetime import datetime, timedelta, timezone

ROOT = Path(__file__).resolve().parents[2]
INDEX = pd.date_range('2022-01-01', '2026-01-01', freq='h', inclusive='left', tz='Asia/Shanghai')

def loads():
    (ROOT/'data/load/cleaned').mkdir(parents=True,exist_ok=True)
    (ROOT/'test/reports/load').mkdir(parents=True,exist_ok=True)
    source = ROOT / 'data_raw/load/MP_EVData/station_hourly.xlsx'
    book = openpyxl.load_workbook(source, read_only=True, data_only=True)
    sessions = openpyxl.load_workbook(source.parent/'charging_session.xlsx', read_only=True, data_only=True)
    report = {}
    for sheet in book:
        rows = list(sheet.values)
        print(sheet.title, len(rows), str(rows[:2])[:250], flush=True)
        pairs = [(r[0], r[1]) for r in rows if len(r)>1 and isinstance(r[0], __import__('datetime').datetime)]
        if not pairs:
            continue
        # A7/A8 published field is session_count, not kW. Reconstruct every station
        # consistently from transaction energy using exact hourly overlap.
        from datetime import datetime, timedelta
        hourly = {t:0. for t,_ in pairs}
        invalid_sessions = 0
        expected_energy = 0.
        coverage_start, coverage_end = min(hourly), max(hourly)+timedelta(hours=1)
        for r in sessions[sheet.title].iter_rows(min_row=2, max_col=3, values_only=True):
            start,end,energy = r
            if not isinstance(start,datetime) or not isinstance(end,datetime) or end<=start or not isinstance(energy,(int,float)) or not np.isfinite(energy) or energy<0:
                invalid_sessions += 1
                continue
            power = energy / ((end-start).total_seconds()/3600)
            expected_energy += power*max(0.,(min(end,coverage_end)-max(start,coverage_start)).total_seconds())/3600
            t = start
            while t<end:
                hour=t.replace(minute=0,second=0,microsecond=0)
                stop=min(hour+timedelta(hours=1),end)
                if hour in hourly: hourly[hour] += power*(stop-t).total_seconds()/3600
                t=stop
        pairs=list(hourly.items())
        energy_error = sum(hourly.values())-expected_energy
        if abs(energy_error)>max(1e-6,expected_energy*1e-10):
            raise ValueError('Hourly reconstruction violates energy conservation')
        raw = pd.DataFrame(pairs, columns=['time', 'load_kw'])
        raw['time'] = pd.to_datetime(raw.time).dt.tz_localize('Asia/Shanghai')
        raw['load_kw'] = pd.to_numeric(raw.load_kw, errors='coerce')
        raw.loc[(raw.load_kw < 0) | ~np.isfinite(raw.load_kw), 'load_kw'] = np.nan
        if (raw.time.dt.minute.ne(0) | raw.time.dt.second.ne(0)).any():
            raise ValueError('Non-hourly source timestamps; inspect before resampling')
        duplicates = int(raw.time.duplicated().sum())
        # Average duplicate published hourly measurements; never sum power readings.
        series = raw.groupby('time').load_kw.mean().reindex(INDEX)
        frame = pd.DataFrame({'load_kw':series, 'month':INDEX.month, 'weekday':INDEX.dayofweek, 'hour':INDEX.hour})
        observed = frame.load_kw.notna()
        frame['fill_method'] = np.where(observed, 'observed', '')
        for keys in [['month','weekday','hour'], ['month','hour'], ['weekday','hour'], ['hour']]:
            means = frame.loc[observed].groupby(keys).load_kw.mean()
            lookup = pd.MultiIndex.from_frame(frame[keys]) if len(keys)>1 else frame[keys[0]]
            values = pd.Series(means.reindex(lookup).to_numpy(), index=INDEX)
            mask = frame.load_kw.isna() & values.notna()
            frame.loc[mask, 'load_kw'] = values[mask]
            frame.loc[mask, 'fill_method'] = 'mean_'+'_'.join(keys)
        if frame.load_kw.isna().any():
            raise ValueError(f'{sheet.title}: no same-hour evidence for filling')
        mu, sigma = float(frame.load_kw.mean()), float(frame.load_kw.std(ddof=0))
        scale = sigma if sigma else 1.
        frame['load_z'] = (frame.load_kw-mu)/scale
        frame.index.name = 'interval_start_bjt'
        frame[['load_kw','load_z','fill_method']].to_csv(ROOT/'data/load/cleaned'/f'{sheet.title}_hourly.csv', encoding='utf-8-sig')
        result = dict(station=sheet.title,standardization=dict(mean_kw=mu,std_kw=sigma,ddof=0,scope='all aligned hours of this station'),n_observed=int(observed.sum()),n_filled=int((~observed).sum()),duplicates=duplicates,source_start=str(raw.time.min()),source_end=str(raw.time.max()),peak_kw=float(frame.load_kw.max()))
        result.update(load_source='session energy allocated by exact hourly overlap; constant average power during connection',invalid_sessions=invalid_sessions,energy_conservation_error_kwh=energy_error,zero_assumption='No transaction in published coverage interpreted as zero; completeness of collection unknown')
        report[sheet.title]=result
        print('EXTRACTED',sheet.title,result['n_observed'],result['n_filled'],flush=True)
    book.close()
    sessions.close()
    (ROOT/'test/reports/load/summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')

def native_weather():
    """Rebuild hourly weather offline from unchanged downloaded native points."""
    from model.preprocess import fetch_gfs as gfs, fetch_era5_land as era
    raw=ROOT/'data_raw/weather'
    run_index=pd.read_csv(raw/'gfs/run_manifest.csv')
    rows=[]
    for row in run_index.to_dict('records'):
        def stamp(name): return datetime.fromisoformat(row[name].replace('Z','+00:00'))
        point=(raw/'gfs'/row['point_file']).resolve()
        if not point.is_relative_to((raw/'gfs').resolve()): raise ValueError('Raw point outside GFS directory')
        data=json.loads(point.read_text(encoding='utf-8'))
        run=(stamp('forecast_init_utc'),stamp('decision_cutoff_utc'),int(row['lead_hours']),stamp('forecast_available_by_utc'))
        rows.extend(gfs.standardize(run,data,float(row['grid_latitude']),float(row['grid_longitude'])))
    forecast=pd.DataFrame(rows)
    values={}; metadata=[]
    files=sorted((raw/'era5/points').glob('*.npz'))
    if len(files)!=4: raise ValueError('Expected four ERA5 native groups')
    for path in files:
        with np.load(path,allow_pickle=False) as saved:
            metadata.append(json.loads(str(saved['metadata'])))
            values.update({k:saved[k] for k in saved.files if k!='metadata'})
    start=datetime(2022,1,1,tzinfo=era.BJT).astimezone(timezone.utc)
    if len({(m['grid_latitude'],m['grid_longitude']) for m in metadata})!=1:
        raise ValueError('ERA5 variable grids differ')
    if any(m['start_utc']!=era.timestamp(start) for m in metadata): raise ValueError('Wrong native ERA5 start')
    reference=pd.DataFrame(era.build_rows(values,metadata[0],start,len(INDEX)))
    return [forecast,reference]

def weather():
    data=[]
    for f in native_weather():
        f=f.drop(columns=['split'],errors='ignore')
        f.index=pd.to_datetime(f.valid_time_utc,utc=True).dt.tz_convert('Asia/Shanghai')
        if f.index.duplicated().any(): raise ValueError('Duplicate weather times')
        data.append(f.reindex(INDEX))
    common=['ghi_w_m2','air_temp_c','relative_humidity_pct','wind_u_10m_m_s','wind_v_10m_m_s','wind_speed_m_s','precipitation_mm','surface_pressure_pa']
    original=[f[common].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan) for f in data]
    report={}
    for i,name in enumerate(['gfs','era5']):
        f=data[i].copy()
        for col in common:
            missing=original[i][col].isna()
            f[col]=original[i][col].fillna(original[1-i][col])
            f[col+'_source']=np.where(missing & f[col].notna(),'era5_actual' if i==0 else 'gfs',np.where(f[col].isna(),'unresolved',name))
        target=ROOT/'data/weather'/name/'hourly.csv'; target.parent.mkdir(parents=True,exist_ok=True)
        f.to_csv(target,index=False,encoding='utf-8-sig')
        report[name]=dict(hours=len(f),filled_cells=int((original[i].isna() & f[common].notna()).sum().sum()),unresolved_cells=int(f[common].isna().sum().sum()))
    target=ROOT/'test/reports/weather/alignment_report.json'; target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Prepared 35064 hours per weather source from native data_raw inputs.',flush=True)

if __name__=='__main__':
    import sys
    if '--weather-only' not in sys.argv: loads()
    weather()
