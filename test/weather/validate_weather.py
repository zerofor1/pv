"""Validate clean weather time keys, coverage, physical fields and provenance."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
def main():
    expected=pd.date_range('2022-01-01','2026-01-01',inclusive='left',freq='h',tz='Asia/Shanghai').tz_convert('UTC')
    fields=['ghi_w_m2','air_temp_c','relative_humidity_pct','wind_speed_m_s','precipitation_mm','surface_pressure_pa']
    for name in ['gfs','era5']:
        f=pd.read_csv(ROOT/'data/weather'/name/'hourly.csv')
        assert pd.DatetimeIndex(pd.to_datetime(f.valid_time_utc,utc=True)).equals(expected)
        assert (pd.to_datetime(f.interval_end_utc,utc=True)-expected==pd.Timedelta(hours=1)).all()
        assert np.isfinite(f[fields].to_numpy()).all()
        assert f.ghi_w_m2.ge(0).all() and f.wind_speed_m_s.ge(0).all() and f.precipitation_mm.ge(0).all()
        assert f.relative_humidity_pct.between(0,100).all()
        assert np.allclose(f.site_latitude,31.2304) and np.allclose(f.site_longitude,121.4737)
        for field in fields: assert f[field+'_source'].notna().all() and not f[field+'_source'].eq('unresolved').any()
        if name=='gfs':
            assert (pd.to_datetime(f.forecast_available_by_utc,utc=True)<pd.to_datetime(f.decision_cutoff_utc,utc=True)).all()
    report=dict(status='passed',hours=35064,sources=['gfs','era5'])
    (ROOT/'test/reports/weather/validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Weather validation passed: continuous Shanghai hours, source flags and GFS availability.')

if __name__=='__main__': main()
