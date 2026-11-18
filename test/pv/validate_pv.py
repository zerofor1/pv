"""Check PV data alignment, source identity, power bounds and loss accounting."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from model.configuration import read_config,configuration_id

ROOT=Path(__file__).resolve().parents[2]
def main():
    cfg=read_config(); folder=ROOT/'test/reports/pv'
    manifest=json.loads((folder/'input_manifest.json').read_text(encoding='utf-8'))
    for key,name in manifest['input_paths'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==manifest['input_sha256'][key],f'Stale PV source: {key}'
    for name,meta in manifest['outputs'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==meta['sha256'],f'Changed PV result: {name}'
    expected=pd.date_range(cfg['simulation']['start'],cfg['simulation']['end'],inclusive='left',freq='h',tz='Asia/Shanghai').tz_convert('UTC')
    losses=['pv_temperature_reduction_kw','pv_iam_loss_kw','pv_soiling_loss_kw','pv_mismatch_loss_kw',
        'pv_dc_wiring_loss_kw','pv_legacy_dc_loss_kw','pv_inverter_conversion_loss_kw','pv_inverter_clipping_kw','pv_ac_wiring_loss_kw']
    summary={}
    for name in ['era5','gfs']:
        f=pd.read_csv(ROOT/'data/weather'/name/'pv_hourly.csv')
        times=pd.DatetimeIndex(pd.to_datetime(f.valid_time_utc,utc=True))
        assert times.equals(expected) and set(f.site_id)=={cfg['site_id']}
        assert (pd.to_datetime(f.interval_end_utc,utc=True)-times==pd.Timedelta(hours=1)).all()
        assert set(f.pv_config_id)=={configuration_id(cfg['pv'])}
        assert np.allclose(f.site_latitude,cfg['pv']['latitude']) and np.allclose(f.site_longitude,cfg['pv']['longitude'])
        assert np.isfinite(f[losses+['pv_reference_dc_kw','pv_ac_kw']].to_numpy()).all()
        residual=f.pv_reference_dc_kw-f.pv_ac_kw-f[losses].sum(axis=1)
        assert np.abs(residual).max()<1e-4
        assert (f.pv_ac_kw>=0).all() and (f.pv_inverter_ac_kw<=cfg['pv']['ac_capacity_kw']+1e-6).all()
        summary[name]=dict(hours=len(f),energy_kwh=float(f.pv_energy_kwh.sum()),max_loss_residual_kw=float(np.abs(residual).max()))
    report=dict(status='passed',datasets=summary)
    (folder/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('PV validation passed: both four-year series and physical loss ledgers.')

if __name__=='__main__': main()
