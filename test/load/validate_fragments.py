"""Independent reconstruction checks against original covered hourly curves."""
import json
import numpy as np
import pandas as pd
from model.generate_load.fragment_load import ROOT, SEED, CONFIG, Library, actual_weather, choose_setting, read_library, simulate, labels

def main():
    folder=ROOT/'data/load/simulated'
    reports=ROOT/'test/reports/load'
    wide=pd.read_csv(folder/'all_stations_hourly_2022_2025.csv',index_col=0)
    expected=pd.date_range('2022-01-01','2026-01-01',inclusive='left',freq='h',tz='Asia/Shanghai')
    assert pd.to_datetime(wide.index,utc=True).tz_convert('Asia/Shanghai').equals(expected)
    assert wide.shape==(35064,10) and wide.notna().all().all()
    assert np.isfinite(wide.to_numpy()).all() and (wide.to_numpy()>=0).all()
    provenance=pd.read_csv(folder/'daily_provenance.csv')
    assert len(provenance)==14610
    report=json.loads((reports/'generation_report.json').read_text(encoding='utf-8'))
    coverage=json.loads((ROOT/'test/reports/load/coverage_audit.json').read_text(encoding='utf-8'))
    checks={}
    for station,r in report['stations'].items():
        lib=read_library(ROOT/'data/load/cleaned'/f'{station}_hourly.csv')
        excluded=set(coverage['stations'][station]['excluded_template_dates'])
        assert not {str(d.date()) for d in lib.days}.intersection(excluded)
        ids={str(d.date()):i for i,d in enumerate(lib.days)}
        trace=provenance.loc[provenance.station==station]
        assert len(trace)==1461 and trace.target_date.is_unique
        assert not set(trace.energy_source_date).intersection(excluded)
        assert not set(trace.shape_source_date).intersection(excluded)
        energy_ids=np.array([ids[d] for d in trace.energy_source_date])
        shape_ids=np.array([ids[d] for d in trace.shape_source_date])
        curves=wide[station+'_kw'].to_numpy().reshape(-1,24)
        energy_error=float(np.max(np.abs(curves.sum(axis=1)-lib.energy[energy_ids])))
        shape_error=float(np.max(np.abs(curves-lib.curves[shape_ids]*trace.shape_scale.to_numpy()[:,None])))
        assert energy_error<1e-6 and shape_error<1e-6
        assert np.array_equal(curves==0,lib.curves[shape_ids]==0)
        tt=labels(pd.DatetimeIndex(pd.to_datetime(trace.target_date)))[0]
        assert np.array_equal(tt,lib.types[energy_ids]) and np.array_equal(tt,lib.types[shape_ids])
        assert trace.shape_scale.between(.9-1e-12,1.1+1e-12).all()
        for _,block in trace.groupby('block_id',sort=False):
            assert (pd.to_datetime(block.energy_source_date).diff().dropna()==pd.Timedelta(days=1)).all()
        assert np.array_equal(trace.shape_replaced,energy_ids!=shape_ids)
        expected_weights=[float(lib.boundary_weights(curves[i,0],curves[i-1,-1] if i else None)) for i in range(len(curves))]
        assert np.allclose(expected_weights,trace.final_boundary_weight)
        assert trace.weather_conditioned.eq(r['weather_conditioned']).all()
        evaluation=r['evaluation']
        training_dates=set(evaluation['training_dates'])
        validation_dates=set(evaluation['validation_dates'])
        test_dates=set(evaluation['test_dates'])
        assert not (training_dates|validation_dates|test_dates).intersection(excluded)
        assert not (training_dates & (validation_dates|test_dates)) and not (validation_dates & test_dates)
        selected,gain=choose_setting(evaluation['validation_settings'])
        assert selected['block_days']==r['block_days'] and selected['weather']==r['weather_conditioned']
        assert r['block_days'] in CONFIG['block_candidates']
        single=pd.read_csv(folder/f'{station}_hourly_2022_2025.csv')
        assert np.allclose(single.load_kw,curves.ravel())
        assert np.allclose(single.load_z,(curves.ravel()-lib.mean)/lib.std)
        checks[station]=dict(max_daily_energy_error_kwh=energy_error,max_hourly_shape_error_kw=shape_error,
                            all_source_days_observed=True,calendar_match=True,zero_pattern_preserved=True,
                            final_scaled_boundary_weights_verified=True,independent_test_dates=True,
                            incomplete_source_days_excluded=True)
    test_trace=pd.read_csv(reports/'test_provenance.csv')
    with np.load(reports/'test_ensembles.npz') as ensembles:
        for station,r in report['stations'].items():
            ev=r['evaluation']; lib=read_library(ROOT/'data/load/cleaned'/f'{station}_hourly.csv')
            lookup={str(d.date()):i for i,d in enumerate(lib.days)}
            train=Library(lib.days[[lookup[d] for d in ev['training_dates']]],lib.curves[[lookup[d] for d in ev['training_dates']]])
            assert train.mean==float(train.curves.mean())
            assert train.mean==ev['training_standardization']['mean_kw'] and train.std==ev['training_standardization']['std_kw']
            assert np.allclose(train.weather_scale,ev['training_standardization']['weather_scale'],rtol=1e-12,atol=1e-12)
            values=ensembles[station+'_load_kw']; dates=ensembles[station+'_dates']
            assert values.shape==(CONFIG['test_draws'],len(ev['test_dates']),24)
            assert list(dates)==ev['test_dates'] and np.isfinite(values).all() and (values>=0).all()
            for _,row in test_trace.loc[test_trace.station==station].iterrows():
                assert row.energy_source_date in ev['training_dates'] and row.shape_source_date in ev['training_dates']
                d=ev['test_dates'].index(row.target_date)
                curve=values[int(row.draw),d]
                e=lib.energy[lookup[row.energy_source_date]]
                template=lib.curves[lookup[row.shape_source_date]]
                assert abs(curve.sum()-e)<1e-6 and np.allclose(curve,template*row.shape_scale)
    assert not report['stations']['A10']['primary_annual_analysis']
    assert actual_weather().shape==(1461,2)
    # Repeat a short sample with the same seed; output and provenance must match.
    lib=read_library(ROOT/'data/load/cleaned/A1_hourly.csv')
    days=expected[::24][:14]
    a,ta=simulate(lib,days,3,np.random.default_rng(SEED))
    b,tb=simulate(lib,days,3,np.random.default_rng(SEED))
    assert np.array_equal(a,b) and ta.equals(tb)
    result=dict(status='passed',hours_per_station=35064,stations=10,provenance_days=14610,
        deterministic_sampling=True,source_reconstruction_checks=checks,
        independent_test_ensembles=True,test_draws_per_station=CONFIG['test_draws'],weather_source='ERA5 actuals',
        validation_note='Validation selects block/weather settings; separate blocked tests use train-only donors and statistics. Historical-conditioned generation, not forecasting or dispatch-cost validation.')
    (reports/'validation_checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
