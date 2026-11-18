"""Historical-weather truth scenarios with paired fragments and held-out ensembles."""
from pathlib import Path
from datetime import date
from functools import lru_cache
import json
import hashlib
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
import chinese_calendar as calendar
from test.load.source_quality import audit_sources

ROOT=Path(__file__).resolve().parents[2]
CONFIG=json.loads(Path(__file__).with_name('config.json').read_text(encoding='utf-8'))
SEED=CONFIG['seed']
SCALE_LIMIT=CONFIG['scale_limit']

@lru_cache(maxsize=3000)
def day_label(d):
    off,name=calendar.get_holiday_detail(d)
    return (0 if calendar.is_workday(d) else 2 if name else 1,
            str(name) if off and name else '',date(2000,d.month,d.day).timetuple().tm_yday)

def labels(days):
    rows=[day_label(ts.date()) for ts in days]
    return tuple(np.array([r[i] for r in rows]) for i in range(3))

@lru_cache(maxsize=1)
def actual_weather():
    """ERA5 actuals are intentional: generating assumed truth, not forecasting."""
    if CONFIG['weather_source']!='era5_reference':
        raise ValueError('This generator requires ERA5 actual-weather conditioning')
    f=pd.read_csv(ROOT/'data/weather/era5/hourly.csv')
    f.index=pd.DatetimeIndex(pd.to_datetime(f.valid_time_utc,utc=True).dt.tz_convert('Asia/Shanghai'))
    if f.index.duplicated().any(): raise ValueError('Duplicate ERA5 times')
    hourly=f[['air_temp_c','precipitation_mm']].apply(pd.to_numeric,errors='coerce')
    if not np.isfinite(hourly.to_numpy()).all(): raise ValueError('ERA5 actuals incomplete')
    groups=hourly.groupby(hourly.index.normalize())
    counts=groups.size()
    if not counts.eq(24).all(): raise ValueError('Incomplete ERA5 days')
    daily=pd.DataFrame({'temperature_c':groups.air_temp_c.mean(),
                        'log_rain':np.log1p(groups.precipitation_mm.sum())})
    return daily

class Library:
    def __init__(self,days,curves):
        self.days=pd.DatetimeIndex(days).as_unit('ns')
        self.curves=np.asarray(curves,dtype=float)
        if self.curves.shape!=(len(days),24) or not np.isfinite(curves).all() or (curves<0).any():
            raise ValueError('Invalid observed daily library')
        self.energy=self.curves.sum(axis=1)
        self.types,self.holidays,self.season=labels(self.days)
        self.mean=float(self.curves.mean())
        self.std=float(self.curves.std()) or 1.
        self.z=(self.curves-self.mean)/self.std
        # Weather scales and all transition statistics use these training dates only.
        self.weather=actual_weather().reindex(self.days).to_numpy()
        if not np.isfinite(self.weather).all(): raise ValueError('ERA5 dates missing')
        self.weather_scale=np.maximum(self.weather.std(axis=0),[2.,.3])
        self.windows={size:np.array([i for i in range(max(0,len(days)-size+1))
            if (self.days[i+size-1]-self.days[i]).days==size-1],dtype=int) for size in range(1,8)}
        self.consecutive=np.diff(self.days.asi8)==pd.Timedelta(days=1).value
        self.overnight=self.curves[1:,0][self.consecutive]-self.curves[:-1,-1][self.consecutive]
        self.previous_levels=self.curves[:-1,-1][self.consecutive]
        self.level_edges=np.quantile(self.previous_levels,[1/3,2/3])
        self.level_bins=np.searchsorted(self.level_edges,self.previous_levels)
        self.jump_kernels=[]
        for b in range(3):
            jumps=self.overnight[self.level_bins==b]
            if len(jumps)<8: jumps=self.overnight
            bandwidth=max(float(np.std(jumps))*.3,self.std*.03,1e-6)
            self.jump_kernels.append((np.quantile(jumps,np.linspace(.02,.98,25)),bandwidth))
        energy_delta=np.diff(np.log1p(self.energy))[self.consecutive]
        self.energy_transition_scale=max(float(energy_delta.std()),.15)
        self._candidates={}
        self._targets={}

    def target_features(self,days):
        key=(str(days[0].date()),len(days))
        if key not in self._targets:
            self._targets[key]=(labels(days),actual_weather().reindex(days).to_numpy())
        return self._targets[key]

    def candidate(self,target,length):
        key=(str(target[0].date()),length)
        if key in self._candidates: return self._candidates[key]
        tt,th,ts=labels(target[:length]); fallback=None
        for size in range(length,0,-1):
            starts=self.windows[size]
            if not len(starts): continue
            ids=starts[:,None]+np.arange(size)
            good=(self.types[ids]==tt[:size]).all(axis=1)
            if not good.any(): continue
            starts=starts[good]; ids=starts[:,None]+np.arange(size)
            d=np.abs(self.season[ids]-ts[:size]); distance=np.minimum(d,366-d).mean(axis=1)
            mismatch=((self.holidays[ids]!=th[:size]) & (th[:size]!='')).sum(axis=1)
            near=distance<=CONFIG['season_window_days']
            w=np.exp(-distance/45-1.5*mismatch)
            if near.any():
                result=(size,starts[near],w[near]/w[near].sum(),distance[near]); break
            if fallback is None: fallback=(size,starts,w/w.sum(),distance)
        else:
            if fallback is None: raise ValueError('No matching basic day type')
            result=fallback
        self._candidates[key]=result
        return result

    def boundary_weights(self,first,previous):
        if previous is None: return np.ones_like(np.asarray(first),dtype=float)
        b=int(np.searchsorted(self.level_edges,previous))
        locations,bandwidth=self.jump_kernels[b]
        diff=np.asarray(first)[...,None]-previous-locations
        # Density over SIGNED historical jumps; does not force a zero midnight jump.
        density=np.exp(-.5*(diff/bandwidth)**2).mean(axis=-1)
        return .03+.97*density

    def weather_weights(self,ids,target_weather):
        delta=(self.weather[ids]-target_weather)/self.weather_scale
        axis=tuple(range(1,delta.ndim))
        return np.exp(-.5*np.mean(delta**2,axis=axis))+.03

def read_observed_days(path,coverage=None):
    f=pd.read_csv(path,usecols=['interval_start_bjt','load_kw','fill_method'])
    f.index=pd.DatetimeIndex(pd.to_datetime(f.interval_start_bjt,utc=True).dt.tz_convert('Asia/Shanghai'))
    f=f.loc[f.fill_method.eq('observed')]
    if coverage is None:
        coverage=json.loads((ROOT/'test/reports/load/coverage_audit.json').read_text(encoding='utf-8'))
    info=coverage['stations'][path.stem.split('_')[0]]
    excluded=set(info['excluded_template_dates'])
    curves,days=[],[]
    for d,part in f.sort_index().groupby(f.index.normalize()):
        if (str(d.date()) not in excluded and info['first_evidence_date']<=str(d.date())<=info['last_evidence_date'] and len(part)==24
            and part.index.equals(pd.date_range(d,periods=24,freq='h'))
            and np.isfinite(part.load_kw).all() and part.load_kw.ge(0).all()):
            days.append(d); curves.append(part.load_kw.to_numpy())
    return pd.DatetimeIndex(days),np.asarray(curves).reshape(-1,24)


def read_library(path,coverage=None):
    return Library(*read_observed_days(path,coverage))

def choose_shape(lib,day,energy,own,rng,weather,replace_probability):
    """Preserve the paired donor, optionally change one day to a near-energy shape."""
    if rng.random()>=replace_probability: return own
    _,starts,weights,_=lib.candidate(pd.DatetimeIndex([day]),1)
    den=lib.energy[starts]
    ratio=np.divide(energy,den,out=np.ones_like(den),where=den>0)
    good=((den==0)==(energy==0)) & (ratio>=1-SCALE_LIMIT) & (ratio<=1+SCALE_LIMIT) & (starts!=own)
    if not good.any(): return own
    starts=starts[good]; weights=weights[good]
    weights*=np.exp(-np.abs(np.log(np.maximum(ratio[good],1e-12)))/.05)
    if weather:
        target=actual_weather().loc[day].to_numpy()
        weights*=lib.weather_weights(starts,target)
    weights/=weights.sum()
    return int(rng.choice(starts,p=weights))

def simulate(lib,target_days,block_days,rng,weather=False,replace_probability=None):
    target_days=pd.DatetimeIndex(target_days)
    replace_probability=CONFIG['shape_replace_probability'] if replace_probability is None else replace_probability
    _,weather_target=lib.target_features(target_days)
    generated,trace=[],[]; position=0; previous=None; previous_energy=None
    while position<len(target_days):
        size,starts,weights,distance=lib.candidate(target_days[position:],min(block_days,len(target_days)-position))
        weights=weights.copy(); ids=starts[:,None]+np.arange(size)
        if weather: weights*=lib.weather_weights(ids,weather_target[position:position+size])
        if previous_energy is not None:
            # Match the observed predecessor of each donor to the preceding simulated day.
            prev=np.maximum(starts-1,0)
            consecutive=(starts>0) & ((lib.days[starts]-lib.days[prev]).days==1)
            delta=np.log1p(lib.energy[prev])-np.log1p(previous_energy)
            weights*=np.where(consecutive,.05+.95*np.exp(-.5*(delta/lib.energy_transition_scale)**2),.15)
        weights/=weights.sum()
        proposals=[]
        for _ in range(CONFIG['boundary_proposals']):
            pos=int(rng.choice(len(starts),p=weights)); donor=int(starts[pos])
            energies=lib.energy[donor:donor+size]
            shapes=[choose_shape(lib,target_days[position+j],energies[j],donor+j,rng,weather,replace_probability) for j in range(size)]
            scales=np.divide(energies,lib.energy[shapes],out=np.ones(size),where=lib.energy[shapes]>0)
            curves=lib.curves[shapes]*scales[:,None]
            # Score FINAL scaled shapes, including internal boundaries changed by replacement.
            quality=float(lib.boundary_weights(curves[0,0],previous))
            altered=[]
            for j in range(1,size):
                if shapes[j]!=donor+j or shapes[j-1]!=donor+j-1:
                    altered.append(float(lib.boundary_weights(curves[j,0],curves[j-1,-1])))
            if altered: quality*=float(np.mean(altered))
            proposals.append((donor,shapes,scales,curves,distance[pos],quality))
        q=np.array([p[-1] for p in proposals]); q/=q.sum()
        donor,shapes,scales,curves,season_distance,quality=proposals[int(rng.choice(len(proposals),p=q))]
        for j in range(size):
            shape=shapes[j]
            shape_dist=abs(lib.season[shape]-day_label(target_days[position+j].date())[2])
            shape_dist=min(shape_dist,366-shape_dist)
            trace.append(dict(target_date=str(target_days[position+j].date()),energy_source_date=str(lib.days[donor+j].date()),
                shape_source_date=str(lib.days[shape].date()),energy_kwh=float(lib.energy[donor+j]),shape_scale=float(scales[j]),
                block_id=position,block_length=size,shape_replaced=bool(shape!=donor+j),
                season_fallback=bool(season_distance>CONFIG['season_window_days'] or shape_dist>CONFIG['season_window_days']),
                day_type=['workday','weekend','holiday'][day_label(target_days[position+j].date())[0]],
                weather_conditioned=weather,weather_source='ERA5_actual' if weather else 'calendar_only',
                final_boundary_weight=float(lib.boundary_weights(curves[j,0],previous if j==0 else curves[j-1,-1])),
                proposal_weight=quality))
        generated.extend(curves); previous=curves[-1,-1]; previous_energy=lib.energy[donor+size-1]; position+=size
    curves=np.asarray(generated)
    if not np.allclose(curves.sum(axis=1),[t['energy_kwh'] for t in trace],atol=1e-7): raise AssertionError('Energy mismatch')
    return curves,pd.DataFrame(trace)

def metrics(curves):
    curves=np.asarray(curves); x=curves.ravel(); daily=curves.sum(axis=1)
    active=x>np.quantile(x,.9); edges=np.diff(np.r_[False,active,False].astype(int))
    runs=np.flatnonzero(edges==-1)-np.flatnonzero(edges==1)
    return dict(mean_kw=float(x.mean()),zero_fraction=float(np.mean(x==0)),
        ramp_p95_kw=float(np.quantile(np.abs(np.diff(x)),.95)),lag1=lag_correlation(x),
        midnight_ramp_p95_kw=float(np.quantile(np.abs(curves[1:,0]-curves[:-1,-1]),.95)),
        daily_energy_mean_kwh=float(daily.mean()),daily_energy_p95_kwh=float(np.quantile(daily,.95)),
        daily_peak_p95_kw=float(np.quantile(curves.max(axis=1),.95)),high_run_p95_hours=float(np.quantile(runs,.95)) if len(runs) else 0.)

def lag_correlation(values):
    x=np.asarray(values).ravel()
    if np.std(x[:-1])<1e-12 or np.std(x[1:])<1e-12: return None
    return float(np.corrcoef(x[:-1],x[1:])[0,1])

def split_day_indices(days):
    width=14 if len(days)>=250 else 7
    windows=[np.arange(int((len(days)-width)*f),int((len(days)-width)*f)+width) for f in [.08,.28,.48,.68,.88]]
    validation=[windows[i] for i in [0,2,4]]; test=[windows[i] for i in [1,3]]
    excluded=np.concatenate(windows)
    if len(np.unique(excluded))!=len(excluded): raise ValueError('Overlapping split windows')
    keep=np.ones(len(days),dtype=bool); keep[excluded]=False
    return keep,validation,test


def split_dates(lib):
    keep,validation,test=split_day_indices(lib.days)
    training=Library(lib.days[keep],lib.curves[keep])
    return training,validation,test

def ensemble_score(actual,ensemble):
    """Distribution fidelity; ramps never cross simulated members or held-out windows."""
    scale=max(float(actual.mean()),.01)
    daily=ensemble.sum(axis=2); true_energy=actual.sum(axis=1)
    energy=wasserstein_distance(true_energy,daily.ravel())/(24*scale)
    peaks=wasserstein_distance(actual.max(axis=1),ensemble.max(axis=2).ravel())/scale
    ramps=np.abs(np.diff(ensemble.reshape(len(ensemble),-1),axis=1))
    true_ramp=float(np.quantile(np.abs(np.diff(actual.ravel())),.95))
    ramp=abs(float(np.quantile(ramps,.95,axis=1).mean())-true_ramp)/max(true_ramp,.1*scale)
    midnight=np.abs(ensemble[:,1:,0]-ensemble[:,:-1,-1])
    true_mid=float(np.quantile(np.abs(actual[1:,0]-actual[:-1,-1]),.95))
    mid=abs(float(np.quantile(midnight,.95,axis=1).mean())-true_mid)/max(true_mid,.1*scale)
    three=np.concatenate([np.convolve(d,np.ones(3),mode='valid') for d in daily])
    multi=wasserstein_distance(np.convolve(true_energy,np.ones(3),mode='valid'),three)/(72*scale)
    member_correlations=[lag_correlation(x) for x in ensemble]
    valid_correlations=[c for c in member_correlations if c is not None]
    actual_correlation=lag_correlation(actual)
    corr=abs(float(np.mean(valid_correlations))-actual_correlation) if valid_correlations and actual_correlation is not None else 0.
    zero=abs(float(np.mean(ensemble==0))-float(np.mean(actual==0)))
    quant=np.quantile(daily,[.1,.5,.9],axis=0)
    sorted_daily=np.sort(daily,axis=0); n=len(daily)
    crps=np.mean(np.abs(daily-true_energy),axis=0)-np.sum((2*np.arange(1,n+1)-n-1)[:,None]*sorted_daily,axis=0)/n**2
    hourly_q=np.quantile(ensemble,[.1,.5,.9],axis=0)
    threshold=float(np.quantile(actual,.9))
    def durations(x):
        edge=np.diff(np.r_[False,x.ravel()>threshold,False].astype(int))
        return np.flatnonzero(edge==-1)-np.flatnonzero(edge==1)
    actual_runs=durations(actual)
    sampled_runs=np.concatenate([durations(member) for member in ensemble])
    run_distance=float(wasserstein_distance(actual_runs,sampled_runs)) if len(actual_runs) and len(sampled_runs) else 0.
    return dict(score=float(energy+peaks+ramp+.5*mid+multi+corr+zero),daily_energy_w1_relative=float(energy),
        daily_peak_w1_relative=float(peaks),ramp_p95_relative_error=float(ramp),midnight_ramp_p95_relative_error=float(mid),
        three_day_energy_w1_relative=float(multi),lag1_absolute_error=float(corr),zero_fraction_error=float(zero),
        actual_lag1_defined=actual_correlation is not None,defined_member_lag1_count=len(valid_correlations),
        daily_energy_crps_kwh=float(np.mean(crps)),daily_energy_80pct_coverage=float(np.mean((true_energy>=quant[0]) & (true_energy<=quant[2]))),
        hourly_80pct_coverage=float(np.mean((actual>=hourly_q[0]) & (actual<=hourly_q[2]))),
        high_load_duration_w1_hours=run_distance,high_load_threshold_kw=threshold,
        daily_energy_p10_p50_p90_kwh=quant.tolist(),actual_daily_energy_kwh=true_energy.tolist(),
        hour_of_day_p10_p50_p90_kw=np.quantile(ensemble.reshape(-1,24),[.1,.5,.9],axis=0).tolist(),
        actual_hour_of_day_p10_p50_p90_kw=np.quantile(actual,[.1,.5,.9],axis=0).tolist())

def draw_ensemble(lib,days,length,weather,station,phase,fold,count,replace_probability=None):
    draws=[]; traces=[]
    for i in range(count):
        rng=np.random.default_rng(np.random.SeedSequence([SEED,station,phase,fold,i]))
        curve,trace=simulate(lib,days,length,rng,weather,replace_probability)
        trace.insert(0,'draw',i); draws.append(curve); traces.append(trace)
    return np.asarray(draws),pd.concat(traces,ignore_index=True)

def select_and_test(lib,station):
    training,validation,test=split_dates(lib)
    excluded={str(d.date()) for ids in validation+test for d in lib.days[ids]}
    rows=[]
    for length in CONFIG['block_candidates']:
        for weather in [False,True]:
            for fold,ids in enumerate(validation):
                for seed in CONFIG['validation_seeds']:
                    draws,trace=draw_ensemble(training,lib.days[ids],length,weather,station,seed,fold,CONFIG['validation_draws_per_seed'])
                    if set(trace.energy_source_date).intersection(excluded) or set(trace.shape_source_date).intersection(excluded):
                        raise AssertionError('Source date leakage')
                    m=ensemble_score(lib.curves[ids],draws)
                    rows.append(dict(block_days=length,weather=weather,fold=fold,seed=seed,score=m['score']))
            print(f'A{station}: validated block={length}, ERA5={weather}',flush=True)
    settings=[]
    for length in CONFIG['block_candidates']:
        for weather in [False,True]:
            scores=[r['score'] for r in rows if r['block_days']==length and r['weather']==weather]
            settings.append(dict(block_days=length,weather=weather,score=float(np.mean(scores)),score_std=float(np.std(scores))))
    chosen,gain=choose_setting(settings)
    tests=[]; arrays=[]; dates=[]; traces=[]
    # Untouched tests are inspected ONLY after freezing block length/weather selection.
    for fold,ids in enumerate(test):
        draws,trace=draw_ensemble(training,lib.days[ids],chosen['block_days'],chosen['weather'],station,900,fold,CONFIG['test_draws'])
        baseline,_=draw_ensemble(training,lib.days[ids],1,False,station,901,fold,CONFIG['test_draws'],0.)
        if set(trace.energy_source_date).intersection(excluded) or set(trace.shape_source_date).intersection(excluded):
            raise AssertionError('Test source leakage')
        trace.insert(0,'fold',fold); trace.insert(0,'station',f'A{station}')
        arrays.append(draws); dates.extend(str(d.date()) for d in lib.days[ids]); traces.append(trace)
        tests.append(dict(start=str(lib.days[ids[0]].date()),end=str(lib.days[ids[-1]].date()),
            selected=ensemble_score(lib.curves[ids],draws),paired_one_day_baseline=ensemble_score(lib.curves[ids],baseline)))
    return chosen,dict(training_dates=[str(d.date()) for d in training.days],
        training_standardization=dict(mean_kw=training.mean,std_kw=training.std,weather_scale=training.weather_scale.tolist()),
        validation_dates=[str(d.date()) for ids in validation for d in lib.days[ids]],test_dates=dates,
        selection='All 1/3/7-day settings compete; ERA5 used only when validation gain >= configured threshold; tests not used for selection',
        validation_settings=settings,validation_folds=rows,weather_validation_relative_gain=float(gain),
        test_draws=CONFIG['test_draws'],test_windows=tests),np.concatenate(arrays,axis=1),np.array(dates),pd.concat(traces,ignore_index=True)

def choose_setting(settings):
    plain=min([s for s in settings if not s['weather']],key=lambda s:s['score'])
    conditioned=min([s for s in settings if s['weather']],key=lambda s:s['score'])
    gain=(plain['score']-conditioned['score'])/max(plain['score'],1e-9)
    return (conditioned if gain>=CONFIG['weather_min_relative_gain'] else plain),gain

def main():
    coverage=audit_sources()
    out=ROOT/'data/load/simulated'; out.mkdir(parents=True,exist_ok=True)
    reports=ROOT/'test/reports/load'; reports.mkdir(parents=True,exist_ok=True)
    days=pd.date_range('2022-01-01','2026-01-01',inclusive='left',freq='D',tz='Asia/Shanghai')
    index=pd.date_range('2022-01-01','2026-01-01',inclusive='left',freq='h',tz='Asia/Shanghai')
    wide=pd.DataFrame(index=index); wide.index.name='interval_start_bjt'
    provenance=[]; test_provenance=[]; ensembles={}
    report=dict(method='Paired multi-day fragments with limited daily shape replacement and ERA5-actual validation',
        config=CONFIG,weather_source_file='data/weather/era5/hourly.csv',
        weather_source_sha256=hashlib.sha256((ROOT/'data/weather/era5/hourly.csv').read_bytes()).hexdigest(),
        transaction_source_sha256=coverage['source_sha256'],stations={},limitations=['Assumed truth scenarios under historical ERA5, not measured four-year load or a forecasting backtest.',
        'Zero loads within transaction coverage remain an unverified collection assumption.',
        'Cross-station dependence is not modeled. Shanghai equipment/tariff/PV dispatch-cost evaluation remains pending.',
        'A10 annual output is exploratory; its independent evaluation is confined to available observed seasons.',
        'Statistical test metrics do not establish improved real dispatch costs.'])
    for path in sorted((ROOT/'data/load/cleaned').glob('A*_hourly.csv'),key=lambda p:int(p.stem[1:].split('_')[0])):
        station=path.stem.split('_')[0]; number=int(station[1:]); lib=read_library(path,coverage)
        chosen,evaluation,test_draws,test_dates,test_trace=select_and_test(lib,number)
        rng=np.random.default_rng(np.random.SeedSequence([SEED,number,1000]))
        generated,trace=simulate(lib,days,chosen['block_days'],rng,chosen['weather'])
        values=generated.ravel()
        if len(values)!=35064 or not np.isfinite(values).all() or (values<0).any(): raise AssertionError('Invalid full series')
        single=pd.DataFrame({'load_kw':values,'load_z':(values-lib.mean)/lib.std,'is_synthetic':True},index=index)
        single.index.name=wide.index.name
        single.to_csv(out/f'{station}_hourly_2022_2025.csv',encoding='utf-8-sig'); wide[station+'_kw']=values
        trace.insert(0,'station',station); provenance.append(trace); test_provenance.append(test_trace)
        ensembles[station+'_load_kw']=test_draws; ensembles[station+'_dates']=test_dates
        report['stations'][station]=dict(observed_days=len(lib.days),block_days=chosen['block_days'],weather_conditioned=chosen['weather'],
            weather_source='ERA5_actual' if chosen['weather'] else 'calendar_only',primary_annual_analysis=station!='A10',
            standardization=dict(mean_kw=lib.mean,std_kw=lib.std,ddof=0,scope='observed complete days within transaction evidence; train-only in validation/test'),
            season_fallback_days=int(trace.season_fallback.sum()),distinct_shape_days=int(trace.shape_replaced.sum()),
            shape_scale_min=float(trace.shape_scale.min()),shape_scale_max=float(trace.shape_scale.max()),
            source_start=str(lib.days.min()),source_end=str(lib.days.max()),evaluation=evaluation,
            reference_metrics=metrics(lib.curves),generated_metrics=metrics(generated))
        print(f'{station}: block={chosen["block_days"]}, ERA5={chosen["weather"]}, replaced={trace.shape_replaced.sum()}, tests={len(test_dates)} days x {CONFIG["test_draws"]}',flush=True)
    wide.to_csv(out/'all_stations_hourly_2022_2025.csv',encoding='utf-8-sig')
    pd.concat(provenance,ignore_index=True).to_csv(out/'daily_provenance.csv',index=False,encoding='utf-8-sig')
    pd.concat(test_provenance,ignore_index=True).to_csv(reports/'test_provenance.csv',index=False,encoding='utf-8-sig')
    np.savez_compressed(reports/'test_ensembles.npz',**ensembles)
    (reports/'generation_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print('DONE: four-year assumed-truth loads and independent scenario ensembles.',flush=True)

if __name__=='__main__': main()
