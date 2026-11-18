"""Audit evidence for zero loads without relabeling ambiguous in-coverage zeros."""
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import numpy as np
import openpyxl
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

def incomplete_day_reasons(observed, first, last):
    """Conservatively require a full hourly grid AND a full evidence envelope.

    Boundary zeros before the first connection or after the last disconnection
    cannot establish full-day coverage. Internal zero periods are not excluded.
    """
    excluded={}
    for day,part in observed.groupby(observed.index.normalize()):
        reasons=[]
        expected=pd.date_range(day,periods=24,freq='h')
        if not part.index.equals(expected) or not np.isfinite(part.load_kw.to_numpy()).all():
            reasons.append('incomplete_or_invalid_hour_grid')
        left=day.to_pydatetime().replace(tzinfo=None)
        right=left+timedelta(days=1)
        if right<=first or left>=last:
            reasons.append('outside_transaction_envelope')
        else:
            if left<first<right:
                reasons.append('first_day_coverage_unconfirmed')
            if left<last<right:
                reasons.append('last_day_incomplete_coverage')
        if reasons:
            excluded[str(day.date())]=reasons
    return excluded

def audit_sources():
    source = ROOT/'data_raw/load/MP_EVData/charging_session.xlsx'
    book = openpyxl.load_workbook(source, read_only=True, data_only=True)
    result = dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        zero_policy='Internal no-transaction zeros remain unverified and unchanged. Templates exclude incomplete hourly grids, days outside the transaction envelope, and first/last days not fully enclosed by evidence. Boundary exclusion is conservative, not proof of missing transactions. Raw and aligned load records are unchanged.',
        stations={})
    for path in sorted((ROOT/'data/load/cleaned').glob('A*_hourly.csv')):
        station = path.stem.split('_')[0]
        first, last, invalid, count = None, None, 0, 0
        for start, end, energy in book[station].iter_rows(min_row=2,max_col=3,values_only=True):
            if (not isinstance(start,datetime) or not isinstance(end,datetime) or end<=start
                or not isinstance(energy,(int,float)) or not np.isfinite(energy) or energy<0):
                invalid += 1
                continue
            count += 1
            first = start if first is None else min(first,start)
            last = end if last is None else max(last,end)
        if first is None:
            raise ValueError(f'{station}: no valid transactions')
        f = pd.read_csv(path)
        f.index = pd.DatetimeIndex(pd.to_datetime(f.interval_start_bjt,utc=True).dt.tz_convert('Asia/Shanghai'))
        observed = f.loc[f.fill_method.eq('observed')]
        days = observed.groupby(observed.index.normalize()).load_kw.agg(['count','sum'])
        complete = days.loc[days['count'].eq(24)]
        excluded=incomplete_day_reasons(observed,first,last)
        first_day, last_day = str(first.date()), str((last-timedelta(microseconds=1)).date())
        outside = (complete.index.date < first.date()) | (complete.index.date > (last-timedelta(microseconds=1)).date())
        zero = complete['sum'].eq(0)
        result['stations'][station] = dict(valid_sessions=count,invalid_sessions=invalid,
            first_connection=first.isoformat(),last_disconnection=last.isoformat(),
            first_evidence_date=first_day,last_evidence_date=last_day,
            excluded_template_dates=sorted(excluded),excluded_template_reasons=excluded,
            eligible_template_days=sum(str(d.date()) not in excluded for d in complete.index),
            published_complete_days=len(complete),excluded_outside_envelope_dates=[str(d.date()) for d in complete.index[outside]],
            ambiguous_internal_zero_dates=[str(d.date()) for d in complete.index[zero & ~outside]],
            published_zero_days=int(zero.sum()),all_zeros_meter_verified=False)
    book.close()
    target=ROOT/'test/reports/load/coverage_audit.json'
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result

if __name__=='__main__':
    r=audit_sources()
    print({s:v['excluded_template_dates'] for s,v in r['stations'].items()})
