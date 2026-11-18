"""Fill verified GFS point caches through the public Dynamical forecast API."""

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from tempfile import gettempdir

import requests

from model.preprocess import fetch_gfs as gfs
from model.preprocess.fetch_gfs_mirror import run_index


API = "https://api.dynamical.org/v1/forecasts"
LOCAL = threading.local()


def fetch_day(day, cache_dir):
    if not hasattr(LOCAL, "session"):
        LOCAL.session = requests.Session()
        LOCAL.session.trust_env = False
    gfs.LOCAL.session = LOCAL.session
    run_info = gfs.select_run(day)
    init, decision, lead, _ = run_info
    path = cache_dir / f"{gfs.VERSION}_235_1206_f{lead:03d}_run{run_index(init)}.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        list(gfs.standardize(run_info, data, 31.25, 121.5))
        return day, "cached"
    query = {
        "dataProductId": gfs.PRODUCT,
        "location": gfs.LOCATION,
        "variables": list(gfs.VARIABLES),
        "initTime": gfs.iso_utc(init),
        "validTimeStart": gfs.iso_utc(decision),
        "validTimeEnd": gfs.iso_utc(decision + 25 * gfs.HOUR),
    }
    for attempt in range(6):
        try:
            response = LOCAL.session.post(API, json={"queries": [query]}, timeout=(15, 120))
            if response.status_code in (408, 429, 500, 502, 503, 504):
                retry = response.headers.get("Retry-After", "")
                delay = float(retry) if retry.isdecimal() else min(30, 2 ** attempt)
                if attempt == 5:
                    response.raise_for_status()
                time.sleep(delay)
                continue
            response.raise_for_status()
            payload = response.json()
            result, = payload["results"]
            forecast, = result["forecasts"]
            assert result["dataProduct"] == {"id": gfs.PRODUCT, "version": gfs.VERSION}
            assert result["point"]["selected"]["latitude"] == 31.25
            assert result["point"]["selected"]["longitude"] == 121.5
            assert forecast["initTime"] == gfs.iso_utc(init)
            assert forecast["leadTimeHours"] == list(range(lead, lead + 25))
            assert forecast["validTimes"] == [gfs.iso_utc(decision + h * gfs.HOUR) for h in range(25)]
            assert {name: val["unit"] for name, val in result["variables"].items()} == gfs.VARIABLES
            data = forecast["data"]
            assert set(data) == set(gfs.VARIABLES)
            list(gfs.standardize(run_info, data, 31.25, 121.5))
            draft = path.with_suffix(f".{os.getpid()}.tmp")
            draft.write_text(json.dumps(data), encoding="utf-8")
            draft.replace(path)
            raw_path = cache_dir / "api_responses" / path.name
            raw_path.parent.mkdir(exist_ok=True)
            raw_path.write_text(json.dumps(payload), encoding="utf-8")
            return day, "downloaded"
        except (requests.ConnectionError, requests.Timeout):
            if attempt == 5:
                raise
            time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(f"API download failed for {day}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path(__file__).resolve().parents[2] / "data_raw" / "weather" / "gfs" / "points")
    args = parser.parse_args()
    days = [args.start + timedelta(days=i) for i in reversed(range((args.end - args.start).days))]
    if not days:
        parser.error("--end must follow --start")
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = {pool.submit(fetch_day, day, args.cache_dir): day for day in days}
        for i, job in enumerate(as_completed(jobs), 1):
            day, status = job.result()
            print(f"{i}/{len(days)} {day} {status}", flush=True)


if __name__ == "__main__":
    main()
