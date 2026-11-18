"""Build the Shanghai day-ahead GFS forecast dataset."""

import argparse
import asyncio
import csv
import json
import math
import os
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from tempfile import gettempdir
from typing import cast
from xml.etree import ElementTree

import requests
import numpy as np
import zarr
from aiohttp import ClientConnectionError, ClientPayloadError, ClientResponseError, ClientTimeout
from zarr.storage import FsspecStore


HERE = Path(__file__).resolve().parent
UTC = timezone.utc
BJT = timezone(timedelta(hours=8))
HOUR = timedelta(hours=1)
BUCKET = "https://noaa-gfs-bdp-pds.s3.amazonaws.com/"
PRODUCT = "noaa-gfs-forecast"
VERSION = "0.2.7"
ARCHIVE = (
    "https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop/"
    f"dynamical/{PRODUCT}/v{VERSION}.zarr"
)
LOCATION = {"latitude": 31.2304, "longitude": 121.4737}
XML = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
VARIABLES = {
    "temperature_2m": "degree_Celsius",
    "wind_u_10m": "m s-1",
    "wind_v_10m": "m s-1",
    "pressure_surface": "Pa",
    "relative_humidity_2m": "percent",
    "total_cloud_cover_atmosphere": "percent",
    "downward_short_wave_radiation_flux_surface": "W m-2",
    "precipitation_surface": "kg m-2 s-1",
}
WEATHER_LIMITS = {
    "ghi_w_m2": (0, 1500),
    "air_temp_c": (-30, 60),
    "dew_point_estimated_c": (-65, 60),
    "relative_humidity_pct": (0, 100),
    "wind_speed_m_s": (0, 100),
    "total_cloud_cover_pct": (0, 100),
    "precipitation_mm": (0, 300),
    "surface_pressure_pa": (85000, 110000),
}
LOCAL = threading.local()
UNAVAILABLE_RUNS = {
    datetime(2022, 11, 29, 12, tzinfo=UTC),
    datetime(2022, 11, 29, 18, tzinfo=UTC),
    datetime(2022, 11, 30, 0, tzinfo=UTC),
    datetime(2022, 11, 30, 6, tzinfo=UTC),
}


def iso_utc(value):
    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def request(method, url, **kwargs):
    if not hasattr(LOCAL, "session"):
        LOCAL.session = requests.Session()
    error: Exception = RuntimeError(f"Request failed for {url}")
    for attempt in range(5):
        try:
            response = LOCAL.session.request(method, url, timeout=(15, 90), **kwargs)
            if response.status_code not in (408, 429, 500, 502, 503, 504):
                response.raise_for_status()
                return response
            error = RuntimeError(f"HTTP {response.status_code}: {response.text[:180]}")
            retry_after = response.headers.get("Retry-After", "")
        except (requests.ConnectionError, requests.Timeout) as exc:
            error = exc
            retry_after = ""
        if attempt < 4:
            delay = float(retry_after) if retry_after.isdecimal() else min(2 ** attempt, 20)
            time.sleep(delay)
    raise error


def select_run(day):
    """Use the latest candidate whose entire input window was in the NOAA archive."""
    decision = datetime.combine(day, datetime.min.time(), BJT).astimezone(UTC)
    audit_dir = Path(gettempdir()) / "shanghai-gfs-availability"
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_path = audit_dir / f"{day}.json"
    if audit_path.exists():
        cached = json.loads(audit_path.read_text(encoding="utf-8"))
        cached_init = datetime.fromisoformat(cached["init"])
        if cached_init not in UNAVAILABLE_RUNS:
            return (
                cached_init, decision, cached["lead"],
                datetime.fromisoformat(cached["available"]),
            )
    for start_lead in range(4, 65, 6):
        run = decision - start_lead * HOUR
        if run in UNAVAILABLE_RUNS:
            continue
        directory = "atmos/" if run.date() >= date(2021, 3, 22) else ""
        prefix = f"gfs.{run:%Y%m%d}/{run:%H}/{directory}gfs.t{run:%H}z.pgrb2.0p25.f"
        response = request(
            "GET",
            BUCKET,
            params={
                "list-type": 2,
                "prefix": prefix,
                "start-after": f"{prefix}{start_lead - 1:03d}.idx",
                "max-keys": 50,
            },
        )
        contents = ElementTree.fromstring(response.content).findall("s3:Contents", XML)
        objects = {
            entry.findtext("s3:Key", namespaces=XML): datetime.fromisoformat(
                entry.findtext("s3:LastModified", default="", namespaces=XML).replace("Z", "+00:00")
            )
            for entry in contents
        }
        required = [f"{prefix}{lead:03d}" for lead in range(start_lead, start_lead + 25)]
        if all(key in objects and objects[key] < decision for key in required):
            available = max(objects[key] for key in required)
            temp = audit_path.with_suffix(f".{os.getpid()}.tmp")
            temp.write_text(json.dumps({"init": run.isoformat(), "lead": start_lead, "available": available.isoformat()}), encoding="utf-8")
            temp.replace(audit_path)
            return run, decision, start_lead, available
    raise ValueError(f"{day}: no complete GFS cycle published before midnight (BJT)")


class RetryingArchive(FsspecStore):
    async def get(self, key, prototype, byte_range=None):
        for attempt in range(5):
            try:
                return await super().get(key, prototype, byte_range)
            except ClientResponseError as exc:
                if attempt == 4 or exc.status not in (408, 429, 500, 502, 503, 504):
                    raise
            except (ClientConnectionError, ClientPayloadError, TimeoutError):
                if attempt == 4:
                    raise
            await asyncio.sleep(2 ** attempt)


def load_field(array, run_indices, lead_start, lat_index, lon_index):
    return array.oindex[run_indices, lead_start:lead_start + 25, lat_index, lon_index].tolist()


def hourly_mean(previous_mean, current_mean, end_lead):
    window_hours = (end_lead - 1) % 6 + 1
    return window_hours * current_mean - (window_hours - 1) * previous_mean


def standardize(run_info, data, grid_lat, grid_lon):
    init, decision, start_lead, available = run_info
    for field, values in data.items():
        if len(values) != 25 or any(not math.isfinite(v) for v in values):
            raise ValueError(f"{iso_utc(init)}: missing or non-finite values in {field}")

    radiation = data["downward_short_wave_radiation_flux_surface"]
    clouds = data["total_cloud_cover_atmosphere"]
    for hour in range(24):
        start = decision + hour * HOUR
        end = start + HOUR
        local = start.astimezone(BJT)
        end_lead = start_lead + hour + 1
        ghi_raw = hourly_mean(radiation[hour], radiation[hour + 1], end_lead)
        cloud_raw = hourly_mean(clouds[hour], clouds[hour + 1], end_lead)
        temp = data["temperature_2m"][hour]
        rh = data["relative_humidity_2m"][hour]
        u = data["wind_u_10m"][hour]
        v = data["wind_v_10m"][hour]
        speed = math.hypot(u, v)
        gamma = math.log(rh / 100) + 17.625 * temp / (243.04 + temp)
        row = {
            "site_id": "shanghai_reference",
            "local_date": local.date().isoformat(),
            "split": (
                "train" if local.date() < date(2024, 3, 1)
                else "validation" if local.date() < date(2024, 5, 1)
                else "test"
            ),
            "interval_start_bjt": local.isoformat(),
            "interval_end_bjt": end.astimezone(BJT).isoformat(),
            "valid_time_utc": iso_utc(start),
            "interval_end_utc": iso_utc(end),
            "decision_cutoff_utc": iso_utc(decision),
            "forecast_init_utc": iso_utc(init),
            "forecast_init_bjt": init.astimezone(BJT).isoformat(),
            "forecast_available_by_utc": iso_utc(available),
            "lead_hours": end_lead - 1,
            "interval_end_lead_hours": end_lead,
            "site_latitude": LOCATION["latitude"],
            "site_longitude": LOCATION["longitude"],
            "grid_latitude": grid_lat,
            "grid_longitude": grid_lon,
            "ghi_w_m2": max(0.0, ghi_raw),
            "ghi_clipped": int(ghi_raw < 0),
            "air_temp_c": temp,
            "dew_point_estimated_c": round(243.04 * gamma / (17.625 - gamma), 4),
            "relative_humidity_pct": rh,
            "wind_u_10m_m_s": u,
            "wind_v_10m_m_s": v,
            "wind_speed_m_s": round(speed, 5),
            "wind_direction_deg": round(math.degrees(math.atan2(-u, -v)) % 360, 4) % 360 if speed else "",
            "total_cloud_cover_pct": min(100.0, max(0.0, cloud_raw)),
            "cloud_cover_clipped": int(not 0 <= cloud_raw <= 100),
            "precipitation_mm": round(data["precipitation_surface"][hour + 1] * 3600, 6),
            "surface_pressure_pa": data["pressure_surface"][hour],
            "source": PRODUCT,
            "source_version": VERSION,
        }
        for field, (low, high) in WEATHER_LIMITS.items():
            if not low <= row[field] <= high:
                raise ValueError(f"{iso_utc(start)}: {field} out of range: {row[field]}")
        yield row


def write_csv(output, rows):
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(f".{os.getpid()}.tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(output)


def read_forecasts(runs, cache_dir):
    store = RetryingArchive.from_url(
        ARCHIVE,
        storage_options={"client_kwargs": {"trust_env": True, "timeout": ClientTimeout(total=90)}},
        read_only=True,
    )
    group = zarr.open_group(store, mode="r")
    init_times = np.asarray(cast(zarr.Array, group["init_time"])[:])
    init_index = {int(timestamp): i for i, timestamp in enumerate(init_times)}
    latitudes = np.asarray(cast(zarr.Array, group["latitude"])[:], dtype=float)
    longitudes = np.asarray(cast(zarr.Array, group["longitude"])[:], dtype=float)
    lat_idx = int(abs(latitudes - LOCATION["latitude"]).argmin())
    lon_idx = int(abs(longitudes - LOCATION["longitude"]).argmin())
    leads = np.asarray(cast(zarr.Array, group["lead_time"])[:]).tolist()
    fields = {var: cast(zarr.Array, group[var]) for var in VARIABLES}
    for name, array in fields.items():
        if array.attrs["units"] != VARIABLES[name]:
            raise ValueError(f"Unexpected units for {name}")
    for lead in sorted({run[2] for run in runs}):
        if leads[lead:lead + 25] != [h * 3600 for h in range(lead, lead + 25)]:
            raise ValueError(f"Incorrect archive lead time coordinates for f{lead:03d}")

    rows = []
    # Read runs with the same lead window together so each variable uses one array slice.
    cache_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool, zarr.config.set({"async.concurrency": 10}):
        for lead in sorted({run[2] for run in runs}):
            group_runs = [run for run in runs if run[2] == lead]
            for offset in range(0, len(group_runs), 24):
                batch = group_runs[offset:offset + 24]
                indices = [init_index[int(run[0].timestamp())] for run in batch]
                key = f"{VERSION}_{lat_idx}_{lon_idx}_f{lead:03d}_{indices[0]}_{indices[-1]}"
                cache = cache_dir / f"{key}.json"
                saved = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
                is_cached = saved.get("indices") == indices and set(saved.get("data", {})) == set(VARIABLES)
                if is_cached:
                    values = saved["data"]
                else:
                    pending = {
                        name: pool.submit(load_field, array, indices, lead, lat_idx, lon_idx)
                        for name, array in fields.items()
                    }
                    values = {name: future.result() for name, future in pending.items()}
                for pos, run in enumerate(batch):
                    data = {name: series[pos] for name, series in values.items()}
                    rows.extend(standardize(run, data, float(latitudes[lat_idx]), float(longitudes[lon_idx])))
                if not is_cached:
                    temporary = cache.with_suffix(f".{os.getpid()}.tmp")
                    temporary.write_text(json.dumps({"indices": indices, "data": values}), encoding="utf-8")
                    temporary.replace(cache)
                print(f"Downloaded and standardized {len(rows)} / {24 * len(runs)} hours", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2022, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 1, 1), help="Exclusive BJT date")
    parser.add_argument("--output", type=Path, default=HERE.parents[1] / "data" / "weather" / "gfs" / "hourly.csv")
    parser.add_argument("--cache-dir", type=Path, help="Optional root for raw and mirror caches")
    args = parser.parse_args()
    num_days = (args.end - args.start).days
    if num_days <= 0:
        parser.error("--end must follow --start")
    days = [args.start + timedelta(days=offset) for offset in range(num_days)]
    rows = []
    # Reuse complete daily snapshots already in the output, including the first two-year build.
    if args.output.exists():
        with args.output.open(encoding="utf-8-sig", newline="") as stream:
            previous = list(csv.DictReader(stream))
        by_day = {}
        for row in previous:
            by_day.setdefault(row["local_date"], []).append(row)
        for day in days:
            daily = by_day.get(day.isoformat(), [])
            first = datetime.combine(day, datetime.min.time(), BJT)
            expected = [iso_utc(first + h * HOUR) for h in range(24)]
            if [r["valid_time_utc"] for r in daily] == expected:
                rows.extend(daily)
        finished = {row["local_date"] for row in rows}
        days = [day for day in days if day.isoformat() not in finished]
        print(f"Reused {len(rows)} existing hours", flush=True)
    with ThreadPoolExecutor(max_workers=12) as pool:
        runs = list(pool.map(select_run, days))
    print(f"Verified pre-decision availability for {len(runs)} new days", flush=True)
    early = [run for run in runs if run[0] < datetime(2021, 5, 1, tzinfo=UTC)]
    mirrored = [run for run in runs if run[0] >= datetime(2021, 5, 1, tzinfo=UTC)]
    if early:
        raise ValueError('Shanghai downloader supports the mirror archive from 2021-05 onward; the project target is 2022-2025.')
    if mirrored:
        from model.preprocess.fetch_gfs_mirror import collect
        mirror_cache = args.cache_dir / "mirror" if args.cache_dir else Path(__file__).resolve().parents[2] / "data_raw" / "weather" / "gfs" / "points"
        rows.extend(collect(mirrored, mirror_cache))
    rows.sort(key=lambda row: row["valid_time_utc"])
    first = datetime.combine(args.start, datetime.min.time(), BJT)
    expected = [iso_utc(first + i * HOUR) for i in range(24 * num_days)]
    if [row["valid_time_utc"] for row in rows] != expected:
        raise ValueError("Output has duplicate or missing time slots")
    write_csv(args.output, rows)
    print(f"Saved {len(rows)} hours to {args.output}", flush=True)
    print("Hours by split:", dict(Counter(row["split"] for row in rows)), flush=True)
    print("Runs by cycle:", dict(Counter(row["forecast_init_utc"][11:13] for row in rows[::24])), flush=True)
    print("Clipped GHI/cloud:", sum(int(r["ghi_clipped"]) for r in rows), sum(int(r["cloud_cover_clipped"]) for r in rows), flush=True)


if __name__ == "__main__":
    main()
