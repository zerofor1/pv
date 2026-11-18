"""Download official ECMWF ERA5-Land ARCO point series and align hourly periods."""

import argparse
import asyncio
import csv
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from tempfile import gettempdir
from typing import cast

import numpy as np
import requests
import zarr
from aiohttp import ClientConnectionError, ClientPayloadError, ClientResponseError, ClientTimeout
from zarr.storage import FsspecStore


HERE = Path(__file__).resolve().parent
UTC = timezone.utc
BJT = timezone(timedelta(hours=8))
HOUR = timedelta(hours=1)
LOCATION = (31.2304, 121.4737)
BASE = "https://arco.datastores.ecmwf.int"
GROUPS = {
    "temperature": ("007", "2m-temperature", {
        "temperature": ("2m_temperature", "t2m"),
        "dewpoint": ("2m_dewpoint_temperature", "d2m"),
    }),
    "wind": ("008", "wind", {
        "wind_u": ("10m_u_component_of_wind", "u10"),
        "wind_v": ("10m_v_component_of_wind", "v10"),
    }),
    "pressure_precipitation": ("009", "pressure-precipitation", {
        "pressure": ("surface_pressure", "sp"),
        "precipitation": ("total_precipitation", "tp"),
    }),
    "radiation": ("010", "radiation-heat", {
        "radiation": ("surface_solar_radiation_downwards", "ssrd"),
    }),
}


def timestamp(value):
    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def api_key():
    value = os.environ.get("CDSAPI_KEY", "")
    config = Path(os.environ.get("CDSAPI_RC", str(Path.home() / ".cdsapirc")))
    if not value and config.is_file():
        for line in config.read_text(encoding="utf-8").splitlines():
            name, separator, content = line.partition(":")
            if separator and name.strip() == "key":
                value = content.strip().strip("\"'")
    if not value:
        raise RuntimeError(
            "CDS API key missing. Configure key in ~/.cdsapirc or CDSAPI_KEY, "
            "and accept the ERA5-Land licence on the CDS website."
        )
    return value


class RetryingArchive(FsspecStore):
    async def get(self, key, prototype, byte_range=None):
        for attempt in range(5):
            try:
                return await super().get(key, prototype, byte_range)
            except ClientResponseError as exc:
                if exc.status in (401, 403):
                    raise RuntimeError("ERA5-Land access denied. Check the local CDS API key and licence acceptance.") from None
                if attempt == 4 or exc.status not in (408, 429, 500, 502, 503, 504):
                    raise
            except (ClientConnectionError, ClientPayloadError, TimeoutError):
                if attempt == 4:
                    raise
            await asyncio.sleep(2 ** attempt)


def decode_times(array):
    values = array[:]
    units = array.attrs.get("units", "")
    step, marker, origin = units.partition(" since ")
    factors = {"seconds": 1, "hours": 3600, "days": 86400, "milliseconds": .001, "microseconds": .000001}
    if not marker or step not in factors:
        raise ValueError(f"Unsupported ERA5-Land time units: {units}")
    base = datetime.fromisoformat(origin.strip().replace("Z", "+00:00"))
    if base.tzinfo is None:
        base = base.replace(tzinfo=UTC)
    return np.asarray(values, dtype=float) * factors[step] + base.timestamp()


def read_group(group_name, key, start, end, cache_dir):
    number, directory, variables = GROUPS[group_name]
    url = f"{BASE}/cadl-arco-geo-{number}/arco/reanalysis_era5_land/sfc-{directory}/geoChunked.zarr"
    cache = cache_dir / f"{group_name}_{start:%Y%m%d%H}_{end:%Y%m%d%H}.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as saved:
            values = {name: saved[name] for name in variables}
            metadata = json.loads(str(saved["metadata"]))
        print(f"ERA5-Land {group_name}: cached {len(next(iter(values.values())))} hours", flush=True)
        return values, metadata

    headers = {"Authorization": f"Bearer {key}"}
    # Check authentication before Zarr converts HTTP errors into missing-store errors.
    session = requests.Session()
    session.trust_env = False
    response = session.get(f"{url}/.zmetadata", headers=headers, timeout=(15, 60))
    if response.status_code in (401, 403):
        raise RuntimeError("ERA5-Land access denied. Check the local CDS API key and licence acceptance.")
    response.raise_for_status()
    store = RetryingArchive.from_url(
        url,
        storage_options={"headers": headers, "client_kwargs": {"trust_env": False, "timeout": ClientTimeout(total=120)}},
        read_only=True,
    )
    # This archive is consolidated Zarr v2. Probing v3 metadata returns HTTP 403.
    group = zarr.open_group(store, mode="r", zarr_format=2, use_consolidated=True)
    latitudes = np.asarray(cast(zarr.Array, group["latitude"])[:], dtype=float)
    longitudes = np.asarray(cast(zarr.Array, group["longitude"])[:], dtype=float)
    lat_index = int(abs(latitudes - LOCATION[0]).argmin())
    lon_index = int(abs(longitudes - LOCATION[1]).argmin())
    times = decode_times(group["time"])
    first = int(np.searchsorted(times, start.timestamp()))
    stop = int(np.searchsorted(times, end.timestamp(), side="right"))
    expected = np.arange(start.timestamp(), end.timestamp() + 3600, 3600)
    if not np.array_equal(times[first:stop], expected):
        raise ValueError(f"ERA5-Land {group_name}: incomplete hourly time coordinates")

    values, attributes = {}, {}
    for name, aliases in variables.items():
        found = [alias for alias in aliases if alias in group]
        if len(found) != 1:
            raise ValueError(f"ERA5-Land {group_name}: expected exactly one of {aliases}")
        array = cast(zarr.Array, group[found[0]])
        dimension_names = array.attrs.get("_ARRAY_DIMENSIONS", getattr(array.metadata, "dimension_names", None) or [])
        if not isinstance(dimension_names, (list, tuple)):
            raise ValueError(f"Invalid ERA5-Land dimensions for {found[0]}: {dimension_names}")
        dimensions = [str(name) for name in dimension_names]
        selection = {"time": slice(first, stop), "latitude": lat_index, "longitude": lon_index}
        if set(dimensions) != set(selection):
            raise ValueError(f"Unexpected ERA5-Land dimensions for {found[0]}: {dimensions}")
        values[name] = np.asarray(array[tuple(selection[dim] for dim in dimensions)], dtype=float)
        if values[name].shape != expected.shape or not np.isfinite(values[name]).all():
            raise ValueError(f"ERA5-Land {name}: missing values or unexpected shape")
        attributes[name] = {"variable": found[0], "units": array.attrs.get("units"), "attributes": dict(array.attrs)}
        unit = str(attributes[name]["units"]).replace("**", "^")
        allowed = {
            "temperature": {"K"}, "dewpoint": {"K"}, "pressure": {"Pa"},
            "wind_u": {"m s-1", "m s^-1", "m/s"}, "wind_v": {"m s-1", "m s^-1", "m/s"},
            "precipitation": {"m"}, "radiation": {"J m-2", "J m^-2", "J/m^2"},
        }
        if unit not in allowed[name]:
            raise ValueError(f"Unexpected ERA5-Land units for {name}: {unit}")
    metadata = {
        "url": url,
        "grid_latitude": float(latitudes[lat_index]),
        "grid_longitude": float(longitudes[lon_index]),
        "start_utc": timestamp(start),
        "end_utc_inclusive": timestamp(end),
        "fetched_at_utc": timestamp(datetime.now(UTC)),
        "variables": attributes,
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    draft = cache.with_suffix(f".{os.getpid()}.tmp")
    with draft.open("wb") as stream:
        np.savez_compressed(stream, metadata=json.dumps(metadata), **values)
    draft.replace(cache)
    print(f"ERA5-Land {group_name}: downloaded {len(expected)} hours", flush=True)
    return values, metadata


def build_rows(values, metadata, start, hours):
    for hour in range(hours):
        instant = start + hour * HOUR
        end = instant + HOUR
        local = instant.astimezone(BJT)
        temp = float(values["temperature"][hour]) - 273.15
        dewpoint = float(values["dewpoint"][hour]) - 273.15
        rh = 100 * math.exp(17.625 * dewpoint / (243.04 + dewpoint) - 17.625 * temp / (243.04 + temp))
        u, v = float(values["wind_u"][hour]), float(values["wind_v"][hour])
        speed = math.hypot(u, v)
        # Official ARCO energy and precipitation values already represent the preceding hour.
        radiation_j_m2 = float(values["radiation"][hour + 1])
        precipitation_m = float(values["precipitation"][hour + 1])
        row = {
            "site_id": "shanghai_reference",
            "local_date": local.date().isoformat(),
            "split": "train" if local.date() < date(2024, 3, 1) else "validation" if local.date() < date(2024, 5, 1) else "test",
            "interval_start_bjt": local.isoformat(),
            "interval_end_bjt": end.astimezone(BJT).isoformat(),
            "valid_time_utc": timestamp(instant),
            "interval_end_utc": timestamp(end),
            "site_latitude": LOCATION[0],
            "site_longitude": LOCATION[1],
            "grid_latitude": metadata["grid_latitude"],
            "grid_longitude": metadata["grid_longitude"],
            "ghi_w_m2": max(0., radiation_j_m2 / 3600),
            "ghi_clipped": int(radiation_j_m2 < 0),
            "air_temp_c": temp,
            "dew_point_c": dewpoint,
            "relative_humidity_pct": min(100., max(0., rh)),
            "relative_humidity_clipped": int(not 0 <= rh <= 100),
            "wind_u_10m_m_s": u,
            "wind_v_10m_m_s": v,
            "wind_speed_m_s": speed,
            "wind_direction_deg": math.degrees(math.atan2(-u, -v)) % 360 if speed else "",
            "precipitation_mm": max(0., precipitation_m * 1000),
            "precipitation_clipped": int(precipitation_m < 0),
            "surface_pressure_pa": float(values["pressure"][hour]),
            "source": "ECMWF_ERA5-Land_ARCO",
        }
        limits = {"ghi_w_m2": (0, 1500), "air_temp_c": (-30, 60), "dew_point_c": (-65, 60),
                  "relative_humidity_pct": (0, 100), "wind_speed_m_s": (0, 100),
                  "precipitation_mm": (0, 300), "surface_pressure_pa": (85000, 110000)}
        if any(not low <= row[field] <= high for field, (low, high) in limits.items()):
            raise ValueError(f"ERA5-Land physical range check failed at {timestamp(instant)}")
        yield row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2022, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 1, 1), help="Exclusive BJT date")
    parser.add_argument("--output", type=Path, default=HERE.parents[1] / "data" / "weather" / "era5" / "hourly.csv")
    parser.add_argument("--cache-dir", type=Path, default=Path(__file__).resolve().parents[2] / "data_raw" / "weather" / "era5" / "points")
    args = parser.parse_args()
    hours = (args.end - args.start).days * 24
    if hours <= 0:
        parser.error("--end must follow --start")
    key = api_key()
    start = datetime.combine(args.start, datetime.min.time(), BJT).astimezone(UTC)
    end = start + hours * HOUR
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(read_group, name, key, start, end, args.cache_dir) for name in GROUPS]
        results = [job.result() for job in jobs]
    values = {name: data for result, _ in results for name, data in result.items()}
    metadata = [meta for _, meta in results]
    coordinates = {(meta["grid_latitude"], meta["grid_longitude"]) for meta in metadata}
    if len(coordinates) != 1:
        raise ValueError("ERA5-Land variable groups use different grid points")
    rows = list(build_rows(values, metadata[0], start, hours))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    draft = args.output.with_suffix(f".{os.getpid()}.tmp")
    with draft.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    draft.replace(args.output)
    args.output.with_suffix(".sources.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(rows)} ERA5-Land hours to {args.output}", flush=True)


if __name__ == "__main__":
    main()
