"""Read the GFS Zarr mirror with resumable HTTP byte range requests."""

import argparse
import json
import math
import os
import struct
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from tempfile import gettempdir

import google_crc32c
import numpy as np
import numcodecs.blosc as blosc

from model.preprocess import fetch_gfs as gfs


CHUNK_SHAPE = (105, 121, 121)
RAW_SIZE = math.prod(CHUNK_SHAPE) * 4
INDEX_SIZE = 2 * 6 * 6 * 16 + 4
EPOCH = datetime(2021, 5, 1, tzinfo=timezone.utc)


def read_range(url, byte_range, size):
    response = gfs.request("GET", url, headers={"Range": byte_range})
    if response.status_code != 206 or len(response.content) != size:
        raise ValueError(f"Invalid byte range response for {url}")
    return response.content


def point_series(job):
    """Return (variable, initialization index, 25 source values) for one point."""
    name, run_index, start_lead, lat_index, lon_index = job
    lat_outer, lat = divmod(lat_index, 726)
    lon_outer, lon = divmod(lon_index, 726)
    lat_inner, lat = divmod(lat, 121)
    lon_inner, lon = divmod(lon, 121)
    lead_outer, lead = divmod(start_lead, 210)
    lead_inner, lead = divmod(lead, 105)
    if lead + 25 > 105:
        raise ValueError("Forecast selection crosses a source chunk")
    url = f"{gfs.ARCHIVE}/{name}/c/{run_index}/{lead_outer}/{lat_outer}/{lon_outer}"
    footer = read_range(url, f"bytes=-{INDEX_SIZE}", INDEX_SIZE)
    if google_crc32c.value(footer[:-4]) != struct.unpack_from("<I", footer, INDEX_SIZE - 4)[0]:
        raise ValueError(f"Source shard checksum mismatch for {url}")
    chunk_position = (lead_inner * 6 + lat_inner) * 6 + lon_inner
    offset, total = struct.unpack_from("<QQ", footer, chunk_position * 16)
    if offset == (1 << 64) - 1 or not 16 <= total <= RAW_SIZE + 16:
        raise ValueError(f"Missing source chunk for {name}, run {run_index}")
    end_bytes = ((lead + 24) * 121 * 121 + lat * 121 + lon + 1) * 4
    initial_size = min(total, max(4096, round(total * end_bytes / RAW_SIZE * 1.22) + 4096))
    chunk = read_range(url, f"bytes={offset}-{offset + initial_size - 1}", initial_size)
    capacity = total
    decoded_size = RAW_SIZE
    if initial_size < total:
        version, codec, flags, typesize, nbytes, blocksize, cbytes = struct.unpack_from("<4B3I", chunk)
        if cbytes != total or nbytes != RAW_SIZE:
            raise ValueError(f"Invalid Blosc dimensions for {name}, run {run_index}")
        if version == 2 and flags & 0x0E == 0 and blocksize > 0:
            blocks = math.ceil(nbytes / blocksize)
            needed = math.ceil(end_bytes / blocksize)
            if needed < blocks and len(chunk) >= 16 + 4 * blocks:
                starts = struct.unpack_from(f"<{blocks}I", chunk, 16)
                if starts[0] >= 16 + 4 * blocks and tuple(sorted(set(starts))) == starts and starts[-1] < total:
                    capacity = starts[needed]
                    decoded_size = needed * blocksize
                    header = struct.pack("<4B3I", version, codec, flags, typesize, decoded_size, blocksize, capacity)
                    chunk = header + chunk[16:]
    if len(chunk) < capacity:
        chunk += read_range(url, f"bytes={offset + len(chunk)}-{offset + capacity - 1}", capacity - len(chunk))
    decoded = blosc.decompress(chunk[:capacity])
    if len(decoded) != decoded_size:
        raise ValueError(f"Unexpected decoded size for {name}, run {run_index}")
    positions = np.arange(lead, lead + 25) * 121 * 121 + lat * 121 + lon
    values = np.frombuffer(decoded, dtype="<f4")[positions].tolist()
    return name, run_index, values


def validate_metadata():
    def read_metadata(name):
        return gfs.request("GET", f"{gfs.ARCHIVE}/{name}/zarr.json").json()

    with ThreadPoolExecutor(max_workers=8) as pool:
        metadata_by_name = dict(zip(gfs.VARIABLES, pool.map(read_metadata, gfs.VARIABLES)))
    for name, units in gfs.VARIABLES.items():
        metadata = metadata_by_name[name]
        assert metadata["data_type"] == "float32"
        assert metadata["attributes"]["units"] == units
        assert metadata["chunk_grid"]["configuration"]["chunk_shape"] == [1, 210, 726, 726]
        config = metadata["codecs"][0]["configuration"]
        assert config["chunk_shape"] == [1, 105, 121, 121]
        assert config["index_location"] == "end"
        assert config["codecs"][1]["name"] == "blosc"


def run_index(run):
    hours = (run - EPOCH).total_seconds() / 3600
    if hours < 0 or hours % 6:
        raise ValueError(f"Run is outside the mirror time domain: {run}")
    return int(hours / 6)


def collect(runs, cache_dir, workers=32):
    cache_dir.mkdir(parents=True, exist_ok=True)
    validate_metadata()
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for pos in range(0, len(runs), 12):
            batch = runs[pos:pos + 12]
            data_by_run = {}
            jobs = []
            for run, _, lead, _ in batch:
                index = run_index(run)
                path = cache_dir / f"{gfs.VERSION}_235_1206_f{lead:03d}_run{index}.json"
                if path.exists():
                    data_by_run[index] = json.loads(path.read_text(encoding="utf-8"))
                else:
                    jobs.extend((name, index, lead, 235, 1206) for name in gfs.VARIABLES)
            for name, index, values in pool.map(point_series, jobs):
                data_by_run.setdefault(index, {})[name] = values
            for run_info in batch:
                run, _, lead, _ = run_info
                index = run_index(run)
                data = data_by_run[index]
                if set(data) != set(gfs.VARIABLES):
                    raise ValueError(f"Incomplete source variables for {run}")
                standardized = list(gfs.standardize(run_info, data, 31.25, 121.5))
                rows.extend(standardized)
                path = cache_dir / f"{gfs.VERSION}_235_1206_f{lead:03d}_run{index}.json"
                if not path.exists():
                    temporary = path.with_suffix(f".{os.getpid()}.tmp")
                    temporary.write_text(json.dumps(data), encoding="utf-8")
                    temporary.replace(path)
            print(f"Downloaded {len(rows)} / {len(runs) * 24} hours", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--cache-dir", type=Path, default=Path(__file__).resolve().parents[2] / "data_raw" / "weather" / "gfs" / "points")
    args = parser.parse_args()
    days = [args.start + timedelta(days=i) for i in range((args.end - args.start).days)]
    if not days:
        parser.error("--end must follow --start")
    with ThreadPoolExecutor(max_workers=12) as pool:
        runs = list(pool.map(gfs.select_run, days))
    print(f"Verified release cutoff for {len(runs)} days", flush=True)
    rows = collect(runs, args.cache_dir, args.workers)
    rows.sort(key=lambda row: row["valid_time_utc"])
    first = datetime.combine(args.start, datetime.min.time(), gfs.BJT)
    assert [row["valid_time_utc"] for row in rows] == [
        gfs.iso_utc(first + i * gfs.HOUR) for i in range(len(days) * 24)
    ]
    gfs.write_csv(args.output, rows)
    print(f"Saved {len(rows)} verified hours to {args.output}", flush=True)


if __name__ == "__main__":
    main()
