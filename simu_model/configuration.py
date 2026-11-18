"""Read shared station parameters and legacy per-model JSON files."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "model" / "pv" / "config.json"


def read_config(path=DEFAULT_CONFIG, section=None):
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if "station_config" in data:
        return read_config((path.parent / data["station_config"]).resolve(), data.get("section", section))
    return data[section] if section and section in data else data


def configuration_id(data):
    payload = json.dumps(data, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
