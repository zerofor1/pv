"""Simulate hourly AC photovoltaic output for a fixed-tilt array."""

import argparse
import calendar
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from model.configuration import configuration_id, read_config
BJT = timezone(timedelta(hours=8))
HOUR = timedelta(hours=1)
INPUT_FIELDS = (
    "valid_time_utc", "interval_end_utc", "ghi_w_m2", "air_temp_c", "wind_speed_m_s",
)
OUTPUT_FIELDS = (
    "solar_elevation_deg", "solar_azimuth_deg", "daylight_fraction", "ghi_used_w_m2",
    "dni_w_m2", "dhi_w_m2", "poa_direct_w_m2", "poa_sky_diffuse_w_m2",
    "poa_ground_diffuse_w_m2", "poa_global_w_m2", "module_temp_c", "pv_dc_gross_kw",
    "pv_dc_kw", "pv_inverter_clipping_kw", "pv_ac_kw", "pv_energy_kwh",
    "pv_reference_dc_kw", "pv_temperature_reduction_kw", "pv_iam_loss_kw",
    "pv_soiling_loss_kw", "pv_mismatch_loss_kw", "pv_dc_wiring_loss_kw",
    "pv_legacy_dc_loss_kw", "pv_inverter_conversion_loss_kw", "pv_ac_wiring_loss_kw",
    "pv_inverter_ac_kw", "module_area_m2", "projected_module_area_m2",
    "pv_config_id",
)


@dataclass(frozen=True)
class Site:
    latitude: float
    longitude: float
    tilt_deg: float
    azimuth_deg: float
    dc_capacity_kwp: float
    ac_capacity_kw: float
    gamma_per_c: float
    albedo: float
    inverter_nominal_efficiency: float
    dc_loss_fraction: float
    faiman_u0: float
    faiman_u1: float
    wind_speed_multiplier: float
    module_count: int = 0
    module_power_w: float = 0
    module_length_m: float = 0
    module_width_m: float = 0
    iam_b0: float = 0
    soiling_loss_fraction: float = 0
    mismatch_loss_fraction: float = 0
    dc_wiring_loss_fraction: float = 0
    ac_wiring_loss_fraction: float = 0

    def __post_init__(self):
        bounds = {
            "latitude": (-90, 90), "longitude": (-180, 180),
            "tilt_deg": (0, 90), "azimuth_deg": (0, 360), "albedo": (0, 1),
            "inverter_nominal_efficiency": (0, 1), "dc_loss_fraction": (0, 1),
            "iam_b0": (0, 1), "soiling_loss_fraction": (0, 1),
            "mismatch_loss_fraction": (0, 1), "dc_wiring_loss_fraction": (0, 1),
            "ac_wiring_loss_fraction": (0, 1),
        }
        for field, value in vars(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{field}: expected a finite number")
            if field in bounds:
                low, high = bounds[field]
                if not low <= value <= high:
                    raise ValueError(f"{field}: expected a value in [{low}, {high}]")
        for field in ("dc_capacity_kwp", "ac_capacity_kw", "inverter_nominal_efficiency",
                      "faiman_u0", "wind_speed_multiplier"):
            if getattr(self, field) <= 0:
                raise ValueError(f"{field}: expected a positive value")
        if self.faiman_u1 < 0:
            raise ValueError("faiman_u1: expected a nonnegative value")
        module_values = (self.module_count, self.module_power_w, self.module_length_m, self.module_width_m)
        if any(module_values):
            if not all(value > 0 for value in module_values) or int(self.module_count) != self.module_count:
                raise ValueError("Module count and dimensions must be positive, with an integer count")
            if not math.isclose(self.module_count * self.module_power_w / 1000, self.dc_capacity_kwp):
                raise ValueError("Module count times rated power must equal DC capacity")


def solar_position(time, latitude, longitude):
    """NOAA fractional-year equations, with azimuth clockwise from north."""
    # Use local solar clock time so the fractional year is continuous at the station.
    solar_clock = time.astimezone(timezone.utc) + timedelta(minutes=4 * longitude)
    hour = solar_clock.hour + solar_clock.minute / 60 + solar_clock.second / 3600
    day = solar_clock.timetuple().tm_yday
    angle = math.tau * (day - 1 + (hour - 12) / 24) / (365 + calendar.isleap(solar_clock.year))
    cos1, sin1 = math.cos(angle), math.sin(angle)
    cos2, sin2 = math.cos(2 * angle), math.sin(2 * angle)
    equation_minutes = 229.18 * (0.000075 + 0.001868 * cos1 - 0.032077 * sin1
                               - 0.014615 * cos2 - 0.040849 * sin2)
    declination = (0.006918 - 0.399912 * cos1 + 0.070257 * sin1
                   - 0.006758 * cos2 + 0.000907 * sin2
                   - 0.002697 * math.cos(3 * angle) + 0.00148 * math.sin(3 * angle))
    hour_angle = math.radians((hour * 60 + equation_minutes) / 4 - 180)
    latitude = math.radians(latitude)
    up = (math.sin(latitude) * math.sin(declination)
          + math.cos(latitude) * math.cos(declination) * math.cos(hour_angle))
    east = -math.cos(declination) * math.sin(hour_angle)
    north = (math.cos(latitude) * math.sin(declination)
             - math.sin(latitude) * math.cos(declination) * math.cos(hour_angle))
    elevation = math.asin(max(-1, min(1, up)))
    azimuth = math.atan2(east, north) % math.tau
    return elevation, azimuth


def erbs(ghi, mean_cos_zenith, extraterrestrial_dni):
    # Use pvlib's low-sun thresholds to avoid dividing by a small projection.
    if mean_cos_zenith < math.cos(math.radians(87)):
        return 0.0, ghi
    kt = min(1.0, ghi / (extraterrestrial_dni * max(mean_cos_zenith, 0.065)))
    if kt <= 0.22:
        diffuse_fraction = 1 - 0.09 * kt
    elif kt <= 0.8:
        diffuse_fraction = 0.9511 - 0.1604 * kt + 4.388 * kt**2 - 16.638 * kt**3 + 12.336 * kt**4
    else:
        diffuse_fraction = 0.165
    dhi = diffuse_fraction * ghi
    return (ghi - dhi) / mean_cos_zenith, dhi


def simulate_hour(start, ghi, air_temp, wind_speed, site):
    """Return hourly mean irradiance and power for [start, start + 1 hour)."""
    middle = start + HOUR / 2
    elevation, azimuth = solar_position(middle, site.latitude, site.longitude)
    tilt = math.radians(site.tilt_deg)
    panel_azimuth = math.radians(site.azimuth_deg)
    cos_tilt = math.cos(tilt)
    horizontal, panel, effective_panel = [], [], []
    # Sample the hour to handle sunrise/sunset and changing beam projection.
    for minute in range(5, 60, 10):
        sun_elevation, sun_azimuth = solar_position(
            start + timedelta(minutes=minute), site.latitude, site.longitude,
        )
        if sun_elevation > 0:
            up = math.sin(sun_elevation)
            horizontal.append(up)
            incidence = (up * cos_tilt + math.cos(sun_elevation) * math.sin(tilt)
                         * math.cos(sun_azimuth - panel_azimuth))
            panel.append(max(0.0, incidence))
            iam = max(0.0, 1 - site.iam_b0 * (1 / max(incidence, 1e-9) - 1))
            effective_panel.append(max(0.0, incidence) * iam)
    if not horizontal:
        ghi = 0.0
    # Erbs uses an hour's extraterrestrial horizontal energy to determine clearness.
    day = (middle.astimezone(timezone.utc) + timedelta(minutes=4 * site.longitude)).timetuple().tm_yday
    extraterrestrial_dni = 1367 * (1 + 0.033 * math.cos(math.tau * day / 365))
    dni, dhi = erbs(ghi, sum(horizontal) / 6, extraterrestrial_dni)
    poa_direct = dni * sum(panel) / 6
    poa_sky = dhi * (1 + cos_tilt) / 2
    poa_ground = ghi * site.albedo * (1 - cos_tilt) / 2
    poa = poa_direct + poa_sky + poa_ground

    module_temp = air_temp + poa / (site.faiman_u0 + site.faiman_u1 * wind_speed * site.wind_speed_multiplier)
    temp_factor = max(0.0, 1 + site.gamma_per_c * (module_temp - 25))
    reference_dc = site.dc_capacity_kwp * poa / 1000
    dc_gross = reference_dc * temp_factor
    effective_poa = dni * sum(effective_panel) / 6 + poa_sky + poa_ground
    after_iam = site.dc_capacity_kwp * effective_poa / 1000 * temp_factor
    after_soiling = after_iam * (1 - site.soiling_loss_fraction)
    after_mismatch = after_soiling * (1 - site.mismatch_loss_fraction)
    after_wiring = after_mismatch * (1 - site.dc_wiring_loss_fraction)
    dc = after_wiring * (1 - site.dc_loss_fraction)
    inverter_dc_rating = site.ac_capacity_kw / site.inverter_nominal_efficiency
    load = dc / inverter_dc_rating
    # Multiply the efficiency curve by DC power algebraically to avoid division at zero load.
    ac_unclipped = max(0.0, site.ac_capacity_kw / 0.9637 * (-0.0162 * load**2 + 0.9858 * load - 0.0059))
    inverter_ac = min(site.ac_capacity_kw, ac_unclipped)
    ac = inverter_ac * (1 - site.ac_wiring_loss_fraction)
    area = site.module_count * site.module_length_m * site.module_width_m
    return {
        "solar_elevation_deg": math.degrees(elevation), "solar_azimuth_deg": math.degrees(azimuth),
        "daylight_fraction": len(horizontal) / 6, "ghi_used_w_m2": ghi,
        "dni_w_m2": dni, "dhi_w_m2": dhi, "poa_direct_w_m2": poa_direct,
        "poa_sky_diffuse_w_m2": poa_sky, "poa_ground_diffuse_w_m2": poa_ground,
        "poa_global_w_m2": poa, "module_temp_c": module_temp, "pv_dc_gross_kw": dc_gross,
        "pv_dc_kw": dc, "pv_inverter_clipping_kw": ac_unclipped - inverter_ac,
        "pv_ac_kw": ac, "pv_energy_kwh": ac * (HOUR / timedelta(hours=1)),
        "pv_reference_dc_kw": reference_dc, "pv_temperature_reduction_kw": reference_dc - dc_gross,
        "pv_iam_loss_kw": dc_gross - after_iam, "pv_soiling_loss_kw": after_iam - after_soiling,
        "pv_mismatch_loss_kw": after_soiling - after_mismatch,
        "pv_dc_wiring_loss_kw": after_mismatch - after_wiring,
        "pv_legacy_dc_loss_kw": after_wiring - dc,
        "pv_inverter_conversion_loss_kw": dc - ac_unclipped,
        "pv_ac_wiring_loss_kw": inverter_ac - ac, "pv_inverter_ac_kw": inverter_ac,
        "module_area_m2": area, "projected_module_area_m2": area * cos_tilt,
    }


def simulate_rows(rows, site):
    seen = set()
    for line, row in enumerate(rows, start=2):
        try:
            if None in row or None in row.values():
                raise ValueError("row has a different number of columns than the header")
            start, end = (datetime.fromisoformat(row[field].replace("Z", "+00:00"))
                          for field in INPUT_FIELDS[:2])
            if start.utcoffset() is None or end.utcoffset() is None or end - start != HOUR:
                raise ValueError("expected offset-aware timestamps spanning exactly one hour")
            key = (row.get("site_id"), start)
            if key in seen:
                raise ValueError("duplicate site and start time")
            if "site_latitude" in row or "site_longitude" in row:
                for field in ("latitude", "longitude"):
                    if not math.isclose(float(row["site_" + field]), getattr(site, field), abs_tol=1e-5):
                        raise ValueError(f"site_{field} differs from the configured site")
            ghi, air_temp, wind_speed = (float(row[field]) for field in INPUT_FIELDS[2:])
            if not all(map(math.isfinite, (ghi, air_temp, wind_speed))) or ghi < 0 or wind_speed < 0:
                raise ValueError("weather must be finite, with nonnegative GHI and wind speed")
            results = simulate_hour(start, ghi, air_temp, wind_speed, site)
            if not all(map(math.isfinite, results.values())):
                raise ValueError("non-finite simulation result")
        except (KeyError, AttributeError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"weather CSV line {line}: {exc}") from exc
        seen.add(key)
        yield {**row, **{field: round(value, 6) for field, value in results.items()},
               "pv_config_id": configuration_id(vars(site))}


def demo_weather(site):
    first = datetime(2024, 6, 18, tzinfo=BJT).astimezone(timezone.utc)
    for i in range(48):
        start = first + i * HOUR
        end = start + HOUR
        mid = start + HOUR / 2
        elevation, _ = solar_position(mid, site.latitude, site.longitude)
        sun = max(0.0, math.sin(elevation))
        cloud_factor = 1.0 if i < 24 else (0.30 if 11 <= i % 24 <= 15 else 0.60)
        yield {
            "site_id": "demo", "source": "synthetic_demo",
            "valid_time_utc": start.isoformat(), "interval_end_utc": end.isoformat(),
            "interval_start_bjt": start.astimezone(BJT).isoformat(),
            "interval_end_bjt": end.astimezone(BJT).isoformat(),
            "site_latitude": site.latitude, "site_longitude": site.longitude,
            "ghi_w_m2": round(1050 * sun**1.25 * cloud_factor, 3),
            "air_temp_c": round(27 + 6 * sun, 3), "wind_speed_m_s": 2.0,
        }


def write_results(output, rows):
    rows = iter(rows)
    first = next(rows, None)
    if first is None:
        raise ValueError("weather CSV contains no data rows")
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8-sig", newline="", dir=output.parent,
                            suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            writer = csv.DictWriter(handle, fieldnames=list(first))
            writer.writeheader()
            writer.writerow(first)
            writer.writerows(rows)
            handle.close()
            temporary.replace(output)
        finally:
            handle.close()
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--weather", type=Path, help="Normalized hourly weather CSV")
    inputs.add_argument("--demo", action="store_true", help="Generate two days of synthetic weather")
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.resolve() in {path.resolve() for path in (args.config, args.weather) if path}:
            raise ValueError("output must differ from the input files")
        site = Site(**read_config(args.config, "pv"))
        if args.demo:
            write_results(args.output, simulate_rows(demo_weather(site), site))
        else:
            with args.weather.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                fields = reader.fieldnames or []
                missing = set(INPUT_FIELDS) - set(fields)
                if missing:
                    raise ValueError(f"missing weather columns: {', '.join(sorted(missing))}")
                if len(fields) != len(set(fields)) or set(fields).intersection(OUTPUT_FIELDS):
                    raise ValueError("weather header contains duplicate or reserved output columns")
                write_results(args.output, simulate_rows(reader, site))
    except (OSError, ValueError, TypeError, csv.Error) as exc:
        parser.error(str(exc))
    print(f"Saved hourly PV simulation to {args.output}")


if __name__ == "__main__":
    main()
