"""Sample whole Shanghai station days and evaluate against an independent hourly GMM."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits

from .fragment_load import CONFIG, ROOT, read_observed_days, split_day_indices


REPORTS = ROOT / "test/reports/load"
LEGACY_GMM = ROOT / "项目材料/历史备份/旧GMM"
METHODS = ("whole_day", "hourly_gmm")
CONDITIONS = ("weekday", "weekend")
METRICS = ("daily_energy", "daily_peak", "hourly_ramp", "typical_curve")
PROFILE_QUANTILES = {"p10": .10, "p50": .50, "p90": .90}


def day_types(days):
    """Use weekdays Monday–Friday and weekends Saturday–Sunday in local time."""
    return np.where(days.day_of_week < 5, "weekday", "weekend")


def sample_whole_days(training_days, training_kw, target_days, draws, rng):
    """Return (draw, target, hour) kW and (draw, target) donor dates."""
    training_types = day_types(training_days)
    target_types = day_types(target_days)
    indices = np.empty((draws, len(target_days)), dtype=int)
    for condition in CONDITIONS:
        targets = target_types == condition
        if targets.any():
            candidates = np.flatnonzero(training_types == condition)
            if not len(candidates):
                raise ValueError(f"No {condition} templates in training")
            indices[:, targets] = rng.choice(candidates, size=(draws, targets.sum()))
    return training_kw[indices], np.asarray(training_days.strftime("%Y-%m-%d"), dtype="U10")[indices]


def fit_hourly_gmm(training_kw, component_grid, seed):
    """Refit the legacy station-wide 1-D mixture with training hours only."""
    mean_kw = float(training_kw.mean())
    std_kw = float(training_kw.std()) or 1.
    values = ((training_kw - mean_kw) / std_kw).reshape(-1, 1)
    distinct = len(np.unique(values))
    candidates = []
    best = None
    with threadpool_limits(limits=1):
        for k in component_grid:
            if k > distinct:
                continue
            fit = GaussianMixture(n_components=k, n_init=3, max_iter=500, random_state=seed)
            fit.fit(values)
            bic = float(fit.bic(values))
            candidates.append(dict(k=k, bic=bic, converged=bool(fit.converged_)))
            if fit.converged_ and (best is None or bic < best[0]):
                best = bic, fit
    if best is None:
        raise ValueError("No hourly GMM candidate converged")
    _, model = best
    components = [
        dict(weight=float(weight), mean_kw=float(mean * std_kw + mean_kw),
             std_kw=float(np.sqrt(variance) * std_kw))
        for weight, mean, variance in zip(model.weights_, model.means_.ravel(),
                                          model.covariances_.ravel())
    ]
    return dict(mean_kw=mean_kw, std_kw=std_kw, fit_hours=len(values),
                bic_candidates=candidates, components=components)


def sample_hourly_gmm(fit, shape, rng):
    components = fit["components"]
    ids = rng.choice(len(components), size=shape, p=[c["weight"] for c in components])
    mean = np.array([c["mean_kw"] for c in components])
    std = np.array([c["std_kw"] for c in components])
    raw = rng.normal(mean[ids], std[ids])
    return np.maximum(raw, 0), raw < 0


def statistics(curves):
    """Use all curves, keeping the 23 intraday ramps within each 24-hour day."""
    ramps = np.diff(curves, axis=1)
    samples = {
        "daily_energy_kwh": curves.sum(axis=1),
        "daily_peak_kw": curves.max(axis=1),
        "abs_hourly_ramp_kw": np.abs(ramps).ravel(),
        "signed_hourly_ramp_kw": ramps.ravel(),
    }
    summary = {}
    for name, values in samples.items():
        quantiles = np.quantile(values, [.10, .50, .90, .95], method="inverted_cdf")
        summary[name] = dict(mean=float(values.mean()), std=float(values.std()),
                             **dict(zip(("p10", "p50", "p90", "p95"), quantiles.tolist())))
    summary["typical_curve_kw"] = {
        name: np.quantile(curves, q, axis=0, method="inverted_cdf").tolist()
        for name, q in PROFILE_QUANTILES.items()
    }
    summary["zero_fraction"] = float(np.mean(curves == 0))
    return samples, summary


def score(reference, generated, scale_kw):
    """Equal-weight normalized distribution errors. Smaller loss is better."""
    real, real_summary = reference
    sim, sim_summary = generated
    energy = wasserstein_distance(real["daily_energy_kwh"], sim["daily_energy_kwh"])
    peak = wasserstein_distance(real["daily_peak_kw"], sim["daily_peak_kw"])
    ramp = wasserstein_distance(real["abs_hourly_ramp_kw"], sim["abs_hourly_ramp_kw"])
    ramp_tail = abs(sim_summary["abs_hourly_ramp_kw"]["p95"] -
                    real_summary["abs_hourly_ramp_kw"]["p95"])
    profile = np.array([real_summary["typical_curve_kw"][q] for q in PROFILE_QUANTILES])
    sim_profile = np.array([sim_summary["typical_curve_kw"][q] for q in PROFILE_QUANTILES])
    profile_mae = float(np.abs(profile - sim_profile).mean())
    errors = dict(daily_energy=energy / (24 * scale_kw), daily_peak=peak / scale_kw,
                  hourly_ramp=(ramp + ramp_tail) / (2 * scale_kw),
                  typical_curve=profile_mae / scale_kw)
    return dict(normalized_errors=errors, **grade(errors),
                raw_errors=dict(energy_w1_kwh=energy, peak_w1_kw=peak, ramp_w1_kw=ramp,
                                ramp_p95_abs_error_kw=ramp_tail, typical_curve_mae_kw=profile_mae))


def grade(errors):
    loss = float(np.mean([errors[name] for name in METRICS]))
    return dict(loss=loss, score=100 / (1 + loss))


def summarize_groups(groups, weights):
    errors = {
        name: float(np.average([g["normalized_errors"][name] for g in groups], weights=weights))
        for name in METRICS
    }
    return dict(normalized_errors=errors, **grade(errors))


def compare_station(path, coverage, legacy_path, draws, seed):
    days, curves = read_observed_days(path, coverage)
    keep, validation, test = split_day_indices(days)
    test_indices = np.concatenate(test)
    train_days, target_days = days[keep], days[test_indices]
    train_kw = curves[keep]
    legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
    component_grid = sorted({int(candidate["k"]) for candidate in legacy["bic_candidates"]})
    fit = fit_hourly_gmm(train_kw, component_grid, seed)

    station = path.stem.split("_")[0]
    streams = np.random.SeedSequence([seed, int(station[1:])]).spawn(2)
    whole, donors = sample_whole_days(train_days, train_kw, target_days, draws,
                                    np.random.default_rng(streams[0]))
    hourly, clipped = sample_hourly_gmm(fit, whole.shape, np.random.default_rng(streams[1]))
    actual = curves[test_indices]
    scale_kw = max(float(train_kw.mean()), .01)
    groups = {}
    for condition in CONDITIONS:
        mask = day_types(target_days) == condition
        if not mask.any():
            continue
        reference = statistics(actual[mask])
        methods = {"whole_day": statistics(whole[:, mask].reshape(-1, 24)),
                   "hourly_gmm": statistics(hourly[:, mask].reshape(-1, 24))}
        groups[condition] = dict(
            test_days=int(mask.sum()),
            train_days=int((day_types(train_days) == condition).sum()),
            real=reference[1],
            gmm_negative_draw_fraction=float(clipped[:, mask].mean()),
            methods={name: dict(statistics=metrics[1], **score(reference, metrics, scale_kw))
                     for name, metrics in methods.items()},
        )

    summary = {
        method: summarize_groups([g["methods"][method] for g in groups.values()],
                                 weights=[g["test_days"] for g in groups.values()])
        for method in METHODS
    }
    report = dict(
        cleaned_file=path.relative_to(ROOT).as_posix(),
        cleaned_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        legacy_gmm_file=legacy_path.relative_to(ROOT).as_posix(),
        legacy_gmm_scope=legacy["fit_scope"], gmm_component_grid=component_grid,
        gmm_train_fit=fit, normalization_scale_kw=scale_kw,
        train_dates=train_days.strftime("%Y-%m-%d").tolist(),
        validation_dates=days[np.concatenate(validation)].strftime("%Y-%m-%d").tolist(),
        test_windows=[days[w].strftime("%Y-%m-%d").tolist() for w in test],
        groups=groups, summary=summary,
    )
    arrays = dict(dates=np.asarray(target_days.strftime("%Y-%m-%d"), dtype="U10"),
                  real_kw=actual, whole_day_kw=whole, whole_day_donor_date=donors,
                  hourly_gmm_kw=hourly)
    return report, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=CONFIG["test_draws"],
                        help="generated curves per station and held-out date")
    parser.add_argument("--seed", type=int, default=CONFIG["seed"])
    parser.add_argument("--output", type=Path, default=REPORTS / "daily_baseline")
    args = parser.parse_args()
    if args.draws < 1 or not 0 <= args.seed <= 2**32 - 1:
        parser.error("--draws must be positive and --seed must be a uint32")

    coverage_path = REPORTS / "coverage_audit.json"
    coverage_bytes = coverage_path.read_bytes()
    coverage = json.loads(coverage_bytes)
    report = dict(seed=args.seed, draws_per_date=args.draws, timezone="Asia/Shanghai",
                  conditions=dict(weekday="Monday–Friday", weekend="Saturday–Sunday"),
                  coverage_audit_sha256=hashlib.sha256(coverage_bytes).hexdigest(),
                  transaction_source_sha256=coverage["source_sha256"], stations={})
    arrays = {}
    for station in sorted(coverage["stations"], key=lambda s: int(s[1:])):
        path = ROOT / "data/load/cleaned" / f"{station}_hourly.csv"
        result, scenarios = compare_station(path, coverage, LEGACY_GMM / f"{station}_gmm.json",
                                            args.draws, args.seed)
        report["stations"][station] = result
        arrays.update({f"{station}_{key}": value for key, value in scenarios.items()})
        print(f"{station}: {len(scenarios['dates'])} test days, "
              f"whole day {result['summary']['whole_day']['score']:.2f}, "
              f"hourly GMM {result['summary']['hourly_gmm']['score']:.2f}", flush=True)

    report["macro_average"] = {
        method: summarize_groups([s["summary"][method] for s in report["stations"].values()],
                                 weights=None)
        for method in METHODS
    }
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "scenarios.npz", **arrays)
    (args.output / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
