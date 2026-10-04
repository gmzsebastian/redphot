"""Batch-level consistency checks and final light-curve assembly for redphot.

The functions in this module consume the tables produced by the image,
calibration, science-photometry, and difference-photometry stages.  They never
delete input measurements.  Rejected epochs, unstable stars, and isolated
outliers remain present with explicit flags and inclusion decisions.
"""

from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import traceback

import numpy as np
from astropy import units as u
from astropy.stats import sigma_clip
from astropy.table import MaskedColumn, Table, vstack

from .config import (
    get_default_settings,
    merge_settings,
    normalize_instrument_name,
    resolve_settings,
)
from . import progress as _progress


def _finite_float(value, default=None):
    """Return a finite float or ``default``."""

    if value is None or np.ma.is_masked(value):
        return default
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def _row_value(row, name, default=None):
    """Read a possibly masked table or mapping value."""

    if isinstance(row, Mapping):
        value = row.get(name, default)
    elif hasattr(row, "colnames") and name in row.colnames:
        value = row[name]
    else:
        value = default
    return default if np.ma.is_masked(value) else value


def _image_id(record, index=0):
    """Return the persistent image identifier used throughout the pipeline."""

    metadata = record.get("metadata") or {}
    return str(record.get("image_id") or metadata.get("filename") or "image_{:04d}".format(index))


def _append_flag(value, flag):
    """Append one semicolon-delimited flag without duplication."""

    flags = list(filter(None, str(value or "").split(";")))
    if flag and flag not in flags:
        flags.append(flag)
    return ";".join(flags)


def _records_table(records):
    """Convert scalar dictionaries into a masked Astropy table."""

    if not records:
        return Table(masked=True)
    names = list(dict.fromkeys(name for record in records for name in record))
    table = Table(masked=True)
    for name in names:
        values = [record.get(name) for record in records]
        present = [value for value in values if value is not None]
        if present and all(isinstance(value, (bool, np.bool_)) for value in present):
            table[name] = MaskedColumn(
                [False if value is None else bool(value) for value in values],
                mask=[value is None for value in values],
            )
        elif present and all(
            isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, (bool, np.bool_)) for value in present
        ):
            numeric = np.asarray([np.nan if value is None else float(value) for value in values])
            table[name] = MaskedColumn(
                np.where(np.isfinite(numeric), numeric, 0.0), mask=~np.isfinite(numeric)
            )
        else:
            maximum = max([len(str(value)) for value in present] + [1])
            table[name] = MaskedColumn(
                np.asarray(["" if value is None else str(value) for value in values], dtype="U{}".format(maximum)),
                mask=[value is None for value in values],
            )
    return table


def _ensure_column(table, name, values):
    """Replace or add one column while avoiding narrow string dtypes."""

    if name in table.colnames:
        table.remove_column(name)
    table[name] = values


def _copy_measurements(table, default_kind="science"):
    """Copy a measurement table and add missing batch provenance columns."""

    if table is None:
        return Table(masked=True)
    result = Table(table, masked=True, copy=True)
    count = len(result)
    if "image_kind" not in result.colnames:
        result["image_kind"] = np.full(count, default_kind, dtype="U16")
    if "host_light_included" not in result.colnames:
        result["host_light_included"] = np.full(count, default_kind == "science", dtype=bool)
    if "flags" not in result.colnames:
        result["flags"] = np.full(count, "", dtype="U1024")
    else:
        _ensure_column(
            result, "flags",
            np.asarray([str(value) for value in result["flags"]], dtype="U2048"),
        )
    for name in ("telescope", "site", "instrument", "detector"):
        if name not in result.colnames:
            result[name] = np.full(count, "", dtype="U64")
    return result


def collect_batch_measurements(science_measurements, difference_results=None):
    """Combine science and difference rows while retaining their provenance."""

    tables = []
    science = _copy_measurements(science_measurements, "science")
    if len(science):
        tables.append(science)
    if difference_results is not None:
        if isinstance(difference_results, Table):
            difference_tables = [difference_results]
        elif isinstance(difference_results, Mapping):
            difference_tables = [difference_results.get("measurements")]
        else:
            difference_tables = [
                item.get("measurements") if isinstance(item, Mapping) else item
                for item in difference_results
            ]
        for value in difference_tables:
            table = _copy_measurements(value, "difference")
            if len(table):
                table["image_kind"] = np.full(len(table), "difference", dtype="U16")
                table["host_light_included"] = np.zeros(len(table), dtype=bool)
                tables.append(table)
    return vstack(tables, join_type="outer", metadata_conflicts="silent") if tables else Table(masked=True)


def _measurement_magnitude(row):
    """Return the best available magnitude and uncertainty for one row."""

    for name, error_name, source in (
        ("ensemble_corrected_magnitude", "ensemble_corrected_magnitude_uncertainty", "ensemble"),
        ("calibrated_magnitude", "calibrated_magnitude_uncertainty", "calibrated"),
        ("instrumental_magnitude", "instrumental_magnitude_uncertainty", "instrumental"),
    ):
        value = _finite_float(_row_value(row, name))
        if value is not None:
            return value, _finite_float(_row_value(row, error_name)), source
    flux = _finite_float(_row_value(row, "flux"))
    error = _finite_float(_row_value(row, "flux_uncertainty"))
    exposure = _finite_float(_row_value(row, "exposure_time"), 1.0)
    if flux is None or flux <= 0 or exposure is None or exposure <= 0:
        return None, None, None
    magnitude = -2.5 * np.log10(flux / exposure)
    uncertainty = 2.5 / np.log(10.0) * error / flux if error is not None else None
    return float(magnitude), uncertainty, "relative_instrumental"


def _robust_scatter(values):
    """Return the Gaussian-equivalent median absolute deviation."""

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return None
    center = np.median(values)
    scatter = 1.4826 * np.median(np.abs(values - center))
    if not np.isfinite(scatter) or scatter <= 0:
        scatter = np.std(values)
    return float(scatter) if np.isfinite(scatter) else None


def build_comparison_star_light_curves(measurements, settings=None):
    """Build per-star residual light curves and classify comparison stability."""

    if settings is None:
        settings = get_default_settings()
    configured = settings.get("batch_consistency", {})
    table = _copy_measurements(measurements)
    records = []
    groups = {}
    methods = set(configured.get("comparison_methods", []))
    for index, row in enumerate(table):
        source_type = str(_row_value(row, "source_type", ""))
        if source_type not in {"comparison", "calibration"}:
            continue
        if str(_row_value(row, "image_kind", "science")) != "science":
            continue
        method = str(_row_value(row, "method", ""))
        if methods and method not in methods:
            continue
        if not bool(_row_value(row, "valid", True)):
            continue
        magnitude, uncertainty, magnitude_source = _measurement_magnitude(row)
        if magnitude is None:
            continue
        key = (
            str(_row_value(row, "source_id", "")),
            str(_row_value(row, "filter", "")), method,
        )
        record = {
            "measurement_index": index,
            "image_id": str(_row_value(row, "image_id", "")),
            "source_id": key[0], "filter": key[1], "method": key[2],
            "mjd": _finite_float(_row_value(row, "mjd_mid")),
            "telescope": str(_row_value(row, "telescope", "")),
            "site": str(_row_value(row, "site", "")),
            "magnitude": magnitude, "magnitude_uncertainty": uncertainty,
            "magnitude_source": magnitude_source,
        }
        groups.setdefault(key, []).append(len(records))
        records.append(record)
    baselines = {
        key: float(np.median([records[index]["magnitude"] for index in indices]))
        for key, indices in groups.items()
    }
    epoch_groups = {}
    for key, indices in groups.items():
        for index in indices:
            raw = records[index]["magnitude"] - baselines[key]
            records[index]["baseline_magnitude"] = baselines[key]
            records[index]["raw_residual_mag"] = float(raw)
            epoch_key = (
                records[index]["image_id"], records[index]["filter"],
                records[index]["method"],
            )
            epoch_groups.setdefault(epoch_key, []).append(raw)
    common_modes = {
        key: float(np.median(values)) for key, values in epoch_groups.items()
    }
    for record in records:
        epoch_key = (record["image_id"], record["filter"], record["method"])
        common = common_modes.get(epoch_key, 0.0)
        record["common_mode_mag"] = common
        record["residual_mag"] = float(record["raw_residual_mag"] - common)
    # Scatter that an epoch adds to every star (e.g. crowding in very poor
    # seeing) beyond the reported errors. It is added to each star's errors
    # in that epoch, so one bad image does not make every star look variable;
    # a truly variable star still stands out against the others.
    epoch_excess = {}
    epoch_members = {}
    for index, record in enumerate(records):
        epoch_members.setdefault(
            (record["image_id"], record["filter"], record["method"]), []
        ).append(index)
    for epoch_key, members in epoch_members.items():
        residual_values = np.asarray([records[index]["residual_mag"] for index in members])
        error_values = np.asarray([
            records[index]["magnitude_uncertainty"] or np.nan for index in members
        ], dtype=float)
        scatter = _robust_scatter(residual_values) if len(members) >= 5 else None
        typical_error = (
            float(np.nanmedian(error_values)) if np.any(np.isfinite(error_values)) else 0.0
        )
        excess = (
            float(np.sqrt(max(0.0, scatter ** 2 - typical_error ** 2)))
            if scatter is not None else 0.0
        )
        epoch_excess[epoch_key] = excess
        for index in members:
            records[index]["epoch_excess_scatter_mag"] = excess
    minimum = int(configured.get("minimum_comparison_epochs", 3))
    floor = float(configured.get("comparison_star_error_floor_mag", 0.01))
    chi2_warn = float(configured.get("comparison_star_reduced_chi2_warn", 3.0))
    chi2_fail = float(configured.get("comparison_star_reduced_chi2_fail", 10.0))
    rms_fail = float(configured.get("comparison_star_rms_fail_mag", 0.15))
    stability_records = []
    for key, indices in groups.items():
        baseline = baselines[key]
        residuals = np.asarray([records[index]["residual_mag"] for index in indices])
        rms = float(np.sqrt(np.mean(residuals ** 2))) if residuals.size else None
        robust = _robust_scatter(residuals)
        errors = np.asarray([
            records[index]["magnitude_uncertainty"]
            if records[index]["magnitude_uncertainty"] not in {None, 0.0}
            else np.nan for index in indices
        ])
        valid_errors = np.isfinite(errors) & (errors > 0)
        # Photon-noise errors of bright stars are a few mmag, below the
        # flat-field and calibration systematics, so a small error floor (and
        # each epoch's excess scatter) is added before asking whether a star
        # scatters more than expected.
        excess = np.asarray([
            records[index].get("epoch_excess_scatter_mag", 0.0) for index in indices
        ], dtype=float)
        effective_errors = np.sqrt(errors ** 2 + floor ** 2 + excess ** 2)
        reduced_chi2 = (
            float(np.sum((residuals[valid_errors] / effective_errors[valid_errors]) ** 2) / max(1, np.count_nonzero(valid_errors) - 1))
            if np.count_nonzero(valid_errors) >= 2 else None
        )
        status = "PASS"
        reasons = []
        if len(indices) < minimum:
            # Too few epochs to tell: not evidence that the star varies.
            status = "UNTESTED"
            reasons.append("TOO_FEW_EPOCHS")
        elif reduced_chi2 is not None:
            # A star is unstable only when it scatters significantly more than
            # its (floored) errors allow; a large RMS of a faint star whose
            # errors are equally large is just noise. With enough epochs, a
            # star whose excess comes from a single epoch (a measurement
            # problem in that image, e.g. a neighbor in bad seeing) is not
            # called variable: the chi2 without that epoch decides.
            normalized = np.abs(residuals / effective_errors)
            usable = valid_errors & np.isfinite(normalized)
            if reduced_chi2 >= chi2_warn and np.count_nonzero(usable) > minimum:
                worst = int(np.nanargmax(np.where(usable, normalized, -np.inf)))
                keep = usable.copy()
                keep[worst] = False
                without = float(
                    np.sum(normalized[keep] ** 2) / max(1, np.count_nonzero(keep) - 1)
                )
                if without < chi2_warn:
                    reduced_chi2 = without
                    reasons.append("SINGLE_EPOCH_OUTLIER")
            if reduced_chi2 >= chi2_fail:
                status = "FAIL"
                reasons.append("CHI2_HIGH")
            elif reduced_chi2 >= chi2_warn:
                status = "WARN"
                reasons.append("CHI2_WARN")
                if rms is not None and rms >= rms_fail:
                    status = "FAIL"
                    reasons.append("RMS_HIGH")
        elif rms is not None:
            # No usable errors: fall back to the absolute scatter limits.
            if rms >= rms_fail:
                status = "FAIL"
                reasons.append("RMS_HIGH")
            elif rms >= float(configured.get("comparison_star_rms_warn_mag", 0.05)):
                status = "WARN"
                reasons.append("RMS_WARN")
        stable = status == "PASS"
        for record_index in indices:
            records[record_index]["stable_star"] = stable
            records[record_index]["stability_status"] = status
        stability_records.append(
            {
                "source_id": key[0], "filter": key[1], "method": key[2],
                "epoch_count": len(indices), "baseline_magnitude": baseline,
                "rms_mag": rms, "robust_scatter_mag": robust,
                "reduced_chi2": reduced_chi2, "error_floor_mag": floor,
                "status": status, "stable": stable,
                "unstable": status in {"WARN", "FAIL"},
                "reasons": ";".join(reasons),
            }
        )
    light_curves = _records_table(records)
    stability = _records_table(stability_records)
    for table_value in (light_curves, stability):
        for name in table_value.colnames:
            if "magnitude" in name or name.endswith("_mag"):
                table_value[name].unit = u.mag
    return light_curves, stability


def apply_ensemble_corrections(measurements, comparison_light_curves, stability,
                               settings=None):
    """Optionally apply simple robust telescope and epoch magnitude offsets."""

    if settings is None:
        settings = get_default_settings()
    configured = settings.get("batch_consistency", {}).get("ensemble_correction", {})
    output = _copy_measurements(measurements)
    stable_keys = {
        (str(row["source_id"]), str(row["filter"]), str(row["method"]))
        for row in stability if bool(row["stable"])
    }
    comparison_rows = [
        row for row in comparison_light_curves
        if (str(row["source_id"]), str(row["filter"]), str(row["method"])) in stable_keys
    ]
    minimum = int(configured.get("minimum_stars", 3))
    components = set(configured.get("components", []))
    maximum = float(configured.get("maximum_absolute_correction_mag", 0.50))
    telescope_offsets = {}
    if configured.get("enabled", False) and "telescope" in components:
        groups = {}
        for row in comparison_rows:
            key = (str(row["filter"]), str(row["method"]), str(row["telescope"]))
            groups.setdefault(key, []).append(float(row["raw_residual_mag"]))
        for key, values in groups.items():
            if len(values) >= minimum:
                telescope_offsets[key] = float(np.median(values))
    epoch_offsets = {}
    if configured.get("enabled", False) and "epoch" in components:
        groups = {}
        for row in comparison_rows:
            telescope_key = (str(row["filter"]), str(row["method"]), str(row["telescope"]))
            residual = float(row["raw_residual_mag"]) - telescope_offsets.get(telescope_key, 0.0)
            key = (str(row["image_id"]), str(row["filter"]), str(row["method"]))
            groups.setdefault(key, []).append(residual)
        sigma = float(configured.get("sigma_clip", 3.0))
        iterations = int(configured.get("maximum_iterations", 5))
        for key, values in groups.items():
            if len(values) < minimum:
                continue
            clipped = sigma_clip(values, sigma=sigma, maxiters=iterations, masked=True)
            kept = np.asarray(clipped.compressed(), dtype=float)
            if kept.size >= minimum:
                epoch_offsets[key] = float(np.median(kept))
    corrections = []
    correction_values = []
    correction_errors = []
    corrected_magnitudes = []
    corrected_errors = []
    corrected_fluxes = []
    for row in output:
        telescope_key = (
            str(_row_value(row, "filter", "")), str(_row_value(row, "method", "")),
            str(_row_value(row, "telescope", "")),
        )
        epoch_key = (
            str(_row_value(row, "image_id", "")), str(_row_value(row, "filter", "")),
            str(_row_value(row, "method", "")),
        )
        raw_offset = telescope_offsets.get(telescope_key, 0.0) + epoch_offsets.get(epoch_key, 0.0)
        correction = float(np.clip(-raw_offset, -maximum, maximum)) if configured.get("enabled", False) else 0.0
        magnitude, magnitude_error, _ = _measurement_magnitude(row)
        flux = _finite_float(_row_value(row, "flux"))
        corrected = magnitude + correction if magnitude is not None else None
        corrected_flux = flux * 10 ** (-0.4 * correction) if flux is not None else None
        correction_values.append(correction)
        correction_errors.append(None)
        corrected_magnitudes.append(corrected)
        corrected_errors.append(magnitude_error)
        corrected_fluxes.append(corrected_flux)
        corrections.append(
            {
                "image_id": epoch_key[0], "filter": epoch_key[1], "method": epoch_key[2],
                "telescope": telescope_key[2],
                "telescope_offset_mag": telescope_offsets.get(telescope_key),
                "epoch_offset_mag": epoch_offsets.get(epoch_key),
                "applied_correction_mag": correction,
                "enabled": bool(configured.get("enabled", False)),
            }
        )
    for name, values in (
        ("ensemble_correction_mag", correction_values),
        ("ensemble_correction_uncertainty_mag", correction_errors),
        ("ensemble_corrected_magnitude", corrected_magnitudes),
        ("ensemble_corrected_magnitude_uncertainty", corrected_errors),
        ("ensemble_corrected_flux", corrected_fluxes),
    ):
        numeric = np.asarray([np.nan if value is None else float(value) for value in values])
        output[name] = MaskedColumn(
            np.where(np.isfinite(numeric), numeric, 0.0), mask=~np.isfinite(numeric)
        )
    for name in (
        "ensemble_correction_mag", "ensemble_correction_uncertainty_mag",
        "ensemble_corrected_magnitude", "ensemble_corrected_magnitude_uncertainty",
    ):
        output[name].unit = u.mag
    if "flux" in output.colnames:
        output["ensemble_corrected_flux"].unit = getattr(output["flux"], "unit", None)
    return output, _records_table(corrections)


def _zeropoint_by_image(zeropoints):
    """Return median zeropoint and scatter per image."""

    groups = {}
    for row in ([] if zeropoints is None else zeropoints):
        value = _finite_float(_row_value(row, "zeropoint_mag"))
        if value is None:
            continue
        groups.setdefault(str(_row_value(row, "image_id", "")), []).append(value)
    return {
        key: (float(np.median(values)), _robust_scatter(values))
        for key, values in groups.items()
    }


def _depth_by_image(limits):
    """Return the deepest available 5-sigma magnitude limit per image."""

    result = {}
    for row in ([] if limits is None else limits):
        image_id = str(_row_value(row, "image_id", ""))
        candidates = [
            _finite_float(_row_value(row, name)) for name in getattr(row, "colnames", [])
            if "5sigma" in name and name.endswith("_mag")
        ]
        candidates = [value for value in candidates if value is not None]
        if candidates:
            result[image_id] = max(result.get(image_id, -np.inf), max(candidates))
    return result


def build_epoch_metrics(image_records, zeropoints=None, limits=None, settings=None):
    """Assemble and robustly flag zeropoint, depth, seeing, background, and WCS trends."""

    if settings is None:
        settings = get_default_settings()
    configured = settings.get("batch_consistency", {})
    zp_lookup = _zeropoint_by_image(zeropoints)
    depth_lookup = _depth_by_image(limits)
    records = []
    for index, record in enumerate(image_records or []):
        image_id = _image_id(record, index)
        metadata = record.get("metadata") or {}
        quality = record.get("quality") or {}
        usability = record.get("usability") or record.get("decision") or {}
        astrometry = record.get("astrometry") or {}
        depths = usability.get("global_depths_mag") or {}
        zeropoint, zeropoint_scatter = zp_lookup.get(image_id, (None, None))
        records.append(
            {
                "image_id": image_id,
                "mjd": _finite_float(metadata.get("mjd_mid"), _finite_float(metadata.get("mjd"))),
                "telescope": str(metadata.get("telescope") or ""),
                "site": str(metadata.get("site") or ""),
                "filter": str(metadata.get("filter") or ""),
                "zeropoint_mag": zeropoint,
                "zeropoint_scatter_mag": zeropoint_scatter,
                "depth_5sigma_mag": depth_lookup.get(
                    image_id, _finite_float(depths.get("5sigma"))
                ),
                "seeing_fwhm_arcsec": _finite_float(
                    quality.get("fwhm_arcsec"), _finite_float(metadata.get("fwhm_arcsec"))
                ),
                "background": _finite_float(quality.get("background")),
                "background_rms": _finite_float(quality.get("background_rms")),
                "wcs_rms_arcsec": _finite_float(
                    astrometry.get("refined_rms_arcsec"),
                    _finite_float(astrometry.get("rms_arcsec")),
                ),
                "input_status": str(usability.get("status") or quality.get("status") or "PASS"),
                "status": "PASS", "flags": "",
            }
        )
    metrics = (
        "zeropoint_mag", "depth_5sigma_mag", "seeing_fwhm_arcsec",
        "background", "background_rms", "wcs_rms_arcsec",
    )
    warn_sigma = float(configured.get("metric_outlier_warn_sigma", 3.5))
    fail_value = configured.get("metric_outlier_fail_sigma")
    fail_sigma = float(fail_value) if fail_value is not None else np.inf
    minimum_images = int(configured.get("metric_outlier_minimum_images", 4))
    # The robust scatter of a handful of epochs can be tiny (three images with
    # the same seeing), which would turn ordinary differences into huge
    # "sigmas". Each metric's scatter is therefore never taken below a floor
    # that represents a difference worth flagging (fractions of the median for
    # seeing and sky, magnitudes or arcsec for the others).
    floors = {
        "zeropoint_mag": ("absolute", 0.05),
        "depth_5sigma_mag": ("absolute", 0.2),
        "seeing_fwhm_arcsec": ("fraction", 0.15),
        "background": ("fraction", 0.25),
        "background_rms": ("fraction", 0.15),
        "wcs_rms_arcsec": ("absolute", 0.1),
    }
    floors.update({
        key: tuple(value) for key, value in
        (configured.get("metric_outlier_scatter_floor") or {}).items()
    })
    for metric in metrics:
        by_filter = {}
        for index, record in enumerate(records):
            value = record.get(metric)
            if value is not None:
                by_filter.setdefault(record["filter"], []).append((index, value))
        for group in by_filter.values():
            if len(group) < minimum_images:
                continue
            values = np.asarray([value for _, value in group])
            center = float(np.median(values))
            scatter = _robust_scatter(values) or 0.0
            kind, amount = floors.get(metric, ("absolute", 0.0))
            floor = float(amount) * (abs(center) if kind == "fraction" else 1.0)
            scatter = max(float(scatter), floor)
            if scatter <= 0:
                continue
            for index, value in group:
                deviation = abs(value - center) / scatter
                records[index]["{}_deviation_sigma".format(metric)] = float(deviation)
                if deviation >= fail_sigma:
                    records[index]["status"] = "FAIL"
                    records[index]["flags"] = _append_flag(records[index]["flags"], "{}_OUTLIER".format(metric.upper()))
                elif deviation >= warn_sigma and records[index]["status"] != "FAIL":
                    records[index]["status"] = "WARN"
                    records[index]["flags"] = _append_flag(records[index]["flags"], "{}_OUTLIER".format(metric.upper()))
    for record in records:
        if record["input_status"] == "FAIL":
            record["status"] = "FAIL"
            record["flags"] = _append_flag(record["flags"], "UPSTREAM_IMAGE_FAIL")
        elif record["input_status"] == "WARN" and record["status"] == "PASS":
            record["status"] = "WARN"
            record["flags"] = _append_flag(record["flags"], "UPSTREAM_IMAGE_WARN")
    table = _records_table(records)
    for name in table.colnames:
        if name.endswith("_mag"):
            table[name].unit = u.mag
        elif name.endswith("_arcsec"):
            table[name].unit = u.arcsec
        elif name == "mjd":
            table[name].unit = u.day
    return table


def summarize_problem_groups(epoch_metrics, settings=None):
    """Summarize problematic telescopes, sites, and filters without exclusion."""

    if settings is None:
        settings = get_default_settings()
    configured = settings.get("batch_consistency", {})
    minimum = int(configured.get("minimum_group_images", 2))
    warn_fraction = float(configured.get("problem_group_warn_fraction", 0.25))
    fail_fraction = float(configured.get("problem_group_fail_fraction", 0.50))
    records = []
    for group_type in ("telescope", "site", "filter"):
        groups = {}
        for row in epoch_metrics:
            groups.setdefault(str(row[group_type]), []).append(row)
        for value, rows in groups.items():
            count = len(rows)
            warn_count = sum(str(row["status"]) == "WARN" for row in rows)
            fail_count = sum(str(row["status"]) == "FAIL" for row in rows)
            problem_fraction = (warn_count + fail_count) / count if count else 0.0
            status = "PASS"
            if count >= minimum and problem_fraction >= fail_fraction:
                status = "FAIL"
            elif count >= minimum and problem_fraction >= warn_fraction:
                status = "WARN"
            record = {
                "group_type": group_type, "group_value": value,
                "image_count": count, "warn_count": warn_count,
                "fail_count": fail_count, "problem_fraction": problem_fraction,
                "status": status,
            }
            for metric in (
                "zeropoint_mag", "depth_5sigma_mag", "seeing_fwhm_arcsec",
                "background", "background_rms", "wcs_rms_arcsec",
            ):
                values = [_finite_float(_row_value(row, metric)) for row in rows]
                values = [item for item in values if item is not None]
                record["median_{}".format(metric)] = (
                    float(np.median(values)) if values else None
                )
            records.append(record)
    return _records_table(records)


def compare_photometry_methods(measurements, settings=None):
    """Compare aperture, PSF, science, and difference target measurements."""

    if settings is None:
        settings = get_default_settings()
    configured = settings.get("batch_consistency", {})
    table = _copy_measurements(measurements)
    rows = []
    groups = {}
    target_lookup = {}
    for index, row in enumerate(table):
        if str(_row_value(row, "source_type", "")) != "target":
            continue
        key = (
            str(row["image_id"]), str(row["filter"]),
            str(row["image_kind"]),
        )
        groups.setdefault(key, []).append((index, row))
        target_lookup[(key[0], key[1], "{}:{}".format(key[2], row["method"]))] = row
    preference = configured.get("preferred_order", [])
    floor = float(configured.get("method_disagreement_floor_mag", 0.05))
    warn = float(configured.get("method_disagreement_warn_sigma", 3.0))
    fail = float(configured.get("method_disagreement_fail_sigma", 5.0))
    for key, values in groups.items():
        available = {}
        for index, row in values:
            available["{}:{}".format(row["image_kind"], row["method"])] = (index, row)
        kind_preference = [name for name in preference if name.startswith(key[2] + ":")]
        reference_key = next((name for name in kind_preference if name in available), None)
        reference_mag = None
        reference_error = None
        if reference_key is not None:
            reference_mag, reference_error, _ = _measurement_magnitude(available[reference_key][1])
        group_magnitudes = [
            _measurement_magnitude(row)[0] for _, row in values
        ]
        group_magnitudes = [value for value in group_magnitudes if value is not None]
        comparison_center = (
            float(np.median(group_magnitudes)) if group_magnitudes else None
        )
        for index, row in values:
            magnitude, error, source = _measurement_magnitude(row)
            delta = (
                magnitude - comparison_center
                if magnitude is not None and comparison_center is not None else None
            )
            uncertainty = float(np.sqrt(
                (error or 0.0) ** 2 + floor ** 2
            ))
            significance = delta / uncertainty if delta is not None and uncertainty > 0 else None
            status = "PASS"
            if significance is not None and abs(significance) >= fail:
                status = "FAIL"
            elif significance is not None and abs(significance) >= warn:
                status = "WARN"
            counterpart_key = "{}:{}".format(
                "difference" if key[2] == "science" else "science", row["method"]
            )
            counterpart_mag = None
            counterpart = target_lookup.get((key[0], key[1], counterpart_key))
            if counterpart is not None:
                counterpart_mag, _, _ = _measurement_magnitude(
                    counterpart
                )
            rows.append(
                {
                    "measurement_index": index, "image_id": key[0], "filter": key[1],
                    "image_kind": str(row["image_kind"]), "method": str(row["method"]),
                    "magnitude": magnitude, "magnitude_uncertainty": error,
                    "magnitude_source": source, "reference_result": reference_key,
                    "reference_magnitude": reference_mag,
                    "within_kind_median_magnitude": comparison_center,
                    "delta_magnitude": delta, "disagreement_sigma": significance,
                    "counterpart_magnitude": counterpart_mag,
                    "science_minus_difference_magnitude": (
                        magnitude - counterpart_mag
                        if magnitude is not None and counterpart_mag is not None
                        and key[2] == "science"
                        else counterpart_mag - magnitude
                        if magnitude is not None and counterpart_mag is not None
                        else None
                    ),
                    "status": status,
                    "outlier": status in {"WARN", "FAIL"},
                }
            )
    result = _records_table(rows)
    for name in result.colnames:
        if "magnitude" in name:
            result[name].unit = u.mag
    return result


def _epoch_status_lookup(epoch_metrics):
    """Return status and flags keyed by image identifier."""

    return {
        str(row["image_id"]): (str(row["status"]), str(row["flags"]))
        for row in epoch_metrics
    }


def build_preferred_light_curve(measurements, epoch_metrics, method_comparison,
                                settings=None):
    """Select one transparent preferred target result per image and flag outliers."""

    if settings is None:
        settings = get_default_settings()
    configured = settings.get("batch_consistency", {})
    table = _copy_measurements(measurements)
    epoch_status = _epoch_status_lookup(epoch_metrics)
    method_outliers = {
        int(row["measurement_index"]): str(row["status"])
        for row in method_comparison if bool(row["outlier"])
    }
    groups = {}
    for index, row in enumerate(table):
        if str(_row_value(row, "source_type", "")) == "target":
            groups.setdefault(str(row["image_id"]), []).append((index, row))
    preferred_order = configured.get("preferred_order", [])
    accepted = set(configured.get("accepted_image_statuses", ["PASS", "WARN"]))
    records = []
    for image_id, values in groups.items():
        available = {
            "{}:{}".format(row["image_kind"], row["method"]): (index, row)
            for index, row in values if bool(_row_value(row, "valid", True))
        }
        selected_key = next((key for key in preferred_order if key in available), None)
        if selected_key is None:
            records.append(
                {
                    "image_id": image_id, "included_in_final": False,
                    "status": "FAIL", "flags": "PREFERRED_RESULT_UNAVAILABLE",
                    "selection_reason": "no valid method in configured preference order",
                }
            )
            continue
        index, row = available[selected_key]
        magnitude, magnitude_error, magnitude_source = _measurement_magnitude(row)
        image_status, image_flags = epoch_status.get(image_id, ("PASS", ""))
        flags = str(_row_value(row, "flags", ""))
        if image_flags:
            for flag in image_flags.split(";"):
                flags = _append_flag(flags, flag)
        method_status = method_outliers.get(index)
        if method_status is not None:
            flags = _append_flag(flags, "BATCH_MEASUREMENT_OUTLIER")
        included = image_status in accepted and method_status != "FAIL"
        records.append(
            {
                "measurement_index": index,
                "image_id": image_id,
                "mjd": _finite_float(_row_value(row, "mjd_mid")),
                "filter": str(_row_value(row, "filter", "")),
                "telescope": str(_row_value(row, "telescope", "")),
                "site": str(_row_value(row, "site", "")),
                "image_kind": str(row["image_kind"]),
                "method": str(row["method"]),
                "host_light_included": bool(row["host_light_included"]),
                "flux": _finite_float(_row_value(row, "ensemble_corrected_flux"), _finite_float(_row_value(row, "flux"))),
                "flux_uncertainty": _finite_float(_row_value(row, "flux_uncertainty")),
                "snr": _finite_float(_row_value(row, "snr")),
                "magnitude": magnitude,
                "magnitude_uncertainty": magnitude_error,
                "magnitude_source": magnitude_source,
                "classification": str(_row_value(row, "classification", "")),
                "image_status": image_status,
                "method_status": method_status or "PASS",
                "included_in_final": included,
                "flags": flags,
                "selection_reason": "first valid result in configured preference order ({})".format(selected_key),
            }
        )
    temporal_sigma = float(configured.get("temporal_outlier_sigma", 5.0))
    maximum_gap = float(configured.get("temporal_maximum_gap_days", 3.0))
    allowed_rate = float(configured.get("temporal_allowed_rate_mag_per_day", 0.1))
    by_filter = {}
    for index, record in enumerate(records):
        if record.get("mjd") is not None and record.get("magnitude") is not None:
            by_filter.setdefault(record["filter"], []).append(index)
    for indices in by_filter.values():
        indices.sort(key=lambda index: records[index]["mjd"])
        for position in range(1, len(indices) - 1):
            before, current, after = indices[position - 1:position + 2]
            t0, t1, t2 = records[before]["mjd"], records[current]["mjd"], records[after]["mjd"]
            if t1 - t0 > maximum_gap or t2 - t1 > maximum_gap or t2 == t0:
                continue
            expected = records[before]["magnitude"] + (
                records[after]["magnitude"] - records[before]["magnitude"]
            ) * (t1 - t0) / (t2 - t0)
            error = records[current].get("magnitude_uncertainty") or float(
                configured.get("method_disagreement_floor_mag", 0.05)
            )
            residual = records[current]["magnitude"] - expected
            significance = residual / max(error, 0.01)
            # A transient really changes between epochs: a point is only an
            # outlier when it departs from its neighbors by more than its
            # errors AND by more than the source could plausibly change in the
            # time to the farther neighbor (temporal_allowed_rate_mag_per_day).
            allowance = allowed_rate * max(t1 - t0, t2 - t1)
            records[current]["temporal_residual_mag"] = float(residual)
            records[current]["temporal_outlier_sigma"] = float(significance)
            records[current]["temporal_allowance_mag"] = float(allowance)
            if abs(significance) >= temporal_sigma and abs(residual) > allowance:
                records[current]["flags"] = _append_flag(
                    records[current]["flags"], "BATCH_MEASUREMENT_OUTLIER"
                )
    result = _records_table(records)
    for name in result.colnames:
        if name in {"magnitude", "magnitude_uncertainty", "temporal_residual_mag"}:
            result[name].unit = u.mag
        elif name == "mjd":
            result[name].unit = u.day
    return result


def analyze_batch_consistency(
    science_measurements,
    difference_results=None,
    image_records=None,
    zeropoints=None,
    limits=None,
    settings=None,
):
    """Run all batch consistency checks and assemble final light-curve products."""

    if settings is None:
        settings = get_default_settings()
    settings = merge_settings(get_default_settings(), settings)
    if not settings.get("batch_consistency", {}).get("enabled", True):
        raise RuntimeError("Batch consistency checks are disabled")
    measurements = collect_batch_measurements(science_measurements, difference_results)
    comparison_light_curves, stability = build_comparison_star_light_curves(
        measurements, settings
    )
    corrected, corrections = apply_ensemble_corrections(
        measurements, comparison_light_curves, stability, settings
    )
    unstable_keys = {
        (str(row["source_id"]), str(row["filter"]), str(row["method"]))
        for row in stability if bool(row["unstable"])
    }
    for index, row in enumerate(corrected):
        key = (
            str(_row_value(row, "source_id", "")),
            str(_row_value(row, "filter", "")),
            str(_row_value(row, "method", "")),
        )
        if key in unstable_keys:
            corrected["flags"][index] = _append_flag(
                corrected["flags"][index], "COMPARISON_STAR_UNSTABLE"
            )
    epoch_metrics = build_epoch_metrics(image_records or [], zeropoints, limits, settings)
    if len(epoch_metrics) == 0:
        image_records = []
        seen = set()
        for row in corrected:
            image_id = str(row["image_id"])
            if image_id in seen:
                continue
            seen.add(image_id)
            image_records.append(
                {
                    "image_id": image_id,
                    "metadata": {
                        "mjd_mid": _finite_float(_row_value(row, "mjd_mid")),
                        "filter": str(_row_value(row, "filter", "")),
                        "telescope": str(_row_value(row, "telescope", "")),
                        "site": str(_row_value(row, "site", "")),
                    },
                }
            )
        epoch_metrics = build_epoch_metrics(image_records, zeropoints, limits, settings)
    group_summary = summarize_problem_groups(epoch_metrics, settings)
    method_comparison = compare_photometry_methods(corrected, settings)
    epoch_lookup = _epoch_status_lookup(epoch_metrics)
    for index, row in enumerate(corrected):
        epoch_status, epoch_flags = epoch_lookup.get(str(row["image_id"]), ("PASS", ""))
        if epoch_status != "PASS":
            corrected["flags"][index] = _append_flag(
                corrected["flags"][index], "BATCH_EPOCH_PROBLEM"
            )
            for flag in filter(None, epoch_flags.split(";")):
                corrected["flags"][index] = _append_flag(
                    corrected["flags"][index], flag
                )
    for row in method_comparison:
        if bool(row["outlier"]):
            index = int(row["measurement_index"])
            corrected["flags"][index] = _append_flag(
                corrected["flags"][index], "BATCH_MEASUREMENT_OUTLIER"
            )
    preferred = build_preferred_light_curve(
        corrected, epoch_metrics, method_comparison, settings
    )
    # Count stars, not star-method pairs: a star measured three ways is still
    # one star.
    unstable_count = len({
        (str(row["source_id"]), str(row["filter"]))
        for row in stability if bool(row["unstable"])
    })
    untested_count = len({
        (str(row["source_id"]), str(row["filter"]))
        for row in stability if str(row["status"]) == "UNTESTED"
    })
    tested_count = len({
        (str(row["source_id"]), str(row["filter"]))
        for row in stability if str(row["status"]) != "UNTESTED"
    })
    failed_epochs = sum(str(row["status"]) == "FAIL" for row in epoch_metrics)
    outlier_count = sum(bool(row["outlier"]) for row in method_comparison)
    status = "FAIL" if len(preferred) == 0 else "WARN" if any(
        value > 0 for value in (unstable_count, failed_epochs, outlier_count)
    ) else "PASS"
    return {
        "status": status,
        "measurements": corrected,
        "comparison_light_curves": comparison_light_curves,
        "comparison_stability": stability,
        "ensemble_corrections": corrections,
        "epoch_metrics": epoch_metrics,
        "group_summary": group_summary,
        "method_comparison": method_comparison,
        "preferred_light_curve": preferred,
        "unstable_comparison_count": unstable_count,
        "untested_comparison_count": untested_count,
        "tested_comparison_count": tested_count,
        "failed_epoch_count": failed_epochs,
        "measurement_outlier_count": outlier_count,
        "ensemble_enabled": bool(
            settings.get("batch_consistency", {}).get("ensemble_correction", {}).get("enabled", False)
        ),
    }


def save_batch_consistency_products(products, output_directory, object_name="field",
                                    settings=None, overwrite=None):
    """Save batch tables and a compact JSON consistency summary."""

    if settings is None:
        settings = get_default_settings()
    configured = settings.get("batch_consistency", {})
    if overwrite is None:
        overwrite = settings.get("output", {}).get("overwrite", False)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    stem = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in str(object_name)
    )
    requests = {
        "comparison_light_curves": configured.get("save_comparison_light_curves", True),
        "comparison_stability": configured.get("save_stability_table", True),
        "ensemble_corrections": configured.get("save_epoch_metrics", True),
        "epoch_metrics": configured.get("save_epoch_metrics", True),
        "group_summary": configured.get("save_group_summary", True),
        "method_comparison": configured.get("save_method_comparison", True),
        "preferred_light_curve": configured.get("save_preferred_light_curve", True),
        "measurements": configured.get("save_all_flagged_measurements", True),
    }
    paths = {}
    for name, enabled in requests.items():
        table = products.get(name)
        if not enabled or table is None:
            continue
        path = output / "{}_{}.ecsv".format(stem, name)
        table.write(path, format="ascii.ecsv", overwrite=bool(overwrite))
        paths[name] = str(path)
    if configured.get("save_summary", True):
        path = output / "{}_batch_consistency.json".format(stem)
        if path.exists() and not overwrite:
            raise FileExistsError(str(path))
        summary = {
            "status": products.get("status"),
            "unstable_comparison_count": products.get("unstable_comparison_count"),
            "failed_epoch_count": products.get("failed_epoch_count"),
            "measurement_outlier_count": products.get("measurement_outlier_count"),
            "ensemble_enabled": products.get("ensemble_enabled"),
            "preferred_light_curve_rows": len(products.get("preferred_light_curve", [])),
        }
        path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        paths["summary"] = str(path)
    return paths


# ---------------------------------------------------------------------------
# Function-based pipeline control
# ---------------------------------------------------------------------------


PIPELINE_STATUSES = (
    "PASS", "WARN", "FAIL", "APPROVED", "REJECTED", "SKIPPED", "STALE"
)


def _utc_now():
    """Return a compact UTC timestamp for persistent state records."""

    return datetime.now(timezone.utc).isoformat()


def _json_value(value):
    """Convert state metadata into JSON-compatible scalar containers."""

    if value is None or np.ma.is_masked(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _hash_value(value):
    """Return a deterministic SHA-256 digest for dependency tracking."""

    text = json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _pipeline_event(state, status, stage, image_id=None, message=""):
    """Append one status transition to the persistent run history."""

    if status not in PIPELINE_STATUSES:
        raise ValueError("Unknown pipeline status: {}".format(status))
    state.setdefault("events", []).append({
        "time": _utc_now(), "status": status, "stage": stage,
        "image_id": image_id, "message": str(message or ""),
    })


def _stage_definitions():
    """Return the ordered built-in stage dependency schema.

    ``settings`` lists the settings each stage reads: a whole section
    (``"psf"``) or one entry of a section (``"psf.maximum_stars"``).
    ``ignore`` removes parts of a listed section that a later stage reads
    instead, so changing them does not redo this stage. Settings read by an
    earlier stage do not need to be listed again: a change there redoes that
    stage and everything after it.
    """

    return [
        {"name": "read", "scope": "image", "requires": [],
         "settings": ["input", "metadata", "instrument"]},
        {"name": "region", "scope": "image", "requires": ["read"],
         "settings": ["crop"]},
        {"name": "masks", "scope": "image", "requires": ["region"],
         "settings": ["masks"], "ignore": ["masks.cosmic_rays"]},
        {"name": "cosmic_rays", "scope": "image", "requires": ["masks"],
         "settings": ["masks.cosmic_rays"]},
        {"name": "fringe", "scope": "image", "requires": ["cosmic_rays"],
         "settings": ["fringe"]},
        {"name": "background", "scope": "image", "requires": ["fringe"],
         "settings": ["background"]},
        {"name": "source_quality", "scope": "image", "requires": ["background"],
         "settings": ["source_detection", "image_quality"],
         # Read by usability (step 10) only: the comparisons across the batch
         # and the zeropoint/catalog checks.
         "ignore": ["image_quality." + name for name in (
             "usability", "batch_minimum_images", "batch_fwhm_ratio_warn",
             "batch_fwhm_ratio_fail", "batch_ellipticity_offset_warn",
             "batch_ellipticity_offset_fail", "batch_background_ratio_warn",
             "batch_background_ratio_fail", "batch_background_rms_ratio_warn",
             "batch_background_rms_ratio_fail", "zeropoint_scatter_warn_mag",
             "zeropoint_scatter_fail_mag", "minimum_catalog_recovery_warn",
             "minimum_catalog_recovery_fail", "expected_target_magnitude",
             "zeropoint_offset_warn_mag", "zeropoint_offset_fail_mag",
             "minimum_useful_depth_mag", "wcs_rms_warn_arcsec", "wcs_rms_fail_arcsec")]},
        {"name": "astrometry", "scope": "image", "requires": ["source_quality"],
         "settings": ["astrometry", "catalogs"],
         "ignore": ["catalogs.comparison_stars", "catalogs.photometry_catalog",
                    "catalogs.photometry_catalog_by_filter",
                    "catalogs.photometric_match_arcsec"]},
        {"name": "star_selection", "scope": "batch", "requires": ["astrometry"],
         "settings": ["catalogs", "psf.maximum_stars", "psf.minimum_stars",
                      "calibration.catalog",
                      "image_quality.usability.quick_aperture_fwhm",
                      "image_quality.usability.quick_aperture_correction_fwhm",
                      "image_quality.usability.quick_sky_annulus_fwhm"]},
        {"name": "usability", "scope": "batch", "requires": ["star_selection"],
         "settings": ["image_quality"]},
        {"name": "alignment", "scope": "batch", "requires": ["usability"],
         "settings": ["astrometry", "target_position"]},
        {"name": "psf", "scope": "image", "requires": ["alignment"],
         "settings": ["psf"]},
        {"name": "science_photometry", "scope": "image", "requires": ["psf"],
         "settings": ["apertures", "background", "target_position"]},
        {"name": "calibration", "scope": "batch", "requires": ["science_photometry"],
         "settings": ["calibration", "upper_limits"]},
        {"name": "templates", "scope": "batch", "requires": ["calibration"],
         "settings": ["subtraction." + name for name in (
             "enabled", "template_path", "template_source", "template_survey_priority",
             "template_surveys", "survey_names", "survey_filter_map",
             "download_pixel_scale_arcsec", "download_timeout_s", "maximum_mosaic_pixels",
             "cache_directory", "use_cached_templates", "save_downloaded_templates",
             "template_margin_arcmin", "allow_approximate_filter_match",
             "approximate_filter_matches", "minimum_coverage_fraction",
             "minimum_footprint_coverage", "resampling_order", "resampling_tile_rows")]},
        {"name": "subtraction", "scope": "image", "requires": ["templates"],
         "settings": ["subtraction"], "ignore": ["subtraction.photometry"]},
        {"name": "difference_photometry", "scope": "image", "requires": ["subtraction"],
         "settings": ["subtraction", "apertures", "upper_limits"]},
        {"name": "batch_consistency", "scope": "batch",
         "requires": ["difference_photometry"], "settings": ["batch_consistency"]},
        # outputs can run at any time (requires nothing), but its signature
        # follows every other stage so finished products are rebuilt whenever
        # anything upstream changes.
        {"name": "outputs", "scope": "batch", "requires": [],
         "signature_requires": "all", "settings": ["diagnostics", "output"]},
    ]


def _setting_at(settings, path):
    """Value of a dotted settings path (``None`` when it does not exist)."""

    value = settings
    for key in path.split("."):
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def _without_path(value, parts):
    """Copy of a nested mapping without the entry at ``parts``."""

    if not isinstance(value, Mapping) or not parts or parts[0] not in value:
        return value
    copied = dict(value)
    if len(parts) == 1:
        copied.pop(parts[0])
    else:
        copied[parts[0]] = _without_path(copied[parts[0]], parts[1:])
    return copied


def _settings_subset(settings, definition):
    """The settings one stage reads (see :func:`_stage_definitions`)."""

    subset = {
        name: _setting_at(settings, name) for name in definition.get("settings", [])
    }
    for path in definition.get("ignore", []):
        section, *rest = path.split(".")
        if section in subset:
            subset[section] = _without_path(subset[section], rest)
    return subset


def _paths_overlap(first, second):
    """True when one dotted settings path contains the other."""

    first, second = first.split("."), second.split(".")
    size = min(len(first), len(second))
    return first[:size] == second[:size]


def stage_reads_setting(stage_name, path):
    """Whether a stage's results depend on the setting at dotted ``path``.

    Example: ``stage_reads_setting("star_selection", "psf.maximum_stars")``
    is True, ``stage_reads_setting("star_selection", "psf.box_size_pixels")``
    is False.
    """

    definition = next(
        (item for item in _stage_definitions() if item["name"] == stage_name), None
    )
    if definition is None:
        raise KeyError("Unknown pipeline stage: {}".format(stage_name))
    if any(_paths_overlap(path, ignored) and len(path.split(".")) >= len(ignored.split("."))
           for ignored in definition.get("ignore", [])):
        return False
    return any(_paths_overlap(path, name) for name in definition.get("settings", []))


def _override_paths(overrides, prefix=()):
    """Dotted paths of every value set in a nested overrides mapping."""

    paths = []
    for key, value in (overrides or {}).items():
        here = prefix + (str(key),)
        if isinstance(value, Mapping) and value:
            paths.extend(_override_paths(value, here))
        else:
            paths.append(".".join(here))
    return paths


def first_stage_reading(overrides):
    """Name of the first stage that reads any setting in ``overrides``.

    ``"read"`` when no stage lists them, so an unknown setting redoes
    everything.
    """

    paths = _override_paths(overrides)
    for item in _stage_definitions():
        if any(stage_reads_setting(item["name"], path) for path in paths):
            return item["name"]
    return "read"


def pipeline_stage_names():
    """Return the public ordered stage names used by all run modes."""

    return [item["name"] for item in _stage_definitions()]


def _stage_lookup(stage_functions=None):
    definitions = {item["name"]: dict(item) for item in _stage_definitions()}
    runners = _default_stage_functions()
    runners.update(stage_functions or {})
    for name, definition in definitions.items():
        definition["runner"] = runners.get(name)
    unknown = set(runners) - set(definitions)
    if unknown:
        raise KeyError("Unknown pipeline stages: {}".format(", ".join(sorted(unknown))))
    return definitions


def _input_fingerprint(path):
    path = Path(path)
    try:
        stat = path.stat()
    except OSError:
        return {"path": str(path), "exists": False}
    return {
        "path": str(path.resolve()), "exists": True,
        "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns),
    }


def _unique_image_ids(paths):
    counts = {}
    identifiers = []
    for path in paths:
        base = Path(path).name
        counts[base] = counts.get(base, 0) + 1
        identifiers.append(base if counts[base] == 1 else "{}__{}".format(base, counts[base]))
    return identifiers


def initialize_pipeline(
    paths,
    settings=None,
    instrument_name=None,
    target=None,
    image_overrides=None,
    run_directory=None,
    filter_settings=None,
):
    """Create serializable state and in-memory context for a new run.

    Instrument defaults are applied before run settings. Filter defaults and
    user ``filter_settings`` are applied after each FITS header reveals its
    filter, then that image's overrides are applied last.
    """

    from .image import discover_fits_files

    run_settings = deepcopy(settings or {})
    base_settings = resolve_settings(
        instrument_name=instrument_name,
        run_settings=run_settings,
    )
    files = discover_fits_files(paths, settings=base_settings)
    if not files:
        raise FileNotFoundError("No readable FITS inputs were discovered")
    run_directory = Path(
        run_directory or base_settings.get("output", {}).get("directory", "redphot_output")
    )
    identifiers = _unique_image_ids(files)
    overrides = image_overrides or {}
    state = {
        "schema_version": 1,
        "run_id": "{}-{}".format(
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            _hash_value([str(path) for path in files])[:10],
        ),
        "created": _utc_now(), "updated": _utc_now(),
        "run_directory": str(run_directory),
        "stage_order": pipeline_stage_names(),
        "settings": _json_value(base_settings),
        "run_settings": _json_value(run_settings),
        "instrument_name": instrument_name,
        "filter_settings": _json_value(filter_settings or {}),
        "batch_stages": {}, "images": {}, "events": [],
        # Image order of the run (the JSON file stores the images sorted by
        # name); load_pipeline_state restores it so image numbers stay put.
        "image_order": list(identifiers),
    }
    context = {
        "settings": base_settings, "target": target, "images": {},
        "shared": {}, "run_settings": run_settings,
        "instrument_name": instrument_name,
        "filter_settings": deepcopy(filter_settings or {}),
    }
    context["_state"] = state
    for image_id, path in zip(identifiers, files):
        by_image = deepcopy(
            overrides.get(image_id, overrides.get(Path(path).name, {}))
        )
        image_settings = resolve_settings(
            instrument_name=instrument_name,
            run_settings=run_settings,
            image_name=Path(path).name,
            image_overrides={Path(path).name: by_image} if by_image else None,
        )
        state["images"][image_id] = {
            "path": str(path), "input_fingerprint": _input_fingerprint(path),
            "status": "STALE", "failed_stage": None,
            "overrides": _json_value(by_image), "review_decisions": {},
            "stages": {},
        }
        context["images"][image_id] = {
            "path": str(path), "settings": image_settings,
            "record": {"image_id": image_id, "path": str(path),
                       "settings": image_settings},
            "products": {},
        }
        _pipeline_event(state, "STALE", "read", image_id, "new input")
    return state, context


def _state_paths(state, settings=None):
    configured = (settings or state.get("settings") or {}).get("pipeline", {})
    root = Path(state["run_directory"])
    return (
        root / configured.get("state_filename", "pipeline_state.json"),
        root / configured.get("checkpoint_filename", "pipeline_context.pkl"),
    )


def _exact_wcs_values(wcs):
    """The numbers of a WCS that its FITS header rounds to ~14 digits."""

    core = wcs.wcs
    cd_form = bool(core.has_cd() and not core.has_pc())
    values = {
        "crpix": np.array(core.crpix, dtype=float),
        "crval": np.array(core.crval, dtype=float),
        # Keep the CD or PC form: code that later sets one of them (for
        # example the alignment) must find the WCS in the form it had.
        "cd": np.array(core.cd, dtype=float) if cd_form else None,
        "pc": None if cd_form else np.array(core.get_pc(), dtype=float),
        "cdelt": None if cd_form else np.array(core.get_cdelt(), dtype=float),
        "lonpole": float(core.lonpole), "latpole": float(core.latpole),
        "pv": [tuple(item) for item in core.get_pv()],
        "pixel_shape": wcs.pixel_shape,
        "sip": None,
    }
    if wcs.sip is not None:
        sip = wcs.sip
        values["sip"] = tuple(
            None if value is None else np.array(value, dtype=float)
            for value in (sip.a, sip.b, sip.ap, sip.bp, sip.crpix)
        )
    return values


def _rebuild_wcs(header_text, values):
    """Rebuild a WCS saved by :class:`_CheckpointPickler` with its exact numbers."""

    import warnings

    from astropy.io import fits
    from astropy.wcs import WCS, Sip

    import re

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        header = fits.Header.fromstring(header_text)
        if values.get("cd") is not None:
            # The header carries the matrix as PC with CDELT = 1; put it back
            # as CD so the rebuilt WCS has the form of the original.
            for key in list(header.keys()):
                if re.match(r"^(PC\d+_\d+|CDELT\d+)$", key):
                    del header[key]
            for row, line in enumerate(values["cd"], 1):
                for column, value in enumerate(line, 1):
                    header["CD{}_{}".format(row, column)] = float(value)
        wcs = WCS(header, relax=True)
        core = wcs.wcs
        core.crpix = values["crpix"]
        core.crval = values["crval"]
        if values.get("cd") is not None:
            core.cd = values["cd"]
        else:
            core.pc = values["pc"]
            core.cdelt = values["cdelt"]
        core.lonpole = values["lonpole"]
        core.latpole = values["latpole"]
        if values["pv"]:
            core.set_pv(values["pv"])
        if values["sip"] is not None:
            wcs.sip = Sip(*values["sip"])
        if values["pixel_shape"] is not None:
            wcs.pixel_shape = values["pixel_shape"]
        core.set()
    return wcs


class _CheckpointPickler(pickle.Pickler):
    """Pickler that keeps WCS numbers exact.

    Astropy pickles a WCS through its FITS header, which rounds every number
    to about 14 digits; a reloaded run would then measure positions very
    slightly differently from the run that saved it, and results that should
    be unchanged would no longer be identical.
    """

    def reducer_override(self, obj):
        from astropy.wcs import WCS

        if type(obj) is WCS and not any(
            getattr(obj, name, None) is not None
            for name in ("cpdis1", "cpdis2", "det2im1", "det2im2")
        ):
            try:
                return _rebuild_wcs, (obj.to_header_string(relax=True),
                                      _exact_wcs_values(obj))
            except Exception:  # anything unusual: astropy's own pickling
                return NotImplemented
        return NotImplemented


def save_pipeline_state(state, context):
    """Atomically save readable run state and a local scientific checkpoint."""

    state["updated"] = _utc_now()
    state_path, checkpoint_path = _state_paths(state, context.get("settings"))
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_temporary = state_path.with_suffix(state_path.suffix + ".tmp")
    checkpoint_temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    state_temporary.write_text(json.dumps(_json_value(state), indent=2, sort_keys=True) + "\n")
    with checkpoint_temporary.open("wb") as handle:
        _CheckpointPickler(handle, protocol=pickle.HIGHEST_PROTOCOL).dump(context)
    state_temporary.replace(state_path)
    checkpoint_temporary.replace(checkpoint_path)
    return {"state": str(state_path), "checkpoint": str(checkpoint_path)}


def load_pipeline_state(run_directory, settings=None):
    """Load a previous local run and its trusted redphot checkpoint.

    Pickle checkpoints must only be loaded from runs created locally by the
    user. If the product checkpoint is missing, completed stages are marked
    ``STALE`` so they can be rebuilt safely from the readable JSON state.
    """

    provisional = merge_settings(get_default_settings(), settings or {})
    state_path = Path(run_directory) / provisional["pipeline"]["state_filename"]
    if not state_path.exists():
        raise FileNotFoundError(str(state_path))
    state = json.loads(state_path.read_text())
    checkpoint_path = Path(run_directory) / provisional["pipeline"]["checkpoint_filename"]
    if checkpoint_path.exists():
        with checkpoint_path.open("rb") as handle:
            context = pickle.load(handle)
    else:
        context = {
            "settings": merge_settings(get_default_settings(), state.get("settings", {})),
            "target": None,
            "shared": {},
            "images": {},
            "run_settings": deepcopy(state.get("run_settings", state.get("settings", {}))),
            "instrument_name": state.get("instrument_name"),
            "filter_settings": deepcopy(state.get("filter_settings", {})),
        }
        for image_id, image in state["images"].items():
            image_settings = merge_settings(
                context["settings"], image.get("overrides", {})
            )
            context["images"][image_id] = {
                "path": image["path"], "settings": image_settings,
                "record": {"image_id": image_id, "path": image["path"],
                           "settings": image_settings}, "products": {},
            }
            for stage in image.get("stages", {}).values():
                if stage.get("status") in {"PASS", "WARN", "APPROVED"}:
                    stage["status"] = "STALE"
        for stage in state.get("batch_stages", {}).values():
            if stage.get("status") in {"PASS", "WARN", "APPROVED"}:
                stage["status"] = "STALE"
    order = state.get("image_order") or [
        image_id for image_id in context.get("images", {}) if image_id in state["images"]
    ]
    order = [image_id for image_id in order if image_id in state["images"]]
    order += [image_id for image_id in state["images"] if image_id not in order]
    state["images"] = {image_id: state["images"][image_id] for image_id in order}
    state["image_order"] = order
    context["_state"] = state
    return state, context


def _stage_signature(state, context, definition, image_id=None):
    settings = (
        context["images"][image_id]["settings"] if image_id is not None
        else context["settings"]
    )
    subset = _settings_subset(settings, definition)
    dependencies = {}
    required_names = definition.get("requires", [])
    if definition.get("signature_requires") == "all":
        required_names = [
            name for name in pipeline_stage_names() if name != definition["name"]
        ]
    for required in required_names:
        if required in state.get("batch_stages", {}):
            entry = state["batch_stages"][required]
            digests = entry.get("image_digests") or {}
            if image_id is not None and image_id in digests:
                # An image stage depends only on this image's part of the
                # batch result (see _batch_image_digests), so redoing a
                # batch stage redoes later steps only for images whose
                # part changed.
                dependencies[required] = {
                    "digest": digests[image_id],
                    "status": state["images"][image_id].get("stages", {})
                    .get(required, {}).get("status"),
                }
                continue
            dependencies[required] = {
                "batch": (entry.get("signature"), entry.get("status")),
                "image_reviews": {
                    key: value.get("stages", {}).get(required, {}).get("status")
                    for key, value in state["images"].items()
                    if required in value.get("stages", {})
                },
            }
        elif image_id is not None:
            entry = state["images"][image_id].get("stages", {}).get(required, {})
            dependencies[required] = (
                entry.get("signature"), entry.get("status"), entry.get("review_status")
            )
        else:
            dependencies[required] = {
                key: (
                    value.get("stages", {}).get(required, {}).get("signature"),
                    value.get("stages", {}).get(required, {}).get("status"),
                ) for key, value in state["images"].items()
            }
    payload = {"stage": definition["name"], "settings": subset,
               "dependencies": dependencies}
    if image_id is not None:
        payload["input"] = state["images"][image_id].get("input_fingerprint")
    else:
        # Per-image settings (overrides, instrument profiles) that differ
        # from the run settings in what this batch stage reads.
        differing = {}
        for key, image in context.get("images", {}).items():
            image_subset = _settings_subset(image.get("settings") or settings, definition)
            if image_subset != subset:
                differing[key] = image_subset
        if differing:
            payload["image_settings"] = differing
    return _hash_value(payload)


# ---------------------------------------------------------------------------
# Per-image digests of batch results
# ---------------------------------------------------------------------------
# What the image stages after a batch stage read from its result, split per
# image. Keys of the result holding tables or lists with an ``image_id`` are
# reduced to that image's rows; "shared" keys apply to every image.
_BATCH_IMAGE_PARTS = {
    "star_selection": {"rows": ["measurements", "summaries"], "shared": []},
    "usability": {"rows": ["decisions", "star_residuals"], "shared": []},
    # projections and residuals only feed the plots; the target position is
    # compared to pipeline.rerun_position_tolerance_mas (_shared_digest_part).
    "alignment": {"rows": ["alignments"], "shared": ["target_solution"]},
    "calibration": {"rows": ["measurements", "zeropoints", "calibration_stars",
                             "aperture_corrections", "limits"], "shared": []},
    "templates": {"rows": [], "shared": ["flags"]},
}


def _feed_hash(digest, value, cache=None):
    """Add a stable byte form of ``value`` to a hashlib object."""

    from astropy.wcs import WCS

    if cache is not None and id(value) in cache:
        digest.update(cache[id(value)][1].encode())
        return
    if cache is not None and isinstance(value, np.ndarray) and value.size > 100000:
        # Large arrays shared by several images (templates) are hashed once;
        # the cache keeps a reference so the id cannot be reused.
        cache[id(value)] = (value, _content_hash(value))
        digest.update(cache[id(value)][1].encode())
        return
    if value is None or isinstance(value, (bool, int, float, str, bytes, np.generic)):
        digest.update(repr(value).encode())
    elif isinstance(value, Table):
        digest.update(b"<table>")
        for name in value.colnames:
            digest.update(str(name).encode())
            _feed_hash(digest, value[name], cache)
    elif isinstance(value, np.ndarray):
        digest.update("{}{}{}".format(type(value).__name__, value.dtype.str,
                                      value.shape).encode())
        unit = getattr(value, "unit", None)
        if unit is not None:
            digest.update(str(unit).encode())
        if np.ma.isMaskedArray(value):
            digest.update(np.ascontiguousarray(np.ma.getmaskarray(value)).view(np.uint8))
            value = np.ma.getdata(value)
        if value.dtype.hasobject:
            for item in value.ravel():
                _feed_hash(digest, item, cache)
        else:
            # Hash the array's memory directly (no copy for contiguous arrays).
            digest.update(np.ascontiguousarray(value).reshape(-1).view(np.uint8))
    elif isinstance(value, Mapping):
        for key in sorted(value, key=str):
            digest.update(str(key).encode())
            _feed_hash(digest, value[key], cache)
    elif isinstance(value, (list, tuple)):
        digest.update("<{}>".format(len(value)).encode())
        for item in value:
            _feed_hash(digest, item, cache)
    elif isinstance(value, set):
        for item in sorted(value, key=repr):
            _feed_hash(digest, item, cache)
    elif isinstance(value, WCS):
        digest.update(value.to_header_string(relax=True).encode())
    elif hasattr(value, "data") and hasattr(value, "wcs") and hasattr(value, "mask"):
        for part in (value.data, value.mask, value.wcs, getattr(value, "uncertainty", None)):
            _feed_hash(digest, getattr(part, "array", part), cache)
    elif hasattr(value, "ra") and hasattr(value, "dec"):
        _feed_hash(digest, np.asarray(value.ra.deg), cache)
        _feed_hash(digest, np.asarray(value.dec.deg), cache)
    else:
        import re

        digest.update(re.sub(r" at 0x[0-9a-fA-F]+", "", repr(value)).encode())


def _content_hash(value, cache=None):
    digest = hashlib.sha256()
    _feed_hash(digest, value, cache)
    return digest.hexdigest()


def _rows_for_image(value, image_id):
    """The rows of a table or list that belong to one image."""

    if isinstance(value, Table):
        if "image_id" not in value.colnames or len(value) == 0:
            return value
        return value[np.asarray(value["image_id"]).astype(str) == str(image_id)]
    if isinstance(value, (list, tuple)):
        return [
            item for item in value
            if not isinstance(item, Mapping) or str(item.get("image_id")) == str(image_id)
        ]
    return value


def _template_part(result, context, image_id):
    """The template (and footprint) one image is subtracted against."""

    templates = (result or {}).get("templates") or {}
    metadata = context["images"][image_id]["record"].get("metadata") or {}
    template = templates.get(str(metadata.get("filter")), templates.get("default"))
    if not isinstance(template, Mapping):
        return template
    part = {key: value for key, value in template.items() if key != "science_footprints"}
    part["science_footprints"] = _rows_for_image(
        template.get("science_footprints") or [], image_id
    )
    return part


def _shared_digest_part(stage, result, settings, keys, previous=None):
    """The shared part of a batch result as the later image stages use it.

    For alignment only the target position (and whether it is frozen) is
    used. When it moved by less than ``pipeline.rerun_position_tolerance_mas``
    from the position the previous digests used (``previous``), that previous
    position is kept, so the images are not all redone for a negligible move.
    Returns the shared part and the position used.
    """

    shared = {key: result.get(key) for key in keys}
    target = shared.get("target_solution")
    position = None
    if stage == "alignment" and isinstance(target, Mapping):
        position = [_finite_float(target.get("ra_deg")), _finite_float(target.get("dec_deg"))]
        tolerance = float((settings or {}).get("pipeline", {}).get(
            "rerun_position_tolerance_mas", 10.0) or 0.0)
        if (tolerance > 0 and previous is not None and None not in position
                and None not in list(previous)):
            ra0, dec0 = (float(value) for value in previous)
            moved = 3.6e6 * float(np.hypot(
                (position[0] - ra0) * np.cos(np.deg2rad(dec0)), position[1] - dec0))
            if moved < tolerance:
                position = [ra0, dec0]
        shared["target_solution"] = {
            "position": position, "frozen": target.get("frozen"),
            "version": target.get("version"),
        }
    return shared, position


def _batch_image_digests(state, context, definition, result, previous=None):
    """One digest per image of the part of a batch result that image uses.

    Returns ``None`` for batch stages without per-image parts (their later
    image stages then follow the whole batch signature).
    """

    stage = definition["name"]
    parts = _BATCH_IMAGE_PARTS.get(stage)
    if parts is None or not isinstance(result, Mapping):
        return None
    expected = parts["rows"] + parts["shared"] + (["templates"] if stage == "templates" else [])
    if not any(key in result for key in expected):
        return None  # not the usual result (a replaced stage function)
    batch_entry = state.get("batch_stages", {}).get(stage, {})
    cache = {}
    digests = {}
    shared_part, position = _shared_digest_part(
        stage, result, context.get("settings"), parts["shared"],
        (previous or {}).get("digest_target_position"),
    )
    if position is not None:
        batch_entry["digest_target_position"] = position
    shared = _content_hash(shared_part, cache)
    for image_id, image in state["images"].items():
        upstream = []
        for required in definition.get("requires", []):
            if required in state.get("batch_stages", {}):
                entry = state["batch_stages"][required]
                upstream.append((entry.get("image_digests") or {}).get(
                    image_id, entry.get("signature")))
            else:
                entry = image.get("stages", {}).get(required, {})
                upstream.append((entry.get("signature"), entry.get("status"),
                                 entry.get("review_status")))
        own = {key: _rows_for_image(result.get(key), image_id) for key in parts["rows"]}
        if stage == "templates":
            own["template"] = _template_part(result, context, image_id)
        digests[image_id] = _content_hash(
            {"stage": stage, "status": batch_entry.get("status"), "shared": shared,
             "own": own, "upstream": upstream}, cache,
        )
    return digests


def _mark_images_after_batch(state, stage_name, image_ids, reason):
    """Mark the image stages after a batch stage stale for some images.

    Stops at the next batch stage: that one compares its own per-image
    results when it runs again.
    """

    definitions = {item["name"]: item for item in _stage_definitions()}
    for name in _downstream_names(stage_name)[1:]:
        if definitions[name]["scope"] == "batch":
            break
        for identifier in image_ids:
            entry = state["images"][identifier].get("stages", {}).get(name)
            if entry and entry.get("status") not in {"STALE", "REJECTED"} and not (
                entry.get("status") == "SKIPPED" and not entry.get("blocked", False)
            ):
                entry["status"] = "STALE"
                entry["review_status"] = None
                entry["stale_reason"] = reason
                _pipeline_event(state, "STALE", name, identifier, reason)
            _update_image_status(state, identifier)


def _stage_status(result):
    """Infer PASS/WARN/FAIL/SKIPPED from one scientific stage result."""

    if result is None:
        return "PASS"
    if isinstance(result, Mapping):
        for key in ("status", "quality_status", "metadata_status", "automatic_status"):
            value = result.get(key)
            if str(value).upper() in PIPELINE_STATUSES:
                return str(value).upper()
        for key in ("info", "quality", "decision"):
            nested = result.get(key)
            if isinstance(nested, Mapping):
                status = _stage_status(nested)
                if status != "PASS":
                    return status
        if result.get("skipped") not in (None, False):
            return "SKIPPED"
        flags = result.get("flags", result.get("quality_flags", []))
        if flags:
            return "WARN"
    return "PASS"


def _accepted_status(status):
    return status in {"PASS", "WARN", "APPROVED", "SKIPPED"}


def _accepted_entry(entry):
    return bool(entry) and _accepted_status(entry.get("status")) and not bool(
        entry.get("blocked", False)
    )


def _record_for(context, image_id):
    return context["images"][image_id]["record"]


def _active_image_ids(state, required_stage=None):
    values = []
    for image_id, image in state["images"].items():
        if image.get("status") == "REJECTED":
            continue
        if required_stage is not None:
            entry = image.get("stages", {}).get(required_stage, {})
            if not _accepted_entry(entry):
                continue
        values.append(image_id)
    return values


def _dependency_ready(state, definition, image_id=None):
    for required in definition.get("requires", []):
        batch = state.get("batch_stages", {}).get(required)
        if batch is not None:
            if not _accepted_entry(batch):
                return False, "batch dependency {} is {}".format(required, batch.get("status"))
            continue
        if image_id is not None:
            entry = state["images"][image_id].get("stages", {}).get(required, {})
            if not _accepted_entry(entry):
                return False, "dependency {} is {}".format(required, entry.get("status"))
        else:
            active = _active_image_ids(state, required)
            if not active:
                return False, "no image completed dependency {}".format(required)
    return True, None


def _dependency_stale(state, definition, image_id):
    """True when a stage this one needs is marked STALE (waiting to be redone)."""

    for required in definition.get("requires", []):
        batch = state.get("batch_stages", {}).get(required)
        entry = batch if batch is not None else (
            state["images"][image_id].get("stages", {}).get(required, {})
        )
        if (entry or {}).get("status") == "STALE":
            return True
    return False


def _stage_entry(status, signature, result=None, error=None, blocked=False):
    return {
        "status": status, "signature": signature, "updated": _utc_now(),
        "error": error, "result_type": None if result is None else type(result).__name__,
        "blocked": bool(blocked),
    }


def _update_image_status(state, image_id):
    image = state["images"][image_id]
    statuses = [entry.get("status") for entry in image.get("stages", {}).values()]
    if any(value == "REJECTED" for value in statuses):
        image["status"] = "REJECTED"
    elif any(value == "FAIL" for value in statuses):
        image["status"] = "FAIL"
    elif any(value == "STALE" for value in statuses):
        image["status"] = "STALE"
    elif statuses and any(value == "WARN" for value in statuses):
        image["status"] = "WARN"
    elif statuses:
        image["status"] = "PASS"


def _rejection_stage(image):
    """Return the first stage at which an image was rejected, or ``None``."""

    for name in pipeline_stage_names():
        entry = image.get("stages", {}).get(name, {})
        decision = image.get("review_decisions", {}).get(name, {})
        if entry.get("status") == "REJECTED" or decision.get("decision") == "REJECTED":
            return name
    return None


# Image stages that replace the working image.  Each one must start from the
# output of the previous one, never from its own earlier output, so a stage can
# be rerun for one image (for example with new background settings) without
# compounding its correction.
CCD_STAGES = ("read", "region", "masks", "cosmic_rays", "fringe", "background")


def _restore_stage_input(context, image_id, stage):
    """Point the working image at the output of the stage before ``stage``."""

    image = context["images"][image_id]
    outputs = image.get("stage_ccd")
    if not outputs or stage == "read":
        return
    names = pipeline_stage_names()
    earlier = [
        name for name in CCD_STAGES
        if names.index(name) < names.index(stage) and outputs.get(name) is not None
    ]
    if not earlier:
        return
    ccd = outputs[earlier[-1]]
    image["working_ccd"] = ccd
    image["record"]["ccd"] = ccd
    image["record"]["shape"] = getattr(ccd, "shape", image["record"].get("shape"))


def _store_stage_output(context, image_id, stage):
    """Remember the working image produced by one CCD stage."""

    if stage not in CCD_STAGES:
        return
    image = context["images"][image_id]
    if image.get("working_ccd") is None:
        return
    outputs = image.setdefault("stage_ccd", {})
    outputs[stage] = image["working_ccd"]
    # A rebuilt stage invalidates the stored outputs of every later CCD stage.
    for later in CCD_STAGES[CCD_STAGES.index(stage) + 1:]:
        outputs.pop(later, None)


def _execute_image_stage(state, context, definition, image_id):
    stage = definition["name"]
    image = state["images"][image_id]
    ready, reason = _dependency_ready(state, definition, image_id)
    signature = _stage_signature(state, context, definition, image_id)
    failed_stage = image.get("failed_stage")
    failed_entry = image.get("stages", {}).get(failed_stage, {})
    failure_blocks = (
        failed_stage in pipeline_stage_names()
        and failed_entry.get("status") == "FAIL"
        and pipeline_stage_names().index(stage) > pipeline_stage_names().index(failed_stage)
    )
    rejected_at = _rejection_stage(image) if image.get("status") == "REJECTED" else None
    names = pipeline_stage_names()
    if rejected_at == stage:
        # Keep the review decision itself; it is changed only by review_image.
        return image["stages"].get(stage, {}).get("status", "REJECTED")
    if image.get("status") == "REJECTED" and (
        rejected_at is None or names.index(stage) > names.index(rejected_at)
    ):
        status = "SKIPPED"
        image["stages"][stage] = _stage_entry(
            status, signature, error="image rejected", blocked=True
        )
        _pipeline_event(state, status, stage, image_id, "image rejected")
        return status
    # Stages before the rejecting gate keep their products and are reused or
    # rebuilt normally, so a rejection never erases completed work.
    if failure_blocks:
        status = "SKIPPED"
        message = "blocked by failed stage {}".format(failed_stage)
        image["stages"][stage] = _stage_entry(
            status, signature, error=message, blocked=True
        )
        _pipeline_event(state, status, stage, image_id, message)
        return status
    previous = image.get("stages", {}).get(stage, {})
    if not ready:
        if _accepted_entry(previous) and _dependency_stale(state, definition, image_id):
            # The step this one needs is waiting to be redone; keep the
            # current result until then (it is checked again afterwards).
            return previous["status"]
        status = "SKIPPED"
        image["stages"][stage] = _stage_entry(
            status, signature, error=reason, blocked=True
        )
        _pipeline_event(state, status, stage, image_id, reason)
        _update_image_status(state, image_id)
        return status
    if previous.get("signature") == signature and _accepted_entry(previous):
        return previous["status"]
    runner = definition.get("runner")
    if runner is None:
        raise RuntimeError("No function is registered for stage {}".format(stage))
    try:
        _restore_stage_input(context, image_id, stage)
        result = runner(context, image_id, context["images"][image_id]["settings"])
        _store_stage_output(context, image_id, stage)
        # The read stage resolves the image's settings again with the
        # instrument found in its header; sign the result with the settings
        # it was made with, or the next pass would redo it.
        signature = _stage_signature(state, context, definition, image_id)
        status = _stage_status(result)
        context["images"][image_id]["products"][stage] = result
        image["stages"][stage] = _stage_entry(status, signature, result=result)
        if status == "FAIL":
            image["failed_stage"] = stage
        elif image.get("failed_stage") == stage:
            image["failed_stage"] = None
        _pipeline_event(state, status, stage, image_id)
    except Exception as error:
        status = "FAIL"
        detail = traceback.format_exc() if context["settings"]["pipeline"].get("save_tracebacks", True) else str(error)
        image["stages"][stage] = _stage_entry(status, signature, error=detail)
        image["failed_stage"] = stage
        _pipeline_event(state, status, stage, image_id, str(error))
    _update_image_status(state, image_id)
    return status


def _apply_batch_image_statuses(state, stage, result):
    """Copy per-image statuses returned by batch stages into image histories."""

    rows = []
    if stage == "usability" and isinstance(result, Mapping):
        rows = result.get("decisions", [])
    elif stage == "alignment" and isinstance(result, Mapping):
        rows = result.get("alignments", [])
    elif stage == "star_selection" and isinstance(result, Mapping):
        rows = result.get("summaries", [])
    for row in rows:
        image_id = str(row.get("image_id"))
        if image_id not in state["images"]:
            continue
        status = str(row.get("status", row.get("automatic_status", "PASS"))).upper()
        if status not in PIPELINE_STATUSES:
            status = "WARN" if row.get("flags") else "PASS"
        state["images"][image_id]["stages"][stage] = _stage_entry(status, None, row)
        _update_image_status(state, image_id)


def _execute_batch_stage(state, context, definition):
    stage = definition["name"]
    ready, reason = _dependency_ready(state, definition)
    signature = _stage_signature(state, context, definition)
    if not ready:
        state["batch_stages"][stage] = _stage_entry(
            "SKIPPED", signature, error=reason, blocked=True
        )
        _pipeline_event(state, "SKIPPED", stage, message=reason)
        return "SKIPPED"
    previous = state.get("batch_stages", {}).get(stage, {})
    if previous.get("signature") == signature and _accepted_entry(previous):
        return previous["status"]
    runner = definition.get("runner")
    if runner is None:
        raise RuntimeError("No function is registered for stage {}".format(stage))
    try:
        result = runner(context, None, context["settings"])
        status = _stage_status(result)
        context["shared"][stage] = result
        state["batch_stages"][stage] = _stage_entry(status, signature, result=result)
        _apply_batch_image_statuses(state, stage, result)
        old_digests = previous.get("image_digests")
        digests = _batch_image_digests(state, context, definition, result, previous)
        if digests is not None:
            state["batch_stages"][stage]["image_digests"] = digests
            changed = [
                image_id for image_id in digests
                if (old_digests or {}).get(image_id) != digests[image_id]
                and state["images"][image_id].get("status") != "REJECTED"
            ]
            state["batch_stages"][stage]["changed_images"] = changed
            _mark_images_after_batch(
                state, stage, changed, "{} result changed".format(stage)
            )
        _pipeline_event(state, status, stage)
    except Exception as error:
        status = "FAIL"
        detail = traceback.format_exc() if context["settings"]["pipeline"].get("save_tracebacks", True) else str(error)
        state["batch_stages"][stage] = _stage_entry(status, signature, error=detail)
        _pipeline_event(state, status, stage, message=str(error))
    return status


def _automatic_review(state, stage, mode):
    if mode == "none":
        return
    accepted = {"PASS"} if mode == "approve_pass" else {"PASS", "WARN", "SKIPPED"}
    for image_id, image in state["images"].items():
        entry = image.get("stages", {}).get(stage)
        if entry is None:
            continue
        if entry.get("status") in {"APPROVED", "REJECTED"} or entry.get("blocked", False):
            # Keep existing (possibly manual) decisions; blocked entries never ran.
            continue
        automatic = entry.get("status")
        entry["automatic_status"] = automatic
        decision = "APPROVED" if automatic in accepted else "REJECTED"
        entry["status"] = decision
        entry["review_status"] = decision
        image.setdefault("review_decisions", {})[stage] = {
            "decision": decision, "automatic_status": automatic,
            "time": _utc_now(), "note": "automatic review",
        }
        _pipeline_event(state, decision, stage, image_id, "automatic review")
        _update_image_status(state, image_id)


def _stage_updates(state, stage_name):
    """Timestamps of every entry of one stage, to detect what a run changed."""

    values = {None: (state.get("batch_stages", {}).get(stage_name) or {}).get("updated")}
    for identifier, image in state["images"].items():
        values[identifier] = (image.get("stages", {}).get(stage_name) or {}).get("updated")
    return values


def _write_stage_diagnostics(state, context, stage_name, before):
    """Draw plots for the entries of ``stage_name`` that changed since ``before``."""

    if stage_name == "outputs":
        return
    after = _stage_updates(state, stage_name)
    changed = [key for key, value in after.items() if key is not None and value != before.get(key)]
    batch_changed = after.get(None) != before.get(None)
    if not changed and not batch_changed:
        return
    try:
        from .stage_reports import PER_IMAGE_BATCH_STAGES, write_stage_diagnostics

        if batch_changed and stage_name in PER_IMAGE_BATCH_STAGES:
            changed = None  # a rerun batch stage redraws every image's view
        write_stage_diagnostics(state, context, stage_name, changed, batch=batch_changed)
    except Exception as error:  # plotting must never stop a run
        state.setdefault("diagnostic_errors", []).append(
            {"stage": stage_name, "time": _utc_now(), "error": repr(error)}
        )


def _short_error(entry):
    """Last line of a stage error, for one-line progress messages."""

    text = str((entry or {}).get("error") or "").strip()
    return text.splitlines()[-1][:110] if text else ""


def _image_progress(state, stage_name, identifier, before, number, total, timer):
    """One progress line for an image after a stage ran (or was reused)."""

    entry = state["images"][identifier].get("stages", {}).get(stage_name) or {}
    status = entry.get("status") or "?"
    label = "[{}/{}] {}".format(number, total, identifier)
    if entry.get("updated") == before.get(identifier):
        _progress.progress("{}  {} (unchanged, kept)".format(label, status))
        return
    detail = ""
    if status in {"FAIL", "SKIPPED"} or entry.get("blocked"):
        detail = "  " + _short_error(entry)
    _progress.progress("{}  {}  {}{}".format(label, status, timer, detail))


def run_pipeline_stage(state, context, stage_name, image_id=None,
                       stage_functions=None, mode="automatic", save=True):
    """Run exactly one named stage for one image or the eligible batch.

    When ``diagnostics.save_stage_plots`` is on (the default), the figures and
    ``summary.csv`` of every entry that changed are written to
    ``<run_directory>/diagnostics/<NN>_<stage>/``.

    Stages listed in ``pipeline.review_gates`` (usability and psf by default)
    decide automatically in every mode: PASS and WARN images are approved,
    FAIL images are rejected and skipped by later stages. Nothing waits for a
    manual decision; :func:`review_image` can still change a decision later.
    ``mode`` is kept for compatibility and no longer changes what a stage does.
    """

    _progress.configure(context["settings"].get("pipeline", {}).get("verbose", True))
    before = _stage_updates(state, stage_name)
    definitions = _stage_lookup(stage_functions)
    if stage_name not in definitions:
        raise KeyError("Unknown pipeline stage: {}".format(stage_name))
    definition = definitions[stage_name]
    if definition["scope"] == "image":
        identifiers = [image_id] if image_id is not None else list(state["images"])
        _progress.stage_started(stage_name, len(identifiers), "image")
        for number, identifier in enumerate(identifiers, 1):
            if identifier not in state["images"]:
                raise KeyError("Unknown image: {}".format(identifier))
            timer = _progress.Timer()
            _execute_image_stage(state, context, definition, identifier)
            _image_progress(state, stage_name, identifier, before, number,
                            len(identifiers), timer)
    else:
        if image_id is not None:
            raise ValueError("{} is a batch stage".format(stage_name))
        _progress.stage_started(stage_name, len(state["images"]), "batch")
        _execute_batch_stage(state, context, definition)
        entry = state.get("batch_stages", {}).get(stage_name) or {}
        if entry.get("updated") == before.get(None):
            _progress.progress("{} (unchanged, kept)".format(entry.get("status")))
        else:
            detail = _short_error(entry) if entry.get("status") in {"FAIL", "SKIPPED"} else ""
            _progress.progress("batch status {}{}".format(
                entry.get("status"), "  " + detail if detail else ""))
    gates = set(context["settings"].get("pipeline", {}).get("review_gates", []))
    if definition.get("review") or stage_name in gates:
        _automatic_review(
            state, stage_name,
            context["settings"]["pipeline"].get("automatic_review", "approve_pass_warn"),
        )
        rejected = [
            identifier for identifier, image in state["images"].items()
            if (image.get("stages", {}).get(stage_name) or {}).get("status") == "REJECTED"
            and (image.get("review_decisions", {}).get(stage_name) or {}).get("note")
            == "automatic review"
        ]
        if rejected:
            _progress.progress("rejected automatically (FAIL at {}): {}".format(
                stage_name, ", ".join(rejected)))
    plot_timer = _progress.Timer()
    _write_stage_diagnostics(state, context, stage_name, before)
    if stage_name != "outputs" and _stage_updates(state, stage_name) != before:
        _progress.progress("plots written ({})".format(plot_timer))
    if save and context["settings"]["pipeline"].get("save_state_after_stage", True):
        save_timer = _progress.Timer()
        save_pipeline_state(state, context)
        _progress.progress("run state saved ({})".format(save_timer))
    _progress.stage_finished()
    return state, context


def run_pipeline_through(state, context, through_stage=None, stage_functions=None,
                         mode="automatic", save=True):
    """Run every stage in order (up to ``through_stage``).

    Review gates decide automatically (see :func:`run_pipeline_stage`), so the
    run never stops to wait for a decision.
    """

    names = pipeline_stage_names()
    if through_stage is not None:
        if through_stage not in names:
            raise KeyError("Unknown pipeline stage: {}".format(through_stage))
        names = names[:names.index(through_stage) + 1]
    for name in names:
        run_pipeline_stage(
            state, context, name, stage_functions=stage_functions, mode=mode, save=save
        )
    return state, context


def run_one_image(path, settings=None, instrument_name=None, target=None,
                  image_overrides=None, run_directory=None, through_stage=None,
                  stage_functions=None, mode="automatic", filter_settings=None):
    """Initialize and process one image with the standard controller."""

    state, context = initialize_pipeline(
        [path], settings, instrument_name, target, image_overrides, run_directory,
        filter_settings,
    )
    return run_pipeline_through(
        state, context, through_stage, stage_functions, mode
    )


def run_batch(paths, settings=None, instrument_name=None, target=None,
              image_overrides=None, run_directory=None, through_stage=None,
              stage_functions=None, mode="automatic", filter_settings=None):
    """Initialize and process a FITS batch while containing image failures."""

    state, context = initialize_pipeline(
        paths, settings, instrument_name, target, image_overrides, run_directory,
        filter_settings,
    )
    return run_pipeline_through(
        state, context, through_stage, stage_functions, mode
    )


def resume_pipeline(run_directory, through_stage=None, stage_functions=None,
                    mode="automatic", settings=None):
    """Resume a previous run, rerunning only stale or incomplete products."""

    state, context = load_pipeline_state(run_directory, settings)
    refresh_pipeline_staleness(state, context, stage_functions)
    return run_pipeline_through(
        state, context, through_stage, stage_functions, mode
    )


def _downstream_names(stage_name):
    names = pipeline_stage_names()
    if stage_name not in names:
        raise KeyError("Unknown pipeline stage: {}".format(stage_name))
    return names[names.index(stage_name):]


def mark_pipeline_stale(state, stage_name, image_id=None, reason="upstream change"):
    """Mark one stage and the products after it stale without deleting them.

    For ``image_id`` (or every image when ``None``) the named stage and the
    image stages that follow it are marked, up to the next batch stage; every
    later batch stage is marked as well. Image stages after a batch stage are
    left alone: when that batch stage runs again it compares each image's part
    of its new result with the old one and marks only the images whose part
    changed, so the other images keep their results.
    """

    definitions = {item["name"]: item for item in _stage_definitions()}
    downstream = _downstream_names(stage_name)
    targets = [image_id] if image_id is not None else list(state["images"])
    after_batch = False
    for name in downstream:
        definition = definitions[name]
        if definition["scope"] == "batch":
            entry = state.get("batch_stages", {}).get(name)
            if entry and (
                entry.get("status") != "SKIPPED" or entry.get("blocked", False)
            ):
                entry["status"] = "STALE"
                entry["stale_reason"] = reason
                _pipeline_event(state, "STALE", name, message=reason)
            if name in _BATCH_IMAGE_PARTS:
                # Later image stages follow this stage's per-image results.
                after_batch = True
            else:
                targets = list(state["images"])
            continue
        if after_batch:
            continue
        for identifier in targets:
            entry = state["images"][identifier].get("stages", {}).get(name)
            if entry and (
                entry.get("status") != "SKIPPED" or entry.get("blocked", False)
            ):
                entry["status"] = "STALE"
                entry["review_status"] = None
                entry["stale_reason"] = reason
                _pipeline_event(state, "STALE", name, identifier, reason)
            _update_image_status(state, identifier)
    return state


def refresh_pipeline_staleness(state, context, stage_functions=None):
    """Detect changed inputs, settings, or upstream signatures after resume."""

    definitions = _stage_lookup(stage_functions)
    for image_id, image in state["images"].items():
        current_input = _input_fingerprint(image["path"])
        if current_input != image.get("input_fingerprint"):
            image["input_fingerprint"] = current_input
            mark_pipeline_stale(state, "read", image_id, "input file changed")
        for name in pipeline_stage_names():
            definition = definitions[name]
            if definition["scope"] != "image":
                continue
            entry = image.get("stages", {}).get(name)
            if not entry or entry.get("status") in {"FAIL", "STALE"}:
                continue
            if entry.get("signature") != _stage_signature(
                state, context, definition, image_id
            ):
                mark_pipeline_stale(state, name, image_id, "dependency or settings changed")
                break
    for name in pipeline_stage_names():
        definition = definitions[name]
        if definition["scope"] != "batch":
            continue
        entry = state.get("batch_stages", {}).get(name)
        if entry and entry.get("status") not in {"FAIL", "STALE"}:
            if entry.get("signature") != _stage_signature(state, context, definition):
                mark_pipeline_stale(state, name, reason="batch dependency or settings changed")
                break
    return state


def set_image_overrides(state, context, image_id, overrides, from_stage=None, save=True):
    """Persist per-image overrides and invalidate only relevant later stages.

    ``from_stage`` defaults to the first stage that reads any of the changed
    settings (:func:`first_stage_reading`).
    """

    if image_id not in state["images"]:
        raise KeyError("Unknown image: {}".format(image_id))
    current = state["images"][image_id].get("overrides", {})
    merged = merge_settings(current, overrides or {})
    state["images"][image_id]["overrides"] = _json_value(merged)
    context["images"][image_id]["settings"] = merge_settings(
        context["images"][image_id]["settings"], overrides or {}
    )
    context["images"][image_id]["record"]["settings"] = context["images"][image_id]["settings"]
    if from_stage is None:
        from_stage = first_stage_reading(overrides)
    mark_pipeline_stale(state, from_stage, image_id, "individual-image override changed")
    _pipeline_event(state, "STALE", from_stage, image_id, "override saved")
    if save:
        save_pipeline_state(state, context)
    return state, context


def _resolve_image_settings(context, image_id, overrides):
    """Re-resolve one image's settings with the documented precedence."""

    image = context["images"][image_id]
    metadata = image["record"].get("metadata") or {}
    instrument_name = context.get("instrument_name")
    if instrument_name is None:
        instrument_name = _metadata_instrument_profile(metadata)
    name = Path(image["path"]).name
    return resolve_settings(
        instrument_name=instrument_name,
        run_settings=context.get("run_settings", {}),
        filter_name=metadata.get("filter"),
        filter_settings=context.get("filter_settings"),
        image_name=name,
        image_overrides={name: overrides} if overrides else None,
    )


def set_run_overrides(state, context, overrides, from_stage=None, save=True):
    """Change run-level settings and invalidate only the affected stages.

    Use this for settings read by batch stages (for example ``catalogs`` for
    star selection) or to change a setting for every image at once.
    Per-image overrides from :func:`set_image_overrides` keep precedence.
    """

    context["run_settings"] = merge_settings(
        context.get("run_settings", {}), overrides or {}
    )
    state["run_settings"] = _json_value(context["run_settings"])
    context["settings"] = resolve_settings(
        instrument_name=context.get("instrument_name"),
        run_settings=context["run_settings"],
    )
    state["settings"] = _json_value(context["settings"])
    for image_id, image in context["images"].items():
        resolved = _resolve_image_settings(
            context, image_id, state["images"][image_id].get("overrides", {})
        )
        image["settings"] = resolved
        image["record"]["settings"] = resolved
    if from_stage is None:
        from_stage = first_stage_reading(overrides)
    mark_pipeline_stale(state, from_stage, reason="run-level override changed")
    _pipeline_event(state, "STALE", from_stage, message="run override saved")
    if save:
        save_pipeline_state(state, context)
    return state, context


def review_image(state, context, image_id, stage_name, decision, note=None,
                 parameter_overrides=None):
    """Approve or reject one image and persist the review decision."""

    if image_id not in state["images"]:
        raise KeyError("Unknown image: {}".format(image_id))
    decision = str(decision).upper()
    if decision not in {"APPROVED", "REJECTED"}:
        raise ValueError("decision must be APPROVED or REJECTED")
    entry = state["images"][image_id].get("stages", {}).get(stage_name)
    if entry is None:
        raise ValueError("{} has not run for {}".format(stage_name, image_id))
    automatic = entry.get("automatic_status", entry.get("status"))
    entry["automatic_status"] = automatic
    entry["status"] = decision
    entry["review_status"] = decision
    state["images"][image_id].setdefault("review_decisions", {})[stage_name] = {
        "decision": decision, "automatic_status": automatic,
        "time": _utc_now(), "note": note,
        "parameter_overrides": _json_value(parameter_overrides or {}),
    }
    if stage_name == "usability":
        usability = context.get("shared", {}).get("usability", {})
        for item in usability.get("decisions", []):
            if str(item.get("image_id")) != image_id:
                continue
            item["manual_status"] = "PASS" if decision == "APPROVED" else "FAIL"
            item["review_state"] = decision
            item["review_note"] = note
            item["decision_source"] = "manual"
            item["status"] = "PASS" if decision == "APPROVED" else "FAIL"
            item["use_image"] = decision == "APPROVED"
            context["images"][image_id]["record"]["decision"] = item
            break
    elif stage_name == "psf":
        product = context["images"][image_id].get("products", {}).get("psf")
        if product is not None:
            from .photometry import apply_psf_review
            context["images"][image_id]["products"]["psf"] = apply_psf_review(
                product,
                {"decision": "approve" if decision == "APPROVED" else "reject",
                 "note": note},
                context["images"][image_id]["settings"],
            )
    _pipeline_event(state, decision, stage_name, image_id, note or "manual review")
    later_names = _downstream_names(stage_name)[1:]
    if later_names:
        mark_pipeline_stale(
            state, later_names[0], image_id, "review decision changed"
        )
    if parameter_overrides:
        set_image_overrides(
            state, context, image_id, parameter_overrides, from_stage=stage_name
        )
    elif decision == "APPROVED":
        for name in later_names:
            later = state["images"][image_id].get("stages", {}).get(name)
            if later and later.get("status") == "SKIPPED":
                later["status"] = "STALE"
    else:
        for name in later_names:
            definition = next(item for item in _stage_definitions() if item["name"] == name)
            if definition["scope"] == "image":
                state["images"][image_id].setdefault("stages", {})[name] = _stage_entry(
                    "SKIPPED", None, error="image rejected at {}".format(stage_name),
                    blocked=True,
                )
    _update_image_status(state, image_id)
    try:
        from .stage_reports import write_stage_diagnostics

        write_stage_diagnostics(state, context, stage_name, [image_id], batch=False,
                                overview=False)
    except Exception as error:  # plotting must never block a review
        state.setdefault("diagnostic_errors", []).append(
            {"stage": stage_name, "time": _utc_now(), "error": repr(error)}
        )
    save_pipeline_state(state, context)
    return state, context


def skip_pipeline_stage(state, context, stage_name, image_id=None, reason="user skipped"):
    """Record an explicit skip; skipped optional stages satisfy dependencies."""

    definition = _stage_lookup()[stage_name]
    if definition["scope"] == "batch":
        state["batch_stages"][stage_name] = _stage_entry("SKIPPED", None, error=reason)
        _pipeline_event(state, "SKIPPED", stage_name, message=reason)
    else:
        identifiers = [image_id] if image_id is not None else list(state["images"])
        for identifier in identifiers:
            state["images"][identifier]["stages"][stage_name] = _stage_entry(
                "SKIPPED", None, error=reason
            )
            _pipeline_event(state, "SKIPPED", stage_name, identifier, reason)
            _update_image_status(state, identifier)
    save_pipeline_state(state, context)
    return state, context


def rerun_image(state, context, image_id, from_stage, through_stage=None,
                stage_functions=None, mode="automatic"):
    """Rerun one image and any required batch stages from a selected point."""

    mark_pipeline_stale(state, from_stage, image_id, "individual image rerun")
    names = _downstream_names(from_stage)
    if through_stage is not None:
        if through_stage not in names:
            raise ValueError("through_stage must not precede from_stage")
        names = names[:names.index(through_stage) + 1]
    for name in names:
        definition = _stage_lookup(stage_functions)[name]
        run_pipeline_stage(
            state, context, name,
            image_id=image_id if definition["scope"] == "image" else None,
            stage_functions=stage_functions, mode=mode,
        )
    return state, context


def _working_ccd(context, image_id):
    image = context["images"][image_id]
    working = image.get("working_ccd")
    return working if working is not None else image["record"].get("ccd")


def _metadata_instrument_profile(metadata):
    """Infer a supported instrument profile from normalized metadata."""

    for name in ("instrument", "detector", "telescope", "site"):
        value = (metadata or {}).get(name)
        profile = normalize_instrument_name(value)
        if profile in {"lco", "keplercam"}:
            return profile
        text = "" if value is None else str(value).strip().lower()
        if name == "site" and text in {"lsc", "elp", "cpt", "coj", "tfn"}:
            return "lco"
        if name in {"instrument", "detector"} and text.startswith(
            ("fa", "fl", "ep", "sq")
        ):
            return "lco"
    return None


def _run_read(context, image_id, settings):
    from .image import read_fits_image

    ccd, metadata = read_fits_image(
        context["images"][image_id]["path"], settings=settings,
        target=context.get("target"),
    )
    image = context["images"][image_id]
    instrument_name = context.get("instrument_name")
    image_overrides = context["_state"]["images"][image_id].get("overrides", {})

    def resolve(profile, header_metadata):
        return resolve_settings(
            instrument_name=profile,
            run_settings=context.get("run_settings", context.get("settings", {})),
            filter_name=header_metadata.get("filter"),
            filter_settings=context.get("filter_settings"),
            image_name=Path(image["path"]).name,
            image_overrides=(
                {Path(image["path"]).name: image_overrides}
                if image_overrides else None
            ),
        )

    if instrument_name is None:
        instrument_name = _metadata_instrument_profile(metadata)
        if instrument_name is not None:
            # The profile was only known after the first read; read again with
            # it so its keyword aliases and fallback values (saturation, gain,
            # read noise) apply exactly as when the instrument is given.
            ccd, metadata = read_fits_image(
                image["path"], settings=resolve(instrument_name, metadata),
                target=context.get("target"),
            )
    resolved = resolve(instrument_name, metadata)
    image["settings"] = resolved
    image["working_ccd"] = ccd
    image["record"].update({
        "ccd": ccd,
        "metadata": metadata,
        "shape": ccd.shape,
        "settings": resolved,
    })
    return {"ccd": ccd, "metadata": metadata,
            "status": metadata.get("metadata_status", "PASS")}


def _run_region(context, image_id, settings):
    from .image import define_processing_region

    image = context["images"][image_id]
    working, region, diagnostics = define_processing_region(
        _working_ccd(context, image_id), image["record"].get("metadata"), settings,
        context.get("target"),
    )
    image["working_ccd"] = working
    image["record"].update({"ccd": working, "region": region, "shape": working.shape})
    return {"region": region, "diagnostics": diagnostics,
            "status": "WARN" if region.get("region_flags") else "PASS"}


def _run_masks(context, image_id, settings):
    from .image import build_masks

    image = context["images"][image_id]
    working, components, info = build_masks(
        _working_ccd(context, image_id), image["record"].get("metadata"), settings,
        context.get("target"),
    )
    image["working_ccd"] = working
    image["record"].update({"ccd": working, "masks": components, "mask_info": info})
    return {"components": components, "info": info,
            "status": "WARN" if info.get("flags") else "PASS"}


def _run_cosmic_rays(context, image_id, settings):
    from .image import apply_cosmic_rays

    image = context["images"][image_id]
    working, products, info = apply_cosmic_rays(
        _working_ccd(context, image_id), image["record"].get("metadata"), settings,
        context.get("target"),
    )
    image["working_ccd"] = working
    image["record"].update({"ccd": working, "cosmic_ray_products": products,
                            "cosmic_ray_info": info})
    status = "SKIPPED" if info.get("skipped") else "WARN" if info.get("flags") else "PASS"
    return {"products": products, "info": info, "status": status}


def _run_fringe(context, image_id, settings):
    from .image import correct_fringe

    image = context["images"][image_id]
    region = image["record"].get("region", {})
    products_before = image["products"].get("region", {})
    crop_slices = products_before.get("diagnostics", {}).get("crop_slices")
    working, products, info = correct_fringe(
        _working_ccd(context, image_id), image["record"].get("metadata"), settings,
        crop_slices=crop_slices,
        source_mask=image["record"].get("masks", {}).get("combined"),
        target=context.get("target"),
    )
    image["working_ccd"] = working
    image["record"].update({"ccd": working, "fringe_products": products,
                            "fringe_info": info})
    status = "SKIPPED" if info.get("skipped") else "WARN" if info.get("flags") else "PASS"
    return {"products": products, "info": info, "region": region, "status": status}


def _run_background(context, image_id, settings):
    from .image import model_background

    image = context["images"][image_id]
    working, products, info = model_background(
        _working_ccd(context, image_id), image["record"].get("metadata"), settings,
        context.get("target"),
    )
    image["working_ccd"] = working
    image["record"].update({"ccd": working, "background_products": products,
                            "background_info": info})
    status = "SKIPPED" if info.get("skipped") else "WARN" if info.get("flags") else "PASS"
    return {"products": products, "info": info, "status": status}


def _run_source_quality(context, image_id, settings):
    from .image import detect_sources_and_measure_quality

    image = context["images"][image_id]
    sources, segmentation, info = detect_sources_and_measure_quality(
        _working_ccd(context, image_id), image["record"].get("metadata"), settings,
        context.get("target"), image["record"].get("masks"),
        image["record"].get("background_products"),
    )
    image["record"].update({"sources": sources, "segmentation": segmentation,
                            "quality": info})
    return {"sources": sources, "segmentation": segmentation, "info": info,
            "status": info.get("quality_status", "PASS")}


def _run_astrometry(context, image_id, settings):
    from .catalogs import solve_astrometry

    image = context["images"][image_id]
    record = image["record"]
    supplied = context.get("shared", {}).get("catalog")
    catalog, matches, refined_wcs, info = solve_astrometry(
        _working_ccd(context, image_id), record.get("sources"),
        record.get("metadata"), settings, catalog=supplied,
        object_name=record.get("metadata", {}).get("object"),
        target=context.get("target"),
        plate_solver=context.get("shared", {}).get("plate_solver"),
    )
    record.update({"catalog": catalog, "matches": matches,
                   "wcs": refined_wcs, "astrometry": info})
    return {"catalog": catalog, "matches": matches, "wcs": refined_wcs,
            "info": info, "status": info.get("quality_status", "PASS")}


def _records_for_stage(context, required_stage=None):
    state = context.get("_state")
    identifiers = (
        list(context["images"]) if state is None
        else _active_image_ids(state, required_stage)
    )
    return [context["images"][image_id]["record"] for image_id in identifiers]


def _run_star_selection(context, image_id, settings):
    from .catalogs import (
        attach_photometric_references,
        build_master_source_table,
        select_comparison_and_psf_stars,
    )

    records = _records_for_stage(context, "astrometry")
    shared = context.setdefault("shared", {})
    _progress.progress("photometric reference catalogs for {} images".format(len(records)))
    references, reference_info = attach_photometric_references(
        records, settings, target=context.get("target"),
        supplied=shared.get("calibration_catalogs"),
    )
    shared["photometric_references"] = references
    shared["photometric_reference_info"] = reference_info
    _progress.progress("matching detections into one source table")
    master, measurements = build_master_source_table(records, settings)
    overrides = context.get("shared", {}).get("star_overrides")
    _progress.progress("choosing zeropoint, PSF, ensemble and astrometry stars "
                       "({} sources)".format(len(master)))
    # Each image is screened with its own settings (per-image overrides).
    image_settings = {
        str(record.get("image_id")): record.get("settings") or settings
        for record in records
    }
    master, measurements, summaries = select_comparison_and_psf_stars(
        master, measurements, settings, overrides, image_settings=image_settings
    )
    flagged = any(item.get("flags") for item in summaries) or any(
        entry.get("error") or not entry.get("matched")
        for entry in reference_info.get("catalogs", {}).values()
    )
    return {"master": master, "measurements": measurements, "summaries": summaries,
            "photometric_references": reference_info,
            "status": "WARN" if flagged else "PASS"}


def _run_usability(context, image_id, settings):
    from .image import assess_image_quality_batch, assess_image_usability

    selection = context["shared"]["star_selection"]
    records = _records_for_stage(context, "astrometry")
    manual = context.get("shared", {}).get("usability_decisions")
    # Compare each image's seeing, shape and sky with the rest of the batch
    # (same filter where possible). The checks go into copies of the quality
    # results, so rerunning this stage never stacks them twice.
    _progress.progress("comparing seeing, shape and sky across {} images".format(len(records)))
    batch_quality = assess_image_quality_batch(
        [record.get("quality") or {} for record in records],
        settings,
        groups=[(record.get("metadata") or {}).get("filter") for record in records],
        exposure_times=[
            (record.get("metadata") or {}).get("exposure_time") for record in records
        ],
    )
    batch_records = [
        dict(record, quality=quality) for record, quality in zip(records, batch_quality)
    ]
    _progress.progress("quick zeropoints, depth and transparency")
    decisions, residuals = assess_image_usability(
        batch_records, selection["measurements"], settings, manual
    )
    for decision, quality in zip(decisions, batch_quality):
        decision["batch_reference"] = quality.get("batch_reference")
    lookup = {str(item["image_id"]): item for item in decisions}
    for identifier, image in context["images"].items():
        if identifier in lookup:
            image["record"]["decision"] = lookup[identifier]
    status = "FAIL" if decisions and all(item["status"] == "FAIL" for item in decisions) else (
        "WARN" if any(item["status"] != "PASS" for item in decisions) else "PASS"
    )
    return {"decisions": decisions, "star_residuals": residuals, "status": status}


def _run_alignment(context, image_id, settings):
    from .alignment import (
        build_alignment_check,
        build_detection_stacks,
        determine_fixed_target_position,
        refine_relative_alignment,
        select_alignment_check_sources,
        validate_fixed_target_projection,
    )

    # Start from the astrometry WCS of every image: a previous alignment
    # replaced record["wcs"] with the aligned one, and aligning that again
    # would add the correction twice.
    for image in context["images"].values():
        astrometry = (image.get("products") or {}).get("astrometry") or {}
        if astrometry.get("wcs") is not None:
            image["record"]["wcs"] = astrometry["wcs"]
    records = _records_for_stage(context, "usability")
    selection = context["shared"]["star_selection"]
    decisions = context["shared"]["usability"]["decisions"]
    _progress.progress("aligning {} images to a common reference".format(len(records)))
    alignments, residuals = refine_relative_alignment(
        records, selection["measurements"], decisions, settings
    )
    _progress.progress("detection stacks")
    stacks = build_detection_stacks(records, alignments, decisions, settings)
    _progress.progress("fixed target position")
    solution, candidates = determine_fixed_target_position(
        records, alignments, stacks, decisions, settings, prior=context.get("target")
    )
    projections = validate_fixed_target_projection(records, alignments, solution, settings)
    alignment_lookup = {str(item["image_id"]): item for item in alignments}
    for identifier, image in context["images"].items():
        if identifier in alignment_lookup:
            image["record"]["alignment"] = alignment_lookup[identifier]
            image["record"]["wcs"] = alignment_lookup[identifier].get("wcs")
    check = None
    try:
        _progress.progress("cutting the check sources out of every aligned image")
        sources = select_alignment_check_sources(
            records, alignments, selection["measurements"], selection.get("master"),
            solution, settings,
        )
        check = build_alignment_check(records, alignments, sources, settings)
    except Exception as error:  # the visual check must never stop the run
        check = {"error": "{}: {}".format(type(error).__name__, error)}
    status = solution.get("status", "PASS")
    return {"alignments": alignments, "residuals": residuals, "stacks": stacks,
            "target_solution": solution, "target_candidates": candidates,
            "projections": projections, "alignment_check": check, "status": status}


def _run_psf(context, image_id, settings):
    from .photometry import construct_psf

    record = _record_for(context, image_id)
    measurements = context["shared"]["star_selection"]["measurements"]
    result = construct_psf(record, measurements, settings)
    return result


def _run_science_photometry(context, image_id, settings):
    from .photometry import perform_science_image_photometry

    record = _record_for(context, image_id)
    target = context["shared"]["alignment"]["target_solution"]
    psf = context["images"][image_id]["products"]["psf"]
    measurements = context["shared"]["star_selection"]["measurements"]
    alignment = record.get("alignment", {})
    result = perform_science_image_photometry(
        record, target, psf, measurements, settings, alignment.get("wcs")
    )
    result["status"] = "WARN" if result.get("target_flags") else "PASS"
    return result


def _combined_science(context):
    tables, results = [], []
    for image in context["images"].values():
        result = image.get("products", {}).get("science_photometry")
        if result is not None:
            results.append(result)
            if len(result.get("measurements", [])):
                tables.append(result["measurements"])
    return (
        vstack(tables, metadata_conflicts="silent") if tables else Table(masked=True),
        results,
    )


def _catalog_collection_from_context(context):
    catalogs = {}
    for image in context["images"].values():
        catalog = image["record"].get("catalog")
        if catalog is not None:
            name = str(catalog.meta.get("catalog_name", "user"))
            catalogs.setdefault(name, catalog)
    catalogs.update(context.get("shared", {}).get("photometric_references") or {})
    supplied = context.get("shared", {}).get("calibration_catalogs")
    if supplied:
        catalogs.update(supplied)
    return catalogs


def _run_calibration(context, image_id, settings):
    from .photometry import calibrate_photometry

    measurements, science_results = _combined_science(context)
    records = _records_for_stage(context, "science_photometry")
    psfs = [image["products"]["psf"] for image in context["images"].values()
            if "psf" in image.get("products", {})]
    return calibrate_photometry(
        measurements, _catalog_collection_from_context(context), records,
        science_results, psfs, settings
    )


def _run_templates(context, image_id, settings):
    from .subtraction import acquire_template, template_footprint_coverage

    if not settings.get("subtraction", {}).get("enabled", False):
        return {"templates": {}, "status": "SKIPPED", "skipped": "disabled"}
    supplied = context.get("shared", {}).get("template_inputs")
    if isinstance(supplied, Mapping) and supplied and "data" not in supplied:
        return {"templates": dict(supplied), "status": "PASS"}
    records = _records_for_stage(context, "science_photometry")
    filters = sorted({str(record.get("metadata", {}).get("filter")) for record in records})
    cache = Path(settings.get("subtraction", {}).get("cache_directory") or "templates").expanduser()
    if not cache.is_absolute() and context.get("_state", {}).get("run_directory"):
        # Downloaded templates live with the run unless an absolute folder is set.
        settings = merge_settings(settings, {"subtraction": {
            "cache_directory": str(Path(context["_state"]["run_directory"]) / cache)}})
    templates = {}
    status, flags = "PASS", []
    minimum = float(settings.get("subtraction", {}).get("minimum_footprint_coverage", 0.999))
    for filter_name in filters:
        _progress.progress("template for filter {}".format(filter_name))
        templates[filter_name] = acquire_template(
            records, filter_name, settings, template_paths=supplied,
            downloader=context.get("shared", {}).get("template_downloader"),
        )
        # Where the images of this filter fall on its template (and whether
        # they land on real template data everywhere).
        footprints = template_footprint_coverage(
            templates[filter_name],
            [record for record in records
             if str(record.get("metadata", {}).get("filter")) == filter_name])
        templates[filter_name]["science_footprints"] = footprints
        for item in footprints:
            coverage = item.get("coverage_fraction")
            if coverage is not None and coverage < minimum:
                status = "WARN"
                if "TEMPLATE_FOOTPRINT_INCOMPLETE" not in flags:
                    flags.append("TEMPLATE_FOOTPRINT_INCOMPLETE")
    return {"templates": templates, "status": status, "flags": flags}


def _run_subtraction(context, image_id, settings):
    from .subtraction import perform_image_subtraction

    if not settings.get("subtraction", {}).get("enabled", False):
        return {"image_id": image_id, "status": "SKIPPED", "skipped": "disabled"}
    record = _record_for(context, image_id)
    templates = context["shared"]["templates"]["templates"]
    filter_name = str(record.get("metadata", {}).get("filter"))
    template = templates.get(filter_name, templates.get("default"))
    if template is None:
        raise ValueError("No template is available for filter {}".format(filter_name))
    shared = context.get("shared", {})
    # Stars used to check the subtraction (residuals, scale, template seeing):
    # the star-selection table (x, y on each science grid, with roles) unless
    # the caller supplied its own.
    quality_stars = shared.get("quality_stars")
    if quality_stars is None:
        quality_stars = (shared.get("star_selection") or {}).get("measurements")
    return perform_image_subtraction(
        record, template, settings, quality_stars, shared.get("pyzogy_runner"),
    )


def _run_difference_photometry(context, image_id, settings):
    from .photometry import perform_difference_image_photometry

    subtraction = context["images"][image_id]["products"].get("subtraction", {})
    if subtraction.get("status") == "SKIPPED":
        return {"image_id": image_id, "status": "SKIPPED", "measurements": Table(masked=True)}
    calibration = context["shared"]["calibration"]
    return perform_difference_image_photometry(
        _record_for(context, image_id), subtraction,
        context["shared"]["alignment"]["target_solution"],
        context["images"][image_id]["products"]["psf"],
        context["shared"]["star_selection"]["measurements"],
        context["images"][image_id]["products"].get("science_photometry"),
        calibration.get("zeropoints"), settings,
    )


def _run_batch_consistency(context, image_id, settings):
    science, _ = _combined_science(context)
    differences = [image["products"]["difference_photometry"]
                   for image in context["images"].values()
                   if "difference_photometry" in image.get("products", {})]
    calibration = context["shared"].get("calibration", {})
    return analyze_batch_consistency(
        calibration.get("measurements", science), differences,
        _records_for_stage(context, "difference_photometry"), calibration.get("zeropoints"),
        calibration.get("limits"), settings,
    )


def _output_derivatives(context):
    values = {}
    for image_id, image in context["images"].items():
        record = image["record"]
        products = {}
        background = record.get("background_products", {})
        products.update({
            "background_model": background.get("background"),
            "background_rms": background.get("background_rms"),
            "background_subtracted": background.get("background_subtracted"),
            "source_mask": record.get("masks", {}).get("combined"),
            "cosmic_ray_mask": record.get("cosmic_ray_products", {}).get("cosmic_mask"),
        })
        psf = image.get("products", {}).get("psf", {})
        products.update({"psf_model": psf.get("model"), "psf_cutouts": psf.get("cutouts"),
                         "psf_residuals": psf.get("residuals")})
        subtraction = image.get("products", {}).get("subtraction", {})
        products.update({"difference_image": subtraction.get("difference"),
                         "aligned_template": subtraction.get("aligned_template", {}).get("data")})
        values[image_id] = {name: value for name, value in products.items() if value is not None}
    return values


_WCS_KEY_PREFIXES = ("CRPIX", "CRVAL", "CTYPE", "CUNIT", "CDELT", "CROTA", "CD1_", "CD2_",
                     "PC1_", "PC2_", "PV1_", "PV2_", "A_", "B_", "AP_", "BP_")
_WCS_KEYS = {"WCSAXES", "LONPOLE", "LATPOLE", "RADESYS", "RADECSYS", "EQUINOX", "EPOCH",
             "MJDREF", "WCSNAME", "A_ORDER", "B_ORDER", "AP_ORDER", "BP_ORDER", "IMAGEW",
             "IMAGEH"}


def _processed_header(ccd, wcs, metadata, extra):
    """Original header with the aligned WCS and redphot processing keywords."""

    from astropy.io import fits

    meta = getattr(ccd, "meta", None)
    header = meta.copy() if isinstance(meta, fits.Header) else fits.Header()
    for key in list(header.keys()):
        if key in _WCS_KEYS or key.startswith(_WCS_KEY_PREFIXES) or key in {
                "DATASEC", "TRIMSEC", "BIASSEC", "CCDSEC", "DETSEC", "BSCALE", "BZERO",
                "BLANK", "SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "EXTEND"}:
            del header[key]
    if wcs is not None:
        header.update(wcs.to_header(relax=True))
    if metadata.get("filter"):
        header["FILTER"] = (str(metadata["filter"]), "normalized filter (redphot)")
    if metadata.get("mjd_mid") is not None:
        header["MJD-MID"] = (float(metadata["mjd_mid"]), "mid-exposure MJD (redphot)")
    for key, value, comment in extra:
        if value is None:
            continue
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            continue
        header[key] = (value, comment)
    return header


def _processed_image_products(context, settings):
    """Per-image processed arrays, headers and mask components for the outputs.

    The processed image is the final cleaned image: the working image after
    the background stage (cut to the usable area, masked, cosmic rays and
    fringes handled, background-subtracted) on its native pixels, with the
    WCS refined by astrometry and aligned to the reference image.
    """

    state = context["_state"]
    shared = context.get("shared", {})
    alignment = shared.get("alignment") or {}
    alignments = {str(item.get("image_id")): item for item in alignment.get("alignments") or []}
    zeropoints = (shared.get("calibration") or {}).get("zeropoints")
    items = []
    for number, (image_id, image) in enumerate(context["images"].items(), 1):
        if image_id not in state["images"]:
            continue
        record = image["record"]
        ccd = (image.get("stage_ccd") or {}).get("background")
        if ccd is None:
            continue
        _progress.progress("processed image {}/{}: {}".format(
            number, len(context["images"]), image_id))
        metadata = record.get("metadata") or {}
        align = alignments.get(image_id) or {}
        wcs = align.get("wcs") or record.get("wcs") or getattr(ccd, "wcs", None)
        wcs = getattr(wcs, "celestial", wcs)
        background = record.get("background_products") or {}
        quality = record.get("quality") or {}
        astrometry = record.get("astrometry") or {}
        region = record.get("region") or {}
        crop = region.get("crop") or {}
        masks = dict(record.get("masks") or {})
        cosmic = (record.get("cosmic_ray_products") or {}).get("cosmic_mask")
        if cosmic is not None:
            masks["cosmic_rays"] = cosmic
        zeropoint = zeropoint_error = zeropoint_method = catalog = None
        if zeropoints is not None and len(zeropoints):
            rows = zeropoints[np.asarray(zeropoints["image_id"], dtype=str) == image_id]
            for method in ("psf", "large_aperture", "small_aperture"):
                chosen = rows[np.asarray(rows["method"], dtype=str) == method] if len(rows) else rows
                if len(chosen):
                    zeropoint = _finite_float(chosen[0]["zeropoint_mag"])
                    if "zeropoint_uncertainty_mag" in chosen.colnames:
                        zeropoint_error = _finite_float(chosen[0]["zeropoint_uncertainty_mag"])
                    catalog = str(chosen[0]["catalog_name"]) if "catalog_name" in chosen.colnames else None
                    zeropoint_method = method
                    break
        slices = crop.get("slices")
        section = None
        if slices:
            (y0, y1), (x0, x1) = slices
            section = "[{}:{},{}:{}]".format(x0 + 1, x1, y0 + 1, y1)
        model = background.get("background")
        rms = background.get("background_rms")
        extra = [
            ("RDPSTAT", str(state["images"][image_id].get("status")), "redphot image status"),
            ("RDPCUT", section, "part of the input array kept (FITS section)"),
            ("BKGSUB", bool((record.get("background_info") or {}).get("subtracted")),
             "broad background subtracted"),
            ("BKGLEVEL", None if model is None else float(np.nanmedian(model)),
             "median subtracted background [ADU]"),
            ("BKGRMS", None if rms is None else float(np.nanmedian(rms)), "median sky RMS"),
            ("FRINGE", bool((record.get("fringe_info") or {}).get("applied")),
             "fringe correction applied"),
            ("FWHM_PX", _finite_float(quality.get("fwhm_pixels")), "seeing FWHM [pixel]"),
            ("FWHM_AS", _finite_float(quality.get("fwhm_arcsec")), "seeing FWHM [arcsec]"),
            ("ASTRMS", _finite_float(astrometry.get("refined_rms_arcsec")),
             "astrometric rms vs catalog [arcsec]"),
            ("ALIGNREF", None if not align else str(align.get("reference_image_id"))[:68],
             "relative-alignment reference image"),
            ("ALIGNRMS", _finite_float(align.get("refined_rms_arcsec")),
             "relative-alignment rms [arcsec]"),
            ("ZP", zeropoint, "zeropoint: mag = -2.5 log10(ADU/s) + ZP"),
            ("ZPERR", zeropoint_error, "zeropoint uncertainty [mag]"),
            ("ZPMETHOD", zeropoint_method, "photometry method of ZP"),
            ("ZPCAT", catalog, "calibration catalog"),
        ]
        header = _processed_header(ccd, wcs, metadata, extra)
        header.add_history("redphot: cut to the usable area, masked, cosmic rays and fringe "
                           "handled as configured, background subtracted; WCS refined by "
                           "astrometry and relative alignment. Input file unchanged.")
        uncertainty = getattr(getattr(ccd, "uncertainty", None), "array", None)
        item = {
            "image_id": image_id,
            "data": np.asarray(ccd.data, dtype=float),
            "header": header,
            "mask_components": {name: value for name, value in masks.items()
                                if name in ("nonfinite", "input", "saturation", "bad_lines",
                                            "amplifier", "trails", "manual", "cosmic_rays")},
            "background": model,
            "background_rms": rms,
            "uncertainty": uncertainty,
        }
        items.append(item)
    return items


def _diagnostic_stage_figures(context, settings):
    """Ordered per-image report items, built lazily one figure at a time.

    Each item points at the PNG written when the stage ran (fast to embed) or,
    when no PNG exists, carries a callable that draws the figure on demand, so
    at most one figure is open while the reports are assembled.
    """

    from .stage_reports import BATCH_STAGES, PER_IMAGE_BATCH_STAGES, stage_figure

    if not settings.get("diagnostics", {}).get("enabled", True):
        return {}
    state = context["_state"]
    values = {}
    for image_id in context["images"]:
        items = []
        image_state = state["images"][image_id]
        for stage_name in pipeline_stage_names():
            if stage_name == "outputs":
                continue
            per_image = stage_name not in BATCH_STAGES or stage_name in PER_IMAGE_BATCH_STAGES
            entry = (
                image_state.get("stages", {}).get(stage_name)
                if per_image else state.get("batch_stages", {}).get(stage_name)
            )
            if entry is None and per_image and stage_name in BATCH_STAGES:
                entry = state.get("batch_stages", {}).get(stage_name)
            if not entry:
                continue
            status = str(entry.get("status", "COMPLETED")).upper()
            if status in ("STALE", "SKIPPED"):
                continue
            item = {"name": stage_name, "status": status, "close_after": True}
            path = entry.get("diagnostic_plot")
            if per_image and stage_name in BATCH_STAGES:
                from .stage_reports import image_file_stem, stage_directory

                candidate = stage_directory(state, stage_name) / "{}.png".format(
                    image_file_stem(image_id))
                path = str(candidate) if candidate.exists() else path
            if path and Path(path).exists():
                item["path"] = path
            else:
                item["figure"] = (
                    lambda stage=stage_name, image=(image_id if per_image else None):
                    stage_figure(state, context, stage, image)
                )
            items.append(item)
            if status == "FAIL" or stage_name == image_state.get("failed_stage"):
                break
        values[image_id] = items
    return values


def _run_outputs(context, image_id, settings):
    from .output import assemble_output_products, output_product_enabled

    state = context["_state"]
    shared = context["shared"]
    output_root = Path(state["run_directory"]) / "products"
    if output_root.exists() and not settings.get("output", {}).get("overwrite", False):
        version = int(shared.get("output_version", 1)) + 1
        shared["output_version"] = version
        output_root = Path(state["run_directory"]) / "products_v{}".format(version)
    selection = shared.get("star_selection", {})
    all_records = []
    for identifier, image in context["images"].items():
        record = image["record"]
        run_image = state["images"][identifier]
        record["status"] = run_image.get("status")
        record["failed_stage"] = run_image.get("failed_stage")
        failed_entry = run_image.get("stages", {}).get(record["failed_stage"], {})
        record["failure_error"] = failed_entry.get("error")
        record["review_decisions"] = deepcopy(run_image.get("review_decisions", {}))
        if run_image.get("review_decisions"):
            record.setdefault("decision", {})["user_decision"] = ";".join(
                "{}={}".format(stage, value.get("decision"))
                for stage, value in run_image["review_decisions"].items()
            )
        all_records.append(record)
    diagnostic_stages = (
        _diagnostic_stage_figures(context, settings)
        if output_product_enabled(settings, "image_pdfs") else {}
    )
    processed = []
    if output_product_enabled(settings, "processed_image"):
        processed = _processed_image_products(context, settings)
    return assemble_output_products(
        all_records, sources=selection.get("master"),
        batch_products=shared.get("batch_consistency"),
        diagnostic_stages=diagnostic_stages,
        derivatives=_output_derivatives(context), settings=settings,
        output_directory=output_root, run_events=state.get("events"),
        processed_images=processed,
    )


def _default_stage_functions():
    """Map each stage name to its plain orchestration function."""

    return {
        "read": _run_read,
        "region": _run_region,
        "masks": _run_masks,
        "cosmic_rays": _run_cosmic_rays,
        "fringe": _run_fringe,
        "background": _run_background,
        "source_quality": _run_source_quality,
        "astrometry": _run_astrometry,
        "star_selection": _run_star_selection,
        "usability": _run_usability,
        "alignment": _run_alignment,
        "psf": _run_psf,
        "science_photometry": _run_science_photometry,
        "calibration": _run_calibration,
        "templates": _run_templates,
        "subtraction": _run_subtraction,
        "difference_photometry": _run_difference_photometry,
        "batch_consistency": _run_batch_consistency,
        "outputs": _run_outputs,
    }


__all__ = [
    "PIPELINE_STATUSES",
    "analyze_batch_consistency",
    "apply_ensemble_corrections",
    "build_comparison_star_light_curves",
    "build_epoch_metrics",
    "build_preferred_light_curve",
    "collect_batch_measurements",
    "compare_photometry_methods",
    "first_stage_reading",
    "initialize_pipeline",
    "load_pipeline_state",
    "mark_pipeline_stale",
    "pipeline_stage_names",
    "refresh_pipeline_staleness",
    "rerun_image",
    "resume_pipeline",
    "review_image",
    "set_run_overrides",
    "run_batch",
    "run_one_image",
    "run_pipeline_stage",
    "run_pipeline_through",
    "save_batch_consistency_products",
    "save_pipeline_state",
    "set_image_overrides",
    "skip_pipeline_stage",
    "stage_reads_setting",
    "summarize_problem_groups",
]
