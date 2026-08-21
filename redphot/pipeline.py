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
    resolve_settings,
    validate_settings,
)


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
    minimum = int(configured.get("minimum_comparison_epochs", 3))
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
        reduced_chi2 = (
            float(np.sum((residuals[valid_errors] / errors[valid_errors]) ** 2) / max(1, np.count_nonzero(valid_errors) - 1))
            if np.count_nonzero(valid_errors) >= 2 else None
        )
        status = "PASS"
        reasons = []
        if len(indices) < minimum:
            status = "WARN"
            reasons.append("TOO_FEW_EPOCHS")
        if rms is not None and rms >= float(configured.get("comparison_star_rms_fail_mag", 0.15)):
            status = "FAIL"
            reasons.append("RMS_HIGH")
        elif rms is not None and rms >= float(configured.get("comparison_star_rms_warn_mag", 0.05)):
            status = "WARN" if status != "FAIL" else status
            reasons.append("RMS_WARN")
        if reduced_chi2 is not None and reduced_chi2 >= float(configured.get("comparison_star_reduced_chi2_fail", 10.0)):
            status = "FAIL"
            reasons.append("CHI2_HIGH")
        elif reduced_chi2 is not None and reduced_chi2 >= float(configured.get("comparison_star_reduced_chi2_warn", 3.0)):
            status = "WARN" if status != "FAIL" else status
            reasons.append("CHI2_WARN")
        stable = status == "PASS"
        for record_index in indices:
            records[record_index]["stable_star"] = stable
            records[record_index]["stability_status"] = status
        stability_records.append(
            {
                "source_id": key[0], "filter": key[1], "method": key[2],
                "epoch_count": len(indices), "baseline_magnitude": baseline,
                "rms_mag": rms, "robust_scatter_mag": robust,
                "reduced_chi2": reduced_chi2, "status": status,
                "stable": stable, "reasons": ";".join(reasons),
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
    fail_sigma = float(configured.get("metric_outlier_fail_sigma", 6.0))
    for metric in metrics:
        by_filter = {}
        for index, record in enumerate(records):
            value = record.get(metric)
            if value is not None:
                by_filter.setdefault(record["filter"], []).append((index, value))
        for group in by_filter.values():
            values = np.asarray([value for _, value in group])
            center = float(np.median(values))
            scatter = _robust_scatter(values)
            if scatter in {None, 0.0}:
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
    maximum_gap = float(configured.get("temporal_maximum_gap_days", 30.0))
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
            significance = (records[current]["magnitude"] - expected) / max(error, 0.01)
            records[current]["temporal_residual_mag"] = float(records[current]["magnitude"] - expected)
            records[current]["temporal_outlier_sigma"] = float(significance)
            if abs(significance) >= temporal_sigma:
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
        for row in stability if not bool(row["stable"])
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
    unstable_count = sum(not bool(row["stable"]) for row in stability)
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
    """Return the ordered built-in stage dependency schema."""

    return [
        {"name": "read", "scope": "image", "requires": [],
         "settings": ["input", "metadata", "instrument"]},
        {"name": "region", "scope": "image", "requires": ["read"],
         "settings": ["crop"]},
        {"name": "masks", "scope": "image", "requires": ["region"],
         "settings": ["masks"]},
        {"name": "cosmic_rays", "scope": "image", "requires": ["masks"],
         "settings": ["masks"]},
        {"name": "fringe", "scope": "image", "requires": ["cosmic_rays"],
         "settings": ["fringe"]},
        {"name": "background", "scope": "image", "requires": ["fringe"],
         "settings": ["background"]},
        {"name": "source_quality", "scope": "image", "requires": ["background"],
         "settings": ["source_detection", "image_quality"]},
        {"name": "astrometry", "scope": "image", "requires": ["source_quality"],
         "settings": ["astrometry", "catalogs"]},
        {"name": "star_selection", "scope": "batch", "requires": ["astrometry"],
         "settings": ["catalogs", "psf"]},
        {"name": "usability", "scope": "batch", "requires": ["star_selection"],
         "settings": ["image_quality"], "review": True},
        {"name": "alignment", "scope": "batch", "requires": ["usability"],
         "settings": ["astrometry", "target_position"]},
        {"name": "psf", "scope": "image", "requires": ["alignment"],
         "settings": ["psf"], "review": True},
        {"name": "science_photometry", "scope": "image", "requires": ["psf"],
         "settings": ["apertures", "background", "target_position"]},
        {"name": "calibration", "scope": "batch", "requires": ["science_photometry"],
         "settings": ["calibration", "upper_limits"]},
        {"name": "templates", "scope": "batch", "requires": ["calibration"],
         "settings": ["subtraction"]},
        {"name": "subtraction", "scope": "image", "requires": ["templates"],
         "settings": ["subtraction"]},
        {"name": "difference_photometry", "scope": "image", "requires": ["subtraction"],
         "settings": ["subtraction", "apertures", "upper_limits"]},
        {"name": "batch_consistency", "scope": "batch",
         "requires": ["difference_photometry"], "settings": ["batch_consistency"]},
        {"name": "outputs", "scope": "batch", "requires": ["batch_consistency"],
         "settings": ["diagnostics", "output"]},
    ]


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
):
    """Create serializable state and in-memory context for a new run."""

    from .image import discover_fits_files

    base_settings = merge_settings(get_default_settings(), settings or {})
    validate_settings(base_settings)
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
        "batch_stages": {}, "images": {}, "events": [],
    }
    context = {
        "settings": base_settings, "target": target, "images": {},
        "shared": {},
    }
    context["_state"] = state
    for image_id, path in zip(identifiers, files):
        by_image = deepcopy(
            overrides.get(image_id, overrides.get(Path(path).name, {}))
        )
        image_settings = resolve_settings(
            instrument_name=instrument_name,
            run_settings=base_settings,
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


def save_pipeline_state(state, context):
    """Atomically save readable run state and a local scientific checkpoint."""

    state["updated"] = _utc_now()
    state_path, checkpoint_path = _state_paths(state, context.get("settings"))
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_temporary = state_path.with_suffix(state_path.suffix + ".tmp")
    checkpoint_temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    state_temporary.write_text(json.dumps(_json_value(state), indent=2, sort_keys=True) + "\n")
    with checkpoint_temporary.open("wb") as handle:
        pickle.dump(context, handle, protocol=pickle.HIGHEST_PROTOCOL)
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
        context = {"settings": merge_settings(get_default_settings(), state.get("settings", {})),
                   "target": None, "shared": {}, "images": {}}
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
    context["_state"] = state
    return state, context


def _stage_signature(state, context, definition, image_id=None):
    settings = (
        context["images"][image_id]["settings"] if image_id is not None
        else context["settings"]
    )
    subset = {name: settings.get(name) for name in definition.get("settings", [])}
    dependencies = {}
    for required in definition.get("requires", []):
        if required in state.get("batch_stages", {}):
            entry = state["batch_stages"][required]
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
        payload["overrides"] = state["images"][image_id].get("overrides", {})
    return _hash_value(payload)


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
    if image.get("status") == "REJECTED":
        status = "SKIPPED"
        image["stages"][stage] = _stage_entry(
            status, signature, error="image rejected", blocked=True
        )
        _pipeline_event(state, status, stage, image_id, "image rejected")
        return status
    if failure_blocks:
        status = "SKIPPED"
        message = "blocked by failed stage {}".format(failed_stage)
        image["stages"][stage] = _stage_entry(
            status, signature, error=message, blocked=True
        )
        _pipeline_event(state, status, stage, image_id, message)
        return status
    if not ready:
        status = "SKIPPED"
        image["stages"][stage] = _stage_entry(
            status, signature, error=reason, blocked=True
        )
        _pipeline_event(state, status, stage, image_id, reason)
        _update_image_status(state, image_id)
        return status
    previous = image.get("stages", {}).get(stage, {})
    if previous.get("signature") == signature and _accepted_status(previous.get("status")):
        return previous["status"]
    runner = definition.get("runner")
    if runner is None:
        raise RuntimeError("No function is registered for stage {}".format(stage))
    try:
        result = runner(context, image_id, context["images"][image_id]["settings"])
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
    if previous.get("signature") == signature and _accepted_status(previous.get("status")):
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


def run_pipeline_stage(state, context, stage_name, image_id=None,
                       stage_functions=None, mode="automatic", save=True):
    """Run exactly one named stage for one image or the eligible batch."""

    definitions = _stage_lookup(stage_functions)
    if stage_name not in definitions:
        raise KeyError("Unknown pipeline stage: {}".format(stage_name))
    definition = definitions[stage_name]
    if definition["scope"] == "image":
        identifiers = [image_id] if image_id is not None else list(state["images"])
        for identifier in identifiers:
            if identifier not in state["images"]:
                raise KeyError("Unknown image: {}".format(identifier))
            _execute_image_stage(state, context, definition, identifier)
    else:
        if image_id is not None:
            raise ValueError("{} is a batch stage".format(stage_name))
        _execute_batch_stage(state, context, definition)
    gates = set(context["settings"].get("pipeline", {}).get("review_gates", []))
    if definition.get("review") or stage_name in gates:
        if mode == "automatic":
            _automatic_review(
                state, stage_name,
                context["settings"]["pipeline"].get("automatic_review", "approve_pass_warn"),
            )
        else:
            for identifier, image in state["images"].items():
                entry = image.get("stages", {}).get(stage_name)
                if entry and entry.get("status") not in {"APPROVED", "REJECTED"}:
                    entry["review_status"] = "PENDING"
                    image.setdefault("review_decisions", {})[stage_name] = {
                        "decision": "PENDING", "time": _utc_now(), "note": None,
                    }
    if save and context["settings"]["pipeline"].get("save_state_after_stage", True):
        save_pipeline_state(state, context)
    return state, context


def run_pipeline_through(state, context, through_stage=None, stage_functions=None,
                         mode="automatic", save=True):
    """Run the same ordered functions automatically or pause at review gates."""

    names = pipeline_stage_names()
    if through_stage is not None:
        if through_stage not in names:
            raise KeyError("Unknown pipeline stage: {}".format(through_stage))
        names = names[:names.index(through_stage) + 1]
    gates = set(context["settings"]["pipeline"].get("review_gates", []))
    for name in names:
        run_pipeline_stage(
            state, context, name, stage_functions=stage_functions, mode=mode, save=save
        )
        if (
            mode == "stepwise"
            and context["settings"]["pipeline"].get("stepwise_stop_at_review", True)
            and name in gates
            and any(
                image.get("stages", {}).get(name, {}).get("review_status") == "PENDING"
                for image in state["images"].values()
            )
        ):
            break
    return state, context


def run_one_image(path, settings=None, instrument_name=None, target=None,
                  image_overrides=None, run_directory=None, through_stage=None,
                  stage_functions=None, mode="automatic"):
    """Initialize and process one image with the standard controller."""

    state, context = initialize_pipeline(
        [path], settings, instrument_name, target, image_overrides, run_directory
    )
    return run_pipeline_through(
        state, context, through_stage, stage_functions, mode
    )


def run_batch(paths, settings=None, instrument_name=None, target=None,
              image_overrides=None, run_directory=None, through_stage=None,
              stage_functions=None, mode="automatic"):
    """Initialize and process a FITS batch while containing image failures."""

    state, context = initialize_pipeline(
        paths, settings, instrument_name, target, image_overrides, run_directory
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
    """Mark one stage and every downstream product stale without deleting it."""

    definitions = {item["name"]: item for item in _stage_definitions()}
    downstream = _downstream_names(stage_name)
    targets = [image_id] if image_id is not None else list(state["images"])
    cascade_all_images = image_id is None
    for name in downstream:
        definition = definitions[name]
        if definition["scope"] == "batch":
            cascade_all_images = True
            entry = state.get("batch_stages", {}).get(name)
            if entry and (
                entry.get("status") != "SKIPPED" or entry.get("blocked", False)
            ):
                entry["status"] = "STALE"
                entry["stale_reason"] = reason
                _pipeline_event(state, "STALE", name, message=reason)
        else:
            stage_targets = list(state["images"]) if cascade_all_images else targets
            for identifier in stage_targets:
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


def set_image_overrides(state, context, image_id, overrides, from_stage=None):
    """Persist per-image overrides and invalidate only relevant later stages."""

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
        changed_sections = set((overrides or {}).keys())
        from_stage = next(
            (item["name"] for item in _stage_definitions()
             if changed_sections.intersection(item.get("settings", []))),
            "read",
        )
    mark_pipeline_stale(state, from_stage, image_id, "individual-image override changed")
    _pipeline_event(state, "STALE", from_stage, image_id, "override saved")
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


def _run_read(context, image_id, settings):
    from .image import read_fits_image

    ccd, metadata = read_fits_image(
        context["images"][image_id]["path"], settings=settings,
        target=context.get("target"),
    )
    image = context["images"][image_id]
    image["working_ccd"] = ccd
    image["record"].update({"ccd": ccd, "metadata": metadata, "shape": ccd.shape})
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
    record.update({"catalog": catalog, "astrometry_matches": matches,
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
    from .catalogs import build_master_source_table, select_comparison_and_psf_stars

    records = _records_for_stage(context, "astrometry")
    master, measurements = build_master_source_table(records, settings)
    overrides = context.get("shared", {}).get("star_overrides")
    master, measurements, summaries = select_comparison_and_psf_stars(
        master, measurements, settings, overrides
    )
    return {"master": master, "measurements": measurements, "summaries": summaries,
            "status": "WARN" if any(item.get("flags") for item in summaries) else "PASS"}


def _run_usability(context, image_id, settings):
    from .image import assess_image_usability

    selection = context["shared"]["star_selection"]
    records = _records_for_stage(context, "astrometry")
    manual = context.get("shared", {}).get("usability_decisions")
    decisions, residuals = assess_image_usability(
        records, selection["measurements"], settings, manual
    )
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
        build_detection_stacks,
        determine_fixed_target_position,
        refine_relative_alignment,
        validate_fixed_target_projection,
    )

    records = _records_for_stage(context, "usability")
    selection = context["shared"]["star_selection"]
    decisions = context["shared"]["usability"]["decisions"]
    alignments, residuals = refine_relative_alignment(
        records, selection["measurements"], decisions, settings
    )
    stacks = build_detection_stacks(records, alignments, decisions, settings)
    solution, candidates = determine_fixed_target_position(
        records, alignments, stacks, decisions, settings, prior=context.get("target")
    )
    projections = validate_fixed_target_projection(records, alignments, solution, settings)
    alignment_lookup = {str(item["image_id"]): item for item in alignments}
    for identifier, image in context["images"].items():
        if identifier in alignment_lookup:
            image["record"]["alignment"] = alignment_lookup[identifier]
            image["record"]["wcs"] = alignment_lookup[identifier].get("wcs")
    status = solution.get("status", "PASS")
    return {"alignments": alignments, "residuals": residuals, "stacks": stacks,
            "target_solution": solution, "target_candidates": candidates,
            "projections": projections, "status": status}


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
    from .subtraction import acquire_template

    if not settings.get("subtraction", {}).get("enabled", False):
        return {"templates": {}, "status": "SKIPPED", "skipped": "disabled"}
    supplied = context.get("shared", {}).get("template_inputs")
    if isinstance(supplied, Mapping) and supplied and "data" not in supplied:
        return {"templates": dict(supplied), "status": "PASS"}
    records = _records_for_stage(context, "science_photometry")
    filters = sorted({str(record.get("metadata", {}).get("filter")) for record in records})
    templates = {}
    for filter_name in filters:
        templates[filter_name] = acquire_template(
            records, filter_name, settings, template_paths=supplied,
            downloader=context.get("shared", {}).get("template_downloader"),
        )
    return {"templates": templates, "status": "PASS"}


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
    return perform_image_subtraction(
        record, template, settings,
        context.get("shared", {}).get("quality_stars"),
        context.get("shared", {}).get("pyzogy_runner"),
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


def _run_outputs(context, image_id, settings):
    from .output import assemble_output_products

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
        record["review_decisions"] = deepcopy(run_image.get("review_decisions", {}))
        if run_image.get("review_decisions"):
            record.setdefault("decision", {})["user_decision"] = ";".join(
                "{}={}".format(stage, value.get("decision"))
                for stage, value in run_image["review_decisions"].items()
            )
        all_records.append(record)
    return assemble_output_products(
        all_records, sources=selection.get("master"),
        batch_products=shared.get("batch_consistency"),
        diagnostic_stages=shared.get("diagnostic_stages"),
        derivatives=_output_derivatives(context), settings=settings,
        output_directory=output_root, run_events=state.get("events"),
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
    "initialize_pipeline",
    "load_pipeline_state",
    "mark_pipeline_stale",
    "pipeline_stage_names",
    "refresh_pipeline_staleness",
    "rerun_image",
    "resume_pipeline",
    "review_image",
    "run_batch",
    "run_one_image",
    "run_pipeline_stage",
    "run_pipeline_through",
    "save_batch_consistency_products",
    "save_pipeline_state",
    "set_image_overrides",
    "skip_pipeline_stage",
    "summarize_problem_groups",
]
