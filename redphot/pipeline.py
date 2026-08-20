"""Batch-level consistency checks and final light-curve assembly for redphot.

The functions in this module consume the tables produced by the image,
calibration, science-photometry, and difference-photometry stages.  They never
delete input measurements.  Rejected epochs, unstable stars, and isolated
outliers remain present with explicit flags and inclusion decisions.
"""

from collections.abc import Mapping
import json
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.stats import sigma_clip
from astropy.table import MaskedColumn, Table, vstack

from .config import get_default_settings, merge_settings


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


__all__ = [
    "analyze_batch_consistency",
    "apply_ensemble_corrections",
    "build_comparison_star_light_curves",
    "build_epoch_metrics",
    "build_preferred_light_curve",
    "collect_batch_measurements",
    "compare_photometry_methods",
    "save_batch_consistency_products",
    "summarize_problem_groups",
]
