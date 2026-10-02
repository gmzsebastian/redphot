"""Per-stage diagnostic plots and summary tables for a redphot run.

Each time a stage runs, :func:`write_stage_diagnostics` writes into
``<run_directory>/diagnostics/<NN>_<stage>/``:

* ``<image>.png`` -- one figure per image for image-level stages (and for the
  batch stages that have a per-image view: star selection, usability, and
  calibration);
* ``batch.png`` -- the run-level figure of a batch stage;
* ``overview.png`` -- the stage across all images (status and key numbers);
* ``summary.csv`` -- one row per image with the status, flags, and the same
  key numbers, readable in any spreadsheet.

Plotting never changes pipeline results; a failure while plotting is
recorded on the stage entry (``diagnostic_error``) and the run continues.
"""

import csv
from pathlib import Path
import re

import numpy as np


STAGE_TITLES = {
    "read": "Read",
    "region": "Region",
    "masks": "Masks",
    "cosmic_rays": "Cosmic rays",
    "fringe": "Fringe",
    "background": "Background",
    "source_quality": "Sources and seeing",
    "astrometry": "Astrometry",
    "star_selection": "Star selection",
    "usability": "Usability",
    "alignment": "Alignment",
    "psf": "PSF",
    "science_photometry": "Science photometry",
    "calibration": "Calibration",
    "templates": "Templates",
    "subtraction": "Subtraction",
    "difference_photometry": "Difference photometry",
    "batch_consistency": "Batch consistency",
    "outputs": "Outputs",
}

BATCH_STAGES = {
    "star_selection", "usability", "alignment", "calibration", "templates",
    "batch_consistency", "outputs",
}

# Batch stages that also get one figure per image.
PER_IMAGE_BATCH_STAGES = {"star_selection", "usability", "calibration"}

CCD_ORDER = ("read", "region", "masks", "cosmic_rays", "fringe", "background")


def stage_directory(state, stage):
    """Folder holding the diagnostics of one stage, e.g. ``diagnostics/06_background``."""

    from .pipeline import pipeline_stage_names

    index = pipeline_stage_names().index(stage) + 1
    return Path(state["run_directory"]) / "diagnostics" / "{:02d}_{}".format(index, stage)


def image_file_stem(image_id):
    """Short, filesystem-safe name for an image (extension removed)."""

    name = str(image_id)
    for suffix in (".fits.fz", ".fits.gz", ".fits", ".fit", ".fz"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    return re.sub(r"[^A-Za-z0-9._+-]+", "_", name).strip("_") or "image"


def _finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _robust(values):
    values = np.asarray(values, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if values.size > 300000:
        values = values[:: int(np.ceil(values.size / 300000))]
    if not values.size:
        return None, None
    median = float(np.median(values))
    return median, float(1.4826 * np.median(np.abs(values - median)))


def _entry(state, stage, image_id=None):
    if image_id is None:
        return state.get("batch_stages", {}).get(stage)
    entry = state["images"][image_id].get("stages", {}).get(stage)
    if entry is None and stage in PER_IMAGE_BATCH_STAGES:
        # Batch stages without per-image entries (calibration) share the batch entry,
        # unless the image never reached them.
        image = state["images"][image_id]
        if image.get("status") == "REJECTED" or image.get("failed_stage"):
            return None
        return state.get("batch_stages", {}).get(stage)
    return entry


def _input_ccd(context, image_id, stage):
    """Working image that entered ``stage`` (falls back to the current one)."""

    image = context["images"][image_id]
    outputs = image.get("stage_ccd") or {}
    if stage in CCD_ORDER:
        earlier = CCD_ORDER[: CCD_ORDER.index(stage)]
    else:
        earlier = CCD_ORDER
    for name in reversed(earlier):
        if outputs.get(name) is not None:
            return outputs[name]
    return image.get("working_ccd") or image["record"].get("ccd")


def _output_ccd(context, image_id, stage):
    image = context["images"][image_id]
    outputs = image.get("stage_ccd") or {}
    if outputs.get(stage) is not None:
        return outputs[stage]
    return _input_ccd(context, image_id, stage)


def _shared(context, stage):
    return (context.get("shared") or {}).get(stage) or {}


def _decision(context, image_id):
    for item in _shared(context, "usability").get("decisions", []) or []:
        if str(item.get("image_id")) == str(image_id):
            return item
    return None


def _star_summary(context, image_id):
    for item in _shared(context, "star_selection").get("summaries", []) or []:
        if str(item.get("image_id")) == str(image_id):
            return item
    return None


def _table_rows(table, column, value):
    if table is None or not len(table) or column not in table.colnames:
        return None
    return table[np.asarray([str(item) for item in table[column]]) == str(value)]


# ---------------------------------------------------------------------------
# Metrics for summary.csv and overview.png
# ---------------------------------------------------------------------------

def metric_specs(stage, settings):
    """Key numbers shown per image in the overview, with their limits."""

    quality = settings.get("image_quality", {})
    astrometry = settings.get("astrometry", {})
    specs = {
        "read": [
            {"key": "sky_median", "label": "Sky level", "spec": "{:.0f}"},
            {"key": "airmass", "label": "Airmass", "spec": "{:.2f}"},
            {"key": "finite_percent", "label": "Finite pixels", "unit": "%", "spec": "{:.2f}"},
        ],
        "region": [
            {"key": "usable_percent", "label": "Usable frame", "unit": "%", "spec": "{:.1f}"},
            {"key": "extra_edge_lines", "label": "Extra edge lines trimmed", "spec": "{:.0f}"},
            {"key": "target_edge_px", "label": "Target to edge", "unit": "px", "spec": "{:.0f}"},
        ],
        "masks": [
            {"key": "masked_percent", "label": "Masked", "unit": "%", "spec": "{:.2f}"},
            {"key": "saturation_percent", "label": "Saturation mask", "unit": "%", "spec": "{:.2f}"},
            {"key": "bad_lines", "label": "Bad rows + columns", "spec": "{:.0f}"},
            {"key": "trails", "label": "Trails", "spec": "{:.0f}"},
        ],
        "cosmic_rays": [
            {"key": "cosmic_percent", "label": "Cosmic-ray pixels", "unit": "%", "spec": "{:.3f}"},
        ],
        "fringe": [{"key": "fringe_scale", "label": "Fringe scale", "spec": "{:.3g}"}],
        "background": [
            {"key": "sky_level", "label": "Sky level", "spec": "{:.0f}"},
            {"key": "sky_rms", "label": "Sky RMS", "spec": "{:.1f}"},
            {"key": "residual_median_sigma", "label": "Residual median", "unit": "σ",
             "spec": "{:+.3f}", "warn": 0.1},
            {"key": "residual_width_sigma", "label": "Residual width", "unit": "σ",
             "spec": "{:.3f}"},
            {"key": "gradient_removed_percent", "label": "Gradient removed", "unit": "%",
             "spec": "{:.0f}"},
        ],
        "source_quality": [
            {"key": "fwhm_arcsec", "label": "FWHM", "unit": "arcsec", "spec": "{:.2f}",
             "warn": quality.get("fwhm_warn_arcsec"), "fail": quality.get("fwhm_fail_arcsec")},
            {"key": "fwhm_scatter_fraction", "label": "FWHM scatter / FWHM", "spec": "{:.2f}",
             "warn": quality.get("fwhm_scatter_warn_fraction"),
             "fail": quality.get("fwhm_scatter_fail_fraction")},
            {"key": "ellipticity", "label": "Ellipticity", "spec": "{:.3f}",
             "warn": quality.get("ellipticity_warn"), "fail": quality.get("ellipticity_fail")},
            {"key": "sources", "label": "Sources", "spec": "{:.0f}"},
        ],
        "astrometry": [
            {"key": "rms_arcsec", "label": "WCS RMS", "unit": "arcsec", "spec": "{:.3f}",
             "warn": astrometry.get("target_rms_arcsec"),
             "fail": astrometry.get("warning_rms_arcsec")},
            {"key": "inliers", "label": "Matched stars", "spec": "{:.0f}"},
        ],
        "star_selection": [
            {"key": "accepted", "label": "Accepted stars", "spec": "{:.0f}"},
            {"key": "calibration_stars", "label": "Calibration stars", "spec": "{:.0f}"},
            {"key": "psf_stars", "label": "PSF stars", "spec": "{:.0f}"},
        ],
        "usability": [
            {"key": "zeropoint_mag", "label": "Quick zeropoint", "unit": "mag", "spec": "{:.3f}"},
            {"key": "transparency_loss_mag", "label": "Transparency loss", "unit": "mag",
             "spec": "{:.3f}"},
            {"key": "depth_5sigma_mag", "label": "5σ depth", "unit": "mag", "spec": "{:.2f}"},
            {"key": "cloud_amplitude_mag", "label": "Cloud amplitude", "unit": "mag",
             "spec": "{:.3f}"},
        ],
        "alignment": [
            {"key": "relative_rms_arcsec", "label": "Alignment RMS", "unit": "arcsec",
             "spec": "{:.3f}"},
            {"key": "shift_px", "label": "Shift to reference", "unit": "px", "spec": "{:.2f}"},
        ],
        "psf": [
            {"key": "fwhm_px", "label": "PSF FWHM", "unit": "px", "spec": "{:.2f}"},
            {"key": "stars_used", "label": "PSF stars used", "spec": "{:.0f}"},
            {"key": "residual_percent", "label": "Median residual", "unit": "%", "spec": "{:.1f}"},
        ],
        "science_photometry": [
            {"key": "snr_psf", "label": "Target S/N (PSF)", "spec": "{:.1f}"},
            {"key": "snr_large_aperture", "label": "Target S/N (large aperture)", "spec": "{:.1f}"},
            {"key": "centroid_offset_arcsec", "label": "Free-centroid offset", "unit": "arcsec",
             "spec": "{:.2f}"},
        ],
        "calibration": [
            {"key": "zeropoint_mag", "label": "Zeropoint (PSF)", "unit": "mag", "spec": "{:.3f}"},
            {"key": "zeropoint_scatter_mag", "label": "Star scatter", "unit": "mag",
             "spec": "{:.3f}"},
            {"key": "stars", "label": "Calibration stars", "spec": "{:.0f}"},
            {"key": "limit_5sigma_mag", "label": "5σ limit", "unit": "mag", "spec": "{:.2f}"},
        ],
        "subtraction": [
            {"key": "residual_fraction", "label": "Star residual fraction", "spec": "{:.3f}"},
            {"key": "noise_ratio", "label": "Noise ratio", "spec": "{:.2f}"},
        ],
        "difference_photometry": [{"key": "snr", "label": "Difference S/N", "spec": "{:.1f}"}],
        "batch_consistency": [
            {"key": "magnitude", "label": "Preferred magnitude", "unit": "mag", "spec": "{:.3f}"},
            {"key": "magnitude_uncertainty", "label": "Uncertainty", "unit": "mag", "spec": "{:.3f}"},
        ],
    }
    return specs.get(stage, [])


def _image_metrics(stage, context, image_id):
    """Numbers summarizing one image for one stage (missing values are None)."""

    image = context["images"][image_id]
    product = image.get("products", {}).get(stage) or {}
    metadata = image["record"].get("metadata") or {}
    values = {}
    if stage == "read":
        data = getattr(product.get("ccd"), "data", None)
        values["sky_median"] = _robust(data)[0] if data is not None else None
        values["airmass"] = metadata.get("airmass")
        values["exposure_time"] = metadata.get("exposure_time")
        fraction = _finite(metadata.get("finite_fraction"))
        values["finite_percent"] = None if fraction is None else 100 * fraction
    elif stage == "region":
        region = product.get("region") or {}
        edges = region.get("empirical_edges") or {}
        fraction = _finite(region.get("valid_fraction_full"))
        values["usable_percent"] = None if fraction is None else 100 * fraction
        values["extra_edge_lines"] = sum(int(edges.get(side) or 0)
                                         for side in ("top", "bottom", "left", "right"))
        values["target_edge_px"] = region.get("target_edge_distance_pixels")
    elif stage == "masks":
        info = product.get("info") or {}
        fractions = info.get("component_fractions") or {}
        lines = info.get("bad_lines") or {}
        values["masked_percent"] = 100 * float(info.get("masked_fraction") or 0)
        values["saturation_percent"] = 100 * float(fractions.get("saturation") or 0)
        values["bad_lines"] = len(lines.get("bad_rows") or []) + len(lines.get("bad_columns") or [])
        values["trails"] = len(info.get("trail_list") or [])
    elif stage == "cosmic_rays":
        info = product.get("info") or {}
        fraction = _finite(info.get("cosmic_pixel_fraction"))
        values["cosmic_percent"] = None if fraction is None else 100 * fraction
    elif stage == "fringe":
        values["fringe_scale"] = (product.get("info") or {}).get("scale")
    elif stage == "background":
        info = product.get("info") or {}
        products = product.get("products") or {}
        values["sky_level"] = _robust(products.get("background"))[0] \
            if products.get("background") is not None else None
        values["sky_rms"] = _robust(products.get("background_rms"))[0] \
            if products.get("background_rms") is not None else None
        reduction = _finite(info.get("gradient_reduction_fraction"))
        values["gradient_removed_percent"] = None if reduction is None else 100 * reduction
        corrected, rms = products.get("background_subtracted"), products.get("background_rms")
        if corrected is not None and rms is not None:
            with np.errstate(all="ignore"):
                normalized = np.asarray(corrected, dtype=float) / np.asarray(rms, dtype=float)
            mask = products.get("background_mask")
            if mask is not None and np.shape(mask) == normalized.shape:
                normalized = normalized[~np.asarray(mask, dtype=bool)]
            values["residual_median_sigma"], values["residual_width_sigma"] = _robust(normalized)
    elif stage == "source_quality":
        info = product.get("info") or {}
        for key in ("fwhm_arcsec", "fwhm_pixels", "fwhm_scatter_fraction", "ellipticity",
                    "background", "background_rms"):
            values[key] = info.get(key)
        values["sources"] = info.get("source_count")
    elif stage == "astrometry":
        info = product.get("info") or {}
        values["rms_arcsec"] = info.get("refined_rms_arcsec") or info.get("original_rms_arcsec")
        values["inliers"] = info.get("inlier_count")
        values["matches"] = info.get("match_count")
    elif stage == "star_selection":
        summary = _star_summary(context, image_id) or {}
        roles = summary.get("role_counts") or {}
        values["candidates"] = summary.get("candidate_count")
        values["accepted"] = summary.get("strictly_accepted_count")
        values["calibration_stars"] = roles.get("calibration")
        values["psf_stars"] = roles.get("psf")
    elif stage == "usability":
        decision = _decision(context, image_id) or {}
        values["zeropoint_mag"] = decision.get("zeropoint_mag")
        values["zeropoint_scatter_mag"] = decision.get("zeropoint_scatter_mag")
        values["transparency_loss_mag"] = decision.get("transparency_attenuation_mag")
        values["cloud_amplitude_mag"] = decision.get("spatial_cloud_amplitude_mag")
        values["depth_5sigma_mag"] = (decision.get("global_depths_mag") or {}).get("5sigma")
        values["use_image"] = decision.get("use_image")
    elif stage == "alignment":
        for item in _shared(context, "alignment").get("alignments", []) or []:
            if str(item.get("image_id")) == str(image_id):
                values["relative_rms_arcsec"] = item.get("refined_rms_arcsec")
                x, y = item.get("translation_x_pixels"), item.get("translation_y_pixels")
                if _finite(x) is not None and _finite(y) is not None:
                    values["shift_px"] = float(np.hypot(x, y))
                values["common_stars"] = item.get("common_star_count")
    elif stage == "psf":
        values["fwhm_px"] = product.get("fwhm_pixels")
        values["stars_used"] = product.get("star_count_used")
        fraction = _finite(product.get("residual_median_fraction"))
        values["residual_percent"] = None if fraction is None else 100 * fraction
        values["correlation"] = product.get("correlation_median")
        values["model"] = product.get("model_type")
    elif stage == "science_photometry":
        table = product.get("measurements")
        rows = _table_rows(table, "source_type", "target")
        if rows is not None:
            for row in rows:
                values["snr_" + str(row["method"])] = _finite(row["snr"])
                values["flux_" + str(row["method"])] = _finite(row["flux"])
        free = (product.get("target_diagnostics") or {}).get("free_centroid") or {}
        values["centroid_offset_arcsec"] = free.get("offset_arcsec")
    elif stage == "calibration":
        calibration = _shared(context, "calibration")
        rows = _table_rows(calibration.get("zeropoints"), "image_id", image_id)
        if rows is not None and len(rows):
            methods = [str(value) for value in rows["method"]]
            row = rows[methods.index("psf")] if "psf" in methods else rows[0]
            values["zeropoint_mag"] = _finite(row["zeropoint_mag"])
            values["zeropoint_scatter_mag"] = _finite(row["zeropoint_scatter_mag"])
            values["stars"] = _finite(row["star_count"])
            values["catalog"] = str(row["catalog_name"])
        limits = _table_rows(calibration.get("limits"), "image_id", image_id)
        if limits is not None and len(limits):
            methods = [str(value) for value in limits["method"]]
            row = limits[methods.index("psf")] if "psf" in methods else limits[0]
            values["limit_5sigma_mag"] = _finite(row["empty_limit_5sigma_mag"]) or \
                _finite(row["analytic_limit_5sigma_mag"])
    elif stage == "subtraction":
        quality = product.get("quality") or {}
        values["residual_fraction"] = quality.get("median_residual_fraction")
        values["noise_ratio"] = quality.get("noise_ratio")
    elif stage == "difference_photometry":
        preferred = product.get("preferred_result") or {}
        values["snr"] = preferred.get("snr")
        values["classification"] = preferred.get("classification")
    elif stage == "batch_consistency":
        rows = _table_rows(_shared(context, "batch_consistency").get("preferred_light_curve"),
                           "image_id", image_id)
        if rows is not None and len(rows):
            row = rows[0]
            values["magnitude"] = _finite(row["magnitude"])
            values["magnitude_uncertainty"] = _finite(row["magnitude_uncertainty"])
            values["included"] = bool(row["included_in_final"])
            values["method"] = "{}:{}".format(row["image_kind"], row["method"])
    return values


def _flags_for(stage, context, image_id):
    image = context["images"][image_id]
    product = image.get("products", {}).get(stage) or {}
    flags = []
    for source in (product, product.get("info") if isinstance(product, dict) else None,
                   product.get("region") if isinstance(product, dict) else None):
        if not isinstance(source, dict):
            continue
        for key in ("flags", "quality_flags", "region_flags", "target_flags"):
            for flag in source.get(key) or []:
                if str(flag) not in flags:
                    flags.append(str(flag))
    if stage == "usability":
        for flag in (_decision(context, image_id) or {}).get("quality_flags") or []:
            if str(flag) not in flags:
                flags.append(str(flag))
    return flags


def stage_summary_rows(state, context, stage):
    """One summary row per image for ``stage`` (status, flags, key numbers)."""

    rows = []
    for image_id, image in state["images"].items():
        entry = _entry(state, stage, image_id) or {}
        if not entry and stage in BATCH_STAGES and image.get("status") != "REJECTED" \
                and not image.get("failed_stage"):
            entry = _entry(state, stage) or {}
        record = context["images"][image_id]["record"]
        metadata = record.get("metadata") or {}
        error = str(entry.get("error") or "").strip().splitlines()
        row = {
            "image_id": image_id,
            "label": image_file_stem(image_id),
            "filter": metadata.get("filter"),
            "mjd": metadata.get("mjd_mid", metadata.get("mjd")),
            "status": entry.get("status") or "NOT RUN",
            "review": entry.get("review_status") or "",
            "flags": ";".join(_flags_for(stage, context, image_id)) if entry else "",
            "note": error[-1][:200] if error else "",
        }
        if entry and entry.get("status") not in (None, "STALE", "SKIPPED") \
                and not entry.get("blocked"):
            try:
                row.update(_image_metrics(stage, context, image_id))
            except Exception as problem:  # summaries must never stop a run
                row["note"] = "metrics unavailable: {}".format(problem)
        rows.append(row)
    return rows


def write_stage_summary(state, context, stage):
    """Write ``summary.csv`` for one stage and return its path."""

    rows = stage_summary_rows(state, context, stage)
    directory = stage_directory(state, stage)
    directory.mkdir(parents=True, exist_ok=True)
    base = ["image_id", "filter", "mjd", "status", "review", "flags", "note"]
    extra = []
    for row in rows:
        for key in row:
            if key not in base and key != "label" and key not in extra:
                extra.append(key)
    path = directory / "summary.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=base + extra, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            clean = {}
            for key, value in row.items():
                number = _finite(value) if not isinstance(value, (bool, str)) else None
                clean[key] = "{:.6g}".format(number) if number is not None else (
                    "" if value is None else value)
            writer.writerow(clean)
    return path, rows


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _status_card(state, context, stage, image_id, entry):
    from .diagnostics import plot_stage_status, _image_subtitle

    metadata = {}
    if image_id is not None:
        metadata = context["images"][image_id]["record"].get("metadata") or {}
    reason = entry.get("error") or entry.get("stale_reason") or "No reason recorded."
    return plot_stage_status(
        STAGE_TITLES.get(stage, stage), entry.get("status"),
        image_file_stem(image_id) if image_id else "run",
        _image_subtitle(metadata) if image_id else None, reason,
    )


_ENABLE_HINTS = {
    "cosmic_rays": "Enable with run settings {'masks': {'cosmic_rays': {'enabled': True}}} "
                   "(needs astroscrappy).",
    "fringe": "Fringe correction needs a fringe map: {'fringe': {'enabled': True, 'map_path': ...}}.",
    "subtraction": "Enable with {'subtraction': {'enabled': True}} (needs a template and Hotpants).",
    "difference_photometry": "Runs only when subtraction is enabled.",
    "templates": "Runs only when subtraction is enabled.",
}


def _skipped_card(state, context, stage, image_ids):
    from .diagnostics import plot_stage_status

    reasons = {}
    for image_id in image_ids:
        product = context["images"][image_id].get("products", {}).get(stage) or {}
        reason = product.get("skipped") or (product.get("info") or {}).get("skipped") or \
            (_entry(state, stage, image_id) or {}).get("error") or "not run"
        reasons[str(reason)] = reasons.get(str(reason), 0) + 1
    text = "\n".join("{} image(s): {}".format(count, reason) for reason, count in reasons.items())
    hint = _ENABLE_HINTS.get(stage)
    if hint:
        text += "\n\n" + hint
    return plot_stage_status(STAGE_TITLES.get(stage, stage), "SKIPPED",
                             "{} images".format(len(image_ids)), None, text)


def stage_figure(state, context, stage, image_id=None):
    """Build the diagnostic figure for one stage (and image), or ``None``.

    Batch stages use ``image_id=None`` for their run-level figure; star
    selection, usability, and calibration also accept an ``image_id``.
    """

    from . import diagnostics as plots

    entry = _entry(state, stage, image_id) or {}
    status = entry.get("status")
    if not entry or status in ("SKIPPED", "STALE"):
        # Skipped stages get one shared card (see write_stage_diagnostics); stages
        # blocked by a rejection or an earlier failure get no figure of their own.
        return None
    if status == "FAIL":
        return _status_card(state, context, stage, image_id, entry)
    if image_id is not None:
        image = context["images"][image_id]
        record = image["record"]
        product = image.get("products", {}).get(stage) or {}
        metadata = record.get("metadata") or {}
        settings = image.get("settings") or context.get("settings") or {}
    else:
        product = _shared(context, stage)
        metadata, settings = {}, context.get("settings") or {}

    if stage == "read":
        return plots.plot_read_diagnostics(product.get("ccd"), product.get("metadata") or metadata,
                                           status=status)
    if stage == "region":
        return plots.plot_region_diagnostics(_input_ccd(context, image_id, stage),
                                             product.get("region"), product.get("diagnostics"),
                                             metadata, status=status)
    if stage == "masks":
        return plots.plot_mask_diagnostics(_output_ccd(context, image_id, stage),
                                           product.get("components"), product.get("info"),
                                           metadata, status=status)
    if stage == "cosmic_rays":
        return plots.plot_cosmic_ray_diagnostics(_input_ccd(context, image_id, stage),
                                                 product.get("products"), product.get("info"),
                                                 metadata, status=status)
    if stage == "fringe":
        return plots.plot_fringe_diagnostics(_input_ccd(context, image_id, stage),
                                             product.get("products"), product.get("info"),
                                             metadata, status=status)
    if stage == "background":
        return plots.plot_background_diagnostics(_input_ccd(context, image_id, stage),
                                                 product.get("products"), product.get("info"),
                                                 metadata, status=status)
    if stage == "source_quality":
        return plots.plot_image_quality_diagnostics(_output_ccd(context, image_id, "background"),
                                                    product.get("sources"), product.get("segmentation"),
                                                    product.get("info"), metadata,
                                                    settings=settings, status=status)
    if stage == "astrometry":
        return plots.plot_astrometry_diagnostics(_output_ccd(context, image_id, "background"),
                                                 product.get("catalog"), product.get("matches"),
                                                 product.get("info"), metadata,
                                                 settings=settings, status=status)
    if stage == "star_selection":
        if image_id is None:
            return None
        shared = _shared(context, stage)
        return plots.plot_star_selection_diagnostics(
            _output_ccd(context, image_id, "background"), shared.get("measurements"), image_id,
            _star_summary(context, image_id), metadata,
            reference_info=shared.get("photometric_references"), status=status)
    if stage == "usability":
        if image_id is None:
            return None
        return plots.plot_image_usability_diagnostics(
            _output_ccd(context, image_id, "background"), _decision(context, image_id) or {},
            _shared(context, stage).get("star_residuals"), metadata, status=status)
    if stage == "alignment":
        if image_id is not None:
            return None
        shared = _shared(context, stage)
        return plots.plot_alignment_target_diagnostics(
            shared.get("stacks"), shared.get("target_solution"), shared.get("target_candidates"),
            shared.get("projections"), status=status)
    if stage == "psf":
        return plots.plot_psf_diagnostics(product, metadata=metadata, status=status)
    if stage == "science_photometry":
        return plots.plot_science_photometry_diagnostics(product, metadata=metadata, status=status)
    if stage == "calibration":
        shared = _shared(context, stage)
        if image_id is None:
            return plots.plot_calibration_diagnostics(shared, status=status)
        return plots.plot_calibration_image_diagnostics(shared, image_id, metadata, status=status)
    if stage in ("subtraction", "difference_photometry", "templates"):
        if status == "SKIPPED" or (product or {}).get("skipped"):
            return None
        if stage == "subtraction":
            return plots.plot_subtraction_diagnostics(product, record, metadata=metadata,
                                                      status=status)
        if stage == "difference_photometry":
            return plots.plot_difference_photometry_diagnostics(product, metadata=metadata,
                                                                status=status)
        return None
    if stage == "batch_consistency":
        return plots.plot_batch_consistency_diagnostics(_shared(context, stage), status=status)
    return None


def _save(figure, path, dpi):
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi)
    plt.close(figure)
    return str(path)


def write_stage_diagnostics(state, context, stage, image_ids=None, batch=True, plots=True,
                            overview=True):
    """Write the plots, overview, and summary table of one stage.

    Parameters
    ----------
    image_ids : sequence of str, optional
        Images whose figures should be (re)drawn; ``None`` redraws all.
    batch : bool
        Redraw the run-level figure of a batch stage.
    plots : bool
        ``False`` rewrites only ``summary.csv`` (used after review decisions).

    Returns
    -------
    dict
        Paths written, keyed by ``summary``, ``overview``, ``batch`` and image IDs.
    """

    import matplotlib
    import matplotlib.pyplot as plt

    configured = (context.get("settings") or {}).get("diagnostics", {})
    if not configured.get("enabled", True) or not configured.get("save_stage_plots", True):
        return {}
    dpi = int(configured.get("stage_plot_dpi", 110))
    directory = stage_directory(state, stage)
    written = {}
    summary_path, rows = write_stage_summary(state, context, stage)
    written["summary"] = str(summary_path)
    if not plots:
        return written
    is_batch = stage in BATCH_STAGES
    targets = list(state["images"]) if image_ids is None else [
        image_id for image_id in image_ids if image_id in state["images"]]
    interactive = matplotlib.is_interactive()
    plt.ioff()
    try:
        if not is_batch or stage in PER_IMAGE_BATCH_STAGES:
            skipped = []
            for image_id in targets:
                entry = _entry(state, stage, image_id)
                if not entry:
                    continue
                if entry.get("status") == "SKIPPED":
                    # Disabled stages get one card for the stage; blocked ones none.
                    if not entry.get("blocked"):
                        skipped.append(image_id)
                    old = directory / "{}.png".format(image_file_stem(image_id))
                    if old.exists():
                        old.unlink()
                    entry.pop("diagnostic_plot", None)
                    continue
                path = directory / "{}.png".format(image_file_stem(image_id))
                try:
                    figure = stage_figure(state, context, stage, image_id)
                    if figure is None:
                        if path.exists():
                            path.unlink()
                        entry.pop("diagnostic_plot", None)
                        continue
                    written[image_id] = _save(figure, path, dpi)
                    if entry is not _entry(state, stage):
                        entry["diagnostic_plot"] = written[image_id]
                        entry.pop("diagnostic_error", None)
                except Exception as problem:
                    plt.close("all")
                    entry["diagnostic_error"] = "{}: {}".format(type(problem).__name__, problem)
            if skipped:
                written["skipped"] = _save(_skipped_card(state, context, stage, skipped),
                                           directory / "skipped.png", dpi)
        if is_batch:
            entry = _entry(state, stage)
            if entry and batch and stage not in ("outputs",):
                path = directory / "batch.png"
                try:
                    figure = stage_figure(state, context, stage, None)
                    if figure is None and entry.get("status") in ("FAIL", "SKIPPED"):
                        figure = _status_card(state, context, stage, None, entry)
                    if figure is not None:
                        written["batch"] = _save(figure, path, dpi)
                        entry["diagnostic_plot"] = written["batch"]
                except Exception as problem:
                    plt.close("all")
                    entry["diagnostic_error"] = "{}: {}".format(type(problem).__name__, problem)
        specs = metric_specs(stage, context.get("settings") or {})
        has_numbers = any(_finite(row.get(spec["key"])) is not None
                          for spec in specs for row in rows)
        overview_path = directory / "overview.png"
        if not has_numbers and overview_path.exists():
            overview_path.unlink()
        if specs and has_numbers and overview:
            from .diagnostics import plot_stage_overview

            try:
                figure = plot_stage_overview(STAGE_TITLES.get(stage, stage), rows, specs)
                written["overview"] = _save(figure, directory / "overview.png", dpi)
            except Exception:
                plt.close("all")
    finally:
        if interactive:
            plt.ion()
    return written


__all__ = [
    "STAGE_TITLES",
    "image_file_stem",
    "metric_specs",
    "stage_directory",
    "stage_figure",
    "stage_summary_rows",
    "write_stage_diagnostics",
    "write_stage_summary",
]
