"""PSF construction and review functions for redphot photometry.

The functions operate on one image at a time and never alter science pixels.
They consume the approved PSF-star roles produced by :mod:`redphot.catalogs`,
build a normalized empirical or analytic model, and return an explicit review
decision that downstream target photometry can require.
"""

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.table import MaskedColumn, Table
from scipy import ndimage
from scipy.optimize import least_squares

from .config import get_default_settings, merge_settings


PSF_DOWNSTREAM_STAGES = (
    "psf",
    "target_photometry",
    "aperture_correction",
    "calibration",
    "subtraction",
    "upper_limits",
    "diagnostics",
)


def _finite_float(value, default=None):
    """Return a finite float or ``default`` for missing table values."""

    if value is None or np.ma.is_masked(value):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _row_value(row, name, default=None):
    """Read one optional value from an Astropy row or mapping."""

    if isinstance(row, Mapping):
        value = row.get(name, default)
    elif hasattr(row, "colnames") and name in row.colnames:
        value = row[name]
    else:
        value = default
    if np.ma.is_masked(value):
        return default
    if isinstance(value, np.generic):
        return value.item()
    return value


def _image_id(record, index=0):
    """Return the identifier used by source-selection tables."""

    if record.get("image_id") is not None:
        return str(record["image_id"])
    metadata = record.get("metadata") or {}
    return str(metadata.get("filename") or "image_{:04d}".format(index))


def _prepared_data(record):
    """Return the preferred background-subtracted array for PSF modeling."""

    for name in ("prepared_ccd", "working_ccd"):
        value = record.get(name)
        if value is not None:
            data = getattr(value, "data", value)
            if np.ndim(data) == 2:
                return np.asarray(data, dtype=float)
    products = record.get("background_products") or {}
    corrected = products.get("background_subtracted")
    if corrected is not None:
        return np.asarray(corrected, dtype=float)
    value = record.get("ccd")
    if value is not None:
        data = getattr(value, "data", value)
        if np.ndim(data) == 2:
            return np.asarray(data, dtype=float)
    raise ValueError("The image record requires a prepared CCDData or 2D array")


def _combined_mask(record, shape):
    """Combine image, processing, and non-finite masks without changing them."""

    mask = np.zeros(shape, dtype=bool)
    ccd = None
    for name in ("prepared_ccd", "working_ccd", "ccd"):
        if record.get(name) is not None:
            ccd = record[name]
            break
    if ccd is not None and getattr(ccd, "mask", None) is not None:
        mask |= np.asarray(ccd.mask, dtype=bool)
    masks = record.get("masks") or {}
    for name in ("combined", "bad_pixels", "saturation", "trails", "cosmic_rays"):
        value = masks.get(name)
        if value is not None and np.shape(value) == shape:
            mask |= np.asarray(value, dtype=bool)
    cosmic = record.get("cosmic_ray_products") or {}
    for name in ("mask", "cosmic_ray_mask"):
        value = cosmic.get(name)
        if value is not None and np.shape(value) == shape:
            mask |= np.asarray(value, dtype=bool)
    mask |= ~np.isfinite(_prepared_data(record))
    return mask


def _selected_psf_rows(measurements, image_id, settings, return_rejections=False):
    """Return approved PSF-role rows sorted by a stable quality score."""

    if measurements is None or len(measurements) == 0:
        return []
    required = {"image_id", "persistent_id", "x", "y", "role_psf"}
    if not required.issubset(set(measurements.colnames)):
        missing = sorted(required - set(measurements.colnames))
        raise ValueError("PSF measurements are missing columns: {}".format(missing))
    selected = []
    rejected = []
    psf = settings.get("psf", {})
    minimum_snr = float(psf.get("minimum_star_snr", 30.0))
    for row in measurements:
        if str(row["image_id"]) != str(image_id) or not bool(row["role_psf"]):
            continue
        if "image_accepted" in measurements.colnames and not bool(row["image_accepted"]):
            continue
        snr = _finite_float(_row_value(row, "snr"), 0.0)
        rejection_reason = None
        if snr < minimum_snr:
            rejection_reason = "LOW_SNR"
        elif psf.get("reject_saturated", True) and bool(_row_value(row, "saturated", False)):
            rejection_reason = "SATURATED"
        elif psf.get("reject_masked", True) and bool(_row_value(row, "masked", False)):
            rejection_reason = "MASKED"
        elif bool(_row_value(row, "trail_overlap", False)):
            rejection_reason = "ARTIFACT"
        reasons = str(_row_value(row, "rejection_reasons", ""))
        if rejection_reason is None and psf.get("reject_blended", True) and any(
            word in reasons.upper() for word in ("BLEND", "CROWD", "NEIGHBOR")
        ):
            rejection_reason = "BLENDED"
        if rejection_reason is not None:
            rejected.append(
                {
                    "persistent_id": str(row["persistent_id"]),
                    "x": _finite_float(row["x"]),
                    "y": _finite_float(row["y"]),
                    "snr": snr,
                    "fwhm_pixels": _finite_float(_row_value(row, "fwhm_pixels")),
                    "ellipticity": _finite_float(_row_value(row, "ellipticity")),
                    "used": False,
                    "rejection_reason": rejection_reason,
                }
            )
            continue
        ellipticity = _finite_float(_row_value(row, "ellipticity"), 0.5)
        fwhm = _finite_float(_row_value(row, "fwhm_pixels"))
        image_fwhm = _finite_float(_row_value(row, "image_fwhm_pixels"), fwhm)
        width_penalty = 0.0
        if fwhm is not None and image_fwhm not in (None, 0):
            width_penalty = abs(fwhm - image_fwhm) / image_fwhm
        score = snr / (1.0 + 5.0 * ellipticity + width_penalty)
        selected.append((float(score), row))
    selected.sort(key=lambda item: (-item[0], str(item[1]["persistent_id"])))
    maximum = int(psf.get("maximum_stars", 20))
    for _, row in selected[maximum:]:
        rejected.append(
            {
                "persistent_id": str(row["persistent_id"]),
                "x": _finite_float(row["x"]),
                "y": _finite_float(row["y"]),
                "snr": _finite_float(_row_value(row, "snr")),
                "fwhm_pixels": _finite_float(_row_value(row, "fwhm_pixels")),
                "ellipticity": _finite_float(_row_value(row, "ellipticity")),
                "used": False,
                "rejection_reason": "LOWER_RANKED",
            }
        )
    rows = [row for _, row in selected[:maximum]]
    return (rows, rejected) if return_rejections else rows


def _border_background(cutout, mask, width, sigma):
    """Measure a light local constant background from a cutout border."""

    border = np.zeros(cutout.shape, dtype=bool)
    width = max(1, min(int(width), min(cutout.shape) // 3))
    border[:width, :] = True
    border[-width:, :] = True
    border[:, :width] = True
    border[:, -width:] = True
    values = cutout[border & ~mask & np.isfinite(cutout)]
    if values.size < 8:
        return 0.0
    clipped = sigma_clip(values, sigma=float(sigma), maxiters=5, masked=True)
    finite = np.asarray(clipped.compressed(), dtype=float)
    return float(np.median(finite)) if finite.size else 0.0


def _extract_star(data, mask, row, settings):
    """Extract, center, locally flatten, and normalize one PSF-star cutout."""

    psf = settings.get("psf", {})
    size = int(psf.get("box_size_pixels", 25))
    half = size // 2
    x = float(row["x"])
    y = float(row["y"])
    ix = int(np.floor(x + 0.5))
    iy = int(np.floor(y + 0.5))
    x0, x1 = ix - half, ix + half + 1
    y0, y1 = iy - half, iy + half + 1
    if x0 < 0 or y0 < 0 or x1 > data.shape[1] or y1 > data.shape[0]:
        return None, "EDGE"
    cutout = np.array(data[y0:y1, x0:x1], dtype=float, copy=True)
    cutout_mask = np.array(mask[y0:y1, x0:x1], dtype=bool, copy=True)
    masked_fraction = float(np.count_nonzero(cutout_mask) / cutout_mask.size)
    if masked_fraction > float(psf.get("maximum_masked_fraction", 0.05)):
        return None, "MASKED"
    background = _border_background(
        cutout,
        cutout_mask,
        psf.get("local_background_border_pixels", 3),
        psf.get("sigma_clip", 3.0),
    )
    cutout -= background
    local_x = x - x0
    local_y = y - y0
    center = half
    shift = (center - local_y, center - local_x)
    centered = ndimage.shift(
        cutout, shift, order=3, mode="constant", cval=np.nan, prefilter=True
    )
    centered_mask = ndimage.shift(
        cutout_mask.astype(float), shift, order=0, mode="constant", cval=1.0
    ) > 0.5
    centered_mask |= ~np.isfinite(centered)
    fwhm = _finite_float(_row_value(row, "fwhm_pixels"), 3.0)
    radius = min(
        half - 1,
        max(2.0, float(psf.get("normalization_radius_fwhm", 2.5)) * fwhm),
    )
    yy, xx = np.indices(centered.shape, dtype=float)
    aperture = (xx - center) ** 2 + (yy - center) ** 2 <= radius ** 2
    usable = aperture & ~centered_mask
    flux = float(np.sum(centered[usable])) if np.any(usable) else np.nan
    if not np.isfinite(flux) or flux <= 0:
        return None, "NONPOSITIVE_FLUX"
    normalized = centered / flux
    normalized[centered_mask] = np.nan
    return {
        "persistent_id": str(row["persistent_id"]),
        "x": x,
        "y": y,
        "snr": _finite_float(_row_value(row, "snr")),
        "fwhm_pixels": fwhm,
        "ellipticity": _finite_float(_row_value(row, "ellipticity")),
        "background": background,
        "normalization_flux": flux,
        "masked_fraction": masked_fraction,
        "cutout": normalized,
        "valid": ~centered_mask,
    }, None


def _normalize_model(model):
    """Clip unstable negative wings and normalize a PSF image to unit sum."""

    model = np.asarray(model, dtype=float)
    model = np.where(np.isfinite(model), model, 0.0)
    model = np.maximum(model, 0.0)
    total = float(np.sum(model))
    if not np.isfinite(total) or total <= 0:
        raise ValueError("The PSF model has no positive finite normalization")
    return model / total


def _empirical_model(stars, oversampling, sigma):
    """Build a subpixel-aligned, sigma-clipped empirical effective PSF."""

    cube = np.asarray([star["cutout"] for star in stars], dtype=float)
    clipped = sigma_clip(
        np.ma.masked_invalid(cube),
        sigma=float(sigma),
        maxiters=5,
        axis=0,
        masked=True,
    )
    native = np.ma.median(clipped, axis=0).filled(np.nan)
    native = _normalize_model(native)
    if int(oversampling) > 1:
        oversampled = ndimage.zoom(native, int(oversampling), order=3, prefilter=True)
        oversampled = _normalize_model(oversampled)
    else:
        oversampled = native.copy()
    return native, oversampled, {}


def _analytic_array(shape, kind, width, beta=2.5):
    """Return a centered circular Gaussian or Moffat array."""

    yy, xx = np.indices(shape, dtype=float)
    cy = (shape[0] - 1) / 2.0
    cx = (shape[1] - 1) / 2.0
    radius2 = (xx - cx) ** 2 + (yy - cy) ** 2
    if kind == "gaussian":
        array = np.exp(-0.5 * radius2 / max(width, 1.0e-3) ** 2)
    else:
        array = (1.0 + radius2 / max(width, 1.0e-3) ** 2) ** (-beta)
    return _normalize_model(array)


def _analytic_model(stars, kind, oversampling, settings):
    """Fit a simple circular analytic model to the median normalized star."""

    psf = settings.get("psf", {})
    cube = np.asarray([star["cutout"] for star in stars], dtype=float)
    target = np.ma.median(np.ma.masked_invalid(cube), axis=0).filled(0.0)
    valid = np.isfinite(target)
    fwhm_values = [star["fwhm_pixels"] for star in stars if star["fwhm_pixels"]]
    initial_fwhm = float(np.median(fwhm_values)) if fwhm_values else 3.0
    beta0 = float(psf.get("analytic_beta", 2.5))
    if kind == "gaussian":
        width0 = max(0.4, initial_fwhm / 2.354820045)
        lower, upper = [0.2], [max(target.shape) / 2.0]

        def residual(parameters):
            return (_analytic_array(target.shape, kind, parameters[0]) - target)[valid]

        fit = least_squares(residual, [width0], bounds=(lower, upper))
        width = float(fit.x[0])
        beta = None
        fwhm_model = 2.354820045 * width
    else:
        width0 = initial_fwhm / (
            2.0 * np.sqrt(max(2.0 ** (1.0 / beta0) - 1.0, 1.0e-6))
        )
        if psf.get("fit_analytic_beta", True):
            lower, upper = [0.2, 1.1], [max(target.shape) / 2.0, 10.0]

            def residual(parameters):
                return (
                    _analytic_array(target.shape, kind, parameters[0], parameters[1])
                    - target
                )[valid]

            fit = least_squares(residual, [width0, beta0], bounds=(lower, upper))
            width, beta = (float(value) for value in fit.x)
        else:
            lower, upper = [0.2], [max(target.shape) / 2.0]

            def residual(parameters):
                return (
                    _analytic_array(target.shape, kind, parameters[0], beta0) - target
                )[valid]

            fit = least_squares(residual, [width0], bounds=(lower, upper))
            width, beta = float(fit.x[0]), beta0
        fwhm_model = 2.0 * width * np.sqrt(2.0 ** (1.0 / beta) - 1.0)
    native = _analytic_array(target.shape, kind, width, beta or beta0)
    if int(oversampling) > 1:
        oversampled = _analytic_array(
            (target.shape[0] * int(oversampling), target.shape[1] * int(oversampling)),
            kind,
            width * int(oversampling),
            beta or beta0,
        )
    else:
        oversampled = native.copy()
    parameters = {
        "width_pixels": width,
        "beta": beta,
        "model_fwhm_pixels": float(fwhm_model),
        "fit_cost": float(fit.cost),
        "fit_success": bool(fit.success),
    }
    return native, oversampled, parameters


def _build_model(stars, requested_model, empirical_minimum, oversampling, settings):
    """Choose and build the empirical model or configured analytic fallback."""

    psf = settings.get("psf", {})
    if requested_model == "empirical" and len(stars) >= empirical_minimum:
        native, oversampled, parameters = _empirical_model(
            stars, oversampling, psf.get("sigma_clip", 3.0)
        )
        model_type = "empirical_epsf"
    else:
        analytic = (
            requested_model
            if requested_model in {"moffat", "gaussian"}
            else str(psf.get("fallback_model", "moffat"))
        )
        native, oversampled, parameters = _analytic_model(
            stars, analytic, oversampling, settings
        )
        model_type = "analytic_{}".format(analytic)
    return native, oversampled, parameters, model_type


def _star_residuals(stars, model):
    """Measure scale, correlation, and fractional residuals for each star."""

    results = []
    residual_images = []
    flat_model = np.asarray(model, dtype=float)
    for star in stars:
        observed = np.asarray(star["cutout"], dtype=float)
        valid = np.isfinite(observed) & np.isfinite(flat_model)
        denominator = float(np.sum(flat_model[valid] ** 2))
        scale = (
            float(np.sum(observed[valid] * flat_model[valid]) / denominator)
            if denominator > 0
            else np.nan
        )
        residual = observed - scale * flat_model
        residual[~valid] = np.nan
        observed_norm = float(np.sum(observed[valid] ** 2))
        fraction = (
            float(np.sqrt(np.sum(residual[valid] ** 2) / observed_norm))
            if observed_norm > 0
            else np.inf
        )
        if np.count_nonzero(valid) > 2:
            correlation = float(np.corrcoef(observed[valid], flat_model[valid])[0, 1])
        else:
            correlation = np.nan
        peak = float(np.nanmax(np.abs(observed)))
        peak_fraction = float(np.nanmax(np.abs(residual)) / peak) if peak > 0 else np.inf
        results.append(
            {
                "persistent_id": star["persistent_id"],
                "fit_scale": scale,
                "residual_fraction": fraction,
                "peak_residual_fraction": peak_fraction,
                "correlation": correlation,
            }
        )
        residual_images.append(residual)
    return results, np.asarray(residual_images, dtype=float)


def _spatial_support(stars, shape, settings):
    """Measure whether selected stars support a spatially varying model."""

    psf = settings.get("psf", {})
    grid_y, grid_x = (int(value) for value in psf.get("spatial_grid", [3, 3]))
    occupied = set()
    for star in stars:
        cell_x = min(grid_x - 1, int(grid_x * star["x"] / max(1, shape[1])))
        cell_y = min(grid_y - 1, int(grid_y * star["y"] / max(1, shape[0])))
        occupied.add((cell_y, cell_x))
    supported = (
        len(stars) >= int(psf.get("minimum_spatial_stars", 20))
        and len(occupied) >= int(psf.get("minimum_spatial_cells", 6))
    )
    return supported, len(occupied)


def _psf_star_table(records):
    """Return a masked table describing every considered PSF star."""

    table = Table(masked=True)
    table["persistent_id"] = [str(row.get("persistent_id", "")) for row in records]
    table["used"] = [bool(row.get("used", False)) for row in records]
    table["rejection_reason"] = [str(row.get("rejection_reason", "")) for row in records]
    numeric = (
        "x",
        "y",
        "snr",
        "fwhm_pixels",
        "ellipticity",
        "background",
        "normalization_flux",
        "masked_fraction",
        "fit_scale",
        "residual_fraction",
        "peak_residual_fraction",
        "correlation",
    )
    for name in numeric:
        values = np.asarray(
            [_finite_float(row.get(name), np.nan) for row in records], dtype=float
        )
        invalid = ~np.isfinite(values)
        table[name] = MaskedColumn(np.where(invalid, 0.0, values), mask=invalid)
    return table


def psf_dependency_signature(image_record, measurements, settings=None):
    """Return a stable signature for PSF inputs and PSF-specific settings."""

    if settings is None:
        settings = image_record.get("settings") or get_default_settings()
    image_id = _image_id(image_record)
    selected = _selected_psf_rows(measurements, image_id, settings)
    stars = [
        {
            "id": str(row["persistent_id"]),
            "x": round(float(row["x"]), 6),
            "y": round(float(row["y"]), 6),
        }
        for row in selected
    ]
    metadata = image_record.get("metadata") or {}
    payload = {
        "image_id": image_id,
        "path": metadata.get("path"),
        "shape": list(_prepared_data(image_record).shape),
        "upstream_signature": image_record.get("upstream_signature"),
        "stars": stars,
        "psf": settings.get("psf", {}),
        "background": settings.get("background", {}),
        "fringe": settings.get("fringe", {}),
        "cosmic_rays": settings.get("masks", {}).get("cosmic_rays", {}),
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def plan_psf_rerun(existing_result, image_record, measurements, settings=None):
    """Report whether PSF inputs changed and which products become stale."""

    current = psf_dependency_signature(image_record, measurements, settings=settings)
    previous = None if existing_result is None else existing_result.get("dependency_signature")
    changed = previous != current
    return {
        "psf_changed": changed,
        "previous_signature": previous,
        "current_signature": current,
        "rerun_stages": list(PSF_DOWNSTREAM_STAGES) if changed else [],
        "preserve_upstream_stages": True,
    }


def apply_psf_review(result, manual_review=None, settings=None):
    """Apply an optional PSF approval, warning, or rejection decision."""

    if settings is None:
        settings = get_default_settings()
    psf = settings.get("psf", {})
    requested = (
        manual_review.get("decision")
        if isinstance(manual_review, Mapping)
        else manual_review
    )
    note = manual_review.get("note") if isinstance(manual_review, Mapping) else None
    aliases = {
        None: None,
        "auto": None,
        "pending": None,
        "approve": "PASS",
        "approved": "PASS",
        "pass": "PASS",
        "warn": "WARN",
        "reject": "FAIL",
        "rejected": "FAIL",
        "fail": "FAIL",
    }
    key = requested.lower() if isinstance(requested, str) else requested
    if key not in aliases:
        raise ValueError("Manual PSF decision must be auto, approve, warn, or reject")
    manual_status = aliases[key]
    result["manual_status"] = manual_status
    result["review_note"] = None if note is None else str(note)
    result["review_state"] = "REVIEWED" if manual_status is not None else "PENDING"
    result["decision_source"] = "manual" if manual_status is not None else "automatic"
    result["status"] = manual_status or result["automatic_status"]
    approved_statuses = set(psf.get("approved_statuses", ["PASS", "WARN"]))
    requires_manual = bool(psf.get("require_manual_review", False))
    result["review_required"] = bool(requires_manual and manual_status is None)
    result["approved_for_photometry"] = bool(
        result.get("model") is not None
        and result["status"] in approved_statuses
        and not result["review_required"]
    )
    if result["review_required"] and "PSF_REVIEW_REQUIRED" not in result["quality_flags"]:
        result["quality_flags"].append("PSF_REVIEW_REQUIRED")
    return result


def construct_psf(image_record, measurements, settings=None, manual_review=None):
    """Construct and validate a normalized PSF for one image.

    The function uses only rows assigned the independent ``role_psf`` role.
    Cutouts are drawn from the prepared image, locally flattened, aligned to
    their fixed detector coordinates, and normalized. Poor residual stars are
    removed iteratively. An analytic Gaussian or Moffat model is used when an
    empirical effective PSF lacks enough clean stars.

    Returns
    -------
    dict
        Model arrays, star table, residual cube, measurements, provenance,
        automatic quality decision, and effective PSF review decision.
    """

    if settings is None:
        settings = image_record.get("settings") or get_default_settings()
    else:
        settings = deepcopy(settings)
    psf = settings.get("psf", {})
    image_id = _image_id(image_record)
    signature = psf_dependency_signature(image_record, measurements, settings)
    result = {
        "image_id": image_id,
        "filename": (image_record.get("metadata") or {}).get("filename", image_id),
        "enabled": bool(psf.get("enabled", True)),
        "model": None,
        "model_native": None,
        "model_type": None,
        "model_parameters": {},
        "oversampling": int(psf.get("oversampling", 2)),
        "normalization": None,
        "fwhm_pixels": None,
        "model_fwhm_pixels": None,
        "ellipticity": None,
        "position_angle_deg": None,
        "star_count_considered": 0,
        "star_count_used": 0,
        "star_count_rejected": 0,
        "stars": Table(masked=True),
        "cutouts": np.empty((0, 0, 0)),
        "residuals": np.empty((0, 0, 0)),
        "residual_median_fraction": None,
        "residual_maximum_fraction": None,
        "correlation_median": None,
        "spatial_order_requested": int(psf.get("spatial_order", 0)),
        "spatial_order_used": 0,
        "spatial_support": False,
        "spatial_cells_occupied": 0,
        "automatic_status": "PASS",
        "status": "PASS",
        "quality_flags": [],
        "reasons": [],
        "settings_used": deepcopy(psf),
        "dependency_signature": signature,
        "model_version": int(psf.get("model_version", 1)),
        "downstream_stages": list(PSF_DOWNSTREAM_STAGES[1:]),
    }
    if not result["enabled"]:
        result["automatic_status"] = "FAIL"
        result["quality_flags"].append("PSF_MODEL_FAILED")
        result["reasons"].append("PSF construction is disabled")
        return apply_psf_review(result, manual_review, settings)

    data = _prepared_data(image_record)
    mask = _combined_mask(image_record, data.shape)
    rows, pre_rejected = _selected_psf_rows(
        measurements, image_id, settings, return_rejections=True
    )
    result["star_count_considered"] = len(rows) + len(pre_rejected)
    records = list(pre_rejected)
    stars = []
    for row in rows:
        star, reason = _extract_star(data, mask, row, settings)
        if star is None:
            records.append(
                {
                    "persistent_id": str(row["persistent_id"]),
                    "x": _finite_float(row["x"]),
                    "y": _finite_float(row["y"]),
                    "snr": _finite_float(_row_value(row, "snr")),
                    "fwhm_pixels": _finite_float(_row_value(row, "fwhm_pixels")),
                    "ellipticity": _finite_float(_row_value(row, "ellipticity")),
                    "used": False,
                    "rejection_reason": reason,
                }
            )
        else:
            star["used"] = True
            star["rejection_reason"] = ""
            records.append(star)
            stars.append(star)

    fallback_minimum = int(psf.get("minimum_fallback_stars", 1))
    empirical_minimum = int(psf.get("minimum_stars", 5))
    if len(stars) < fallback_minimum:
        result["automatic_status"] = "FAIL"
        result["quality_flags"].extend(["TOO_FEW_PSF_STARS", "PSF_MODEL_FAILED"])
        result["reasons"].append("No usable approved PSF stars remain")
        result["stars"] = _psf_star_table(records)
        result["star_count_rejected"] = len(records)
        return apply_psf_review(result, manual_review, settings)

    requested_model = str(psf.get("model", "empirical")).lower()
    maximum_iterations = max(1, int(psf.get("maximum_iterations", 3)))
    residual_limit = float(psf.get("maximum_residual_fraction", 0.20))
    correlation_limit = float(psf.get("minimum_correlation", 0.90))
    native = oversampled = parameters = residual_rows = residual_cube = None
    model_type = None
    try:
        for _ in range(maximum_iterations):
            native, oversampled, parameters, model_type = _build_model(
                stars,
                requested_model,
                empirical_minimum,
                result["oversampling"],
                settings,
            )
            residual_rows, residual_cube = _star_residuals(stars, native)
            bad = [
                (index, row)
                for index, row in enumerate(residual_rows)
                if row["residual_fraction"] > residual_limit
                or not np.isfinite(row["correlation"])
                or row["correlation"] < correlation_limit
            ]
            if not bad or len(stars) <= fallback_minimum:
                break
            worst_index, worst = max(
                bad,
                key=lambda item: (
                    item[1]["residual_fraction"],
                    -np.nan_to_num(item[1]["correlation"], nan=-1.0),
                ),
            )
            rejected = stars.pop(worst_index)
            rejected.update(worst)
            rejected["used"] = False
            rejected["rejection_reason"] = "POOR_PSF_RESIDUAL"

        final_ids = [star["persistent_id"] for star in stars]
        residual_ids = [row["persistent_id"] for row in residual_rows or []]
        if final_ids != residual_ids:
            native, oversampled, parameters, model_type = _build_model(
                stars,
                requested_model,
                empirical_minimum,
                result["oversampling"],
                settings,
            )
            residual_rows, residual_cube = _star_residuals(stars, native)
    except (FloatingPointError, RuntimeError, ValueError) as error:
        result["automatic_status"] = "FAIL"
        result["quality_flags"].append("PSF_MODEL_FAILED")
        result["reasons"].append("PSF construction failed: {}".format(error))
        result["failure_error"] = str(error)
        result["stars"] = _psf_star_table(records)
        result["star_count_rejected"] = len(records)
        return apply_psf_review(result, manual_review, settings)

    residual_by_id = {row["persistent_id"]: row for row in residual_rows or []}
    for record in records:
        residual = residual_by_id.get(record["persistent_id"])
        if residual is not None:
            record.update(residual)
        record["used"] = any(
            star["persistent_id"] == record["persistent_id"] for star in stars
        )
        if not record["used"] and not record.get("rejection_reason"):
            record["rejection_reason"] = "POOR_PSF_RESIDUAL"

    used_residuals = [
        residual_by_id[star["persistent_id"]]
        for star in stars
        if star["persistent_id"] in residual_by_id
    ]
    residual_fractions = np.asarray(
        [row["residual_fraction"] for row in used_residuals], dtype=float
    )
    correlations = np.asarray([row["correlation"] for row in used_residuals], dtype=float)
    result.update(
        {
            "model": oversampled,
            "model_native": native,
            "model_type": model_type,
            "model_parameters": parameters,
            "normalization": float(np.sum(oversampled)),
            "star_count_used": len(stars),
            "star_count_rejected": len(records) - len(stars),
            "stars": _psf_star_table(records),
            "cutouts": np.asarray([star["cutout"] for star in stars], dtype=float),
            "residuals": residual_cube,
            "residual_median_fraction": (
                float(np.nanmedian(residual_fractions)) if residual_fractions.size else None
            ),
            "residual_maximum_fraction": (
                float(np.nanmax(residual_fractions)) if residual_fractions.size else None
            ),
            "correlation_median": (
                float(np.nanmedian(correlations)) if correlations.size else None
            ),
        }
    )
    fwhm_values = np.asarray(
        [star["fwhm_pixels"] for star in stars if star["fwhm_pixels"] is not None],
        dtype=float,
    )
    ellipticities = np.asarray(
        [star["ellipticity"] for star in stars if star["ellipticity"] is not None],
        dtype=float,
    )
    result["fwhm_pixels"] = float(np.median(fwhm_values)) if fwhm_values.size else None
    result["ellipticity"] = (
        float(np.median(ellipticities)) if ellipticities.size else None
    )
    result["model_fwhm_pixels"] = parameters.get(
        "model_fwhm_pixels", result["fwhm_pixels"]
    )
    spatial, cells = _spatial_support(stars, data.shape, settings)
    result["spatial_support"] = spatial
    result["spatial_cells_occupied"] = cells

    if len(stars) < empirical_minimum:
        result["automatic_status"] = "WARN"
        result["quality_flags"].append("TOO_FEW_PSF_STARS")
        result["reasons"].append("An analytic fallback was required for a sparse field")
    median_residual = result["residual_median_fraction"]
    if median_residual is not None:
        if median_residual > float(psf.get("residual_fail_fraction", 0.30)):
            result["automatic_status"] = "FAIL"
        elif median_residual > float(psf.get("residual_warn_fraction", 0.15)):
            result["automatic_status"] = "WARN"
        if result["automatic_status"] != "PASS":
            result["quality_flags"].append("PSF_RESIDUAL_HIGH")
            result["reasons"].append("PSF-star residuals exceed the configured limit")
    if result["normalization"] is None or not np.isclose(result["normalization"], 1.0, atol=1e-6):
        result["automatic_status"] = "FAIL"
        result["quality_flags"].append("PSF_MODEL_FAILED")
        result["reasons"].append("The PSF model is not normalized")
    minimum_fwhm = float(psf.get("minimum_fwhm_pixels", 0.8))
    maximum_fwhm = float(psf.get("maximum_fwhm_pixels", 30.0))
    model_fwhm = _finite_float(result["model_fwhm_pixels"])
    if model_fwhm is not None and not minimum_fwhm <= model_fwhm <= maximum_fwhm:
        result["automatic_status"] = "FAIL"
        result["quality_flags"].append("PSF_MODEL_FAILED")
        result["reasons"].append("The fitted PSF width is outside configured limits")
    result["quality_flags"] = list(dict.fromkeys(result["quality_flags"]))
    result["reasons"] = list(dict.fromkeys(result["reasons"]))

    configured_review = psf.get("manual_decisions", {}).get(image_id)
    if manual_review is None:
        manual_review = configured_review
    return apply_psf_review(result, manual_review, settings)


def construct_psfs(image_records, measurements, settings=None, manual_reviews=None):
    """Construct independent PSF products for a sequence of image records."""

    if settings is None:
        settings = get_default_settings()
    manual_reviews = manual_reviews or {}
    results = []
    for index, record in enumerate(image_records):
        image_id = _image_id(record, index)
        image_settings = record.get("settings") or settings
        review = manual_reviews.get(image_id)
        if isinstance(review, Mapping) and review.get("parameter_overrides"):
            image_settings = merge_settings(
                image_settings, {"psf": review["parameter_overrides"]}
            )
        results.append(
            construct_psf(record, measurements, image_settings, manual_review=review)
        )
    return results


def require_approved_psf(result):
    """Raise a clear error unless the PSF review gate permits photometry."""

    if not result.get("approved_for_photometry", False):
        raise RuntimeError(
            "PSF for {} is not approved for target photometry (status={}, review={})".format(
                result.get("image_id"), result.get("status"), result.get("review_state")
            )
        )
    return True


def _safe_stem(value):
    """Return a filesystem-safe product stem."""

    name = Path(str(value)).name
    for ending in (".fits.fz", ".fits.gz", ".fits", ".fit", ".fts"):
        if name.lower().endswith(ending):
            name = name[: -len(ending)]
            break
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in name)


def _review_summary(result):
    """Return the JSON-safe part of a PSF result."""

    excluded = {"model", "model_native", "cutouts", "residuals", "stars", "settings_used"}
    return {key: value for key, value in result.items() if key not in excluded}


def save_psf_products(result, output_directory, settings=None, overwrite=None):
    """Save PSF models, cutouts, residuals, star measurements, and review state."""

    if settings is None:
        settings = get_default_settings()
    psf = settings.get("psf", {})
    if overwrite is None:
        overwrite = settings.get("output", {}).get("overwrite", False)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    stem = _safe_stem(result.get("filename") or result.get("image_id"))
    paths = {}
    header = fits.Header()
    header["PSFTYPE"] = str(result.get("model_type") or "NONE")
    header["OVERSAMP"] = int(result.get("oversampling", 1))
    header["NPSFSTAR"] = int(result.get("star_count_used", 0))
    header["PSFSTAT"] = str(result.get("status", "FAIL"))
    header["PSFREV"] = int(result.get("model_version", 1))
    if result.get("fwhm_pixels") is not None:
        header["PSFFWHM"] = float(result["fwhm_pixels"])
    if result.get("ellipticity") is not None:
        header["PSFELLIP"] = float(result["ellipticity"])
    if psf.get("save_model", True) and result.get("model") is not None:
        path = output_directory / "{}_psf.fits".format(stem)
        fits.writeto(path, np.asarray(result["model"], dtype=np.float32), header, overwrite=bool(overwrite))
        paths["model"] = str(path)
    if psf.get("save_cutouts", True) and np.size(result.get("cutouts")):
        path = output_directory / "{}_psf_cutouts.fits".format(stem)
        fits.writeto(path, np.asarray(result["cutouts"], dtype=np.float32), overwrite=bool(overwrite))
        paths["cutouts"] = str(path)
    if psf.get("save_residuals", True) and np.size(result.get("residuals")):
        path = output_directory / "{}_psf_residuals.fits".format(stem)
        fits.writeto(path, np.asarray(result["residuals"], dtype=np.float32), overwrite=bool(overwrite))
        paths["residuals"] = str(path)
    if psf.get("save_star_table", True) and len(result.get("stars", [])):
        path = output_directory / "{}_psf_stars.ecsv".format(stem)
        result["stars"].write(path, format="ascii.ecsv", overwrite=bool(overwrite))
        paths["stars"] = str(path)
    if psf.get("save_review", True):
        path = output_directory / "{}_psf_review.json".format(stem)
        if path.exists() and not overwrite:
            raise FileExistsError(str(path))
        path.write_text(json.dumps(_review_summary(result), indent=2, sort_keys=True, default=str) + "\n")
        paths["review"] = str(path)
    return paths


__all__ = [
    "PSF_DOWNSTREAM_STAGES",
    "apply_psf_review",
    "construct_psf",
    "construct_psfs",
    "plan_psf_rerun",
    "psf_dependency_signature",
    "require_approved_psf",
    "save_psf_products",
]
