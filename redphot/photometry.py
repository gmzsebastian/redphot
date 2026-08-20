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
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import StdDevUncertainty
from astropy.stats import sigma_clip
from astropy.table import MaskedColumn, Table, vstack
from astropy.wcs.utils import proj_plane_pixel_scales
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
    for name in ("mask", "cosmic_mask", "cosmic_ray_mask"):
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


def _record_wcs(record, supplied=None):
    """Return the derived image WCS used for forced coordinate projection."""

    if supplied is not None:
        return supplied
    alignment = record.get("alignment") or {}
    for value in (
        alignment.get("wcs"),
        record.get("refined_wcs"),
        getattr(record.get("prepared_ccd"), "wcs", None),
        getattr(record.get("working_ccd"), "wcs", None),
        getattr(record.get("ccd"), "wcs", None),
    ):
        if value is not None and getattr(value, "has_celestial", False):
            return value
    return None


def _pixel_scale_arcsec(record, wcs):
    """Return the image pixel scale in arcseconds per pixel when available."""

    metadata = record.get("metadata") or {}
    scale = _finite_float(metadata.get("pixel_scale"))
    if scale is not None and scale > 0:
        return scale
    if wcs is not None:
        try:
            return float(np.mean(np.abs(proj_plane_pixel_scales(wcs.celestial))) * 3600.0)
        except (AttributeError, TypeError, ValueError):
            pass
    return None


def _standard_deviation(record, data, settings):
    """Return a per-pixel standard-deviation array and its provenance."""

    ccd = record.get("prepared_ccd")
    if ccd is None:
        ccd = record.get("working_ccd")
    if ccd is None:
        ccd = record.get("ccd")
    uncertainty = None if ccd is None else getattr(ccd, "uncertainty", None)
    if uncertainty is not None:
        try:
            array = np.asarray(
                uncertainty.represent_as(StdDevUncertainty).array, dtype=float
            )
        except (AttributeError, TypeError, ValueError):
            array = None
        if array is not None and array.shape == data.shape:
            valid = np.isfinite(array) & (array > 0)
            if np.any(valid):
                replacement = float(np.median(array[valid]))
                return np.where(valid, array, replacement), "ccd_uncertainty"

    products = record.get("background_products") or {}
    rms = products.get("background_rms")
    if rms is not None and np.shape(rms) == data.shape:
        base = np.asarray(rms, dtype=float)
        source = "background_rms_map"
    else:
        quality = record.get("quality") or {}
        scalar = _finite_float(quality.get("background_rms"))
        if scalar is None or scalar <= 0:
            finite = data[np.isfinite(data)]
            scalar = float(np.std(finite)) if finite.size else 1.0
        base = np.full(data.shape, scalar, dtype=float)
        source = "background_rms_scalar"
    valid = np.isfinite(base) & (base > 0)
    replacement = float(np.median(base[valid])) if np.any(valid) else 1.0
    variance = np.where(valid, base, replacement) ** 2
    apertures = settings.get("apertures", {})
    metadata = record.get("metadata") or {}
    gain = _finite_float(metadata.get("gain"))
    if apertures.get("add_poisson_noise_when_needed", True) and gain is not None and gain > 0:
        variance += np.maximum(data, 0.0) / gain
        source += "+poisson"
    floor = float(apertures.get("minimum_uncertainty", 1.0e-6))
    return np.sqrt(np.maximum(variance, floor ** 2)), source


def _measurement_masks(record, shape):
    """Return the combined mask plus separately traceable artifact masks."""

    components = dict(record.get("masks") or {})
    cosmic = record.get("cosmic_ray_products") or {}
    for name in ("cosmic_mask", "cosmic_ray_mask", "mask"):
        value = cosmic.get(name)
        if value is not None and np.shape(value) == shape:
            components["cosmic_rays"] = np.asarray(value, dtype=bool)
            break
    normalized = {}
    for name, value in components.items():
        if value is not None and np.shape(value) == shape:
            normalized[name] = np.asarray(value, dtype=bool)
    normalized["combined"] = _combined_mask(record, shape)
    return normalized


def _radial_weights(
    shape, x, y, outer_radius, subpixels, inner_radius=0.0, method="exact"
):
    """Return fractional pixel weights and image slices for a circle or annulus."""

    outer_radius = float(outer_radius)
    try:
        from photutils.aperture import CircularAnnulus, CircularAperture

        if inner_radius > 0:
            aperture = CircularAnnulus(
                (x, y), r_in=float(inner_radius), r_out=outer_radius
            )
        else:
            aperture = CircularAperture((x, y), r=outer_radius)
        aperture_mask = aperture.to_mask(
            method=method, subpixels=max(1, int(subpixels))
        )
        image_slice, mask_slice = aperture_mask.get_overlap_slices(shape)
        if image_slice is None:
            return (slice(0, 0), slice(0, 0)), np.empty((0, 0))
        return image_slice, np.asarray(aperture_mask.data[mask_slice], dtype=float)
    except (ImportError, TypeError, ValueError):
        pass

    x0 = max(0, int(np.floor(x - outer_radius - 0.5)))
    x1 = min(shape[1], int(np.ceil(x + outer_radius + 0.5)))
    y0 = max(0, int(np.floor(y - outer_radius - 0.5)))
    y1 = min(shape[0], int(np.ceil(y + outer_radius + 0.5)))
    if x0 >= x1 or y0 >= y1:
        return (slice(0, 0), slice(0, 0)), np.empty((0, 0))
    yy, xx = np.indices((y1 - y0, x1 - x0), dtype=float)
    xx += x0
    yy += y0
    subpixels = max(1, int(subpixels))
    offsets = (np.arange(subpixels, dtype=float) + 0.5) / subpixels - 0.5
    weights = np.zeros(xx.shape, dtype=float)
    outer2 = outer_radius ** 2
    inner2 = float(inner_radius) ** 2
    for dy in offsets:
        for dx in offsets:
            radius2 = (xx + dx - x) ** 2 + (yy + dy - y) ** 2
            weights += (radius2 <= outer2) & (radius2 >= inner2)
    weights /= subpixels ** 2
    return (slice(y0, y1), slice(x0, x1)), weights


def _local_background(data, standard_deviation, mask, x, y, fwhm, settings):
    """Measure a sigma-clipped local sky in the configured annulus."""

    apertures = settings.get("apertures", {})
    inner = float(apertures.get("sky_inner_radius_fwhm", 4.0)) * fwhm
    outer = float(apertures.get("sky_outer_radius_fwhm", 7.0)) * fwhm
    slices, weights = _radial_weights(
        data.shape,
        x,
        y,
        outer,
        apertures.get("subpixels", 5),
        inner_radius=inner,
        method=apertures.get("subpixel_method", "exact"),
    )
    if not apertures.get("local_background", True):
        values = standard_deviation[slices]
        valid = np.isfinite(values) & ~mask[slices] & (weights > 0)
        rms = float(np.median(values[valid])) if np.any(valid) else None
        return {
            "value": 0.0,
            "rms": rms,
            "error": 0.0,
            "pixel_count": int(np.count_nonzero(valid)),
            "inner_radius_pixels": inner,
            "outer_radius_pixels": outer,
            "valid": True,
            "slices": slices,
            "weights": weights,
        }
    local = data[slices]
    valid = np.isfinite(local) & ~mask[slices] & (weights > 0)
    values = local[valid]
    minimum = int(apertures.get("minimum_sky_pixels", 50))
    result = {
        "value": None,
        "rms": None,
        "error": None,
        "pixel_count": int(values.size),
        "inner_radius_pixels": inner,
        "outer_radius_pixels": outer,
        "valid": False,
        "slices": slices,
        "weights": weights,
    }
    if values.size < minimum:
        return result
    clipped = sigma_clip(
        values,
        sigma=float(apertures.get("local_background_sigma_clip", 3.0)),
        maxiters=int(apertures.get("local_background_maximum_iterations", 5)),
        masked=True,
    )
    kept = np.asarray(clipped.compressed(), dtype=float)
    if kept.size < minimum:
        result["pixel_count"] = int(kept.size)
        return result
    estimator = apertures.get("local_background_estimator", "median")
    background = float(np.mean(kept)) if estimator == "mean" else float(np.median(kept))
    median = float(np.median(kept))
    rms = float(1.4826 * np.median(np.abs(kept - median)))
    if not np.isfinite(rms) or rms <= 0:
        rms = float(np.std(kept))
    result.update(
        {
            "value": background,
            "rms": rms,
            "error": rms / np.sqrt(kept.size) if np.isfinite(rms) else None,
            "pixel_count": int(kept.size),
            "valid": bool(np.isfinite(background) and np.isfinite(rms)),
        }
    )
    return result


def _artifact_flags(source_type, masks, slices, weights):
    """Return explicit flags for artifact masks overlapping one footprint."""

    flags = []
    footprint = weights > 0
    combined = masks.get("combined")
    if combined is not None and np.any(combined[slices] & footprint):
        flags.append("TARGET_MASKED" if source_type == "target" else "MASKED_PIXELS")
    cosmic = masks.get("cosmic_rays")
    if cosmic is not None and np.any(cosmic[slices] & footprint):
        flags.append("TARGET_COSMIC_RAY" if source_type == "target" else "COSMIC_RAY_OVERLAP")
    trails = masks.get("trails")
    if trails is not None and np.any(trails[slices] & footprint):
        flags.append("TARGET_TRAIL" if source_type == "target" else "TRAIL_OVERLAP")
    return flags


def _aperture_measurement(
    data,
    variance,
    masks,
    source,
    radius,
    background,
    settings,
    method,
):
    """Measure one signed circular-aperture flux and propagated uncertainty."""

    apertures = settings.get("apertures", {})
    slices, weights = _radial_weights(
        data.shape,
        source["x"],
        source["y"],
        radius,
        apertures.get("subpixels", 5),
        method=apertures.get("subpixel_method", "exact"),
    )
    local_data = data[slices]
    local_variance = variance[slices]
    local_mask = masks["combined"][slices]
    valid = (
        (weights > 0)
        & ~local_mask
        & np.isfinite(local_data)
        & np.isfinite(local_variance)
        & (local_variance > 0)
    )
    expected_area = np.pi * radius ** 2
    measured_area = float(np.sum(weights[valid]))
    coverage = measured_area / expected_area if expected_area > 0 else 0.0
    flags = _artifact_flags(source["source_type"], masks, slices, weights)
    minimum_coverage = float(apertures.get("minimum_unmasked_fraction", 0.80))
    if coverage < minimum_coverage:
        flags.extend(["APERTURE_INCOMPLETE", "INSUFFICIENT_UNMASKED_PIXELS"])
    if not background["valid"]:
        flags.append("BAD_LOCAL_BACKGROUND")
    background_value = background["value"] if background["valid"] else 0.0
    background_error = background["error"] if background["valid"] else None
    flux = None
    uncertainty = None
    if np.any(valid):
        used_weights = weights[valid]
        flux = float(np.sum(used_weights * (local_data[valid] - background_value)))
        flux_variance = float(np.sum(used_weights ** 2 * local_variance[valid]))
        if background_error is not None and np.isfinite(background_error):
            flux_variance += (float(np.sum(used_weights)) * background_error) ** 2
        uncertainty = float(np.sqrt(max(flux_variance, 0.0)))
    snr = (
        flux / uncertainty
        if flux is not None and uncertainty is not None and uncertainty > 0
        else None
    )
    return {
        "method": method,
        "flux": flux,
        "flux_uncertainty": uncertainty,
        "snr": snr,
        "aperture_radius_pixels": float(radius),
        "model_type": None,
        "coverage_fraction": float(np.clip(coverage, 0.0, 1.0)),
        "unmasked_weight": measured_area,
        "flags": list(dict.fromkeys(flags)),
        "valid": bool(
            flux is not None
            and coverage >= minimum_coverage
            and (background["valid"] or not apertures.get("local_background", True))
        ),
    }


def _psf_footprint(data, variance, masks, x, y, model):
    """Place a normalized native PSF at a fixed subpixel detector position."""

    model = np.asarray(model, dtype=float)
    half_y = model.shape[0] // 2
    half_x = model.shape[1] // 2
    ix = int(np.floor(x + 0.5))
    iy = int(np.floor(y + 0.5))
    shifted = ndimage.shift(
        model,
        (y - iy, x - ix),
        order=3,
        mode="constant",
        cval=0.0,
        prefilter=True,
    )
    shifted = np.maximum(np.where(np.isfinite(shifted), shifted, 0.0), 0.0)
    normalization = float(np.sum(shifted))
    if normalization <= 0:
        raise ValueError("The shifted PSF footprint has no finite normalization")
    shifted /= normalization
    raw_x0, raw_y0 = ix - half_x, iy - half_y
    raw_x1, raw_y1 = raw_x0 + model.shape[1], raw_y0 + model.shape[0]
    x0, x1 = max(0, raw_x0), min(data.shape[1], raw_x1)
    y0, y1 = max(0, raw_y0), min(data.shape[0], raw_y1)
    if x0 >= x1 or y0 >= y1:
        return None
    mx0, mx1 = x0 - raw_x0, x1 - raw_x0
    my0, my1 = y0 - raw_y0, y1 - raw_y0
    slices = (slice(y0, y1), slice(x0, x1))
    return {
        "slices": slices,
        "data": np.asarray(data[slices], dtype=float),
        "variance": np.asarray(variance[slices], dtype=float),
        "mask": np.asarray(masks["combined"][slices], dtype=bool),
        "model": shifted[my0:my1, mx0:mx1],
        "origin": (x0, y0),
    }


def _fit_psf_flux(footprint, background):
    """Fit a signed amplitude at a fixed position with inverse-variance weights."""

    observed = footprint["data"]
    variance = footprint["variance"]
    model = footprint["model"]
    valid = (
        ~footprint["mask"]
        & np.isfinite(observed)
        & np.isfinite(variance)
        & (variance > 0)
        & np.isfinite(model)
    )
    if not np.any(valid):
        return None, None, valid
    sky = background["value"] if background["valid"] else 0.0
    inverse_variance = 1.0 / variance[valid]
    denominator = float(np.sum(model[valid] ** 2 * inverse_variance))
    if denominator <= 0:
        return None, None, valid
    flux = float(
        np.sum(model[valid] * (observed[valid] - sky) * inverse_variance)
        / denominator
    )
    flux_variance = 1.0 / denominator
    sky_error = background["error"] if background["valid"] else None
    if sky_error is not None and np.isfinite(sky_error):
        sensitivity = float(np.sum(model[valid] * inverse_variance) / denominator)
        flux_variance += (sensitivity * sky_error) ** 2
    return flux, float(np.sqrt(max(flux_variance, 0.0))), valid


def _diagnostic_free_centroid(footprint, background, forced_flux, search_pixels):
    """Fit a diagnostic PSF offset without changing the forced measurement."""

    observed = footprint["data"]
    variance = footprint["variance"]
    base_model = footprint["model"]
    valid = (
        ~footprint["mask"]
        & np.isfinite(observed)
        & np.isfinite(variance)
        & (variance > 0)
    )
    if np.count_nonzero(valid) < 6:
        return None
    sky = background["value"] if background["valid"] else 0.0
    inverse_sigma = 1.0 / np.sqrt(variance[valid])

    def model_and_flux(offset):
        shifted = ndimage.shift(
            base_model,
            (offset[1], offset[0]),
            order=3,
            mode="constant",
            cval=0.0,
            prefilter=True,
        )
        denominator = float(np.sum(shifted[valid] ** 2 / variance[valid]))
        if denominator <= 0:
            return shifted, forced_flux
        amplitude = float(
            np.sum(shifted[valid] * (observed[valid] - sky) / variance[valid])
            / denominator
        )
        return shifted, amplitude

    def residual(offset):
        shifted, amplitude = model_and_flux(offset)
        return (observed[valid] - sky - amplitude * shifted[valid]) * inverse_sigma

    try:
        fit = least_squares(
            residual,
            [0.0, 0.0],
            bounds=([-search_pixels, -search_pixels], [search_pixels, search_pixels]),
        )
    except (FloatingPointError, RuntimeError, ValueError):
        return None
    shifted, amplitude = model_and_flux(fit.x)
    return {
        "offset_x_pixels": float(fit.x[0]),
        "offset_y_pixels": float(fit.x[1]),
        "offset_pixels": float(np.hypot(*fit.x)),
        "flux": amplitude,
        "success": bool(fit.success),
        "cost": float(fit.cost),
        "model": amplitude * shifted + sky,
    }


def _psf_measurement(
    data,
    variance,
    masks,
    source,
    background,
    psf_result,
    settings,
    pixel_scale,
):
    """Measure one fixed-position signed PSF flux and diagnostic free centroid."""

    model = psf_result.get("model_native")
    flags = []
    diagnostics = None
    if model is None:
        flags.append("TARGET_FIT_FAILED")
        return {
            "method": "psf",
            "flux": None,
            "flux_uncertainty": None,
            "snr": None,
            "aperture_radius_pixels": None,
            "model_type": psf_result.get("model_type"),
            "coverage_fraction": 0.0,
            "unmasked_weight": 0.0,
            "flags": flags,
            "valid": False,
        }, diagnostics
    try:
        footprint = _psf_footprint(
            data, variance, masks, source["x"], source["y"], model
        )
    except ValueError:
        footprint = None
    if footprint is None:
        flags.append("TARGET_FIT_FAILED")
        return {
            "method": "psf",
            "flux": None,
            "flux_uncertainty": None,
            "snr": None,
            "aperture_radius_pixels": None,
            "model_type": psf_result.get("model_type"),
            "coverage_fraction": 0.0,
            "unmasked_weight": 0.0,
            "flags": flags,
            "valid": False,
        }, diagnostics
    flux, uncertainty, valid = _fit_psf_flux(footprint, background)
    model_weights = footprint["model"]
    coverage = float(np.sum(model_weights[valid]))
    flags.extend(
        _artifact_flags(
            source["source_type"], masks, footprint["slices"], model_weights
        )
    )
    minimum = float(settings.get("apertures", {}).get("minimum_unmasked_fraction", 0.80))
    if coverage < minimum:
        flags.append("INSUFFICIENT_UNMASKED_PIXELS")
    if not background["valid"]:
        flags.append("BAD_LOCAL_BACKGROUND")
    if flux is None or uncertainty is None:
        flags.append("TARGET_FIT_FAILED")
    snr = flux / uncertainty if flux is not None and uncertainty not in (None, 0) else None
    fixed_model = None
    residual = None
    if flux is not None:
        sky = background["value"] if background["valid"] else 0.0
        fixed_model = flux * model_weights + sky
        residual = footprint["data"] - fixed_model
        residual[~valid] = np.nan
    free = None
    target_settings = settings.get("target_position", {})
    if source["source_type"] == "target" and target_settings.get("diagnostic_recenter", True):
        search_arcsec = float(target_settings.get("centroid_search_radius_arcsec", 3.0))
        search_pixels = search_arcsec / pixel_scale if pixel_scale else 3.0
        free = _diagnostic_free_centroid(
            footprint, background, flux or 0.0, max(0.25, search_pixels)
        )
        if free is not None:
            free["x"] = source["x"] + free["offset_x_pixels"]
            free["y"] = source["y"] + free["offset_y_pixels"]
            free["offset_arcsec"] = (
                free["offset_pixels"] * pixel_scale if pixel_scale else None
            )
            threshold = float(
                target_settings.get("maximum_diagnostic_offset_arcsec", 1.0)
            )
            comparison = free["offset_arcsec"] if pixel_scale else free["offset_pixels"]
            limit = threshold if pixel_scale else threshold
            if comparison is not None and comparison > limit:
                flags.append("TARGET_CENTROID_OFFSET")
    diagnostics = {
        "data": footprint["data"],
        "model": fixed_model,
        "residual": residual,
        "mask": footprint["mask"],
        "origin": footprint["origin"],
        "fixed_x": source["x"],
        "fixed_y": source["y"],
        "free_centroid": free,
    }
    return {
        "method": "psf",
        "flux": flux,
        "flux_uncertainty": uncertainty,
        "snr": snr,
        "aperture_radius_pixels": None,
        "model_type": psf_result.get("model_type"),
        "coverage_fraction": float(np.clip(coverage, 0.0, 1.0)),
        "unmasked_weight": float(np.count_nonzero(valid)),
        "flags": list(dict.fromkeys(flags)),
        "valid": bool(
            flux is not None
            and coverage >= minimum
            and (
                background["valid"]
                or not settings.get("apertures", {}).get("local_background", True)
            )
        ),
    }, diagnostics


def _science_sources(image_id, measurements, target_solution, wcs):
    """Build the target and unique approved comparison/calibration source list."""

    coordinate = SkyCoord(
        float(target_solution["ra_deg"]),
        float(target_solution["dec_deg"]),
        unit="deg",
        frame="icrs",
    )
    x, y = wcs.world_to_pixel(coordinate)
    sources = [
        {
            "source_id": "target",
            "source_type": "target",
            "roles": "target",
            "x": float(x),
            "y": float(y),
            "ra_deg": float(coordinate.ra.deg),
            "dec_deg": float(coordinate.dec.deg),
        }
    ]
    if measurements is None or len(measurements) == 0:
        return sources
    role_names = ("psf", "calibration", "ensemble", "qc_anchor")
    seen = set()
    for row in measurements:
        if str(row["image_id"]) != str(image_id):
            continue
        roles = [
            role for role in role_names
            if "role_{}".format(role) in measurements.colnames
            and bool(row["role_{}".format(role)])
        ]
        if not roles:
            continue
        if "image_accepted" in measurements.colnames and not bool(row["image_accepted"]):
            continue
        source_id = str(row["persistent_id"])
        if source_id in seen:
            continue
        seen.add(source_id)
        sx, sy = float(row["x"]), float(row["y"])
        try:
            sky = wcs.pixel_to_world(sx, sy).icrs
            ra, dec = float(sky.ra.deg), float(sky.dec.deg)
        except (AttributeError, TypeError, ValueError):
            ra = dec = None
        sources.append(
            {
                "source_id": source_id,
                "source_type": "calibration" if "calibration" in roles else "comparison",
                "roles": ";".join(roles),
                "x": sx,
                "y": sy,
                "ra_deg": ra,
                "dec_deg": dec,
            }
        )
    return sources


def _measurement_table(rows, flux_unit):
    """Convert science-image measurement rows to a masked Astropy table."""

    table = Table(masked=True)
    string_fields = (
        "image_id", "filename", "filter", "source_id", "source_type", "roles",
        "method", "coordinate_version", "model_type", "psf_version",
        "uncertainty_source", "flags",
    )
    boolean_fields = ("fixed_position", "valid")
    numeric_fields = (
        "mjd_mid", "x", "y", "ra_deg", "dec_deg", "flux", "flux_uncertainty",
        "snr", "local_background", "local_background_rms", "local_background_error",
        "local_background_pixels", "aperture_radius_pixels", "sky_inner_radius_pixels",
        "sky_outer_radius_pixels", "coverage_fraction", "unmasked_weight",
        "free_centroid_x", "free_centroid_y", "centroid_offset_pixels",
        "centroid_offset_arcsec", "free_centroid_flux", "fwhm_pixels",
        "psf_normalization",
    )
    for name in string_fields:
        table[name] = [str(row.get(name, "")) for row in rows]
    for name in boolean_fields:
        table[name] = [bool(row.get(name, False)) for row in rows]
    for name in numeric_fields:
        values = np.asarray(
            [_finite_float(row.get(name), np.nan) for row in rows], dtype=float
        )
        invalid = ~np.isfinite(values)
        table[name] = MaskedColumn(np.where(invalid, 0.0, values), mask=invalid)
    for name in ("x", "y", "aperture_radius_pixels", "sky_inner_radius_pixels", "sky_outer_radius_pixels", "free_centroid_x", "free_centroid_y", "centroid_offset_pixels", "fwhm_pixels"):
        table[name].unit = u.pixel
    for name in ("ra_deg", "dec_deg"):
        table[name].unit = u.deg
    table["centroid_offset_arcsec"].unit = u.arcsec
    for name in ("flux", "flux_uncertainty", "local_background", "local_background_rms", "local_background_error", "free_centroid_flux"):
        table[name].unit = flux_unit
    table.meta["flux_unit"] = str(flux_unit)
    table.meta["signed_fluxes"] = True
    table.meta["free_centroids_diagnostic_only"] = True
    return table


def perform_science_image_photometry(
    image_record,
    target_solution,
    psf_result,
    measurements=None,
    settings=None,
    wcs=None,
):
    """Perform forced target and comparison-star photometry on one image.

    Small-aperture, reference-aperture, and PSF measurements share a single
    fixed position per source. The target position always comes from the
    frozen sky-coordinate solution. A free-centroid PSF fit is retained only
    as a diagnostic and never replaces the forced flux.
    """

    if settings is None:
        settings = image_record.get("settings") or get_default_settings()
    apertures = settings.get("apertures", {})
    if not apertures.get("enabled", True):
        raise RuntimeError("Science-image photometry is disabled")
    if not target_solution.get("frozen", False):
        raise ValueError("Target photometry requires a frozen target-coordinate solution")
    if apertures.get("perform_psf", True) and apertures.get("require_approved_psf", True):
        require_approved_psf(psf_result)
    wcs = _record_wcs(image_record, supplied=wcs)
    if wcs is None:
        raise ValueError("Science-image photometry requires a celestial derived WCS")
    data = _prepared_data(image_record)
    standard_deviation, uncertainty_source = _standard_deviation(
        image_record, data, settings
    )
    variance = standard_deviation ** 2
    masks = _measurement_masks(image_record, data.shape)
    image_id = _image_id(image_record)
    metadata = image_record.get("metadata") or {}
    ccd = None
    for name in ("prepared_ccd", "working_ccd", "ccd"):
        if image_record.get(name) is not None:
            ccd = image_record[name]
            break
    flux_unit = getattr(ccd, "unit", u.adu) if ccd is not None else u.adu
    fwhm = _finite_float(psf_result.get("fwhm_pixels"))
    if fwhm is None:
        fwhm = _finite_float((image_record.get("quality") or {}).get("fwhm_pixels"), 4.0)
    pixel_scale = _pixel_scale_arcsec(image_record, wcs)
    sources = _science_sources(image_id, measurements, target_solution, wcs)
    rows = []
    target_diagnostics = None
    target_flags = []
    for source in sources:
        background = _local_background(
            data,
            standard_deviation,
            masks["combined"],
            source["x"],
            source["y"],
            fwhm,
            settings,
        )
        source_measurements = []
        if apertures.get("perform_small_aperture", True):
            source_measurements.append(
                _aperture_measurement(
                    data,
                    variance,
                    masks,
                    source,
                    float(apertures.get("small_radius_fwhm", 1.0)) * fwhm,
                    background,
                    settings,
                    "small_aperture",
                )
            )
        if apertures.get("perform_large_aperture", True):
            source_measurements.append(
                _aperture_measurement(
                    data,
                    variance,
                    masks,
                    source,
                    float(apertures.get("large_radius_fwhm", 2.5)) * fwhm,
                    background,
                    settings,
                    "large_aperture",
                )
            )
        if apertures.get("perform_psf", True):
            psf_measurement, diagnostics = _psf_measurement(
                data,
                variance,
                masks,
                source,
                background,
                psf_result,
                settings,
                pixel_scale,
            )
            source_measurements.append(psf_measurement)
            if source["source_type"] == "target":
                target_diagnostics = diagnostics
        if source["source_type"] == "target":
            target_flags = list(
                dict.fromkeys(
                    flag
                    for measurement in source_measurements
                    for flag in measurement["flags"]
                )
            )
        free = (
            target_diagnostics.get("free_centroid")
            if source["source_type"] == "target" and target_diagnostics is not None
            else None
        )
        for measurement in source_measurements:
            flags = list(measurement["flags"])
            if source["source_type"] == "target":
                flags = list(dict.fromkeys(flags + target_flags))
            rows.append(
                {
                    "image_id": image_id,
                    "filename": metadata.get("filename", image_id),
                    "filter": metadata.get("filter", ""),
                    "mjd_mid": metadata.get("mjd_mid", metadata.get("mjd")),
                    "source_id": source["source_id"],
                    "source_type": source["source_type"],
                    "roles": source["roles"],
                    "method": measurement["method"],
                    "fixed_position": True,
                    "coordinate_version": target_solution.get("version", ""),
                    "x": source["x"],
                    "y": source["y"],
                    "ra_deg": source["ra_deg"],
                    "dec_deg": source["dec_deg"],
                    "flux": measurement["flux"],
                    "flux_uncertainty": measurement["flux_uncertainty"],
                    "snr": measurement["snr"],
                    "local_background": background["value"],
                    "local_background_rms": background["rms"],
                    "local_background_error": background["error"],
                    "local_background_pixels": background["pixel_count"],
                    "aperture_radius_pixels": measurement["aperture_radius_pixels"],
                    "sky_inner_radius_pixels": background["inner_radius_pixels"],
                    "sky_outer_radius_pixels": background["outer_radius_pixels"],
                    "model_type": measurement["model_type"],
                    "psf_version": "psf-v{}".format(
                        psf_result.get("model_version", 1)
                    ),
                    "psf_normalization": psf_result.get("normalization"),
                    "coverage_fraction": measurement["coverage_fraction"],
                    "unmasked_weight": measurement["unmasked_weight"],
                    "uncertainty_source": uncertainty_source,
                    "free_centroid_x": None if free is None else free.get("x"),
                    "free_centroid_y": None if free is None else free.get("y"),
                    "centroid_offset_pixels": None if free is None else free.get("offset_pixels"),
                    "centroid_offset_arcsec": None if free is None else free.get("offset_arcsec"),
                    "free_centroid_flux": None if free is None else free.get("flux"),
                    "fwhm_pixels": fwhm,
                    "flags": ";".join(flags),
                    "valid": measurement["valid"],
                }
            )
    table = _measurement_table(rows, flux_unit)
    if target_diagnostics is not None:
        context_radius = max(
            float(apertures.get("diagnostic_cutout_radius_fwhm", 5.0)),
            float(apertures.get("sky_outer_radius_fwhm", 7.0)),
        ) * fwhm
        target_x = sources[0]["x"]
        target_y = sources[0]["y"]
        context_slices, _ = _radial_weights(
            data.shape,
            target_x,
            target_y,
            context_radius,
            1,
            method="center",
        )
        target_diagnostics.update(
            {
                "small_radius_pixels": float(apertures.get("small_radius_fwhm", 1.0)) * fwhm,
                "large_radius_pixels": float(apertures.get("large_radius_fwhm", 2.5)) * fwhm,
                "sky_inner_radius_pixels": float(apertures.get("sky_inner_radius_fwhm", 4.0)) * fwhm,
                "sky_outer_radius_pixels": float(apertures.get("sky_outer_radius_fwhm", 7.0)) * fwhm,
                "context_data": np.asarray(data[context_slices], dtype=float),
                "context_mask": np.asarray(masks["combined"][context_slices], dtype=bool),
                "context_origin": (
                    int(context_slices[1].start), int(context_slices[0].start)
                ),
            }
        )
    return {
        "image_id": image_id,
        "filename": metadata.get("filename", image_id),
        "target_coordinate_version": target_solution.get("version"),
        "fixed_target_ra_deg": float(target_solution["ra_deg"]),
        "fixed_target_dec_deg": float(target_solution["dec_deg"]),
        "pixel_scale_arcsec": pixel_scale,
        "fwhm_pixels": fwhm,
        "flux_unit": str(flux_unit),
        "uncertainty_source": uncertainty_source,
        "measurements": table,
        "target_diagnostics": target_diagnostics,
        "target_flags": list(dict.fromkeys(target_flags)),
        "source_count": len(sources),
        "method_count": len(set(table["method"])) if len(table) else 0,
    }


def perform_science_photometry(
    image_records,
    target_solution,
    psf_results,
    measurements=None,
    alignments=None,
    settings=None,
):
    """Perform science-image photometry for a complete image batch."""

    if settings is None:
        settings = get_default_settings()
    if isinstance(psf_results, Mapping):
        psf_lookup = dict(psf_results)
    else:
        psf_lookup = {str(item["image_id"]): item for item in psf_results}
    alignment_lookup = {
        str(item["image_id"]): item for item in (alignments or [])
    }
    results = []
    tables = []
    for index, record in enumerate(image_records):
        image_id = _image_id(record, index)
        if image_id not in psf_lookup:
            raise ValueError("No PSF result is available for {}".format(image_id))
        image_settings = record.get("settings") or settings
        alignment = alignment_lookup.get(image_id)
        wcs = None if alignment is None else alignment.get("wcs")
        result = perform_science_image_photometry(
            record,
            target_solution,
            psf_lookup[image_id],
            measurements=measurements,
            settings=image_settings,
            wcs=wcs,
        )
        results.append(result)
        tables.append(result["measurements"])
    combined = (
        vstack(tables, metadata_conflicts="silent")
        if tables else Table(masked=True)
    )
    return combined, results


def save_science_photometry_products(
    result, output_directory, settings=None, overwrite=None
):
    """Save one image's measurement table and target diagnostic cutouts."""

    if settings is None:
        settings = get_default_settings()
    apertures = settings.get("apertures", {})
    if overwrite is None:
        overwrite = settings.get("output", {}).get("overwrite", False)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    stem = _safe_stem(result.get("filename") or result.get("image_id"))
    paths = {}
    if apertures.get("save_measurement_table", True):
        path = output_directory / "{}_science_photometry.ecsv".format(stem)
        result["measurements"].write(
            path, format="ascii.ecsv", overwrite=bool(overwrite)
        )
        paths["measurements"] = str(path)
    diagnostics = result.get("target_diagnostics")
    if apertures.get("save_target_cutouts", True) and diagnostics is not None:
        hdus = [fits.PrimaryHDU()]
        extension_names = {
            "data": "DATA",
            "model": "MODEL",
            "residual": "RESIDUAL",
            "mask": "MASK",
            "context_data": "CONTEXT",
            "context_mask": "CONTMASK",
        }
        for name, extension in extension_names.items():
            value = diagnostics.get(name)
            if value is None:
                continue
            array = np.asarray(
                value, dtype=np.uint8 if name.endswith("mask") else np.float32
            )
            hdus.append(fits.ImageHDU(array, name=extension))
        path = output_directory / "{}_target_photometry.fits".format(stem)
        fits.HDUList(hdus).writeto(path, overwrite=bool(overwrite))
        paths["target_cutouts"] = str(path)
    return paths


__all__ = [
    "PSF_DOWNSTREAM_STAGES",
    "apply_psf_review",
    "construct_psf",
    "construct_psfs",
    "perform_science_image_photometry",
    "perform_science_photometry",
    "plan_psf_rerun",
    "psf_dependency_signature",
    "require_approved_psf",
    "save_science_photometry_products",
    "save_psf_products",
]
