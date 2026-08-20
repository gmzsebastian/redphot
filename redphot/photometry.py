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
from astropy.nddata import CCDData, StdDevUncertainty
from astropy.stats import sigma_clip
from astropy.table import MaskedColumn, Table, vstack
from astropy.wcs.utils import proj_plane_pixel_scales
from scipy import ndimage
from scipy.optimize import least_squares

from .catalogs import normalize_catalog_name
from .config import get_default_settings, merge_settings, normalize_filter_name


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
        "image_id", "filename", "filter", "telescope", "site", "instrument",
        "detector", "source_id", "source_type", "roles",
        "method", "coordinate_version", "model_type", "psf_version",
        "uncertainty_source", "flags",
    )
    boolean_fields = ("fixed_position", "valid")
    numeric_fields = (
        "mjd_mid", "exposure_time", "airmass", "x", "y", "ra_deg", "dec_deg",
        "flux", "flux_uncertainty",
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
    table["exposure_time"].unit = u.s
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
                    "telescope": metadata.get("telescope", ""),
                    "site": metadata.get("site", ""),
                    "instrument": metadata.get("instrument", ""),
                    "detector": metadata.get("detector", ""),
                    "mjd_mid": metadata.get("mjd_mid", metadata.get("mjd")),
                    "exposure_time": metadata.get("exposure_time"),
                    "airmass": metadata.get("airmass"),
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


def route_calibration_catalog(filter_name, settings=None, available_catalogs=None):
    """Return the configured photometric catalog for a normalized filter."""

    if settings is None:
        settings = get_default_settings()
    band = normalize_filter_name(filter_name)
    calibration = settings.get("calibration", {})
    configured = calibration.get("catalog", "auto")
    if configured in (None, "auto"):
        configured = settings.get("catalogs", {}).get(
            "photometry_catalog_by_filter", {}
        ).get(band)
        if configured is None:
            configured = settings.get("catalogs", {}).get("photometry_catalog")
    routed = normalize_catalog_name(configured)
    available = [normalize_catalog_name(value) for value in (available_catalogs or [])]
    if routed in (None, "auto") and available:
        routed = "user" if "user" in available else available[0]
    return routed, band


def _catalog_collection(catalogs):
    """Normalize one table or a mapping of tables into catalog-name groups."""

    if catalogs is None:
        return {}
    if isinstance(catalogs, Mapping):
        return {
            normalize_catalog_name(name) or "user": Table(table, masked=True, copy=False)
            for name, table in catalogs.items()
        }
    table = Table(catalogs, masked=True, copy=False)
    if "catalog_name" in table.colnames:
        names = list(dict.fromkeys(str(value) for value in table["catalog_name"]))
        return {
            normalize_catalog_name(name) or "user": table[
                np.asarray(table["catalog_name"], dtype=str) == name
            ]
            for name in names
        }
    name = normalize_catalog_name(table.meta.get("catalog_name")) or "user"
    return {name: table}


def _catalog_identifier_column(table):
    """Return the source identifier column in a normalized or master catalog."""

    for name in ("persistent_id", "source_id"):
        if name in table.colnames:
            return name
    return None


def _catalog_band_column(filter_name, catalog_name, settings):
    """Return the configured normalized magnitude column for one filter."""

    mapping = settings.get("calibration", {}).get("filter_column_map", {})
    value = mapping.get(filter_name)
    if isinstance(value, Mapping):
        value = value.get(catalog_name) or value.get("default")
    if value is None:
        value = "mag_{}".format(filter_name)
    value = str(value)
    return value if value.startswith("mag_") else "mag_{}".format(value)


def _catalog_value(table, column, index):
    """Read one finite catalog number, returning ``None`` when masked."""

    if column not in table.colnames:
        return None
    return _finite_float(table[column][index])


def _catalog_entry(catalogs, source_id, routed_catalog, band, settings):
    """Look up one source's routed magnitude, uncertainty, and color."""

    calibration = settings.get("calibration", {})
    names = []
    if routed_catalog is not None:
        names.append(routed_catalog)
    if calibration.get("allow_catalog_fallback", True):
        names.extend(name for name in catalogs if name not in names)
    for name in names:
        table = catalogs.get(name)
        if table is None:
            continue
        identifier = _catalog_identifier_column(table)
        if identifier is None:
            continue
        identifiers = np.asarray(table[identifier], dtype=str)
        match = np.flatnonzero(identifiers == str(source_id))
        if match.size == 0 and ":" in str(source_id):
            match = np.flatnonzero(identifiers == str(source_id).split(":", 1)[1])
        if match.size == 0:
            continue
        index = int(match[0])
        magnitude_column = _catalog_band_column(band, name, settings)
        magnitude = _catalog_value(table, magnitude_column, index)
        if magnitude is None:
            continue
        error_column = magnitude_column.replace("mag_", "magerr_", 1)
        magnitude_error = _catalog_value(table, error_column, index)
        color = _catalog_value(table, "catalog_color", index)
        if color is None:
            color_bands = settings.get("catalogs", {}).get(
                "comparison_stars", {}
            ).get("color_bands", ["g", "r"])
            first = _catalog_value(table, "mag_{}".format(color_bands[0]), index)
            second = _catalog_value(table, "mag_{}".format(color_bands[1]), index)
            color = None if first is None or second is None else first - second
        system = calibration.get("magnitude_system_by_catalog", {}).get(name)
        return {
            "catalog_name": name,
            "routed_catalog": routed_catalog,
            "magnitude_band": magnitude_column.replace("mag_", "", 1),
            "magnitude_system": system,
            "catalog_magnitude": magnitude,
            "catalog_magnitude_error": magnitude_error,
            "catalog_color": color,
            "catalog_fallback": name != routed_catalog,
        }
    return None


def _science_row_value(row, name, default=None):
    """Return one optional unmasked science-table value."""

    if name not in row.colnames or np.ma.is_masked(row[name]):
        return default
    value = row[name]
    return value.item() if isinstance(value, np.generic) else value


def _instrumental_magnitude(flux, uncertainty, exposure):
    """Return a rate-based instrumental magnitude and uncertainty."""

    flux = _finite_float(flux)
    uncertainty = _finite_float(uncertainty)
    exposure = _finite_float(exposure)
    if flux is None or flux <= 0 or exposure is None or exposure <= 0:
        return None, None
    magnitude = -2.5 * np.log10(flux / exposure)
    error = (
        2.5 / np.log(10.0) * uncertainty / flux
        if uncertainty is not None and uncertainty >= 0
        else None
    )
    return float(magnitude), None if error is None else float(error)


def _calibration_records(measurements, catalogs, settings):
    """Build calibration-star records from valid role assignments and catalog data."""

    calibration = settings.get("calibration", {})
    excluded_flags = set(calibration.get("excluded_measurement_flags", []))
    records = []
    available = list(catalogs)
    for index, row in enumerate(measurements):
        roles = str(_science_row_value(row, "roles", ""))
        if "calibration" not in roles.split(";"):
            continue
        image_id = str(row["image_id"])
        method = str(row["method"])
        filter_name = normalize_filter_name(_science_row_value(row, "filter"))
        routed, band = route_calibration_catalog(filter_name, settings, available)
        source_id = str(row["source_id"])
        entry = _catalog_entry(catalogs, source_id, routed, band, settings)
        reasons = []
        flux = _finite_float(_science_row_value(row, "flux"))
        flux_error = _finite_float(_science_row_value(row, "flux_uncertainty"))
        exposure = _finite_float(_science_row_value(row, "exposure_time"))
        snr = _finite_float(_science_row_value(row, "snr"))
        if not bool(_science_row_value(row, "valid", False)):
            reasons.append("MEASUREMENT_INVALID")
        flags = set(filter(None, str(_science_row_value(row, "flags", "")).split(";")))
        if flags & excluded_flags:
            reasons.append("MEASUREMENT_FLAGGED")
        if flux is None or flux <= 0 or flux_error is None or exposure in (None, 0):
            reasons.append("FLUX_INVALID")
        if snr is None or snr < float(calibration.get("minimum_star_snr", 10.0)):
            reasons.append("SNR_LOW")
        if entry is None:
            reasons.append("CATALOG_MAGNITUDE_MISSING")
        elif calibration.get("require_routed_catalog", False) and entry["catalog_fallback"]:
            reasons.append("CATALOG_MISMATCH")
        if entry is not None:
            catalog_error = entry["catalog_magnitude_error"]
            if catalog_error is None:
                if not calibration.get("allow_missing_catalog_error", True):
                    reasons.append("CATALOG_ERROR_MISSING")
            elif catalog_error > float(
                calibration.get("maximum_catalog_error_mag", 0.10)
            ):
                reasons.append("CATALOG_ERROR_HIGH")
        instrumental, instrumental_error = _instrumental_magnitude(
            flux, flux_error, exposure
        )
        if instrumental is None:
            reasons.append("INSTRUMENTAL_MAGNITUDE_INVALID")
        individual_zeropoint = (
            None
            if entry is None or instrumental is None
            else entry["catalog_magnitude"] - instrumental
        )
        records.append(
            {
                "measurement_index": index,
                "image_id": image_id,
                "method": method,
                "filter": band,
                "source_id": source_id,
                "routed_catalog": routed,
                "catalog_name": None if entry is None else entry["catalog_name"],
                "magnitude_band": None if entry is None else entry["magnitude_band"],
                "magnitude_system": None if entry is None else entry["magnitude_system"],
                "catalog_magnitude": None if entry is None else entry["catalog_magnitude"],
                "catalog_magnitude_error": None if entry is None else entry["catalog_magnitude_error"],
                "catalog_color": None if entry is None else entry["catalog_color"],
                "flux": flux,
                "flux_uncertainty": flux_error,
                "snr": snr,
                "x": _finite_float(_science_row_value(row, "x")),
                "y": _finite_float(_science_row_value(row, "y")),
                "airmass": _finite_float(_science_row_value(row, "airmass")),
                "exposure_time": exposure,
                "instrumental_magnitude": instrumental,
                "instrumental_magnitude_uncertainty": instrumental_error,
                "individual_zeropoint": individual_zeropoint,
                "input_accepted": not reasons,
                "inlier": False,
                "unstable": False,
                "zeropoint_residual": None,
                "calibrated_residual": None,
                "rejection_reason": ";".join(dict.fromkeys(reasons)),
            }
        )
    return records


def _robust_scatter(values):
    """Return a finite MAD scatter with a standard-deviation fallback."""

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    median = float(np.median(values))
    scatter = float(1.4826 * np.median(np.abs(values - median)))
    if values.size > 1 and (not np.isfinite(scatter) or scatter == 0):
        scatter = float(np.std(values, ddof=1))
    return scatter if np.isfinite(scatter) else None


def _solve_zeropoint_group(records, indices, unstable_pairs, settings):
    """Solve one robust inverse-variance method-specific zeropoint."""

    calibration = settings.get("calibration", {})
    usable = []
    for index in indices:
        record = records[index]
        if (record["source_id"], record["method"]) in unstable_pairs:
            record["unstable"] = True
            record["rejection_reason"] = ";".join(
                filter(None, [record["rejection_reason"], "STAR_UNSTABLE"])
            )
            continue
        if record["input_accepted"] and record["individual_zeropoint"] is not None:
            usable.append(index)
    usable.sort(key=lambda index: -(records[index]["snr"] or 0.0))
    usable = usable[: int(calibration.get("maximum_stars", 100))]
    inlier = np.ones(len(usable), dtype=bool)
    sigma = float(calibration.get("sigma_clip", 3.0))
    for _ in range(int(calibration.get("maximum_iterations", 5))):
        active = [index for index, keep in zip(usable, inlier) if keep]
        if not active:
            break
        values = np.asarray(
            [records[index]["individual_zeropoint"] for index in active], dtype=float
        )
        center = float(np.median(values))
        scatter = _robust_scatter(values)
        errors = []
        for index in active:
            record = records[index]
            errors.append(
                np.hypot(
                    record["instrumental_magnitude_uncertainty"] or 0.0,
                    record["catalog_magnitude_error"] or 0.0,
                )
            )
        scale = max(scatter or 0.0, float(np.median(errors)) if errors else 0.0, 1.0e-4)
        new_inlier = np.array(
            [
                abs(records[index]["individual_zeropoint"] - center) <= sigma * scale
                for index in usable
            ],
            dtype=bool,
        )
        if np.array_equal(new_inlier, inlier):
            break
        inlier = new_inlier
    accepted = [index for index, keep in zip(usable, inlier) if keep]
    minimum = int(calibration.get("minimum_stars", 3))
    first = records[usable[0]] if usable else records[indices[0]]
    solution = {
        "image_id": first["image_id"],
        "method": first["method"],
        "filter": first["filter"],
        "catalog_name": first["routed_catalog"],
        "magnitude_band": first["magnitude_band"],
        "magnitude_system": first["magnitude_system"],
        "zeropoint_mag": None,
        "zeropoint_uncertainty_mag": None,
        "zeropoint_scatter_mag": None,
        "star_count": len(accepted),
        "rejected_star_count": len(indices) - len(accepted),
        "status": "FAIL",
        "flags": [],
        "airmass": None,
    }
    if len(accepted) < minimum:
        solution["flags"].append("TOO_FEW_CALIBRATION_STARS")
        return solution
    values = np.asarray(
        [records[index]["individual_zeropoint"] for index in accepted], dtype=float
    )
    variances = []
    for index in accepted:
        record = records[index]
        error = np.hypot(
            record["instrumental_magnitude_uncertainty"] or 0.0,
            record["catalog_magnitude_error"] or 0.0,
        )
        variances.append(max(error, 1.0e-4) ** 2)
    weights = 1.0 / np.asarray(variances)
    zeropoint = float(np.sum(weights * values) / np.sum(weights))
    scatter = _robust_scatter(values)
    formal = float(np.sqrt(1.0 / np.sum(weights)))
    uncertainty = max(
        formal,
        0.0 if scatter is None else scatter / np.sqrt(len(values)),
    )
    solution.update(
        {
            "zeropoint_mag": zeropoint,
            "zeropoint_uncertainty_mag": uncertainty,
            "zeropoint_scatter_mag": scatter,
            "status": "PASS",
            "airmass": float(np.median([
                records[index]["airmass"] for index in accepted
                if records[index]["airmass"] is not None
            ])) if any(records[index]["airmass"] is not None for index in accepted) else None,
        }
    )
    warn = float(calibration.get("zeropoint_scatter_warn_mag", 0.10))
    fail = float(calibration.get("zeropoint_scatter_fail_mag", 0.30))
    if scatter is not None and scatter > fail:
        solution["status"] = "FAIL"
        solution["flags"].append("ZEROPOINT_SCATTER_HIGH")
    elif scatter is not None and scatter > warn:
        solution["status"] = "WARN"
        solution["flags"].append("ZEROPOINT_SCATTER_HIGH")
    for index in indices:
        records[index]["inlier"] = index in accepted
        individual = records[index]["individual_zeropoint"]
        if records[index]["input_accepted"] and individual is not None:
            residual = individual - zeropoint
            records[index]["zeropoint_residual"] = residual
            records[index]["calibrated_residual"] = -residual
        if (
            index not in accepted
            and records[index]["input_accepted"]
            and not records[index]["rejection_reason"]
        ):
            records[index]["rejection_reason"] = "ZEROPOINT_OUTLIER"
    return solution


def _solve_all_zeropoints(records, settings, unstable_pairs=None):
    """Solve every image and method group, updating calibration-star records."""

    unstable_pairs = unstable_pairs or set()
    groups = {}
    for index, record in enumerate(records):
        key = (record["image_id"], record["method"])
        groups.setdefault(key, []).append(index)
    solutions = []
    for key in sorted(groups):
        solutions.append(
            _solve_zeropoint_group(
                records, groups[key], unstable_pairs, settings
            )
        )
    return solutions


def _unstable_calibration_stars(records, settings):
    """Identify repeat stars whose zeropoint residuals vary across images."""

    calibration = settings.get("calibration", {})
    minimum = int(calibration.get("minimum_epochs_for_stability", 2))
    maximum = float(calibration.get("maximum_star_rms_mag", 0.10))
    grouped = {}
    for record in records:
        if record["input_accepted"] and record["zeropoint_residual"] is not None:
            grouped.setdefault((record["source_id"], record["method"]), []).append(
                record["zeropoint_residual"]
            )
    return {
        key for key, values in grouped.items()
        if len(values) >= minimum
        and (_robust_scatter(values) or 0.0) > maximum
    }


def _aperture_corrections(measurements, settings):
    """Calculate robust method-to-reference aperture corrections per image."""

    calibration = settings.get("calibration", {})
    reference = calibration.get("reference_aperture_method", "large_aperture")
    minimum = int(calibration.get("minimum_aperture_correction_stars", 3))
    rows = []
    image_ids = list(dict.fromkeys(str(value) for value in measurements["image_id"]))
    methods = list(dict.fromkeys(str(value) for value in measurements["method"]))
    for image_id in image_ids:
        image_rows = measurements[np.asarray(measurements["image_id"], dtype=str) == image_id]
        by_source = {}
        for row in image_rows:
            if "calibration" not in str(row["roles"]).split(";"):
                continue
            magnitude, error = _instrumental_magnitude(
                _science_row_value(row, "flux"),
                _science_row_value(row, "flux_uncertainty"),
                _science_row_value(row, "exposure_time"),
            )
            if magnitude is not None and bool(_science_row_value(row, "valid", False)):
                by_source.setdefault(str(row["source_id"]), {})[str(row["method"])] = (
                    magnitude, error
                )
        for method in methods:
            differences = []
            for values in by_source.values():
                if reference in values and method in values:
                    differences.append(values[reference][0] - values[method][0])
            if method == reference:
                differences = [0.0] * max(minimum, len(differences))
            if differences:
                clipped = sigma_clip(
                    differences,
                    sigma=float(calibration.get("aperture_correction_sigma_clip", 3.0)),
                    maxiters=int(calibration.get("maximum_iterations", 5)),
                    masked=True,
                )
                accepted = np.asarray(clipped.compressed(), dtype=float)
            else:
                accepted = np.array([], dtype=float)
            correction = float(np.median(accepted)) if len(accepted) >= minimum else None
            scatter = _robust_scatter(accepted)
            rows.append(
                {
                    "image_id": image_id,
                    "method": method,
                    "reference_method": reference,
                    "aperture_correction_mag": correction,
                    "aperture_correction_uncertainty_mag": (
                        None if scatter is None else scatter / np.sqrt(len(accepted))
                    ),
                    "scatter_mag": scatter,
                    "star_count": len(accepted),
                    "status": "PASS" if correction is not None else "FAIL",
                    "flags": "" if correction is not None else "APERTURE_CORRECTION_FAILED",
                }
            )
    return _records_table(rows)


def _linear_trend(x, y):
    """Return a centered linear slope and correlation for finite values."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if len(x) < 3 or np.ptp(x) <= 0:
        return None, None, len(x)
    normalized = (x - np.min(x)) / np.ptp(x)
    slope = float(np.polyfit(normalized, y, 1)[0])
    correlation = (
        0.0 if np.ptp(y) == 0
        else float(np.corrcoef(normalized, y)[0, 1])
    )
    return slope, correlation, len(x)


def _calibration_trends(records, settings):
    """Measure residual trends against catalog, measurement, and detector variables."""

    calibration = settings.get("calibration", {})
    methods = sorted({record["method"] for record in records})
    rows = []
    variables = {
        "magnitude": "catalog_magnitude",
        "color": "catalog_color",
        "snr": "snr",
        "detector_x": "x",
        "detector_y": "y",
        "airmass": "airmass",
    }
    for method in methods:
        selected = [record for record in records if record["method"] == method and record["inlier"]]
        residuals = [record["calibrated_residual"] for record in selected]
        for variable, key in variables.items():
            slope, correlation, count = _linear_trend(
                [np.nan if record[key] is None else record[key] for record in selected],
                residuals,
            )
            warning = (
                slope is not None
                and abs(slope) > float(calibration.get("trend_slope_warn_mag", 0.05))
            )
            rows.append(
                {
                    "method": method,
                    "variable": variable,
                    "slope_mag_per_span": slope,
                    "correlation": correlation,
                    "point_count": count,
                    "status": "WARN" if warning else "PASS",
                }
            )
    return _records_table(rows)


def _records_table(records):
    """Convert scalar dictionaries with missing values into a masked table."""

    if not records:
        return Table(masked=True)
    names = list(dict.fromkeys(name for record in records for name in record))
    table = Table(masked=True)
    for name in names:
        values = [record.get(name) for record in records]
        nonmissing = [value for value in values if value is not None]
        if nonmissing and all(isinstance(value, (bool, np.bool_)) for value in nonmissing):
            table[name] = MaskedColumn(
                [False if value is None else bool(value) for value in values],
                mask=[value is None for value in values],
            )
        elif nonmissing and all(
            isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, (bool, np.bool_))
            for value in nonmissing
        ):
            numeric = np.asarray(
                [np.nan if value is None else float(value) for value in values]
            )
            table[name] = MaskedColumn(
                np.where(np.isfinite(numeric), numeric, 0.0),
                mask=~np.isfinite(numeric),
            )
        else:
            table[name] = MaskedColumn(
                ["" if value is None else str(value) for value in values],
                mask=[value is None for value in values],
            )
    return table


def _add_numeric_columns(table, values_by_name):
    """Add masked floating columns to an existing table copy."""

    for name, values in values_by_name.items():
        numeric = np.asarray(
            [np.nan if value is None else float(value) for value in values], dtype=float
        )
        table[name] = MaskedColumn(
            np.where(np.isfinite(numeric), numeric, 0.0),
            mask=~np.isfinite(numeric),
        )


def _append_flag(value, flag):
    """Append one semicolon-delimited flag without duplication."""

    flags = list(filter(None, str(value or "").split(";")))
    if flag and flag not in flags:
        flags.append(flag)
    return ";".join(flags)


def _empty_source_mask(record, science_result, settings):
    """Build a conservative source-exclusion mask for empty apertures."""

    data = _prepared_data(record)
    exclusion = np.zeros(data.shape, dtype=bool)
    products = record.get("background_products") or {}
    for name in ("detected_source_mask", "protected_source_mask"):
        value = products.get(name)
        if value is not None and np.shape(value) == data.shape:
            exclusion |= np.asarray(value, dtype=bool)
    if settings.get("upper_limits", {}).get("exclude_sources", True):
        fwhm = float(science_result.get("fwhm_pixels") or 4.0)
        radius = float(
            settings.get("upper_limits", {}).get(
                "empty_aperture_source_exclusion_fwhm", 3.0
            )
        ) * fwhm
        table = science_result.get("measurements")
        seen = set()
        if table is not None:
            for row in table:
                key = (float(row["x"]), float(row["y"]))
                if key in seen:
                    continue
                seen.add(key)
                slices, weights = _radial_weights(
                    data.shape, key[0], key[1], radius, 1, method="center"
                )
                exclusion[slices] |= weights > 0
    return exclusion


def _empty_aperture_noise(record, science_result, psf_result, method, settings):
    """Measure empirical blank-sky flux scatter for one photometry method."""

    limit_settings = settings.get("upper_limits", {})
    data = _prepared_data(record)
    standard_deviation, _ = _standard_deviation(record, data, settings)
    variance = standard_deviation ** 2
    masks = _measurement_masks(record, data.shape)
    exclusion = _empty_source_mask(record, science_result, settings)
    target = science_result.get("target_diagnostics") or {}
    center_x = _finite_float(target.get("fixed_x"), data.shape[1] / 2.0)
    center_y = _finite_float(target.get("fixed_y"), data.shape[0] / 2.0)
    pixel_scale = _finite_float(science_result.get("pixel_scale_arcsec"))
    local_arcsec = _finite_float(
        limit_settings.get("empty_aperture_local_radius_arcsec")
    )
    local_pixels = (
        local_arcsec / pixel_scale
        if local_arcsec is not None and pixel_scale is not None and pixel_scale > 0
        else max(data.shape)
    )
    fwhm = float(science_result.get("fwhm_pixels") or 4.0)
    aperture_settings = settings.get("apertures", {})
    method_radius = {
        "small_aperture": float(aperture_settings.get("small_radius_fwhm", 1.0)) * fwhm,
        "large_aperture": float(aperture_settings.get("large_radius_fwhm", 2.5)) * fwhm,
        "psf": (
            max(np.shape(psf_result.get("model_native"))) / 2.0
            if psf_result.get("model_native") is not None else fwhm * 2.5
        ),
    }[method]
    margin = max(
        method_radius,
        float(aperture_settings.get("sky_outer_radius_fwhm", 7.0)) * fwhm,
    ) + 1.0
    requested = int(limit_settings.get("number_empty_apertures", 100))
    maximum_attempts = requested * int(
        limit_settings.get("maximum_empty_aperture_attempts_factor", 30)
    )
    seed_text = "{}:{}:{}".format(
        limit_settings.get("empty_aperture_random_seed", 12345),
        science_result.get("image_id"),
        method,
    )
    seed = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    fluxes = []
    positions = []
    x_low = max(margin, center_x - local_pixels)
    x_high = min(data.shape[1] - margin, center_x + local_pixels)
    y_low = max(margin, center_y - local_pixels)
    y_high = min(data.shape[0] - margin, center_y + local_pixels)
    if x_low >= x_high or y_low >= y_high:
        return {
            "noise": None,
            "fluxes": np.array([], dtype=float),
            "positions": np.empty((0, 2), dtype=float),
            "accepted_count": 0,
            "requested_count": requested,
            "status": "FAIL",
        }
    for _ in range(maximum_attempts):
        if len(fluxes) >= requested:
            break
        x = float(rng.uniform(x_low, x_high))
        y = float(rng.uniform(y_low, y_high))
        slices, weights = _radial_weights(
            data.shape, x, y, method_radius, 1, method="center"
        )
        if weights.size == 0 or np.any(exclusion[slices] & (weights > 0)):
            continue
        if (
            limit_settings.get("exclude_masked_regions", True)
            and np.any(masks["combined"][slices] & (weights > 0))
        ):
            continue
        background = _local_background(
            data, standard_deviation, masks["combined"], x, y, fwhm, settings
        )
        source = {"x": x, "y": y, "source_type": "empty"}
        if method == "psf":
            measurement, _ = _psf_measurement(
                data,
                variance,
                masks,
                source,
                background,
                psf_result,
                settings,
                pixel_scale,
            )
        else:
            measurement = _aperture_measurement(
                data,
                variance,
                masks,
                source,
                method_radius,
                background,
                settings,
                method,
            )
        flux = _finite_float(measurement.get("flux"))
        if measurement.get("valid") and flux is not None:
            fluxes.append(flux)
            positions.append((x, y))
    minimum = int(limit_settings.get("minimum_empty_apertures", 20))
    noise = _robust_scatter(fluxes) if len(fluxes) >= minimum else None
    return {
        "noise": noise,
        "fluxes": np.asarray(fluxes, dtype=float),
        "positions": np.asarray(positions, dtype=float),
        "accepted_count": len(fluxes),
        "requested_count": requested,
        "status": "PASS" if noise is not None and noise > 0 else "FAIL",
    }


def _magnitude_limit(zeropoint, flux_limit, exposure):
    """Convert a positive count limit to a calibrated rate magnitude."""

    if None in (zeropoint, flux_limit, exposure):
        return None
    if flux_limit <= 0 or exposure <= 0:
        return None
    return float(zeropoint - 2.5 * np.log10(flux_limit / exposure))


def _limit_products(
    measurements,
    solutions,
    image_records,
    science_results,
    psf_results,
    settings,
):
    """Calculate analytic and empirical empty-aperture limits per method."""

    limit_settings = settings.get("upper_limits", {})
    if not limit_settings.get("enabled", True) or not limit_settings.get(
        "calculate_on_science", True
    ):
        return Table(masked=True)
    sigma_levels = [float(value) for value in limit_settings.get("sigma_levels", [3.0, 5.0])]
    solution_lookup = {
        (str(row["image_id"]), str(row["method"])): row for row in solutions
    }
    record_lookup = {
        _image_id(record, index): record for index, record in enumerate(image_records or [])
    }
    science_lookup = {
        str(result["image_id"]): result for result in (science_results or [])
    }
    if isinstance(psf_results, Mapping):
        psf_lookup = dict(psf_results)
    else:
        psf_lookup = {
            str(result["image_id"]): result for result in (psf_results or [])
        }
    rows = []
    target_rows = measurements[
        np.asarray(measurements["source_type"], dtype=str) == "target"
    ]
    for target_row in target_rows:
        image_id = str(target_row["image_id"])
        method = str(target_row["method"])
        solution = solution_lookup.get((image_id, method))
        zeropoint = None if solution is None else _finite_float(solution["zeropoint_mag"])
        exposure = _finite_float(_science_row_value(target_row, "exposure_time"))
        uncertainty = _finite_float(_science_row_value(target_row, "flux_uncertainty"))
        row = {
            "image_id": image_id,
            "method": method,
            "filter": str(target_row["filter"]),
            "zeropoint_mag": zeropoint,
            "empty_aperture_count": None,
            "empty_aperture_noise": None,
            "empty_aperture_status": "NOT_REQUESTED",
            "flags": "",
        }
        for sigma in sigma_levels:
            label = "{}sigma".format(int(sigma) if sigma.is_integer() else sigma)
            analytic_flux = (
                sigma * uncertainty
                if limit_settings.get("analytic", True) and uncertainty is not None
                else None
            )
            row["analytic_flux_{}".format(label)] = analytic_flux
            row["analytic_limit_{}_mag".format(label)] = _magnitude_limit(
                zeropoint, analytic_flux, exposure
            )
        if (
            limit_settings.get("empty_apertures", True)
            and method in limit_settings.get("empty_aperture_methods", [])
            and image_id in record_lookup
            and image_id in science_lookup
            and image_id in psf_lookup
        ):
            empty = _empty_aperture_noise(
                record_lookup[image_id],
                science_lookup[image_id],
                psf_lookup[image_id],
                method,
                settings,
            )
            row["empty_aperture_count"] = empty["accepted_count"]
            row["empty_aperture_noise"] = empty["noise"]
            row["empty_aperture_status"] = empty["status"]
            if empty["status"] == "FAIL":
                row["flags"] = "EMPTY_APERTURE_LIMIT_FAILED"
            for sigma in sigma_levels:
                label = "{}sigma".format(int(sigma) if sigma.is_integer() else sigma)
                flux_limit = None if empty["noise"] is None else sigma * empty["noise"]
                row["empty_flux_{}".format(label)] = flux_limit
                row["empty_limit_{}_mag".format(label)] = _magnitude_limit(
                    zeropoint, flux_limit, exposure
                )
        else:
            for sigma in sigma_levels:
                label = "{}sigma".format(int(sigma) if sigma.is_integer() else sigma)
                row["empty_flux_{}".format(label)] = None
                row["empty_limit_{}_mag".format(label)] = None
        rows.append(row)
    return _records_table(rows)


def _calibrated_measurements(
    measurements, catalogs, solutions, calibration_records, corrections, limits, settings
):
    """Append instrumental and calibrated quantities without replacing fluxes."""

    output = Table(measurements, masked=True, copy=True)
    solution_lookup = {
        (str(row["image_id"]), str(row["method"])): row for row in solutions
    }
    correction_lookup = {
        (str(row["image_id"]), str(row["method"])): row for row in corrections
    }
    calibration_lookup = {
        int(record["measurement_index"]): record for record in calibration_records
    }
    numeric = {
        name: [] for name in (
            "instrumental_magnitude", "instrumental_magnitude_uncertainty",
            "zeropoint_mag", "zeropoint_uncertainty_mag", "zeropoint_scatter_mag",
            "calibrated_magnitude", "calibrated_magnitude_uncertainty",
            "catalog_magnitude", "catalog_magnitude_error", "catalog_color",
            "catalog_residual_mag", "aperture_correction_mag",
        )
    }
    classifications = []
    calibration_statuses = []
    calibration_catalogs = []
    magnitude_bands = []
    magnitude_systems = []
    calibration_inliers = []
    flags = []
    detection_sigma = float(settings.get("calibration", {}).get("detection_sigma", 3.0))
    available = list(catalogs)
    for index, row in enumerate(output):
        image_id, method = str(row["image_id"]), str(row["method"])
        solution = solution_lookup.get((image_id, method))
        correction = correction_lookup.get((image_id, method))
        flux = _finite_float(_science_row_value(row, "flux"))
        flux_error = _finite_float(_science_row_value(row, "flux_uncertainty"))
        exposure = _finite_float(_science_row_value(row, "exposure_time"))
        instrumental, instrumental_error = _instrumental_magnitude(
            flux, flux_error, exposure
        )
        zeropoint = None if solution is None else _finite_float(solution["zeropoint_mag"])
        zeropoint_error = None if solution is None else _finite_float(solution["zeropoint_uncertainty_mag"])
        scatter = None if solution is None else _finite_float(solution["zeropoint_scatter_mag"])
        calibrated = (
            None if instrumental is None or zeropoint is None
            else instrumental + zeropoint
        )
        calibrated_error = (
            None
            if calibrated is None
            else float(np.hypot(instrumental_error or 0.0, zeropoint_error or 0.0))
        )
        filter_name = normalize_filter_name(_science_row_value(row, "filter"))
        routed, band = route_calibration_catalog(filter_name, settings, available)
        entry = _catalog_entry(
            catalogs, str(row["source_id"]), routed, band, settings
        ) if str(row["source_type"]) != "target" else None
        catalog_magnitude = None if entry is None else entry["catalog_magnitude"]
        numeric["instrumental_magnitude"].append(instrumental)
        numeric["instrumental_magnitude_uncertainty"].append(instrumental_error)
        numeric["zeropoint_mag"].append(zeropoint)
        numeric["zeropoint_uncertainty_mag"].append(zeropoint_error)
        numeric["zeropoint_scatter_mag"].append(scatter)
        numeric["calibrated_magnitude"].append(calibrated)
        numeric["calibrated_magnitude_uncertainty"].append(calibrated_error)
        numeric["catalog_magnitude"].append(catalog_magnitude)
        numeric["catalog_magnitude_error"].append(
            None if entry is None else entry["catalog_magnitude_error"]
        )
        numeric["catalog_color"].append(None if entry is None else entry["catalog_color"])
        numeric["catalog_residual_mag"].append(
            None if calibrated is None or catalog_magnitude is None
            else calibrated - catalog_magnitude
        )
        numeric["aperture_correction_mag"].append(
            None if correction is None else _finite_float(correction["aperture_correction_mag"])
        )
        valid = bool(_science_row_value(row, "valid", False))
        snr = _finite_float(_science_row_value(row, "snr"))
        if not valid or flux is None or flux_error is None:
            classification = "measurement_failure"
        elif flux > 0 and snr is not None and snr >= detection_sigma:
            classification = "detection"
        else:
            classification = "nondetection"
        classifications.append(classification)
        calibration_statuses.append("FAIL" if solution is None else str(solution["status"]))
        calibration_catalogs.append("" if solution is None else str(solution["catalog_name"]))
        magnitude_bands.append("" if solution is None else str(solution["magnitude_band"]))
        magnitude_systems.append("" if solution is None else str(solution["magnitude_system"]))
        calibration_inliers.append(bool(calibration_lookup.get(index, {}).get("inlier", False)))
        row_flags = str(_science_row_value(row, "flags", ""))
        if classification == "nondetection":
            row_flags = _append_flag(row_flags, "NONDETECTION")
        if solution is None or str(solution["status"]) == "FAIL":
            row_flags = _append_flag(row_flags, "CALIBRATION_FAILED")
        flags.append(row_flags)
    _add_numeric_columns(output, numeric)
    output["classification"] = classifications
    output["calibration_status"] = calibration_statuses
    output["calibration_catalog"] = calibration_catalogs
    output["magnitude_band"] = magnitude_bands
    output["magnitude_system"] = magnitude_systems
    output["calibration_inlier"] = calibration_inliers
    output["flags"] = flags
    magnitude_columns = [
        name for name in output.colnames
        if "magnitude" in name or "zeropoint" in name
        or name.endswith("_mag") or name == "catalog_color"
    ]
    for name in magnitude_columns:
        output[name].unit = u.mag
    for name in limits.colnames:
        if name in {"image_id", "method", "filter", "flags", "empty_aperture_status"}:
            continue
        lookup = {
            (str(row["image_id"]), str(row["method"])): _finite_float(row[name])
            for row in limits
        }
        values = [lookup.get((str(row["image_id"]), str(row["method"]))) for row in output]
        _add_numeric_columns(output, {name: values})
        if "limit" in name and name.endswith("_mag"):
            output[name].unit = u.mag
        elif "flux" in name or name == "empty_aperture_noise":
            output[name].unit = getattr(output["flux"], "unit", None)
    output.meta["instrumental_flux_preserved"] = True
    output.meta["calibration_version"] = 1
    return output


def calibrate_photometry(
    measurements,
    catalogs,
    image_records=None,
    science_results=None,
    psf_results=None,
    settings=None,
):
    """Calibrate instrumental photometry and calculate detection limits.

    Zeropoints are solved independently for each image and measurement method.
    The input signed flux table is copied and retained in full. Catalog-star
    residuals are sigma clipped, repeat-variable stars are removed in a second
    pass, and measurement and zeropoint uncertainties are propagated into the
    calibrated magnitude columns.
    """

    if settings is None:
        settings = get_default_settings()
    if not settings.get("calibration", {}).get("enabled", True):
        raise RuntimeError("Photometric calibration is disabled")
    measurements = Table(measurements, masked=True, copy=True)
    catalog_collection = _catalog_collection(catalogs)
    records = _calibration_records(measurements, catalog_collection, settings)
    first_solutions = _solve_all_zeropoints(records, settings)
    unstable = _unstable_calibration_stars(records, settings)
    if unstable:
        for record in records:
            record["inlier"] = False
            record["zeropoint_residual"] = None
            record["calibrated_residual"] = None
            reasons = [
                value for value in record["rejection_reason"].split(";")
                if value and value != "ZEROPOINT_OUTLIER"
            ]
            record["rejection_reason"] = ";".join(reasons)
        solutions_records = _solve_all_zeropoints(records, settings, unstable)
    else:
        solutions_records = first_solutions
    zeropoints = _records_table(solutions_records)
    calibration_stars = _records_table(records)
    corrections = _aperture_corrections(measurements, settings)
    correction_lookup = {
        (str(row["image_id"]), str(row["method"])): row for row in corrections
    }
    if len(zeropoints):
        correction_values = []
        correction_errors = []
        for row in zeropoints:
            correction = correction_lookup.get((str(row["image_id"]), str(row["method"])))
            correction_values.append(
                None if correction is None else _finite_float(correction["aperture_correction_mag"])
            )
            correction_errors.append(
                None if correction is None else _finite_float(correction["aperture_correction_uncertainty_mag"])
            )
        _add_numeric_columns(
            zeropoints,
            {
                "aperture_correction_mag": correction_values,
                "aperture_correction_uncertainty_mag": correction_errors,
            },
        )
    trends = _calibration_trends(records, settings)
    limits = _limit_products(
        measurements,
        zeropoints,
        image_records,
        science_results,
        psf_results,
        settings,
    )
    calibrated = _calibrated_measurements(
        measurements,
        catalog_collection,
        zeropoints,
        records,
        corrections,
        limits,
        settings,
    )
    flux_unit = getattr(measurements["flux"], "unit", None)
    for table in (zeropoints, calibration_stars, corrections, trends, limits):
        for name in table.colnames:
            if (
                "magnitude" in name
                or "zeropoint" in name
                or name.endswith("_mag")
                or name == "catalog_color"
                or name == "slope_mag_per_span"
            ):
                table[name].unit = u.mag
            elif flux_unit is not None and (
                "flux" in name or name == "empty_aperture_noise"
            ):
                table[name].unit = flux_unit
    return {
        "measurements": calibrated,
        "zeropoints": zeropoints,
        "calibration_stars": calibration_stars,
        "aperture_corrections": corrections,
        "trends": trends,
        "limits": limits,
        "unstable_stars": sorted("{}:{}".format(*value) for value in unstable),
        "catalogs_available": sorted(catalog_collection),
        "status": (
            "FAIL"
            if len(zeropoints) == 0 or all(str(value) == "FAIL" for value in zeropoints["status"])
            else "WARN"
            if any(str(value) != "PASS" for value in zeropoints["status"])
            else "PASS"
        ),
        "artificial_star_injection_enabled": bool(
            settings.get("upper_limits", {}).get("injection_recovery", False)
        ),
        "artificial_star_injection_implemented": False,
    }


def save_calibration_products(
    products, output_directory, object_name="field", settings=None, overwrite=None
):
    """Save calibrated measurements, zeropoints, residuals, corrections, and limits."""

    if settings is None:
        settings = get_default_settings()
    if overwrite is None:
        overwrite = settings.get("output", {}).get("overwrite", False)
    calibration = settings.get("calibration", {})
    limits_settings = settings.get("upper_limits", {})
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    stem = _safe_stem(object_name)
    paths = {}
    requests = {
        "measurements": calibration.get("save_calibrated_table", True),
        "zeropoints": calibration.get("save_zeropoints", True),
        "calibration_stars": calibration.get("save_calibration_stars", True),
        "aperture_corrections": calibration.get("save_aperture_corrections", True),
        "trends": calibration.get("save_summary", True),
        "limits": limits_settings.get("save_limit_table", True),
    }
    for name, enabled in requests.items():
        table = products.get(name)
        if not enabled or table is None:
            continue
        path = output_directory / "{}_{}.ecsv".format(stem, name)
        table.write(path, format="ascii.ecsv", overwrite=bool(overwrite))
        paths[name] = str(path)
    if calibration.get("save_summary", True):
        path = output_directory / "{}_calibration_summary.json".format(stem)
        if path.exists() and not overwrite:
            raise FileExistsError(str(path))
        summary = {
            "status": products.get("status"),
            "catalogs_available": products.get("catalogs_available", []),
            "unstable_stars": products.get("unstable_stars", []),
            "artificial_star_injection_enabled": products.get(
                "artificial_star_injection_enabled", False
            ),
            "artificial_star_injection_implemented": False,
        }
        path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        paths["summary"] = str(path)
    return paths


def _difference_image_record(science_record, subtraction_result, settings):
    """Build a non-destructive image record for a generated difference image."""

    difference = subtraction_result.get("difference")
    if difference is None or np.ndim(difference) != 2:
        raise ValueError("A valid two-dimensional difference image is required")
    data = np.asarray(difference, dtype=float)
    aligned = subtraction_result.get("aligned_template") or {}
    wcs = aligned.get("wcs") or _record_wcs(science_record)
    if wcs is None:
        raise ValueError("Difference-image photometry requires a celestial WCS")
    masks = _measurement_masks(science_record, data.shape)
    combined = masks["combined"].copy()
    aligned_mask = aligned.get("mask")
    if aligned_mask is not None and np.shape(aligned_mask) == data.shape:
        combined |= np.asarray(aligned_mask, dtype=bool)
    combined |= ~np.isfinite(data)
    quality = subtraction_result.get("quality") or {}
    rms = _finite_float(quality.get("background_rms"))
    if rms is None or rms <= 0:
        finite = data[~combined]
        rms = _robust_scatter(finite) if finite.size else None
    if rms is None or rms <= 0:
        raise ValueError("The difference image has no usable noise estimate")
    science_ccd = None
    for name in ("prepared_ccd", "working_ccd", "ccd"):
        if science_record.get(name) is not None:
            science_ccd = science_record[name]
            break
    unit = getattr(science_ccd, "unit", u.adu)
    ccd = CCDData(
        data,
        unit=unit,
        wcs=deepcopy(wcs),
        mask=combined,
        uncertainty=StdDevUncertainty(np.full(data.shape, rms, dtype=float)),
    )
    metadata = deepcopy(science_record.get("metadata") or {})
    metadata["filename"] = "{}_difference.fits".format(
        _safe_stem(metadata.get("filename") or _image_id(science_record))
    )
    image_id = _image_id(science_record)
    return {
        "image_id": image_id,
        "prepared_ccd": ccd,
        "metadata": metadata,
        "quality": {
            "background": quality.get("background"),
            "background_rms": rms,
            "fwhm_pixels": (science_record.get("quality") or {}).get("fwhm_pixels"),
        },
        "masks": {"combined": combined},
        "background_products": {
            key: value for key, value in (science_record.get("background_products") or {}).items()
            if key in {"detected_source_mask", "protected_source_mask"}
            and np.shape(value) == data.shape
        },
        "settings": settings,
        "science_record": science_record,
        "subtraction_result": subtraction_result,
    }


def _difference_psf(science_psf, subtraction_result, difference_psf=None):
    """Return the PSF appropriate for the transient signal in the difference."""

    if difference_psf is not None:
        return deepcopy(difference_psf), "supplied_difference_psf"
    parameters = subtraction_result.get("parameters") or {}
    method = subtraction_result.get("method")
    if method == "hotpants" and parameters.get("convolve") == "science":
        fwhm = _finite_float(parameters.get("template_fwhm_pixels"))
        if fwhm is not None:
            size = max(9, int(np.ceil(6.0 * fwhm)))
            if size % 2 == 0:
                size += 1
            sigma = fwhm / 2.354820045
            model = _analytic_array((size, size), "gaussian", sigma)
            model = _normalize_model(model)
            result = deepcopy(science_psf)
            result.update(
                {
                    "model": model,
                    "model_native": model,
                    "model_type": "difference_gaussian",
                    "fwhm_pixels": fwhm,
                    "normalization": float(np.sum(model)),
                    "approved_for_photometry": True,
                    "status": "PASS",
                    "review_state": "DERIVED",
                }
            )
            return result, "analytic_template_seeing"
    return deepcopy(science_psf), "science_psf"


def _target_rows(table):
    """Return the target-only subset of a measurement table."""

    if table is None or len(table) == 0 or "source_type" not in table.colnames:
        return Table(masked=True)
    return table[np.asarray(table["source_type"], dtype=str) == "target"]


def _science_measurement_table(science_photometry):
    """Normalize a science result mapping or direct table to a copied table."""

    if science_photometry is None:
        return Table(masked=True)
    if isinstance(science_photometry, Mapping):
        value = science_photometry.get("measurements")
    else:
        value = science_photometry
    return Table(masked=True) if value is None else Table(value, masked=True, copy=True)


def _zeropoint_lookup(zeropoints, image_id, method):
    """Return a method-specific zeropoint and uncertainty when available."""

    if zeropoints is None:
        return None, None
    if isinstance(zeropoints, Mapping):
        value = zeropoints.get((str(image_id), str(method)), zeropoints.get(str(method)))
        if isinstance(value, Mapping):
            return _finite_float(value.get("zeropoint_mag")), _finite_float(
                value.get("zeropoint_uncertainty_mag")
            )
        return _finite_float(value), None
    for row in zeropoints:
        names = row.colnames
        if (
            "image_id" in names and "method" in names
            and str(row["image_id"]) == str(image_id)
            and str(row["method"]) == str(method)
        ):
            return _finite_float(row["zeropoint_mag"]), _finite_float(
                row["zeropoint_uncertainty_mag"]
                if "zeropoint_uncertainty_mag" in names else None
            )
    return None, None


def _difference_dipole(data, x, y, fwhm, rms, settings):
    """Measure significant positive and negative lobes around the target."""

    configured = settings.get("subtraction", {}).get("photometry", {})
    radius = max(3.0, 2.5 * float(fwhm))
    slices, weights = _radial_weights(data.shape, x, y, radius, 1, method="center")
    patch = np.asarray(data[slices], dtype=float)
    yy, xx = np.indices(patch.shape, dtype=float)
    xx += slices[1].start
    yy += slices[0].start
    inside = (weights > 0) & np.isfinite(patch)
    threshold = float(configured.get("dipole_sigma", 3.0)) * float(rms)
    positive = inside & (patch > threshold)
    negative = inside & (patch < -threshold)
    positive_flux = float(np.sum(patch[positive])) if np.any(positive) else 0.0
    negative_flux = float(np.sum(-patch[negative])) if np.any(negative) else 0.0

    def centroid(selection, values):
        if not np.any(selection) or np.sum(values[selection]) <= 0:
            return None
        total = np.sum(values[selection])
        return (
            float(np.sum(xx[selection] * values[selection]) / total),
            float(np.sum(yy[selection] * values[selection]) / total),
        )

    positive_center = centroid(positive, patch)
    negative_center = centroid(negative, -patch)
    separation = None
    if positive_center is not None and negative_center is not None:
        separation = float(np.hypot(
            positive_center[0] - negative_center[0],
            positive_center[1] - negative_center[1],
        ))
    ratio = min(positive_flux, negative_flux) / max(positive_flux, negative_flux, 1.0e-12)
    detected = bool(
        positive_flux > 0 and negative_flux > 0
        and ratio >= float(configured.get("dipole_ratio_threshold", 0.25))
        and separation is not None
        and separation >= float(configured.get("dipole_minimum_separation_pixels", 0.5))
    )
    return {
        "detected": detected,
        "positive_flux": positive_flux,
        "negative_flux": -negative_flux,
        "absolute_lobe_ratio": float(ratio),
        "separation_pixels": separation,
        "positive_centroid": positive_center,
        "negative_centroid": negative_center,
        "threshold": threshold,
        "slices": slices,
    }


def _annotate_measurement_origin(table, image_kind, host_light_included):
    """Add common provenance and selection columns to a measurement table."""

    result = Table(table, masked=True, copy=True)
    count = len(result)
    result["image_kind"] = [str(image_kind)] * count
    result["host_light_included"] = [bool(host_light_included)] * count
    result["preferred"] = [False] * count
    classifications = (
        [str(value) for value in result["classification"]]
        if "classification" in result.colnames else [""] * count
    )
    result["classification"] = np.asarray(classifications, dtype="U32")
    result["selection_reason"] = np.full(count, "", dtype="U256")
    if "flags" in result.colnames:
        result["flags"] = np.asarray([str(value) for value in result["flags"]], dtype="U1024")
    return result


def _difference_comparison(science_table, difference_table):
    """Compare target fluxes method by method without discarding either table."""

    science = {str(row["method"]): row for row in _target_rows(science_table)}
    difference = {str(row["method"]): row for row in _target_rows(difference_table)}
    rows = []
    for method in sorted(set(science) | set(difference)):
        science_flux = _finite_float(science.get(method)["flux"]) if method in science else None
        science_error = _finite_float(science.get(method)["flux_uncertainty"]) if method in science else None
        difference_flux = _finite_float(difference.get(method)["flux"]) if method in difference else None
        difference_error = _finite_float(difference.get(method)["flux_uncertainty"]) if method in difference else None
        delta = (
            difference_flux - science_flux
            if difference_flux is not None and science_flux is not None else None
        )
        combined_error = (
            np.hypot(difference_error, science_error)
            if difference_error is not None and science_error is not None else None
        )
        rows.append(
            {
                "method": method,
                "science_flux": science_flux,
                "science_uncertainty": science_error,
                "difference_flux": difference_flux,
                "difference_uncertainty": difference_error,
                "difference_minus_science": delta,
                "difference_significance": (
                    delta / combined_error if delta is not None and combined_error not in {None, 0.0}
                    else None
                ),
            }
        )
    return _records_table(rows)


def perform_difference_image_photometry(
    science_record,
    subtraction_result,
    target_solution,
    psf_result,
    measurements=None,
    science_photometry=None,
    zeropoints=None,
    settings=None,
    difference_psf_result=None,
):
    """Perform and validate forced photometry on one difference image.

    All three science-image methods are repeated with the same frozen sky
    coordinate and base schema.  Signed fluxes are retained.  Empirical blank
    apertures can inflate underestimated formal errors but never reduce them.
    The preferred-result decision is stored as provenance rather than removing
    any measurement.
    """

    if settings is None:
        settings = science_record.get("settings") or get_default_settings()
    settings = merge_settings(get_default_settings(), settings)
    configured = settings.get("subtraction", {}).get("photometry", {})
    if not configured.get("enabled", True):
        raise RuntimeError("Difference-image photometry is disabled")
    if not target_solution.get("frozen", False):
        raise ValueError("Difference photometry requires a frozen target coordinate")
    if subtraction_result.get("difference") is None:
        raise ValueError("The subtraction result contains no difference image")
    subtraction_accepted = subtraction_result.get("status") == "PASS"
    difference_flags = []
    if not subtraction_accepted:
        difference_flags.append("DIFFERENCE_SUBTRACTION_REJECTED")
        if configured.get("require_accepted_subtraction", True):
            science_annotated = _annotate_measurement_origin(
                _science_measurement_table(science_photometry), "science", True
            )
            preferred = None
            reason = "subtraction rejected; first valid science method used"
            for method in configured.get("preferred_science_methods", []):
                matches = [
                    index for index, row in enumerate(science_annotated)
                    if str(row["source_type"]) == "target"
                    and str(row["method"]) == method and bool(row["valid"])
                ]
                if not matches:
                    continue
                index = matches[0]
                row = science_annotated[index]
                snr = _finite_float(row["snr"])
                classification = str(row["classification"])
                if not classification:
                    classification = (
                        "detection" if snr is not None and abs(snr) >= float(
                            configured.get("detection_sigma", 3.0)
                        ) else "nondetection"
                    )
                    science_annotated["classification"][index] = classification
                science_annotated["preferred"][index] = True
                science_annotated["selection_reason"][index] = reason
                preferred = {
                    "image_kind": "science", "method": method,
                    "row_index": index, "host_light_included": True,
                    "flux": _finite_float(row["flux"]),
                    "flux_uncertainty": _finite_float(row["flux_uncertainty"]),
                    "snr": snr, "classification": classification,
                    "reason": reason,
                }
                break
            if preferred is None:
                difference_flags.append("PREFERRED_RESULT_UNAVAILABLE")
            return {
                "image_id": _image_id(science_record),
                "status": "WARN" if preferred is not None else "FAIL",
                "flags": difference_flags,
                "measurements": Table(masked=True),
                "all_measurements": science_annotated,
                "science_measurements": science_annotated,
                "comparison": Table(masked=True),
                "limits": Table(masked=True),
                "preferred_result": preferred,
                "selection_rule": reason if preferred is not None else "accepted subtraction required",
                "preferred_host_light_included": None if preferred is None else True,
                "subtraction_result": subtraction_result,
            }
    difference_record = _difference_image_record(science_record, subtraction_result, settings)
    difference_psf, psf_source = _difference_psf(
        psf_result, subtraction_result, difference_psf_result
    )
    local_settings = merge_settings(
        settings,
        {
            "apertures": {"add_poisson_noise_when_needed": False},
            "upper_limits": {
                "minimum_empty_apertures": configured.get("minimum_empty_apertures", 20),
                "calculate_on_difference": True,
            },
        },
    )
    base = perform_science_image_photometry(
        difference_record,
        target_solution,
        difference_psf,
        measurements=measurements,
        settings=local_settings,
        wcs=_record_wcs(difference_record),
    )
    difference_table = _annotate_measurement_origin(
        base["measurements"], "difference", False
    )
    target_indices = [
        index for index, row in enumerate(difference_table)
        if str(row["source_type"]) == "target"
    ]
    empty_results = {}
    uncertainty_validation = {}
    limit_rows = []
    sigma_levels = [
        float(value) for value in settings.get("upper_limits", {}).get("sigma_levels", [3, 5])
    ]
    detection_sigma = float(configured.get("detection_sigma", 3.0))
    calculate_limits = bool(
        settings.get("upper_limits", {}).get("enabled", True)
        and settings.get("upper_limits", {}).get("calculate_on_difference", True)
    )
    uncertainty_failed = False
    for index in target_indices:
        row = difference_table[index]
        method = str(row["method"])
        formal = _finite_float(row["flux_uncertainty"])
        empty = None
        if configured.get("validate_with_empty_apertures", True):
            empty = _empty_aperture_noise(
                difference_record, base, difference_psf, method, local_settings
            )
        empty_results[method] = empty
        empirical = None if empty is None else _finite_float(empty.get("noise"))
        ratio = empirical / formal if empirical is not None and formal not in {None, 0.0} else None
        adopted = formal
        if (
            empirical is not None and formal is not None and empirical > formal
            and configured.get("inflate_underestimated_uncertainties", True)
        ):
            adopted = empirical
        uncertainty_validation[method] = {
            "formal": formal,
            "empirical": empirical,
            "ratio": ratio,
            "adopted": adopted,
        }
        difference_table["flux_uncertainty"][index] = (
            np.ma.masked if adopted is None else adopted
        )
        flux = _finite_float(row["flux"])
        snr = flux / adopted if flux is not None and adopted not in {None, 0.0} else None
        difference_table["snr"][index] = np.ma.masked if snr is None else snr
        flags = list(filter(None, str(row["flags"]).split(";")))
        if ratio is not None and ratio > float(configured.get("uncertainty_warn_ratio", 1.25)):
            flags.append("DIFFERENCE_UNCERTAINTY_UNDERESTIMATED")
            difference_flags.append("DIFFERENCE_UNCERTAINTY_UNDERESTIMATED")
        if ratio is not None and ratio > float(configured.get("uncertainty_fail_ratio", 2.0)):
            uncertainty_failed = True
        classification = (
            "measurement_failure" if not bool(row["valid"]) or flux is None or adopted is None
            else "detection" if abs(snr) >= detection_sigma
            else "nondetection"
        )
        difference_table["classification"][index] = classification
        difference_table["flags"][index] = ";".join(dict.fromkeys(flags))
        zeropoint, zeropoint_error = _zeropoint_lookup(
            zeropoints, row["image_id"], method
        )
        limit_row = {
            "image_id": str(row["image_id"]),
            "method": method,
            "filter": str(row["filter"]),
            "formal_uncertainty": formal,
            "empirical_uncertainty": empirical,
            "uncertainty_ratio": ratio,
            "adopted_uncertainty": adopted,
            "empty_aperture_count": None if empty is None else empty.get("accepted_count"),
            "empty_aperture_status": "NOT_REQUESTED" if empty is None else empty.get("status"),
            "zeropoint_mag": zeropoint,
            "zeropoint_uncertainty_mag": zeropoint_error,
        }
        exposure = _finite_float(row["exposure_time"])
        for sigma in sigma_levels:
            label = "{}sigma".format(int(sigma) if sigma.is_integer() else sigma)
            flux_limit = None if adopted is None else sigma * adopted
            limit_row["flux_{}".format(label)] = flux_limit
            limit_row["limit_{}_mag".format(label)] = _magnitude_limit(
                zeropoint, flux_limit, exposure
            )
        if calculate_limits:
            limit_rows.append(limit_row)
    limits = _records_table(limit_rows)
    flux_unit = getattr(difference_table["flux"], "unit", None)
    for name in limits.colnames:
        if name.endswith("_mag"):
            limits[name].unit = u.mag
        elif flux_unit is not None and (
            "uncertainty" in name or name.startswith("flux_")
        ):
            limits[name].unit = flux_unit
    additions = {
        "formal_flux_uncertainty": [],
        "empirical_flux_uncertainty": [],
        "uncertainty_ratio": [],
        "zeropoint_mag": [],
        "zeropoint_uncertainty_mag": [],
        "calibrated_magnitude": [],
        "calibrated_magnitude_uncertainty": [],
    }
    for row in difference_table:
        method = str(row["method"])
        validation = (
            uncertainty_validation.get(method, {})
            if str(row["source_type"]) == "target" else {}
        )
        formal = validation.get("formal")
        empirical = validation.get("empirical")
        ratio = validation.get("ratio")
        zeropoint, zeropoint_error = _zeropoint_lookup(
            zeropoints, row["image_id"], method
        )
        flux = _finite_float(row["flux"])
        error = _finite_float(row["flux_uncertainty"])
        exposure = _finite_float(row["exposure_time"])
        magnitude = (
            zeropoint - 2.5 * np.log10(flux / exposure)
            if zeropoint is not None and flux is not None and flux > 0
            and exposure is not None and exposure > 0 else None
        )
        magnitude_error = (
            np.hypot(2.5 / np.log(10.0) * error / flux, zeropoint_error or 0.0)
            if magnitude is not None and error is not None else None
        )
        for name, value in (
            ("formal_flux_uncertainty", formal),
            ("empirical_flux_uncertainty", empirical),
            ("uncertainty_ratio", ratio),
            ("zeropoint_mag", zeropoint),
            ("zeropoint_uncertainty_mag", zeropoint_error),
            ("calibrated_magnitude", magnitude),
            ("calibrated_magnitude_uncertainty", magnitude_error),
        ):
            additions[name].append(value)
    _add_numeric_columns(difference_table, additions)
    for index, row in enumerate(difference_table):
        if str(row["classification"]):
            continue
        flux = _finite_float(row["flux"])
        uncertainty = _finite_float(row["flux_uncertainty"])
        snr = flux / uncertainty if flux is not None and uncertainty not in {None, 0.0} else None
        difference_table["classification"][index] = (
            "measurement_failure" if not bool(row["valid"]) or snr is None
            else "detection" if abs(snr) >= detection_sigma
            else "nondetection"
        )
    for name in ("zeropoint_mag", "zeropoint_uncertainty_mag", "calibrated_magnitude", "calibrated_magnitude_uncertainty"):
        difference_table[name].unit = u.mag
    for name in ("formal_flux_uncertainty", "empirical_flux_uncertainty"):
        difference_table[name].unit = difference_table["flux"].unit

    target = base.get("target_diagnostics") or {}
    rms = _finite_float((difference_record.get("quality") or {}).get("background_rms"), 1.0)
    target_x = _finite_float(target.get("fixed_x"))
    target_y = _finite_float(target.get("fixed_y"))
    if target_x is None or target_y is None:
        coordinate = SkyCoord(
            float(target_solution["ra_deg"]), float(target_solution["dec_deg"]),
            unit="deg", frame="icrs",
        )
        target_x, target_y = _record_wcs(difference_record).world_to_pixel(coordinate)
    dipole = _difference_dipole(
        np.asarray(subtraction_result["difference"], dtype=float),
        target_x, target_y, base.get("fwhm_pixels", 4.0),
        rms, settings,
    )
    if dipole["detected"]:
        difference_flags.append("DIFFERENCE_DIPOLE")
    science_table = _science_measurement_table(science_photometry)
    comparison = _difference_comparison(science_table, difference_table)
    if flux_unit is not None:
        for name in comparison.colnames:
            if "flux" in name or "uncertainty" in name or name == "difference_minus_science":
                comparison[name].unit = flux_unit
    inverted = False
    science_by_method = {str(row["method"]): row for row in _target_rows(science_table)}
    for row in _target_rows(difference_table):
        method = str(row["method"])
        science_row = science_by_method.get(method)
        difference_flux = _finite_float(row["flux"])
        difference_error = _finite_float(row["flux_uncertainty"])
        science_flux = None if science_row is None else _finite_float(science_row["flux"])
        if (
            difference_flux is not None and difference_error not in {None, 0.0}
            and science_flux is not None and science_flux > 0
            and difference_flux < -float(configured.get("inverted_residual_sigma", 3.0)) * difference_error
        ):
            inverted = True
    if inverted:
        difference_flags.append("DIFFERENCE_INVERTED_RESIDUAL")
    difference_flags = list(dict.fromkeys(difference_flags))
    if difference_flags:
        for index in target_indices:
            current = str(difference_table["flags"][index])
            difference_table["flags"][index] = ";".join(
                dict.fromkeys(list(filter(None, current.split(";"))) + difference_flags)
            )

    science_annotated = _annotate_measurement_origin(science_table, "science", True)
    preferred = None
    selection_reason = None
    severe = {"DIFFERENCE_DIPOLE", "DIFFERENCE_INVERTED_RESIDUAL"}
    difference_allowed = (
        subtraction_accepted
        and not uncertainty_failed
        and not severe.intersection(difference_flags)
    )
    if configured.get("prefer_difference_when_valid", True) and difference_allowed:
        for method in configured.get("preferred_difference_methods", []):
            matches = [
                index for index in target_indices
                if str(difference_table["method"][index]) == method
                and bool(difference_table["valid"][index])
            ]
            if matches:
                index = matches[0]
                preferred = {
                    "image_kind": "difference", "method": method,
                    "row_index": index, "host_light_included": False,
                }
                selection_reason = (
                    "accepted difference image; first valid method in configured preference order"
                )
                difference_table["preferred"][index] = True
                difference_table["selection_reason"][index] = selection_reason
                break
    if preferred is None and len(science_annotated):
        science_targets = [
            index for index, row in enumerate(science_annotated)
            if str(row["source_type"]) == "target"
        ]
        for method in configured.get("preferred_science_methods", []):
            matches = [
                index for index in science_targets
                if str(science_annotated["method"][index]) == method
                and bool(science_annotated["valid"][index])
            ]
            if matches:
                index = matches[0]
                preferred = {
                    "image_kind": "science", "method": method,
                    "row_index": index, "host_light_included": True,
                }
                selection_reason = (
                    "difference result unavailable or rejected; first valid science method used"
                )
                science_annotated["preferred"][index] = True
                science_annotated["selection_reason"][index] = selection_reason
                break
    if preferred is None:
        difference_flags.append("PREFERRED_RESULT_UNAVAILABLE")
    else:
        table = difference_table if preferred["image_kind"] == "difference" else science_annotated
        selected = table[preferred["row_index"]]
        classification = str(selected["classification"])
        if not classification:
            selected_snr = _finite_float(selected["snr"])
            classification = (
                "measurement_failure" if not bool(selected["valid"])
                else "detection" if selected_snr is not None and abs(selected_snr) >= detection_sigma
                else "nondetection"
            )
            table["classification"][preferred["row_index"]] = classification
        preferred.update(
            {
                "flux": _finite_float(selected["flux"]),
                "flux_uncertainty": _finite_float(selected["flux_uncertainty"]),
                "snr": _finite_float(selected["snr"]),
                "classification": classification,
                "reason": selection_reason,
            }
        )
    all_measurements = (
        vstack([science_annotated, difference_table], metadata_conflicts="silent")
        if len(science_annotated) else difference_table
    )
    base.update(
        {
            "status": (
                "FAIL" if preferred is None
                else "WARN" if difference_flags else "PASS"
            ),
            "flags": list(dict.fromkeys(difference_flags)),
            "measurements": difference_table,
            "all_measurements": all_measurements,
            "science_measurements": science_annotated,
            "comparison": comparison,
            "limits": limits,
            "empty_apertures": empty_results,
            "uncertainty_validation": uncertainty_validation,
            "dipole": dipole,
            "inverted_residual": inverted,
            "preferred_result": preferred,
            "selection_rule": selection_reason,
            "preferred_host_light_included": (
                None if preferred is None else preferred["host_light_included"]
            ),
            "difference_psf_source": psf_source,
            "difference_record": difference_record,
            "subtraction_result": subtraction_result,
        }
    )
    return base


def perform_difference_photometry(
    image_records,
    subtraction_results,
    target_solution,
    psf_results,
    measurements=None,
    science_results=None,
    zeropoints=None,
    settings=None,
    difference_psf_results=None,
    continue_on_error=True,
):
    """Perform difference-image photometry for a complete image batch."""

    if settings is None:
        settings = get_default_settings()
    subtraction_lookup = {
        str(item["image_id"]): item for item in subtraction_results
    }
    psf_lookup = dict(psf_results) if isinstance(psf_results, Mapping) else {
        str(item["image_id"]): item for item in psf_results
    }
    science_lookup = {
        str(item["image_id"]): item for item in (science_results or [])
    }
    difference_psf_lookup = (
        dict(difference_psf_results) if isinstance(difference_psf_results, Mapping)
        else {str(item["image_id"]): item for item in (difference_psf_results or [])}
    )
    results = []
    tables = []
    for index, record in enumerate(image_records):
        image_id = _image_id(record, index)
        try:
            if image_id not in subtraction_lookup:
                raise ValueError(
                    "No subtraction result is available for {}".format(image_id)
                )
            if image_id not in psf_lookup:
                raise ValueError("No PSF result is available for {}".format(image_id))
            result = perform_difference_image_photometry(
                record,
                subtraction_lookup[image_id],
                target_solution,
                psf_lookup[image_id],
                measurements,
                science_lookup.get(image_id),
                zeropoints,
                record.get("settings") or settings,
                difference_psf_lookup.get(image_id),
            )
        except Exception as error:
            if not continue_on_error:
                raise
            result = {
                "image_id": image_id,
                "status": "FAIL",
                "flags": ["DIFFERENCE_PHOTOMETRY_FAILED"],
                "error": str(error),
                "measurements": Table(masked=True),
                "all_measurements": _annotate_measurement_origin(
                    _science_measurement_table(science_lookup.get(image_id)),
                    "science", True,
                ),
                "comparison": Table(masked=True),
                "limits": Table(masked=True),
                "preferred_result": None,
                "preferred_host_light_included": None,
            }
        results.append(result)
        if len(result.get("measurements", [])):
            tables.append(result["measurements"])
    combined = vstack(tables, metadata_conflicts="silent") if tables else Table(masked=True)
    return combined, results


def save_difference_photometry_products(
    result, output_directory, settings=None, overwrite=None
):
    """Save difference measurements, limits, comparisons, and selection summary."""

    if settings is None:
        settings = get_default_settings()
    configured = settings.get("subtraction", {}).get("photometry", {})
    if overwrite is None:
        overwrite = settings.get("output", {}).get("overwrite", False)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    stem = _safe_stem(result.get("filename") or result.get("image_id"))
    paths = {}
    requests = {
        "difference_photometry": ("measurements", configured.get("save_measurements", True)),
        "all_photometry": ("all_measurements", configured.get("save_measurements", True)),
        "science_difference_comparison": ("comparison", configured.get("save_comparison", True)),
        "difference_limits": ("limits", configured.get("save_limits", True)),
    }
    for suffix, (key, enabled) in requests.items():
        table = result.get(key)
        if not enabled or table is None:
            continue
        path = output / "{}_{}.ecsv".format(stem, suffix)
        table.write(path, format="ascii.ecsv", overwrite=bool(overwrite))
        paths[key] = str(path)
    if configured.get("save_summary", True):
        path = output / "{}_difference_photometry.json".format(stem)
        if path.exists() and not overwrite:
            raise FileExistsError(str(path))
        summary = {
            "image_id": result.get("image_id"),
            "status": result.get("status"),
            "flags": result.get("flags", []),
            "preferred_result": result.get("preferred_result"),
            "selection_rule": result.get("selection_rule"),
            "preferred_host_light_included": result.get(
                "preferred_host_light_included"
            ),
            "difference_psf_source": result.get("difference_psf_source"),
            "dipole": {
                key: value for key, value in (result.get("dipole") or {}).items()
                if key != "slices"
            },
            "inverted_residual": result.get("inverted_residual"),
        }
        path.write_text(json.dumps(summary, indent=2, default=str, sort_keys=True) + "\n")
        paths["summary"] = str(path)
    return paths


__all__ = [
    "PSF_DOWNSTREAM_STAGES",
    "apply_psf_review",
    "calibrate_photometry",
    "construct_psf",
    "construct_psfs",
    "perform_science_image_photometry",
    "perform_science_photometry",
    "perform_difference_image_photometry",
    "perform_difference_photometry",
    "plan_psf_rerun",
    "psf_dependency_signature",
    "require_approved_psf",
    "route_calibration_catalog",
    "save_calibration_products",
    "save_difference_photometry_products",
    "save_science_photometry_products",
    "save_psf_products",
]
