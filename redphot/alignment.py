"""Relative alignment, derived detection stacks, and fixed target positions.

All WCS refinements and resampling products are derived in memory.  The
science arrays supplied by the caller are never modified or resampled, which
keeps direct aperture and PSF photometry on the original detector pixels.
"""

from copy import deepcopy
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.stats import SigmaClip, sigma_clipped_stats
from astropy.table import MaskedColumn, Table
from astropy.wcs.utils import proj_plane_pixel_scales

from .catalogs import refine_wcs_from_matches
from .config import get_default_settings


def _image_id(record, index):
    """Return a stable image identifier from a processing record."""

    metadata = record.get("metadata") or {}
    return str(
        record.get("image_id")
        or metadata.get("filename")
        or "image_{:04d}".format(index)
    )


def _finite_float(value):
    """Return a finite float or ``None`` for a missing scalar."""

    if value is None or np.ma.is_masked(value):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _decision_map(decisions):
    """Normalize list, table, or mapping decisions by image ID."""

    if decisions is None:
        return {}
    if isinstance(decisions, dict):
        if all(isinstance(value, dict) for value in decisions.values()):
            return {str(key): value for key, value in decisions.items()}
        decisions = [decisions]
    output = {}
    for row in decisions:
        image_id = row.get("image_id") if hasattr(row, "get") else row["image_id"]
        output[str(image_id)] = row
    return output


def _decision_value(decision, name, default=None):
    """Read an unmasked value from a dictionary or Astropy row."""

    if decision is None:
        return default
    try:
        value = decision.get(name, default)
    except AttributeError:
        value = decision[name] if name in decision.colnames else default
    return default if value is None or np.ma.is_masked(value) else value


def _status_allowed(decision, allowed, require_approval):
    """Return whether one saved usability decision permits an operation."""

    if decision is None:
        return not require_approval
    status = str(_decision_value(decision, "status", "WARN"))
    use_image = bool(_decision_value(decision, "use_image", status != "FAIL"))
    return use_image and status in set(allowed)


def _record_wcs(record):
    """Return the best available derived celestial WCS from a record."""

    wcs = record.get("refined_wcs") or record.get("wcs")
    if wcs is None and record.get("ccd") is not None:
        wcs = getattr(record["ccd"], "wcs", None)
    if wcs is None or not getattr(wcs, "has_celestial", False):
        return None
    return deepcopy(wcs.celestial)


def _measurement_rows(measurements, image_id, astrometry_only=True):
    """Return unique common-star rows for one image."""

    if measurements is None or len(measurements) == 0:
        return Table(masked=True)
    rows = measurements[
        np.asarray(measurements["image_id"], dtype=str) == str(image_id)
    ]
    if astrometry_only and "role_astrometry" in rows.colnames:
        rows = rows[np.asarray(rows["role_astrometry"], dtype=bool)]
    identities = set()
    keep = []
    for index, value in enumerate(rows["persistent_id"]):
        identity = str(value)
        if identity not in identities:
            identities.add(identity)
            keep.append(index)
    return rows[keep]


def _reference_score(record, rows, decision):
    """Return a sortable score favoring well-sampled sharp exposures."""

    quality = record.get("quality") or {}
    astrometry = record.get("astrometry") or {}
    status = str(_decision_value(decision, "status", "WARN"))
    status_rank = {"PASS": 0, "WARN": 1, "FAIL": 2}.get(status, 2)
    rms = _finite_float(astrometry.get("refined_rms_arcsec"))
    seeing = _finite_float(quality.get("fwhm_arcsec"))
    noise = _finite_float(quality.get("background_rms"))
    return (
        status_rank,
        np.inf if rms is None else rms,
        np.inf if seeing is None else seeing,
        np.inf if noise is None else noise,
        -len(rows),
    )


def _relative_match_table(current_rows, reference_rows, current_wcs, reference_wcs):
    """Build a catalog-style match table from persistent common stars."""

    reference_lookup = {
        str(row["persistent_id"]): row for row in reference_rows
    }
    current_lookup = {str(row["persistent_id"]): row for row in current_rows}
    common = sorted(set(reference_lookup) & set(current_lookup))
    table = Table(masked=True)
    if not common:
        return table
    x = np.asarray([float(current_lookup[key]["x"]) for key in common])
    y = np.asarray([float(current_lookup[key]["y"]) for key in common])
    reference_x = np.asarray(
        [float(reference_lookup[key]["x"]) for key in common]
    )
    reference_y = np.asarray(
        [float(reference_lookup[key]["y"]) for key in common]
    )
    reference_sky = reference_wcs.pixel_to_world(reference_x, reference_y).icrs
    expected_x, expected_y = current_wcs.world_to_pixel(reference_sky)
    measured_sky = current_wcs.pixel_to_world(x, y).icrs
    longitude, latitude = measured_sky.spherical_offsets_to(reference_sky)
    separation = measured_sky.separation(reference_sky)
    table["persistent_id"] = common
    table["x"] = x * u.pixel
    table["y"] = y * u.pixel
    table["catalog_x_original"] = np.asarray(expected_x) * u.pixel
    table["catalog_y_original"] = np.asarray(expected_y) * u.pixel
    table["catalog_ra"] = reference_sky.ra.deg * u.deg
    table["catalog_dec"] = reference_sky.dec.deg * u.deg
    table["separation_original_arcsec"] = separation.arcsec * u.arcsec
    table["residual_ra_original_arcsec"] = longitude.arcsec * u.arcsec
    table["residual_dec_original_arcsec"] = latitude.arcsec * u.arcsec
    table["inlier"] = np.ones(len(table), dtype=bool)
    return table


def _relative_astrometry_settings(settings):
    """Translate relative-alignment settings into the WCS fitter interface."""

    output = deepcopy(settings)
    target = output.get("target_position", {})
    astrometry = output.setdefault("astrometry", {})
    astrometry.update(
        {
            "minimum_matches": int(
                target.get("relative_alignment_minimum_common_stars", 6)
            ),
            "sigma_clip": float(target.get("relative_alignment_sigma_clip", 4.0)),
            "maximum_iterations": int(
                target.get("relative_alignment_maximum_iterations", 3)
            ),
            "target_rms_arcsec": float(
                target.get("relative_alignment_target_rms_arcsec", 0.20)
            ),
            "warning_rms_arcsec": float(
                target.get("relative_alignment_warn_rms_arcsec", 0.50)
            ),
            "maximum_translation_pixels": float(
                target.get("relative_alignment_maximum_translation_pixels", 20.0)
            ),
            "maximum_rotation_degrees": float(
                target.get("relative_alignment_maximum_rotation_degrees", 2.0)
            ),
            "maximum_scale_change_fraction": float(
                target.get(
                    "relative_alignment_maximum_scale_change_fraction", 0.02
                )
            ),
            "minimum_improvement_fraction": 0.01,
            "fit_translation": True,
            "fit_rotation": True,
            "fit_scale": True,
            "reject_unsafe_solution": True,
            "refine_wcs": bool(
                target.get("relative_alignment_enabled", True)
            ),
        }
    )
    return output


def refine_relative_alignment(image_records, measurements, usability_decisions=None,
                              settings=None):
    """Refine image WCS solutions against one common-star reference exposure.

    The returned WCS objects are independent copies. Science pixels and the
    WCS objects attached to input CCDData instances are not modified.

    Returns
    -------
    alignments : list of dict
        Per-image derived WCS, residuals, eligibility, and quality information.
    residuals : astropy.table.Table
        One row per common-star residual after relative refinement.
    """

    if settings is None:
        settings = get_default_settings()
    target_settings = settings.get("target_position", {})
    decisions = _decision_map(usability_decisions)
    require = bool(target_settings.get("require_usability_approval", True))
    coordinate_statuses = target_settings.get(
        "coordinate_allowed_statuses", ["PASS", "WARN"]
    )
    stack_statuses = target_settings.get("stack_allowed_statuses", ["PASS", "WARN"])
    candidates = []
    rows_by_image = {}
    for index, record in enumerate(image_records):
        image_id = _image_id(record, index)
        rows = _measurement_rows(measurements, image_id)
        rows_by_image[image_id] = rows
        decision = decisions.get(image_id)
        wcs = _record_wcs(record)
        if wcs is not None and _status_allowed(
            decision, coordinate_statuses, require
        ):
            candidates.append(
                (_reference_score(record, rows, decision), index, image_id)
            )
    if not candidates:
        raise ValueError(
            "No usability-approved image with a celestial WCS is available "
            "for relative alignment"
        )
    _, reference_index, reference_id = min(candidates)
    reference_record = image_records[reference_index]
    reference_wcs = _record_wcs(reference_record)
    reference_rows = rows_by_image[reference_id]
    fit_settings = _relative_astrometry_settings(settings)
    minimum = int(
        target_settings.get("relative_alignment_minimum_common_stars", 6)
    )
    warning_rms = float(
        target_settings.get("relative_alignment_warn_rms_arcsec", 0.50)
    )
    failure_rms = float(
        target_settings.get("relative_alignment_fail_rms_arcsec", 1.50)
    )
    alignments = []
    residual_rows = []

    for index, record in enumerate(image_records):
        image_id = _image_id(record, index)
        decision = decisions.get(image_id)
        original_wcs = _record_wcs(record)
        coordinate_allowed = _status_allowed(
            decision, coordinate_statuses, require
        )
        stack_allowed = _status_allowed(decision, stack_statuses, require)
        info = {
            "image_id": image_id,
            "reference_image_id": reference_id,
            "is_reference": image_id == reference_id,
            "common_star_count": 0,
            "inlier_count": 0,
            "rejected_match_count": 0,
            "original_rms_arcsec": None,
            "refined_rms_arcsec": None,
            "translation_x_pixels": None,
            "translation_y_pixels": None,
            "rotation_degrees": None,
            "scale": None,
            "refinement_adopted": False,
            "status": "PASS",
            "flags": [],
            "coordinate_eligible": False,
            "stack_eligible": False,
            "wcs": original_wcs,
            "original_wcs_header": None,
            "refined_wcs_header": None,
        }
        if original_wcs is None:
            info["status"] = "FAIL"
            info["flags"].append("RELATIVE_ALIGNMENT_FAILED")
            alignments.append(info)
            continue
        info["original_wcs_header"] = original_wcs.to_header(relax=True)
        if image_id == reference_id:
            info.update(
                {
                    "common_star_count": len(reference_rows),
                    "inlier_count": len(reference_rows),
                    "original_rms_arcsec": 0.0,
                    "refined_rms_arcsec": 0.0,
                    "scale": 1.0,
                    "refined_wcs_header": original_wcs.to_header(relax=True),
                }
            )
        else:
            matches = _relative_match_table(
                rows_by_image[image_id], reference_rows, original_wcs, reference_wcs
            )
            info["common_star_count"] = len(matches)
            if len(matches) < minimum:
                info["status"] = "FAIL"
                info["flags"].append("RELATIVE_ALIGNMENT_FAILED")
            else:
                shape = record["ccd"].shape if record.get("ccd") is not None else None
                refined_wcs, matches, fit_info = refine_wcs_from_matches(
                    original_wcs,
                    matches,
                    settings=fit_settings,
                    shape=shape,
                )
                info["wcs"] = refined_wcs
                for name in (
                    "inlier_count", "rejected_match_count", "original_rms_arcsec",
                    "refined_rms_arcsec", "translation_x_pixels",
                    "translation_y_pixels", "rotation_degrees", "scale",
                    "refinement_adopted", "original_wcs_header",
                    "refined_wcs_header",
                ):
                    info[name] = fit_info.get(name)
                rms = info["refined_rms_arcsec"]
                if rms is None or rms > failure_rms:
                    info["status"] = "FAIL"
                    info["flags"].append("RELATIVE_ALIGNMENT_FAILED")
                elif rms > warning_rms:
                    info["status"] = "WARN"
                    info["flags"].append("RELATIVE_ALIGNMENT_POOR")
                for match in matches:
                    residual_rows.append(
                        {
                            "image_id": image_id,
                            "reference_image_id": reference_id,
                            "persistent_id": str(match["persistent_id"]),
                            "x": float(match["x"]),
                            "y": float(match["y"]),
                            "residual_ra_arcsec": _finite_float(
                                match["residual_ra_final_arcsec"]
                            ),
                            "residual_dec_arcsec": _finite_float(
                                match["residual_dec_final_arcsec"]
                            ),
                            "separation_arcsec": _finite_float(
                                match["separation_final_arcsec"]
                            ),
                            "inlier": bool(match["inlier"]),
                        }
                    )
        info["coordinate_eligible"] = bool(
            coordinate_allowed and info["status"] != "FAIL"
        )
        info["stack_eligible"] = bool(stack_allowed and info["status"] != "FAIL")
        alignments.append(info)

    residuals = Table(rows=residual_rows, masked=True)
    for name, unit in (
        ("x", u.pixel), ("y", u.pixel),
        ("residual_ra_arcsec", u.arcsec),
        ("residual_dec_arcsec", u.arcsec),
        ("separation_arcsec", u.arcsec),
    ):
        if name in residuals.colnames:
            residuals[name].unit = unit
    return alignments, residuals


def alignment_table(alignments):
    """Convert alignment dictionaries to a persistent scalar table."""

    names = (
        "image_id", "reference_image_id", "is_reference", "common_star_count",
        "inlier_count", "rejected_match_count", "original_rms_arcsec",
        "refined_rms_arcsec", "translation_x_pixels", "translation_y_pixels",
        "rotation_degrees", "scale", "refinement_adopted", "status",
        "coordinate_eligible", "stack_eligible",
    )
    rows = []
    for alignment in alignments:
        row = {name: alignment.get(name) for name in names}
        row["flags"] = ";".join(alignment.get("flags", []))
        rows.append(row)
    table = Table(rows=rows, masked=True)
    for name, unit in (
        ("original_rms_arcsec", u.arcsec),
        ("refined_rms_arcsec", u.arcsec),
        ("translation_x_pixels", u.pixel),
        ("translation_y_pixels", u.pixel),
        ("rotation_degrees", u.deg),
    ):
        if name in table.colnames:
            values = np.asarray(
                [
                    np.nan if _finite_float(row.get(name)) is None else row[name]
                    for row in rows
                ],
                dtype=float,
            )
            table.replace_column(
                name,
                MaskedColumn(
                    np.where(np.isfinite(values), values, 0.0),
                    mask=~np.isfinite(values),
                    name=name,
                    unit=unit,
                ),
            )
    return table


def _reproject_derived_array(data, input_wcs, output_wcs, output_shape, mask,
                             order, tile_rows):
    """Map one derived array onto a reference WCS in row-limited tiles."""

    from scipy.ndimage import map_coordinates

    data = np.asarray(data, dtype=float)
    invalid = ~np.isfinite(data)
    if mask is not None:
        invalid |= np.asarray(mask, dtype=bool)
    output = np.full(output_shape, np.nan, dtype=float)
    valid_output = np.zeros(output_shape, dtype=bool)
    x_grid = np.arange(output_shape[1], dtype=float)
    for y_start in range(0, output_shape[0], tile_rows):
        y_stop = min(output_shape[0], y_start + tile_rows)
        yy, xx = np.meshgrid(
            np.arange(y_start, y_stop, dtype=float), x_grid, indexing="ij"
        )
        sky = output_wcs.pixel_to_world(xx, yy)
        input_x, input_y = input_wcs.world_to_pixel(sky)
        inside = (
            np.isfinite(input_x)
            & np.isfinite(input_y)
            & (input_x >= 0)
            & (input_y >= 0)
            & (input_x <= data.shape[1] - 1)
            & (input_y <= data.shape[0] - 1)
        )
        sampled = map_coordinates(
            np.where(invalid, 0.0, data),
            [input_y, input_x],
            order=order,
            mode="constant",
            cval=0.0,
            prefilter=order > 1,
        )
        sampled_invalid = map_coordinates(
            invalid.astype(float),
            [input_y, input_x],
            order=0,
            mode="constant",
            cval=1.0,
            prefilter=False,
        ) > 0.0
        good = inside & ~sampled_invalid
        tile = output[y_start:y_stop]
        tile[good] = sampled[good]
        valid_output[y_start:y_stop] = good
    return output, valid_output


def _robust_background_rms(data, mask=None):
    """Return sigma-clipped background and RMS for a derived plane."""

    invalid = ~np.isfinite(data)
    if mask is not None:
        invalid |= np.asarray(mask, dtype=bool)
    _, median, rms = sigma_clipped_stats(
        data, mask=invalid, sigma=3.0, maxiters=5
    )
    return _finite_float(median), _finite_float(rms)


def _stack_normalization(record, decision, reference_zeropoint, settings):
    """Return multiplicative scale and provenance for one stack input."""

    target = settings.get("target_position", {})
    mode = target.get("stack_normalization", "zeropoint")
    metadata = record.get("metadata") or {}
    exposure = _finite_float(metadata.get("exposure_time"))
    zeropoint = _finite_float(_decision_value(decision, "zeropoint_mag"))
    scale = 1.0
    applied = "none"
    if mode in {"exposure", "zeropoint"} and exposure is not None and exposure > 0:
        scale /= exposure
        applied = "exposure"
    if (
        mode == "zeropoint"
        and zeropoint is not None
        and reference_zeropoint is not None
    ):
        scale *= 10.0 ** (0.4 * (reference_zeropoint - zeropoint))
        applied = "zeropoint"
    return scale, applied


def _combine_stack_planes(planes, weights, sigma, method):
    """Robustly combine reprojected derived planes."""

    cube = np.asarray(planes, dtype=float)
    valid = np.isfinite(cube)
    if cube.shape[0] >= 3 and sigma is not None:
        clipped = SigmaClip(sigma=float(sigma), maxiters=3)(
            cube, axis=0, masked=True
        )
        valid &= ~np.ma.getmaskarray(clipped)
    coverage = np.count_nonzero(valid, axis=0).astype(np.int16)
    if method == "median":
        with np.errstate(all="ignore"):
            combined = np.nanmedian(np.where(valid, cube, np.nan), axis=0)
        weight_map = coverage.astype(float)
    else:
        weight_cube = np.asarray(weights, dtype=float)[:, None, None] * valid
        total_weight = np.sum(weight_cube, axis=0)
        numerator = np.nansum(np.where(valid, cube, 0.0) * weight_cube, axis=0)
        combined = np.full(total_weight.shape, np.nan, dtype=float)
        np.divide(numerator, total_weight, out=combined, where=total_weight > 0)
        weight_map = total_weight
    combined[coverage == 0] = np.nan
    return combined, weight_map, coverage


def build_detection_stacks(image_records, alignments, usability_decisions=None,
                           settings=None):
    """Build reprojected per-filter and normalized multi-filter detection stacks.

    Only temporary stack inputs are reprojected. Every ``ccd.data`` array in
    ``image_records`` remains unchanged and on its native detector grid.
    """

    if settings is None:
        settings = get_default_settings()
    target = settings.get("target_position", {})
    if target.get("resample_science_images", False):
        raise ValueError("Direct science-image resampling is not permitted")
    if not target.get("build_detection_stack", True):
        return {}
    decisions = _decision_map(usability_decisions)
    alignment_lookup = {item["image_id"]: item for item in alignments}
    reference = next((item for item in alignments if item.get("is_reference")), None)
    if reference is None or reference.get("wcs") is None:
        raise ValueError("A valid relative-alignment reference is required")
    reference_record = next(
        record for index, record in enumerate(image_records)
        if _image_id(record, index) == reference["image_id"]
    )
    output_wcs = deepcopy(reference["wcs"])
    output_shape = tuple(reference_record["ccd"].shape)
    grouped = {}
    for index, record in enumerate(image_records):
        image_id = _image_id(record, index)
        alignment = alignment_lookup.get(image_id)
        if (
            alignment is None
            or not alignment.get("stack_eligible", False)
            or alignment.get("wcs") is None
            or record.get("ccd") is None
        ):
            continue
        filter_name = str((record.get("metadata") or {}).get("filter") or "unknown")
        grouped.setdefault(filter_name, []).append((record, alignment, image_id))

    products = {}
    minimum = int(target.get("stack_minimum_images_per_filter", 1))
    maximum = target.get("stack_maximum_images_per_filter")
    order = int(target.get("stack_reprojection_order", 1))
    tile_rows = int(target.get("stack_reprojection_tile_rows", 256))
    method = target.get("stack_combine", "weighted_mean")
    sigma = target.get("stack_sigma_clip", 4.0)
    for filter_name, group in grouped.items():
        group = sorted(
            group,
            key=lambda item: (
                np.inf if _finite_float((item[0].get("quality") or {}).get("fwhm_arcsec")) is None
                else float((item[0].get("quality") or {}).get("fwhm_arcsec")),
                item[2],
            ),
        )
        if maximum is not None:
            group = group[: int(maximum)]
        if len(group) < minimum:
            continue
        zeropoints = [
            _finite_float(_decision_value(decisions.get(image_id), "zeropoint_mag"))
            for _, _, image_id in group
        ]
        finite_zeropoints = [value for value in zeropoints if value is not None]
        reference_zeropoint = (
            float(np.median(finite_zeropoints)) if finite_zeropoints else None
        )
        planes = []
        weights = []
        contributors = []
        for record, alignment, image_id in group:
            ccd = record["ccd"]
            scale, normalization = _stack_normalization(
                record, decisions.get(image_id), reference_zeropoint, settings
            )
            data = np.asarray(ccd.data, dtype=float) * scale
            mask = getattr(ccd, "mask", None)
            background, rms = _robust_background_rms(data, mask)
            if background is not None:
                data = data - background
            plane, valid = _reproject_derived_array(
                data,
                alignment["wcs"],
                output_wcs,
                output_shape,
                mask,
                order,
                tile_rows,
            )
            plane[~valid] = np.nan
            weight = 1.0
            if target.get("stack_use_inverse_variance_weights", True) and rms not in {None, 0.0}:
                weight = 1.0 / rms ** 2
            planes.append(plane)
            weights.append(weight)
            contributors.append(
                {
                    "image_id": image_id,
                    "scale": float(scale),
                    "normalization": normalization,
                    "background": background,
                    "background_rms": rms,
                    "weight": float(weight),
                    "relative_alignment_rms_arcsec": _finite_float(
                        alignment.get("refined_rms_arcsec")
                    ),
                    "absolute_wcs_rms_arcsec": _finite_float(
                        (record.get("astrometry") or {}).get(
                            "refined_rms_arcsec"
                        )
                    ),
                }
            )
        combined, weight_map, coverage = _combine_stack_planes(
            planes, weights, sigma, method
        )
        _, combined_rms = _robust_background_rms(combined)
        products[filter_name] = {
            "kind": "per_filter",
            "filter": filter_name,
            "data": combined,
            "weight": weight_map,
            "coverage": coverage,
            "wcs": deepcopy(output_wcs),
            "shape": output_shape,
            "contributors": contributors,
            "reference_image_id": reference["image_id"],
            "reference_zeropoint_mag": reference_zeropoint,
            "background_rms": combined_rms,
            "combine_method": method,
            "alignment_uncertainty_arcsec": float(
                np.median(
                    [
                        np.hypot(
                            item.get("relative_alignment_rms_arcsec") or 0.0,
                            item.get("absolute_wcs_rms_arcsec") or 0.0,
                        )
                        for item in contributors
                    ]
                )
            ),
        }

    if target.get("build_multifilter_stack", True) and len(products) >= 2:
        planes = []
        filters = []
        for filter_name, product in products.items():
            plane = np.asarray(product["data"], dtype=float)
            _, rms = _robust_background_rms(plane)
            if target.get("multifilter_normalization", "background_rms") == "background_rms":
                if rms is None or rms <= 0:
                    continue
                plane = plane / rms
            planes.append(plane)
            filters.append(filter_name)
        if len(planes) >= 2:
            combined, weight_map, coverage = _combine_stack_planes(
                planes, np.ones(len(planes)), sigma, method
            )
            _, combined_rms = _robust_background_rms(combined)
            products["multifilter"] = {
                "kind": "multifilter",
                "filter": "multifilter",
                "data": combined,
                "weight": weight_map,
                "coverage": coverage,
                "wcs": deepcopy(output_wcs),
                "shape": output_shape,
                "contributors": filters,
                "reference_image_id": reference["image_id"],
                "background_rms": combined_rms,
                "combine_method": method,
                "alignment_uncertainty_arcsec": float(
                    np.median(
                        [
                            products[name].get(
                                "alignment_uncertainty_arcsec", 0.0
                            )
                            for name in filters
                        ]
                    )
                ),
            }
    if not target.get("build_per_filter_stacks", True):
        products = {
            name: product for name, product in products.items()
            if name == "multifilter"
        }
    return products


def _coordinate_prior(image_records, settings, prior=None):
    """Resolve a user/discovery or metadata coordinate prior."""

    if isinstance(prior, SkyCoord):
        return prior.icrs, "supplied_prior"
    if prior is not None:
        if isinstance(prior, dict):
            ra, dec = prior.get("ra"), prior.get("dec")
        else:
            ra, dec = prior
        return SkyCoord(float(ra), float(dec), unit="deg", frame="icrs"), "supplied_prior"
    target = settings.get("target_position", {})
    if target.get("ra") is not None and target.get("dec") is not None:
        coordinate = SkyCoord(
            target["ra"], target["dec"],
            unit=target.get("coordinate_unit", ["hourangle", "deg"]),
            frame="icrs",
        )
        return coordinate, "user"
    for record in image_records:
        metadata = record.get("metadata") or {}
        for ra_name, dec_name, source in (
            ("adopted_ra_deg", "adopted_dec_deg", "adopted_metadata"),
            ("header_target_ra_deg", "header_target_dec_deg", "header_target"),
        ):
            ra = _finite_float(metadata.get(ra_name))
            dec = _finite_float(metadata.get(dec_name))
            if ra is not None and dec is not None:
                return SkyCoord(ra, dec, unit="deg", frame="icrs"), source
    raise ValueError("A user, discovery, or metadata target-position prior is required")


def _pixel_scale_arcsec(wcs):
    """Return the mean celestial pixel scale in arcseconds."""

    scales = proj_plane_pixel_scales(wcs.celestial) * 3600.0
    value = float(np.mean(scales))
    return value if np.isfinite(value) and value > 0 else None


def _centroid_candidate(data, wcs, prior, fwhm_pixels, settings, source,
                        filter_name=None, mask=None,
                        wcs_uncertainty_arcsec=0.0):
    """Measure a guarded diagnostic centroid near a coordinate prior."""

    from scipy.ndimage import gaussian_filter

    target = settings.get("target_position", {})
    data = np.asarray(data, dtype=float)
    invalid = ~np.isfinite(data)
    if mask is not None:
        invalid |= np.asarray(mask, dtype=bool)
    x_prior, y_prior = wcs.world_to_pixel(prior)
    pixel_scale = _pixel_scale_arcsec(wcs)
    base = {
        "source": source,
        "filter": filter_name,
        "accepted": False,
        "rejection_reason": None,
        "ra_deg": None,
        "dec_deg": None,
        "x": _finite_float(x_prior),
        "y": _finite_float(y_prior),
        "snr": None,
        "offset_from_prior_arcsec": None,
        "uncertainty_arcsec": None,
        "width_fwhm_pixels": None,
        "width_ratio": None,
        "ellipticity": None,
        "host_dominated": False,
        "used_in_solution": False,
        "offset_from_final_arcsec": None,
    }
    if pixel_scale is None or not (np.isfinite(x_prior) and np.isfinite(y_prior)):
        base["rejection_reason"] = "WCS_OR_PRIOR_INVALID"
        return base
    if not (0 <= x_prior < data.shape[1] and 0 <= y_prior < data.shape[0]):
        base["rejection_reason"] = "TARGET_OUTSIDE_IMAGE"
        return base
    fwhm_pixels = max(1.0, float(fwhm_pixels))
    search = float(target.get("centroid_search_radius_arcsec", 3.0)) / pixel_scale
    outer = float(target.get("centroid_background_outer_fwhm", 6.0)) * fwhm_pixels
    half_size = int(np.ceil(max(search + 2.0 * fwhm_pixels, outer)))
    x0 = max(0, int(np.floor(x_prior)) - half_size)
    x1 = min(data.shape[1], int(np.floor(x_prior)) + half_size + 1)
    y0 = max(0, int(np.floor(y_prior)) - half_size)
    y1 = min(data.shape[0], int(np.floor(y_prior)) + half_size + 1)
    cutout = np.array(data[y0:y1, x0:x1], copy=True)
    cutout_invalid = invalid[y0:y1, x0:x1]
    yy, xx = np.indices(cutout.shape, dtype=float)
    local_prior_x = float(x_prior) - x0
    local_prior_y = float(y_prior) - y0
    radius = np.hypot(xx - local_prior_x, yy - local_prior_y)
    inner = float(target.get("centroid_background_inner_fwhm", 3.0)) * fwhm_pixels
    annulus = (radius >= inner) & (radius <= outer) & ~cutout_invalid
    if np.count_nonzero(annulus) < 20:
        base["rejection_reason"] = "LOCAL_BACKGROUND_UNAVAILABLE"
        return base
    _, background, rms = sigma_clipped_stats(
        cutout[annulus], sigma=3.0, maxiters=5
    )
    if not np.isfinite(rms) or rms <= 0:
        base["rejection_reason"] = "LOCAL_BACKGROUND_UNAVAILABLE"
        return base
    signal = cutout - float(background)
    smoothed = gaussian_filter(
        np.where(cutout_invalid, 0.0, signal),
        sigma=max(0.5, fwhm_pixels / 2.355),
    )
    search_region = (radius <= search) & ~cutout_invalid
    if not np.any(search_region):
        base["rejection_reason"] = "SEARCH_REGION_MASKED"
        return base
    peak_flat = np.argmax(np.where(search_region, smoothed, -np.inf))
    peak_y, peak_x = np.unravel_index(peak_flat, smoothed.shape)
    aperture_radius = float(
        target.get("centroid_aperture_radius_fwhm", 1.5)
    ) * fwhm_pixels
    aperture = (
        (xx - peak_x) ** 2 + (yy - peak_y) ** 2 <= aperture_radius ** 2
    ) & ~cutout_invalid
    weights = np.where(aperture, np.maximum(signal, 0.0), 0.0)
    total = float(np.sum(weights))
    if total <= 0:
        base["rejection_reason"] = "NO_POSITIVE_TARGET_SIGNAL"
        return base
    centroid_x = float(np.sum(weights * xx) / total)
    centroid_y = float(np.sum(weights * yy) / total)
    dx = xx - centroid_x
    dy = yy - centroid_y
    moment_xx = float(np.sum(weights * dx ** 2) / total)
    moment_yy = float(np.sum(weights * dy ** 2) / total)
    moment_xy = float(np.sum(weights * dx * dy) / total)
    eigenvalues = np.linalg.eigvalsh(
        np.array([[moment_xx, moment_xy], [moment_xy, moment_yy]])
    )
    eigenvalues = np.maximum(eigenvalues, 0.0)
    minor_sigma, major_sigma = np.sqrt(eigenvalues)
    measured_fwhm = 2.355 * np.sqrt(max(major_sigma * minor_sigma, 0.0))
    ellipticity = 1.0 - minor_sigma / major_sigma if major_sigma > 0 else 1.0
    global_x = centroid_x + x0
    global_y = centroid_y + y0
    coordinate = wcs.pixel_to_world(global_x, global_y).icrs
    offset = float(prior.separation(coordinate).arcsec)
    aperture_noise = float(rms) * np.sqrt(np.count_nonzero(aperture))
    snr = total / aperture_noise if aperture_noise > 0 else None
    width_ratio = measured_fwhm / fwhm_pixels
    host_dominated = bool(
        width_ratio > float(target.get("centroid_maximum_width_fwhm", 1.8))
        or ellipticity > float(target.get("centroid_maximum_ellipticity", 0.50))
    )
    minimum_uncertainty = float(
        target.get("minimum_coordinate_uncertainty_arcsec", 0.02)
    )
    centroid_uncertainty = (
        max(minimum_uncertainty, fwhm_pixels * pixel_scale / (2.355 * snr))
        if snr is not None and snr > 0 else None
    )
    uncertainty = (
        None
        if centroid_uncertainty is None
        else float(
            np.hypot(
                centroid_uncertainty,
                max(0.0, float(wcs_uncertainty_arcsec or 0.0)),
            )
        )
    )
    base.update(
        {
            "ra_deg": float(coordinate.ra.deg),
            "dec_deg": float(coordinate.dec.deg),
            "x": global_x,
            "y": global_y,
            "snr": snr,
            "offset_from_prior_arcsec": offset,
            "uncertainty_arcsec": uncertainty,
            "width_fwhm_pixels": measured_fwhm,
            "width_ratio": width_ratio,
            "ellipticity": ellipticity,
            "host_dominated": host_dominated,
        }
    )
    if snr is None or snr < float(target.get("centroid_minimum_snr", 3.0)):
        base["rejection_reason"] = "TARGET_SNR_LOW"
    elif offset > float(target.get("centroid_maximum_offset_arcsec", 2.0)):
        base["rejection_reason"] = "TARGET_CENTROID_OFFSET"
    elif host_dominated:
        base["rejection_reason"] = "TARGET_CENTROID_HOST_DOMINATED"
    else:
        base["accepted"] = True
    return base


def _candidate_table(candidates):
    """Return a masked table with stable target-candidate columns."""

    table = Table(rows=candidates, masked=True)
    numeric = (
        "ra_deg", "dec_deg", "x", "y", "snr", "offset_from_prior_arcsec",
        "uncertainty_arcsec", "width_fwhm_pixels", "width_ratio",
        "ellipticity", "offset_from_final_arcsec",
    )
    for name in numeric:
        if name not in table.colnames:
            continue
        values = np.asarray(
            [np.nan if _finite_float(row.get(name)) is None else float(row[name])
             for row in candidates],
            dtype=float,
        )
        table.replace_column(
            name,
            MaskedColumn(
                np.where(np.isfinite(values), values, 0.0),
                mask=~np.isfinite(values),
                name=name,
            ),
        )
    for name in ("ra_deg", "dec_deg"):
        if name in table.colnames:
            table[name].unit = u.deg
    for name in (
        "offset_from_prior_arcsec", "uncertainty_arcsec",
        "offset_from_final_arcsec",
    ):
        if name in table.colnames:
            table[name].unit = u.arcsec
    for name in ("x", "y", "width_fwhm_pixels"):
        if name in table.colnames:
            table[name].unit = u.pixel
    return table


def _weighted_coordinate(candidates, prior, settings):
    """Combine accepted coordinates in the tangent plane around the prior."""

    target = settings.get("target_position", {})
    accepted = [item for item in candidates if item.get("accepted")]
    if not accepted:
        return prior, float(target.get("prior_uncertainty_arcsec", 1.0)), []
    coordinates = SkyCoord(
        [item["ra_deg"] for item in accepted] * u.deg,
        [item["dec_deg"] for item in accepted] * u.deg,
        frame="icrs",
    )
    longitude, latitude = prior.spherical_offsets_to(coordinates)
    x = longitude.arcsec
    y = latitude.arcsec
    uncertainty = np.asarray(
        [item.get("uncertainty_arcsec") or target.get("prior_uncertainty_arcsec", 1.0)
         for item in accepted],
        dtype=float,
    )
    weights = 1.0 / np.maximum(uncertainty, 1.0e-6) ** 2
    for _ in range(3):
        mean_x = float(np.sum(weights * x) / np.sum(weights))
        mean_y = float(np.sum(weights * y) / np.sum(weights))
        radial = np.hypot(x - mean_x, y - mean_y)
        median = float(np.median(radial))
        scatter = float(1.4826 * np.median(np.abs(radial - median)))
        limit = max(
            float(target.get("maximum_filter_shift_arcsec", 0.50)),
            median + 4.0 * scatter,
        )
        keep = radial <= limit
        if np.all(keep) or np.count_nonzero(keep) == 0:
            break
        x, y, weights = x[keep], y[keep], weights[keep]
        accepted = [item for item, use in zip(accepted, keep) if use]
    mean_x = float(np.sum(weights * x) / np.sum(weights))
    mean_y = float(np.sum(weights * y) / np.sum(weights))
    coordinate = prior.spherical_offsets_by(mean_x * u.arcsec, mean_y * u.arcsec)
    residual = np.hypot(x - mean_x, y - mean_y)
    formal = float(np.sqrt(1.0 / np.sum(weights)))
    empirical = float(np.std(residual, ddof=1) / np.sqrt(len(residual))) if len(residual) > 1 else 0.0
    floor = float(target.get("minimum_coordinate_uncertainty_arcsec", 0.02))
    return coordinate.icrs, max(floor, formal, empirical), accepted


def determine_fixed_target_position(image_records, alignments, stacks,
                                    usability_decisions=None, settings=None,
                                    prior=None, difference_stacks=None, version=None):
    """Determine and freeze a versioned sky coordinate for forced photometry.

    Free centroids produced here are diagnostic candidates only. The returned
    frozen coordinate is the sole position intended for later target fitting.
    Difference-stack products may be supplied by later subtraction processing
    without changing this function's interface.
    """

    if settings is None:
        settings = get_default_settings()
    target = settings.get("target_position", {})
    prior_coordinate, prior_source = _coordinate_prior(
        image_records, settings, prior
    )
    prior_uncertainty = float(target.get("prior_uncertainty_arcsec", 1.0))
    candidates = [
        {
            "source": prior_source,
            "filter": None,
            "accepted": True,
            "rejection_reason": None,
            "ra_deg": float(prior_coordinate.ra.deg),
            "dec_deg": float(prior_coordinate.dec.deg),
            "x": None,
            "y": None,
            "snr": None,
            "offset_from_prior_arcsec": 0.0,
            "uncertainty_arcsec": prior_uncertainty,
            "width_fwhm_pixels": None,
            "width_ratio": None,
            "ellipticity": None,
            "host_dominated": False,
            "used_in_solution": False,
            "offset_from_final_arcsec": None,
        }
    ]
    alignment_lookup = {item["image_id"]: item for item in alignments}
    decisions = _decision_map(usability_decisions)
    eligible_records = []
    for index, record in enumerate(image_records):
        image_id = _image_id(record, index)
        alignment = alignment_lookup.get(image_id)
        if alignment is None or not alignment.get("coordinate_eligible", False):
            continue
        quality = record.get("quality") or {}
        seeing = _finite_float(quality.get("fwhm_arcsec"))
        decision = decisions.get(image_id)
        depth = _finite_float(
            _decision_value(decision, "local_depth_5sigma_mag")
        )
        if depth is None and isinstance(decision, dict):
            depth = _finite_float(
                (decision.get("local_depths_mag") or {}).get("5sigma")
            )
        alignment_rms = _finite_float(alignment.get("refined_rms_arcsec"))
        eligible_records.append(
            (
                np.inf if alignment_rms is None else alignment_rms,
                np.inf if seeing is None else seeing,
                np.inf if depth is None else -depth,
                image_id,
                record,
                alignment,
            )
        )
    eligible_records.sort(key=lambda value: value[:3])
    maximum = int(target.get("maximum_individual_centroids", 5))
    for _, _, _, image_id, record, alignment in eligible_records[:maximum]:
        ccd = record.get("ccd")
        if ccd is None:
            continue
        quality = record.get("quality") or {}
        fwhm = _finite_float(quality.get("fwhm_pixels")) or 4.0
        filter_name = (record.get("metadata") or {}).get("filter")
        candidates.append(
            _centroid_candidate(
                ccd.data,
                alignment["wcs"],
                prior_coordinate,
                fwhm,
                settings,
                "image:{}".format(image_id),
                filter_name=filter_name,
                mask=getattr(ccd, "mask", None),
                wcs_uncertainty_arcsec=float(
                    np.hypot(
                        _finite_float(alignment.get("refined_rms_arcsec")) or 0.0,
                        _finite_float(
                            (record.get("astrometry") or {}).get(
                                "refined_rms_arcsec"
                            )
                        ) or 0.0,
                    )
                ),
            )
        )

    for name, product in stacks.items():
        if name == "multifilter":
            filter_name = "multifilter"
        else:
            filter_name = name
        contributor_fwhm = []
        contributor_ids = {
            item.get("image_id") for item in product.get("contributors", [])
            if isinstance(item, dict)
        }
        for _, _, _, image_id, record, _ in eligible_records:
            if image_id in contributor_ids:
                value = _finite_float((record.get("quality") or {}).get("fwhm_pixels"))
                if value is not None:
                    contributor_fwhm.append(value)
        fwhm = float(np.median(contributor_fwhm)) if contributor_fwhm else 4.0
        candidates.append(
            _centroid_candidate(
                product["data"],
                product["wcs"],
                prior_coordinate,
                fwhm,
                settings,
                "stack:{}".format(name),
                filter_name=filter_name,
                mask=product.get("coverage", 1) <= 0,
                wcs_uncertainty_arcsec=product.get(
                    "alignment_uncertainty_arcsec", 0.0
                ),
            )
        )

    if difference_stacks and target.get("allow_difference_stack_candidates", True):
        for name, product in difference_stacks.items():
            candidates.append(
                _centroid_candidate(
                    product["data"],
                    product["wcs"],
                    prior_coordinate,
                    product.get("fwhm_pixels", 4.0),
                    settings,
                    "difference_stack:{}".format(name),
                    filter_name=product.get("filter", name),
                    mask=product.get("mask"),
                    wcs_uncertainty_arcsec=product.get(
                        "alignment_uncertainty_arcsec", 0.0
                    ),
                )
            )

    flags = []
    stack_candidates = [
        item for item in candidates
        if item.get("accepted") and item["source"].startswith("stack:")
        and item.get("filter") != "multifilter"
    ]
    if len(stack_candidates) >= 2:
        coordinates = SkyCoord(
            [item["ra_deg"] for item in stack_candidates] * u.deg,
            [item["dec_deg"] for item in stack_candidates] * u.deg,
        )
        maximum_shift = 0.0
        for index in range(len(coordinates)):
            maximum_shift = max(
                maximum_shift,
                float(np.max(coordinates[index].separation(coordinates).arcsec)),
            )
        if maximum_shift > float(target.get("maximum_filter_shift_arcsec", 0.50)):
            flags.append("TARGET_FILTER_SHIFT")
            for item in stack_candidates:
                item["accepted"] = False
                item["rejection_reason"] = "TARGET_FILTER_SHIFT"

    if any(item.get("host_dominated") for item in candidates):
        flags.append("TARGET_CENTROID_HOST_DOMINATED")
    if any(item.get("rejection_reason") == "TARGET_CENTROID_OFFSET" for item in candidates):
        flags.append("TARGET_CENTROID_OFFSET")

    fixed_user = (
        prior_source == "user" and target.get("user_position_mode", "prior") == "fixed"
    )
    if fixed_user:
        final_coordinate = prior_coordinate
        uncertainty = prior_uncertainty
        used = [candidates[0]]
    else:
        final_coordinate, uncertainty, used = _weighted_coordinate(
            candidates, prior_coordinate, settings
        )
    used_ids = {id(item) for item in used}
    for item in candidates:
        item["used_in_solution"] = id(item) in used_ids
        if item.get("ra_deg") is not None:
            coordinate = SkyCoord(item["ra_deg"], item["dec_deg"], unit="deg")
            item["offset_from_final_arcsec"] = float(
                final_coordinate.separation(coordinate).arcsec
            )
    if uncertainty > float(target.get("maximum_coordinate_uncertainty_arcsec", 1.0)):
        flags.append("TARGET_POSITION_UNCERTAIN")
    accepted_detections = sum(
        item.get("used_in_solution") and item["source"] != prior_source
        for item in candidates
    )
    if accepted_detections == 0 and not fixed_user:
        flags.append("TARGET_POSITION_UNCERTAIN")
    status = "WARN" if flags else "PASS"
    if version is None:
        version = int(target.get("target_coordinate_version", 1))
    solution = {
        "version": "target-v{}".format(int(version)),
        "version_number": int(version),
        "ra_deg": float(final_coordinate.ra.deg),
        "dec_deg": float(final_coordinate.dec.deg),
        "uncertainty_arcsec": float(uncertainty),
        "frozen": True,
        "forced_photometry": bool(target.get("fixed_position_photometry", True)),
        "free_centroid_diagnostic_only": True,
        "prior_ra_deg": float(prior_coordinate.ra.deg),
        "prior_dec_deg": float(prior_coordinate.dec.deg),
        "prior_source": prior_source,
        "candidate_count": len(candidates),
        "used_candidate_count": len(used),
        "detection_candidate_count": accepted_detections,
        "status": status,
        "flags": list(dict.fromkeys(flags)),
        "provenance": [item["source"] for item in used],
    }
    return solution, _candidate_table(candidates)


def validate_fixed_target_projection(image_records, alignments, target_solution,
                                     settings=None):
    """Confirm the frozen coordinate projects consistently in every image."""

    if settings is None:
        settings = get_default_settings()
    target = settings.get("target_position", {})
    warning_rms = float(target.get("relative_alignment_warn_rms_arcsec", 0.50))
    failure_rms = float(target.get("relative_alignment_fail_rms_arcsec", 1.50))
    coordinate = SkyCoord(
        target_solution["ra_deg"], target_solution["dec_deg"], unit="deg"
    )
    alignment_lookup = {item["image_id"]: item for item in alignments}
    rows = []
    for index, record in enumerate(image_records):
        image_id = _image_id(record, index)
        alignment = alignment_lookup.get(image_id)
        wcs = None if alignment is None else alignment.get("wcs")
        ccd = record.get("ccd")
        row = {
            "image_id": image_id,
            "x": None,
            "y": None,
            "inside_image": False,
            "roundtrip_error_arcsec": None,
            "relative_alignment_rms_arcsec": None if alignment is None else alignment.get("refined_rms_arcsec"),
            "status": "FAIL",
            "flags": "",
        }
        flags = []
        if wcs is None or ccd is None:
            flags.append("RELATIVE_ALIGNMENT_FAILED")
        else:
            x, y = wcs.world_to_pixel(coordinate)
            roundtrip = wcs.pixel_to_world(x, y).icrs
            row["x"] = float(x)
            row["y"] = float(y)
            row["inside_image"] = bool(
                0 <= x < ccd.shape[1] and 0 <= y < ccd.shape[0]
            )
            row["roundtrip_error_arcsec"] = float(
                coordinate.separation(roundtrip).arcsec
            )
            rms = _finite_float(row["relative_alignment_rms_arcsec"])
            if not row["inside_image"]:
                flags.append("TARGET_OUTSIDE_IMAGE")
                row["status"] = "FAIL"
            elif rms is None or rms > failure_rms:
                flags.append("RELATIVE_ALIGNMENT_FAILED")
                row["status"] = "FAIL"
            elif rms > warning_rms:
                flags.append("RELATIVE_ALIGNMENT_POOR")
                row["status"] = "WARN"
            else:
                row["status"] = "PASS"
        row["flags"] = ";".join(flags)
        rows.append(row)
    table = Table(rows=rows, masked=True)
    for name, unit in (
        ("x", u.pixel), ("y", u.pixel),
        ("roundtrip_error_arcsec", u.arcsec),
        ("relative_alignment_rms_arcsec", u.arcsec),
    ):
        if name in table.colnames:
            values = np.asarray(
                [
                    np.nan if _finite_float(row.get(name)) is None else row[name]
                    for row in rows
                ],
                dtype=float,
            )
            table.replace_column(
                name,
                MaskedColumn(
                    np.where(np.isfinite(values), values, 0.0),
                    mask=~np.isfinite(values),
                    name=name,
                    unit=unit,
                ),
            )
    return table


def save_alignment_and_target_products(alignments, residuals, stacks,
                                       target_solution, target_candidates,
                                       projection_table, output_directory,
                                       object_name="field", settings=None,
                                       overwrite=False):
    """Save derived WCS, stack, target-coordinate, and validation products."""

    if settings is None:
        settings = get_default_settings()
    target = settings.get("target_position", {})
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    stem = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in str(object_name)
    ).strip("_") or "field"
    paths = {}
    if target.get("save_alignment_table", True):
        table_path = output_directory / "{}_relative_alignment.ecsv".format(stem)
        alignment_table(alignments).write(
            table_path, format="ascii.ecsv", overwrite=bool(overwrite)
        )
        residual_path = output_directory / "{}_relative_residuals.ecsv".format(stem)
        residuals.write(residual_path, format="ascii.ecsv", overwrite=bool(overwrite))
        paths["alignment_table"] = str(table_path)
        paths["alignment_residuals"] = str(residual_path)
    if target.get("save_alignment_headers", True):
        header_paths = {}
        for alignment in alignments:
            if alignment.get("wcs") is None:
                continue
            name = "".join(
                character if character.isalnum() or character in "-_" else "_"
                for character in alignment["image_id"]
            )
            path = output_directory / "{}_{}_relative_wcs.fits".format(stem, name)
            fits.PrimaryHDU(
                header=alignment["wcs"].to_header(relax=True)
            ).writeto(path, overwrite=bool(overwrite))
            header_paths[alignment["image_id"]] = str(path)
        paths["alignment_headers"] = header_paths
    if target.get("save_stack", True):
        stack_paths = {}
        for name, product in stacks.items():
            path = output_directory / "{}_{}_detection_stack.fits".format(stem, name)
            header = product["wcs"].to_header(relax=True)
            header["NSTACK"] = len(product.get("contributors", []))
            header["STKTYPE"] = product.get("kind", "detection")
            header["TARGRA"] = (target_solution["ra_deg"], "Frozen target RA [deg]")
            header["TARGDEC"] = (target_solution["dec_deg"], "Frozen target Dec [deg]")
            header["TARGVER"] = (target_solution["version"], "Target-coordinate version")
            fits.HDUList(
                [
                    fits.PrimaryHDU(np.asarray(product["data"], dtype=np.float32), header),
                    fits.ImageHDU(np.asarray(product["weight"], dtype=np.float32), name="WEIGHT"),
                    fits.ImageHDU(np.asarray(product["coverage"], dtype=np.int16), name="COVERAGE"),
                ]
            ).writeto(path, overwrite=bool(overwrite))
            stack_paths[name] = str(path)
        paths["stacks"] = stack_paths
    if target.get("save_target_position", True):
        solution_path = output_directory / "{}_fixed_target.ecsv".format(stem)
        solution_row = dict(target_solution)
        solution_row["flags"] = ";".join(target_solution.get("flags", []))
        solution_row["provenance"] = ";".join(target_solution.get("provenance", []))
        solution_table = Table(rows=[solution_row], masked=True)
        solution_table["ra_deg"].unit = u.deg
        solution_table["dec_deg"].unit = u.deg
        solution_table["prior_ra_deg"].unit = u.deg
        solution_table["prior_dec_deg"].unit = u.deg
        solution_table["uncertainty_arcsec"].unit = u.arcsec
        solution_table.write(
            solution_path, format="ascii.ecsv", overwrite=bool(overwrite)
        )
        projection_path = output_directory / "{}_fixed_target_projection.ecsv".format(stem)
        projection_table.write(
            projection_path, format="ascii.ecsv", overwrite=bool(overwrite)
        )
        paths["target_position"] = str(solution_path)
        paths["target_projection"] = str(projection_path)
    if target.get("save_target_candidates", True):
        candidate_path = output_directory / "{}_target_candidates.ecsv".format(stem)
        target_candidates.write(
            candidate_path, format="ascii.ecsv", overwrite=bool(overwrite)
        )
        paths["target_candidates"] = str(candidate_path)
    return paths


__all__ = [
    "alignment_table",
    "build_detection_stacks",
    "determine_fixed_target_position",
    "refine_relative_alignment",
    "save_alignment_and_target_products",
    "validate_fixed_target_projection",
]
