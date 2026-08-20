"""Template acquisition, alignment, and image subtraction for redphot.

Science pixels are never resampled or overwritten.  Templates are mosaicked as
needed and sampled onto each science-image grid.  Every subtraction backend is
run with checked output, and the resulting difference is accepted only after
the configured residual-quality checks pass.
"""

from collections.abc import Mapping
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, SkyOffsetFrame
from astropy.io import fits
from astropy.table import MaskedColumn, Table
from astropy.time import Time
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from scipy import ndimage

from .config import get_default_settings, merge_settings, normalize_filter_name


def _finite_float(value, default=None):
    """Return a finite float, or ``default`` when conversion fails."""

    if value is None or np.ma.is_masked(value):
        return default
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def _image_id(record, index=0):
    """Return the persistent identifier for an image record."""

    value = record.get("image_id")
    metadata = record.get("metadata") or {}
    return str(value or metadata.get("filename") or "image_{:04d}".format(index))


def _record_ccd(record):
    """Return the preferred CCDData-like value from an image record."""

    for name in ("prepared_ccd", "working_ccd", "ccd"):
        value = record.get(name)
        if value is not None:
            return value
    return None


def _record_data(record):
    """Return a floating two-dimensional image without modifying its source."""

    ccd = _record_ccd(record)
    value = record.get("data") if ccd is None else getattr(ccd, "data", ccd)
    if value is None or np.ndim(value) != 2:
        raise ValueError("Image records require a two-dimensional CCD or data array")
    return np.asarray(value, dtype=float)


def _record_wcs(record):
    """Return the best celestial WCS stored in a record."""

    for value in (
        record.get("wcs"),
        (record.get("alignment") or {}).get("wcs"),
        (record.get("astrometry") or {}).get("refined_wcs"),
        getattr(_record_ccd(record), "wcs", None),
    ):
        if value is not None and getattr(value, "has_celestial", False):
            return deepcopy(value.celestial)
    return None


def _record_mask(record, shape):
    """Combine finite-pixel, CCD, and named artifact masks."""

    data = _record_data(record)
    mask = ~np.isfinite(data)
    ccd = _record_ccd(record)
    if ccd is not None and getattr(ccd, "mask", None) is not None:
        candidate = np.asarray(ccd.mask, dtype=bool)
        if candidate.shape == shape:
            mask |= candidate
    for value in (record.get("masks") or {}).values():
        if isinstance(value, np.ndarray) and value.shape == shape:
            mask |= np.asarray(value, dtype=bool)
    value = record.get("mask")
    if value is not None and np.shape(value) == shape:
        mask |= np.asarray(value, dtype=bool)
    return mask


def _robust_location_scale(values):
    """Return a robust median and Gaussian-equivalent MAD."""

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return None, None
    center = float(np.median(values))
    scale = float(1.4826 * np.median(np.abs(values - center)))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.std(values))
    return center, scale if np.isfinite(scale) else None


def _pixel_scale_arcsec(wcs):
    """Return the representative celestial pixel scale in arcseconds."""

    if wcs is None:
        return None
    scales = np.abs(proj_plane_pixel_scales(wcs.celestial)) * 3600.0
    scales = scales[np.isfinite(scales) & (scales > 0)]
    return float(np.median(scales)) if scales.size else None


def _metadata_mjd(metadata):
    """Return an MJD from normalized numeric or ISO metadata."""

    value = _finite_float(metadata.get("mjd"))
    if value is not None:
        return value
    date_obs = metadata.get("date_obs")
    if date_obs is None:
        return None
    try:
        return float(Time(date_obs, scale="utc").mjd)
    except (TypeError, ValueError):
        return None


def science_footprint_union(image_records, settings=None):
    """Measure the union of science WCS footprints plus the safety margin.

    Returns a tangent-plane bounding region suitable for a survey request.  RA
    wraparound and high-declination fields are handled through a sky-offset
    frame rather than by taking minima and maxima directly in RA.
    """

    if settings is None:
        settings = get_default_settings()
    corners = []
    scales = []
    contributors = []
    for index, record in enumerate(image_records):
        wcs = _record_wcs(record)
        if wcs is None:
            continue
        shape = _record_data(record).shape
        x = np.array([-0.5, shape[1] - 0.5, shape[1] - 0.5, -0.5])
        y = np.array([-0.5, -0.5, shape[0] - 0.5, shape[0] - 0.5])
        try:
            sky = wcs.pixel_to_world(x, y).icrs
        except Exception:
            continue
        if not np.all(np.isfinite(sky.ra.deg)) or not np.all(np.isfinite(sky.dec.deg)):
            continue
        corners.append(sky)
        contributors.append(_image_id(record, index))
        scale = _pixel_scale_arcsec(wcs)
        if scale is not None:
            scales.append(scale)
    if not corners:
        raise ValueError("No image has a valid celestial WCS footprint")
    all_corners = SkyCoord(
        np.concatenate([value.ra.deg for value in corners]) * u.deg,
        np.concatenate([value.dec.deg for value in corners]) * u.deg,
        frame="icrs",
    )
    xyz = np.mean(all_corners.cartesian.xyz.value, axis=1)
    cartesian_center = SkyCoord(
        x=xyz[0], y=xyz[1], z=xyz[2], representation_type="cartesian", frame="icrs"
    )
    center = SkyCoord(
        ra=cartesian_center.spherical.lon,
        dec=cartesian_center.spherical.lat,
        frame="icrs",
    )
    offset = all_corners.transform_to(SkyOffsetFrame(origin=center))
    margin = float(settings.get("subtraction", {}).get("template_margin_arcmin", 2.0))
    lon = offset.lon.to_value(u.arcmin)
    lat = offset.lat.to_value(u.arcmin)
    width = float(np.ptp(lon) + 2.0 * margin)
    height = float(np.ptp(lat) + 2.0 * margin)
    return {
        "center": center,
        "corners": all_corners,
        "width_arcmin": width,
        "height_arcmin": height,
        "margin_arcmin": margin,
        "pixel_scale_arcsec": float(np.median(scales)) if scales else None,
        "contributors": contributors,
    }


def read_template(path, metadata=None):
    """Read the first numeric 2D FITS HDU into a normalized template record."""

    path = Path(path).expanduser()
    with fits.open(path, memmap=False) as hdulist:
        selected = None
        for index, hdu in enumerate(hdulist):
            data = getattr(hdu, "data", None)
            if data is not None and np.ndim(data) == 2 and np.issubdtype(
                np.asarray(data).dtype, np.number
            ):
                selected = index
                break
        if selected is None:
            raise ValueError("Template contains no numeric two-dimensional image")
        data = np.asarray(hdulist[selected].data, dtype=float).copy()
        header = hdulist[0].header.copy()
        if selected != 0:
            header.extend(hdulist[selected].header, update=True, unique=True)
    wcs = WCS(header).celestial
    if not wcs.has_celestial:
        wcs = None
    values = dict(metadata or {})
    keyword_map = {
        "filter": ("FILTER", "FILTER1", "BAND"),
        "mjd": ("MJD-OBS", "MJD", "OBSMJD"),
        "date_obs": ("DATE-OBS",),
        "exposure_time": ("EXPTIME",),
        "saturation": ("SATURATE", "SATLEVEL", "MAXLIN"),
        "fwhm_arcsec": ("FWHM", "L1FWHM", "SEEING"),
        "depth_mag": ("DEPTH", "MAGLIM", "LIMMAG"),
    }
    for name, keywords in keyword_map.items():
        if values.get(name) is None:
            values[name] = next((header[key] for key in keywords if key in header), None)
    values["filter"] = normalize_filter_name(values.get("filter"))
    return {
        "data": data,
        "mask": ~np.isfinite(data),
        "wcs": wcs,
        "header": header,
        "metadata": values,
        "path": str(path.resolve()),
        "source": values.get("source", "user"),
        "hdu": selected,
    }


def _template_paths_for_filter(template_path, filter_name):
    """Resolve a string, sequence, or filter-keyed template path setting."""

    if template_path is None:
        return []
    if isinstance(template_path, Mapping):
        value = template_path.get(filter_name, template_path.get("default"))
    else:
        value = template_path
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        return [value]
    return list(value)


def _default_survey_downloader(footprint, filter_name, survey, settings):
    """Download survey cutouts through Astroquery SkyView.

    Survey names are configurable because SkyView holdings can change.  A
    caller may supply a dedicated downloader function for archive-specific
    authentication or tile services.
    """

    from astroquery.skyview import SkyView

    subtraction = settings.get("subtraction", {})
    SkyView.TIMEOUT = float(subtraction.get("download_timeout_s", 120))
    survey_name = subtraction.get("survey_names", {}).get(survey, survey)
    survey_filter = subtraction.get("survey_filter_map", {}).get(
        survey, {}
    ).get(filter_name)
    if survey_filter:
        survey_name = survey_name.format(filter=survey_filter)
    scale = subtraction.get("download_pixel_scale_arcsec")
    scale = _finite_float(scale, footprint.get("pixel_scale_arcsec")) or 1.0
    pixels = max(
        32,
        int(np.ceil(max(footprint["width_arcmin"], footprint["height_arcmin"]) * 60.0 / scale)),
    )
    images = SkyView.get_images(
        position=footprint["center"],
        survey=[survey_name],
        width=footprint["width_arcmin"] * u.arcmin,
        height=footprint["height_arcmin"] * u.arcmin,
        pixels=[pixels, pixels],
    )
    records = []
    for hdulist in images:
        hdu = hdulist[0]
        data = np.asarray(hdu.data, dtype=float)
        wcs = WCS(hdu.header).celestial
        records.append(
            {
                "data": data,
                "mask": ~np.isfinite(data),
                "wcs": wcs,
                "header": hdu.header.copy(),
                "metadata": {"filter": filter_name, "survey": survey},
                "source": survey,
                "path": None,
            }
        )
    if not records:
        raise RuntimeError("Survey returned no template images")
    return records


def _resample_array(data, input_wcs, output_wcs, output_shape, mask=None,
                    order=3, tile_rows=256):
    """Resample one derived array onto an output WCS in memory-limited tiles."""

    data = np.asarray(data, dtype=float)
    invalid = ~np.isfinite(data)
    if mask is not None:
        invalid |= np.asarray(mask, dtype=bool)
    output = np.full(output_shape, np.nan, dtype=float)
    footprint = np.zeros(output_shape, dtype=bool)
    for y0 in range(0, output_shape[0], int(tile_rows)):
        y1 = min(output_shape[0], y0 + int(tile_rows))
        yy, xx = np.meshgrid(
            np.arange(y0, y1, dtype=float),
            np.arange(output_shape[1], dtype=float),
            indexing="ij",
        )
        sky = output_wcs.pixel_to_world(xx, yy)
        input_x, input_y = input_wcs.world_to_pixel(sky)
        inside = (
            np.isfinite(input_x) & np.isfinite(input_y)
            & (input_x >= -0.5) & (input_y >= -0.5)
            & (input_x <= data.shape[1] - 0.5)
            & (input_y <= data.shape[0] - 0.5)
        )
        sample_x = np.clip(input_x, 0, data.shape[1] - 1)
        sample_y = np.clip(input_y, 0, data.shape[0] - 1)
        sampled = ndimage.map_coordinates(
            np.where(invalid, 0.0, data), [sample_y, sample_x], order=int(order),
            mode="constant", cval=0.0, prefilter=int(order) > 1,
        )
        sampled_bad = ndimage.map_coordinates(
            invalid.astype(float), [sample_y, sample_x], order=0,
            mode="constant", cval=1.0, prefilter=False,
        ) > 0
        good = inside & ~sampled_bad
        tile = output[y0:y1]
        tile[good] = sampled[good]
        footprint[y0:y1] = good
    return output, footprint


def mosaic_template_tiles(templates, footprint, settings=None):
    """Mosaic all template tiles onto one tangent-plane union grid."""

    if settings is None:
        settings = get_default_settings()
    if not templates:
        raise ValueError("At least one template tile is required")
    subtraction = settings.get("subtraction", {})
    scales = [_pixel_scale_arcsec(item.get("wcs")) for item in templates]
    scales = [value for value in scales if value is not None]
    scale = _finite_float(subtraction.get("download_pixel_scale_arcsec"))
    scale = scale or (float(np.median(scales)) if scales else footprint.get("pixel_scale_arcsec"))
    if scale is None or scale <= 0:
        raise ValueError("Template mosaic requires a valid pixel scale")
    width = max(2, int(np.ceil(footprint["width_arcmin"] * 60.0 / scale)))
    height = max(2, int(np.ceil(footprint["height_arcmin"] * 60.0 / scale)))
    maximum = int(subtraction.get("maximum_mosaic_pixels", 100_000_000))
    if width * height > maximum:
        raise ValueError("Requested template mosaic exceeds maximum_mosaic_pixels")
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [footprint["center"].ra.deg, footprint["center"].dec.deg]
    wcs.wcs.crpix = [(width + 1.0) / 2.0, (height + 1.0) / 2.0]
    wcs.wcs.cdelt = [-scale / 3600.0, scale / 3600.0]
    total = np.zeros((height, width), dtype=float)
    weight = np.zeros((height, width), dtype=np.int16)
    for template in templates:
        if template.get("wcs") is None:
            continue
        plane, valid = _resample_array(
            template["data"], template["wcs"], wcs, (height, width),
            template.get("mask"), subtraction.get("resampling_order", 3),
            subtraction.get("resampling_tile_rows", 256),
        )
        total[valid] += plane[valid]
        weight[valid] += 1
    mosaic = np.full((height, width), np.nan, dtype=float)
    np.divide(total, weight, out=mosaic, where=weight > 0)
    metadata = dict(templates[0].get("metadata") or {})
    metadata["tile_count"] = len(templates)
    return {
        "data": mosaic,
        "mask": weight == 0,
        "coverage": weight,
        "wcs": wcs,
        "header": wcs.to_header(relax=True),
        "metadata": metadata,
        "source": templates[0].get("source", "user"),
        "tiles": templates,
        "pixel_scale_arcsec": scale,
    }


def acquire_template(image_records, filter_name=None, settings=None,
                     template_paths=None, downloader=None):
    """Acquire and mosaic a user or supported-survey template.

    The optional ``downloader`` receives ``footprint``, ``filter_name``,
    ``survey``, and ``settings`` and must return one template record or a list
    of records.  This keeps archive-specific code replaceable while providing
    SkyView as the built-in network route.
    """

    if settings is None:
        settings = get_default_settings()
    subtraction = settings.get("subtraction", {})
    footprint = science_footprint_union(image_records, settings)
    if filter_name is None:
        names = {
            normalize_filter_name((item.get("metadata") or {}).get("filter"))
            for item in image_records
        }
        names.discard(None)
        if len(names) != 1:
            raise ValueError("Specify filter_name when science images span multiple filters")
        filter_name = names.pop()
    filter_name = normalize_filter_name(filter_name)
    configured_path = template_paths
    if configured_path is None:
        configured_path = subtraction.get("template_path")
    paths = _template_paths_for_filter(configured_path, filter_name)
    tiles = []
    acquisition = {"filter": filter_name, "attempts": [], "from_cache": False}
    for path in paths:
        tiles.append(read_template(path, {"filter": filter_name, "source": "user"}))
    if not tiles:
        source = str(subtraction.get("template_source", "auto")).lower()
        if source == "user":
            raise FileNotFoundError(
                "subtraction.template_source is 'user' but no template path was supplied"
            )
        cache = Path(subtraction.get("cache_directory", "redphot_cache/templates")).expanduser()
        cache_path = cache / "{}_template.fits".format(filter_name)
        if subtraction.get("use_cached_templates", True) and cache_path.exists():
            tiles = [read_template(cache_path, {"filter": filter_name, "source": "cache"})]
            acquisition["from_cache"] = True
        else:
            fetch = downloader or _default_survey_downloader
            errors = []
            surveys = subtraction.get("template_survey_priority", [])
            if source not in {"auto", "survey"}:
                surveys = [source]
            for survey in surveys:
                try:
                    result = fetch(footprint, filter_name, survey, settings)
                    tiles = result if isinstance(result, list) else [result]
                    acquisition["attempts"].append({"survey": survey, "status": "PASS"})
                    break
                except Exception as error:
                    errors.append("{}: {}".format(survey, error))
                    acquisition["attempts"].append(
                        {"survey": survey, "status": "FAIL", "error": str(error)}
                    )
            if not tiles:
                raise RuntimeError("Template acquisition failed: {}".format("; ".join(errors)))
    mosaic = mosaic_template_tiles(tiles, footprint, settings)
    mosaic["acquisition"] = acquisition
    mosaic["requested_footprint"] = footprint
    if not paths and subtraction.get("save_downloaded_templates", True):
        cache = Path(subtraction.get("cache_directory", "redphot_cache/templates")).expanduser()
        cache.mkdir(parents=True, exist_ok=True)
        cache_path = cache / "{}_template.fits".format(filter_name)
        fits.PrimaryHDU(mosaic["data"], mosaic["wcs"].to_header()).writeto(
            cache_path, overwrite=True
        )
        mosaic["cached_path"] = str(cache_path)
    return mosaic


def validate_template(template, image_records, settings=None, filter_name=None):
    """Validate template coverage, band, depth, seeing, saturation, epoch, and WCS."""

    if settings is None:
        settings = get_default_settings()
    subtraction = settings.get("subtraction", {})
    metadata = template.get("metadata") or {}
    data = np.asarray(template.get("data"), dtype=float)
    mask = np.asarray(template.get("mask", ~np.isfinite(data)), dtype=bool)
    flags = []
    checks = []

    def add(name, value, passed, flag):
        checks.append({"name": name, "value": value, "passed": bool(passed), "flag": None if passed else flag})
        if not passed and flag not in flags:
            flags.append(flag)

    wcs = template.get("wcs")
    add("wcs", bool(wcs is not None and getattr(wcs, "has_celestial", False)),
        wcs is not None and getattr(wcs, "has_celestial", False), "TEMPLATE_WCS_INVALID")
    coverage = float(np.count_nonzero(~mask & np.isfinite(data)) / data.size) if data.size else 0.0
    add("coverage_fraction", coverage,
        coverage >= float(subtraction.get("minimum_coverage_fraction", 0.99)),
        "TEMPLATE_COVERAGE_INCOMPLETE")
    if filter_name is None:
        science_filters = {
            normalize_filter_name((record.get("metadata") or {}).get("filter"))
            for record in image_records
        }
        science_filters.discard(None)
        filter_name = next(iter(science_filters)) if len(science_filters) == 1 else None
    template_filter = normalize_filter_name(metadata.get("filter"))
    filter_ok = filter_name is None or template_filter == filter_name
    if not filter_ok and subtraction.get("allow_approximate_filter_match", False):
        allowed = subtraction.get("approximate_filter_matches", {}).get(filter_name, [])
        filter_ok = template_filter in [normalize_filter_name(value) for value in allowed]
    if subtraction.get("require_filter_match", True):
        add("filter_match", "{} -> {}".format(template_filter, filter_name), filter_ok,
            "TEMPLATE_FILTER_MISMATCH")
    saturation = _finite_float(metadata.get("saturation"))
    if saturation is not None:
        saturated_fraction = float(np.count_nonzero(data >= saturation) / data.size)
        add("saturated_fraction", saturated_fraction,
            saturated_fraction <= float(subtraction.get("maximum_template_saturated_fraction", 0.01)),
            "TEMPLATE_SATURATION_HIGH")
    template_fwhm = _finite_float(
        metadata.get("fwhm_arcsec"),
        _finite_float((template.get("quality") or {}).get("fwhm_arcsec")),
    )
    science_fwhm = [
        _finite_float((record.get("quality") or {}).get("fwhm_arcsec"))
        for record in image_records
    ]
    science_fwhm = [value for value in science_fwhm if value is not None]
    if template_fwhm is not None and science_fwhm:
        ratio = template_fwhm / max(float(np.median(science_fwhm)), 1.0e-6)
        add("fwhm_ratio", ratio,
            ratio <= float(subtraction.get("maximum_template_fwhm_ratio", 2.0)),
            "TEMPLATE_SEEING_POOR")
    template_depth = _finite_float(metadata.get("depth_mag"))
    science_depths = []
    for record in image_records:
        usability = record.get("usability") or {}
        depths = usability.get("global_depths_mag") or {}
        value = _finite_float(
            usability.get("global_depth_5sigma_mag"),
            _finite_float(depths.get("5sigma")),
        )
        if value is not None:
            science_depths.append(value)
    if template_depth is not None and science_depths:
        margin = template_depth - float(np.median(science_depths))
        add("depth_margin_mag", margin,
            margin >= float(subtraction.get("minimum_template_depth_margin_mag", 0.5)),
            "TEMPLATE_DEPTH_INSUFFICIENT")
    transient_mjd = _finite_float(subtraction.get("transient_epoch_mjd"))
    template_mjd = _metadata_mjd(metadata)
    if subtraction.get("require_pretransient_template", True) and transient_mjd is not None:
        add("pretransient", template_mjd, template_mjd is not None and template_mjd < transient_mjd,
            "TEMPLATE_POST_TRANSIENT")
    return {
        "status": "PASS" if not flags else "FAIL",
        "flags": flags,
        "checks": checks,
        "filter": template_filter,
        "coverage_fraction": coverage,
    }


def align_template_to_science(science_record, template, settings=None):
    """Resample only the template onto the unchanged science-image grid."""

    if settings is None:
        settings = get_default_settings()
    subtraction = settings.get("subtraction", {})
    if not subtraction.get("keep_science_grid", True) or not subtraction.get(
        "resample_template_only", True
    ):
        raise ValueError("RedPhot subtraction requires a fixed science grid")
    science = _record_data(science_record)
    science_wcs = _record_wcs(science_record)
    template_wcs = template.get("wcs")
    if science_wcs is None or template_wcs is None:
        raise ValueError("Science and template both require celestial WCS")
    aligned, footprint = _resample_array(
        template["data"], template_wcs, science_wcs, science.shape,
        template.get("mask"), subtraction.get("resampling_order", 3),
        subtraction.get("resampling_tile_rows", 256),
    )
    return {
        "data": aligned,
        "mask": ~footprint,
        "coverage": footprint,
        "coverage_fraction": float(np.count_nonzero(footprint) / footprint.size),
        "wcs": science_wcs,
        "science_shape": science.shape,
        "science_pixels_resampled": False,
    }


def _seeing_pixels(record, fallback=4.0):
    """Return FWHM in pixels from image-quality metadata."""

    quality = record.get("quality") or {}
    pixels = _finite_float(quality.get("fwhm_pixels"))
    if pixels is not None:
        return pixels
    arcsec = _finite_float(quality.get("fwhm_arcsec"))
    scale = _pixel_scale_arcsec(_record_wcs(record))
    return arcsec / scale if arcsec is not None and scale else float(fallback)


def choose_hotpants_parameters(science_record, template, aligned_template,
                               settings=None):
    """Choose conservative Hotpants parameters from seeing, noise, and masks."""

    if settings is None:
        settings = get_default_settings()
    subtraction = settings.get("subtraction", {})
    configured = subtraction.get("hotpants", {})
    science = _record_data(science_record)
    template_data = np.asarray(aligned_template["data"], dtype=float)
    science_mask = _record_mask(science_record, science.shape)
    template_mask = np.asarray(aligned_template["mask"], dtype=bool)
    science_background, science_rms = _robust_location_scale(science[~science_mask])
    template_background, template_rms = _robust_location_scale(template_data[~template_mask])
    science_fwhm = _seeing_pixels(science_record)
    template_fwhm = _finite_float((template.get("metadata") or {}).get("fwhm_pixels"))
    if template_fwhm is None:
        arcsec = _finite_float((template.get("metadata") or {}).get("fwhm_arcsec"))
        scale = _pixel_scale_arcsec(aligned_template.get("wcs"))
        template_fwhm = arcsec / scale if arcsec is not None and scale else science_fwhm
    broader = max(science_fwhm, template_fwhm)
    kernel_radius = max(5, int(np.ceil(2.5 * broader)))
    stamp_grid = max(3, min(12, int(np.sqrt(science.size) / max(20 * broader, 1))))
    science_saturation = _finite_float((science_record.get("metadata") or {}).get("saturation"))
    template_saturation = _finite_float((template.get("metadata") or {}).get("saturation"))
    parameters = {
        "convolve": "template" if template_fwhm <= science_fwhm else "science",
        "science_fwhm_pixels": science_fwhm,
        "template_fwhm_pixels": template_fwhm,
        "science_background": science_background,
        "template_background": template_background,
        "science_rms": science_rms,
        "template_rms": template_rms,
        "science_lower": science_background - 5 * science_rms,
        "template_lower": template_background - 5 * template_rms,
        "science_upper": science_saturation or science_background + 500 * science_rms,
        "template_upper": template_saturation or template_background + 500 * template_rms,
        "kernel_radius": kernel_radius,
        "stamp_radius": max(kernel_radius + 3, int(np.ceil(4 * broader))),
        "stamp_grid_x": stamp_grid,
        "stamp_grid_y": stamp_grid,
        "stamps_per_cell": 3,
        "kernel_order": 1,
        "background_order": 1,
        "gaussian_components": [(6, 0.7), (4, 1.5), (2, 3.0)],
    }
    aliases = {
        "kernel_order": "kernel_order",
        "background_order": "background_order",
        "kernel_radius": "kernel_radius",
    }
    for source, destination in aliases.items():
        value = configured.get(source, "auto")
        if value != "auto":
            parameters[destination] = int(value)
    grid = configured.get("stamp_grid", "auto")
    if grid != "auto":
        parameters["stamp_grid_x"] = int(grid[0])
        parameters["stamp_grid_y"] = int(grid[1])
    stamp_count = configured.get("stamp_count", "auto")
    if stamp_count != "auto":
        parameters["stamps_per_cell"] = int(stamp_count)
    for setting_name, parameter_names in (
        ("lower_threshold", ("science_lower", "template_lower")),
        ("upper_threshold", ("science_upper", "template_upper")),
    ):
        value = configured.get(setting_name, "auto")
        if value == "auto":
            continue
        if isinstance(value, Mapping):
            parameters[parameter_names[0]] = float(value.get("science"))
            parameters[parameter_names[1]] = float(value.get("template"))
        else:
            parameters[parameter_names[0]] = float(value)
            parameters[parameter_names[1]] = float(value)
    return parameters


def match_background_and_scale(science_record, aligned_template, settings=None,
                               quality_stars=None):
    """Estimate the robust linear relation between template and science pixels.

    The returned relation is ``science = scale * template + background``.  It
    is an initialization and diagnostic for Hotpants, which performs its own
    spatial kernel solution, and is applied directly for the PyZOGY route.
    """

    if settings is None:
        settings = get_default_settings()
    subtraction = settings.get("subtraction", {})
    science = _record_data(science_record)
    template = np.asarray(aligned_template["data"], dtype=float)
    mask = _record_mask(science_record, science.shape) | np.asarray(
        aligned_template["mask"], dtype=bool
    )
    valid = ~mask & np.isfinite(science) & np.isfinite(template)
    minimum = int(subtraction.get("minimum_scale_pixels", 1000))
    if np.count_nonzero(valid) < minimum:
        raise ValueError("Too few common pixels to match template scale")
    science_values = science[valid]
    template_values = template[valid]
    science_center, _ = _robust_location_scale(science_values)
    template_center, template_rms = _robust_location_scale(template_values)
    positions = _quality_positions(science_record, quality_stars)
    aperture_ratios = []
    if positions:
        radius = max(3.0, 2.5 * _seeing_pixels(science_record))
        for x, y, _ in positions:
            science_flux = _aperture_sum(science - science_center, x, y, radius, mask)
            template_flux = _aperture_sum(template - template_center, x, y, radius, mask)
            if (
                science_flux is not None and template_flux is not None
                and science_flux > 0 and template_flux > 0
            ):
                aperture_ratios.append(science_flux / template_flux)
    if len(aperture_ratios) >= 3:
        ratios = np.asarray(aperture_ratios, dtype=float)
        ratio_center, ratio_scatter = _robust_location_scale(ratios)
        keep_ratio = np.ones(ratios.size, dtype=bool)
        if ratio_scatter not in {None, 0.0}:
            keep_ratio = np.abs(ratios - ratio_center) <= float(
                subtraction.get("scale_sigma_clip", 3.0)
            ) * ratio_scatter
        scale = float(np.median(ratios[keep_ratio]))
        return {
            "scale": scale,
            "background": float(science_center - scale * template_center),
            "scatter": float(1.4826 * np.median(np.abs(ratios[keep_ratio] - scale))),
            "pixel_count": int(np.count_nonzero(valid)),
            "rejected_pixel_count": 0,
            "star_count": int(np.count_nonzero(keep_ratio)),
            "method": "quality_star_apertures",
        }
    source_like = template_values > template_center + 2.0 * (template_rms or 0.0)
    if np.count_nonzero(source_like) >= minimum:
        science_values = science_values[source_like]
        template_values = template_values[source_like]
    maximum_samples = 200_000
    if science_values.size > maximum_samples:
        indices = np.linspace(0, science_values.size - 1, maximum_samples).astype(int)
        science_values = science_values[indices]
        template_values = template_values[indices]
    design = np.column_stack([template_values, np.ones(template_values.size)])
    keep = np.ones(template_values.size, dtype=bool)
    sigma = float(subtraction.get("scale_sigma_clip", 3.0))
    iterations = int(subtraction.get("scale_maximum_iterations", 5))
    coefficients = np.array([1.0, 0.0])
    for _ in range(iterations):
        coefficients = np.linalg.lstsq(design[keep], science_values[keep], rcond=None)[0]
        residual = science_values - design @ coefficients
        center, scatter = _robust_location_scale(residual[keep])
        if scatter in {None, 0.0}:
            break
        updated = np.abs(residual - center) <= sigma * scatter
        if np.count_nonzero(updated) < minimum or np.array_equal(updated, keep):
            break
        keep = updated
    residual = science_values[keep] - design[keep] @ coefficients
    _, scatter = _robust_location_scale(residual)
    return {
        "scale": float(coefficients[0]),
        "background": float(coefficients[1]),
        "scatter": scatter,
        "pixel_count": int(np.count_nonzero(keep)),
        "rejected_pixel_count": int(keep.size - np.count_nonzero(keep)),
        "star_count": 0,
        "method": "robust_pixels",
    }


def _hotpants_command(executable, science_path, template_path, output_path,
                      parameters, settings, science_mask_path=None,
                      template_mask_path=None):
    """Build a shell-free Hotpants command line."""

    command = [
        executable,
        "-inim", str(science_path), "-tmplim", str(template_path),
        "-outim", str(output_path),
        "-c", "t" if parameters["convolve"] == "template" else "i",
        "-il", "{:.8g}".format(parameters["science_lower"]),
        "-iu", "{:.8g}".format(parameters["science_upper"]),
        "-tl", "{:.8g}".format(parameters["template_lower"]),
        "-tu", "{:.8g}".format(parameters["template_upper"]),
        "-r", str(parameters["kernel_radius"]),
        "-rss", str(parameters["stamp_radius"]),
        "-nsx", str(parameters["stamp_grid_x"]),
        "-nsy", str(parameters["stamp_grid_y"]),
        "-nss", str(parameters["stamps_per_cell"]),
        "-ko", str(parameters["kernel_order"]),
        "-bgo", str(parameters["background_order"]),
        "-hki",
        "-ng", str(len(parameters["gaussian_components"])),
    ]
    for degree, sigma in parameters["gaussian_components"]:
        command.extend([str(degree), "{:.4g}".format(sigma)])
    if science_mask_path is not None:
        command.extend(["-imi", str(science_mask_path)])
    if template_mask_path is not None:
        command.extend(["-tmi", str(template_mask_path)])
    command.extend(str(value) for value in settings.get("hotpants", {}).get("extra_arguments", []))
    return command


def _run_hotpants(science, aligned_template, header, parameters, settings,
                  science_mask=None, template_mask=None):
    """Execute Hotpants in a temporary directory and validate its FITS output."""

    executable = settings.get("hotpants_executable", "hotpants")
    resolved = shutil.which(executable)
    if resolved is None:
        raise FileNotFoundError("Hotpants executable not found: {}".format(executable))
    with tempfile.TemporaryDirectory(prefix="redphot_hotpants_") as directory:
        directory = Path(directory)
        science_path = directory / "science.fits"
        template_path = directory / "template.fits"
        science_mask_path = directory / "science_mask.fits"
        template_mask_path = directory / "template_mask.fits"
        output_path = directory / "difference.fits"
        fits.PrimaryHDU(np.asarray(science, dtype=np.float32), header).writeto(science_path)
        fits.PrimaryHDU(np.asarray(aligned_template, dtype=np.float32), header).writeto(template_path)
        if science_mask is not None:
            fits.PrimaryHDU(np.asarray(science_mask, dtype=np.uint8)).writeto(
                science_mask_path
            )
        else:
            science_mask_path = None
        if template_mask is not None:
            fits.PrimaryHDU(np.asarray(template_mask, dtype=np.uint8)).writeto(
                template_mask_path
            )
        else:
            template_mask_path = None
        command = _hotpants_command(
            resolved, science_path, template_path, output_path, parameters, settings,
            science_mask_path, template_mask_path,
        )
        completed = subprocess.run(
            command, capture_output=True, text=True,
            timeout=float(settings.get("execution_timeout_s", 300)), check=False,
        )
        log = {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        if completed.returncode != 0:
            raise RuntimeError("Hotpants failed with exit code {}: {}".format(
                completed.returncode, completed.stderr.strip()
            ))
        if not output_path.exists():
            raise RuntimeError("Hotpants reported success but produced no difference image")
        difference, output_header = fits.getdata(output_path, header=True)
        difference = np.asarray(difference, dtype=float)
        if difference.shape != science.shape or not np.isfinite(difference).any():
            raise RuntimeError("Hotpants produced an invalid difference image")
        log["kernel_header"] = {
            key: output_header[key]
            for key in output_header
            if key.startswith(("CONVOL", "KSUM", "REGION"))
        }
    return difference, log


def _run_pyzogy(science, aligned_template, science_record, template, settings,
                 runner=None):
    """Run PyZOGY through an injected or installed functional entry point."""

    if runner is None:
        try:
            import pyzogy
        except ImportError as error:
            raise FileNotFoundError("PyZOGY is not installed") from error
        runner = getattr(pyzogy, "run_subtraction", None)
        if runner is None:
            raise RuntimeError(
                "Installed PyZOGY has no run_subtraction function; provide pyzogy_runner"
            )
    result = runner(
        science=np.asarray(science, dtype=float),
        reference=np.asarray(aligned_template, dtype=float),
        science_variance=science_record.get("variance"),
        reference_variance=template.get("variance"),
        science_psf=(science_record.get("psf_result") or {}).get("model_native"),
        reference_psf=(template.get("psf_result") or {}).get("model_native"),
        settings=settings.get("pyzogy", {}),
    )
    if isinstance(result, Mapping):
        difference = result.get("difference")
        log = {key: value for key, value in result.items() if key != "difference"}
    else:
        difference, log = result, {}
    difference = np.asarray(difference, dtype=float)
    if difference.shape != science.shape or not np.isfinite(difference).any():
        raise RuntimeError("PyZOGY produced an invalid difference image")
    return difference, log


def _aperture_sum(data, x, y, radius, mask=None):
    """Return a fractional-pixel-free circular aperture sum for diagnostics."""

    y0 = max(0, int(np.floor(y - radius)))
    y1 = min(data.shape[0], int(np.ceil(y + radius)) + 1)
    x0 = max(0, int(np.floor(x - radius)))
    x1 = min(data.shape[1], int(np.ceil(x + radius)) + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    inside = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
    values = np.asarray(data[y0:y1, x0:x1], dtype=float)
    good = inside & np.isfinite(values)
    if mask is not None:
        good &= ~np.asarray(mask[y0:y1, x0:x1], dtype=bool)
    return float(np.sum(values[good])) if np.count_nonzero(good) >= 3 else None


def _quality_positions(science_record, quality_stars):
    """Select finite detector positions for non-variable quality stars."""

    if quality_stars is None:
        quality_stars = science_record.get("measurements")
    if quality_stars is None:
        return []
    positions = []
    image_id = _image_id(science_record)
    for row in quality_stars:
        names = getattr(row, "colnames", [])
        if "image_id" in names and str(row["image_id"]) != image_id:
            continue
        role_ok = True
        for name in ("role_qc_anchor", "role_calibration", "role_psf"):
            if name in names and bool(row[name]):
                role_ok = True
                break
        x = _finite_float(row["x"] if "x" in names else None)
        y = _finite_float(row["y"] if "y" in names else None)
        if role_ok and x is not None and y is not None:
            positions.append((x, y, str(row["source_id"]) if "source_id" in names else None))
    return positions


def evaluate_subtraction(science_record, aligned_template, difference,
                         settings=None, quality_stars=None):
    """Evaluate residuals, dipoles, flux bias, background, and blank noise."""

    if settings is None:
        settings = get_default_settings()
    subtraction = settings.get("subtraction", {})
    science = _record_data(science_record)
    difference = np.asarray(difference, dtype=float)
    mask = _record_mask(science_record, science.shape) | np.asarray(
        aligned_template["mask"], dtype=bool
    ) | ~np.isfinite(difference)
    background, rms = _robust_location_scale(difference[~mask])
    _, science_rms = _robust_location_scale(science[~mask])
    positions = _quality_positions(science_record, quality_stars)
    fwhm = _seeing_pixels(science_record)
    radius = max(2.0, 1.5 * fwhm)
    rows = []
    residual_fractions = []
    dipoles = []
    for x, y, source_id in positions:
        science_flux = _aperture_sum(science, x, y, radius, mask)
        residual_flux = _aperture_sum(difference, x, y, radius, mask)
        if science_flux is None or residual_flux is None or science_flux == 0:
            continue
        residual_fraction = abs(residual_flux / science_flux)
        y0, y1 = max(0, int(y - radius)), min(science.shape[0], int(y + radius) + 1)
        x0, x1 = max(0, int(x - radius)), min(science.shape[1], int(x + radius) + 1)
        patch = difference[y0:y1, x0:x1]
        threshold = 3.0 * (rms or 0.0)
        positive = np.count_nonzero(patch > threshold)
        negative = np.count_nonzero(patch < -threshold)
        dipole = min(positive, negative) / max(positive + negative, 1)
        residual_fractions.append(residual_fraction)
        dipoles.append(dipole)
        rows.append((source_id, x, y, science_flux, residual_flux, residual_fraction, dipole))
    table = Table(
        rows=rows,
        names=("source_id", "x", "y", "science_flux", "residual_flux",
               "residual_fraction", "dipole_fraction"),
        masked=True,
    )
    count = int(subtraction.get("blank_aperture_count", 50))
    rng = np.random.default_rng(int(subtraction.get("blank_aperture_seed", 12345)))
    blank = []
    source_mask = mask.copy()
    for x, y, _ in positions:
        yy, xx = np.ogrid[:science.shape[0], :science.shape[1]]
        source_mask |= (xx - x) ** 2 + (yy - y) ** 2 <= (3 * fwhm) ** 2
    attempts = 0
    while len(blank) < count and attempts < count * 30:
        attempts += 1
        x = rng.uniform(radius, science.shape[1] - radius)
        y = rng.uniform(radius, science.shape[0] - radius)
        if source_mask[int(round(y)), int(round(x))]:
            continue
        flux = _aperture_sum(difference, x, y, radius, source_mask)
        if flux is not None:
            blank.append(flux)
    _, blank_rms = _robust_location_scale(blank)
    expected_noise = None
    if science_rms is not None:
        expected_noise = science_rms * np.sqrt(np.pi * radius ** 2)
    noise_ratio = (
        blank_rms / expected_noise
        if blank_rms is not None and expected_noise not in {None, 0.0} else None
    )
    residual = float(np.median(residual_fractions)) if residual_fractions else None
    dipole = float(np.median(dipoles)) if dipoles else None
    flux_bias = (
        abs(float(np.median([row[4] for row in rows])))
        / max(float(np.median(np.abs([row[3] for row in rows]))), 1.0e-12)
        if rows else None
    )
    flags = []
    minimum = int(subtraction.get("minimum_quality_stars", 3))
    if len(rows) < minimum:
        flags.append("SUBTRACTION_RESIDUAL_HIGH")
    if residual is not None and residual > float(subtraction.get("maximum_residual_fraction", 0.10)):
        flags.append("SUBTRACTION_RESIDUAL_HIGH")
    if dipole is not None and dipole > float(subtraction.get("maximum_dipole_fraction", 0.20)):
        flags.append("SUBTRACTION_DIPOLE")
    if flux_bias is not None and flux_bias > float(subtraction.get("maximum_flux_bias_fraction", 0.10)):
        flags.append("SUBTRACTION_FLUX_LOSS")
    if noise_ratio is not None and noise_ratio > float(subtraction.get("maximum_noise_ratio", 2.0)):
        flags.append("SUBTRACTION_NOISE_HIGH")
    return {
        "status": "PASS" if not flags else "FAIL",
        "flags": list(dict.fromkeys(flags)),
        "background": background,
        "background_rms": rms,
        "median_residual_fraction": residual,
        "median_dipole_fraction": dipole,
        "flux_bias_fraction": flux_bias,
        "blank_aperture_rms": blank_rms,
        "blank_aperture_count": len(blank),
        "noise_ratio": noise_ratio,
        "star_residuals": table,
        "blank_aperture_fluxes": np.asarray(blank, dtype=float),
    }


def perform_image_subtraction(science_record, template, settings=None,
                              quality_stars=None, pyzogy_runner=None):
    """Align, subtract, validate, and classify one science/template pair."""

    if settings is None:
        settings = get_default_settings()
    settings = merge_settings(get_default_settings(), settings)
    subtraction = settings.get("subtraction", {})
    image_id = _image_id(science_record)
    result = {
        "image_id": image_id,
        "status": "FAIL",
        "flags": [],
        "method": subtraction.get("method", "hotpants"),
        "science_grid_unchanged": True,
        "difference": None,
        "aligned_template": None,
        "parameters": {},
        "log": {},
        "template": template,
    }
    try:
        template_check = validate_template(
            template, [science_record], settings,
            normalize_filter_name((science_record.get("metadata") or {}).get("filter")),
        )
        result["template_validation"] = template_check
        if template_check["status"] == "FAIL":
            result["flags"].extend(template_check["flags"])
            return result
        aligned = align_template_to_science(science_record, template, settings)
        result["aligned_template"] = aligned
        if aligned["coverage_fraction"] < float(subtraction.get("minimum_coverage_fraction", 0.99)):
            result["flags"].append("TEMPLATE_COVERAGE_INCOMPLETE")
            return result
        parameters = choose_hotpants_parameters(science_record, template, aligned, settings)
        scale_match = match_background_and_scale(
            science_record, aligned, settings, quality_stars
        )
        parameters["initial_photometric_scale"] = scale_match["scale"]
        parameters["initial_background_offset"] = scale_match["background"]
        result["scale_match"] = scale_match
        result["parameters"] = parameters
        science = _record_data(science_record)
        header = _record_wcs(science_record).to_header(relax=True)
        methods = [subtraction.get("method", "hotpants")]
        fallback = subtraction.get("fallback_method")
        if fallback and fallback not in methods:
            methods.append(fallback)
        errors = []
        for method in methods:
            try:
                if method == "hotpants":
                    difference, log = _run_hotpants(
                        science, aligned["data"], header, parameters, subtraction,
                        _record_mask(science_record, science.shape), aligned["mask"],
                    )
                elif method == "pyzogy":
                    matched_template = aligned["data"]
                    if subtraction.get("photometric_scale", True):
                        matched_template = scale_match["scale"] * matched_template
                    if subtraction.get("background_match", True):
                        matched_template = matched_template + scale_match["background"]
                    difference, log = _run_pyzogy(
                        science, matched_template, science_record, template,
                        subtraction, pyzogy_runner,
                    )
                else:
                    raise ValueError("Unknown subtraction backend: {}".format(method))
                result["method"] = method
                result["difference"] = difference
                result["log"] = log
                break
            except FileNotFoundError as error:
                errors.append("{}: {}".format(method, error))
                result["flags"].append("SUBTRACTION_BACKEND_MISSING")
            except Exception as error:
                errors.append("{}: {}".format(method, error))
        if result["difference"] is None:
            result["flags"].append("SUBTRACTION_FAILED")
            result["error"] = "; ".join(errors)
            return result
        quality = evaluate_subtraction(
            science_record, aligned, result["difference"], settings, quality_stars
        )
        result["quality"] = quality
        result["flags"].extend(quality["flags"])
        result["flags"] = list(dict.fromkeys(result["flags"]))
        result["status"] = "PASS" if quality["status"] == "PASS" else "FAIL"
    except Exception as error:
        result["flags"].append("SUBTRACTION_FAILED")
        result["flags"] = list(dict.fromkeys(result["flags"]))
        result["error"] = str(error)
    return result


def perform_subtractions(image_records, templates, settings=None,
                         quality_stars=None, pyzogy_runner=None):
    """Perform checked subtraction for every image using filter-keyed templates."""

    if settings is None:
        settings = get_default_settings()
    results = []
    for index, record in enumerate(image_records):
        filter_name = normalize_filter_name((record.get("metadata") or {}).get("filter"))
        if isinstance(templates, Mapping) and "data" not in templates:
            template = templates.get(filter_name, templates.get("default"))
        else:
            template = templates
        if template is None:
            results.append({
                "image_id": _image_id(record, index), "status": "FAIL",
                "flags": ["TEMPLATE_MISSING"], "difference": None,
            })
            continue
        results.append(perform_image_subtraction(
            record, template, settings, quality_stars, pyzogy_runner
        ))
    return results


def save_template_products(template, output_directory, filter_name=None,
                           settings=None, overwrite=None):
    """Save a template mosaic, coverage map, and acquisition summary."""

    if settings is None:
        settings = get_default_settings()
    if overwrite is None:
        overwrite = settings.get("output", {}).get("overwrite", False)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    filter_name = normalize_filter_name(
        filter_name or (template.get("metadata") or {}).get("filter")
    ) or "unknown"
    stem = "template_{}".format(filter_name)
    header = template["wcs"].to_header(relax=True)
    header["TMPLSRC"] = str(template.get("source", "unknown"))[:68]
    paths = {}
    path = output / "{}_mosaic.fits".format(stem)
    fits.PrimaryHDU(template["data"], header).writeto(path, overwrite=bool(overwrite))
    paths["template"] = str(path)
    if template.get("coverage") is not None:
        path = output / "{}_coverage.fits".format(stem)
        fits.PrimaryHDU(np.asarray(template["coverage"], dtype=np.int16), header).writeto(
            path, overwrite=bool(overwrite)
        )
        paths["coverage"] = str(path)
    path = output / "{}_acquisition.json".format(stem)
    if path.exists() and not overwrite:
        raise FileExistsError(str(path))
    footprint = template.get("requested_footprint") or {}
    summary = {
        "filter": filter_name,
        "source": template.get("source"),
        "tile_count": len(template.get("tiles", [])),
        "cached_path": template.get("cached_path"),
        "acquisition": template.get("acquisition", {}),
        "requested_center_ra_deg": (
            None if footprint.get("center") is None else footprint["center"].ra.deg
        ),
        "requested_center_dec_deg": (
            None if footprint.get("center") is None else footprint["center"].dec.deg
        ),
        "requested_width_arcmin": footprint.get("width_arcmin"),
        "requested_height_arcmin": footprint.get("height_arcmin"),
    }
    path.write_text(json.dumps(summary, indent=2, default=str, sort_keys=True) + "\n")
    paths["acquisition"] = str(path)
    return paths


def subtraction_table(results):
    """Return a compact masked table of subtraction decisions and metrics."""

    rows = []
    for result in results:
        quality = result.get("quality") or {}
        aligned = result.get("aligned_template") or {}
        rows.append({
            "image_id": result.get("image_id"),
            "status": result.get("status"),
            "method": result.get("method"),
            "flags": ",".join(result.get("flags", [])),
            "coverage_fraction": aligned.get("coverage_fraction"),
            "background": quality.get("background"),
            "background_rms": quality.get("background_rms"),
            "residual_fraction": quality.get("median_residual_fraction"),
            "dipole_fraction": quality.get("median_dipole_fraction"),
            "flux_bias_fraction": quality.get("flux_bias_fraction"),
            "noise_ratio": quality.get("noise_ratio"),
            "error": result.get("error"),
        })
    table = Table(rows=rows, masked=True)
    for name in (
        "coverage_fraction", "background", "background_rms", "residual_fraction",
        "dipole_fraction", "flux_bias_fraction", "noise_ratio",
    ):
        values = np.array([_finite_float(row.get(name), np.nan) for row in rows])
        table.replace_column(name, MaskedColumn(values, mask=~np.isfinite(values), name=name))
    return table


def save_subtraction_products(result, output_directory, settings=None,
                              overwrite=None):
    """Save accepted or failed subtraction products, logs, and parameters."""

    if settings is None:
        settings = get_default_settings()
    subtraction = settings.get("subtraction", {})
    if overwrite is None:
        overwrite = settings.get("output", {}).get("overwrite", False)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    stem = Path(str(result.get("image_id", "image"))).name
    for ending in (".fits.fz", ".fits.gz", ".fits", ".fit", ".fts"):
        if stem.lower().endswith(ending):
            stem = stem[:-len(ending)]
            break
    paths = {}
    aligned = result.get("aligned_template")
    if aligned is not None and subtraction.get("save_aligned_template", True):
        path = output / "{}_aligned_template.fits".format(stem)
        fits.PrimaryHDU(aligned["data"], aligned["wcs"].to_header()).writeto(
            path, overwrite=bool(overwrite)
        )
        paths["aligned_template"] = str(path)
    difference = result.get("difference")
    if difference is not None and subtraction.get("save_difference", True):
        path = output / "{}_difference.fits".format(stem)
        header = aligned["wcs"].to_header() if aligned is not None else fits.Header()
        header["SUBSTAT"] = str(result.get("status", "FAIL"))
        header["SUBMETH"] = str(result.get("method", "unknown"))
        fits.PrimaryHDU(difference, header).writeto(path, overwrite=bool(overwrite))
        paths["difference"] = str(path)
    if subtraction.get("save_parameters", True):
        path = output / "{}_subtraction_parameters.json".format(stem)
        if path.exists() and not overwrite:
            raise FileExistsError(str(path))
        summary = {
            "image_id": result.get("image_id"), "status": result.get("status"),
            "method": result.get("method"), "flags": result.get("flags", []),
            "parameters": result.get("parameters", {}), "error": result.get("error"),
        }
        path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        paths["parameters"] = str(path)
    if subtraction.get("save_logs", True) and result.get("log"):
        path = output / "{}_subtraction.log".format(stem)
        if path.exists() and not overwrite:
            raise FileExistsError(str(path))
        path.write_text(json.dumps(result["log"], indent=2, default=str) + "\n")
        paths["log"] = str(path)
    quality = result.get("quality") or {}
    table = quality.get("star_residuals")
    if table is not None and subtraction.get("save_quality_table", True):
        path = output / "{}_subtraction_quality.ecsv".format(stem)
        table.write(path, format="ascii.ecsv", overwrite=bool(overwrite))
        paths["quality"] = str(path)
    return paths


__all__ = [
    "acquire_template",
    "align_template_to_science",
    "choose_hotpants_parameters",
    "evaluate_subtraction",
    "mosaic_template_tiles",
    "match_background_and_scale",
    "perform_image_subtraction",
    "perform_subtractions",
    "read_template",
    "save_subtraction_products",
    "save_template_products",
    "science_footprint_union",
    "subtraction_table",
    "validate_template",
]
