"""
Discover and read reduced optical FITS images for redphot.

The functions in this module never modify an input file.  They locate science,
mask, variance, and WCS information across ordinary or compressed FITS HDUs and
return an Astropy ``CCDData`` object with a separate normalized metadata
dictionary.
"""

from collections.abc import Mapping
from glob import glob
from pathlib import Path
import warnings

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import (
    CCDData,
    Cutout2D,
    InverseVariance,
    StdDevUncertainty,
    VarianceUncertainty,
)
from astropy.table import MaskedColumn, Table
from astropy.time import Time
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from .config import get_default_settings, normalize_filter_name
from .metadata import normalize_and_validate_metadata


FITS_ENDINGS = (
    ".fits",
    ".fit",
    ".fts",
    ".fits.fz",
    ".fit.fz",
    ".fts.fz",
    ".fits.gz",
    ".fit.gz",
    ".fts.gz",
)

SCIENCE_EXTNAMES = ("SCI", "IMAGE", "IM1", "IM2")
MASK_EXTNAMES = ("MASK", "DQ", "BPM", "BADPIX", "BAD_PIXEL_MASK")
VARIANCE_EXTNAMES = ("VAR", "VARIANCE")
ERROR_EXTNAMES = ("ERR", "ERROR", "RMS", "SIGMA", "UNCERT", "UNCERTAINTY")
INVERSE_VARIANCE_EXTNAMES = ("IVAR", "INVVAR", "INVERSE_VARIANCE", "WEIGHT")
AUXILIARY_EXTNAMES = set(
    MASK_EXTNAMES
    + VARIANCE_EXTNAMES
    + ERROR_EXTNAMES
    + INVERSE_VARIANCE_EXTNAMES
)

METADATA_FIELDS = (
    "path",
    "filename",
    "object",
    "telescope",
    "instrument",
    "detector",
    "site",
    "exposure_time",
    "exposure_time_from_times",
    "date_obs",
    "date_end",
    "date_start_utc",
    "date_mid_utc",
    "date_end_utc",
    "mjd",
    "mjd_utc",
    "mjd_start",
    "mjd_mid",
    "mjd_end",
    "mjd_header",
    "mjd_header_difference_s",
    "time_reference",
    "filter",
    "gain",
    "read_noise",
    "saturation",
    "nonlinearity",
    "airmass",
    "pointing_ra",
    "pointing_dec",
    "target_ra",
    "target_dec",
    "pointing_ra_deg",
    "pointing_dec_deg",
    "header_target_ra_deg",
    "header_target_dec_deg",
    "wcs_center_ra_deg",
    "wcs_center_dec_deg",
    "user_target_ra_deg",
    "user_target_dec_deg",
    "adopted_ra_deg",
    "adopted_dec_deg",
    "adopted_position_source",
    "adopted_position_is_preliminary",
    "wcs_valid",
    "pixel_scale",
    "binning",
    "binning_x",
    "binning_y",
    "reduction_level",
    "pipeline_version",
    "pipeline_fwhm_arcsec",
    "pipeline_ellipticity",
    "pipeline_zeropoint_mag",
    "pipeline_saturated_fraction",
    "pipeline_wcs_error",
    "metadata_valid",
    "metadata_status",
    "quality_flags",
    "data_hdu",
    "data_extname",
    "header_hdu",
    "wcs_hdu",
    "mask_hdu",
    "variance_hdu",
    "uncertainty_type",
    "shape_y",
    "shape_x",
    "dtype",
    "finite_fraction",
)

FLOAT_METADATA_FIELDS = {
    "exposure_time",
    "exposure_time_from_times",
    "mjd",
    "mjd_utc",
    "mjd_start",
    "mjd_mid",
    "mjd_end",
    "mjd_header",
    "mjd_header_difference_s",
    "gain",
    "read_noise",
    "saturation",
    "nonlinearity",
    "airmass",
    "pixel_scale",
    "pointing_ra_deg",
    "pointing_dec_deg",
    "header_target_ra_deg",
    "header_target_dec_deg",
    "wcs_center_ra_deg",
    "wcs_center_dec_deg",
    "user_target_ra_deg",
    "user_target_dec_deg",
    "adopted_ra_deg",
    "adopted_dec_deg",
    "pipeline_fwhm_arcsec",
    "pipeline_ellipticity",
    "pipeline_zeropoint_mag",
    "pipeline_saturated_fraction",
    "pipeline_wcs_error",
    "finite_fraction",
}

BOOLEAN_METADATA_FIELDS = {
    "adopted_position_is_preliminary",
    "wcs_valid",
    "metadata_valid",
}

METADATA_UNITS = {
    "exposure_time": u.s,
    "exposure_time_from_times": u.s,
    "mjd": u.day,
    "mjd_utc": u.day,
    "mjd_start": u.day,
    "mjd_mid": u.day,
    "mjd_end": u.day,
    "mjd_header": u.day,
    "mjd_header_difference_s": u.s,
    "gain": u.electron / u.adu,
    "read_noise": u.electron,
    "saturation": u.adu,
    "nonlinearity": u.adu,
    "pixel_scale": u.arcsec / u.pixel,
    "pointing_ra_deg": u.deg,
    "pointing_dec_deg": u.deg,
    "header_target_ra_deg": u.deg,
    "header_target_dec_deg": u.deg,
    "wcs_center_ra_deg": u.deg,
    "wcs_center_dec_deg": u.deg,
    "user_target_ra_deg": u.deg,
    "user_target_dec_deg": u.deg,
    "adopted_ra_deg": u.deg,
    "adopted_dec_deg": u.deg,
    "pipeline_fwhm_arcsec": u.arcsec,
    "pipeline_zeropoint_mag": u.mag,
}

INTEGER_METADATA_FIELDS = {
    "binning_x",
    "binning_y",
    "data_hdu",
    "header_hdu",
    "wcs_hdu",
    "mask_hdu",
    "variance_hdu",
    "shape_y",
    "shape_x",
}


def is_fits_path(path):
    """Return ``True`` when a path has a supported FITS filename ending."""

    return str(path).lower().endswith(FITS_ENDINGS)


def _is_compressed_fits(path):
    """Return whether a FITS path uses gzip or tile-compressed naming."""

    return str(path).lower().endswith((".fz", ".gz"))


def discover_fits_files(paths=None, settings=None):
    """Discover supported FITS images from files, directories, or glob patterns.

    Parameters
    ----------
    paths : str, pathlib.Path, or sequence, optional
        Input file, directory, glob expression, or collection of them.  If not
        supplied, ``settings['input']['paths']`` is used.
    settings : mapping, optional
        Resolved redphot settings.  The ``input.recursive`` value controls
        directory traversal.

    Returns
    -------
    list of pathlib.Path
        Unique absolute paths in deterministic filename order.

    Raises
    ------
    FileNotFoundError
        If an explicitly supplied path does not exist or a glob has no matches.
    ValueError
        If an explicit file is not a supported FITS file or no FITS files are
        discovered.
    """

    if settings is None:
        settings = get_default_settings()

    input_settings = settings.get("input", {})
    recursive = bool(input_settings.get("recursive", False))
    allow_compressed = bool(input_settings.get("allow_compressed", True))

    if paths is None:
        paths = input_settings.get("paths", [])
    if isinstance(paths, (str, Path)):
        paths = [paths]
    else:
        paths = list(paths or [])

    if not paths:
        raise ValueError("No FITS input paths were supplied.")

    discovered = []
    for supplied_path in paths:
        text = str(supplied_path)
        has_glob = any(character in text for character in "*?[")

        if has_glob:
            matches = [Path(item) for item in glob(text, recursive=recursive)]
            if not matches:
                raise FileNotFoundError("No files match {!r}.".format(text))
            for match in matches:
                if (
                    match.is_file()
                    and is_fits_path(match)
                    and (allow_compressed or not _is_compressed_fits(match))
                ):
                    discovered.append(match)
            continue

        path = Path(supplied_path).expanduser()
        if not path.exists():
            raise FileNotFoundError("FITS input does not exist: {}".format(path))

        if path.is_file():
            if not is_fits_path(path):
                raise ValueError("Unsupported FITS filename: {}".format(path))
            if _is_compressed_fits(path) and not allow_compressed:
                raise ValueError(
                    "Compressed FITS input is disabled in the settings: {}".format(
                        path
                    )
                )
            discovered.append(path)
            continue

        iterator = path.rglob("*") if recursive else path.iterdir()
        discovered.extend(
            item
            for item in iterator
            if item.is_file()
            and is_fits_path(item)
            and (allow_compressed or not _is_compressed_fits(item))
        )

    unique = {}
    for path in discovered:
        resolved = path.resolve()
        unique[str(resolved)] = resolved

    files = sorted(unique.values(), key=lambda item: str(item).lower())
    if not files:
        raise ValueError("No supported FITS images were discovered.")

    return files


def _hdu_extname(hdu):
    """Return a normalized extension name for an HDU."""

    name = hdu.header.get("EXTNAME", getattr(hdu, "name", ""))
    return str(name or "").strip().upper()


def _resolve_hdu_index(hdulist, selector, label):
    """Resolve an integer, extension name, or ``(name, version)`` selector."""

    if selector is None:
        return None

    if isinstance(selector, str) and selector.strip().lstrip("+-").isdigit():
        selector = int(selector)

    if isinstance(selector, (int, np.integer)):
        index = int(selector)
        if index < 0:
            index += len(hdulist)
        if index < 0 or index >= len(hdulist):
            raise ValueError(
                "The requested {} HDU {!r} does not exist.".format(label, selector)
            )
        return index

    try:
        index = hdulist.index_of(selector)
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise ValueError(
            "The requested {} HDU {!r} does not exist.".format(label, selector)
        ) from error

    return int(index)


def _hdu_search_order(
    hdulist, preferred=None, configured=None, include_remaining=True
):
    """Return unique valid HDU indices in the configured fallback order."""

    order = []
    requested = []
    if preferred is not None:
        requested.append(preferred)
    requested.extend(configured or [0, 1, 2])
    if include_remaining:
        requested.extend(range(len(hdulist)))

    for item in requested:
        try:
            index = _resolve_hdu_index(hdulist, item, "search")
        except ValueError:
            continue
        if index not in order:
            order.append(index)

    return order


def _numeric_2d_array(hdu):
    """Return an HDU's numeric two-dimensional array or ``None``."""

    try:
        data = hdu.data
    except (OSError, TypeError, ValueError):
        return None

    if data is None:
        return None

    array = np.asanyarray(data)
    if array.ndim != 2 or 0 in array.shape:
        return None
    if not np.issubdtype(array.dtype, np.number):
        return None
    if np.issubdtype(array.dtype, np.complexfloating):
        return None

    return array


def _finite_fraction(array):
    """Return the fraction of finite pixels in an array."""

    if array.size == 0:
        return 0.0
    return float(np.count_nonzero(np.isfinite(array)) / array.size)


def _combined_header(hdulist, index):
    """Combine the primary and selected extension headers without mutation."""

    header = hdulist[0].header.copy()
    if index != 0:
        header.extend(hdulist[index].header, update=True, useblanks=False)
    return header


def _wcs_for_data_hdu(hdulist, data_index, header_index, search_order):
    """Find a usable celestial WCS for a science array."""

    candidates = []
    if header_index is not None:
        candidates.append(header_index)
    candidates.append(data_index)
    candidates.extend(search_order)

    seen = set()
    for index in candidates:
        if index in seen:
            continue
        seen.add(index)

        header = _combined_header(hdulist, index)
        if data_index != index:
            header.extend(
                hdulist[data_index].header,
                update=False,
                unique=True,
                useblanks=False,
            )

        try:
            candidate = WCS(header, relax=True)
            if candidate.has_celestial:
                return candidate.celestial, index
        except Exception:
            continue

    return None, None


def _coerce_target_coordinate(target, settings):
    """Convert a target specification into ``SkyCoord`` when possible."""

    if target is None:
        target_settings = settings.get("target_position", {})
        ra = target_settings.get("ra")
        dec = target_settings.get("dec")
    elif isinstance(target, SkyCoord):
        return target.icrs
    elif isinstance(target, Mapping):
        ra = target.get("ra")
        dec = target.get("dec")
    else:
        try:
            ra, dec = target
        except (TypeError, ValueError) as error:
            raise ValueError(
                "target must be SkyCoord, a mapping, or an (RA, Dec) pair."
            ) from error

    if ra is None or dec is None:
        return None

    if isinstance(ra, (int, float, np.number)) and isinstance(
        dec, (int, float, np.number)
    ):
        return SkyCoord(float(ra), float(dec), unit=(u.deg, u.deg), frame="icrs")

    units = settings.get("target_position", {}).get(
        "coordinate_unit", ["hourangle", "deg"]
    )
    return SkyCoord(ra, dec, unit=tuple(units), frame="icrs")


def _target_coordinate_from_headers(hdulist, settings, search_order):
    """Find a target coordinate in headers, falling back to telescope pointing."""

    keywords = settings.get("metadata", {}).get("keywords", {})
    coordinate_pairs = (
        (keywords.get("target_ra", []), keywords.get("target_dec", [])),
        (keywords.get("pointing_ra", []), keywords.get("pointing_dec", [])),
    )

    for ra_keywords, dec_keywords in coordinate_pairs:
        ra, _ = _find_header_value(
            hdulist, ra_keywords, search_order, check_duplicates=False
        )
        dec, _ = _find_header_value(
            hdulist, dec_keywords, search_order, check_duplicates=False
        )
        if ra is None or dec is None:
            continue
        try:
            return _coerce_target_coordinate((ra, dec), settings)
        except (TypeError, ValueError):
            continue

    return None


def _target_inside_hdu(hdulist, index, target, header_index, search_order):
    """Return whether a target projects inside an HDU's pixel boundaries."""

    if target is None:
        return False

    array = _numeric_2d_array(hdulist[index])
    wcs, _ = _wcs_for_data_hdu(hdulist, index, header_index, search_order)
    if array is None or wcs is None:
        return False

    try:
        x, y = wcs.world_to_pixel(target)
    except Exception:
        return False

    ny, nx = array.shape
    return bool(
        np.isfinite(x)
        and np.isfinite(y)
        and -0.5 <= x < nx - 0.5
        and -0.5 <= y < ny - 0.5
    )


def select_science_hdu(hdulist, settings, target=None):
    """Select the science HDU using overrides, WCS coverage, and extension names.

    An explicit ``input.data_hdu`` is strict.  Automatic selection inspects HDUs
    0, 1, and 2 first, followed by all remaining HDUs.  If more than one valid
    science array exists, an array containing the supplied target wins.  The
    configured preferred extension names and then HDU search order break ties.

    Returns
    -------
    int
        Selected HDU index.
    """

    input_settings = settings.get("input", {})
    explicit = input_settings.get("data_hdu")
    if explicit is not None:
        index = _resolve_hdu_index(hdulist, explicit, "science data")
        if _numeric_2d_array(hdulist[index]) is None:
            raise ValueError(
                "The requested science HDU {!r} is not a numeric 2D image.".format(
                    explicit
                )
            )
        return index

    configured_order = input_settings.get("hdu_search_order", [0, 1, 2])
    search_order = _hdu_search_order(
        hdulist,
        configured=configured_order,
        include_remaining=input_settings.get("search_remaining_hdus", True),
    )
    minimum_finite = float(input_settings.get("minimum_finite_fraction", 0.5))

    preferred_names = list(input_settings.get("preferred_extnames", []))
    for name in SCIENCE_EXTNAMES:
        if name not in preferred_names:
            preferred_names.append(name)
    preferred_names = [str(name).strip().upper() for name in preferred_names]

    header_index = _resolve_hdu_index(
        hdulist, input_settings.get("header_hdu"), "header"
    )
    target = _coerce_target_coordinate(target, settings)
    if target is None:
        target = _target_coordinate_from_headers(hdulist, settings, search_order)

    candidates = []
    for order_position, index in enumerate(search_order):
        array = _numeric_2d_array(hdulist[index])
        if array is None:
            continue

        extname = _hdu_extname(hdulist[index])
        if extname in AUXILIARY_EXTNAMES:
            continue

        finite_fraction = _finite_fraction(array)
        if finite_fraction < minimum_finite:
            continue

        contains_target = False
        if input_settings.get("prefer_target_hdu", True):
            contains_target = _target_inside_hdu(
                hdulist, index, target, header_index, search_order
            )
        try:
            name_priority = len(preferred_names) - preferred_names.index(extname)
        except ValueError:
            name_priority = 0

        score = (
            int(contains_target),
            name_priority,
            finite_fraction,
            -order_position,
        )
        candidates.append((score, index))

    if not candidates:
        raise ValueError("No usable numeric 2D science image was found in the FITS file.")

    candidates.sort(reverse=True)
    return candidates[0][1]


def _matching_auxiliary_hdu(
    hdulist, data_index, selector, names, label, search_order
):
    """Find a shape-matched mask or uncertainty HDU."""

    science_shape = _numeric_2d_array(hdulist[data_index]).shape
    if selector is not None:
        index = _resolve_hdu_index(hdulist, selector, label)
        array = _numeric_2d_array(hdulist[index])
        if array is None or array.shape != science_shape:
            raise ValueError(
                "The requested {} HDU {!r} is not a numeric 2D image with shape {}."
                .format(label, selector, science_shape)
            )
        return index

    science_header = hdulist[data_index].header
    science_extver = science_header.get("EXTVER")
    science_chip = science_header.get("CCDCHIP", science_header.get("CCDNUM"))
    normalized_names = {str(name).strip().upper() for name in names}

    candidates = []
    for order_position, index in enumerate(search_order):
        if index == data_index:
            continue
        array = _numeric_2d_array(hdulist[index])
        if array is None or array.shape != science_shape:
            continue

        extname = _hdu_extname(hdulist[index])
        if extname not in normalized_names:
            continue

        header = hdulist[index].header
        extver_match = int(
            science_extver is not None and header.get("EXTVER") == science_extver
        )
        chip_match = int(
            science_chip is not None
            and header.get("CCDCHIP", header.get("CCDNUM")) == science_chip
        )
        candidates.append(((extver_match, chip_match, -order_position), index))

    if not candidates:
        return None

    candidates.sort(reverse=True)
    return candidates[0][1]


def _find_header_value(hdulist, keywords, search_order, check_duplicates=True):
    """Find the first usable keyword, preferring its last duplicate card."""

    for index in search_order:
        header = hdulist[index].header
        for keyword in keywords or []:
            if keyword not in header:
                continue

            values = []
            for card in header.cards:
                if card.keyword.upper() != str(keyword).upper():
                    continue
                value = card.value
                if value is None or np.ma.is_masked(value):
                    continue
                if isinstance(value, str) and not value.strip():
                    continue
                if isinstance(value, (float, np.floating)) and not np.isfinite(value):
                    continue
                values.append(value)

            if not values:
                continue

            source = {"hdu": index, "keyword": str(keyword)}
            if len(values) > 1:
                source["duplicate_values"] = list(values)
                source["duplicate_count"] = len(values)
                conflict = len({repr(value) for value in values}) > 1
                source["duplicate_conflict"] = conflict
                if check_duplicates and conflict:
                    warnings.warn(
                        "Conflicting duplicate {} cards in HDU {}; using the last "
                        "value {!r}.".format(keyword, index, values[-1]),
                        RuntimeWarning,
                    )

            return values[-1], source

    return None, None


def _as_float(value):
    """Convert a scalar header value to float, returning ``None`` if invalid."""

    if value is None or np.ma.is_masked(value):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _as_clean_string(value, strip=True):
    """Convert a header value to a clean string or ``None``."""

    if value is None or np.ma.is_masked(value):
        return None
    result = str(value)
    if strip:
        result = result.strip()
    return result or None


def _parse_binning(value):
    """Parse common FITS binning representations into integer x and y values."""

    if value is None:
        return None, None
    if isinstance(value, (tuple, list, np.ndarray)) and len(value) >= 2:
        parts = value[:2]
    else:
        text = str(value).lower().replace("x", " ").replace(",", " ")
        parts = text.split()
    try:
        if len(parts) == 1:
            number = int(float(parts[0]))
            return number, number
        return int(float(parts[0])), int(float(parts[1]))
    except (TypeError, ValueError, IndexError):
        return None, None


def _normalize_time(metadata, sources, metadata_settings):
    """Normalize JD/MJD or DATE-OBS to the configured MJD reference."""

    original = _as_float(metadata.get("mjd"))
    source = sources.get("mjd") or {}
    keyword = str(source.get("keyword", "")).upper()

    if original is not None and (keyword == "JD" or original > 2000000.0):
        original -= 2400000.5

    if original is None and metadata.get("date_obs") is not None:
        try:
            scale = metadata_settings.get("canonical_time_scale", "utc")
            original = float(Time(metadata["date_obs"], scale=scale).mjd)
            sources["mjd"] = dict(sources.get("date_obs") or {})
            sources["mjd"]["derived_from"] = "date_obs"
        except (TypeError, ValueError):
            original = None

    metadata["mjd_header"] = original
    reference = str(metadata_settings.get("time_reference", "start")).lower()
    canonical = original
    exposure = metadata.get("exposure_time")

    if (
        canonical is not None
        and exposure is not None
        and metadata_settings.get("convert_time_to_mid_exposure", True)
    ):
        half_exposure_days = float(exposure) / 172800.0
        if reference == "start":
            canonical += half_exposure_days
            reference = "mid"
        elif reference == "end":
            canonical -= half_exposure_days
            reference = "mid"

    metadata["mjd"] = canonical
    metadata["time_reference"] = reference if canonical is not None else None


def extract_metadata(hdulist, settings, data_index, wcs_index=None):
    """Extract normalized metadata by searching the configured header sequence.

    Each keyword is searched independently, so values may come from different
    HDUs.  Explicit metadata overrides take precedence over header values, then
    instrument fallback values are used.  Missing values remain ``None``.

    Returns
    -------
    dict
        Normalized scalar metadata.  The private ``_sources`` mapping records
        the HDU and FITS keyword used for each extracted field.
    """

    input_settings = settings.get("input", {})
    metadata_settings = settings.get("metadata", {})
    explicit_header = _resolve_hdu_index(
        hdulist, input_settings.get("header_hdu"), "header"
    )
    search_order = _hdu_search_order(
        hdulist,
        preferred=explicit_header,
        configured=input_settings.get("hdu_search_order", [0, 1, 2]),
        include_remaining=input_settings.get("search_remaining_hdus", True),
    )

    keywords = metadata_settings.get("keywords", {})
    fallback_values = metadata_settings.get("fallback_values", {})
    metadata = {}
    sources = {}

    for field, field_keywords in keywords.items():
        value, source = _find_header_value(
            hdulist,
            field_keywords,
            search_order,
            check_duplicates=metadata_settings.get("check_duplicate_cards", True),
        )
        override_key = "{}_override".format(field)
        override = metadata_settings.get(override_key)
        if override is not None:
            value = override
            source = {"override": override_key}
        elif value is None and field in fallback_values:
            value = fallback_values.get(field)
            if value is not None:
                source = {"fallback": "instrument"}
        metadata[field] = value
        if source is not None:
            sources[field] = source

    string_fields = {
        "object",
        "telescope",
        "instrument",
        "detector",
        "site",
        "date_obs",
        "date_end",
        "filter",
        "pointing_ra",
        "pointing_dec",
        "target_ra",
        "target_dec",
        "binning",
        "reduction_level",
        "pipeline_version",
    }
    numeric_fields = {
        "exposure_time",
        "mjd",
        "gain",
        "read_noise",
        "saturation",
        "airmass",
        "pixel_scale",
    }
    strip_strings = metadata_settings.get("strip_string_values", True)

    for field in string_fields:
        if field in metadata:
            metadata[field] = _as_clean_string(metadata[field], strip_strings)
    for field in numeric_fields:
        if field in metadata:
            metadata[field] = _as_float(metadata[field])

    if metadata_settings.get("normalize_filter", True):
        metadata["filter"] = normalize_filter_name(metadata.get("filter"))

    binning_x, binning_y = _parse_binning(metadata.get("binning"))
    metadata["binning_x"] = binning_x
    metadata["binning_y"] = binning_y
    _normalize_time(metadata, sources, metadata_settings)

    metadata["header_hdu"] = search_order[0] if search_order else None
    metadata["wcs_hdu"] = wcs_index
    metadata["_sources"] = sources
    return metadata


def _data_unit(header):
    """Return a valid Astropy unit from BUNIT, defaulting to ADU."""

    value = header.get("BUNIT")
    if value is None or not str(value).strip():
        return u.adu
    try:
        return u.Unit(str(value).strip())
    except (TypeError, ValueError):
        warnings.warn(
            "Unrecognized FITS BUNIT {!r}; using ADU.".format(value),
            RuntimeWarning,
        )
        return u.adu


def _read_uncertainty(hdulist, index, unit):
    """Read an uncertainty extension and return it with its invalid-pixel mask."""

    if index is None:
        return None, None, None

    array = np.array(_numeric_2d_array(hdulist[index]), dtype=float, copy=True)
    extname = _hdu_extname(hdulist[index])

    if extname in ERROR_EXTNAMES:
        invalid = ~np.isfinite(array) | (array < 0)
        uncertainty = StdDevUncertainty(array, unit=unit)
        kind = "standard_deviation"
    elif extname in INVERSE_VARIANCE_EXTNAMES:
        invalid = ~np.isfinite(array) | (array <= 0)
        uncertainty = InverseVariance(array, unit=unit ** -2)
        kind = "inverse_variance"
    else:
        invalid = ~np.isfinite(array) | (array < 0)
        uncertainty = VarianceUncertainty(array, unit=unit ** 2)
        kind = "variance"

    return uncertainty, invalid, kind


def read_fits_image(path, settings=None, target=None):
    """Read one reduced FITS image into ``CCDData`` without modifying the file.

    Parameters
    ----------
    path : str or pathlib.Path
        A supported ordinary, gzip-compressed, or tile-compressed FITS file.
    settings : mapping, optional
        Fully resolved settings for this image.  General defaults are used when
        omitted.
    target : astropy.coordinates.SkyCoord, mapping, or pair, optional
        Target coordinate used to choose among multiple science arrays.  Numeric
        pairs are interpreted as decimal degrees; string pairs use the units in
        ``target_position.coordinate_unit``.

    Returns
    -------
    ccd : astropy.nddata.CCDData
        Independent in-memory science data with WCS, mask, and uncertainty when
        those components are available.
    metadata : dict
        Normalized metadata with ``None`` for missing values.
    """

    if settings is None:
        settings = get_default_settings()

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError("FITS input does not exist: {}".format(path))
    if not is_fits_path(path):
        raise ValueError("Unsupported FITS filename: {}".format(path))

    input_settings = settings.get("input", {})
    if _is_compressed_fits(path) and not input_settings.get(
        "allow_compressed", True
    ):
        raise ValueError(
            "Compressed FITS input is disabled in the settings: {}".format(path)
        )
    open_options = {
        "mode": "readonly",
        "memmap": bool(input_settings.get("memmap", False)),
        "checksum": bool(input_settings.get("verify_checksum", True)),
    }

    with fits.open(path, **open_options) as hdulist:
        data_index = select_science_hdu(hdulist, settings, target=target)
        header_index = _resolve_hdu_index(
            hdulist, input_settings.get("header_hdu"), "header"
        )
        search_order = _hdu_search_order(
            hdulist,
            preferred=data_index,
            configured=input_settings.get("hdu_search_order", [0, 1, 2]),
            include_remaining=input_settings.get("search_remaining_hdus", True),
        )
        wcs, wcs_index = _wcs_for_data_hdu(
            hdulist, data_index, header_index, search_order
        )

        mask_index = _matching_auxiliary_hdu(
            hdulist,
            data_index,
            input_settings.get("mask_hdu"),
            MASK_EXTNAMES,
            "mask",
            search_order,
        )
        variance_index = _matching_auxiliary_hdu(
            hdulist,
            data_index,
            input_settings.get("variance_hdu"),
            VARIANCE_EXTNAMES + ERROR_EXTNAMES + INVERSE_VARIANCE_EXTNAMES,
            "variance",
            search_order,
        )

        array = np.array(_numeric_2d_array(hdulist[data_index]), copy=True)
        original_dtype = str(array.dtype)
        header = _combined_header(hdulist, data_index)
        unit = _data_unit(header)

        mask = None
        if mask_index is not None:
            mask_data = np.array(
                _numeric_2d_array(hdulist[mask_index]), copy=True
            )
            mask = ~np.isfinite(mask_data) | (mask_data != 0)

        if settings.get("masks", {}).get("mask_nonfinite", True):
            nonfinite = ~np.isfinite(array)
            mask = nonfinite if mask is None else (mask | nonfinite)

        uncertainty, invalid_uncertainty, uncertainty_type = _read_uncertainty(
            hdulist, variance_index, unit
        )
        if invalid_uncertainty is not None:
            mask = (
                invalid_uncertainty
                if mask is None
                else (mask | invalid_uncertainty)
            )

        metadata = extract_metadata(
            hdulist, settings, data_index, wcs_index=wcs_index
        )
        metadata.update(
            {
                "path": str(path),
                "filename": path.name,
                "data_hdu": data_index,
                "data_extname": _hdu_extname(hdulist[data_index]) or None,
                "mask_hdu": mask_index,
                "variance_hdu": variance_index,
                "uncertainty_type": uncertainty_type,
                "shape_y": int(array.shape[0]),
                "shape_x": int(array.shape[1]),
                "dtype": original_dtype,
                "finite_fraction": _finite_fraction(array),
            }
        )
        metadata = normalize_and_validate_metadata(
            hdulist,
            metadata,
            settings,
            wcs=wcs,
            shape=array.shape,
            target=target,
        )

    ccd = CCDData(
        array,
        unit=unit,
        meta=header,
        wcs=wcs,
        mask=mask,
        uncertainty=uncertainty,
    )
    return ccd, metadata


# ---------------------------------------------------------------------------
# Step 5: valid processing region and cropping
#
# These functions define the usable science region of an image and produce a
# cropped working view.  They never modify the input ``CCDData``; each returns
# new arrays and a fresh ``CCDData`` so any correction can be inspected and
# reversed.  Header section keywords and empirical edge trimming build a
# valid-pixel mask; the angular crop then limits later processing to the region
# of interest while keeping the WCS, mask, and uncertainty consistent.
# ---------------------------------------------------------------------------


def parse_fits_section(value, shape=None):
    """Parse a FITS image section string into 0-based, half-open array bounds.

    Parameters
    ----------
    value : str
        A FITS section such as ``'[1:1024,1:4096]'``.  The two ranges are the
        one-based, inclusive column (``x``) and row (``y``) limits.  Reversed
        ranges (image flips) are accepted and normalized.
    shape : tuple of int, optional
        ``(ny, nx)`` array shape.  When given, the bounds are clipped to the
        array.

    Returns
    -------
    tuple of int or None
        ``(x_start, x_stop, y_start, y_stop)`` in numpy convention, or ``None``
        when the value is missing or cannot be parsed.
    """

    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.lstrip("[").rstrip("]")
    parts = text.split(",")
    if len(parts) != 2:
        return None

    bounds = []
    for part in parts:
        limits = part.strip().split(":")
        try:
            if len(limits) == 1:
                low = high = int(float(limits[0]))
            else:
                low = int(float(limits[0]))
                high = int(float(limits[1]))
        except (TypeError, ValueError):
            return None
        bounds.append((min(low, high), max(low, high)))

    (x_low, x_high), (y_low, y_high) = bounds
    x_start, x_stop = x_low - 1, x_high
    y_start, y_stop = y_low - 1, y_high

    if shape is not None:
        ny, nx = shape
        x_start = max(0, min(x_start, nx))
        x_stop = max(0, min(x_stop, nx))
        y_start = max(0, min(y_start, ny))
        y_stop = max(0, min(y_stop, ny))

    if x_stop <= x_start or y_stop <= y_start:
        return None
    return x_start, x_stop, y_start, y_stop


def header_section_region(header, shape, settings):
    """Build a valid-pixel mask from a header science section when it applies.

    The configured ``TRIMSEC``/``DATASEC`` (and optionally ``CCDSEC``/``DETSEC``)
    keywords are tried in order.  A section is only used when it references a
    strict sub-region of the current array.  If the section extends beyond the
    array or matches its size, the frame is treated as already trimmed and no
    mask is produced.

    Returns
    -------
    valid : numpy.ndarray or None
        Boolean ``True``-inside mask, or ``None`` when no section applies.
    info : dict
        Keyword, bounds, and whether the section was applied.
    """

    crop_settings = settings.get("crop", {})
    section_keywords = crop_settings.get("section_keywords", {})

    order = []
    if crop_settings.get("use_trimsec", True):
        order.extend(section_keywords.get("trimsec", ["TRIMSEC"]))
    if crop_settings.get("use_datasec", True):
        order.extend(section_keywords.get("datasec", ["DATASEC"]))
    if crop_settings.get("use_ccdsec", False):
        order.extend(section_keywords.get("ccdsec", ["CCDSEC"]))
    if crop_settings.get("use_detsec", False):
        order.extend(section_keywords.get("detsec", ["DETSEC"]))

    ny, nx = shape
    for keyword in order:
        bounds = parse_fits_section(header.get(keyword))
        if bounds is None:
            continue
        x_start, x_stop, y_start, y_stop = bounds
        section_width = x_stop - x_start
        section_height = y_stop - y_start
        fits_array = (
            x_start >= 0
            and y_start >= 0
            and x_stop <= nx
            and y_stop <= ny
        )
        same_size = section_width == nx and section_height == ny
        if not fits_array or same_size:
            reason = "already_trimmed" if same_size else "outside_array"
            return None, {
                "keyword": keyword,
                "bounds": bounds,
                "applied": False,
                "reason": reason,
            }
        valid = np.zeros(shape, dtype=bool)
        valid[y_start:y_stop, x_start:x_stop] = True
        return valid, {"keyword": keyword, "bounds": bounds, "applied": True}

    return None, {"keyword": None, "bounds": None, "applied": False}


def detect_empirical_edges(data, base_valid, settings):
    """Trim contiguous unusable rows and columns inward from each border.

    Only border rows/columns are examined, so interior sources are never
    removed.  A line is considered bad when it is largely non-finite, dominated
    by a single constant value (such as zeros on a blank edge), or has a median
    that departs strongly from the robust global median (extreme border glow or
    unexposed regions).

    Returns
    -------
    edge_invalid : numpy.ndarray
        Boolean mask that is ``True`` on trimmed border pixels.
    info : dict
        Number of rows/columns trimmed on each side and the grow width.
    """

    crop_settings = settings.get("crop", {})
    ny, nx = data.shape
    edge_invalid = np.zeros(data.shape, dtype=bool)
    info = {"top": 0, "bottom": 0, "left": 0, "right": 0, "grow_pixels": 0}

    if not crop_settings.get("detect_empirical_edges", True):
        return edge_invalid, info

    finite = np.isfinite(data)
    usable = finite & np.asarray(base_valid, dtype=bool)
    sample = data[usable]
    if sample.size < 100:
        return edge_invalid, info

    median = float(np.median(sample))
    mad = float(np.median(np.abs(sample - median)))
    robust_std = mad * 1.4826 if mad > 0 else float(np.std(sample))

    sigma = float(crop_settings.get("edge_sigma", 5.0))
    min_finite = float(crop_settings.get("edge_min_finite_fraction", 0.5))
    max_constant = float(crop_settings.get("edge_max_constant_fraction", 0.5))
    scan_fraction = float(crop_settings.get("edge_scan_fraction", 0.15))

    def line_is_bad(values):
        line_finite = np.isfinite(values)
        if line_finite.mean() < min_finite:
            return True
        good = values[line_finite]
        if good.size == 0:
            return True
        zero_fraction = float(np.mean(good == 0))
        constant_fraction = max(
            zero_fraction, float(np.mean(good == np.median(good)))
        )
        if constant_fraction > max_constant:
            return True
        if robust_std > 0 and abs(float(np.median(good)) - median) > sigma * robust_std:
            return True
        return False

    def count_bad(lines, limit):
        count = 0
        for step in range(limit):
            if line_is_bad(lines(step)):
                count = step + 1
            else:
                break
        return count

    max_rows = max(1, int(scan_fraction * ny))
    max_cols = max(1, int(scan_fraction * nx))

    info["top"] = count_bad(lambda step: data[step, :], max_rows)
    info["bottom"] = count_bad(lambda step: data[ny - 1 - step, :], max_rows)
    info["left"] = count_bad(lambda step: data[:, step], max_cols)
    info["right"] = count_bad(lambda step: data[:, nx - 1 - step], max_cols)

    grow = int(crop_settings.get("edge_grow_pixels", 2))
    info["grow_pixels"] = grow

    top = min(info["top"] + grow, ny) if info["top"] else 0
    bottom = min(info["bottom"] + grow, ny) if info["bottom"] else 0
    left = min(info["left"] + grow, nx) if info["left"] else 0
    right = min(info["right"] + grow, nx) if info["right"] else 0

    if top:
        edge_invalid[:top, :] = True
    if bottom:
        edge_invalid[ny - bottom:, :] = True
    if left:
        edge_invalid[:, :left] = True
    if right:
        edge_invalid[:, nx - right:] = True

    return edge_invalid, info


def _valid_section_bounds(value, shape):
    """Resolve a ``valid_section`` override into 0-based half-open bounds."""

    if value is None:
        return None
    if isinstance(value, str):
        return parse_fits_section(value, shape=shape)
    try:
        x1, x2, y1, y2 = value
    except (TypeError, ValueError):
        return None
    section = "[{}:{},{}:{}]".format(int(x1), int(x2), int(y1), int(y2))
    return parse_fits_section(section, shape=shape)


def build_valid_region(ccd, metadata=None, settings=None):
    """Build a full-frame valid-pixel mask (``True`` = usable).

    Combines existing non-finite/mask pixels, the header science section,
    empirical edge trimming, a uniform ``edge_crop_pixels`` border, and an
    explicit ``valid_section`` override.  When ``valid_section`` is supplied it
    is authoritative and the automatic section/edge steps are skipped.

    Returns
    -------
    valid : numpy.ndarray
        Boolean valid-pixel mask.
    info : dict
        Provenance for each contributing step and the overall valid fraction.
    """

    if settings is None:
        settings = get_default_settings()
    crop_settings = settings.get("crop", {})

    data = np.asarray(ccd.data)
    shape = data.shape
    ny, nx = shape

    valid = np.isfinite(data)
    existing_mask = getattr(ccd, "mask", None)
    if existing_mask is not None:
        valid &= ~np.asarray(existing_mask, dtype=bool)

    info = {}
    explicit = _valid_section_bounds(crop_settings.get("valid_section"), shape)
    if explicit is not None:
        x_start, x_stop, y_start, y_stop = explicit
        section_valid = np.zeros(shape, dtype=bool)
        section_valid[y_start:y_stop, x_start:x_stop] = True
        valid &= section_valid
        info["valid_section"] = {"bounds": explicit, "applied": True}
        info["header_section"] = {"keyword": None, "bounds": None, "applied": False}
        info["empirical_edges"] = {
            "top": 0,
            "bottom": 0,
            "left": 0,
            "right": 0,
            "grow_pixels": 0,
        }
    else:
        info["valid_section"] = {"bounds": None, "applied": False}
        header = ccd.meta if getattr(ccd, "meta", None) is not None else {}
        section_valid, section_info = header_section_region(
            header, shape, settings
        )
        info["header_section"] = section_info
        if section_valid is not None:
            valid &= section_valid
        edge_invalid, edge_info = detect_empirical_edges(data, valid, settings)
        info["empirical_edges"] = edge_info
        valid &= ~edge_invalid

    border = int(crop_settings.get("edge_crop_pixels", 0) or 0)
    if border > 0:
        uniform = np.zeros(shape, dtype=bool)
        uniform[border:ny - border, border:nx - border] = True
        valid &= uniform
    info["edge_crop_pixels"] = border
    info["valid_fraction"] = float(valid.mean()) if valid.size else 0.0

    return valid, info


def _pixel_scale_arcsec(ccd, metadata, settings):
    """Return the image pixel scale in arcsec/pixel from WCS or metadata."""

    wcs = getattr(ccd, "wcs", None)
    if wcs is not None and wcs.has_celestial:
        try:
            scales = proj_plane_pixel_scales(wcs.celestial)
            arcsec = float(np.mean(scales)) * 3600.0
            if np.isfinite(arcsec) and arcsec > 0:
                return arcsec
        except Exception:
            pass
    scale = _as_float(metadata.get("pixel_scale")) if metadata else None
    if scale is not None and scale > 0:
        return scale
    return None


def _fwhm_guess_pixels(settings):
    """Return the configured FWHM guess in pixels used for edge buffers."""

    return float(
        settings.get("source_detection", {}).get("fwhm_guess_pixels", 4.0)
    )


def _crop_center_pixel(ccd, metadata, settings, target):
    """Return the crop-center pixel ``(x, y)`` and its sky coordinate."""

    crop_settings = settings.get("crop", {})
    data = np.asarray(ccd.data)
    ny, nx = data.shape
    field_center = ((nx - 1) / 2.0, (ny - 1) / 2.0)
    wcs = getattr(ccd, "wcs", None)

    if crop_settings.get("center_on", "target") == "field":
        return field_center, None

    center = _coerce_target_coordinate(target, settings)
    if center is None and metadata is not None:
        ra = metadata.get("adopted_ra_deg")
        dec = metadata.get("adopted_dec_deg")
        if ra is not None and dec is not None:
            center = SkyCoord(float(ra), float(dec), unit="deg", frame="icrs")

    if center is not None and wcs is not None and wcs.has_celestial:
        try:
            x, y = wcs.world_to_pixel(center)
            if np.isfinite(x) and np.isfinite(y):
                return (float(x), float(y)), center
        except Exception:
            pass
    return field_center, center


def crop_to_processing_region(
    ccd, metadata=None, settings=None, target=None, valid=None
):
    """Crop a science image to the configured angular processing footprint.

    The crop is centered on the target (or the field center when
    ``crop.center_on`` is ``'field'``) and sized by ``crop.size_arcmin``.  Data,
    WCS, mask, uncertainty, and an optional valid-pixel mask are cropped
    consistently.  When cropping is disabled or no size or pixel scale is
    available, the input is returned unchanged.  The input image is not
    modified.

    Returns
    -------
    cropped_ccd : astropy.nddata.CCDData
        Cropped working image with an updated WCS (or the input when no crop is
        applied).
    cropped_valid : numpy.ndarray or None
        The ``valid`` mask cropped to the new footprint, when supplied.
    info : dict
        Crop center, pixel scale, resulting shape, and array slices.
    """

    if settings is None:
        settings = get_default_settings()
    crop_settings = settings.get("crop", {})
    data = np.asarray(ccd.data)
    ny, nx = data.shape
    wcs = getattr(ccd, "wcs", None)

    (center_x, center_y), _ = _crop_center_pixel(ccd, metadata, settings, target)
    size_arcmin = crop_settings.get("size_arcmin")
    info = {
        "requested_arcmin": size_arcmin,
        "center_x": center_x,
        "center_y": center_y,
        "applied": False,
    }

    if not crop_settings.get("enabled", True):
        info["reason"] = "disabled"
        return ccd, valid, info
    if size_arcmin is None:
        info["reason"] = "no_size"
        return ccd, valid, info

    pixel_scale = _pixel_scale_arcsec(ccd, metadata, settings)
    if pixel_scale is None:
        warnings.warn(
            "Cannot crop without a pixel scale; keeping the full frame.",
            RuntimeWarning,
        )
        info["reason"] = "no_pixel_scale"
        return ccd, valid, info

    size_pixels = float(size_arcmin) * 60.0 / pixel_scale
    side = max(1, int(round(size_pixels)))
    size = (min(side, ny), min(side, nx))
    mode = crop_settings.get("crop_mode", "trim")
    position = (center_x, center_y)

    try:
        data_cut = Cutout2D(
            data, position, size, wcs=wcs, mode=mode, fill_value=np.nan, copy=True
        )
    except Exception as error:
        warnings.warn(
            "Cropping failed ({}); keeping the full frame.".format(error),
            RuntimeWarning,
        )
        info["reason"] = "cutout_failed"
        return ccd, valid, info

    cropped_data = np.asarray(data_cut.data)

    cropped_mask = None
    if getattr(ccd, "mask", None) is not None:
        mask_cut = Cutout2D(
            np.asarray(ccd.mask, dtype=np.uint8),
            position,
            size,
            mode=mode,
            fill_value=1,
            copy=True,
        )
        cropped_mask = mask_cut.data.astype(bool)

    cropped_uncertainty = None
    if getattr(ccd, "uncertainty", None) is not None:
        uncertainty = ccd.uncertainty
        uncertainty_cut = Cutout2D(
            np.asarray(uncertainty.array, dtype=float),
            position,
            size,
            mode=mode,
            fill_value=np.nan,
            copy=True,
        )
        cropped_uncertainty = uncertainty.__class__(
            uncertainty_cut.data, unit=uncertainty.unit
        )

    cropped_valid = None
    if valid is not None:
        valid_cut = Cutout2D(
            np.asarray(valid, dtype=np.uint8),
            position,
            size,
            mode=mode,
            fill_value=0,
            copy=True,
        )
        cropped_valid = valid_cut.data.astype(bool)

    header = ccd.meta.copy() if getattr(ccd, "meta", None) is not None else None
    cropped_ccd = CCDData(
        cropped_data,
        unit=ccd.unit,
        meta=header,
        wcs=data_cut.wcs,
        mask=cropped_mask,
        uncertainty=cropped_uncertainty,
    )

    slices = data_cut.slices_original
    info.update(
        {
            "applied": True,
            "pixel_scale_arcsec": pixel_scale,
            "size_pixels": size,
            "shape": cropped_data.shape,
            "slices": (
                (slices[0].start, slices[0].stop),
                (slices[1].start, slices[1].stop),
            ),
        }
    )
    return cropped_ccd, cropped_valid, info


def _check_target_region(ccd, metadata, settings, target, valid=None):
    """Check whether the target lies safely inside the usable region."""

    crop_settings = settings.get("crop", {})
    data = np.asarray(ccd.data)
    ny, nx = data.shape
    wcs = getattr(ccd, "wcs", None)
    flags = []
    info = {
        "target_x": None,
        "target_y": None,
        "target_inside": None,
        "target_edge_distance_pixels": None,
        "flags": flags,
    }

    center = _coerce_target_coordinate(target, settings)
    if center is None and metadata is not None:
        ra = metadata.get("adopted_ra_deg")
        dec = metadata.get("adopted_dec_deg")
        if ra is not None and dec is not None:
            center = SkyCoord(float(ra), float(dec), unit="deg", frame="icrs")

    if center is None or wcs is None or not wcs.has_celestial:
        return info

    try:
        x, y = wcs.world_to_pixel(center)
        x = float(x)
        y = float(y)
    except Exception:
        return info
    if not (np.isfinite(x) and np.isfinite(y)):
        return info

    info["target_x"] = x
    info["target_y"] = y
    inside = bool(-0.5 <= x < nx - 0.5 and -0.5 <= y < ny - 0.5)
    info["target_inside"] = inside
    if not inside:
        if crop_settings.get("require_target_inside", True):
            flags.append("TARGET_OUTSIDE_IMAGE")
        return info

    edge_distance = float(min(x + 0.5, y + 0.5, nx - 0.5 - x, ny - 0.5 - y))
    info["target_edge_distance_pixels"] = edge_distance
    minimum = crop_settings.get("target_edge_distance_fwhm", 5.0) * _fwhm_guess_pixels(
        settings
    )
    near_edge = edge_distance < minimum

    if valid is not None:
        xi = int(round(x))
        yi = int(round(y))
        valid_array = np.asarray(valid)
        if 0 <= yi < valid_array.shape[0] and 0 <= xi < valid_array.shape[1]:
            if not bool(valid_array[yi, xi]):
                near_edge = True

    if near_edge:
        flags.append("TARGET_NEAR_EDGE")
    return info


def define_processing_region(ccd, metadata=None, settings=None, target=None):
    """Define the valid processing region and return a cropped working image.

    This builds the full-frame valid-pixel mask (header science section plus
    empirical edge trimming and any overrides), folds the invalid pixels into a
    copy's mask, crops to the configured angular footprint, and checks that the
    target lies safely inside the result.  The input ``ccd`` is never modified.

    Parameters
    ----------
    ccd : astropy.nddata.CCDData
        Science image from :func:`read_fits_image`.
    metadata : mapping, optional
        Normalized metadata; the adopted target position is used when no
        explicit ``target`` is given.
    settings : mapping, optional
        Fully resolved settings for this image.
    target : astropy.coordinates.SkyCoord, mapping, or pair, optional
        Target coordinate used for centering and the target-region check.

    Returns
    -------
    working : astropy.nddata.CCDData
        Cropped, masked working image with an updated WCS.
    region : dict
        Section, trimmed-edge, crop, target-region, and quality-flag records.
    diagnostics : dict
        Full-frame valid mask and crop geometry for later plotting.
    """

    if settings is None:
        settings = get_default_settings()

    data = np.asarray(ccd.data)
    valid, valid_info = build_valid_region(ccd, metadata, settings)

    invalid = ~valid
    combined_mask = invalid
    if getattr(ccd, "mask", None) is not None:
        combined_mask = np.asarray(ccd.mask, dtype=bool) | invalid

    masked_ccd = CCDData(
        np.array(data, copy=True),
        unit=ccd.unit,
        meta=ccd.meta.copy() if getattr(ccd, "meta", None) is not None else None,
        wcs=getattr(ccd, "wcs", None),
        mask=combined_mask,
        uncertainty=getattr(ccd, "uncertainty", None),
    )

    working, cropped_valid, crop_info = crop_to_processing_region(
        masked_ccd, metadata, settings, target=target, valid=valid
    )

    flags = []
    edges = valid_info.get("empirical_edges", {})
    if any(edges.get(side) for side in ("top", "bottom", "left", "right")):
        flags.append("BAD_EDGES")

    target_info = _check_target_region(
        working, metadata, settings, target, cropped_valid
    )
    for flag in target_info.pop("flags", []):
        if flag not in flags:
            flags.append(flag)

    region = {
        "valid_fraction_full": valid_info.get("valid_fraction"),
        "valid_section": valid_info.get("valid_section"),
        "header_section": valid_info.get("header_section"),
        "empirical_edges": valid_info.get("empirical_edges"),
        "edge_crop_pixels": valid_info.get("edge_crop_pixels"),
        "crop": crop_info,
        "region_flags": flags,
    }
    region.update(target_info)

    diagnostics = {
        "full_valid_mask": valid,
        "full_shape": data.shape,
        "crop_slices": crop_info.get("slices"),
        "crop_center": (crop_info.get("center_x"), crop_info.get("center_y")),
    }
    return working, region, diagnostics


# ---------------------------------------------------------------------------
# Step 6: pixel and artifact masks
#
# Each defect type produces its own boolean component so it can be plotted and
# reasoned about independently; ``build_masks`` combines them into one working
# mask.  Morphological growth and connected-component labelling use
# ``scipy.ndimage``, which is imported lazily so this module still imports when
# scipy is absent (those growth steps are then skipped with a warning).  The
# input image is never modified.
# ---------------------------------------------------------------------------


_NDIMAGE_WARNED = False


def _try_ndimage():
    """Return ``scipy.ndimage`` or ``None``, warning once when unavailable."""

    global _NDIMAGE_WARNED
    try:
        from scipy import ndimage
    except ImportError:
        if not _NDIMAGE_WARNED:
            warnings.warn(
                "scipy is not available; morphological mask steps (saturation "
                "halo growth and trail detection) are skipped.",
                RuntimeWarning,
            )
            _NDIMAGE_WARNED = True
        return None
    return ndimage


def _disk_structure(radius):
    """Return a circular boolean structuring element of the given radius."""

    radius = int(radius)
    if radius < 1:
        return None
    grid_y, grid_x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return (grid_x ** 2 + grid_y ** 2) <= radius ** 2


def make_saturation_mask(data, metadata=None, settings=None):
    """Mask saturated and nonlinear pixels and grow bleed and halo regions.

    The saturation and nonlinearity levels come from the settings overrides or
    the image metadata.  When both are present the lower (more conservative)
    level defines the bright cores used for halo growth, so wings, bleed
    columns, and halos around bright stars are excluded from later selection.

    Returns
    -------
    components : dict of numpy.ndarray
        ``saturated``, ``nonlinear``, and the grown ``saturation`` mask.
    info : dict
        Levels used and the saturated-pixel fraction.
    """

    if settings is None:
        settings = get_default_settings()
    masks_settings = settings.get("masks", {})
    shape = np.asarray(data).shape

    saturated = np.zeros(shape, dtype=bool)
    nonlinear = np.zeros(shape, dtype=bool)

    sat_level = masks_settings.get("saturation_level")
    if sat_level is None and metadata is not None:
        sat_level = _as_float(metadata.get("saturation"))
    nonlinear_level = masks_settings.get("nonlinearity_level")
    if nonlinear_level is None and metadata is not None:
        nonlinear_level = _as_float(metadata.get("nonlinearity"))

    finite = np.isfinite(data)
    if masks_settings.get("mask_saturated", True) and sat_level is not None:
        saturated = finite & (data >= float(sat_level))
    if masks_settings.get("mask_nonlinear", True) and nonlinear_level is not None:
        nonlinear = finite & (data >= float(nonlinear_level))

    levels = [level for level in (sat_level, nonlinear_level) if level is not None]
    effective = None
    if levels:
        if masks_settings.get("prefer_nonlinearity_limit", True):
            effective = min(levels)
        else:
            effective = sat_level if sat_level is not None else nonlinear_level
    core = finite & (data >= float(effective)) if effective is not None else (
        np.zeros(shape, dtype=bool)
    )

    saturation_mask = core.copy()
    grow = int(masks_settings.get("saturation_grow_pixels", 5))
    halo_radius = int(
        round(
            masks_settings.get("saturation_halo_fwhm", 5.0)
            * _fwhm_guess_pixels(settings)
        )
    )
    if core.any() and (grow > 0 or halo_radius > 0):
        ndimage = _try_ndimage()
        if ndimage is not None:
            if grow > 0:
                saturation_mask |= ndimage.binary_dilation(core, iterations=grow)
            disk = _disk_structure(halo_radius)
            if disk is not None:
                saturation_mask |= ndimage.binary_dilation(core, structure=disk)

    info = {
        "saturation_level": None if sat_level is None else float(sat_level),
        "nonlinearity_level": (
            None if nonlinear_level is None else float(nonlinear_level)
        ),
        "effective_level": None if effective is None else float(effective),
        "saturated_fraction": float(saturated.mean()) if saturated.size else 0.0,
    }
    components = {
        "saturated": saturated,
        "nonlinear": nonlinear,
        "saturation": saturation_mask,
    }
    return components, info


def _line_bad_indices(profile, sigma):
    """Return indices of a 1D profile that deviate strongly from the median."""

    finite = np.isfinite(profile)
    if int(finite.sum()) < 10:
        return np.array([], dtype=int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(profile)
        scale = np.nanmedian(np.abs(profile - median)) * 1.4826
    if not np.isfinite(scale) or scale <= 0:
        return np.array([], dtype=int)
    deviation = np.abs(profile - median)
    return np.where(finite & (deviation > sigma * scale))[0]


def make_line_defect_mask(data, valid=None, settings=None):
    """Mask hot or dead rows and columns using robust profile statistics.

    Row and column medians are computed over the usable pixels; a line whose
    median departs from the robust global level by more than ``bad_line_sigma``
    times the median absolute deviation is masked.  A high default sigma keeps
    ordinary stars and galaxies from being mistaken for defects.

    Returns
    -------
    mask : numpy.ndarray
        Boolean mask of bad rows and columns.
    info : dict
        Lists of the flagged row and column indices.
    """

    if settings is None:
        settings = get_default_settings()
    masks_settings = settings.get("masks", {})
    shape = np.asarray(data).shape
    ny, nx = shape
    mask = np.zeros(shape, dtype=bool)
    info = {"bad_rows": [], "bad_columns": []}

    detect_rows = masks_settings.get("detect_bad_rows", True)
    detect_columns = masks_settings.get("detect_bad_columns", True)
    if not (detect_rows or detect_columns):
        return mask, info

    work = np.array(data, dtype=float)
    if valid is not None:
        work[~np.asarray(valid, dtype=bool)] = np.nan
    work[~np.isfinite(work)] = np.nan

    sigma = float(masks_settings.get("bad_line_sigma", 6.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if detect_columns:
            columns = _line_bad_indices(np.nanmedian(work, axis=0), sigma)
            info["bad_columns"] = columns.tolist()
            for column in columns:
                mask[:, column] = True
        if detect_rows:
            rows = _line_bad_indices(np.nanmedian(work, axis=1), sigma)
            info["bad_rows"] = rows.tolist()
            for row in rows:
                mask[row, :] = True

    grow = int(masks_settings.get("bad_line_grow_pixels", 0))
    if grow > 0 and mask.any():
        ndimage = _try_ndimage()
        if ndimage is not None:
            mask = ndimage.binary_dilation(mask, iterations=grow)
    return mask, info


def make_amplifier_seam_mask(data, valid=None, settings=None):
    """Mask amplifier seams from explicit boundaries and abrupt median jumps.

    Explicit ``masks.amplifier_boundaries`` entries are always applied.  When
    ``detect_amplifier_boundaries`` is enabled, columns or rows with an extreme
    step between adjacent medians are also masked.  A high default sigma keeps
    the detector's smooth structure from being flagged.

    Returns
    -------
    mask : numpy.ndarray
        Boolean seam mask.
    info : dict
        Seam column and row indices that were masked.
    """

    if settings is None:
        settings = get_default_settings()
    masks_settings = settings.get("masks", {})
    shape = np.asarray(data).shape
    ny, nx = shape
    mask = np.zeros(shape, dtype=bool)
    info = {"seam_columns": [], "seam_rows": []}

    grow = int(masks_settings.get("amplifier_seam_grow_pixels", 1))

    for boundary in masks_settings.get("amplifier_boundaries", []) or []:
        axis = str(boundary.get("axis", "x")).lower()
        try:
            index = int(boundary.get("index"))
        except (TypeError, ValueError):
            continue
        width = int(boundary.get("width", 1)) + grow
        if axis in ("x", "column", "col"):
            mask[:, max(0, index - width):index + width + 1] = True
            info["seam_columns"].append(index)
        else:
            mask[max(0, index - width):index + width + 1, :] = True
            info["seam_rows"].append(index)

    if masks_settings.get("detect_amplifier_boundaries", True):
        work = np.array(data, dtype=float)
        if valid is not None:
            work[~np.asarray(valid, dtype=bool)] = np.nan
        work[~np.isfinite(work)] = np.nan
        sigma = float(masks_settings.get("amplifier_seam_sigma", 8.0))
        edge_margin = int(masks_settings.get("amplifier_seam_edge_margin", 20))

        def seam_indices(profile):
            steps = np.abs(np.diff(profile))
            finite = np.isfinite(steps)
            if int(finite.sum()) < 10:
                return np.array([], dtype=int)
            median = np.nanmedian(steps)
            scale = np.nanmedian(np.abs(steps - median)) * 1.4826
            if not np.isfinite(scale) or scale <= 0:
                return np.array([], dtype=int)
            candidates = np.where(finite & (steps > median + sigma * scale))[0]
            if edge_margin > 0:
                length = steps.size
                candidates = candidates[
                    (candidates >= edge_margin)
                    & (candidates < length - edge_margin)
                ]
            return candidates

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for index in seam_indices(np.nanmedian(work, axis=0)):
                lo = max(0, index - grow)
                hi = index + 1 + grow + 1
                mask[:, lo:hi] = True
                info["seam_columns"].append(int(index))
            for index in seam_indices(np.nanmedian(work, axis=1)):
                lo = max(0, index - grow)
                hi = index + 1 + grow + 1
                mask[lo:hi, :] = True
                info["seam_rows"].append(int(index))

    return mask, info


def detect_trails(data, base_mask=None, settings=None, exclude_mask=None):
    """Detect long, thin linear features such as asteroid or satellite trails.

    Bright pixels are thresholded against a robust background, grouped into
    connected components, and each component is accepted as a trail only when it
    is long, narrow, and highly elongated.  Compact sources therefore remain
    untouched.  Detection requires ``scipy``; without it the step is skipped.

    Parameters
    ----------
    base_mask : numpy.ndarray, optional
        Pixels to ignore during detection (edges, existing mask).
    exclude_mask : numpy.ndarray, optional
        Additional pixels to exclude, typically the saturation mask so that
        bleed and diffraction spikes are not mistaken for trails.

    Returns
    -------
    trail_mask : numpy.ndarray
        Boolean mask of accepted, grown trails.
    trails : list of dict
        Per-trail geometry (length, width, elongation, centroid, angle).
    info : dict
        Detection summary.
    """

    if settings is None:
        settings = get_default_settings()
    masks_settings = settings.get("masks", {})
    shape = np.asarray(data).shape
    trail_mask = np.zeros(shape, dtype=bool)
    trails = []
    info = {"n_trails": 0, "detected": False, "skipped": None}

    if not masks_settings.get("detect_trails", True):
        info["skipped"] = "disabled"
        return trail_mask, trails, info

    ndimage = _try_ndimage()
    if ndimage is None:
        info["skipped"] = "scipy_missing"
        return trail_mask, trails, info

    valid = np.isfinite(data)
    if base_mask is not None:
        valid &= ~np.asarray(base_mask, dtype=bool)
    if exclude_mask is not None:
        valid &= ~np.asarray(exclude_mask, dtype=bool)

    sample = data[valid]
    if sample.size < 100:
        info["skipped"] = "too_few_pixels"
        return trail_mask, trails, info

    median = float(np.median(sample))
    mad = float(np.median(np.abs(sample - median))) * 1.4826
    scale = mad if mad > 0 else float(np.std(sample))
    if scale <= 0:
        return trail_mask, trails, info

    sigma = float(masks_settings.get("trail_sigma", 5.0))
    binary = (data - median > sigma * scale) & valid
    if not binary.any():
        return trail_mask, trails, info

    min_length = float(masks_settings.get("trail_min_length_pixels", 50))
    max_width = float(masks_settings.get("trail_max_width_pixels", 20))
    min_elongation = float(masks_settings.get("trail_min_elongation", 4.0))
    min_pixels = int(masks_settings.get("trail_min_pixels", 20))

    # Use 8-connectivity so thin diagonal trails form a single component.
    labels, count = ndimage.label(binary, structure=np.ones((3, 3), dtype=int))
    for label in range(1, count + 1):
        component = labels == label
        pixels = int(component.sum())
        if pixels < min_pixels:
            continue
        ys, xs = np.where(component)
        coords = np.column_stack([xs, ys]).astype(float)
        centered = coords - coords.mean(axis=0)
        covariance = np.cov(centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        major = eigenvectors[:, int(np.argmax(eigenvalues))]
        minor = eigenvectors[:, int(np.argmin(eigenvalues))]
        projection_major = centered @ major
        projection_minor = centered @ minor
        length = float(projection_major.max() - projection_major.min())
        width = float(projection_minor.max() - projection_minor.min())
        elongation = length / max(width, 1.0)
        if length >= min_length and width <= max_width and elongation >= min_elongation:
            trail_mask |= component
            trails.append(
                {
                    "label": int(label),
                    "pixels": pixels,
                    "length_pixels": length,
                    "width_pixels": width,
                    "elongation": elongation,
                    "centroid_x": float(xs.mean()),
                    "centroid_y": float(ys.mean()),
                    "angle_deg": float(
                        np.degrees(np.arctan2(major[1], major[0]))
                    ),
                }
            )

    grow = int(masks_settings.get("trail_grow_pixels", 5))
    if trail_mask.any() and grow > 0:
        trail_mask = ndimage.binary_dilation(trail_mask, iterations=grow)

    info["n_trails"] = len(trails)
    info["detected"] = bool(trails)
    return trail_mask, trails, info


def _region_center_pixels(region, wcs):
    """Resolve a manual-region center to pixel coordinates."""

    if "x" in region and "y" in region:
        try:
            return float(region["x"]), float(region["y"])
        except (TypeError, ValueError):
            return None, None
    if "ra" in region and "dec" in region and wcs is not None and wcs.has_celestial:
        try:
            coordinate = SkyCoord(
                float(region["ra"]), float(region["dec"]),
                unit="deg", frame="icrs",
            )
            x, y = wcs.world_to_pixel(coordinate)
            return float(x), float(y)
        except Exception:
            return None, None
    return None, None


def _polygon_vertices_pixels(region, wcs):
    """Return manual-polygon vertices as an ``(N, 2)`` array of pixel columns."""

    vertices = region.get("vertices")
    if not vertices:
        return None
    if region.get("sky") and wcs is not None and wcs.has_celestial:
        try:
            values = np.asarray(vertices, dtype=float)
            coordinate = SkyCoord(
                values[:, 0], values[:, 1], unit="deg", frame="icrs"
            )
            x, y = wcs.world_to_pixel(coordinate)
            return np.column_stack([np.asarray(x, float), np.asarray(y, float)])
        except Exception:
            return None
    try:
        return np.asarray(vertices, dtype=float)
    except (TypeError, ValueError):
        return None


def _polygon_mask(shape, vertices):
    """Rasterize a polygon into a boolean mask using ray casting over its bbox."""

    ny, nx = shape
    mask = np.zeros(shape, dtype=bool)
    if vertices is None or len(vertices) < 3:
        return mask

    x_min = max(0, int(np.floor(vertices[:, 0].min())))
    x_max = min(nx, int(np.ceil(vertices[:, 0].max())) + 1)
    y_min = max(0, int(np.floor(vertices[:, 1].min())))
    y_max = min(ny, int(np.ceil(vertices[:, 1].max())) + 1)
    if x_max <= x_min or y_max <= y_min:
        return mask

    grid_y, grid_x = np.mgrid[y_min:y_max, x_min:x_max]
    inside = np.zeros(grid_x.shape, dtype=bool)
    n = len(vertices)
    j = n - 1
    for i in range(n):
        xi, yi = vertices[i]
        xj, yj = vertices[j]
        crosses = ((yi > grid_y) != (yj > grid_y)) & (
            grid_x < (xj - xi) * (grid_y - yi) / (yj - yi + 1e-12) + xi
        )
        inside ^= crosses
        j = i
    mask[y_min:y_max, x_min:x_max] = inside
    return mask


def make_manual_mask(shape, wcs=None, settings=None, pixel_scale=None):
    """Build a mask from user-specified circular, rectangular, or polygon regions.

    Each entry in ``masks.manual_regions`` is a dictionary with a ``type`` of
    ``circle``, ``rect``, or ``polygon``:

    - ``circle``: ``x``/``y`` (pixels) or ``ra``/``dec`` (degrees) center with a
      ``radius`` in pixels or ``radius_arcsec``.
    - ``rect``: ``x1``, ``x2``, ``y1``, ``y2`` one-based inclusive pixel bounds,
      or a FITS ``section`` string.
    - ``polygon``: ``vertices`` as ``[[x, y], ...]`` in pixels, or ``[[ra, dec],
      ...]`` in degrees when ``sky`` is true.

    Returns
    -------
    mask : numpy.ndarray
        Combined boolean mask of all regions.
    info : dict
        Summary of the regions that were applied.
    """

    if settings is None:
        settings = get_default_settings()
    regions = settings.get("masks", {}).get("manual_regions", []) or []
    ny, nx = shape
    mask = np.zeros(shape, dtype=bool)
    applied = []

    for region in regions:
        region_type = str(region.get("type", "circle")).lower()

        if region_type in ("circle", "disk"):
            center_x, center_y = _region_center_pixels(region, wcs)
            radius = region.get("radius")
            if (
                radius is None
                and region.get("radius_arcsec") is not None
                and pixel_scale
            ):
                radius = float(region["radius_arcsec"]) / float(pixel_scale)
            if center_x is None or radius is None:
                continue
            radius = float(radius)
            x0 = max(0, int(np.floor(center_x - radius)))
            x1 = min(nx, int(np.ceil(center_x + radius)) + 1)
            y0 = max(0, int(np.floor(center_y - radius)))
            y1 = min(ny, int(np.ceil(center_y + radius)) + 1)
            if x1 <= x0 or y1 <= y0:
                continue
            grid_y, grid_x = np.ogrid[y0:y1, x0:x1]
            disk = (grid_x - center_x) ** 2 + (grid_y - center_y) ** 2 <= radius ** 2
            mask[y0:y1, x0:x1] |= disk
            applied.append({"type": "circle", "x": center_x, "y": center_y,
                            "radius": radius})

        elif region_type in ("rect", "rectangle", "box"):
            if region.get("section") is not None:
                bounds = parse_fits_section(region["section"], shape=shape)
            elif all(key in region for key in ("x1", "x2", "y1", "y2")):
                bounds = parse_fits_section(
                    "[{}:{},{}:{}]".format(
                        int(region["x1"]), int(region["x2"]),
                        int(region["y1"]), int(region["y2"]),
                    ),
                    shape=shape,
                )
            else:
                bounds = None
            if bounds is None:
                continue
            x_start, x_stop, y_start, y_stop = bounds
            mask[y_start:y_stop, x_start:x_stop] = True
            applied.append({"type": "rect", "bounds": bounds})

        elif region_type == "polygon":
            vertices = _polygon_vertices_pixels(region, wcs)
            polygon_mask = _polygon_mask(shape, vertices)
            if polygon_mask.any():
                mask |= polygon_mask
                applied.append({"type": "polygon", "vertices": int(len(vertices))})

    return mask, {"applied": applied, "count": len(applied)}


def check_mask_overlaps(mask, positions, radii):
    """Return, for each position, whether a circular region hits masked pixels.

    Parameters
    ----------
    mask : numpy.ndarray
        Boolean mask to test against (for example the combined mask or the
        trail mask).
    positions : sequence of (x, y)
        Pixel positions to test.
    radii : float or sequence of float
        Test radius in pixels, shared or per position.

    Returns
    -------
    numpy.ndarray
        Boolean array, ``True`` where the region around a position overlaps a
        masked pixel.  Used by later stages to drop individual comparison, PSF,
        or calibration stars without rejecting the whole image.
    """

    mask = np.asarray(mask, dtype=bool)
    ny, nx = mask.shape
    positions = list(positions)
    if np.isscalar(radii):
        radii = [float(radii)] * len(positions)

    results = []
    for (x, y), radius in zip(positions, radii):
        x = float(x)
        y = float(y)
        radius = float(radius)
        x0 = max(0, int(np.floor(x - radius)))
        x1 = min(nx, int(np.ceil(x + radius)) + 1)
        y0 = max(0, int(np.floor(y - radius)))
        y1 = min(ny, int(np.ceil(y + radius)) + 1)
        if x1 <= x0 or y1 <= y0:
            results.append(False)
            continue
        sub = mask[y0:y1, x0:x1]
        if not sub.any():
            results.append(False)
            continue
        grid_y, grid_x = np.ogrid[y0:y1, x0:x1]
        within = (grid_x - x) ** 2 + (grid_y - y) ** 2 <= radius ** 2
        results.append(bool((sub & within).any()))

    return np.array(results, dtype=bool)


def _target_region_radius(settings):
    """Return the target test radius in pixels for mask-overlap checks."""

    masks_settings = settings.get("masks", {})
    radius_fwhm = masks_settings.get("target_overlap_radius_fwhm")
    if radius_fwhm is None:
        radius_fwhm = settings.get("apertures", {}).get(
            "sky_outer_radius_fwhm", 7.0
        )
    return float(radius_fwhm) * _fwhm_guess_pixels(settings)


def build_masks(ccd, metadata=None, settings=None, target=None, valid=None):
    """Build all pixel and artifact masks for a prepared working image.

    Combines the incoming mask (existing data quality, non-finite pixels, and
    invalid edges from earlier stages) with saturation and nonlinearity masks
    and their halos, hot/dead rows and columns, amplifier seams, detected
    trails, and user regions.  Every component is kept separately for
    diagnostics, and the union becomes the working mask.  The target is tested
    against the trail and combined masks so that a target-crossing trail raises
    a strong flag while defects elsewhere only mask local pixels.  The input
    ``ccd`` is not modified.

    Returns
    -------
    working : astropy.nddata.CCDData
        Copy of the image with the combined mask applied.
    components : dict of numpy.ndarray
        Individual boolean mask components and the ``combined`` mask.
    info : dict
        Flags, per-defect summaries, per-component fractions, and target
        overlap results.
    """

    if settings is None:
        settings = get_default_settings()
    masks_settings = settings.get("masks", {})

    data = np.asarray(ccd.data)
    shape = data.shape
    wcs = getattr(ccd, "wcs", None)

    base = (
        np.asarray(ccd.mask, dtype=bool)
        if getattr(ccd, "mask", None) is not None
        else np.zeros(shape, dtype=bool)
    )
    nonfinite = ~np.isfinite(data)

    saturation_components, saturation_info = make_saturation_mask(
        data, metadata, settings
    )
    saturation_full = saturation_components["saturation"]

    line_valid = ~(base | nonfinite | saturation_full)
    line_mask, line_info = make_line_defect_mask(data, line_valid, settings)
    # Exclude already-flagged bad lines so a hot column is not also reported as
    # an amplifier seam.
    amplifier_mask, amplifier_info = make_amplifier_seam_mask(
        data, line_valid & ~line_mask, settings
    )

    trail_exclude = base | nonfinite | saturation_full | line_mask | amplifier_mask
    trail_mask, trails, trail_info = detect_trails(
        data, base_mask=trail_exclude, settings=settings,
        exclude_mask=saturation_full,
    )

    pixel_scale = _pixel_scale_arcsec(ccd, metadata, settings)
    manual_mask, manual_info = make_manual_mask(
        shape, wcs, settings, pixel_scale
    )

    combined = (
        base
        | nonfinite
        | saturation_full
        | saturation_components["nonlinear"]
        | line_mask
        | amplifier_mask
        | trail_mask
        | manual_mask
    )

    components = {
        "input": base,
        "nonfinite": nonfinite,
        "saturated": saturation_components["saturated"],
        "nonlinear": saturation_components["nonlinear"],
        "saturation": saturation_full,
        "bad_lines": line_mask,
        "amplifier": amplifier_mask,
        "trails": trail_mask,
        "manual": manual_mask,
        "combined": combined,
    }

    flags = []
    if saturation_info["saturated_fraction"] > masks_settings.get(
        "saturation_high_fraction", 0.02
    ):
        flags.append("SATURATION_HIGH")
    if line_info["bad_rows"] or line_info["bad_columns"]:
        flags.append("BAD_ROWS_OR_COLUMNS")
    if amplifier_info["seam_columns"] or amplifier_info["seam_rows"]:
        flags.append("BAD_ROWS_OR_COLUMNS")
    if trails:
        flags.append("TRAIL_PRESENT")

    target_info = {
        "target_x": None,
        "target_y": None,
        "target_masked": None,
        "target_trail": None,
    }
    center = _coerce_target_coordinate(target, settings)
    if center is None and metadata is not None:
        ra = metadata.get("adopted_ra_deg")
        dec = metadata.get("adopted_dec_deg")
        if ra is not None and dec is not None:
            center = SkyCoord(float(ra), float(dec), unit="deg", frame="icrs")

    if center is not None and wcs is not None and wcs.has_celestial:
        try:
            x, y = wcs.world_to_pixel(center)
            x = float(x)
            y = float(y)
        except Exception:
            x = y = None
        if x is not None and np.isfinite(x) and np.isfinite(y):
            target_info["target_x"] = x
            target_info["target_y"] = y
            radius = _target_region_radius(settings)
            trail_hit = bool(
                check_mask_overlaps(trail_mask, [(x, y)], [radius])[0]
            )
            masked_hit = bool(
                check_mask_overlaps(combined, [(x, y)], [radius])[0]
            )
            target_info["target_trail"] = trail_hit
            target_info["target_masked"] = masked_hit
            if trail_hit:
                flags.append("TARGET_TRAIL")
            if masked_hit:
                flags.append("TARGET_MASKED")

    deduplicated = []
    for flag in flags:
        if flag not in deduplicated:
            deduplicated.append(flag)
    flags = deduplicated

    working = CCDData(
        np.array(data, copy=True),
        unit=ccd.unit,
        meta=ccd.meta.copy() if getattr(ccd, "meta", None) is not None else None,
        wcs=wcs,
        mask=combined,
        uncertainty=getattr(ccd, "uncertainty", None),
    )

    info = {
        "flags": flags,
        "saturation": saturation_info,
        "bad_lines": line_info,
        "amplifier": amplifier_info,
        "trails": trail_info,
        "trail_list": trails,
        "manual": manual_info,
        "target": target_info,
        "masked_fraction": float(combined.mean()) if combined.size else 0.0,
        "component_fractions": {
            name: float(component.mean()) if component.size else 0.0
            for name, component in components.items()
        },
    }
    return working, components, info


# ---------------------------------------------------------------------------
# Step 7: optional cosmic-ray and fringe correction
#
# Both stages are opt-in and reversible.  Cosmic-ray handling produces a mask
# and, in ``clean`` mode, a cleaned derivative that is kept separate from the
# measurement data (masking is preferred over pixel replacement).  Fringe
# correction subtracts a scaled additive fringe model and keeps both the model
# and corrected derivative.  Disabling either stage returns the image
# unchanged.  The input ``ccd`` is never modified.
# ---------------------------------------------------------------------------


_ASTROSCRAPPY_WARNED = False


def _try_astroscrappy():
    """Return ``astroscrappy`` or ``None``, warning once when unavailable."""

    global _ASTROSCRAPPY_WARNED
    try:
        import astroscrappy
    except ImportError:
        if not _ASTROSCRAPPY_WARNED:
            warnings.warn(
                "astroscrappy is not available; cosmic-ray handling is skipped.",
                RuntimeWarning,
            )
            _ASTROSCRAPPY_WARNED = True
        return None
    return astroscrappy


def _project_target_pixel(ccd, metadata, settings, target):
    """Return the target pixel ``(x, y)`` or ``None`` using WCS projection."""

    wcs = getattr(ccd, "wcs", None)
    if wcs is None or not wcs.has_celestial:
        return None
    center = _coerce_target_coordinate(target, settings)
    if center is None and metadata is not None:
        ra = metadata.get("adopted_ra_deg")
        dec = metadata.get("adopted_dec_deg")
        if ra is not None and dec is not None:
            center = SkyCoord(float(ra), float(dec), unit="deg", frame="icrs")
    if center is None:
        return None
    try:
        x, y = wcs.world_to_pixel(center)
        x = float(x)
        y = float(y)
    except Exception:
        return None
    if not (np.isfinite(x) and np.isfinite(y)):
        return None
    return x, y


def apply_cosmic_rays(
    ccd, metadata=None, settings=None, target=None, psf_positions=None
):
    """Detect and optionally clean cosmic rays with the L.A.Cosmic algorithm.

    Uses ``astroscrappy`` in one of four modes from
    ``masks.cosmic_rays.mode``:

    - ``off``: return the image unchanged.
    - ``detect_only``: compute a cosmic-ray mask and diagnostics without
      changing the working mask.
    - ``mask``: add the cosmic-ray mask to the working mask (preferred for
      photometry).
    - ``clean``: add the mask and also produce a cleaned derivative, kept
      separate so repaired pixels are not treated as measurements.

    Cosmic rays intersecting the target position or a supplied PSF-star core
    raise ``TARGET_COSMIC_RAY``.  Gain, read noise, and saturation come from the
    metadata unless overridden.  The input ``ccd`` is not modified.

    Returns
    -------
    working : astropy.nddata.CCDData
        Image with the cosmic-ray mask applied (``mask``/``clean``) or an
        unchanged copy (``off``/``detect_only``).
    products : dict
        ``cosmic_mask`` and ``cleaned`` (an array in ``clean`` mode, else
        ``None``).
    info : dict
        Mode, parameters, cosmic-ray counts, flags, and target overlap.
    """

    if settings is None:
        settings = get_default_settings()
    cosmic_settings = settings.get("masks", {}).get("cosmic_rays", {})
    data = np.asarray(ccd.data)
    shape = data.shape

    mode = cosmic_settings.get("mode", "mask")
    enabled = cosmic_settings.get("enabled", False)
    products = {"cosmic_mask": np.zeros(shape, dtype=bool), "cleaned": None}
    info = {
        "mode": mode,
        "applied": False,
        "skipped": None,
        "cosmic_pixel_count": 0,
        "cosmic_pixel_fraction": 0.0,
        "target_overlap": None,
        "flags": [],
        "parameters": {},
    }

    if not enabled or mode == "off":
        info["skipped"] = "disabled" if not enabled else "off"
        return ccd, products, info

    astroscrappy = _try_astroscrappy()
    if astroscrappy is None:
        info["skipped"] = "astroscrappy_missing"
        return ccd, products, info

    existing_mask = (
        np.asarray(ccd.mask, dtype=bool)
        if getattr(ccd, "mask", None) is not None
        else None
    )

    gain = cosmic_settings.get("gain")
    if gain is None and cosmic_settings.get("use_image_gain", True) and metadata:
        gain = _as_float(metadata.get("gain"))
    read_noise = cosmic_settings.get("read_noise")
    if (
        read_noise is None
        and cosmic_settings.get("use_image_read_noise", True)
        and metadata
    ):
        read_noise = _as_float(metadata.get("read_noise"))
    saturation = cosmic_settings.get("saturation")
    if (
        saturation is None
        and cosmic_settings.get("use_image_saturation", True)
        and metadata
    ):
        saturation = _as_float(metadata.get("saturation"))

    detect_kwargs = {
        "sigclip": float(cosmic_settings.get("sigclip", 4.5)),
        "sigfrac": float(cosmic_settings.get("sigfrac", 0.3)),
        "objlim": float(cosmic_settings.get("objlim", 5.0)),
        "niter": int(cosmic_settings.get("niter", 4)),
        "cleantype": cosmic_settings.get("cleantype", "meanmask"),
    }
    if gain is not None:
        detect_kwargs["gain"] = float(gain)
    if read_noise is not None:
        detect_kwargs["readnoise"] = float(read_noise)
    if saturation is not None:
        detect_kwargs["satlevel"] = float(saturation)
    info["parameters"] = dict(detect_kwargs)

    inmask = existing_mask if existing_mask is not None else None
    try:
        cosmic_mask, cleaned = astroscrappy.detect_cosmics(
            np.ascontiguousarray(data, dtype=np.float32),
            inmask=inmask,
            **detect_kwargs,
        )
    except Exception as error:
        warnings.warn(
            "Cosmic-ray detection failed ({}); image unchanged.".format(error),
            RuntimeWarning,
        )
        info["skipped"] = "detection_failed"
        return ccd, products, info

    cosmic_mask = np.asarray(cosmic_mask, dtype=bool)
    grow = int(cosmic_settings.get("grow_pixels", 1))
    if grow > 0 and cosmic_mask.any():
        ndimage = _try_ndimage()
        if ndimage is not None:
            cosmic_mask = ndimage.binary_dilation(cosmic_mask, iterations=grow)

    products["cosmic_mask"] = cosmic_mask
    info["applied"] = True
    info["cosmic_pixel_count"] = int(cosmic_mask.sum())
    info["cosmic_pixel_fraction"] = (
        float(cosmic_mask.mean()) if cosmic_mask.size else 0.0
    )

    fwhm = _fwhm_guess_pixels(settings)
    target_xy = _project_target_pixel(ccd, metadata, settings, target)
    if target_xy is not None:
        radius = float(cosmic_settings.get("target_radius_fwhm", 3.0)) * fwhm
        overlap = bool(check_mask_overlaps(cosmic_mask, [target_xy], [radius])[0])
        info["target_overlap"] = overlap
        if overlap:
            info["flags"].append("TARGET_COSMIC_RAY")

    if psf_positions:
        radius = float(cosmic_settings.get("psf_core_radius_fwhm", 1.0)) * fwhm
        psf_hits = check_mask_overlaps(
            cosmic_mask, list(psf_positions), [radius] * len(psf_positions)
        )
        info["psf_overlap_indices"] = np.where(psf_hits)[0].tolist()
        if psf_hits.any() and "TARGET_COSMIC_RAY" not in info["flags"]:
            # A PSF-star hit is not a target hit, but record it for the caller.
            info.setdefault("flags", [])

    if mode == "detect_only":
        return ccd, products, info

    combined_mask = cosmic_mask
    if existing_mask is not None:
        combined_mask = existing_mask | cosmic_mask

    if mode == "clean":
        products["cleaned"] = np.asarray(cleaned, dtype=float)

    working = CCDData(
        np.array(data, copy=True),
        unit=ccd.unit,
        meta=ccd.meta.copy() if getattr(ccd, "meta", None) is not None else None,
        wcs=getattr(ccd, "wcs", None),
        mask=combined_mask,
        uncertainty=getattr(ccd, "uncertainty", None),
    )
    return working, products, info


def _load_fringe_map(path):
    """Load the first 2D image array from a fringe-map FITS file."""

    with fits.open(path, mode="readonly", memmap=False) as hdulist:
        for hdu in hdulist:
            array = _numeric_2d_array(hdu)
            if array is not None:
                return np.array(array, dtype=float), hdu.header.copy()
    raise ValueError("No 2D image found in fringe map: {}".format(path))


def _align_fringe_map(map_array, map_header, data_shape, metadata, settings,
                      crop_slices):
    """Match a fringe map to the working image and validate compatibility.

    Returns ``(aligned_map, info)`` where ``aligned_map`` is ``None`` when the
    map cannot be used.
    """

    fringe_settings = settings.get("fringe", {})
    info = {"ok": True, "reason": None, "alignment": None}

    if map_array.shape == data_shape:
        aligned = map_array
        info["alignment"] = "direct"
    elif crop_slices is not None:
        (y0, y1), (x0, x1) = crop_slices
        sub = map_array[y0:y1, x0:x1]
        if sub.shape == data_shape:
            aligned = sub
            info["alignment"] = "cropped_to_science"
        else:
            info["ok"] = False
            info["reason"] = "shape_mismatch"
            return None, info
    else:
        if fringe_settings.get("validate_shape", True):
            info["ok"] = False
            info["reason"] = "shape_mismatch"
            return None, info
        aligned = map_array
        info["alignment"] = "unchecked"

    if fringe_settings.get("validate_binning", True) and metadata is not None:
        map_binning = map_header.get("CCDSUM")
        image_binning = metadata.get("binning")
        if (
            map_binning is not None
            and image_binning is not None
            and str(map_binning).strip() != str(image_binning).strip()
        ):
            info["ok"] = False
            info["reason"] = "binning_mismatch"
            return None, info

    if fringe_settings.get("check_filter", True) and metadata is not None:
        map_filter = map_header.get("FILTER")
        if map_filter is not None:
            map_filter_norm = normalize_filter_name(map_filter)
            image_filter = metadata.get("filter")
            if image_filter is not None and map_filter_norm != image_filter:
                info["ok"] = False
                info["reason"] = "filter_mismatch"
                return None, info

    return aligned, info


def _fringe_scale_lstsq(data, fringe, valid, settings):
    """Estimate an additive fringe scale by sigma-clipped least squares."""

    fringe_settings = settings.get("fringe", {})
    sigma = float(fringe_settings.get("sigma_clip", 3.0))
    iterations = int(fringe_settings.get("maximum_iterations", 3))
    source_sigma = float(fringe_settings.get("source_sigma", 3.0))

    data = np.asarray(data, dtype=float)
    fringe = np.asarray(fringe, dtype=float)
    good = np.asarray(valid, dtype=bool) & np.isfinite(data) & np.isfinite(fringe)
    if int(good.sum()) < 50:
        return None, {"reason": "too_few_pixels", "n_pixels": int(good.sum())}

    background = float(np.median(data[good]))
    residual_data = data - background

    spread = np.median(np.abs(residual_data[good] - np.median(residual_data[good])))
    spread *= 1.4826
    if spread > 0:
        good &= residual_data < source_sigma * spread

    scale = 0.0
    rms_before = float(np.std(residual_data[good])) if good.any() else 0.0
    for _ in range(max(1, iterations)):
        selected = good
        denominator = float(np.sum(fringe[selected] ** 2))
        if denominator <= 0:
            return None, {"reason": "zero_fringe_power"}
        scale = float(np.sum(residual_data[selected] * fringe[selected]) / denominator)
        residual = residual_data - scale * fringe
        median = float(np.median(residual[selected]))
        clip = np.median(np.abs(residual[selected] - median)) * 1.4826
        if clip <= 0:
            break
        good = selected & (np.abs(residual - median) < sigma * clip)
        if int(good.sum()) < 50:
            good = selected
            break

    residual = residual_data - scale * fringe
    rms_after = float(np.std(residual[good])) if good.any() else 0.0
    info = {
        "reason": None,
        "n_pixels": int(good.sum()),
        "rms_before": rms_before,
        "rms_after": rms_after,
        "background": background,
    }
    return scale, info


def _read_control_points(settings):
    """Return an ``(N, 4)`` array of bright/dark control pairs, or ``None``."""

    fringe_settings = settings.get("fringe", {})
    points = fringe_settings.get("control_points")
    if points is None:
        path = fringe_settings.get("control_points_path")
        if path is None:
            return None
        try:
            points = np.loadtxt(path)
        except (OSError, ValueError):
            return None
    points = np.atleast_2d(np.asarray(points, dtype=float))
    if points.ndim != 2 or points.shape[1] < 4:
        return None
    return points[:, :4]


def _fringe_scale_control_pairs(data, fringe, settings):
    """Estimate the fringe scale from bright/dark control pairs."""

    fringe_settings = settings.get("fringe", {})
    pairs = _read_control_points(settings)
    if pairs is None:
        return None, {"reason": "no_control_points"}

    ny, nx = data.shape
    scales = []
    for bx, by, dx, dy in pairs:
        bxi, byi, dxi, dyi = int(bx), int(by), int(dx), int(dy)
        if not (
            0 <= byi < ny and 0 <= bxi < nx and 0 <= dyi < ny and 0 <= dxi < nx
        ):
            continue
        fringe_difference = fringe[byi, bxi] - fringe[dyi, dxi]
        if abs(fringe_difference) <= 0:
            continue
        data_difference = data[byi, bxi] - data[dyi, dxi]
        scales.append(data_difference / fringe_difference)

    minimum_pairs = int(fringe_settings.get("minimum_control_pairs", 10))
    if len(scales) < minimum_pairs:
        return None, {"reason": "too_few_pairs", "n_pairs": len(scales)}

    scales = np.asarray(scales, dtype=float)
    sigma = float(fringe_settings.get("sigma_clip", 3.0))
    for _ in range(int(fringe_settings.get("maximum_iterations", 3))):
        median = float(np.median(scales))
        spread = float(np.median(np.abs(scales - median))) * 1.4826
        if spread <= 0:
            break
        keep = np.abs(scales - median) < sigma * spread
        if keep.sum() < minimum_pairs:
            break
        scales = scales[keep]

    return float(np.median(scales)), {
        "reason": None,
        "n_pairs": int(scales.size),
    }


def correct_fringe(
    ccd, metadata=None, settings=None, crop_slices=None, source_mask=None,
    target=None,
):
    """Subtract a scaled additive fringe model for eligible i/z images.

    The stage runs only when ``fringe.enabled`` is set and the image is eligible
    (``fringe.eligible`` or its filter is in ``fringe.filters``, and, when
    configured, its instrument is allowed).  The fringe map is loaded and
    validated against the working image (shape, binning, filter), scaled by
    least squares or bright/dark control pairs, and subtracted.  A scale outside
    ``[scale_minimum, scale_maximum]`` raises ``FRINGE_CORRECTION_FAILED`` and
    leaves the image unchanged.  The input ``ccd`` is not modified.

    Returns
    -------
    working : astropy.nddata.CCDData
        Fringe-corrected image, or an unchanged copy when the stage is skipped.
    products : dict
        ``fringe_model``, ``corrected`` (arrays) and the applied ``scale``.
    info : dict
        Eligibility, validation, scaling, residual metrics, and flags.
    """

    if settings is None:
        settings = get_default_settings()
    fringe_settings = settings.get("fringe", {})
    data = np.asarray(ccd.data)
    shape = data.shape

    products = {"fringe_model": None, "corrected": None, "scale": None}
    info = {"applied": False, "skipped": None, "scale": None, "flags": [],
            "validation": None, "scaling": None}

    if not fringe_settings.get("enabled", False):
        info["skipped"] = "disabled"
        return ccd, products, info

    image_filter = metadata.get("filter") if metadata else None
    eligible = fringe_settings.get("eligible", False) or (
        image_filter is not None
        and image_filter in fringe_settings.get("filters", [])
    )
    allowed_instruments = fringe_settings.get("instruments")
    if allowed_instruments and metadata is not None:
        instrument = normalize_instrument_name(metadata.get("instrument"))
        if instrument not in {
            normalize_instrument_name(name) for name in allowed_instruments
        }:
            eligible = False
    if not eligible:
        info["skipped"] = "not_eligible"
        return ccd, products, info

    map_path = fringe_settings.get("map_path")
    if map_path is None:
        info["skipped"] = "no_map"
        return ccd, products, info

    try:
        map_array, map_header = _load_fringe_map(map_path)
    except (OSError, ValueError) as error:
        warnings.warn(
            "Could not read fringe map ({}); image unchanged.".format(error),
            RuntimeWarning,
        )
        info["skipped"] = "map_unreadable"
        info["flags"].append("FRINGE_CORRECTION_FAILED")
        return ccd, products, info

    aligned, validation = _align_fringe_map(
        map_array, map_header, shape, metadata, settings, crop_slices
    )
    info["validation"] = validation
    if aligned is None:
        info["skipped"] = validation.get("reason", "validation_failed")
        info["flags"].append("FRINGE_CORRECTION_FAILED")
        return ccd, products, info

    valid = np.isfinite(data)
    if getattr(ccd, "mask", None) is not None:
        valid &= ~np.asarray(ccd.mask, dtype=bool)
    if source_mask is not None:
        valid &= ~np.asarray(source_mask, dtype=bool)

    method = fringe_settings.get("scale_method", "lstsq")
    if method == "control_pairs":
        scale, scaling = _fringe_scale_control_pairs(data, aligned, settings)
        if scale is None:
            scale, scaling = _fringe_scale_lstsq(data, aligned, valid, settings)
            scaling["fallback"] = "lstsq"
    else:
        scale, scaling = _fringe_scale_lstsq(data, aligned, valid, settings)
    info["scaling"] = scaling

    if scale is None:
        info["skipped"] = scaling.get("reason", "scaling_failed")
        info["flags"].append("FRINGE_CORRECTION_FAILED")
        return ccd, products, info

    scale_minimum = fringe_settings.get("scale_minimum")
    scale_maximum = fringe_settings.get("scale_maximum")
    invalid_scale = (
        (scale_minimum is not None and scale < scale_minimum)
        or (scale_maximum is not None and scale > scale_maximum)
    )
    if invalid_scale and fringe_settings.get("reject_invalid_scale", True):
        info["skipped"] = "scale_out_of_range"
        info["scale"] = scale
        info["flags"].append("FRINGE_CORRECTION_FAILED")
        return ccd, products, info

    fringe_model = scale * aligned
    corrected = data - fringe_model
    products["fringe_model"] = fringe_model
    products["corrected"] = corrected
    products["scale"] = scale
    info["applied"] = True
    info["scale"] = scale

    working = CCDData(
        np.array(corrected, copy=True),
        unit=ccd.unit,
        meta=ccd.meta.copy() if getattr(ccd, "meta", None) is not None else None,
        wcs=getattr(ccd, "wcs", None),
        mask=(
            np.asarray(ccd.mask, dtype=bool)
            if getattr(ccd, "mask", None) is not None
            else None
        ),
        uncertainty=getattr(ccd, "uncertainty", None),
    )
    return working, products, info


def metadata_table(metadata_rows):
    """Convert normalized metadata dictionaries into a masked Astropy table."""

    rows = list(metadata_rows)
    table = Table(masked=True)

    for field in METADATA_FIELDS:
        values = [row.get(field) for row in rows]
        mask = [value is None or np.ma.is_masked(value) for value in values]

        if field in FLOAT_METADATA_FIELDS:
            data = [np.nan if missing else float(value) for value, missing in zip(values, mask)]
            column = MaskedColumn(data, mask=mask, name=field, dtype=float)
        elif field in INTEGER_METADATA_FIELDS:
            data = [0 if missing else int(value) for value, missing in zip(values, mask)]
            column = MaskedColumn(data, mask=mask, name=field, dtype=int)
        elif field in BOOLEAN_METADATA_FIELDS:
            data = [False if missing else bool(value) for value, missing in zip(values, mask)]
            column = MaskedColumn(data, mask=mask, name=field, dtype=bool)
        else:
            data = [
                ""
                if missing
                else ",".join(value)
                if isinstance(value, (list, tuple))
                else str(value)
                for value, missing in zip(values, mask)
            ]
            column = MaskedColumn(data, mask=mask, name=field)

        if field in METADATA_UNITS:
            column.unit = METADATA_UNITS[field]
        table.add_column(column)

    return table


def read_fits_batch(paths=None, settings=None, target=None, continue_on_error=False):
    """Discover and read a batch of FITS files in deterministic order.

    Parameters
    ----------
    paths : str, pathlib.Path, or sequence, optional
        Files, directories, or glob patterns passed to ``discover_fits_files``.
    settings : mapping, optional
        Settings shared by this call.  Resolve per-image overrides before calling
        ``read_fits_image`` individually when images require different settings.
    target : astropy.coordinates.SkyCoord, mapping, or pair, optional
        Target coordinate used during science-HDU selection.
    continue_on_error : bool, optional
        Continue reading other files after a failure.  Failure messages are
        stored in ``metadata.meta['read_errors']``.

    Returns
    -------
    images : list of astropy.nddata.CCDData
        Successfully read images in the same order as the metadata rows.
    metadata : astropy.table.Table
        Masked table containing normalized scalar metadata.
    """

    if settings is None:
        settings = get_default_settings()

    files = discover_fits_files(paths, settings=settings)
    images = []
    rows = []
    errors = []

    for path in files:
        try:
            ccd, metadata = read_fits_image(path, settings=settings, target=target)
        except (OSError, ValueError, IndexError) as error:
            if not continue_on_error:
                raise
            errors.append({"path": str(path), "error": str(error)})
            warnings.warn(
                "Skipping {}: {}".format(path, error),
                RuntimeWarning,
            )
            continue
        images.append(ccd)
        rows.append(metadata)

    table = metadata_table(rows)
    table.meta["read_errors"] = errors
    return images, table


__all__ = [
    "FITS_ENDINGS",
    "SCIENCE_EXTNAMES",
    "apply_cosmic_rays",
    "build_masks",
    "build_valid_region",
    "check_mask_overlaps",
    "correct_fringe",
    "crop_to_processing_region",
    "define_processing_region",
    "detect_empirical_edges",
    "detect_trails",
    "discover_fits_files",
    "extract_metadata",
    "header_section_region",
    "is_fits_path",
    "make_amplifier_seam_mask",
    "make_line_defect_mask",
    "make_manual_mask",
    "make_saturation_mask",
    "metadata_table",
    "parse_fits_section",
    "read_fits_batch",
    "read_fits_image",
    "select_science_hdu",
]
