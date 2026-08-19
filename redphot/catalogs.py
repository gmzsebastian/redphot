"""Catalog queries, caching, source matching, and conservative WCS refinement.

All functions use Astropy tables and plain dictionaries.  WCS refinement is
limited to translation, rotation, and a uniform scale change.  Science pixels
are never resampled or modified.
"""

from copy import deepcopy
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import warnings

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import MaskedColumn, Table, unique, vstack
from astropy.time import Time
from astropy.wcs import WCS

from .config import get_default_settings


CATALOG_ALIASES = {
    "gaia": "gaia",
    "gaiadr3": "gaia",
    "gaia-dr3": "gaia",
    "ps1": "ps1",
    "panstarrs": "ps1",
    "pan-starrs": "ps1",
    "sdss": "sdss",
    "apass": "apass",
    "skymapper": "skymapper",
    "sky-mapper": "skymapper",
    "user": "user",
    "local": "user",
}


# Candidate input columns are intentionally generous because VizieR column
# spellings differ between releases and user-supplied tables.
COMMON_COLUMN_CANDIDATES = {
    "source_id": ["source_id", "Source", "objID", "ObjectId", "recno", "ID"],
    "ra": ["ra", "RA_ICRS", "RAJ2000", "RAICRS", "RAdeg", "RA"],
    "dec": ["dec", "DE_ICRS", "DEJ2000", "DEICRS", "DEdeg", "DEC", "DE"],
    "pmra": ["pmra", "pmRA", "pmRA*", "PMRA"],
    "pmdec": ["pmdec", "pmDE", "PMDEC"],
    "parallax": ["parallax", "Plx", "plx"],
    "ref_epoch": ["ref_epoch", "Epoch", "epoch", "epRA"],
    "morphology_score": ["morphology_score", "ClassStar", "class_star"],
    "ruwe": ["ruwe", "RUWE"],
    "variable_flag": ["variable_flag", "VarFlag", "varFlag", "Variable"],
    "quality_flag": ["quality_flag", "Qual", "flags", "q_mode"],
    "source_class": ["source_class", "class", "Class", "cl"],
    "kron_mag": ["kron_mag", "iKmag", "rKmag", "gKmag"],
    "mag_u": ["mag_u", "umag", "u_mag", "uPSF"],
    "mag_v": ["mag_v", "vmag", "v_mag", "vPSF"],
    "mag_g": ["mag_g", "gmag", "g_mag", "g'mag", "gPSF"],
    "mag_r": ["mag_r", "rmag", "r_mag", "r'mag", "rPSF"],
    "mag_i": ["mag_i", "imag", "i_mag", "i'mag", "iPSF"],
    "mag_z": ["mag_z", "zmag", "z_mag", "zPSF"],
    "mag_y": ["mag_y", "ymag", "y_mag"],
    "mag_B": ["mag_B", "Bmag", "B_mag"],
    "mag_V": ["mag_V", "Vmag", "V_mag"],
    "mag_R": ["mag_R", "Rmag", "R_mag"],
    "mag_I": ["mag_I", "Imag", "I_mag"],
    "mag_G": ["mag_G", "Gmag", "phot_g_mean_mag"],
    "mag_BP": ["mag_BP", "BPmag", "phot_bp_mean_mag"],
    "mag_RP": ["mag_RP", "RPmag", "phot_rp_mean_mag"],
}


MAGNITUDE_NAMES = (
    "u",
    "v",
    "g",
    "r",
    "i",
    "z",
    "y",
    "B",
    "V",
    "R",
    "I",
    "G",
    "BP",
    "RP",
)


def normalize_catalog_name(catalog_name):
    """Return the standard short name for a supported catalog."""

    if catalog_name is None:
        return None
    key = str(catalog_name).strip().lower().replace("_", "-")
    return CATALOG_ALIASES.get(key, key)


def _safe_name(value):
    """Return a filesystem-safe object or catalog name."""

    cleaned = re.sub(r"[^A-Za-z0-9._+-]+", "_", str(value).strip())
    return cleaned.strip("_") or "field"


def catalog_cache_path(object_name, catalog_name, settings=None, radius_arcmin=None):
    """Return the deterministic path for a normalized cached catalog."""

    if settings is None:
        settings = get_default_settings()
    catalog_settings = settings.get("catalogs", {})
    if radius_arcmin is None:
        radius_arcmin = float(catalog_settings.get("search_radius_arcmin", 10.0))
    cache_format = catalog_settings.get("cache_format", "ecsv")
    suffix = ".ecsv" if cache_format == "ecsv" else ".fits"
    filename = "{}_{}_{:.2f}arcmin{}".format(
        _safe_name(object_name or "field"),
        _safe_name(normalize_catalog_name(catalog_name)),
        float(radius_arcmin),
        suffix,
    )
    return Path(catalog_settings.get("cache_directory", "catalogs")) / filename


def _canonical_column_key(name):
    """Normalize a column spelling for case-insensitive matching."""

    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _find_column(table, candidates, case_sensitive=False):
    """Return the first matching column name from a candidate sequence."""

    exact = {str(name): str(name) for name in table.colnames}
    for candidate in candidates:
        if candidate in exact:
            return exact[candidate]
    if case_sensitive:
        return None
    canonical = {
        _canonical_column_key(name): str(name) for name in table.colnames
    }
    for candidate in candidates:
        match = canonical.get(_canonical_column_key(candidate))
        if match is not None:
            return match
    return None


def _numeric_column(table, source_name, unit=None):
    """Convert one table column to a masked floating column."""

    length = len(table)
    if source_name is None:
        column = MaskedColumn(np.zeros(length), mask=True, dtype=float)
    else:
        source = table[source_name]
        source_mask = np.ma.getmaskarray(source)
        try:
            if getattr(source, "unit", None) is not None and unit is not None:
                values = u.Quantity(source).to_value(unit)
            else:
                values = np.asarray(source, dtype=float)
        except (TypeError, ValueError, u.UnitConversionError):
            values = np.full(length, np.nan)
        invalid = ~np.isfinite(values)
        column = MaskedColumn(
            np.where(invalid, 0.0, values),
            mask=source_mask | invalid,
            dtype=float,
        )
    if unit is not None:
        column.unit = unit
    return column


def _string_column(table, source_name):
    """Convert one table column to a masked string column."""

    length = len(table)
    if source_name is None:
        return MaskedColumn([""] * length, mask=True, dtype="U64")
    source = table[source_name]
    mask = np.ma.getmaskarray(source)
    values = ["" if masked else str(value) for value, masked in zip(source, mask)]
    return MaskedColumn(values, mask=mask, dtype="U64")


def _catalog_characteristics(table, selected, catalog_name):
    """Derive portable morphology, variability, and quality indicators."""

    length = len(table)
    point_source = MaskedColumn(
        np.zeros(length, dtype=bool), mask=True, dtype=bool
    )
    known_variable = MaskedColumn(
        np.zeros(length, dtype=bool), mask=True, dtype=bool
    )
    catalog_quality = MaskedColumn(
        np.zeros(length, dtype=bool), mask=True, dtype=bool
    )

    morphology = _numeric_column(table, selected.get("morphology_score"))
    morphology_values = np.asarray(morphology.filled(np.nan), dtype=float)
    valid_morphology = np.isfinite(morphology_values)
    point_source[valid_morphology] = morphology_values[valid_morphology] >= 0.8
    point_source.mask[valid_morphology] = False

    source_class = _string_column(table, selected.get("source_class"))
    for index, value in enumerate(source_class):
        if np.ma.is_masked(value):
            continue
        label = str(value).strip().lower()
        if label:
            point_source[index] = label in {"star", "stellar", "6", "qso"}
            point_source.mask[index] = False

    if catalog_name == "gaia":
        point_source[:] = True
        point_source.mask[:] = False

    psf_source = selected.get("mag_i") or selected.get("mag_r")
    kron_source = selected.get("kron_mag")
    if catalog_name == "ps1" and psf_source is not None and kron_source is not None:
        psf = _numeric_column(table, psf_source)
        kron = _numeric_column(table, kron_source)
        difference = np.abs(
            np.asarray(psf.filled(np.nan), dtype=float)
            - np.asarray(kron.filled(np.nan), dtype=float)
        )
        valid = np.isfinite(difference)
        point_source[valid] = difference[valid] <= 0.10
        point_source.mask[valid] = False

    variable_flag = _string_column(table, selected.get("variable_flag"))
    nonvariable_labels = {
        "",
        "0",
        "n",
        "no",
        "false",
        "constant",
        "not_available",
        "notavailable",
    }
    for index, value in enumerate(variable_flag):
        if np.ma.is_masked(value):
            continue
        label = str(value).strip().lower()
        known_variable[index] = label not in nonvariable_labels
        known_variable.mask[index] = False

    ruwe = _numeric_column(table, selected.get("ruwe"))
    ruwe_values = np.asarray(ruwe.filled(np.nan), dtype=float)
    valid_ruwe = np.isfinite(ruwe_values)
    catalog_quality[valid_ruwe] = ruwe_values[valid_ruwe] <= 1.4
    catalog_quality.mask[valid_ruwe] = False

    quality_flag = _string_column(table, selected.get("quality_flag"))
    if catalog_name in {"skymapper", "sdss"}:
        for index, value in enumerate(quality_flag):
            if np.ma.is_masked(value):
                continue
            label = str(value).strip().lower()
            if label:
                catalog_quality[index] = label in {"0", "good", "clean", "primary"}
                catalog_quality.mask[index] = False

    return {
        "morphology_score": morphology,
        "ruwe": ruwe,
        "variable_flag": variable_flag,
        "quality_flag": quality_flag,
        "source_class": source_class,
        "point_source": point_source,
        "known_variable": known_variable,
        "catalog_quality": catalog_quality,
    }


def normalize_catalog(table, catalog_name="user", column_map=None):
    """Normalize a catalog to redphot column names and physical units.

    The output always contains ``source_id``, ``ra``, ``dec``, proper motion,
    parallax, reference epoch, and standard magnitude/error columns. Missing
    values are represented by masked table entries, never numeric sentinels.

    Parameters
    ----------
    table : astropy.table.Table
        Raw queried or user-supplied catalog.
    catalog_name : str, optional
        Catalog identifier stored in the output metadata.
    column_map : mapping, optional
        Explicit mapping from normalized names such as ``ra`` or ``mag_r`` to
        input column names. Explicit entries take precedence over candidates.

    Returns
    -------
    astropy.table.Table
        Normalized masked catalog.
    """

    table = Table(table, masked=True, copy=True)
    column_map = dict(column_map or {})
    normalized_name = normalize_catalog_name(catalog_name) or "user"
    output = Table(masked=True)

    selected = {}
    for standard_name, candidates in COMMON_COLUMN_CANDIDATES.items():
        explicit = column_map.get(standard_name)
        selected[standard_name] = (
            explicit
            if explicit in table.colnames
            else _find_column(
                table,
                candidates,
                case_sensitive=standard_name.startswith("mag_"),
            )
        )

    output["source_id"] = _string_column(table, selected["source_id"])
    output["catalog_name"] = [normalized_name] * len(table)
    output["ra"] = _numeric_column(table, selected["ra"], u.deg)
    output["dec"] = _numeric_column(table, selected["dec"], u.deg)
    output["pmra"] = _numeric_column(table, selected["pmra"], u.mas / u.yr)
    output["pmdec"] = _numeric_column(table, selected["pmdec"], u.mas / u.yr)
    output["parallax"] = _numeric_column(table, selected["parallax"], u.mas)
    output["ref_epoch"] = _numeric_column(table, selected["ref_epoch"], u.yr)

    characteristics = _catalog_characteristics(table, selected, normalized_name)
    for name, column in characteristics.items():
        output[name] = column

    for band in MAGNITUDE_NAMES:
        magnitude_name = "mag_{}".format(band)
        magnitude_source = selected.get(magnitude_name)
        output[magnitude_name] = _numeric_column(table, magnitude_source, u.mag)

        error_name = "magerr_{}".format(band)
        explicit_error = column_map.get(error_name)
        candidates = []
        if magnitude_source is not None:
            candidates.extend(
                [
                    "e_{}".format(magnitude_source),
                    "{}err".format(magnitude_source),
                    "e{}".format(magnitude_source),
                ]
            )
        candidates.extend(
            [
                error_name,
                "e_{}mag".format(band),
                "e_{}_mag".format(band),
            ]
        )
        error_source = (
            explicit_error
            if explicit_error in table.colnames
            else _find_column(table, candidates, case_sensitive=True)
        )
        output[error_name] = _numeric_column(table, error_source, u.mag)

    output["catalog_row"] = np.arange(len(table), dtype=int)
    invalid_coordinates = np.ma.getmaskarray(output["ra"]) | np.ma.getmaskarray(
        output["dec"]
    )
    output = output[~invalid_coordinates]
    output.meta.update(deepcopy(table.meta))
    output.meta["catalog_name"] = normalized_name
    output.meta["normalized"] = True
    output.meta["normalization_version"] = 2
    output.meta["source_columns"] = selected
    return output


def _catalog_query_center(center, object_name, ccd, metadata):
    """Resolve the catalog-query center without requiring name resolution."""

    if isinstance(center, SkyCoord):
        return center.icrs
    if center is not None:
        if isinstance(center, str):
            return center
        try:
            return SkyCoord(center[0], center[1], unit="deg", frame="icrs")
        except (TypeError, ValueError, IndexError):
            pass
    if metadata is not None:
        for ra_name, dec_name in (
            ("adopted_ra_deg", "adopted_dec_deg"),
            ("header_target_ra_deg", "header_target_dec_deg"),
            ("wcs_center_ra_deg", "wcs_center_dec_deg"),
        ):
            ra = metadata.get(ra_name)
            dec = metadata.get(dec_name)
            if ra is not None and dec is not None:
                return SkyCoord(float(ra), float(dec), unit="deg", frame="icrs")
    wcs = None if ccd is None else getattr(ccd, "wcs", None)
    if wcs is not None and wcs.has_celestial:
        ny, nx = ccd.shape
        return wcs.pixel_to_world((nx - 1) / 2.0, (ny - 1) / 2.0).icrs
    if object_name:
        return str(object_name)
    raise ValueError("A coordinate, usable WCS, or object name is required")


def _catalog_cache_label(object_name, center, ccd, metadata, use_object_name):
    """Return an object or coordinate label that avoids cache collisions."""

    if use_object_name and object_name:
        return str(object_name)
    try:
        resolved = _catalog_query_center(center, None, ccd, metadata)
    except ValueError:
        return "field"
    if isinstance(resolved, SkyCoord):
        return "field_{:.6f}_{:+.6f}".format(resolved.ra.deg, resolved.dec.deg)
    return str(resolved)


def query_catalog(
    object_name=None,
    catalog_name=None,
    center=None,
    ccd=None,
    metadata=None,
    settings=None,
    save=None,
):
    """Query, normalize, cache, and optionally save an astrometric catalog.

    Gaia, PS1, SDSS, APASS, and SkyMapper are queried through VizieR. A local
    user catalog is loaded when ``catalogs.local_catalog_path`` is configured.
    The object name is used for the cache filename and, when no coordinates or
    usable WCS are supplied, as the remote name-resolution query center.

    Returns
    -------
    catalog : astropy.table.Table
        Normalized catalog.
    info : dict
        Query service, cache use, catalog ID, center, and saved path.
    """

    if settings is None:
        settings = get_default_settings()
    catalog_settings = settings.get("catalogs", {})
    if object_name is None and metadata is not None:
        object_name = metadata.get("object")
    catalog_name = normalize_catalog_name(
        catalog_name
        or catalog_settings.get("astrometry_catalog")
        or settings.get("astrometry", {}).get("catalog")
        or "gaia"
    )
    radius = float(catalog_settings.get("search_radius_arcmin", 10.0))
    use_object_name = bool(catalog_settings.get("use_object_name", True))
    cache_label = _catalog_cache_label(
        object_name, center, ccd, metadata, use_object_name
    )
    cache_path = catalog_cache_path(
        cache_label,
        catalog_name,
        settings,
        radius,
    )
    info = {
        "catalog_name": catalog_name,
        "catalog_id": None,
        "object_name": object_name,
        "radius_arcmin": radius,
        "cache_path": str(cache_path),
        "loaded_from_cache": False,
        "queried": False,
        "saved": False,
        "row_count": 0,
        "query_center": None,
    }

    use_cache = catalog_settings.get("cache_enabled", True)
    force = catalog_settings.get("force_new_query", False)
    if use_cache and cache_path.exists() and not force:
        catalog = Table.read(cache_path)
        if int(catalog.meta.get("normalization_version", 0)) >= 2:
            info["loaded_from_cache"] = True
            info["row_count"] = len(catalog)
            return catalog, info

    local_path = catalog_settings.get("local_catalog_path")
    if catalog_name == "user" or local_path is not None:
        if local_path is None:
            raise ValueError(
                "catalogs.local_catalog_path is required for a user catalog"
            )
        raw = Table.read(Path(local_path))
        catalog = normalize_catalog(
            raw,
            "user",
            column_map=catalog_settings.get("user_column_map", {}),
        )
        info["catalog_id"] = str(local_path)
    else:
        if catalog_settings.get("query_service", "vizier") != "vizier":
            raise ValueError("Only the 'vizier' query service is supported")
        try:
            from astroquery.vizier import Vizier
        except ImportError as error:
            raise ImportError(
                "Catalog queries require astroquery; install redphot dependencies"
            ) from error

        catalog_id = catalog_settings.get("catalog_ids", {}).get(catalog_name)
        if catalog_id is None:
            raise ValueError("No VizieR catalog ID configured for {}".format(catalog_name))
        query_center = _catalog_query_center(
            center,
            object_name if use_object_name else None,
            ccd,
            metadata,
        )
        info["catalog_id"] = catalog_id
        info["query_center"] = (
            query_center.to_string("decimal")
            if isinstance(query_center, SkyCoord)
            else str(query_center)
        )
        vizier = Vizier(
            columns=["**"],
            row_limit=int(catalog_settings.get("row_limit", -1)),
            timeout=float(catalog_settings.get("query_timeout_seconds", 120)),
        )
        tables = vizier.query_region(
            query_center,
            radius=radius * u.arcmin,
            catalog=catalog_id,
            cache=not force,
        )
        if not tables:
            raise RuntimeError(
                "No {} sources were returned around {}".format(
                    catalog_name, object_name or info["query_center"]
                )
            )
        raw = tables[0]
        raw.meta["vizier_catalog_id"] = catalog_id
        catalog = normalize_catalog(raw, catalog_name)
        info["queried"] = True

    if save is None:
        save = catalog_settings.get("save_catalog", True)
    if save or use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        output_format = (
            "ascii.ecsv"
            if catalog_settings.get("cache_format", "ecsv") == "ecsv"
            else "fits"
        )
        catalog.write(cache_path, format=output_format, overwrite=True)
        info["saved"] = True
    info["row_count"] = len(catalog)
    return catalog, info


def _observation_time(metadata, observation_time=None):
    """Return an Astropy UTC observation time from an explicit value or metadata."""

    if observation_time is not None:
        return observation_time if isinstance(observation_time, Time) else Time(observation_time)
    if metadata is None:
        return None
    for name in ("mjd_mid", "mjd_utc", "mjd"):
        value = metadata.get(name)
        if value is not None:
            return Time(float(value), format="mjd", scale="utc")
    return None


def propagate_catalog_to_epoch(catalog, metadata=None, observation_time=None, settings=None):
    """Propagate Gaia coordinates to the image epoch using Astropy space motion.

    Rows without finite proper motions retain their reference coordinates.
    All catalogs receive ``ra_epoch`` and ``dec_epoch`` columns so downstream
    projection and matching have a uniform interface.
    """

    if settings is None:
        settings = get_default_settings()
    output = Table(catalog, masked=True, copy=True)
    ra = np.asarray(output["ra"].filled(np.nan), dtype=float)
    dec = np.asarray(output["dec"].filled(np.nan), dtype=float)
    ra_epoch = np.array(ra, copy=True)
    dec_epoch = np.array(dec, copy=True)
    epoch = _observation_time(metadata, observation_time)
    catalog_name = normalize_catalog_name(output.meta.get("catalog_name"))
    propagate = (
        catalog_name == "gaia"
        and settings.get("catalogs", {}).get("propagate_gaia_proper_motion", True)
        and epoch is not None
    )
    propagated = np.zeros(len(output), dtype=bool)
    if propagate and len(output):
        pmra = np.asarray(output["pmra"].filled(np.nan), dtype=float)
        pmdec = np.asarray(output["pmdec"].filled(np.nan), dtype=float)
        ref_epoch = np.asarray(output["ref_epoch"].filled(np.nan), dtype=float)
        ref_epoch[~np.isfinite(ref_epoch)] = 2016.0
        valid = (
            np.isfinite(ra)
            & np.isfinite(dec)
            & np.isfinite(pmra)
            & np.isfinite(pmdec)
        )
        if np.any(valid):
            coordinates = SkyCoord(
                ra=ra[valid] * u.deg,
                dec=dec[valid] * u.deg,
                pm_ra_cosdec=pmra[valid] * u.mas / u.yr,
                pm_dec=pmdec[valid] * u.mas / u.yr,
                obstime=Time(ref_epoch[valid], format="jyear", scale="tcb"),
                frame="icrs",
            )
            # Gaia rows often lack a useful parallax/radial velocity. ERFA then
            # assumes a harmless large distance and emits one warning per row.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                moved = coordinates.apply_space_motion(new_obstime=epoch)
            ra_epoch[valid] = moved.ra.to_value(u.deg)
            dec_epoch[valid] = moved.dec.to_value(u.deg)
            propagated[valid] = True

    output["ra_epoch"] = ra_epoch * u.deg
    output["dec_epoch"] = dec_epoch * u.deg
    output["proper_motion_applied"] = propagated
    output.meta["observation_epoch_mjd"] = None if epoch is None else float(epoch.mjd)
    output.meta["proper_motion_count"] = int(np.count_nonzero(propagated))
    return output


def catalog_coordinates(catalog):
    """Return epoch-corrected catalog coordinates as an ICRS SkyCoord."""

    ra_name = "ra_epoch" if "ra_epoch" in catalog.colnames else "ra"
    dec_name = "dec_epoch" if "dec_epoch" in catalog.colnames else "dec"
    return SkyCoord(
        np.asarray(catalog[ra_name], dtype=float) * u.deg,
        np.asarray(catalog[dec_name], dtype=float) * u.deg,
        frame="icrs",
    )


def _plain_icrs(coordinates):
    """Return ICRS positions without frame attributes such as observation time."""

    icrs = coordinates.icrs
    return SkyCoord(icrs.ra, icrs.dec, frame="icrs")


def project_catalog_to_image(catalog, wcs, shape=None):
    """Project normalized catalog coordinates through the selected image WCS."""

    if wcs is None or not wcs.has_celestial:
        raise ValueError("A celestial WCS is required to project a catalog")
    output = Table(catalog, masked=True, copy=True)
    coordinates = catalog_coordinates(output)
    x, y = wcs.world_to_pixel(coordinates)
    output["x"] = np.asarray(x, dtype=float) * u.pixel
    output["y"] = np.asarray(y, dtype=float) * u.pixel
    finite = np.isfinite(x) & np.isfinite(y)
    if shape is None:
        in_image = finite
    else:
        ny, nx = shape
        in_image = finite & (x >= 0) & (x <= nx - 1) & (y >= 0) & (y <= ny - 1)
    output["in_image"] = in_image
    return output


def _unique_nearest_matches(detected_coordinates, catalog_coordinate, maximum_separation):
    """Return one-to-one nearest-neighbor match indices within a radius."""

    if len(detected_coordinates) == 0 or len(catalog_coordinate) == 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([]) * u.arcsec
    catalog_index, separation, _ = detected_coordinates.match_to_catalog_sky(
        catalog_coordinate
    )
    candidates = np.flatnonzero(separation <= maximum_separation)
    candidates = candidates[np.argsort(separation[candidates])]
    kept_detected = []
    kept_catalog = []
    used_catalog = set()
    for detected_index in candidates:
        matched_catalog = int(catalog_index[detected_index])
        if matched_catalog in used_catalog:
            continue
        kept_detected.append(int(detected_index))
        kept_catalog.append(matched_catalog)
        used_catalog.add(matched_catalog)
    kept_detected = np.asarray(kept_detected, dtype=int)
    kept_catalog = np.asarray(kept_catalog, dtype=int)
    return kept_detected, kept_catalog, separation[kept_detected]


def match_catalog_sources(sources, catalog, wcs, settings=None):
    """Match detected image sources to an epoch-corrected normalized catalog."""

    if settings is None:
        settings = get_default_settings()
    if wcs is None or not wcs.has_celestial:
        raise ValueError("A celestial WCS is required for source matching")
    x = np.asarray(sources["x"], dtype=float)
    y = np.asarray(sources["y"], dtype=float)
    detected_coordinates = _plain_icrs(wcs.pixel_to_world(x, y))
    catalog_coordinate = catalog_coordinates(catalog)
    maximum = float(
        settings.get("astrometry", {}).get(
            "maximum_match_separation_arcsec", 5.0
        )
    ) * u.arcsec
    detected_index, catalog_index, separation = _unique_nearest_matches(
        detected_coordinates, catalog_coordinate, maximum
    )

    matches = Table(masked=True)
    matches["source_index"] = detected_index
    matches["catalog_index"] = catalog_index
    matches["source_label"] = (
        np.asarray(sources["label"], dtype=int)[detected_index]
        if "label" in sources.colnames
        else detected_index
    )
    matches["x"] = x[detected_index] * u.pixel
    matches["y"] = y[detected_index] * u.pixel
    projected = project_catalog_to_image(catalog, wcs)
    matches["catalog_x_original"] = (
        np.asarray(projected["x"], dtype=float)[catalog_index] * u.pixel
    )
    matches["catalog_y_original"] = (
        np.asarray(projected["y"], dtype=float)[catalog_index] * u.pixel
    )
    matches["catalog_ra"] = catalog_coordinate.ra.deg[catalog_index] * u.deg
    matches["catalog_dec"] = catalog_coordinate.dec.deg[catalog_index] * u.deg
    matches["separation_original_arcsec"] = separation.to_value(u.arcsec) * u.arcsec
    longitude, latitude = detected_coordinates[detected_index].spherical_offsets_to(
        catalog_coordinate[catalog_index]
    )
    matches["residual_ra_original_arcsec"] = longitude.to_value(u.arcsec) * u.arcsec
    matches["residual_dec_original_arcsec"] = latitude.to_value(u.arcsec) * u.arcsec
    matches["inlier"] = np.ones(len(matches), dtype=bool)
    return matches


def _rms(values):
    """Return a finite root-mean-square value or None."""

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.sqrt(np.mean(values ** 2))) if values.size else None


def _fit_similarity(x, y, target_x, target_y, settings):
    """Fit the configured similarity transform from detections to catalog pixels."""

    astrometry = settings.get("astrometry", {})
    fit_rotation = astrometry.get("fit_rotation", True)
    fit_scale = astrometry.get("fit_scale", True)
    fit_translation = astrometry.get("fit_translation", True)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    target_x = np.asarray(target_x, dtype=float)
    target_y = np.asarray(target_y, dtype=float)

    if fit_rotation or fit_scale:
        design = np.zeros((2 * len(x), 4), dtype=float)
        values = np.empty(2 * len(x), dtype=float)
        design[0::2, 0] = x
        design[0::2, 1] = -y
        design[0::2, 2] = 1.0
        design[1::2, 0] = y
        design[1::2, 1] = x
        design[1::2, 3] = 1.0
        values[0::2] = target_x
        values[1::2] = target_y
        a, b, tx, ty = np.linalg.lstsq(design, values, rcond=None)[0]
        scale = float(np.hypot(a, b))
        angle = float(np.arctan2(b, a))
        if not fit_rotation:
            angle = 0.0
        if not fit_scale:
            scale = 1.0
        a = scale * np.cos(angle)
        b = scale * np.sin(angle)
    else:
        a, b = 1.0, 0.0
        tx, ty = 0.0, 0.0

    transform = np.array([[a, -b], [b, a]], dtype=float)
    if fit_translation:
        transformed = transform @ np.vstack((x, y))
        tx = float(np.mean(target_x - transformed[0]))
        ty = float(np.mean(target_y - transformed[1]))
    else:
        tx, ty = 0.0, 0.0
    return transform, np.array([tx, ty]), scale, np.rad2deg(angle)


def _compose_similarity_wcs(original_wcs, transform, translation):
    """Compose a pixel-space similarity correction with an existing WCS."""

    refined = deepcopy(original_wcs.celestial)
    old_matrix = np.asarray(refined.pixel_scale_matrix, dtype=float)
    old_crpix = np.asarray(refined.wcs.crpix, dtype=float)
    new_matrix = old_matrix @ transform
    constant = old_matrix @ (translation + 1.0 - old_crpix)
    new_crpix = 1.0 - np.linalg.solve(new_matrix, constant)
    refined.wcs.cd = new_matrix
    refined.wcs.crpix = new_crpix
    refined.wcs.set()
    return refined


def _residuals_for_wcs(matches, wcs):
    """Return RA, Dec, and radial residuals for a match table and WCS."""

    detected = _plain_icrs(wcs.pixel_to_world(
        np.asarray(matches["x"], dtype=float),
        np.asarray(matches["y"], dtype=float),
    ))
    catalog = SkyCoord(
        np.asarray(matches["catalog_ra"], dtype=float) * u.deg,
        np.asarray(matches["catalog_dec"], dtype=float) * u.deg,
        frame="icrs",
    )
    longitude, latitude = detected.spherical_offsets_to(catalog)
    radial = detected.separation(catalog)
    return (
        longitude.to_value(u.arcsec),
        latitude.to_value(u.arcsec),
        radial.to_value(u.arcsec),
    )


def _sigma_clip_inliers(radial, settings):
    """Return a robust residual inlier mask."""

    radial = np.asarray(radial, dtype=float)
    finite = np.isfinite(radial)
    if np.count_nonzero(finite) < 3:
        return finite
    values = radial[finite]
    target_rms = float(
        settings.get("astrometry", {}).get("target_rms_arcsec", 0.5)
    )
    if np.max(values) <= target_rms:
        return finite
    median = np.median(values)
    scatter = 1.4826 * np.median(np.abs(values - median))
    if not np.isfinite(scatter) or scatter <= 0:
        return finite
    threshold = median + float(
        settings.get("astrometry", {}).get("sigma_clip", 4.0)
    ) * scatter
    return finite & (radial <= threshold)


def _target_coordinate(target, metadata, settings):
    """Resolve the target coordinate for WCS projection checks."""

    if isinstance(target, SkyCoord):
        return target.icrs
    if target is not None:
        try:
            return SkyCoord(target[0], target[1], unit="deg", frame="icrs")
        except (TypeError, ValueError, IndexError):
            pass
    target_settings = settings.get("target_position", {})
    if target_settings.get("ra") is not None and target_settings.get("dec") is not None:
        units = target_settings.get("coordinate_unit", ["hourangle", "deg"])
        return SkyCoord(
            target_settings["ra"], target_settings["dec"], unit=units, frame="icrs"
        )
    if metadata is not None:
        ra = metadata.get("adopted_ra_deg")
        dec = metadata.get("adopted_dec_deg")
        if ra is not None and dec is not None:
            return SkyCoord(float(ra), float(dec), unit="deg", frame="icrs")
    return None


def _target_projection_record(wcs, target, shape):
    """Measure target pixel projection and round-trip WCS consistency."""

    if wcs is None or target is None:
        return None
    x, y = wcs.world_to_pixel(target)
    round_trip = wcs.pixel_to_world(x, y).icrs
    ny, nx = shape
    return {
        "x": float(x),
        "y": float(y),
        "inside_image": bool(0 <= x < nx and 0 <= y < ny),
        "round_trip_error_arcsec": float(target.separation(round_trip).arcsec),
    }


def plate_solve_with_astrometry_net(ccd, metadata=None, settings=None):
    """Run the local Astrometry.net ``solve-field`` command as a fallback.

    A temporary ordinary FITS file is written because compressed or extension
    based input formats are not uniformly supported by external solvers. The
    temporary directory is removed automatically. Only the solved WCS is
    returned; neither the input FITS file nor the in-memory science pixels are
    changed.

    Raises
    ------
    FileNotFoundError
        The configured ``solve-field`` executable is unavailable.
    RuntimeError
        The solver exits unsuccessfully or does not create a celestial WCS.
    subprocess.TimeoutExpired
        The configured timeout is exceeded.
    """

    if settings is None:
        settings = get_default_settings()
    astrometry = settings.get("astrometry", {})
    command = str(astrometry.get("plate_solver_command", "solve-field"))
    executable = shutil.which(command)
    if executable is None and Path(command).is_file():
        executable = str(Path(command).resolve())
    if executable is None:
        raise FileNotFoundError(
            "Astrometry.net executable not found: {}".format(command)
        )

    with tempfile.TemporaryDirectory(prefix="redphot_astrometry_") as directory:
        directory = Path(directory)
        input_path = directory / "input.fits"
        solved_path = directory / "solved.fits"
        header = fits.Header()
        input_wcs = getattr(ccd, "wcs", None)
        if input_wcs is not None:
            header.update(input_wcs.to_header(relax=True))
        fits.PrimaryHDU(
            data=np.asarray(ccd.data),
            header=header,
        ).writeto(input_path, overwrite=True)

        arguments = [
            executable,
            "--dir",
            str(directory),
            "--new-fits",
            str(solved_path),
            "--overwrite",
            "--no-plots",
        ]
        pixel_scale = None if metadata is None else metadata.get("pixel_scale")
        if pixel_scale is not None and np.isfinite(float(pixel_scale)):
            tolerance = float(
                astrometry.get("plate_solver_scale_tolerance_fraction", 0.20)
            )
            scale = float(pixel_scale)
            arguments.extend(
                [
                    "--scale-units",
                    "arcsecperpix",
                    "--scale-low",
                    str(scale * (1.0 - tolerance)),
                    "--scale-high",
                    str(scale * (1.0 + tolerance)),
                ]
            )
        target = _target_coordinate(None, metadata, settings)
        if target is not None:
            arguments.extend(
                [
                    "--ra",
                    str(target.ra.deg),
                    "--dec",
                    str(target.dec.deg),
                    "--radius",
                    str(astrometry.get("plate_solver_search_radius_degrees", 2.0)),
                ]
            )
        arguments.append(str(input_path))

        completed = subprocess.run(
            arguments,
            check=False,
            capture_output=True,
            text=True,
            timeout=float(astrometry.get("plate_solver_timeout_seconds", 300)),
        )
        if completed.returncode != 0 or not solved_path.exists():
            message = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(
                "Astrometry.net plate solve failed: {}".format(
                    message[-1000:] if message else "no solved FITS was produced"
                )
            )
        solved_wcs = WCS(fits.getheader(solved_path)).celestial
        if not solved_wcs.has_celestial:
            raise RuntimeError("Astrometry.net output does not contain a celestial WCS")
        return solved_wcs


def refine_wcs_from_matches(wcs, matches, settings=None, shape=None, target=None):
    """Verify and conservatively refine a WCS from matched sources.

    Iterative robust clipping rejects mismatches. The candidate solution is
    adopted only when enough matches remain, it improves the absolute RMS by
    the configured amount, and translation/rotation/scale changes remain
    within safety limits.

    Returns
    -------
    refined_wcs : astropy.wcs.WCS
        Adopted refined WCS, or an independent copy of the original.
    matches : astropy.table.Table
        Match table with inlier and final residual columns.
    info : dict
        Match counts, RMS values, fitted changes, flags, and WCS headers.
    """

    if settings is None:
        settings = get_default_settings()
    astrometry = settings.get("astrometry", {})
    original = deepcopy(wcs.celestial)
    matches = Table(matches, masked=True, copy=True)
    minimum = int(astrometry.get("minimum_matches", 6))
    original_radial = np.asarray(
        matches["separation_original_arcsec"], dtype=float
    )
    original_rms = _rms(original_radial)
    info = {
        "match_count": len(matches),
        "inlier_count": 0,
        "rejected_match_count": 0,
        "original_rms_arcsec": original_rms,
        "refined_rms_arcsec": None,
        "improvement_fraction": None,
        "translation_x_pixels": None,
        "translation_y_pixels": None,
        "translation_pixels": None,
        "rotation_degrees": None,
        "scale": None,
        "scale_change_fraction": None,
        "refinement_attempted": False,
        "refinement_adopted": False,
        "refinement_reason": None,
        "quality_status": "PASS",
        "flags": [],
        "original_wcs_header": original.to_header(relax=True),
        "refined_wcs_header": original.to_header(relax=True),
        "target_original": None,
        "target_refined": None,
    }
    if shape is not None:
        info["target_original"] = _target_projection_record(original, target, shape)

    if len(matches) < minimum:
        info["refinement_reason"] = "too_few_matches"
        info["quality_status"] = "FAIL"
        info["flags"].append("WCS_TOO_FEW_MATCHES")
        matches["inlier"] = np.zeros(len(matches), dtype=bool)
        return original, matches, info

    target_rms = float(astrometry.get("target_rms_arcsec", 0.5))
    if not astrometry.get("refine_wcs", True) or (
        original_rms is not None and original_rms <= target_rms
    ):
        info["refinement_reason"] = (
            "disabled" if not astrometry.get("refine_wcs", True) else "already_good"
        )
        ra_residual, dec_residual, radial = _residuals_for_wcs(matches, original)
        inliers = _sigma_clip_inliers(radial, settings)
        matches["inlier"] = inliers
        matches["residual_ra_final_arcsec"] = ra_residual * u.arcsec
        matches["residual_dec_final_arcsec"] = dec_residual * u.arcsec
        matches["separation_final_arcsec"] = radial * u.arcsec
        info["inlier_count"] = int(np.count_nonzero(inliers))
        info["rejected_match_count"] = int(len(matches) - np.count_nonzero(inliers))
        info["refined_rms_arcsec"] = _rms(radial[inliers])
        if shape is not None:
            info["target_refined"] = _target_projection_record(original, target, shape)
        return original, matches, info

    info["refinement_attempted"] = True
    x = np.asarray(matches["x"], dtype=float)
    y = np.asarray(matches["y"], dtype=float)
    target_x = np.asarray(matches["catalog_x_original"], dtype=float)
    target_y = np.asarray(matches["catalog_y_original"], dtype=float)
    inliers = np.ones(len(matches), dtype=bool)
    candidate = original
    transform = np.eye(2)
    translation = np.zeros(2)
    scale = 1.0
    rotation = 0.0
    for _ in range(int(astrometry.get("maximum_iterations", 3))):
        if np.count_nonzero(inliers) < minimum:
            break
        transform, translation, scale, rotation = _fit_similarity(
            x[inliers], y[inliers], target_x[inliers], target_y[inliers], settings
        )
        candidate = _compose_similarity_wcs(original, transform, translation)
        _, _, radial = _residuals_for_wcs(matches, candidate)
        updated = _sigma_clip_inliers(radial, settings)
        if np.array_equal(updated, inliers):
            break
        inliers = updated

    ra_residual, dec_residual, radial = _residuals_for_wcs(matches, candidate)
    refined_rms = _rms(radial[inliers])
    translation_size = float(np.hypot(*translation))
    scale_change = abs(float(scale) - 1.0)
    improvement = (
        None
        if original_rms in {None, 0.0} or refined_rms is None
        else 1.0 - refined_rms / original_rms
    )
    safe = (
        translation_size
        <= float(astrometry.get("maximum_translation_pixels", 50.0))
        and abs(rotation)
        <= float(astrometry.get("maximum_rotation_degrees", 10.0))
        and scale_change
        <= float(astrometry.get("maximum_scale_change_fraction", 0.10))
    )
    improved = (
        improvement is not None
        and improvement
        >= float(astrometry.get("minimum_improvement_fraction", 0.05))
    )
    enough = np.count_nonzero(inliers) >= minimum
    adopted = enough and improved and (
        safe or not astrometry.get("reject_unsafe_solution", True)
    )

    final_wcs = candidate if adopted else original
    if not adopted:
        ra_residual, dec_residual, radial = _residuals_for_wcs(matches, original)
        inliers = _sigma_clip_inliers(radial, settings)
        info["refinement_reason"] = (
            "too_few_inliers"
            if not enough
            else "unsafe_solution"
            if not safe
            else "insufficient_improvement"
        )
        info["flags"].append("WCS_REFINEMENT_FAILED")
        info["quality_status"] = "WARN"
    else:
        info["refinement_reason"] = "improved"

    matches["inlier"] = inliers
    matches["residual_ra_final_arcsec"] = ra_residual * u.arcsec
    matches["residual_dec_final_arcsec"] = dec_residual * u.arcsec
    matches["separation_final_arcsec"] = radial * u.arcsec
    final_rms = _rms(radial[inliers])
    info.update(
        {
            "inlier_count": int(np.count_nonzero(inliers)),
            "rejected_match_count": int(len(matches) - np.count_nonzero(inliers)),
            "refined_rms_arcsec": final_rms,
            "improvement_fraction": improvement,
            "translation_x_pixels": float(translation[0]),
            "translation_y_pixels": float(translation[1]),
            "translation_pixels": translation_size,
            "rotation_degrees": float(rotation),
            "scale": float(scale),
            "scale_change_fraction": scale_change,
            "refinement_adopted": adopted,
            "refined_wcs_header": final_wcs.to_header(relax=True),
        }
    )
    warning_rms = float(astrometry.get("warning_rms_arcsec", 1.0))
    if final_rms is None or final_rms > warning_rms:
        info["quality_status"] = "WARN" if final_rms is not None else "FAIL"
        info["flags"].append("WCS_POOR")
    if shape is not None:
        info["target_refined"] = _target_projection_record(final_wcs, target, shape)
    return final_wcs, matches, info


def _attempt_plate_solve(ccd, metadata, settings, plate_solver):
    """Call a user solver or the configured local Astrometry.net fallback."""

    solver = plate_solver or plate_solve_with_astrometry_net
    try:
        solved = solver(ccd=ccd, metadata=metadata, settings=settings)
    except (
        FileNotFoundError,
        OSError,
        RuntimeError,
        ValueError,
        subprocess.TimeoutExpired,
    ) as error:
        return None, str(error)
    if solved is None or not getattr(solved, "has_celestial", False):
        return None, "solver did not return a celestial WCS"
    return solved.celestial, None


def solve_astrometry(
    ccd,
    sources,
    metadata=None,
    settings=None,
    catalog=None,
    object_name=None,
    center=None,
    target=None,
    plate_solver=None,
):
    """Run catalog acquisition, epoch propagation, matching, and WCS refinement.

    ``plate_solver`` may be an optional function accepting ``ccd``, ``metadata``,
    and ``settings`` and returning an Astropy WCS. Otherwise the configured
    local Astrometry.net ``solve-field`` executable is used. A solver is called
    only when ``astrometry.plate_solve_fallback`` is enabled and the existing
    WCS is missing or remains unusable.

    Returns
    -------
    catalog : astropy.table.Table
        Normalized, epoch-corrected, and image-projected catalog.
    matches : astropy.table.Table
        Matched detections and residuals.
    refined_wcs : astropy.wcs.WCS or None
        Adopted derived WCS. Science data are unchanged.
    info : dict
        Query, matching, fitting, fallback, and target-projection diagnostics.
    """

    if settings is None:
        settings = get_default_settings()
    astrometry = settings.get("astrometry", {})
    if object_name is None and metadata is not None:
        object_name = metadata.get("object")
    query_info = None
    if catalog is None:
        catalog, query_info = query_catalog(
            object_name=object_name,
            center=center,
            ccd=ccd,
            metadata=metadata,
            settings=settings,
        )
    else:
        catalog = normalize_catalog(
            catalog,
            catalog.meta.get("catalog_name", "user"),
            settings.get("catalogs", {}).get("user_column_map", {}),
        ) if not catalog.meta.get("normalized", False) else Table(catalog, copy=True)

    catalog = propagate_catalog_to_epoch(catalog, metadata=metadata, settings=settings)
    original_wcs = getattr(ccd, "wcs", None)
    input_wcs_header = (
        None
        if original_wcs is None
        else original_wcs.celestial.to_header(relax=True)
    )
    working_wcs = deepcopy(original_wcs.celestial) if original_wcs is not None else None
    fallback = {
        "configured": bool(astrometry.get("plate_solve_fallback", False)),
        "needed": working_wcs is None,
        "attempted": False,
        "succeeded": False,
        "reason": None,
    }

    if working_wcs is None and fallback["configured"]:
        fallback["attempted"] = True
        working_wcs, fallback["reason"] = _attempt_plate_solve(
            ccd, metadata, settings, plate_solver
        )
        fallback["succeeded"] = working_wcs is not None

    empty_matches = Table(masked=True)
    if working_wcs is None or not working_wcs.has_celestial:
        return catalog, empty_matches, None, {
            "query": query_info,
            "fallback": fallback,
            "input_wcs_header": input_wcs_header,
            "quality_status": "FAIL",
            "flags": ["WCS_MISSING"],
            "match_count": 0,
        }

    projected = project_catalog_to_image(catalog, working_wcs, ccd.shape)
    matches = match_catalog_sources(sources, projected, working_wcs, settings)
    target_coordinate = _target_coordinate(target, metadata, settings)
    refined_wcs, matches, fit_info = refine_wcs_from_matches(
        working_wcs,
        matches,
        settings=settings,
        shape=ccd.shape,
        target=target_coordinate,
    )

    poor = (
        fit_info.get("quality_status") == "FAIL"
        or fit_info.get("refined_rms_arcsec") is None
        or fit_info.get("refined_rms_arcsec", np.inf)
        > float(astrometry.get("warning_rms_arcsec", 1.0))
    )
    fallback["needed"] = bool(poor)
    if poor and fallback["configured"] and not fallback["attempted"]:
        fallback["attempted"] = True
        solved_wcs, fallback["reason"] = _attempt_plate_solve(
            ccd, metadata, settings, plate_solver
        )
        if solved_wcs is not None:
            fallback["succeeded"] = True
            projected = project_catalog_to_image(catalog, solved_wcs, ccd.shape)
            matches = match_catalog_sources(sources, projected, solved_wcs, settings)
            refined_wcs, matches, fit_info = refine_wcs_from_matches(
                solved_wcs,
                matches,
                settings=settings,
                shape=ccd.shape,
                target=target_coordinate,
            )

    final_catalog = project_catalog_to_image(catalog, refined_wcs, ccd.shape)
    info = dict(fit_info)
    info["input_wcs_header"] = input_wcs_header
    info["query"] = query_info
    info["fallback"] = fallback
    info["catalog_name"] = catalog.meta.get("catalog_name")
    info["catalog_row_count"] = len(catalog)
    info["catalog_in_image_count"] = int(np.count_nonzero(final_catalog["in_image"]))
    return final_catalog, matches, refined_wcs, info


def save_astrometry_products(
    matches,
    info,
    output_directory,
    filename,
    settings=None,
    overwrite=None,
):
    """Save the match table and derived WCS-only FITS header."""

    if settings is None:
        settings = get_default_settings()
    if overwrite is None:
        overwrite = settings.get("output", {}).get("overwrite", False)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    stem = Path(filename).name
    for ending in (".fits.fz", ".fits.gz", ".fits", ".fit", ".fts"):
        if stem.lower().endswith(ending):
            stem = stem[: -len(ending)]
            break
    paths = {}
    astrometry = settings.get("astrometry", {})
    if astrometry.get("save_match_table", True) and len(matches):
        path = output_directory / "{}_astrometry_matches.ecsv".format(stem)
        matches.write(path, format="ascii.ecsv", overwrite=bool(overwrite))
        paths["matches"] = str(path)
    header = info.get("refined_wcs_header")
    if astrometry.get("save_refined_header", True) and header is not None:
        path = output_directory / "{}_refined_wcs.fits".format(stem)
        fits.PrimaryHDU(header=header).writeto(path, overwrite=bool(overwrite))
        paths["refined_wcs"] = str(path)
    return paths


# ---------------------------------------------------------------------------
# Persistent source tables and role-based star selection
# ---------------------------------------------------------------------------


ROLE_NAMES = (
    "astrometry",
    "psf",
    "calibration",
    "ensemble",
    "qc_anchor",
)


def _optional_value(table, column, index, default=None):
    """Return one unmasked table value or a default."""

    if column not in table.colnames:
        return default
    value = table[column][index]
    if np.ma.is_masked(value):
        return default
    if isinstance(value, np.generic):
        return value.item()
    return value


def _source_identity(catalog, index):
    """Return the catalog identity and reference coordinate for one row."""

    catalog_name = normalize_catalog_name(catalog.meta.get("catalog_name")) or "catalog"
    source_id = _optional_value(catalog, "source_id", index)
    ra = float(_optional_value(catalog, "ra", index, np.nan))
    dec = float(_optional_value(catalog, "dec", index, np.nan))
    identity = None
    if source_id is not None and str(source_id).strip():
        identity = "{}:{}".format(catalog_name, str(source_id).strip())
    return identity, ra, dec


def _assign_persistent_id(
    catalog,
    index,
    identity_map,
    persistent_ids,
    persistent_coordinates,
    tolerance_arcsec,
):
    """Assign a stable ID, cross-identifying different catalogs by position."""

    identity, ra, dec = _source_identity(catalog, index)
    if identity is not None and identity in identity_map:
        return identity_map[identity]

    persistent_id = None
    if persistent_coordinates and np.isfinite(ra) and np.isfinite(dec):
        coordinate = SkyCoord(ra, dec, unit="deg", frame="icrs")
        known = SkyCoord(
            [item[0] for item in persistent_coordinates] * u.deg,
            [item[1] for item in persistent_coordinates] * u.deg,
            frame="icrs",
        )
        nearest, separation, _ = coordinate.match_to_catalog_sky(known)
        if separation.arcsec <= tolerance_arcsec:
            persistent_id = persistent_ids[int(nearest)]

    if persistent_id is None:
        if identity is not None:
            persistent_id = identity
        else:
            persistent_id = "sky:{:.7f}:{:+.7f}".format(ra, dec)
        persistent_ids.append(persistent_id)
        persistent_coordinates.append((ra, dec))
    if identity is not None:
        identity_map[identity] = persistent_id
    return persistent_id


def _image_record_id(record, index):
    """Return the persistent identifier for an image record."""

    if record.get("image_id") is not None:
        return str(record["image_id"])
    metadata = record.get("metadata") or {}
    return str(metadata.get("filename") or "image_{:04d}".format(index))


def _record_shape(record):
    """Return ``(ny, nx)`` from an image record."""

    if record.get("shape") is not None:
        shape = tuple(record["shape"])
    elif record.get("ccd") is not None:
        shape = tuple(record["ccd"].shape)
    else:
        shape = None
    if shape is None or len(shape) != 2:
        raise ValueError("Every image record requires a 2D shape or CCDData")
    return int(shape[0]), int(shape[1])


def _distance_from_mask(mask):
    """Return the distance to the nearest true mask pixel."""

    if mask is None:
        return None
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return None
    from scipy.ndimage import distance_transform_edt

    return distance_transform_edt(~mask)


def _sample_distance(distance, x, y):
    """Sample a distance map at the nearest valid pixel."""

    if distance is None:
        return np.inf
    ix = int(np.clip(round(x), 0, distance.shape[1] - 1))
    iy = int(np.clip(round(y), 0, distance.shape[0] - 1))
    return float(distance[iy, ix])


def _mask_overlap(mask, x, y, radius):
    """Return whether a circular footprint overlaps a boolean mask."""

    if mask is None:
        return False
    mask = np.asarray(mask, dtype=bool)
    x0 = max(0, int(np.floor(x - radius)))
    x1 = min(mask.shape[1], int(np.ceil(x + radius)) + 1)
    y0 = max(0, int(np.floor(y - radius)))
    y1 = min(mask.shape[0], int(np.ceil(y + radius)) + 1)
    if x0 >= x1 or y0 >= y1:
        return True
    yy, xx = np.ogrid[y0:y1, x0:x1]
    circle = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
    return bool(np.any(mask[y0:y1, x0:x1] & circle))


def _nearest_source_distances(sources):
    """Return nearest-neighbor detector distances for a source table."""

    if len(sources) < 2:
        return np.full(len(sources), np.inf)
    from scipy.spatial import cKDTree

    positions = np.column_stack(
        (np.asarray(sources["x"], dtype=float), np.asarray(sources["y"], dtype=float))
    )
    distances, _ = cKDTree(positions).query(positions, k=2)
    return np.asarray(distances[:, 1], dtype=float)


def _catalog_magnitude(catalog, index, filter_name):
    """Return the most relevant catalog magnitude and uncertainty."""

    requested = "mag_{}".format(filter_name) if filter_name else None
    candidates = [requested, "mag_G", "mag_r", "mag_g", "mag_i"]
    for name in candidates:
        if name is None or name not in catalog.colnames:
            continue
        value = _optional_value(catalog, name, index)
        if value is None or not np.isfinite(float(value)):
            continue
        error_name = name.replace("mag_", "magerr_", 1)
        error = _optional_value(catalog, error_name, index)
        return (
            float(value),
            None if error is None else float(error),
            name.replace("mag_", "", 1),
        )
    return None, None, None


def _measurement_table(rows):
    """Convert per-image source dictionaries to a masked Astropy table."""

    string_fields = (
        "persistent_id",
        "image_id",
        "filter",
        "magnitude_band",
        "rejection_reasons",
    )
    integer_fields = ("source_index", "catalog_index", "source_label")
    boolean_fields = (
        "saturated",
        "near_edge",
        "masked",
        "trail_overlap",
        "image_accepted",
    ) + tuple("role_{}".format(role) for role in ROLE_NAMES)
    numeric_fields = (
        "x",
        "y",
        "flux",
        "snr",
        "fwhm_pixels",
        "fwhm_arcsec",
        "ellipticity",
        "catalog_separation_arcsec",
        "magnitude",
        "magnitude_error",
        "edge_distance_pixels",
        "saturation_distance_pixels",
        "trail_distance_pixels",
        "neighbor_distance_pixels",
        "detector_x_fraction",
        "detector_y_fraction",
        "image_fwhm_pixels",
    )
    table = Table(masked=True)
    for name in string_fields:
        table[name] = [str(row.get(name, "")) for row in rows]
    for name in integer_fields:
        table[name] = [int(row.get(name, 0)) for row in rows]
    for name in boolean_fields:
        table[name] = [bool(row.get(name, False)) for row in rows]
    for name in numeric_fields:
        values = np.asarray([row.get(name, np.nan) for row in rows], dtype=float)
        if name in {
            "saturation_distance_pixels",
            "trail_distance_pixels",
            "neighbor_distance_pixels",
        }:
            invalid = np.isnan(values)
        else:
            invalid = ~np.isfinite(values)
        table[name] = MaskedColumn(
            np.where(invalid, 0.0, values), mask=invalid, dtype=float
        )
    for name in (
        "x",
        "y",
        "edge_distance_pixels",
        "saturation_distance_pixels",
        "trail_distance_pixels",
        "neighbor_distance_pixels",
        "fwhm_pixels",
        "image_fwhm_pixels",
    ):
        table[name].unit = u.pixel
    table["fwhm_arcsec"].unit = u.arcsec
    table["catalog_separation_arcsec"].unit = u.arcsec
    table["magnitude"].unit = u.mag
    table["magnitude_error"].unit = u.mag
    return table


def build_master_source_table(image_records, settings=None):
    """Build persistent master and per-image source tables.

    Each image record supplies ``sources``, ``catalog``, ``matches``, and either
    ``shape`` or ``ccd``. Optional entries are ``metadata``, ``masks``, and
    ``quality``. Catalog identities are retained when possible; sources from
    different catalogs are cross-identified by their reference coordinates.

    Returns
    -------
    master : astropy.table.Table
        One row per persistent sky source.
    measurements : astropy.table.Table
        One row per source detection per image, linked by ``persistent_id``.
    """

    if settings is None:
        settings = get_default_settings()
    star_settings = settings.get("catalogs", {}).get("comparison_stars", {})
    tolerance = float(star_settings.get("persistent_match_arcsec", 0.5))
    identity_map = {}
    persistent_ids = []
    persistent_coordinates = []
    master_parts = []
    measurement_rows = []

    for image_index, record in enumerate(image_records):
        sources = record.get("sources")
        catalog = record.get("catalog")
        matches = record.get("matches")
        if sources is None or catalog is None or matches is None:
            raise ValueError(
                "Each image record requires sources, catalog, and matches tables"
            )
        image_id = _image_record_id(record, image_index)
        ny, nx = _record_shape(record)
        metadata = record.get("metadata") or {}
        image_filter = metadata.get("filter")
        quality = record.get("quality") or {}
        image_fwhm = quality.get("fwhm_pixels")
        masks = record.get("masks") or {}
        combined_mask = masks.get("combined")
        saturation_mask = masks.get("saturation")
        trail_mask = masks.get("trails")
        saturation_distance = _distance_from_mask(saturation_mask)
        trail_distance = _distance_from_mask(trail_mask)
        neighbor_distances = _nearest_source_distances(sources)

        catalog_indices = np.asarray(matches["catalog_index"], dtype=int)
        part = Table(catalog[catalog_indices], masked=True, copy=True)
        assigned_ids = []
        for catalog_index in catalog_indices:
            assigned_ids.append(
                _assign_persistent_id(
                    catalog,
                    int(catalog_index),
                    identity_map,
                    persistent_ids,
                    persistent_coordinates,
                    tolerance,
                )
            )
        part["persistent_id"] = assigned_ids
        for name in (
            "x",
            "y",
            "in_image",
            "ra_epoch",
            "dec_epoch",
            "proper_motion_applied",
        ):
            if name in part.colnames:
                part.remove_column(name)
        master_parts.append(part)

        source_indices = np.asarray(matches["source_index"], dtype=int)
        separations = (
            np.asarray(matches["separation_final_arcsec"], dtype=float)
            if "separation_final_arcsec" in matches.colnames
            else np.asarray(matches["separation_original_arcsec"], dtype=float)
        )
        for match_index, (source_index, catalog_index, persistent_id) in enumerate(
            zip(source_indices, catalog_indices, assigned_ids)
        ):
            x = float(sources["x"][source_index])
            y = float(sources["y"][source_index])
            source_fwhm = _optional_value(sources, "fwhm_pixels", source_index)
            working_fwhm = (
                float(image_fwhm)
                if image_fwhm is not None
                else float(source_fwhm)
                if source_fwhm is not None
                else 4.0
            )
            aperture_radius = max(1.0, working_fwhm)
            magnitude, magnitude_error, magnitude_band = _catalog_magnitude(
                catalog, int(catalog_index), image_filter
            )
            measurement_rows.append(
                {
                    "persistent_id": persistent_id,
                    "image_id": image_id,
                    "filter": image_filter or "",
                    "source_index": int(source_index),
                    "catalog_index": int(catalog_index),
                    "source_label": int(
                        _optional_value(sources, "label", source_index, source_index)
                    ),
                    "x": x,
                    "y": y,
                    "flux": _optional_value(sources, "flux", source_index, np.nan),
                    "snr": _optional_value(sources, "snr", source_index, np.nan),
                    "fwhm_pixels": source_fwhm,
                    "fwhm_arcsec": _optional_value(
                        sources, "fwhm_arcsec", source_index, np.nan
                    ),
                    "ellipticity": _optional_value(
                        sources, "ellipticity", source_index, np.nan
                    ),
                    "catalog_separation_arcsec": float(separations[match_index]),
                    "magnitude": magnitude,
                    "magnitude_error": magnitude_error,
                    "magnitude_band": magnitude_band or "",
                    "edge_distance_pixels": float(
                        min(x, y, nx - 1 - x, ny - 1 - y)
                    ),
                    "saturation_distance_pixels": _sample_distance(
                        saturation_distance, x, y
                    ),
                    "trail_distance_pixels": _sample_distance(trail_distance, x, y),
                    "neighbor_distance_pixels": float(
                        neighbor_distances[source_index]
                    ),
                    "detector_x_fraction": x / max(1, nx - 1),
                    "detector_y_fraction": y / max(1, ny - 1),
                    "image_fwhm_pixels": working_fwhm,
                    "saturated": bool(
                        _optional_value(sources, "saturated", source_index, False)
                    ),
                    "near_edge": bool(
                        _optional_value(sources, "near_edge", source_index, False)
                    ),
                    "masked": _mask_overlap(
                        combined_mask, x, y, aperture_radius
                    ),
                    "trail_overlap": _mask_overlap(
                        trail_mask, x, y, aperture_radius
                    ),
                    "image_accepted": False,
                    "rejection_reasons": "",
                }
            )

    if not master_parts:
        return Table(masked=True), _measurement_table([])
    combined_master = vstack(master_parts, metadata_conflicts="silent")
    master = unique(combined_master, keys="persistent_id", keep="first")
    measurements = _measurement_table(measurement_rows)
    observation_counts = {
        persistent_id: len(
            set(
                str(image_id)
                for image_id in measurements["image_id"][
                    np.asarray(measurements["persistent_id"] == persistent_id)
                ]
            )
        )
        for persistent_id in master["persistent_id"]
    }
    master["observation_count"] = [
        observation_counts[str(persistent_id)] for persistent_id in master["persistent_id"]
    ]
    master.meta["image_count"] = len(image_records)
    return master, measurements


def _master_neighbor_separation(master):
    """Return nearest-neighbor sky separations for the master table."""

    if len(master) < 2:
        return np.full(len(master), np.inf)
    coordinates = SkyCoord(
        np.asarray(master["ra"], dtype=float) * u.deg,
        np.asarray(master["dec"], dtype=float) * u.deg,
        frame="icrs",
    )
    _, separation, _ = coordinates.match_to_catalog_sky(coordinates, nthneighbor=2)
    return separation.to_value(u.arcsec)


def _reason_string(reasons):
    """Return deterministic semicolon-separated rejection reasons."""

    return ";".join(dict.fromkeys(reasons))


def _reason_set(value):
    """Convert a rejection-reason cell to a set."""

    if value is None or np.ma.is_masked(value) or not str(value):
        return set()
    return {item for item in str(value).split(";") if item}


def _screen_master_catalog(master, settings):
    """Apply catalog-level morphology, variability, motion, quality, and color cuts."""

    output = Table(master, masked=True, copy=True)
    star_settings = settings.get("catalogs", {}).get("comparison_stars", {})
    neighbor_separation = _master_neighbor_separation(output)
    output["nearest_catalog_neighbor_arcsec"] = neighbor_separation * u.arcsec
    accepted = []
    reason_values = []
    color_values = []
    color_bands = star_settings.get("color_bands", ["g", "r"])
    color_one = "mag_{}".format(color_bands[0])
    color_two = "mag_{}".format(color_bands[1])

    for index in range(len(output)):
        reasons = []
        point_source = _optional_value(output, "point_source", index)
        if star_settings.get("require_point_source", True):
            if point_source is False:
                reasons.append("NOT_POINT_SOURCE")
            elif point_source is None and not star_settings.get(
                "allow_unknown_morphology", True
            ):
                reasons.append("MORPHOLOGY_UNKNOWN")

        proper_motion_values = [
            _optional_value(output, "pmra", index),
            _optional_value(output, "pmdec", index),
        ]
        if all(value is not None for value in proper_motion_values):
            total_motion = float(np.hypot(*proper_motion_values))
            if total_motion > float(
                star_settings.get("maximum_proper_motion_mas_per_year", 200.0)
            ):
                reasons.append("PROPER_MOTION_HIGH")

        ruwe = _optional_value(output, "ruwe", index)
        if ruwe is not None and float(ruwe) > float(
            star_settings.get("maximum_ruwe", 1.4)
        ):
            reasons.append("RUWE_HIGH")

        known_variable = _optional_value(output, "known_variable", index)
        if star_settings.get("reject_known_variables", True) and known_variable is True:
            reasons.append("KNOWN_VARIABLE")

        catalog_quality = _optional_value(output, "catalog_quality", index)
        if star_settings.get("require_catalog_quality", True):
            if catalog_quality is False:
                reasons.append("CATALOG_QUALITY_BAD")
            elif catalog_quality is None and not star_settings.get(
                "allow_unknown_catalog_quality", True
            ):
                reasons.append("CATALOG_QUALITY_UNKNOWN")

        if neighbor_separation[index] < float(
            star_settings.get("minimum_catalog_separation_arcsec", 2.0)
        ):
            reasons.append("CATALOG_CROWDING")

        first = _optional_value(output, color_one, index)
        second = _optional_value(output, color_two, index)
        color = None if first is None or second is None else float(first) - float(second)
        color_values.append(np.nan if color is None else color)
        if color is None:
            if star_settings.get("require_color", False):
                reasons.append("COLOR_MISSING")
        elif not (
            float(star_settings.get("minimum_color", -0.5))
            <= color
            <= float(star_settings.get("maximum_color", 2.5))
        ):
            reasons.append("COLOR_OUT_OF_RANGE")

        accepted.append(not reasons)
        reason_values.append(_reason_string(reasons))

    output["catalog_color"] = MaskedColumn(
        np.where(np.isfinite(color_values), color_values, 0.0),
        mask=~np.isfinite(color_values),
        unit=u.mag,
    )
    output["catalog_accepted"] = accepted
    output["catalog_rejection_reasons"] = np.asarray(reason_values, dtype="U512")
    for role in ROLE_NAMES:
        output["role_{}".format(role)] = np.zeros(len(output), dtype=bool)
    return output


def _inside_excluded_detector_region(row, regions):
    """Return whether a measurement lies in a configured normalized region."""

    x = float(row["detector_x_fraction"])
    y = float(row["detector_y_fraction"])
    for region in regions:
        if (
            float(region.get("x_min", 0.0)) <= x <= float(region.get("x_max", 1.0))
            and float(region.get("y_min", 0.0))
            <= y
            <= float(region.get("y_max", 1.0))
        ):
            return True
    return False


def _screen_measurements(master, measurements, settings):
    """Apply filter-specific catalog and detector-level screening."""

    output = Table(measurements, masked=True, copy=True)
    star_settings = settings.get("catalogs", {}).get("comparison_stars", {})
    master_reasons = {
        str(row["persistent_id"]): _reason_set(row["catalog_rejection_reasons"])
        for row in master
    }
    regions = star_settings.get("excluded_detector_regions", [])
    accepted = []
    reason_values = []
    for row in output:
        reasons = set(master_reasons.get(str(row["persistent_id"]), set()))
        fwhm = float(row["image_fwhm_pixels"])
        magnitude = None if np.ma.is_masked(row["magnitude"]) else float(row["magnitude"])
        magnitude_error = (
            None
            if np.ma.is_masked(row["magnitude_error"])
            else float(row["magnitude_error"])
        )
        if magnitude is None:
            reasons.add("MAGNITUDE_MISSING")
        elif magnitude < float(star_settings.get("minimum_magnitude", 10.0)):
            reasons.add("MAGNITUDE_TOO_BRIGHT")
        elif magnitude > float(star_settings.get("maximum_magnitude", 22.0)):
            reasons.add("MAGNITUDE_TOO_FAINT")
        if magnitude_error is None:
            if not star_settings.get("allow_missing_magnitude_error", True):
                reasons.add("MAGNITUDE_ERROR_MISSING")
        elif magnitude_error > float(star_settings.get("maximum_magnitude_error", 0.10)):
            reasons.add("MAGNITUDE_ERROR_HIGH")

        if float(row["edge_distance_pixels"]) < float(
            star_settings.get("minimum_edge_distance_fwhm", 5.0)
        ) * fwhm or bool(row["near_edge"]):
            reasons.add("NEAR_EDGE")
        if bool(row["saturated"]):
            reasons.add("SATURATED")
        saturation_distance = (
            np.inf
            if np.ma.is_masked(row["saturation_distance_pixels"])
            else float(row["saturation_distance_pixels"])
        )
        if saturation_distance < float(
            star_settings.get("minimum_saturation_distance_fwhm", 8.0)
        ) * fwhm:
            reasons.add("SATURATION_HALO")
        if bool(row["masked"]):
            reasons.add("MASKED")
        trail_distance = (
            np.inf
            if np.ma.is_masked(row["trail_distance_pixels"])
            else float(row["trail_distance_pixels"])
        )
        if bool(row["trail_overlap"]) or trail_distance < float(
            star_settings.get("minimum_trail_distance_fwhm", 5.0)
        ) * fwhm:
            reasons.add("TRAIL_NEARBY")
        if not np.ma.is_masked(row["snr"]) and float(row["snr"]) < float(
            star_settings.get("minimum_snr", 10.0)
        ):
            reasons.add("SNR_LOW")
        if not np.ma.is_masked(row["ellipticity"]) and float(
            row["ellipticity"]
        ) > float(star_settings.get("maximum_ellipticity", 0.35)):
            reasons.add("ELLIPTICITY_HIGH")
        if not np.ma.is_masked(row["fwhm_pixels"]):
            deviation = abs(float(row["fwhm_pixels"]) - fwhm) / max(fwhm, 1e-6)
            if deviation > float(
                star_settings.get("maximum_fwhm_deviation_fraction", 0.50)
            ):
                reasons.add("FWHM_OUTLIER")
        if float(row["neighbor_distance_pixels"]) < float(
            star_settings.get("minimum_neighbor_distance_fwhm", 3.0)
        ) * fwhm:
            reasons.add("NEIGHBOR_CONTAMINATION")
        if _inside_excluded_detector_region(row, regions):
            reasons.add("DETECTOR_REGION_EXCLUDED")
        accepted.append(not reasons)
        reason_values.append(_reason_string(sorted(reasons)))
    output["image_accepted"] = accepted
    output["rejection_reasons"] = np.asarray(reason_values, dtype="U512")
    return output


def _role_candidates(measurements, role, settings):
    """Return role-specific eligibility without coupling unrelated criteria."""

    star_settings = settings.get("catalogs", {}).get("comparison_stars", {})
    safety = {
        "NEAR_EDGE",
        "SATURATED",
        "SATURATION_HALO",
        "MASKED",
        "TRAIL_NEARBY",
        "DETECTOR_REGION_EXCLUDED",
    }
    role_rejections = {
        "astrometry": safety
        | {
            "SNR_LOW",
            "NOT_POINT_SOURCE",
            "MORPHOLOGY_UNKNOWN",
            "PROPER_MOTION_HIGH",
            "RUWE_HIGH",
            "CATALOG_QUALITY_BAD",
        },
        "psf": safety
        | {
            "SNR_LOW",
            "NOT_POINT_SOURCE",
            "MORPHOLOGY_UNKNOWN",
            "ELLIPTICITY_HIGH",
            "FWHM_OUTLIER",
            "NEIGHBOR_CONTAMINATION",
            "CATALOG_CROWDING",
        },
        "calibration": None,
        "ensemble": None,
        "qc_anchor": None,
    }
    eligible = np.zeros(len(measurements), dtype=bool)
    for index, row in enumerate(measurements):
        reasons = _reason_set(row["rejection_reasons"])
        forbidden = role_rejections[role]
        accepted = not reasons if forbidden is None else not bool(reasons & forbidden)
        if not accepted or np.ma.is_masked(row["snr"]):
            continue
        snr = float(row["snr"])
        minimum_snr = {
            "astrometry": star_settings.get("minimum_snr", 10.0),
            "psf": star_settings.get("psf_minimum_snr", 30.0),
            "calibration": star_settings.get("calibration_minimum_snr", 10.0),
            "ensemble": star_settings.get("ensemble_minimum_snr", 15.0),
            "qc_anchor": star_settings.get("qc_minimum_snr", 50.0),
        }[role]
        if snr < float(minimum_snr):
            continue
        if role == "psf" and not np.ma.is_masked(row["ellipticity"]):
            if float(row["ellipticity"]) > float(
                star_settings.get("psf_maximum_ellipticity", 0.25)
            ):
                continue
        eligible[index] = True
    return eligible


def _spatially_distributed_selection(measurements, candidates, maximum, grid, score):
    """Select high-scoring stars in round-robin detector grid cells."""

    indices = np.flatnonzero(candidates)
    selected = np.zeros(len(measurements), dtype=bool)
    if len(indices) == 0 or maximum <= 0:
        return selected
    grid_x, grid_y = int(grid[0]), int(grid[1])
    cells = {}
    for index in indices:
        x_cell = min(grid_x - 1, int(float(measurements["detector_x_fraction"][index]) * grid_x))
        y_cell = min(grid_y - 1, int(float(measurements["detector_y_fraction"][index]) * grid_y))
        cells.setdefault((x_cell, y_cell), []).append(index)
    for cell in cells:
        cells[cell].sort(key=lambda item: float(score[item]), reverse=True)
    while np.count_nonzero(selected) < min(maximum, len(indices)):
        changed = False
        for cell in sorted(cells):
            if cells[cell] and np.count_nonzero(selected) < maximum:
                selected[cells[cell].pop(0)] = True
                changed = True
        if not changed:
            break
    return selected


def _selection_overrides(settings, overrides):
    """Combine configured and call-specific star-selection overrides."""

    from .config import merge_settings

    configured = settings.get("catalogs", {}).get("comparison_stars", {})
    base = {
        "global_include": configured.get("global_include", []),
        "global_exclude": configured.get("global_exclude", []),
        "global_role_add": configured.get("global_role_add", {}),
        "global_role_remove": configured.get("global_role_remove", {}),
        "image_overrides": configured.get("image_overrides", {}),
        "user_include_overrides_safety": configured.get(
            "user_include_overrides_safety", False
        ),
    }
    return merge_settings(base, overrides or {})


def _apply_selection_overrides(master, measurements, settings, overrides):
    """Apply global and per-image additions/removals by persistent source ID."""

    resolved = _selection_overrides(settings, overrides)
    global_include = {str(value) for value in resolved.get("global_include", [])}
    global_exclude = {str(value) for value in resolved.get("global_exclude", [])}
    safety_reasons = {
        "SATURATED",
        "SATURATION_HALO",
        "MASKED",
        "TRAIL_NEARBY",
    }
    force_safety = bool(resolved.get("user_include_overrides_safety", False))

    for index, row in enumerate(master):
        persistent_id = str(row["persistent_id"])
        if persistent_id in global_exclude:
            master["catalog_accepted"][index] = False
            reasons = _reason_set(master["catalog_rejection_reasons"][index])
            reasons.add("USER_GLOBAL_EXCLUDE")
            master["catalog_rejection_reasons"][index] = _reason_string(sorted(reasons))
        elif persistent_id in global_include:
            master["catalog_accepted"][index] = True
            master["catalog_rejection_reasons"][index] = ""

    image_overrides = resolved.get("image_overrides", {})
    for index, row in enumerate(measurements):
        persistent_id = str(row["persistent_id"])
        image_id = str(row["image_id"])
        image_settings = image_overrides.get(image_id, {})
        include = global_include | {
            str(value) for value in image_settings.get("include", [])
        }
        exclude = global_exclude | {
            str(value) for value in image_settings.get("exclude", [])
        }
        if persistent_id in exclude:
            measurements["image_accepted"][index] = False
            reasons = _reason_set(measurements["rejection_reasons"][index])
            reasons.add("USER_IMAGE_EXCLUDE")
            measurements["rejection_reasons"][index] = _reason_string(sorted(reasons))
            for role in ROLE_NAMES:
                measurements["role_{}".format(role)][index] = False
        elif persistent_id in include:
            reasons = _reason_set(measurements["rejection_reasons"][index])
            if force_safety or not (reasons & safety_reasons):
                measurements["image_accepted"][index] = True

    for role in ROLE_NAMES:
        additions = {
            str(value)
            for value in resolved.get("global_role_add", {}).get(role, [])
        }
        removals = {
            str(value)
            for value in resolved.get("global_role_remove", {}).get(role, [])
        }
        column = "role_{}".format(role)
        for index, row in enumerate(measurements):
            image_settings = image_overrides.get(str(row["image_id"]), {})
            additions_image = {
                str(value)
                for value in image_settings.get("role_add", {}).get(role, [])
            }
            removals_image = {
                str(value)
                for value in image_settings.get("role_remove", {}).get(role, [])
            }
            persistent_id = str(row["persistent_id"])
            if persistent_id in removals | removals_image | global_exclude:
                measurements[column][index] = False
            elif persistent_id in additions | additions_image:
                reasons = _reason_set(row["rejection_reasons"])
                if force_safety or not (reasons & safety_reasons):
                    measurements[column][index] = True
    return resolved


def select_comparison_and_psf_stars(master, measurements, settings=None, overrides=None):
    """Screen sources, assign independent roles, and enforce spatial coverage.

    Contamination rejects only the affected star in the affected image. It does
    not reject the image itself. Global and per-image overrides use persistent
    source IDs and can independently add or remove role assignments.

    Returns
    -------
    master : astropy.table.Table
        Catalog-screened master table with aggregate role columns.
    measurements : astropy.table.Table
        Per-image screening, reasons, and independent role assignments.
    summaries : list of dict
        Per-image role counts and non-fatal selection warnings.
    """

    if settings is None:
        settings = get_default_settings()
    master = _screen_master_catalog(master, settings)
    measurements = _screen_measurements(master, measurements, settings)
    star_settings = settings.get("catalogs", {}).get("comparison_stars", {})
    image_ids = list(dict.fromkeys(str(value) for value in measurements["image_id"]))
    total_images = max(1, len(image_ids))
    detection_fraction = {
        str(persistent_id): len(
            set(
                str(image_id)
                for image_id in measurements["image_id"][
                    np.asarray(measurements["persistent_id"] == persistent_id)
                ]
            )
        )
        / total_images
        for persistent_id in master["persistent_id"]
    }
    grid = star_settings.get("spatial_grid", [3, 3])
    for image_id in image_ids:
        image_mask = np.asarray(measurements["image_id"] == image_id)
        image_indices = np.flatnonzero(image_mask)
        image_table = measurements[image_indices]
        snr = np.asarray(image_table["snr"].filled(0.0), dtype=float)
        ellipticity = np.asarray(
            image_table["ellipticity"].filled(1.0), dtype=float
        )
        fwhm = np.asarray(image_table["fwhm_pixels"].filled(np.nan), dtype=float)
        image_fwhm = np.asarray(
            image_table["image_fwhm_pixels"].filled(1.0), dtype=float
        )
        magnitude_error = np.asarray(
            image_table["magnitude_error"].filled(1.0), dtype=float
        )
        magnitude = np.asarray(image_table["magnitude"].filled(99.0), dtype=float)
        psf_score = snr / (1.0 + 5.0 * ellipticity) / (
            1.0 + np.nan_to_num(np.abs(fwhm - image_fwhm) / image_fwhm, nan=1.0)
        )
        calibration_score = snr / (1.0 + 20.0 * magnitude_error)
        qc_score = -magnitude + 1.0e-4 * np.log1p(snr)

        astrometry_candidates = _role_candidates(image_table, "astrometry", settings)
        psf_candidates = _role_candidates(image_table, "psf", settings)
        calibration_candidates = _role_candidates(
            image_table, "calibration", settings
        )
        ensemble_candidates = _role_candidates(image_table, "ensemble", settings)
        ensemble_candidates &= np.array(
            [
                detection_fraction[str(value)]
                >= float(star_settings.get("minimum_epoch_fraction", 0.50))
                for value in image_table["persistent_id"]
            ],
            dtype=bool,
        )
        qc_candidates = _role_candidates(image_table, "qc_anchor", settings)

        psf_selected = _spatially_distributed_selection(
            image_table,
            psf_candidates,
            int(settings.get("psf", {}).get("maximum_stars", 20)),
            grid,
            psf_score,
        )
        calibration_selected = _spatially_distributed_selection(
            image_table,
            calibration_candidates,
            int(star_settings.get("maximum_calibration_stars", 100)),
            grid,
            calibration_score,
        )
        ensemble_selected = _spatially_distributed_selection(
            image_table,
            ensemble_candidates,
            int(star_settings.get("maximum_ensemble_stars", 50)),
            grid,
            calibration_score,
        )
        qc_selected = _spatially_distributed_selection(
            image_table,
            qc_candidates,
            int(star_settings.get("maximum_qc_anchors", 1)),
            [1, 1],
            qc_score,
        )
        role_values = {
            "astrometry": astrometry_candidates,
            "psf": psf_selected,
            "calibration": calibration_selected,
            "ensemble": ensemble_selected,
            "qc_anchor": qc_selected,
        }
        for role, values in role_values.items():
            measurements["role_{}".format(role)][image_indices] = values

    _apply_selection_overrides(master, measurements, settings, overrides)

    for role in ROLE_NAMES:
        role_ids = {
            str(row["persistent_id"])
            for row in measurements
            if bool(row["role_{}".format(role)])
        }
        master["role_{}".format(role)] = [
            str(value) in role_ids for value in master["persistent_id"]
        ]

    summaries = []
    for image_id in image_ids:
        rows = measurements[np.asarray(measurements["image_id"] == image_id)]
        counts = {
            role: int(np.count_nonzero(rows["role_{}".format(role)]))
            for role in ROLE_NAMES
        }
        flags = []
        if counts["psf"] < int(settings.get("psf", {}).get("minimum_stars", 5)):
            flags.append("TOO_FEW_PSF_STARS")
        if counts["calibration"] < int(
            star_settings.get("minimum_catalog_stars", 3)
        ):
            flags.append("TOO_FEW_CALIBRATION_STARS")
        summaries.append(
            {
                "image_id": image_id,
                "candidate_count": len(rows),
                "strictly_accepted_count": int(
                    np.count_nonzero(rows["image_accepted"])
                ),
                "rejected_count": int(np.count_nonzero(~rows["image_accepted"])),
                "role_counts": counts,
                "flags": flags,
                "image_rejected": False,
            }
        )
    return master, measurements, summaries


def save_star_selection_tables(
    master,
    measurements,
    output_directory,
    object_name="field",
    overwrite=False,
):
    """Save master and per-image star-selection tables as ECSV files."""

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    stem = _safe_name(object_name)
    master_path = output_directory / "{}_master_sources.ecsv".format(stem)
    measurement_path = output_directory / "{}_source_measurements.ecsv".format(stem)
    master.write(master_path, format="ascii.ecsv", overwrite=bool(overwrite))
    measurements.write(
        measurement_path, format="ascii.ecsv", overwrite=bool(overwrite)
    )
    return {"master": str(master_path), "measurements": str(measurement_path)}


__all__ = [
    "CATALOG_ALIASES",
    "ROLE_NAMES",
    "build_master_source_table",
    "catalog_cache_path",
    "catalog_coordinates",
    "match_catalog_sources",
    "normalize_catalog",
    "normalize_catalog_name",
    "plate_solve_with_astrometry_net",
    "project_catalog_to_image",
    "propagate_catalog_to_epoch",
    "query_catalog",
    "refine_wcs_from_matches",
    "save_astrometry_products",
    "save_star_selection_tables",
    "select_comparison_and_psf_stars",
    "solve_astrometry",
]
