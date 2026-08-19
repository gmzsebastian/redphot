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
from astropy.table import MaskedColumn, Table
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
    output["ra"] = _numeric_column(table, selected["ra"], u.deg)
    output["dec"] = _numeric_column(table, selected["dec"], u.deg)
    output["pmra"] = _numeric_column(table, selected["pmra"], u.mas / u.yr)
    output["pmdec"] = _numeric_column(table, selected["pmdec"], u.mas / u.yr)
    output["parallax"] = _numeric_column(table, selected["parallax"], u.mas)
    output["ref_epoch"] = _numeric_column(table, selected["ref_epoch"], u.yr)

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
    """Match Step 9 detections to an epoch-corrected normalized catalog."""

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


__all__ = [
    "CATALOG_ALIASES",
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
    "solve_astrometry",
]
