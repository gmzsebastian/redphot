"""
Normalize and validate FITS metadata for redphot.

This module adds validation and provenance to the basic header extraction done
while a FITS file is read.  It uses plain dictionaries and functions so that
all results remain easy to inspect, serialize, and override.
"""

from collections.abc import Mapping
from datetime import datetime, time, timedelta, timezone
import warnings

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

from .config import FILTER_ALIASES, normalize_filter_name


DEFAULT_REQUIRED_FIELDS = ("object", "exposure_time", "mjd", "filter")
DEFAULT_CONFLICT_FIELDS = (
    "exposure_time",
    "gain",
    "read_noise",
    "saturation",
    "nonlinearity",
    "pixel_scale",
    "binning",
)

NUMERIC_FIELDS = {
    "exposure_time",
    "mjd",
    "gain",
    "read_noise",
    "saturation",
    "nonlinearity",
    "airmass",
    "pixel_scale",
    "pipeline_fwhm_arcsec",
    "pipeline_ellipticity",
    "pipeline_zeropoint_mag",
    "pipeline_saturated_fraction",
    "pipeline_wcs_error",
}


def _as_float(value):
    """Convert a finite scalar to float, returning ``None`` when invalid."""

    if value is None or np.ma.is_masked(value):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _clean_string(value):
    """Convert a scalar to a stripped string, returning ``None`` when empty."""

    if value is None or np.ma.is_masked(value):
        return None
    result = str(value).strip()
    return result or None


def _header_indices(hdulist, settings):
    """Return valid HDU indices in configured metadata search order."""

    input_settings = settings.get("input", {})
    requested = []
    explicit = input_settings.get("header_hdu")
    if isinstance(explicit, (int, np.integer)):
        requested.append(int(explicit))
    requested.extend(input_settings.get("hdu_search_order", [0, 1, 2]))
    if input_settings.get("search_remaining_hdus", True):
        requested.extend(range(len(hdulist)))

    indices = []
    for index in requested:
        if not isinstance(index, (int, np.integer)):
            try:
                index = hdulist.index_of(index)
            except (KeyError, IndexError, TypeError, ValueError):
                continue
        index = int(index)
        if index < 0:
            index += len(hdulist)
        if 0 <= index < len(hdulist) and index not in indices:
            indices.append(index)
    return indices


def collect_header_candidates(hdulist, settings, field, keywords=None):
    """Collect every usable card matching a normalized metadata field.

    Returns a list of dictionaries containing ``value``, ``keyword``, ``hdu``,
    and ``card_index``.  Duplicate cards are therefore preserved rather than
    silently collapsed by normal FITS header lookup.
    """

    if keywords is None:
        keywords = (
            settings.get("metadata", {}).get("keywords", {}).get(field, [])
        )
    wanted = {str(keyword).upper() for keyword in keywords or []}
    candidates = []

    for hdu_index in _header_indices(hdulist, settings):
        for card_index, card in enumerate(hdulist[hdu_index].header.cards):
            if card.keyword.upper() not in wanted:
                continue
            value = card.value
            if value is None or np.ma.is_masked(value):
                continue
            if isinstance(value, str) and not value.strip():
                continue
            if isinstance(value, (float, np.floating)) and not np.isfinite(value):
                continue
            candidates.append(
                {
                    "field": field,
                    "value": value,
                    "keyword": card.keyword,
                    "hdu": hdu_index,
                    "card_index": card_index,
                }
            )

    return candidates


def _candidate_number(candidate):
    """Return a candidate's numeric value, converting JD to MJD when needed."""

    value = _as_float(candidate.get("value"))
    if value is None:
        return None
    keyword = str(candidate.get("keyword", "")).upper()
    if keyword == "JD" or value > 2000000.0:
        value -= 2400000.5
    return value


def _values_conflict(values, tolerance=0.0):
    """Return whether usable scalar values disagree beyond a tolerance."""

    values = [value for value in values if value is not None]
    if len(values) < 2:
        return False

    numeric = [_as_float(value) for value in values]
    if all(value is not None for value in numeric):
        return max(numeric) - min(numeric) > float(tolerance)

    normalized = {str(value).strip().casefold() for value in values}
    return len(normalized) > 1


def _find_conflicts(hdulist, settings):
    """Find conflicting duplicates and selected alternative metadata fields."""

    metadata_settings = settings.get("metadata", {})
    keywords_by_field = metadata_settings.get("keywords", {})
    conflict_fields = metadata_settings.get(
        "conflict_fields", DEFAULT_CONFLICT_FIELDS
    )
    conflicts = []

    for field, keywords in keywords_by_field.items():
        candidates = collect_header_candidates(
            hdulist, settings, field, keywords=keywords
        )
        by_keyword = {}
        for candidate in candidates:
            key = (candidate["hdu"], candidate["keyword"].upper())
            by_keyword.setdefault(key, []).append(candidate)

        tolerance = 0.0
        if field == "exposure_time":
            tolerance = metadata_settings.get("exposure_time_tolerance_s", 1.0)

        for (hdu_index, keyword), cards in by_keyword.items():
            values = [card["value"] for card in cards]
            if _values_conflict(values, tolerance=tolerance):
                conflicts.append(
                    {
                        "field": field,
                        "kind": "duplicate_card",
                        "hdu": hdu_index,
                        "keyword": keyword,
                        "values": values,
                    }
                )

        if field not in conflict_fields:
            continue
        values = [candidate["value"] for candidate in candidates]
        distinct_keywords = {
            candidate["keyword"].upper() for candidate in candidates
        }
        if len(distinct_keywords) > 1 and _values_conflict(
            values, tolerance=tolerance
        ):
            conflicts.append(
                {
                    "field": field,
                    "kind": "alternative_keywords",
                    "values": values,
                    "sources": [
                        {
                            "hdu": candidate["hdu"],
                            "keyword": candidate["keyword"],
                        }
                        for candidate in candidates
                    ],
                }
            )

    unique = []
    seen = set()
    for conflict in conflicts:
        identity = (
            conflict.get("field"),
            conflict.get("kind"),
            conflict.get("hdu"),
            conflict.get("keyword"),
            repr(conflict.get("values")),
        )
        if identity not in seen:
            seen.add(identity)
            unique.append(conflict)
    return unique


def _parse_iso_time(value, scale="utc"):
    """Parse a complete FITS date-time value into Astropy ``Time``."""

    text = _clean_string(value)
    if text is None:
        return None
    try:
        return Time(text, scale=scale)
    except (TypeError, ValueError):
        return None


def _parse_end_time(value, start_time, scale="utc"):
    """Parse a complete or time-only exposure end value."""

    parsed = _parse_iso_time(value, scale=scale)
    if parsed is not None:
        return parsed
    if start_time is None:
        return None

    text = _clean_string(value)
    if text is None:
        return None
    try:
        clock = time.fromisoformat(text.rstrip("Z"))
    except ValueError:
        return None

    start_datetime = start_time.to_datetime(timezone=timezone.utc)
    end_datetime = datetime.combine(
        start_datetime.date(), clock, tzinfo=timezone.utc
    )
    if end_datetime < start_datetime:
        end_datetime += timedelta(days=1)
    return Time(end_datetime, scale=scale)


def _time_from_mjd_candidate(candidate, scale="utc"):
    """Convert an MJD/JD header candidate into Astropy ``Time``."""

    value = _candidate_number(candidate)
    if value is None:
        return None
    try:
        return Time(value, format="mjd", scale=scale)
    except (TypeError, ValueError):
        return None


def _seconds_between(first, second):
    """Return ``second - first`` in seconds when both times are available."""

    if first is None or second is None:
        return None
    return float((second - first).to_value(u.s))


def _add_flag(flags, flag):
    """Append a quality flag once while preserving insertion order."""

    if flag not in flags:
        flags.append(flag)


def _resolve_exposure_and_times(hdulist, metadata, settings, conflicts, flags):
    """Resolve exposure duration and canonical start, midpoint, and end times."""

    metadata_settings = settings.get("metadata", {})
    scale = metadata_settings.get("canonical_time_scale", "utc")
    exposure_tolerance = float(
        metadata_settings.get("exposure_time_tolerance_s", 1.0)
    )
    time_tolerance = float(metadata_settings.get("time_tolerance_s", 2.0))

    start = _parse_iso_time(metadata.get("date_obs"), scale=scale)
    end_header = _parse_end_time(metadata.get("date_end"), start, scale=scale)
    elapsed = _seconds_between(start, end_header)
    if elapsed is not None and elapsed <= 0:
        elapsed = None

    exposure_candidates = collect_header_candidates(
        hdulist, settings, "exposure_time"
    )
    numeric_candidates = []
    for candidate in exposure_candidates:
        value = _as_float(candidate["value"])
        if value is not None and value > 0:
            numeric_candidates.append((value, candidate))

    selected_exposure = _as_float(metadata.get("exposure_time"))
    selected_source = metadata.get("_sources", {}).get("exposure_time")
    override_used = bool(selected_source and "override" in selected_source)
    resolve_from_times = metadata_settings.get(
        "resolve_exposure_from_times", True
    )
    derive_from_times = metadata_settings.get(
        "derive_exposure_from_times", True
    )

    if not override_used and numeric_candidates:
        if elapsed is not None and resolve_from_times:
            selected_exposure, selected_candidate = min(
                numeric_candidates, key=lambda item: abs(item[0] - elapsed)
            )
        else:
            selected_exposure, selected_candidate = numeric_candidates[-1]
        selected_source = {
            "hdu": selected_candidate["hdu"],
            "keyword": selected_candidate["keyword"],
            "card_index": selected_candidate["card_index"],
            "resolution": (
                "closest_to_start_end_interval"
                if elapsed is not None
                else "last_header_card"
            ),
        }
    elif (
        selected_exposure is None
        and elapsed is not None
        and derive_from_times
    ):
        selected_exposure = elapsed
        selected_source = {"derived_from": "start_end_interval"}

    if selected_exposure is not None and selected_exposure <= 0:
        selected_exposure = None

    metadata["exposure_time"] = selected_exposure
    metadata["exposure_time_from_times"] = elapsed
    metadata["exposure_time_header_values"] = [
        value for value, _ in numeric_candidates
    ]
    if selected_source is not None:
        metadata.setdefault("_sources", {})["exposure_time"] = selected_source

    exposure_values = [value for value, _ in numeric_candidates]
    if _values_conflict(exposure_values, tolerance=exposure_tolerance):
        _add_flag(flags, "EXPOSURE_TIME_CONFLICT")

    if (
        elapsed is not None
        and selected_exposure is not None
        and abs(elapsed - selected_exposure) > time_tolerance
    ):
        _add_flag(flags, "TIME_CONFLICT")
        conflicts.append(
            {
                "field": "exposure_time",
                "kind": "start_end_interval",
                "header_exposure_s": selected_exposure,
                "start_end_exposure_s": elapsed,
            }
        )

    mjd_candidates = collect_header_candidates(hdulist, settings, "mjd")
    mjd_source = metadata.get("_sources", {}).get("mjd", {})
    primary_mjd_candidate = None
    for candidate in mjd_candidates:
        if (
            candidate["hdu"] == mjd_source.get("hdu")
            and candidate["keyword"].upper()
            == str(mjd_source.get("keyword", "")).upper()
        ):
            primary_mjd_candidate = candidate
            break
    if primary_mjd_candidate is None and mjd_candidates:
        primary_mjd_candidate = mjd_candidates[0]

    mjd_header_time = None
    if primary_mjd_candidate is not None:
        mjd_header_time = _time_from_mjd_candidate(
            primary_mjd_candidate, scale=scale
        )

    reference = str(metadata_settings.get("time_reference", "start")).lower()
    if start is None and mjd_header_time is not None:
        start = mjd_header_time
        if selected_exposure is not None:
            if reference == "mid":
                start = start - selected_exposure / 2.0 * u.s
            elif reference == "end":
                start = start - selected_exposure * u.s

    midpoint = None
    end = end_header
    if start is not None and selected_exposure is not None:
        midpoint = start + selected_exposure / 2.0 * u.s
        expected_end = start + selected_exposure * u.s
        if end is None:
            end = expected_end
        elif abs(_seconds_between(expected_end, end)) > time_tolerance:
            _add_flag(flags, "TIME_CONFLICT")

    expected_reference = start
    if reference == "mid" and midpoint is not None:
        expected_reference = midpoint
    elif reference == "end" and end is not None:
        expected_reference = end

    if mjd_header_time is not None and expected_reference is not None:
        metadata["mjd_header_difference_s"] = abs(
            _seconds_between(expected_reference, mjd_header_time)
        )
    else:
        metadata["mjd_header_difference_s"] = None

    if expected_reference is not None:
        for candidate in mjd_candidates:
            candidate_time = _time_from_mjd_candidate(candidate, scale=scale)
            if candidate_time is None:
                continue
            difference = abs(
                _seconds_between(expected_reference, candidate_time)
            )
            if difference <= time_tolerance:
                continue
            _add_flag(flags, "TIME_CONFLICT")
            conflicts.append(
                {
                    "field": "mjd",
                    "kind": "date_obs_vs_mjd_card",
                    "hdu": candidate["hdu"],
                    "keyword": candidate["keyword"],
                    "value": candidate["value"],
                    "difference_s": difference,
                }
            )

    metadata["date_start_utc"] = None if start is None else start.utc.isot
    metadata["date_mid_utc"] = None if midpoint is None else midpoint.utc.isot
    metadata["date_end_utc"] = None if end is None else end.utc.isot
    metadata["mjd_start"] = None if start is None else float(start.utc.mjd)
    metadata["mjd_mid"] = None if midpoint is None else float(midpoint.utc.mjd)
    metadata["mjd_end"] = None if end is None else float(end.utc.mjd)
    metadata["mjd"] = metadata["mjd_mid"]
    metadata["mjd_utc"] = metadata["mjd_mid"]
    metadata["time_reference"] = "mid" if midpoint is not None else None


def _coordinate_from_values(ra, dec, units=("hourangle", "deg")):
    """Build an ICRS coordinate from numeric or sexagesimal values."""

    if ra is None or dec is None:
        return None
    if isinstance(ra, (int, float, np.number)) and isinstance(
        dec, (int, float, np.number)
    ):
        try:
            return SkyCoord(float(ra), float(dec), unit="deg", frame="icrs")
        except (TypeError, ValueError):
            return None

    ra_text = str(ra).strip()
    dec_text = str(dec).strip()
    coordinate_units = units
    if ":" not in ra_text and len(ra_text.split()) == 1:
        coordinate_units = ("deg", "deg")
    try:
        return SkyCoord(
            ra_text,
            dec_text,
            unit=coordinate_units,
            frame="icrs",
        )
    except (TypeError, ValueError):
        return None


def _user_coordinate(target, settings):
    """Read a user coordinate from an argument or target-position settings."""

    if isinstance(target, SkyCoord):
        return target.icrs
    if target is None:
        target_settings = settings.get("target_position", {})
        ra = target_settings.get("ra")
        dec = target_settings.get("dec")
    elif isinstance(target, Mapping):
        ra = target.get("ra")
        dec = target.get("dec")
    else:
        try:
            ra, dec = target
        except (TypeError, ValueError):
            return None

    units = tuple(
        settings.get("target_position", {}).get(
            "coordinate_unit", ["hourangle", "deg"]
        )
    )
    return _coordinate_from_values(ra, dec, units=units)


def _wcs_center_coordinate(wcs, shape):
    """Return the celestial coordinate at the image-array center."""

    if wcs is None or shape is None or len(shape) != 2:
        return None
    try:
        ny, nx = shape
        coordinate = wcs.pixel_to_world((nx - 1) / 2.0, (ny - 1) / 2.0)
        if isinstance(coordinate, SkyCoord):
            return coordinate.icrs
    except Exception:
        return None
    return None


def _store_coordinate(metadata, prefix, coordinate):
    """Store one coordinate as separate decimal-degree metadata fields."""

    metadata["{}_ra_deg".format(prefix)] = (
        None if coordinate is None else float(coordinate.icrs.ra.deg)
    )
    metadata["{}_dec_deg".format(prefix)] = (
        None if coordinate is None else float(coordinate.icrs.dec.deg)
    )


def _normalize_positions(metadata, settings, wcs, shape, target, flags):
    """Keep pointing, header, WCS, user, and adopted positions independent."""

    coordinate_units = tuple(
        settings.get("metadata", {}).get(
            "header_coordinate_unit", ["hourangle", "deg"]
        )
    )
    pointing = _coordinate_from_values(
        metadata.get("pointing_ra"),
        metadata.get("pointing_dec"),
        units=coordinate_units,
    )
    header_target = _coordinate_from_values(
        metadata.get("target_ra"),
        metadata.get("target_dec"),
        units=coordinate_units,
    )
    wcs_center = _wcs_center_coordinate(wcs, shape)
    user_target = _user_coordinate(target, settings)

    positions = {
        "user": user_target,
        "header_target": header_target,
        "wcs_center": wcs_center,
        "telescope_pointing": pointing,
    }
    configured_precedence = settings.get("target_position", {}).get(
        "metadata_precedence",
        ["user", "header_target", "wcs_center", "telescope_pointing"],
    )
    adopted = None
    adopted_source = None
    for source in configured_precedence:
        if positions.get(source) is not None:
            adopted = positions[source]
            adopted_source = source
            break

    _store_coordinate(metadata, "pointing", pointing)
    _store_coordinate(metadata, "header_target", header_target)
    _store_coordinate(metadata, "wcs_center", wcs_center)
    _store_coordinate(metadata, "user_target", user_target)
    _store_coordinate(metadata, "adopted", adopted)
    metadata["adopted_position_source"] = adopted_source
    metadata["adopted_position_is_preliminary"] = True
    metadata["wcs_valid"] = wcs_center is not None

    if wcs_center is None:
        _add_flag(flags, "WCS_MISSING")


def _read_diagnostic_fields(hdulist, metadata, settings):
    """Read configured upstream-pipeline diagnostic header values."""

    diagnostic_keywords = settings.get("metadata", {}).get(
        "diagnostic_keywords", {}
    )
    sources = metadata.setdefault("_sources", {})

    for field, keywords in diagnostic_keywords.items():
        candidates = collect_header_candidates(
            hdulist, settings, field, keywords=keywords
        )
        if not candidates:
            metadata[field] = None
            continue
        selected = candidates[-1]
        value = selected["value"]
        metadata[field] = (
            _as_float(value) if field in NUMERIC_FIELDS else _clean_string(value)
        )
        sources[field] = {
            "hdu": selected["hdu"],
            "keyword": selected["keyword"],
            "card_index": selected["card_index"],
        }


def _validate_required_fields(metadata, settings, flags):
    """Flag missing required and calibration metadata values."""

    metadata_settings = settings.get("metadata", {})
    required = metadata_settings.get("required_fields", DEFAULT_REQUIRED_FIELDS)
    missing = [field for field in required if metadata.get(field) is None]
    if missing:
        _add_flag(flags, "METADATA_MISSING")

    if metadata.get("filter") is None:
        _add_flag(flags, "FILTER_UNKNOWN")
    else:
        known_filters = set(FILTER_ALIASES.values())
        normalized = normalize_filter_name(metadata["filter"])
        metadata["filter"] = normalized
        if normalized not in known_filters:
            _add_flag(flags, "FILTER_UNKNOWN")

    if metadata.get("gain") is None:
        _add_flag(flags, "GAIN_MISSING")
    if metadata.get("read_noise") is None:
        _add_flag(flags, "READ_NOISE_MISSING")
    if metadata.get("saturation") is None:
        _add_flag(flags, "SATURATION_MISSING")

    return missing


def normalize_and_validate_metadata(
    hdulist,
    metadata,
    settings,
    wcs=None,
    shape=None,
    target=None,
):
    """Return complete normalized and validated metadata for one FITS image.

    The input dictionary is copied.  Header provenance is retained in
    ``_sources``; machine-readable issues appear in ``quality_flags`` and
    detailed disagreements in ``metadata_conflicts``.  Missing values remain
    ``None``.
    """

    metadata = dict(metadata)
    metadata["_sources"] = dict(metadata.get("_sources", {}))
    flags = list(metadata.get("quality_flags", []))
    conflicts = _find_conflicts(hdulist, settings)

    metadata_settings = settings.get("metadata", {})
    for field in NUMERIC_FIELDS:
        if field in metadata:
            metadata[field] = _as_float(metadata[field])

    if "nonlinearity" not in metadata:
        candidates = collect_header_candidates(
            hdulist, settings, "nonlinearity"
        )
        if candidates:
            selected = candidates[-1]
            metadata["nonlinearity"] = _as_float(selected["value"])
            metadata["_sources"]["nonlinearity"] = {
                "hdu": selected["hdu"],
                "keyword": selected["keyword"],
                "card_index": selected["card_index"],
            }
        else:
            metadata["nonlinearity"] = metadata_settings.get(
                "fallback_values", {}
            ).get("nonlinearity")

    _read_diagnostic_fields(hdulist, metadata, settings)
    _resolve_exposure_and_times(
        hdulist, metadata, settings, conflicts, flags
    )
    _normalize_positions(metadata, settings, wcs, shape, target, flags)

    if conflicts and metadata_settings.get("check_conflicts", True):
        _add_flag(flags, "METADATA_CONFLICT")
    missing = _validate_required_fields(metadata, settings, flags)

    metadata["metadata_conflicts"] = conflicts
    metadata["missing_metadata"] = missing
    metadata["quality_flags"] = flags
    metadata["metadata_valid"] = not bool(missing)
    if missing:
        metadata["metadata_status"] = "FAIL"
    elif flags:
        metadata["metadata_status"] = "WARN"
    else:
        metadata["metadata_status"] = "PASS"

    if metadata_settings.get("warn_on_validation", True):
        warning_flags = [
            flag
            for flag in flags
            if flag
            in {
                "METADATA_CONFLICT",
                "EXPOSURE_TIME_CONFLICT",
                "TIME_CONFLICT",
            }
        ]
        if warning_flags:
            warnings.warn(
                "Metadata validation flags: {}".format(
                    ", ".join(warning_flags)
                ),
                RuntimeWarning,
            )

    return metadata


__all__ = [
    "collect_header_candidates",
    "normalize_and_validate_metadata",
]
