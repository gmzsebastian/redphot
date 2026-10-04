"""Assemble traceable reports and output products for redphot runs.

This module deliberately contains only functions.  It provides one central
output policy so scientific stages can return rich in-memory products without
forcing every run to write large intermediate images.
"""

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import MaskedColumn, Table, vstack

from .config import get_default_settings, merge_settings


TABLE_PRODUCTS = (
    "images_table",
    "sources_table",
    "photometry_table",
    "lightcurve_table",
)

FITS_PRODUCTS = (
    "processed_image",
    "cleaned_image",
    "fringe_corrected_image",
    "source_mask",
    "cosmic_ray_mask",
    "trail_mask",
    "saturation_mask",
    "background_mask",
    "background_model",
    "background_rms",
    "background_subtracted",
    "psf_model",
    "psf_cutouts",
    "psf_residuals",
    "downloaded_template",
    "aligned_template",
    "difference_image",
    "difference_mask",
    "subtraction_kernel",
)

# Products not written by the generic derivative writer: the final
# light-curve figure and the processed multi-extension FITS file per image.
SPECIAL_PRODUCTS = ("lightcurve_plot", "processed_image")

_SMALL_PRODUCTS = {
    "images_table", "sources_table", "photometry_table", "lightcurve_table",
    "resolved_config", "run_log", "manifest", "lightcurve_plot",
}

_STANDARD_PRODUCTS = _SMALL_PRODUCTS | {
    "image_pdfs", "batch_pdf", "psf_model", "difference_image",
    "processed_image",
}

_ALL_PRODUCTS = _STANDARD_PRODUCTS | set(FITS_PRODUCTS) | set(SPECIAL_PRODUCTS)


def image_stem(image_id):
    """File-name stem of an image: its ID without the FITS ending, made safe."""

    text = str(image_id)
    for ending in (".fits.fz", ".fits.gz", ".fits", ".fit", ".fz"):
        if text.lower().endswith(ending):
            text = text[: -len(ending)]
            break
    return _safe_name(text)


def _safe_name(value):
    """Return a filesystem-safe but recognizable name."""

    text = str(value or "unknown")
    cleaned = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in text
    ).strip("_")
    return cleaned or "unknown"


def _plain_value(value):
    """Convert common scientific Python values into JSON-compatible values."""

    if value is None or np.ma.is_masked(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "unit") and hasattr(value, "value"):
        return {"value": _plain_value(value.value), "unit": str(value.unit)}
    if isinstance(value, dict):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, set):
        return sorted((_plain_value(item) for item in value), key=str)
    if isinstance(value, (list, tuple)):
        return [_plain_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _canonical_json(value):
    """Serialize a value deterministically for configuration tracing."""

    return json.dumps(
        _plain_value(value), sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )


def configuration_hash(settings):
    """Return the SHA-256 digest of a resolved configuration mapping."""

    return hashlib.sha256(_canonical_json(settings).encode("utf-8")).hexdigest()


def resolve_output_policy(settings=None, profile=None, product_overrides=None):
    """Resolve named save profiles and per-product overrides.

    Profiles are intentionally simple:

    ``minimal``
        Four final ECSV tables, resolved configuration, log, and manifest.
    ``standard``
        Minimal products plus PDF reports, PSF models, and difference images.
    ``full``
        Every supplied table, report, mask, model, and FITS derivative.
    ``custom``
        Use ``output.products`` exactly.

    ``output.product_overrides`` and the function argument are applied last,
    allowing a single product to be enabled or disabled under any profile.
    """

    resolved = merge_settings(get_default_settings(), settings or {})
    configured = resolved.get("output", {})
    selected = str(profile or configured.get("profile", "standard")).lower()
    if selected not in {"minimal", "standard", "full", "custom"}:
        raise ValueError("Unknown output profile: {}".format(selected))

    known = set(configured.get("products", {})) | _ALL_PRODUCTS
    if selected == "custom":
        products = {
            name: bool(value)
            for name, value in configured.get("products", {}).items()
        }
    else:
        enabled = {
            "minimal": _SMALL_PRODUCTS,
            "standard": _STANDARD_PRODUCTS,
            "full": _ALL_PRODUCTS,
        }[selected]
        products = {name: name in enabled for name in known}

    for overrides in (
        configured.get("product_overrides", {}), product_overrides or {}
    ):
        for name, value in overrides.items():
            if not isinstance(value, bool):
                raise TypeError("Output product overrides must be boolean")
            products[str(name)] = value

    return {
        "profile": selected,
        "products": products,
        "image_pdf_stages": configured.get("image_pdf_stages", "completed"),
        "selected_stage_names": list(configured.get("selected_stage_names", [])),
        "fits_dtype": configured.get("fits_dtype", "float32"),
        "fits_compression": configured.get("fits_compression", "none"),
        "include_checksums": bool(configured.get("include_checksums", True)),
        "overwrite": bool(configured.get("overwrite", False)),
    }


def output_product_enabled(settings, product_name, profile=None, overrides=None):
    """Return whether one named product is enabled by the output policy."""

    policy = resolve_output_policy(settings, profile, overrides)
    return bool(policy["products"].get(str(product_name), False))


def _record_id(record, index):
    metadata = record.get("metadata", {}) if isinstance(record, dict) else {}
    return str(
        record.get("image_id")
        or metadata.get("image_id")
        or metadata.get("filename")
        or "image_{:04d}".format(index + 1)
    )


def _record_path(record):
    metadata = record.get("metadata", {})
    return str(record.get("path") or metadata.get("path") or metadata.get("filename") or "")


def _record_status(record):
    for container in (
        record.get("decision", {}), record.get("quality", {}),
        record.get("metadata", {}), record,
    ):
        for name in ("final_status", "status", "metadata_status"):
            if container.get(name) not in (None, ""):
                return str(container[name])
    return "UNKNOWN"


def _record_flags(record):
    values = []
    for container in (
        record.get("metadata", {}), record.get("quality", {}),
        record.get("decision", {}), record,
    ):
        for name in ("quality_flags", "flags", "reasons"):
            value = container.get(name)
            if isinstance(value, str):
                values.extend(filter(None, value.replace(",", ";").split(";")))
            elif value is not None:
                values.extend(str(item) for item in value if item not in (None, ""))
    return ";".join(dict.fromkeys(values))


def build_images_table(image_records, config_digest=None, run_id=None):
    """Build the canonical one-row-per-input ``images.ecsv`` table."""

    rows = []
    for index, record in enumerate(image_records or []):
        metadata = record.get("metadata", {})
        quality = record.get("quality", {})
        decision = record.get("decision", {})
        rows.append({
            "image_id": _record_id(record, index),
            "input_file": _record_path(record),
            "object": metadata.get("object"),
            "filter": metadata.get("filter"),
            "mjd": metadata.get("mjd_mid", metadata.get("mjd")),
            "telescope": metadata.get("telescope"),
            "site": metadata.get("site"),
            "instrument": metadata.get("instrument"),
            "data_hdu": metadata.get("data_hdu"),
            "quality_status": _record_status(record),
            "failed_stage": record.get("failed_stage"),
            "user_decision": decision.get("manual_decision", decision.get("user_decision")),
            "depth_3sigma_mag": decision.get("depth_3sigma_mag", quality.get("depth_3sigma_mag")),
            "depth_5sigma_mag": decision.get("depth_5sigma_mag", quality.get("depth_5sigma_mag")),
            "flags": _record_flags(record),
            "config_sha256": config_digest,
            "run_id": run_id,
        })
    return Table(rows=rows, masked=True)


def _copy_or_empty_table(table):
    if table is None:
        return Table(masked=True)
    return Table(table, masked=True, copy=True)


def _add_trace_column(table, name, values, dtype=None):
    if name in table.colnames:
        return
    if dtype is None:
        table[name] = values
    else:
        table.add_column(MaskedColumn(values, name=name, dtype=dtype))


def add_photometry_traceability(photometry, image_records, config_digest, run_id):
    """Add stable provenance columns without replacing scientific columns."""

    table = _copy_or_empty_table(photometry)
    record_lookup = {
        _record_id(record, index): record
        for index, record in enumerate(image_records or [])
    }
    identifiers, paths, hdus, layers, hashes, runs, calibration_refs = [], [], [], [], [], [], []
    for index, row in enumerate(table):
        image_id = str(row["image_id"]) if "image_id" in table.colnames else ""
        record = record_lookup.get(image_id, {})
        metadata = record.get("metadata", {})
        kind = str(row["image_kind"]) if "image_kind" in table.colnames else "science"
        method = str(row["method"]) if "method" in table.colnames else "unknown"
        source = str(row["source_id"]) if "source_id" in table.colnames else "target"
        digest_text = "|".join((run_id, image_id, kind, method, source, str(index)))
        identifiers.append(hashlib.sha256(digest_text.encode("utf-8")).hexdigest()[:20])
        paths.append(_record_path(record))
        hdus.append(metadata.get("data_hdu"))
        layers.append(kind)
        hashes.append(config_digest)
        runs.append(run_id)
        catalog = row["calibration_catalog"] if "calibration_catalog" in table.colnames else None
        zeropoint = row["zeropoint_mag"] if "zeropoint_mag" in table.colnames else None
        if np.ma.is_masked(catalog):
            catalog = None
        if np.ma.is_masked(zeropoint):
            zeropoint = None
        calibration_refs.append(
            "catalog={};zeropoint={}".format(catalog or "none", zeropoint)
        )
    _add_trace_column(table, "measurement_id", identifiers)
    _add_trace_column(table, "input_file", paths)
    _add_trace_column(table, "input_data_hdu", hdus)
    _add_trace_column(table, "image_layer", layers)
    _add_trace_column(table, "config_sha256", hashes)
    _add_trace_column(table, "run_id", runs)
    _add_trace_column(table, "calibration_reference", calibration_refs)
    table.meta["traceability"] = (
        "measurement_id -> input_file, input_data_hdu, image_layer, "
        "config_sha256, calibration_reference"
    )
    return table


def add_lightcurve_traceability(lightcurve, traced_photometry, image_records,
                                config_digest, run_id):
    """Link preferred light-curve rows back to their full photometry rows."""

    table = _copy_or_empty_table(lightcurve)
    source_ids = []
    calibration = []
    for row in table:
        source = None
        if "measurement_index" in table.colnames:
            value = row["measurement_index"]
            if not np.ma.is_masked(value):
                index = int(value)
                if 0 <= index < len(traced_photometry):
                    source = traced_photometry[index]
        if source is None and len(traced_photometry):
            selection = np.ones(len(traced_photometry), dtype=bool)
            for name in ("image_id", "image_kind", "method"):
                if name in table.colnames and name in traced_photometry.colnames:
                    selection &= (
                        np.asarray(traced_photometry[name], dtype=str) == str(row[name])
                    )
            if "source_type" in traced_photometry.colnames:
                selection &= np.asarray(traced_photometry["source_type"], dtype=str) == "target"
            matches = np.flatnonzero(selection)
            if matches.size:
                source = traced_photometry[int(matches[0])]
        if source is None:
            source_ids.append("")
            calibration.append("catalog=none;zeropoint=None")
        else:
            source_ids.append(str(source["measurement_id"]))
            calibration.append(str(source["calibration_reference"]))
    _add_trace_column(table, "source_measurement_id", source_ids)
    _add_trace_column(table, "calibration_reference", calibration)
    return add_photometry_traceability(
        table, image_records, config_digest, run_id
    )


def build_sources_table(sources, config_digest=None, run_id=None):
    """Normalize a source table or list of source tables for final output."""

    if isinstance(sources, (list, tuple)):
        tables = [_copy_or_empty_table(item) for item in sources if item is not None]
        table = vstack(tables, join_type="outer", metadata_conflicts="silent") if tables else Table(masked=True)
    else:
        table = _copy_or_empty_table(sources)
    _add_trace_column(table, "config_sha256", [config_digest] * len(table))
    _add_trace_column(table, "run_id", [run_id] * len(table))
    return table


def _summary_page(record, preferred_rows=None):
    """Create the first page of one image report (observation and final result)."""

    from .diagnostics import (
        _fmt, _image_subtitle, _new_figure, diagnostic_style, metric_panel,
    )

    metadata = record.get("metadata", {}) or {}
    decision = record.get("decision", {}) or {}
    image_id = _record_id(record, 0)
    status = _record_status(record)
    with diagnostic_style():
        figure, grid = _new_figure(
            "Image report  ·  {}".format(image_id), _image_subtitle(metadata), status,
            size=(11, 8.5), rows=1, columns=2,
        )
        observation = [
            ("Object", metadata.get("object") or "—", None),
            ("Input file", str(_record_path(record) or "—").split("/")[-1], None),
            ("Filter", metadata.get("filter") or "—", None),
            ("MJD (mid)", _fmt(metadata.get("mjd_mid", metadata.get("mjd")), "{:.5f}"), None),
            ("Exposure Time", _fmt(metadata.get("exposure_time"), "{:g}", "s"), None),
            ("Airmass", _fmt(metadata.get("airmass"), "{:.2f}"), None),
            ("Telescope / instrument", "{} / {}".format(metadata.get("telescope", "—"),
                                                       metadata.get("instrument", "—")), None),
            ("Data HDU", str(metadata.get("data_hdu", "—")), None),
        ]
        metric_panel(figure.add_subplot(grid[0, 0]), observation, "Observation")
        depths = decision.get("global_depths_mag") or {}
        result = [
            ("Status", status, status),
            ("Failed stage", record.get("failed_stage") or "none",
             "FAIL" if record.get("failed_stage") else None),
            ("Review decisions", ", ".join(
                "{} {}".format(stage, (value or {}).get("decision", "")).lower()
                for stage, value in (record.get("review_decisions") or {}).items()) or "none", None),
            ("Quick 5σ depth", _fmt(depths.get("5sigma"), "{:.2f}", "mag"), None),
        ]
        for row in (preferred_rows or []):
            values = {name: row[name] for name in row.colnames}
            magnitude = values.get("magnitude", values.get("calibrated_magnitude"))
            result.append((
                "Final {} {}".format(values.get("image_kind", ""), values.get("method", "")).strip(),
                "{} ± {} mag".format(_fmt(magnitude, "{:.3f}"),
                                     _fmt(values.get("magnitude_uncertainty"), "{:.3f}")),
                None,
                "S/N {}  ·  {}".format(_fmt(values.get("snr"), "{:.1f}"),
                                       values.get("classification", "")),
            ))
        if not preferred_rows or not len(preferred_rows):
            result.append(("Final measurement", "none (image may have stopped earlier)", None))
        flags = [flag for flag in str(_record_flags(record) or "").replace(",", ";").split(";")
                 if flag.strip()]
        metric_panel(figure.add_subplot(grid[0, 1]), result, "Result", flags=flags[:12])
    return figure


def _failure_page(record):
    from .diagnostics import plot_stage_status, _image_subtitle

    return plot_stage_status(
        "Processing stopped", _record_status(record), _record_id(record, 0),
        _image_subtitle(record.get("metadata") or {}),
        "Failed stage: {}\n\nFlags / reasons: {}\n\n{}".format(
            record.get("failed_stage") or "unknown",
            _record_flags(record) or "none recorded",
            record.get("failure_error") or "No exception text recorded.",
        ),
    )


def _normalize_stage_items(stages):
    if stages is None:
        return []
    if isinstance(stages, dict):
        values = []
        for name, item in stages.items():
            if isinstance(item, dict):
                normalized = dict(item)
                normalized.setdefault("name", name)
            else:
                normalized = {"name": name, "figure": item, "status": "COMPLETED"}
            values.append(normalized)
        return values
    values = []
    for index, item in enumerate(stages):
        if isinstance(item, dict):
            values.append(dict(item))
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            values.append({"name": item[0], "figure": item[1], "status": "COMPLETED"})
        else:
            values.append({"name": "stage_{:02d}".format(index + 1), "figure": item,
                           "status": "COMPLETED"})
    return values


def _stage_figure(item):
    figure = item.get("figure")
    if callable(figure) and not hasattr(figure, "savefig"):
        figure = figure()
        if figure is None:
            return None, False
        return figure, True
    if figure is not None and hasattr(figure, "savefig"):
        return figure, False
    path = item.get("path")
    if path is None and isinstance(figure, (str, Path)):
        path = figure
    if path is None:
        return None, False
    path = Path(path)
    if not path.exists() or path.suffix.lower() == ".pdf":
        return None, False
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    pixels = mpimg.imread(path)
    height, width = pixels.shape[:2]
    rendered = plt.figure(figsize=(11.0, 11.0 * height / max(width, 1)))
    axis = rendered.add_axes([0, 0, 1, 1])
    axis.imshow(pixels)
    axis.set_axis_off()
    return rendered, True


def make_image_diagnostic_pdf(record, stages, output_path, preferred_rows=None,
                              settings=None, policy=None):
    """Write one summary-led multipage PDF containing all completed stages."""

    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    policy = policy or resolve_output_policy(settings)
    mode = policy.get("image_pdf_stages", "completed")
    selected = set(policy.get("selected_stage_names", []))
    items = _normalize_stage_items(stages)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_path) as pdf:
        summary = _summary_page(record, preferred_rows)
        pdf.savefig(summary, bbox_inches="tight")
        plt.close(summary)
        if mode != "summary_only":
            failed_stage = str(record.get("failed_stage") or "")
            for item in items:
                status = str(item.get("status", "COMPLETED")).upper()
                failed_item = str(item.get("name", "")) == failed_stage or status in {
                    "FAIL", "FAILED", "ERROR"
                }
                if status not in {
                    "COMPLETE", "COMPLETED", "PASS", "WARN", "FAIL", "FAILED", "ERROR",
                    "APPROVED", "REJECTED", "SKIPPED",
                }:
                    continue
                if mode == "selected" and str(item.get("name")) not in selected:
                    if failed_item:
                        break
                    continue
                figure, temporary = _stage_figure(item)
                if figure is not None:
                    pdf.savefig(figure, bbox_inches="tight")
                    if temporary or item.get("close_after", False):
                        plt.close(figure)
                if failed_item:
                    break
        if record.get("failed_stage") or _record_status(record) == "FAIL":
            failure = _failure_page(record)
            pdf.savefig(failure, bbox_inches="tight")
            plt.close(failure)
    return str(output_path)


def make_batch_summary_pdf(batch_products, output_path, show=False):
    """Write the standard batch trends as a one-page summary PDF."""

    import matplotlib.pyplot as plt
    from .diagnostics import plot_batch_consistency_diagnostics

    figure = plot_batch_consistency_diagnostics(batch_products, show=show)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, format="pdf", bbox_inches="tight")
    if not show:
        plt.close(figure)
    return str(output_path)


def _fits_header(record, product_name, config_digest, run_id):
    metadata = record.get("metadata", {})
    header = fits.Header()
    ccd = record.get("ccd")
    candidate = getattr(ccd, "meta", None)
    if isinstance(candidate, fits.Header):
        header = candidate.copy()
    header["RDPID"] = (str(run_id)[:68], "redphot run identifier")
    header["RDPCONF"] = (str(config_digest)[:32], "configuration SHA-256 prefix")
    header["RDPPROD"] = (str(product_name)[:68], "redphot derivative product")
    header["RDPINP"] = (_record_path(record)[-68:], "source input file")
    if metadata.get("data_hdu") is not None:
        header["RDPHDU"] = (int(metadata["data_hdu"]), "source science-data HDU")
    header.add_history("Derived by redphot; original input pixels were not overwritten.")
    return header


def _derivative_data(value):
    if isinstance(value, dict):
        return value.get("data"), value.get("header")
    return getattr(value, "data", value), getattr(value, "meta", None)


def save_fits_derivatives(image_records, derivatives, output_directory, settings=None,
                          policy=None, config_digest=None, run_id=None):
    """Save only enabled FITS derivatives supplied by completed stages."""

    policy = policy or resolve_output_policy(settings)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    record_lookup = {
        _record_id(record, index): record
        for index, record in enumerate(image_records or [])
    }
    paths = []
    for image_id, products in (derivatives or {}).items():
        record = record_lookup.get(str(image_id), {"image_id": str(image_id), "metadata": {}})
        for product_name, value in products.items():
            name = str(product_name)
            if name not in FITS_PRODUCTS or name in SPECIAL_PRODUCTS \
                    or not policy["products"].get(name, False):
                continue
            data, supplied_header = _derivative_data(value)
            if data is None:
                continue
            array = np.asarray(data)
            if array.dtype == bool or name.endswith("mask"):
                array = array.astype(np.uint8)
            elif policy["fits_dtype"] != "preserve":
                array = array.astype(policy["fits_dtype"])
            header = _fits_header(record, name, config_digest, run_id)
            if isinstance(supplied_header, fits.Header):
                for card in supplied_header.cards:
                    if card.keyword not in header and card.keyword not in {"", "HISTORY", "COMMENT"}:
                        try:
                            header.append(card)
                        except (TypeError, ValueError):
                            pass
            suffix = ".fits.gz" if policy["fits_compression"] == "gzip" else ".fits"
            path = output / "{}_{}{}".format(image_stem(image_id), name, suffix)
            fits.writeto(path, array, header=header, overwrite=policy["overwrite"],
                         checksum=policy["include_checksums"])
            paths.append((name, str(path), str(image_id)))
    return paths


# Bits of the MASK extension of processed images (0 = pixel used).
MASK_BITS = {
    "nonfinite": 1,
    "input": 2,
    "saturation": 4,
    "bad_lines": 8,
    "amplifier": 16,
    "trails": 32,
    "manual": 64,
    "cosmic_rays": 128,
}

_MASK_BIT_TEXT = {
    "nonfinite": "non-finite value",
    "input": "input mask / unusable region (outside DATASEC, trimmed edge)",
    "saturation": "saturated star (grown core, bleeds)",
    "bad_lines": "bad row or column",
    "amplifier": "amplifier seam",
    "trails": "satellite or aircraft trail",
    "manual": "manual mask region",
    "cosmic_rays": "cosmic ray",
}


def save_processed_images(items, output_directory, policy=None, config_digest=None,
                          run_id=None):
    """Write the final processed multi-extension FITS file of every image.

    Each item (built by the pipeline) carries the processed science array, its
    header (aligned WCS and processing keywords), the mask components, and
    the background model and RMS. The file sits next to the image's PSF model::

        fits/<image>_processed.fits
            [0] processed image: cut to the usable area, masked pixels kept,
                cosmic rays and fringes handled, background-subtracted, on its
                native pixels with the WCS aligned to the reference image
            [MASK] bit mask (header lists the bits; 0 = pixel used)
            [BKG], [BKGRMS] subtracted background model and its RMS
            [ERR] 1σ uncertainty, when the image has one
    """

    policy = policy or resolve_output_policy()
    output = Path(output_directory)
    entries = []
    dtype = policy.get("fits_dtype", "float32")
    cast = (lambda array: np.asarray(array)) if dtype == "preserve" else \
        (lambda array: np.asarray(array, dtype=dtype))
    for item in items or []:
        image_id = str(item["image_id"])
        stem = image_stem(image_id)
        if policy["products"].get("processed_image", False) and item.get("data") is not None:
            folder = output / "fits"
            folder.mkdir(parents=True, exist_ok=True)
            header = item["header"].copy()
            header["RDPID"] = (str(run_id)[:68], "redphot run identifier")
            header["RDPCONF"] = (str(config_digest)[:32], "configuration SHA-256 prefix")
            header["RDPPROD"] = ("processed_image", "redphot product")
            hdus = [fits.PrimaryHDU(cast(item["data"]), header=header)]
            components = item.get("mask_components") or {}
            bits = np.zeros(np.shape(item["data"]), dtype=np.uint8)
            mask_header = fits.Header()
            mask_header["EXTNAME"] = "MASK"
            mask_header["BUNIT"] = "bitmask"
            for name, bit in MASK_BITS.items():
                component = components.get(name)
                if component is not None and np.shape(component) == bits.shape:
                    bits[np.asarray(component, dtype=bool)] |= bit
                mask_header["MASKB{}".format(int(np.log2(bit)))] = (
                    "{} = {}".format(bit, name), _MASK_BIT_TEXT[name][:46])
            mask_header["MASKFRAC"] = (float(np.mean(bits > 0)), "fraction of pixels masked")
            mask_header.add_comment("0 = pixel used; any other value = excluded (OR of bits)")
            hdus.append(fits.ImageHDU(bits, header=mask_header))
            for name, key in (("BKG", "background"), ("BKGRMS", "background_rms"),
                              ("ERR", "uncertainty")):
                if item.get(key) is not None and np.shape(item[key]) == np.shape(item["data"]):
                    extension = fits.Header()
                    extension["EXTNAME"] = name
                    hdus.append(fits.ImageHDU(cast(item[key]), header=extension))
            path = folder / "{}_processed.fits".format(stem)
            fits.HDUList(hdus).writeto(path, overwrite=policy.get("overwrite", False),
                                       checksum=policy.get("include_checksums", True))
            entries.append(("processed_image", str(path), image_id))
    return entries


def save_light_curve_plot(batch_products, output_directory, object_name, settings=None,
                          overwrite=False):
    """Write the final light curve with every photometry method (PNG and PDF)."""

    import matplotlib.pyplot as plt
    from .diagnostics import plot_final_light_curve

    batch_products = batch_products or {}
    figure = plot_final_light_curve(
        batch_products.get("measurements"), batch_products.get("preferred_light_curve"),
        object_name=object_name, settings=settings,
    )
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix, dpi in ((".png", 150), (".pdf", None)):
        path = output / "{}_lightcurve{}".format(_safe_name(object_name), suffix)
        if path.exists() and not overwrite:
            raise FileExistsError(str(path))
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(str(path))
    plt.close(figure)
    return paths


def write_resolved_configuration(settings, path, overwrite=False):
    """Write the exact resolved configuration as readable, stable JSON."""

    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_plain_value(settings), indent=2, sort_keys=True) + "\n")
    return str(path)


def write_run_log(events, path, run_id, config_digest, overwrite=False):
    """Write a human-readable chronological run log."""

    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "redphot run {}".format(run_id),
        "configuration SHA-256: {}".format(config_digest),
        "written: {}".format(datetime.now(timezone.utc).isoformat()),
        "",
    ]
    for event in events or []:
        if isinstance(event, dict):
            timestamp = event.get("time", event.get("timestamp", ""))
            level = event.get("level", "INFO")
            image_id = event.get("image_id", "batch")
            stage = event.get("stage", "run")
            message = event.get("message", "")
            lines.append("{} {:<7} [{}:{}] {}".format(
                timestamp, str(level).upper(), image_id, stage, message
            ).strip())
        else:
            lines.append(str(event))
    path.write_text("\n".join(lines).rstrip() + "\n")
    return str(path)


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_output_manifest(entries, output_root, run_id, config_digest, checksums=True):
    """Build a machine-readable inventory of every saved product."""

    root = Path(output_root)
    rows = []
    for product_type, path, image_id in entries:
        file_path = Path(path)
        rows.append({
            "product_type": str(product_type),
            "image_id": image_id,
            "path": str(file_path.relative_to(root)) if file_path.is_relative_to(root) else str(file_path),
            "size_bytes": file_path.stat().st_size,
            "sha256": _file_sha256(file_path) if checksums else None,
            "run_id": run_id,
            "config_sha256": config_digest,
        })
    return Table(rows=rows, masked=True)


def _preferred_rows(lightcurve, image_id):
    if lightcurve is None or len(lightcurve) == 0 or "image_id" not in lightcurve.colnames:
        return Table(masked=True)
    return lightcurve[np.asarray(lightcurve["image_id"], dtype=str) == str(image_id)]


def assemble_output_products(
    image_records,
    sources=None,
    photometry=None,
    lightcurve=None,
    batch_products=None,
    diagnostic_stages=None,
    derivatives=None,
    settings=None,
    output_directory=None,
    run_events=None,
    object_name=None,
    profile=None,
    product_overrides=None,
    processed_images=None,
):
    """Assemble final reports, tables, derivatives, configuration, log, and manifest.

    All inputs remain unchanged.  Missing products produce valid empty ECSV
    tables, and failed images still receive reports containing their summary,
    completed stages, and failure page.
    """

    resolved = merge_settings(get_default_settings(), settings or {})
    policy = resolve_output_policy(resolved, profile, product_overrides)
    output = Path(output_directory or resolved["output"]["directory"])
    output.mkdir(parents=True, exist_ok=True)
    table_directory = output
    report_directory = output / "reports"
    fits_directory = output / "fits"
    digest = configuration_hash(resolved)
    run_id = "{}-{}".format(
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"), digest[:12]
    )
    name = _safe_name(
        object_name
        or next((record.get("metadata", {}).get("object") for record in image_records or []
                 if record.get("metadata", {}).get("object")), "field")
    )
    if batch_products:
        photometry = photometry if photometry is not None else batch_products.get("measurements")
        lightcurve = lightcurve if lightcurve is not None else batch_products.get("preferred_light_curve")

    images_table = build_images_table(image_records, digest, run_id)
    sources_table = build_sources_table(sources, digest, run_id)
    photometry_table = add_photometry_traceability(
        photometry, image_records, digest, run_id
    )
    lightcurve_table = add_lightcurve_traceability(
        lightcurve, photometry_table, image_records, digest, run_id
    )

    tables = {
        "images_table": ("images.ecsv", images_table),
        "sources_table": ("sources.ecsv", sources_table),
        "photometry_table": ("photometry.ecsv", photometry_table),
        "lightcurve_table": ("lightcurve.ecsv", lightcurve_table),
    }
    paths = {}
    entries = []
    for product, (filename, table) in tables.items():
        if not policy["products"].get(product, False):
            continue
        path = table_directory / filename
        table.write(path, format="ascii.ecsv", overwrite=policy["overwrite"])
        paths[product] = str(path)
        entries.append((product, str(path), None))

    from .progress import progress

    if policy["products"].get("lightcurve_plot", False) and batch_products:
        progress("final light curve figure")
        try:
            figure_paths = save_light_curve_plot(batch_products, output, name, resolved,
                                                 policy["overwrite"])
            paths["lightcurve_plot"] = figure_paths
            entries.extend(("lightcurve_plot", path, None) for path in figure_paths)
        except Exception as error:  # a figure must never stop the products
            paths["lightcurve_plot_error"] = "{}: {}".format(type(error).__name__, error)

    if processed_images and policy["products"].get("processed_image", False):
        progress("processed FITS images ({})".format(len(processed_images)))
        processed_entries = save_processed_images(
            processed_images, output, policy, digest, run_id)
        paths["processed_images"] = [path for kind, path, _ in processed_entries
                                     if kind == "processed_image"]
        entries.extend(processed_entries)

    if policy["products"].get("image_pdfs", False):
        report_directory.mkdir(exist_ok=True)
        image_paths = {}
        for index, record in enumerate(image_records or []):
            image_id = _record_id(record, index)
            progress("diagnostic report {}/{}: {}".format(index + 1, len(image_records),
                                                          image_id))
            path = report_directory / "{}_diagnostics.pdf".format(_safe_name(image_id))
            stages = (diagnostic_stages or {}).get(image_id, record.get("diagnostics"))
            make_image_diagnostic_pdf(
                record, stages, path, _preferred_rows(lightcurve_table, image_id),
                resolved, policy,
            )
            image_paths[image_id] = str(path)
            entries.append(("image_pdf", str(path), image_id))
        paths["image_pdfs"] = image_paths

    if policy["products"].get("batch_pdf", False) and batch_products:
        report_directory.mkdir(exist_ok=True)
        path = report_directory / "{}_batch_summary.pdf".format(name)
        make_batch_summary_pdf(batch_products, path)
        paths["batch_pdf"] = str(path)
        entries.append(("batch_pdf", str(path), None))

    fits_entries = save_fits_derivatives(
        image_records, derivatives, fits_directory, resolved, policy, digest, run_id
    )
    if fits_entries:
        paths["fits"] = [path for _, path, _ in fits_entries]
        entries.extend(fits_entries)

    if policy["products"].get("resolved_config", False):
        path = output / "resolved_config.json"
        write_resolved_configuration(resolved, path, policy["overwrite"])
        paths["resolved_config"] = str(path)
        entries.append(("resolved_config", str(path), None))

    if policy["products"].get("run_log", False):
        events = list(run_events or [])
        if not events:
            for index, record in enumerate(image_records or []):
                events.append({
                    "level": "ERROR" if _record_status(record) == "FAIL" else "INFO",
                    "image_id": _record_id(record, index), "stage": "final",
                    "message": "status={} flags={}".format(
                        _record_status(record), _record_flags(record) or "none"
                    ),
                })
        path = output / "run.log"
        write_run_log(events, path, run_id, digest, policy["overwrite"])
        paths["run_log"] = str(path)
        entries.append(("run_log", str(path), None))

    manifest = build_output_manifest(
        entries, output, run_id, digest, policy["include_checksums"]
    )
    if policy["products"].get("manifest", False):
        path = output / "manifest.ecsv"
        manifest.write(path, format="ascii.ecsv", overwrite=policy["overwrite"])
        paths["manifest"] = str(path)

    return {
        "paths": paths,
        "policy": deepcopy(policy),
        "manifest": manifest,
        "images": images_table,
        "sources": sources_table,
        "photometry": photometry_table,
        "lightcurve": lightcurve_table,
        "run_id": run_id,
        "config_sha256": digest,
    }


__all__ = [
    "FITS_PRODUCTS",
    "MASK_BITS",
    "SPECIAL_PRODUCTS",
    "save_light_curve_plot",
    "save_processed_images",
    "image_stem",
    "TABLE_PRODUCTS",
    "add_lightcurve_traceability",
    "add_photometry_traceability",
    "assemble_output_products",
    "build_images_table",
    "build_output_manifest",
    "build_sources_table",
    "configuration_hash",
    "make_batch_summary_pdf",
    "make_image_diagnostic_pdf",
    "output_product_enabled",
    "resolve_output_policy",
    "save_fits_derivatives",
    "write_resolved_configuration",
    "write_run_log",
]
