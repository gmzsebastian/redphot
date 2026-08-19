"""Diagnostic plotting functions for redphot processing stages."""

from pathlib import Path

import numpy as np


def _image_limits(array, lower=1.0, upper=99.0):
    """Return finite percentile limits suitable for a diagnostic image."""

    if array is None:
        return None, None
    finite = np.asarray(array, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None, None
    low, high = np.percentile(finite, [lower, upper])
    if low == high:
        high = low + 1.0
    return float(low), float(high)


def _show_image(fig, axis, array, title, cmap="gray", lower=1.0, upper=99.0):
    """Draw an image panel with robust limits and a compact colorbar."""

    axis.set_title(title)
    if array is None:
        axis.text(0.5, 0.5, "Not available", ha="center", va="center")
        axis.set_axis_off()
        return None
    low, high = _image_limits(array, lower=lower, upper=upper)
    image = axis.imshow(
        array,
        origin="lower",
        cmap=cmap,
        vmin=low,
        vmax=high,
        interpolation="nearest",
    )
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set_xlabel("x [pixel]")
    axis.set_ylabel("y [pixel]")
    return image


def plot_background_diagnostics(
    ccd,
    products,
    info,
    metadata=None,
    output_path=None,
    show=False,
):
    """Plot all broad-background products in one nine-panel figure.

    The panels show the input image, expanded background mask, low-resolution
    mesh, interpolated background, RMS map, corrected derivative, row and
    column profiles, and a compact validation summary.

    Parameters
    ----------
    ccd : astropy.nddata.CCDData or numpy.ndarray
        Input image before broad-background subtraction.
    products : mapping
        Products returned by :func:`redphot.image.model_background`.
    info : mapping
        Diagnostic information returned by the same function.
    metadata : mapping, optional
        Normalized image metadata used for the figure title.
    output_path : str or pathlib.Path, optional
        PNG or PDF destination. Parent directories are created as needed.
    show : bool, optional
        Display the figure interactively.

    Returns
    -------
    matplotlib.figure.Figure
        The completed diagnostic figure.
    """

    import matplotlib.pyplot as plt

    data = np.asarray(getattr(ccd, "data", ccd), dtype=float)
    background_mask = products.get("background_mask")
    mesh = products.get("mesh_background")
    mesh_excluded = products.get("mesh_excluded")
    profiles = products.get("profiles", {})

    fig, axes = plt.subplots(3, 3, figsize=(16, 14), constrained_layout=True)
    _show_image(fig, axes[0, 0], data, "Input image")
    _show_image(
        fig,
        axes[0, 1],
        background_mask,
        "Expanded background mask",
        cmap="magma",
        lower=0.0,
        upper=100.0,
    )
    _show_image(fig, axes[0, 2], mesh, "Background mesh", cmap="viridis")
    if mesh is not None and mesh_excluded is not None and np.any(mesh_excluded):
        overlay = np.ma.masked_where(~np.asarray(mesh_excluded, dtype=bool), mesh_excluded)
        axes[0, 2].imshow(
            overlay,
            origin="lower",
            cmap="Reds",
            alpha=0.6,
            interpolation="nearest",
        )

    _show_image(
        fig,
        axes[1, 0],
        products.get("background"),
        "Broad background model",
        cmap="viridis",
    )
    _show_image(
        fig,
        axes[1, 1],
        products.get("background_rms"),
        "Background RMS",
        cmap="viridis",
    )
    _show_image(
        fig,
        axes[1, 2],
        products.get("background_subtracted"),
        "Background-subtracted derivative",
    )

    row_axis = axes[2, 0]
    row_axis.set_title("Median row profiles")
    for key, label in (
        ("row_input", "input"),
        ("row_background", "model"),
        ("row_corrected", "corrected"),
    ):
        values = profiles.get(key)
        if values is not None:
            row_axis.plot(values, label=label, linewidth=1.2)
    row_axis.set_xlabel("row [pixel]")
    row_axis.set_ylabel("median value")
    row_axis.legend(loc="best")
    row_axis.grid(alpha=0.2)

    column_axis = axes[2, 1]
    column_axis.set_title("Median column profiles")
    for key, label in (
        ("column_input", "input"),
        ("column_background", "model"),
        ("column_corrected", "corrected"),
    ):
        values = profiles.get(key)
        if values is not None:
            column_axis.plot(values, label=label, linewidth=1.2)
    column_axis.set_xlabel("column [pixel]")
    column_axis.set_ylabel("median value")
    column_axis.legend(loc="best")
    column_axis.grid(alpha=0.2)

    summary_axis = axes[2, 2]
    summary_axis.set_axis_off()
    before = info.get("gradient_before") or {}
    after = info.get("gradient_after") or {}
    preservation = info.get("source_preservation") or {}
    summary = [
        "Mode: {}".format(info.get("mode")),
        "Measured: {}".format(info.get("measured")),
        "Subtracted: {}".format(info.get("subtracted")),
        "Mesh: {}".format(info.get("effective_box_size")),
        "Excluded meshes: {}".format(
            "--"
            if info.get("excluded_mesh_fraction") is None
            else "{:.1%}".format(info["excluded_mesh_fraction"])
        ),
        "Gradient before: {}".format(
            "--"
            if before.get("peak_to_peak") is None
            else "{:.4g}".format(before["peak_to_peak"])
        ),
        "Gradient after: {}".format(
            "--"
            if after.get("peak_to_peak") is None
            else "{:.4g}".format(after["peak_to_peak"])
        ),
        "Gradient reduction: {}".format(
            "--"
            if info.get("gradient_reduction_fraction") is None
            else "{:.1%}".format(info["gradient_reduction_fraction"])
        ),
        "Median source change: {}".format(
            "--"
            if preservation.get("median_fractional_change") is None
            else "{:.2%}".format(preservation["median_fractional_change"])
        ),
        "Flags: {}".format(", ".join(info.get("flags", [])) or "none"),
    ]
    summary_axis.text(
        0.02,
        0.98,
        "\n".join(summary),
        va="top",
        ha="left",
        family="monospace",
        fontsize=11,
    )

    filename = None if metadata is None else metadata.get("filename")
    title = "Broad-background diagnostics"
    if filename:
        title += " — {}".format(filename)
    fig.suptitle(title, fontsize=15)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_image_quality_diagnostics(
    ccd,
    sources,
    segmentation,
    info,
    metadata=None,
    output_path=None,
    show=False,
):
    """Plot source detections and image-quality measurements.

    The six panels show the source overlay, deblended segmentation image,
    FWHM--ellipticity relation, FWHM distribution, source orientations, and a
    compact PASS/WARN/FAIL summary.  The target-background annulus is drawn on
    the overlay when a target position was available.

    Parameters
    ----------
    ccd : astropy.nddata.CCDData or numpy.ndarray
        Prepared image used for source detection.
    sources : astropy.table.Table
        Source table returned by
        :func:`redphot.image.detect_sources_and_measure_quality`.
    segmentation : numpy.ndarray
        Integer segmentation image returned by the same function.
    info : mapping
        Image-level quality information returned by the same function.
    metadata : mapping, optional
        Normalized image metadata used for the title.
    output_path : str or pathlib.Path, optional
        PNG or PDF destination. Parent directories are created as needed.
    show : bool, optional
        Display the figure interactively.

    Returns
    -------
    matplotlib.figure.Figure
        The completed diagnostic figure.
    """

    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    data = np.asarray(getattr(ccd, "data", ccd), dtype=float)
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)

    overlay = axes[0, 0]
    low, high = _image_limits(data, lower=1.0, upper=99.5)
    overlay.imshow(
        data,
        origin="lower",
        cmap="gray",
        vmin=low,
        vmax=high,
        interpolation="nearest",
    )
    overlay.set_title("Detected sources")
    overlay.set_xlabel("x [pixel]")
    overlay.set_ylabel("y [pixel]")
    if len(sources):
        x = np.asarray(sources["x"], dtype=float)
        y = np.asarray(sources["y"], dtype=float)
        good = np.asarray(sources["good_for_seeing"], dtype=bool)
        saturated = np.asarray(sources["saturated"], dtype=bool)
        ordinary = ~good & ~saturated
        if np.any(ordinary):
            overlay.scatter(
                x[ordinary],
                y[ordinary],
                s=30,
                facecolors="none",
                edgecolors="orange",
                linewidths=0.8,
                label="detected",
            )
        if np.any(good):
            overlay.scatter(
                x[good],
                y[good],
                s=34,
                facecolors="none",
                edgecolors="lime",
                linewidths=0.9,
                label="seeing sample",
            )
        if np.any(saturated):
            overlay.scatter(
                x[saturated],
                y[saturated],
                marker="x",
                s=36,
                color="red",
                linewidths=1.0,
                label="saturated",
            )
        overlay.legend(loc="upper right", fontsize=8)

    target = info.get("local_target_background") or {}
    if target.get("x") is not None:
        for radius, color in (
            (target.get("inner_radius_pixels"), "cyan"),
            (target.get("outer_radius_pixels"), "cyan"),
        ):
            if radius is not None:
                overlay.add_patch(
                    Circle(
                        (target["x"], target["y"]),
                        radius,
                        fill=False,
                        edgecolor=color,
                        linewidth=1.0,
                        linestyle="--",
                    )
                )

    _show_image(
        fig,
        axes[0, 1],
        segmentation,
        "Source segmentation",
        cmap="nipy_spectral",
        lower=0.0,
        upper=100.0,
    )

    shape_axis = axes[0, 2]
    shape_axis.set_title("Source shape measurements")
    if len(sources):
        fwhm = np.asarray(sources["fwhm_arcsec"], dtype=float)
        if not np.any(np.isfinite(fwhm)):
            fwhm = np.asarray(sources["fwhm_pixels"], dtype=float)
            x_label = "FWHM [pixel]"
        else:
            x_label = "FWHM [arcsec]"
        ellipticity = np.asarray(sources["ellipticity"], dtype=float)
        good = np.asarray(sources["good_for_seeing"], dtype=bool)
        shape_axis.scatter(
            fwhm[~good],
            ellipticity[~good],
            s=16,
            color="0.6",
            alpha=0.7,
            label="excluded",
        )
        shape_axis.scatter(
            fwhm[good],
            ellipticity[good],
            s=20,
            color="tab:blue",
            alpha=0.8,
            label="seeing sample",
        )
        shape_axis.set_xlabel(x_label)
        shape_axis.legend(loc="best", fontsize=8)
    else:
        shape_axis.text(0.5, 0.5, "No sources", ha="center", va="center")
        shape_axis.set_xlabel("FWHM")
    shape_axis.set_ylabel("ellipticity")
    shape_axis.grid(alpha=0.2)

    fwhm_axis = axes[1, 0]
    fwhm_axis.set_title("FWHM distribution")
    if len(sources):
        values = np.asarray(sources["fwhm_arcsec"], dtype=float)
        units = "arcsec"
        if not np.any(np.isfinite(values)):
            values = np.asarray(sources["fwhm_pixels"], dtype=float)
            units = "pixel"
        values = values[np.isfinite(values)]
        if values.size:
            fwhm_axis.hist(values, bins="auto", color="tab:blue", alpha=0.8)
        fwhm_axis.set_xlabel("FWHM [{}]".format(units))
    else:
        fwhm_axis.text(0.5, 0.5, "No sources", ha="center", va="center")
    fwhm_axis.set_ylabel("source count")
    fwhm_axis.grid(alpha=0.2)

    orientation_axis = axes[1, 1]
    orientation_axis.set_title("Elongated-source orientations")
    if len(sources):
        ellipticity = np.asarray(sources["ellipticity"], dtype=float)
        orientation = np.asarray(sources["orientation_deg"], dtype=float) % 180.0
        elongated_limit = float(
            info.get("elongated_source_ellipticity", 0.35)
        )
        selected = np.isfinite(orientation) & (ellipticity >= elongated_limit)
        if np.any(selected):
            orientation_axis.hist(
                orientation[selected],
                bins=np.linspace(0, 180, 13),
                color="tab:orange",
                alpha=0.85,
            )
        else:
            orientation_axis.text(
                0.5,
                0.5,
                "No elongated sources",
                ha="center",
                va="center",
            )
    else:
        orientation_axis.text(0.5, 0.5, "No sources", ha="center", va="center")
    orientation_axis.set_xlim(0, 180)
    orientation_axis.set_xlabel("orientation [deg]")
    orientation_axis.set_ylabel("source count")
    orientation_axis.grid(alpha=0.2)

    summary_axis = axes[1, 2]
    summary_axis.set_axis_off()

    def formatted(value, format_string):
        return "--" if value is None else format_string.format(value)

    local = info.get("local_target_background") or {}
    summary = [
        "Status: {}".format(info.get("quality_status", "--")),
        "Sources: {} (seeing: {})".format(
            info.get("source_count", 0), info.get("seeing_source_count", 0)
        ),
        "Background: {}".format(formatted(info.get("background"), "{:.4g}")),
        "Background RMS: {}".format(
            formatted(info.get("background_rms"), "{:.4g}")
        ),
        "FWHM: {} arcsec".format(
            formatted(info.get("fwhm_arcsec"), "{:.3f}")
        ),
        "FWHM scatter: {}".format(
            formatted(info.get("fwhm_scatter_fraction"), "{:.1%}")
        ),
        "Ellipticity: {} ± {}".format(
            formatted(info.get("ellipticity"), "{:.3f}"),
            formatted(info.get("ellipticity_scatter"), "{:.3f}"),
        ),
        "Aligned elongated: {}".format(info.get("globally_elongated")),
        "Saturated sources: {}".format(
            info.get("saturated_source_count", 0)
        ),
        "Masked pixels: {}".format(
            formatted(info.get("masked_pixel_fraction"), "{:.1%}")
        ),
        "Trail pixels: {}".format(
            formatted(info.get("trail_fraction"), "{:.2%}")
        ),
        "Target background: {}".format(
            formatted(local.get("background"), "{:.4g}")
        ),
        "Flags: {}".format(", ".join(info.get("quality_flags", [])) or "none"),
    ]
    failed_checks = [
        "{}: {}".format(check.get("status"), check.get("metric"))
        for check in info.get("checks", [])
    ]
    if failed_checks:
        summary.extend(["", "Triggered limits:"] + failed_checks[:8])
    summary_axis.text(
        0.02,
        0.98,
        "\n".join(summary),
        va="top",
        ha="left",
        family="monospace",
        fontsize=10,
    )

    filename = None if metadata is None else metadata.get("filename")
    title = "Source detection and image quality"
    if filename:
        title += " — {}".format(filename)
    fig.suptitle(title, fontsize=15)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_astrometry_diagnostics(
    ccd,
    catalog,
    matches,
    info,
    metadata=None,
    output_path=None,
    show=False,
):
    """Plot catalog overlays and WCS residual diagnostics.

    Panels show the projected catalog, matched-source residual vectors,
    original and final RA/Dec residuals, radial residual distributions,
    residuals across the detector, and a compact refinement summary.
    """

    import matplotlib.pyplot as plt

    data = np.asarray(getattr(ccd, "data", ccd), dtype=float)
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)

    overlay = axes[0, 0]
    low, high = _image_limits(data, lower=1.0, upper=99.5)
    overlay.imshow(
        data,
        origin="lower",
        cmap="gray",
        vmin=low,
        vmax=high,
        interpolation="nearest",
    )
    overlay.set_title("Catalog overlay")
    if len(catalog) and "in_image" in catalog.colnames:
        inside = np.asarray(catalog["in_image"], dtype=bool)
        overlay.scatter(
            np.asarray(catalog["x"], dtype=float)[inside],
            np.asarray(catalog["y"], dtype=float)[inside],
            s=25,
            facecolors="none",
            edgecolors="cyan",
            linewidths=0.7,
            label="catalog",
        )
    if len(matches):
        overlay.scatter(
            np.asarray(matches["x"], dtype=float),
            np.asarray(matches["y"], dtype=float),
            marker="+",
            s=28,
            color="lime",
            linewidths=0.8,
            label="matched detections",
        )
    if len(catalog) or len(matches):
        overlay.legend(loc="upper right", fontsize=8)
    overlay.set_xlabel("x [pixel]")
    overlay.set_ylabel("y [pixel]")

    vector_axis = axes[0, 1]
    vector_axis.imshow(
        data,
        origin="lower",
        cmap="gray",
        vmin=low,
        vmax=high,
        interpolation="nearest",
    )
    vector_axis.set_title("Original WCS residual vectors (×20)")
    if len(matches):
        x = np.asarray(matches["x"], dtype=float)
        y = np.asarray(matches["y"], dtype=float)
        dx = np.asarray(matches["catalog_x_original"], dtype=float) - x
        dy = np.asarray(matches["catalog_y_original"], dtype=float) - y
        vector_axis.quiver(
            x,
            y,
            dx,
            dy,
            color="yellow",
            angles="xy",
            scale_units="xy",
            scale=0.05,
            width=0.0025,
        )
    vector_axis.set_xlabel("x [pixel]")
    vector_axis.set_ylabel("y [pixel]")

    residual_axis = axes[0, 2]
    residual_axis.set_title("Sky residuals")
    if len(matches):
        original_ra = np.asarray(
            matches["residual_ra_original_arcsec"], dtype=float
        )
        original_dec = np.asarray(
            matches["residual_dec_original_arcsec"], dtype=float
        )
        residual_axis.scatter(
            original_ra,
            original_dec,
            s=22,
            color="0.6",
            alpha=0.7,
            label="original",
        )
        if "residual_ra_final_arcsec" in matches.colnames:
            final_ra = np.asarray(matches["residual_ra_final_arcsec"], dtype=float)
            final_dec = np.asarray(matches["residual_dec_final_arcsec"], dtype=float)
            inlier = np.asarray(matches["inlier"], dtype=bool)
            residual_axis.scatter(
                final_ra[inlier],
                final_dec[inlier],
                s=24,
                color="tab:blue",
                alpha=0.8,
                label="final inliers",
            )
            if np.any(~inlier):
                residual_axis.scatter(
                    final_ra[~inlier],
                    final_dec[~inlier],
                    marker="x",
                    s=30,
                    color="red",
                    label="rejected",
                )
        residual_axis.legend(loc="best", fontsize=8)
    residual_axis.axhline(0, color="0.3", linewidth=0.7)
    residual_axis.axvline(0, color="0.3", linewidth=0.7)
    residual_axis.set_xlabel("RA residual [arcsec]")
    residual_axis.set_ylabel("Dec residual [arcsec]")
    residual_axis.grid(alpha=0.2)

    histogram_axis = axes[1, 0]
    histogram_axis.set_title("Radial residual distribution")
    if len(matches):
        original = np.asarray(
            matches["separation_original_arcsec"], dtype=float
        )
        histogram_axis.hist(
            original[np.isfinite(original)],
            bins="auto",
            histtype="step",
            linewidth=1.5,
            color="0.4",
            label="original",
        )
        if "separation_final_arcsec" in matches.colnames:
            final = np.asarray(matches["separation_final_arcsec"], dtype=float)
            inlier = np.asarray(matches["inlier"], dtype=bool)
            histogram_axis.hist(
                final[inlier & np.isfinite(final)],
                bins="auto",
                alpha=0.65,
                color="tab:blue",
                label="final inliers",
            )
        histogram_axis.legend(loc="best", fontsize=8)
    histogram_axis.set_xlabel("radial residual [arcsec]")
    histogram_axis.set_ylabel("match count")
    histogram_axis.grid(alpha=0.2)

    field_axis = axes[1, 1]
    field_axis.set_title("Final residual across detector")
    if len(matches) and "separation_final_arcsec" in matches.colnames:
        x = np.asarray(matches["x"], dtype=float)
        radial = np.asarray(matches["separation_final_arcsec"], dtype=float)
        inlier = np.asarray(matches["inlier"], dtype=bool)
        field_axis.scatter(
            x[inlier],
            radial[inlier],
            s=22,
            color="tab:blue",
            alpha=0.8,
            label="inlier",
        )
        if np.any(~inlier):
            field_axis.scatter(
                x[~inlier],
                radial[~inlier],
                marker="x",
                s=30,
                color="red",
                label="rejected",
            )
        field_axis.legend(loc="best", fontsize=8)
    field_axis.set_xlabel("x [pixel]")
    field_axis.set_ylabel("radial residual [arcsec]")
    field_axis.grid(alpha=0.2)

    summary_axis = axes[1, 2]
    summary_axis.set_axis_off()

    def formatted(value, format_string):
        return "--" if value is None else format_string.format(value)

    target_original = info.get("target_original") or {}
    target_refined = info.get("target_refined") or {}
    summary = [
        "Status: {}".format(info.get("quality_status", "--")),
        "Catalog: {}".format(info.get("catalog_name", "--")),
        "Catalog rows: {} (image: {})".format(
            info.get("catalog_row_count", "--"),
            info.get("catalog_in_image_count", "--"),
        ),
        "Matches: {} (inliers: {}, rejected: {})".format(
            info.get("match_count", 0),
            info.get("inlier_count", 0),
            info.get("rejected_match_count", 0),
        ),
        "Original RMS: {} arcsec".format(
            formatted(info.get("original_rms_arcsec"), "{:.3f}")
        ),
        "Final RMS: {} arcsec".format(
            formatted(info.get("refined_rms_arcsec"), "{:.3f}")
        ),
        "Refinement adopted: {}".format(info.get("refinement_adopted", False)),
        "Reason: {}".format(info.get("refinement_reason", "--")),
        "Translation: {} pixel".format(
            formatted(info.get("translation_pixels"), "{:.3f}")
        ),
        "Rotation: {} deg".format(
            formatted(info.get("rotation_degrees"), "{:.4f}")
        ),
        "Scale change: {}".format(
            formatted(info.get("scale_change_fraction"), "{:.3%}")
        ),
        "Target round trip (original): {} arcsec".format(
            formatted(target_original.get("round_trip_error_arcsec"), "{:.3g}")
        ),
        "Target round trip (final): {} arcsec".format(
            formatted(target_refined.get("round_trip_error_arcsec"), "{:.3g}")
        ),
        "Flags: {}".format(", ".join(info.get("flags", [])) or "none"),
    ]
    summary_axis.text(
        0.02,
        0.98,
        "\n".join(summary),
        va="top",
        ha="left",
        family="monospace",
        fontsize=10,
    )

    filename = None if metadata is None else metadata.get("filename")
    title = "Catalog matching and WCS refinement"
    if filename:
        title += " — {}".format(filename)
    fig.suptitle(title, fontsize=15)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_star_selection_diagnostics(
    ccd,
    measurements,
    image_id,
    summary=None,
    metadata=None,
    output_path=None,
    show=False,
):
    """Plot accepted, rejected, and role-assigned stars for one image.

    Rejected candidates remain visible with their reason categories, while
    independent plotting symbols show astrometry, PSF, calibration, ensemble,
    and bright quality-control roles.
    """

    import matplotlib.pyplot as plt

    data = np.asarray(getattr(ccd, "data", ccd), dtype=float)
    rows = measurements[np.asarray(measurements["image_id"] == str(image_id))]
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=True)
    low, high = _image_limits(data, lower=1.0, upper=99.5)

    overlay = axes[0, 0]
    overlay.imshow(
        data,
        origin="lower",
        cmap="gray",
        vmin=low,
        vmax=high,
        interpolation="nearest",
    )
    overlay.set_title("Accepted and rejected candidates")
    if len(rows):
        x = np.asarray(rows["x"], dtype=float)
        y = np.asarray(rows["y"], dtype=float)
        accepted = np.asarray(rows["image_accepted"], dtype=bool)
        if np.any(~accepted):
            overlay.scatter(
                x[~accepted],
                y[~accepted],
                marker="x",
                s=38,
                color="red",
                linewidths=0.9,
                label="rejected",
            )
        if np.any(accepted):
            overlay.scatter(
                x[accepted],
                y[accepted],
                s=34,
                facecolors="none",
                edgecolors="lime",
                linewidths=0.9,
                label="accepted",
            )
        overlay.legend(loc="upper right", fontsize=8)
    overlay.set_xlabel("x [pixel]")
    overlay.set_ylabel("y [pixel]")

    role_axis = axes[0, 1]
    role_axis.imshow(
        data,
        origin="lower",
        cmap="gray",
        vmin=low,
        vmax=high,
        interpolation="nearest",
    )
    role_axis.set_title("Independent star roles")
    role_styles = {
        "astrometry": ("o", "cyan"),
        "psf": ("s", "yellow"),
        "calibration": ("^", "lime"),
        "ensemble": ("D", "magenta"),
        "qc_anchor": ("*", "orange"),
    }
    if len(rows):
        x = np.asarray(rows["x"], dtype=float)
        y = np.asarray(rows["y"], dtype=float)
        for role, (marker, color) in role_styles.items():
            selected = np.asarray(rows["role_{}".format(role)], dtype=bool)
            if not np.any(selected):
                continue
            role_axis.scatter(
                x[selected],
                y[selected],
                marker=marker,
                s=48 if role != "qc_anchor" else 90,
                facecolors="none" if role != "qc_anchor" else color,
                edgecolors=color,
                linewidths=1.0,
                label=role.replace("_", " "),
            )
        role_axis.legend(loc="upper right", fontsize=8)
    role_axis.set_xlabel("x [pixel]")
    role_axis.set_ylabel("y [pixel]")

    reason_axis = axes[1, 0]
    reason_axis.set_title("Rejection reasons")
    reason_counts = {}
    for value in rows["rejection_reasons"] if len(rows) else []:
        for reason in str(value).split(";"):
            if reason:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
    if reason_counts:
        ordered = sorted(reason_counts, key=reason_counts.get)
        reason_axis.barh(
            ordered,
            [reason_counts[reason] for reason in ordered],
            color="tab:red",
            alpha=0.8,
        )
    else:
        reason_axis.text(
            0.5, 0.5, "No rejected candidates", ha="center", va="center"
        )
    reason_axis.set_xlabel("candidate count")
    reason_axis.grid(axis="x", alpha=0.2)

    summary_axis = axes[1, 1]
    summary_axis.set_axis_off()
    if summary is None:
        summary = {}
    role_counts = summary.get("role_counts", {})
    lines = [
        "Image: {}".format(image_id),
        "Candidates: {}".format(summary.get("candidate_count", len(rows))),
        "Strictly accepted: {}".format(
            summary.get(
                "strictly_accepted_count",
                int(np.count_nonzero(rows["image_accepted"])) if len(rows) else 0,
            )
        ),
        "Rejected: {}".format(
            summary.get(
                "rejected_count",
                int(np.count_nonzero(~rows["image_accepted"])) if len(rows) else 0,
            )
        ),
        "",
        "Role counts:",
    ]
    for role in role_styles:
        count = role_counts.get(
            role,
            int(np.count_nonzero(rows["role_{}".format(role)])) if len(rows) else 0,
        )
        lines.append("  {:12s} {}".format(role, count))
    lines.extend(
        [
            "",
            "Selection flags: {}".format(
                ", ".join(summary.get("flags", [])) or "none"
            ),
            "Image rejected: {}".format(summary.get("image_rejected", False)),
        ]
    )
    summary_axis.text(
        0.02,
        0.98,
        "\n".join(lines),
        va="top",
        ha="left",
        family="monospace",
        fontsize=11,
    )

    filename = None if metadata is None else metadata.get("filename")
    title = "Comparison and PSF star selection"
    if filename:
        title += " — {}".format(filename)
    fig.suptitle(title, fontsize=15)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_image_usability_diagnostics(
    ccd,
    decision,
    star_residuals=None,
    metadata=None,
    output_path=None,
    show=False,
):
    """Plot the complete image-usability review gate for one exposure.

    The figure combines the target and artifact context, catalog recovery,
    quick stellar zeropoints, limiting depth, global image-quality metrics,
    and the effective automatic or manual decision.
    """

    import matplotlib.pyplot as plt

    data = np.asarray(getattr(ccd, "data", ccd), dtype=float)
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), constrained_layout=True)
    low, high = _image_limits(data, lower=1.0, upper=99.5)

    image_axis = axes[0, 0]
    image_axis.imshow(
        data,
        origin="lower",
        cmap="gray",
        vmin=low,
        vmax=high,
        interpolation="nearest",
    )
    image_axis.set_title("Target and artifact check")
    artifacts = decision.get("target_artifacts", {})
    position = artifacts.get("position")
    if position is not None:
        from matplotlib.patches import Circle

        radius = artifacts.get("radius_pixels", 5.0)
        overlap = bool(artifacts.get("any"))
        image_axis.add_patch(
            Circle(
                position,
                radius,
                fill=False,
                color="red" if overlap else "lime",
                linewidth=1.5,
            )
        )
        image_axis.scatter(
            [position[0]], [position[1]], marker="+", s=80,
            color="red" if overlap else "lime",
        )
    image_axis.set_xlabel("x [pixel]")
    image_axis.set_ylabel("y [pixel]")

    recovery_axis = axes[0, 1]
    recovery_axis.set_title("Catalog and calibration recovery")
    expected = decision.get("catalog_expected_count") or 0
    recovered = decision.get("catalog_recovered_count") or 0
    calibrated = decision.get("calibration_star_count") or 0
    recovery_axis.bar(
        ["expected", "recovered", "calibration"],
        [expected, recovered, calibrated],
        color=["0.65", "tab:blue", "tab:green"],
    )
    recovery_axis.set_ylabel("star count")
    fraction = decision.get("catalog_recovery_fraction")
    recovery_axis.text(
        0.98,
        0.96,
        "Recovery: {}\nPersistent QC anchor: {}\nFallback available: {}".format(
            "--" if fraction is None else "{:.1%}".format(fraction),
            "yes" if decision.get("qc_star_recovered") else "no",
            "yes" if decision.get("qc_fallback_available") else "no",
        ),
        transform=recovery_axis.transAxes,
        ha="right",
        va="top",
    )
    recovery_axis.grid(axis="y", alpha=0.2)

    residual_axis = axes[0, 2]
    residual_axis.set_title("Comparison-star transparency residuals")
    rows = star_residuals
    if rows is not None and len(rows) and "image_id" in rows.colnames:
        rows = rows[
            np.asarray(rows["image_id"], dtype=str) == str(decision["image_id"])
        ]
    if rows is not None and len(rows):
        inlier = np.asarray(rows["inlier"], dtype=bool)
        residual = np.ma.asarray(rows["spatial_residual_mag"], dtype=float)
        valid = inlier & ~np.ma.getmaskarray(residual)
        if np.any(valid):
            points = residual_axis.scatter(
                np.asarray(rows["x"], dtype=float)[valid],
                np.asarray(rows["y"], dtype=float)[valid],
                c=np.asarray(residual.filled(np.nan), dtype=float)[valid],
                cmap="coolwarm_r",
                s=48,
                edgecolors="black",
                linewidths=0.3,
            )
            fig.colorbar(points, ax=residual_axis, label="zeropoint residual [mag]")
        rejected = ~inlier
        if np.any(rejected):
            residual_axis.scatter(
                np.asarray(rows["x"], dtype=float)[rejected],
                np.asarray(rows["y"], dtype=float)[rejected],
                marker="x",
                color="0.4",
                s=25,
                label="clipped/rejected",
            )
            residual_axis.legend(fontsize=8)
        residual_axis.set_xlim(0, data.shape[1] - 1)
        residual_axis.set_ylim(0, data.shape[0] - 1)
    else:
        residual_axis.text(
            0.5, 0.5, "No stellar residuals", ha="center", va="center"
        )
    residual_axis.set_xlabel("x [pixel]")
    residual_axis.set_ylabel("y [pixel]")

    depth_axis = axes[1, 0]
    depth_axis.set_title("Quick limiting depth")
    labels = []
    values = []
    colors = []
    for region, color in (("global", "tab:blue"), ("local", "tab:orange")):
        for sigma_label, value in decision.get(
            "{}_depths_mag".format(region), {}
        ).items():
            if value is not None:
                labels.append("{}\n{}".format(region, sigma_label))
                values.append(value)
                colors.append(color)
    if values:
        depth_axis.bar(labels, values, color=colors, alpha=0.85)
        lower = min(values) - 1.0
        upper = max(values) + 1.0
        expected_magnitude = decision.get("expected_target_magnitude")
        if expected_magnitude is not None:
            depth_axis.axhline(
                expected_magnitude,
                color="red",
                linestyle="--",
                label="expected target",
            )
            lower = min(lower, expected_magnitude - 0.5)
            upper = max(upper, expected_magnitude + 0.5)
            depth_axis.legend(fontsize=8)
        depth_axis.set_ylim(lower, upper)
        depth_axis.invert_yaxis()
    else:
        depth_axis.text(
            0.5, 0.5, "Depth unavailable", ha="center", va="center"
        )
    depth_axis.set_ylabel("limiting magnitude")
    depth_axis.grid(axis="y", alpha=0.2)

    metric_axis = axes[1, 1]
    metric_axis.set_axis_off()
    metric_lines = [
        "Quick zeropoint: {}".format(
            "--" if decision.get("zeropoint_mag") is None
            else "{:.3f} mag".format(decision["zeropoint_mag"])
        ),
        "Zeropoint scatter: {}".format(
            "--" if decision.get("zeropoint_scatter_mag") is None
            else "{:.3f} mag".format(decision["zeropoint_scatter_mag"])
        ),
        "Transparency loss: {}".format(
            "--" if decision.get("transparency_attenuation_mag") is None
            else "{:.3f} mag".format(decision["transparency_attenuation_mag"])
        ),
        "Spatial cloud amplitude: {}".format(
            "--" if decision.get("spatial_cloud_amplitude_mag") is None
            else "{:.3f} mag".format(decision["spatial_cloud_amplitude_mag"])
        ),
        "Seeing: {}".format(
            "--" if decision.get("fwhm_arcsec") is None
            else "{:.2f} arcsec".format(decision["fwhm_arcsec"])
        ),
        "Ellipticity: {}".format(
            "--" if decision.get("ellipticity") is None
            else "{:.3f}".format(decision["ellipticity"])
        ),
        "Background / RMS: {} / {}".format(
            "--" if decision.get("background") is None
            else "{:.4g}".format(decision["background"]),
            "--" if decision.get("background_rms") is None
            else "{:.4g}".format(decision["background_rms"]),
        ),
        "Masked / trail fraction: {} / {}".format(
            "--" if decision.get("masked_pixel_fraction") is None
            else "{:.1%}".format(decision["masked_pixel_fraction"]),
            "--" if decision.get("trail_fraction") is None
            else "{:.1%}".format(decision["trail_fraction"]),
        ),
    ]
    metric_axis.text(
        0.02, 0.98, "\n".join(metric_lines), va="top", ha="left",
        family="monospace", fontsize=10,
    )

    decision_axis = axes[1, 2]
    decision_axis.set_axis_off()
    status = decision.get("status", "WARN")
    status_color = {"PASS": "tab:green", "WARN": "darkorange", "FAIL": "tab:red"}.get(
        status, "0.4"
    )
    decision_axis.text(
        0.02, 0.98, status, va="top", ha="left", fontsize=24,
        fontweight="bold", color=status_color,
    )
    lines = [
        "Automatic: {}".format(decision.get("automatic_status")),
        "Review: {}".format(decision.get("review_state")),
        "Use image: {}".format(decision.get("use_image")),
        "Source: {}".format(decision.get("decision_source")),
    ]
    if decision.get("review_note"):
        lines.append("Note: {}".format(decision["review_note"]))
    lines.append("")
    lines.extend(decision.get("reasons", []) or ["No warnings or failures"])
    decision_axis.text(
        0.02, 0.84, "\n".join(lines), va="top", ha="left",
        fontsize=9, wrap=True,
    )

    filename = None if metadata is None else metadata.get("filename")
    if filename is None:
        filename = decision.get("filename")
    title = "Image usability and limiting-depth review"
    if filename:
        title += " — {}".format(filename)
    fig.suptitle(title, fontsize=15)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


__all__ = [
    "plot_background_diagnostics",
    "plot_astrometry_diagnostics",
    "plot_image_quality_diagnostics",
    "plot_image_usability_diagnostics",
    "plot_star_selection_diagnostics",
]
