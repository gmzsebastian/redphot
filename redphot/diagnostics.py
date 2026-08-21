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


def plot_alignment_target_diagnostics(
    stacks,
    target_solution,
    target_candidates,
    projection_table,
    output_path=None,
    show=False,
):
    """Plot detection stacks, centroid offsets, and fixed-position checks."""

    import matplotlib.pyplot as plt
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    fig, axes = plt.subplots(2, 3, figsize=(18, 11), constrained_layout=True)
    stack_items = list(stacks.items())
    stack_items.sort(key=lambda item: (item[0] != "multifilter", item[0]))
    final_coordinate = SkyCoord(
        target_solution["ra_deg"], target_solution["dec_deg"], unit="deg"
    )
    for panel, axis in enumerate(axes[0]):
        if panel >= len(stack_items):
            axis.set_axis_off()
            continue
        name, product = stack_items[panel]
        data = np.asarray(product["data"], dtype=float)
        low, high = _image_limits(data, lower=1.0, upper=99.7)
        axis.imshow(
            data,
            origin="lower",
            cmap="gray",
            vmin=low,
            vmax=high,
            interpolation="nearest",
        )
        x, y = product["wcs"].world_to_pixel(final_coordinate)
        axis.scatter([x], [y], marker="+", color="lime", s=100, linewidths=1.5)
        axis.set_xlim(max(0, x - 40), min(data.shape[1] - 1, x + 40))
        axis.set_ylim(max(0, y - 40), min(data.shape[0] - 1, y + 40))
        axis.set_title("{} detection stack".format(name))
        axis.set_xlabel("x [pixel]")
        axis.set_ylabel("y [pixel]")

    offset_axis = axes[1, 0]
    offset_axis.set_title("Diagnostic centroid offsets")
    if target_candidates is not None and len(target_candidates):
        finite = (
            ~np.ma.getmaskarray(target_candidates["ra_deg"])
            & ~np.ma.getmaskarray(target_candidates["dec_deg"])
        )
        rows = target_candidates[finite]
        if len(rows):
            coordinates = SkyCoord(
                np.asarray(rows["ra_deg"], dtype=float) * u.deg,
                np.asarray(rows["dec_deg"], dtype=float) * u.deg,
            )
            longitude, latitude = final_coordinate.spherical_offsets_to(coordinates)
            accepted = np.asarray(rows["accepted"], dtype=bool)
            used = np.asarray(rows["used_in_solution"], dtype=bool)
            if np.any(~accepted):
                offset_axis.scatter(
                    longitude.arcsec[~accepted], latitude.arcsec[~accepted],
                    marker="x", color="tab:red", s=45, label="rejected",
                )
            if np.any(accepted & ~used):
                offset_axis.scatter(
                    longitude.arcsec[accepted & ~used],
                    latitude.arcsec[accepted & ~used],
                    facecolors="none", edgecolors="tab:blue", s=55,
                    label="accepted diagnostic",
                )
            if np.any(used):
                offset_axis.scatter(
                    longitude.arcsec[used], latitude.arcsec[used],
                    marker="*", color="tab:green", s=100, label="used",
                )
            for row, x_value, y_value in zip(
                rows, longitude.arcsec, latitude.arcsec
            ):
                offset_axis.annotate(
                    str(row["source"]),
                    (x_value, y_value),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=6,
                )
            offset_axis.legend(fontsize=8)
    offset_axis.axhline(0, color="0.5", linewidth=0.7)
    offset_axis.axvline(0, color="0.5", linewidth=0.7)
    offset_axis.set_xlabel("RA offset from frozen position [arcsec]")
    offset_axis.set_ylabel("Dec offset from frozen position [arcsec]")
    offset_axis.grid(alpha=0.2)
    offset_axis.set_aspect("equal", adjustable="datalim")

    projection_axis = axes[1, 1]
    projection_axis.set_title("Relative alignment at fixed position")
    if projection_table is not None and len(projection_table):
        image_ids = [str(value) for value in projection_table["image_id"]]
        values = np.ma.asarray(
            projection_table["relative_alignment_rms_arcsec"], dtype=float
        )
        numeric = np.asarray(values.filled(np.nan), dtype=float)
        colors = [
            {"PASS": "tab:green", "WARN": "darkorange", "FAIL": "tab:red"}.get(
                str(value), "0.5"
            )
            for value in projection_table["status"]
        ]
        positions = np.arange(len(image_ids))
        projection_axis.bar(positions, np.nan_to_num(numeric), color=colors)
        projection_axis.set_xticks(positions)
        projection_axis.set_xticklabels(image_ids, rotation=45, ha="right", fontsize=7)
    else:
        projection_axis.text(
            0.5, 0.5, "No projection checks", ha="center", va="center"
        )
    projection_axis.set_ylabel("common-star RMS [arcsec]")
    projection_axis.grid(axis="y", alpha=0.2)

    summary_axis = axes[1, 2]
    summary_axis.set_axis_off()
    summary_lines = [
        "Version: {}".format(target_solution.get("version")),
        "Frozen: {}".format(target_solution.get("frozen")),
        "RA: {:.8f} deg".format(target_solution["ra_deg"]),
        "Dec: {:+.8f} deg".format(target_solution["dec_deg"]),
        "Uncertainty: {:.3f} arcsec".format(
            target_solution["uncertainty_arcsec"]
        ),
        "Status: {}".format(target_solution.get("status")),
        "Candidates used: {}/{}".format(
            target_solution.get("used_candidate_count"),
            target_solution.get("candidate_count"),
        ),
        "Free centroids: diagnostic only",
        "",
        "Provenance:",
    ]
    summary_lines.extend(
        "  {}".format(value) for value in target_solution.get("provenance", [])
    )
    summary_lines.extend(
        ["", "Flags: {}".format(", ".join(target_solution.get("flags", [])) or "none")]
    )
    summary_axis.text(
        0.02, 0.98, "\n".join(summary_lines), va="top", ha="left",
        family="monospace", fontsize=10,
    )
    fig.suptitle("Relative alignment and frozen target coordinate", fontsize=15)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.25)
    if show:
        plt.show()
    return fig


def _image_mosaic(cube, columns=4, gap=1):
    """Arrange a small image cube into a compact diagnostic mosaic."""

    cube = np.asarray(cube, dtype=float)
    if cube.ndim != 3 or cube.shape[0] == 0:
        return None
    columns = max(1, min(int(columns), cube.shape[0]))
    rows = int(np.ceil(cube.shape[0] / columns))
    height, width = cube.shape[1:]
    mosaic = np.full(
        (rows * height + (rows - 1) * gap, columns * width + (columns - 1) * gap),
        np.nan,
    )
    for index, image in enumerate(cube):
        row, column = divmod(index, columns)
        y0 = row * (height + gap)
        x0 = column * (width + gap)
        mosaic[y0:y0 + height, x0:x0 + width] = image
    return mosaic


def plot_psf_diagnostics(result, output_path=None, show=False):
    """Plot PSF-star cutouts, model, profiles, residuals, and review gate.

    Parameters
    ----------
    result : mapping
        Result returned by :func:`redphot.photometry.construct_psf`.
    output_path : str or pathlib.Path, optional
        PNG or PDF destination. Parent directories are created when needed.
    show : bool, optional
        Display the figure interactively.

    Returns
    -------
    matplotlib.figure.Figure
        The completed six-panel diagnostic figure.
    """

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    grid = fig.add_gridspec(2, 3)
    cutout_axis = fig.add_subplot(grid[0, 0])
    model_axis = fig.add_subplot(grid[0, 1])
    surface_axis = fig.add_subplot(grid[0, 2], projection="3d")
    profile_axis = fig.add_subplot(grid[1, 0])
    residual_axis = fig.add_subplot(grid[1, 1])
    summary_axis = fig.add_subplot(grid[1, 2])

    cutout_mosaic = _image_mosaic(result.get("cutouts", []))
    _show_image(fig, cutout_axis, cutout_mosaic, "Accepted PSF-star cutouts")
    model = result.get("model_native")
    _show_image(fig, model_axis, model, "Normalized PSF model", cmap="viridis")

    surface_axis.set_title("PSF surface")
    if model is None:
        surface_axis.text2D(0.5, 0.5, "Not available", ha="center", va="center")
    else:
        model_array = np.asarray(model, dtype=float)
        yy, xx = np.indices(model_array.shape)
        surface_axis.plot_surface(
            xx, yy, model_array, cmap="viridis", linewidth=0, antialiased=True
        )
        surface_axis.set_xlabel("x [pixel]")
        surface_axis.set_ylabel("y [pixel]")
        surface_axis.set_zlabel("normalized value")

    profile_axis.set_title("Radial and detector-axis profiles")
    if model is not None:
        model_array = np.asarray(model, dtype=float)
        cy = (model_array.shape[0] - 1) / 2.0
        cx = (model_array.shape[1] - 1) / 2.0
        yy, xx = np.indices(model_array.shape, dtype=float)
        radius = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        radial_index = np.floor(radius).astype(int)
        radial = np.array(
            [
                np.nanmean(model_array[radial_index == index])
                for index in range(radial_index.max() + 1)
            ]
        )
        profile_axis.plot(radial, "o-", label="radial", markersize=3)
        profile_axis.plot(
            np.arange(model_array.shape[1]) - cx,
            model_array[int(round(cy)), :],
            label="x axis",
        )
        profile_axis.plot(
            np.arange(model_array.shape[0]) - cy,
            model_array[:, int(round(cx))],
            label="y axis",
        )
        profile_axis.legend(loc="best")
    profile_axis.set_xlabel("distance [pixel]")
    profile_axis.set_ylabel("normalized value")
    profile_axis.grid(alpha=0.2)

    residual_mosaic = _image_mosaic(result.get("residuals", []))
    _show_image(
        fig,
        residual_axis,
        residual_mosaic,
        "PSF-star residuals",
        cmap="coolwarm",
        lower=2.0,
        upper=98.0,
    )

    summary_axis.set_axis_off()
    summary = [
        "Model: {}".format(result.get("model_type") or "failed"),
        "Status: {}".format(result.get("status")),
        "Automatic: {}".format(result.get("automatic_status")),
        "Review: {}".format(result.get("review_state")),
        "Approved: {}".format(result.get("approved_for_photometry")),
        "Stars used: {}/{}".format(
            result.get("star_count_used"), result.get("star_count_considered")
        ),
        "FWHM: {} pixel".format(
            "--" if result.get("fwhm_pixels") is None
            else "{:.3f}".format(result["fwhm_pixels"])
        ),
        "Ellipticity: {}".format(
            "--" if result.get("ellipticity") is None
            else "{:.3f}".format(result["ellipticity"])
        ),
        "Normalization: {}".format(
            "--" if result.get("normalization") is None
            else "{:.6f}".format(result["normalization"])
        ),
        "Median residual: {}".format(
            "--" if result.get("residual_median_fraction") is None
            else "{:.2%}".format(result["residual_median_fraction"])
        ),
        "Median correlation: {}".format(
            "--" if result.get("correlation_median") is None
            else "{:.4f}".format(result["correlation_median"])
        ),
        "Spatial support: {} ({} cells)".format(
            result.get("spatial_support"), result.get("spatial_cells_occupied")
        ),
        "",
        "Flags: {}".format(", ".join(result.get("quality_flags", [])) or "none"),
    ]
    summary.extend(result.get("reasons", []))
    if result.get("review_note"):
        summary.extend(["", "Review note: {}".format(result["review_note"])])
    summary_axis.text(
        0.02, 0.98, "\n".join(summary), va="top", ha="left",
        family="monospace", fontsize=10,
    )
    fig.suptitle(
        "PSF construction and review — {}".format(result.get("filename", "image")),
        fontsize=15,
    )
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.25)
    if show:
        plt.show()
    return fig


def plot_science_photometry_diagnostics(result, output_path=None, show=False):
    """Plot fixed-position target photometry and centroid diagnostics.

    The panels show the target context and configured apertures, the forced PSF
    model, residuals, masks, signed flux and S/N measurements, and the offset
    of the diagnostic free centroid from the frozen position.
    """

    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    diagnostics = result.get("target_diagnostics") or {}
    table = result.get("measurements")
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)

    context = diagnostics.get("context_data")
    context_mask = diagnostics.get("context_mask")
    _show_image(fig, axes[0, 0], context, "Target and apertures")
    if context is not None:
        origin = diagnostics.get("context_origin", (0, 0))
        center = (
            diagnostics.get("fixed_x", 0.0) - origin[0],
            diagnostics.get("fixed_y", 0.0) - origin[1],
        )
        circles = (
            ("small_radius_pixels", "tab:cyan", "small"),
            ("large_radius_pixels", "yellow", "reference"),
            ("sky_inner_radius_pixels", "tab:orange", "sky inner"),
            ("sky_outer_radius_pixels", "tab:red", "sky outer"),
        )
        for name, color, label in circles:
            radius = diagnostics.get(name)
            if radius is not None:
                axes[0, 0].add_patch(
                    Circle(center, radius, fill=False, color=color, linewidth=1.2, label=label)
                )
        axes[0, 0].legend(loc="upper right", fontsize=7)
        if context_mask is not None and np.any(context_mask):
            axes[0, 0].imshow(
                np.ma.masked_where(~np.asarray(context_mask, dtype=bool), context_mask),
                origin="lower", cmap="Reds", alpha=0.45, interpolation="nearest",
            )

    _show_image(
        fig, axes[0, 1], diagnostics.get("model"), "Forced PSF model", cmap="viridis"
    )
    _show_image(
        fig, axes[0, 2], diagnostics.get("residual"), "Data minus forced model",
        cmap="coolwarm", lower=2.0, upper=98.0,
    )
    _show_image(
        fig, axes[1, 0], context_mask, "Combined measurement mask",
        cmap="magma", lower=0.0, upper=100.0,
    )

    flux_axis = axes[1, 1]
    flux_axis.set_title("Signed target measurements")
    target_rows = None
    if table is not None and len(table):
        target_rows = table[np.asarray(table["source_type"], dtype=str) == "target"]
    if target_rows is not None and len(target_rows):
        methods = [str(value) for value in target_rows["method"]]
        flux = np.ma.asarray(target_rows["flux"], dtype=float).filled(np.nan)
        error = np.ma.asarray(target_rows["flux_uncertainty"], dtype=float).filled(np.nan)
        positions = np.arange(len(methods))
        flux_axis.errorbar(positions, flux, yerr=error, fmt="o", color="tab:blue", capsize=3)
        flux_axis.axhline(0.0, color="0.4", linewidth=1.0)
        flux_axis.set_xticks(positions)
        flux_axis.set_xticklabels(methods, rotation=20, ha="right")
        flux_axis.set_ylabel("flux [{}]".format(result.get("flux_unit", "image unit")))
        for position, row in zip(positions, target_rows):
            snr = row["snr"]
            label = "S/N --" if np.ma.is_masked(snr) else "S/N {:.1f}".format(float(snr))
            flux_axis.annotate(label, (position, float(row["flux"])), xytext=(0, 7),
                               textcoords="offset points", ha="center", fontsize=8)
    else:
        flux_axis.text(0.5, 0.5, "No target measurements", ha="center", va="center")
    flux_axis.grid(alpha=0.2)

    centroid_axis = axes[1, 2]
    centroid_axis.set_title("Fixed versus diagnostic centroid")
    free = diagnostics.get("free_centroid")
    centroid_axis.scatter([0.0], [0.0], marker="+", s=140, color="black", label="fixed")
    if free is not None:
        dx = free.get("offset_x_pixels", 0.0)
        dy = free.get("offset_y_pixels", 0.0)
        centroid_axis.arrow(
            0.0, 0.0, dx, dy, width=0.01, length_includes_head=True,
            color="tab:red", alpha=0.8,
        )
        centroid_axis.scatter([dx], [dy], color="tab:red", label="free diagnostic")
        summary = "Offset: {:.3f} pixel".format(free.get("offset_pixels", np.nan))
        if free.get("offset_arcsec") is not None:
            summary += "\n{:.3f} arcsec".format(free["offset_arcsec"])
        centroid_axis.text(0.03, 0.97, summary, transform=centroid_axis.transAxes,
                           va="top", family="monospace")
        span = max(1.0, abs(dx) * 1.5, abs(dy) * 1.5)
    else:
        centroid_axis.text(0.5, 0.5, "Free fit unavailable", ha="center", va="center")
        span = 1.0
    centroid_axis.set_xlim(-span, span)
    centroid_axis.set_ylim(-span, span)
    centroid_axis.set_aspect("equal")
    centroid_axis.set_xlabel("x offset [pixel]")
    centroid_axis.set_ylabel("y offset [pixel]")
    centroid_axis.axhline(0.0, color="0.8", linewidth=0.8)
    centroid_axis.axvline(0.0, color="0.8", linewidth=0.8)
    centroid_axis.legend(loc="lower right", fontsize=8)
    centroid_axis.grid(alpha=0.2)

    title = "Science-image forced photometry — {}".format(
        result.get("filename", result.get("image_id", "image"))
    )
    flags = result.get("target_flags", [])
    if flags:
        title += " — {}".format(", ".join(flags))
    fig.suptitle(title, fontsize=15)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.25)
    if show:
        plt.show()
    return fig


def plot_calibration_diagnostics(products, output_path=None, show=False):
    """Plot method zeropoints, calibration residual trends, and depth limits."""

    import matplotlib.pyplot as plt

    zeropoints = products.get("zeropoints")
    stars = products.get("calibration_stars")
    limits = products.get("limits")
    fig, axes = plt.subplots(3, 3, figsize=(17, 14), constrained_layout=True)

    zeropoint_axis = axes[0, 0]
    zeropoint_axis.set_title("Method-specific zeropoints")
    if zeropoints is not None and len(zeropoints):
        methods = list(dict.fromkeys(str(value) for value in zeropoints["method"]))
        colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(methods))))
        for color, method in zip(colors, methods):
            selected = zeropoints[np.asarray(zeropoints["method"], dtype=str) == method]
            values = np.ma.asarray(selected["zeropoint_mag"], dtype=float).filled(np.nan)
            errors = np.ma.asarray(
                selected["zeropoint_uncertainty_mag"], dtype=float
            ).filled(np.nan)
            zeropoint_axis.errorbar(
                np.arange(len(selected)), values, yerr=errors, fmt="o-",
                color=color, label=method, capsize=2,
            )
        zeropoint_axis.legend(fontsize=8)
    else:
        zeropoint_axis.text(0.5, 0.5, "No zeropoints", ha="center", va="center")
    zeropoint_axis.set_xlabel("image index")
    zeropoint_axis.set_ylabel("zeropoint [mag]")
    zeropoint_axis.grid(alpha=0.2)

    inlier = None
    if stars is not None and len(stars):
        inlier = stars[np.asarray(stars["inlier"], dtype=bool)]
    variables = (
        ("catalog_magnitude", "Catalog magnitude [mag]", axes[0, 1]),
        ("catalog_color", "Catalog color [mag]", axes[0, 2]),
        ("snr", "S/N", axes[1, 0]),
        ("x", "Detector x [pixel]", axes[1, 1]),
        ("y", "Detector y [pixel]", axes[1, 2]),
        ("airmass", "Airmass", axes[2, 0]),
    )
    for column, label, axis in variables:
        axis.set_title("Residual versus {}".format(label.split(" [")[0].lower()))
        if inlier is not None and len(inlier) and column in inlier.colnames:
            x = np.ma.asarray(inlier[column], dtype=float).filled(np.nan)
            residual = np.ma.asarray(
                inlier["calibrated_residual"], dtype=float
            ).filled(np.nan)
            methods = np.asarray(inlier["method"], dtype=str)
            for method in list(dict.fromkeys(methods)):
                selected = methods == method
                axis.scatter(x[selected], residual[selected], s=18, alpha=0.7, label=method)
            axis.axhline(0.0, color="0.4", linewidth=1.0)
        else:
            axis.text(0.5, 0.5, "Not available", ha="center", va="center")
        axis.set_xlabel(label)
        axis.set_ylabel("calibrated - catalog [mag]")
        axis.grid(alpha=0.2)
    if inlier is not None and len(inlier):
        axes[0, 1].legend(fontsize=7)

    limit_axis = axes[2, 1]
    limit_axis.set_title("Target limiting magnitudes")
    if limits is not None and len(limits):
        columns = [
            name for name in limits.colnames
            if name.endswith("_mag") and "limit_" in name
        ]
        labels = []
        values = []
        for row in limits:
            for name in columns:
                value = row[name]
                if np.ma.is_masked(value):
                    continue
                labels.append("{}\n{}".format(row["method"], name.replace("_mag", "")))
                values.append(float(value))
        if values:
            positions = np.arange(len(values))
            limit_axis.bar(positions, values, color="tab:purple", alpha=0.75)
            limit_axis.set_xticks(positions)
            limit_axis.set_xticklabels(labels, rotation=70, ha="right", fontsize=6)
        else:
            limit_axis.text(0.5, 0.5, "No calibrated limits", ha="center", va="center")
    else:
        limit_axis.text(0.5, 0.5, "No limits", ha="center", va="center")
    limit_axis.set_ylabel("limiting magnitude")
    limit_axis.invert_yaxis()
    limit_axis.grid(axis="y", alpha=0.2)

    summary_axis = axes[2, 2]
    summary_axis.set_axis_off()
    status_counts = {}
    if zeropoints is not None and len(zeropoints):
        for value in zeropoints["status"]:
            status_counts[str(value)] = status_counts.get(str(value), 0) + 1
    classification_counts = {}
    measurements = products.get("measurements")
    if measurements is not None and len(measurements):
        for value in measurements["classification"]:
            classification_counts[str(value)] = classification_counts.get(str(value), 0) + 1
    summary = [
        "Overall status: {}".format(products.get("status")),
        "Catalogs: {}".format(", ".join(products.get("catalogs_available", [])) or "none"),
        "Zeropoints: {}".format(status_counts or "none"),
        "Classifications: {}".format(classification_counts or "none"),
        "Unstable stars: {}".format(len(products.get("unstable_stars", []))),
        "Empty-aperture rows: {}".format(0 if limits is None else len(limits)),
    ]
    if products.get("unstable_stars"):
        summary.extend(["", "Rejected unstable stars:"])
        summary.extend("  {}".format(value) for value in products["unstable_stars"])
    summary_axis.text(
        0.02, 0.98, "\n".join(summary), va="top", ha="left",
        family="monospace", fontsize=10,
    )
    fig.suptitle("Photometric calibration and limiting-depth diagnostics", fontsize=15)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.25)
    if show:
        plt.show()
    return fig


def plot_subtraction_diagnostics(result, science_record=None, output_path=None,
                                 show=False):
    """Plot template preparation, difference quality, and backend diagnostics.

    Parameters
    ----------
    result : mapping
        Result returned by
        :func:`redphot.subtraction.perform_image_subtraction`.
    science_record : mapping, optional
        Input image record.  Supplying it adds the original science panel.
    output_path : str or pathlib.Path, optional
        PNG or PDF destination.
    show : bool, optional
        Display the completed figure interactively.
    """

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(16, 13), constrained_layout=True)
    science = None
    if science_record is not None:
        for name in ("prepared_ccd", "working_ccd", "ccd"):
            value = science_record.get(name)
            if value is not None:
                science = np.asarray(getattr(value, "data", value), dtype=float)
                break
        if science is None and science_record.get("data") is not None:
            science = np.asarray(science_record["data"], dtype=float)
    template = result.get("template") or {}
    aligned = result.get("aligned_template") or {}
    difference = result.get("difference")
    _show_image(fig, axes[0, 0], science, "Science (native grid)")
    _show_image(fig, axes[0, 1], template.get("data"), "Template mosaic")
    _show_image(fig, axes[0, 2], aligned.get("data"), "Aligned template")
    _show_image(fig, axes[1, 0], difference, "Difference", cmap="coolwarm")

    kernel_axis = axes[1, 1]
    kernel_axis.set_title("Kernel choice")
    parameters = result.get("parameters") or {}
    components = parameters.get("gaussian_components", [])
    if components:
        x = np.linspace(-8, 8, 400)
        for degree, sigma in components:
            profile = np.exp(-0.5 * (x / float(sigma)) ** 2)
            kernel_axis.plot(x, profile, label="degree {}, sigma {}".format(degree, sigma))
        kernel_axis.legend(fontsize=8)
        kernel_axis.set_xlabel("kernel coordinate [pixel]")
        kernel_axis.set_ylabel("relative amplitude")
    else:
        kernel_axis.text(0.5, 0.5, "Kernel not available", ha="center", va="center")

    histogram_axis = axes[1, 2]
    histogram_axis.set_title("Difference-pixel distribution")
    if difference is not None:
        values = np.asarray(difference, dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            low, high = np.nanpercentile(values, [0.5, 99.5])
            histogram_axis.hist(values[(values >= low) & (values <= high)], bins=80,
                                histtype="step", color="black")
    histogram_axis.set_xlabel("difference value")
    histogram_axis.set_ylabel("pixels")

    residual_axis = axes[2, 0]
    residual_axis.set_title("Stellar residuals")
    quality = result.get("quality") or {}
    residuals = quality.get("star_residuals")
    if residuals is not None and len(residuals):
        residual_axis.scatter(
            residuals["science_flux"], residuals["residual_fraction"],
            c=residuals["dipole_fraction"], cmap="viridis", edgecolor="none",
        )
        residual_axis.set_xscale("symlog")
        residual_axis.axhline(0.10, color="tab:red", linestyle="--", alpha=0.6)
    else:
        residual_axis.text(0.5, 0.5, "No quality-star measurements",
                           ha="center", va="center")
    residual_axis.set_xlabel("science flux")
    residual_axis.set_ylabel("absolute residual fraction")

    blank_axis = axes[2, 1]
    blank_axis.set_title("Blank-aperture noise")
    blank = np.asarray(quality.get("blank_aperture_fluxes", []), dtype=float)
    blank = blank[np.isfinite(blank)]
    if blank.size:
        blank_axis.hist(blank, bins=min(30, max(5, blank.size // 2)),
                        histtype="stepfilled", alpha=0.5, color="tab:blue")
        blank_axis.axvline(0, color="black", linewidth=1)
    else:
        blank_axis.text(0.5, 0.5, "No blank apertures", ha="center", va="center")
    blank_axis.set_xlabel("aperture flux")

    summary_axis = axes[2, 2]
    summary_axis.set_axis_off()
    summary = [
        "Image: {}".format(result.get("image_id")),
        "Status: {}".format(result.get("status")),
        "Backend: {}".format(result.get("method")),
        "Convolved: {}".format(parameters.get("convolve", "unknown")),
        "Coverage: {}".format(
            "n/a" if aligned.get("coverage_fraction") is None
            else "{:.3f}".format(aligned["coverage_fraction"])
        ),
        "Residual fraction: {}".format(
            "n/a" if quality.get("median_residual_fraction") is None
            else "{:.4f}".format(quality["median_residual_fraction"])
        ),
        "Dipole fraction: {}".format(
            "n/a" if quality.get("median_dipole_fraction") is None
            else "{:.4f}".format(quality["median_dipole_fraction"])
        ),
        "Noise ratio: {}".format(
            "n/a" if quality.get("noise_ratio") is None
            else "{:.3f}".format(quality["noise_ratio"])
        ),
        "Flags: {}".format(", ".join(result.get("flags", [])) or "none"),
    ]
    if result.get("error"):
        summary.extend(["", "Error:", str(result["error"])])
    summary_axis.text(0.02, 0.98, "\n".join(summary), va="top", ha="left",
                      family="monospace", fontsize=10, wrap=True)
    fig.suptitle("Image-subtraction diagnostics", fontsize=15)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.25)
    if show:
        plt.show()
    return fig


def plot_difference_photometry_diagnostics(result, output_path=None, show=False):
    """Plot forced difference photometry, limits, and science comparison."""

    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, axes = plt.subplots(2, 4, figsize=(20, 10), constrained_layout=True)
    subtraction = result.get("subtraction_result") or {}
    difference_record = result.get("difference_record") or {}
    science_record = difference_record.get("science_record") or {}
    science = None
    for name in ("prepared_ccd", "working_ccd", "ccd"):
        value = science_record.get(name)
        if value is not None:
            science = np.asarray(getattr(value, "data", value), dtype=float)
            break
    template = (subtraction.get("template") or {}).get("data")
    aligned_template = (subtraction.get("aligned_template") or {}).get("data")
    _show_image(fig, axes[0, 0], science, "Science")
    _show_image(
        fig, axes[0, 1], template if template is not None else aligned_template,
        "Template",
    )
    diagnostics = result.get("target_diagnostics") or {}
    context = diagnostics.get("context_data")
    _show_image(fig, axes[0, 2], context, "Difference target", cmap="coolwarm")
    if context is not None:
        origin = diagnostics.get("context_origin", (0, 0))
        center = (
            diagnostics.get("fixed_x", 0.0) - origin[0],
            diagnostics.get("fixed_y", 0.0) - origin[1],
        )
        for name, color in (
            ("small_radius_pixels", "cyan"),
            ("large_radius_pixels", "yellow"),
        ):
            radius = diagnostics.get(name)
            if radius is not None:
                axes[0, 2].add_patch(
                    Circle(center, radius, fill=False, color=color, linewidth=1.2)
                )
    _show_image(fig, axes[0, 3], diagnostics.get("model"),
                "Forced difference PSF model", cmap="viridis")
    _show_image(fig, axes[1, 0], diagnostics.get("residual"),
                "Difference minus model", cmap="coolwarm", lower=2, upper=98)

    comparison_axis = axes[1, 1]
    comparison_axis.set_title("Science and difference flux")
    comparison = result.get("comparison")
    if comparison is not None and len(comparison):
        methods = [str(value) for value in comparison["method"]]
        positions = np.arange(len(methods))
        science_flux = np.ma.asarray(comparison["science_flux"], dtype=float).filled(np.nan)
        science_error = np.ma.asarray(
            comparison["science_uncertainty"], dtype=float
        ).filled(np.nan)
        difference_flux = np.ma.asarray(
            comparison["difference_flux"], dtype=float
        ).filled(np.nan)
        difference_error = np.ma.asarray(
            comparison["difference_uncertainty"], dtype=float
        ).filled(np.nan)
        comparison_axis.errorbar(
            positions - 0.08, science_flux, yerr=science_error,
            fmt="o", capsize=3, label="science (host included)",
        )
        comparison_axis.errorbar(
            positions + 0.08, difference_flux, yerr=difference_error,
            fmt="o", capsize=3, label="difference (host removed)",
        )
        comparison_axis.set_xticks(positions)
        comparison_axis.set_xticklabels(methods, rotation=20, ha="right")
        comparison_axis.legend(fontsize=8)
    else:
        comparison_axis.text(0.5, 0.5, "No paired measurements",
                             ha="center", va="center")
    comparison_axis.axhline(0, color="0.5", linewidth=1)
    comparison_axis.set_ylabel("signed flux")
    comparison_axis.grid(alpha=0.2)

    limit_axis = axes[1, 2]
    limit_axis.set_title("Difference-image limits")
    limits = result.get("limits")
    plotted = False
    if limits is not None and len(limits):
        columns = [name for name in limits.colnames if name.startswith("limit_") and name.endswith("_mag")]
        labels, values = [], []
        for row in limits:
            for name in columns:
                value = row[name]
                if not np.ma.is_masked(value) and np.isfinite(float(value)):
                    labels.append("{}\n{}".format(row["method"], name[6:-4]))
                    values.append(float(value))
        if values:
            positions = np.arange(len(values))
            limit_axis.bar(positions, values, color="tab:purple", alpha=0.75)
            limit_axis.set_xticks(positions)
            limit_axis.set_xticklabels(labels, rotation=65, ha="right", fontsize=7)
            limit_axis.invert_yaxis()
            limit_axis.set_ylabel("limiting magnitude")
            plotted = True
    if not plotted:
        limit_axis.text(0.5, 0.5, "No calibrated magnitude limits",
                        ha="center", va="center")

    summary_axis = axes[1, 3]
    summary_axis.set_axis_off()
    preferred = result.get("preferred_result") or {}
    dipole = result.get("dipole") or {}
    summary = [
        "Status: {}".format(result.get("status")),
        "Preferred: {} / {}".format(
            preferred.get("image_kind", "none"), preferred.get("method", "none")
        ),
        "Host light included: {}".format(
            result.get("preferred_host_light_included")
        ),
        "Classification: {}".format(preferred.get("classification", "none")),
        "S/N: {}".format(
            "n/a" if preferred.get("snr") is None else "{:.2f}".format(preferred["snr"])
        ),
        "Difference PSF: {}".format(result.get("difference_psf_source")),
        "Dipole: {}".format(dipole.get("detected", False)),
        "Dipole ratio: {}".format(
            "n/a" if dipole.get("absolute_lobe_ratio") is None
            else "{:.3f}".format(dipole["absolute_lobe_ratio"])
        ),
        "Inverted residual: {}".format(result.get("inverted_residual")),
        "Flags: {}".format(", ".join(result.get("flags", [])) or "none"),
        "",
        "Rule:",
        str(result.get("selection_rule") or "none"),
    ]
    summary_axis.text(0.02, 0.98, "\n".join(summary), va="top", ha="left",
                      family="monospace", fontsize=10, wrap=True)
    fig.suptitle("Difference-image forced photometry", fontsize=15)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.25)
    if show:
        plt.show()
    return fig


def plot_batch_consistency_diagnostics(products, output_path=None, show=False):
    """Plot batch trends, rejected epochs, star stability, and light curves."""

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(17, 14), constrained_layout=True)
    epoch = products.get("epoch_metrics")
    trends_axis = axes[0, 0]
    trends_axis.set_title("Batch metrics over time")
    if epoch is not None and len(epoch):
        mjd = np.ma.asarray(epoch["mjd"], dtype=float).filled(np.nan)
        metric_styles = (
            ("zeropoint_mag", "zeropoint", "o"),
            ("depth_5sigma_mag", "5-sigma depth", "s"),
            ("seeing_fwhm_arcsec", "seeing", "^"),
            ("background_rms", "background RMS", "v"),
            ("wcs_rms_arcsec", "WCS RMS", "D"),
        )
        for name, label, marker in metric_styles:
            if name not in epoch.colnames:
                continue
            values = np.ma.asarray(epoch[name], dtype=float).filled(np.nan)
            valid = np.isfinite(mjd) & np.isfinite(values)
            if np.any(valid):
                normalized = values[valid] - np.nanmedian(values[valid])
                scatter = 1.4826 * np.nanmedian(np.abs(normalized))
                if np.isfinite(scatter) and scatter > 0:
                    normalized /= scatter
                trends_axis.plot(mjd[valid], normalized, marker=marker,
                                 linestyle="-", alpha=0.7, label=label)
        trends_axis.axhline(0, color="0.5", linewidth=1)
        trends_axis.set_xlabel("MJD")
        trends_axis.set_ylabel("median-centered robust units")
        trends_axis.legend(fontsize=8, ncol=2)
    else:
        trends_axis.text(0.5, 0.5, "No epoch metrics", ha="center", va="center")
    trends_axis.grid(alpha=0.2)

    rejected_axis = axes[0, 1]
    rejected_axis.set_title("Accepted and rejected epochs")
    if epoch is not None and len(epoch):
        colors = {"PASS": "tab:green", "WARN": "tab:orange", "FAIL": "tab:red"}
        for status in ("PASS", "WARN", "FAIL"):
            selection = np.asarray(epoch["status"], dtype=str) == status
            if np.any(selection):
                rejected_axis.scatter(
                    np.ma.asarray(epoch["mjd"], dtype=float).filled(np.nan)[selection],
                    np.arange(len(epoch))[selection], color=colors[status],
                    label=status, s=45,
                )
        rejected_axis.set_xlabel("MJD")
        rejected_axis.set_ylabel("image index")
        rejected_axis.legend(fontsize=8)
    else:
        rejected_axis.text(0.5, 0.5, "No image decisions", ha="center", va="center")
    rejected_axis.grid(alpha=0.2)

    stability_axis = axes[1, 0]
    stability_axis.set_title("Comparison-star stability")
    stability = products.get("comparison_stability")
    if stability is not None and len(stability):
        order = np.argsort(np.ma.asarray(stability["rms_mag"], dtype=float).filled(np.inf))
        values = np.ma.asarray(stability["rms_mag"], dtype=float).filled(np.nan)[order]
        labels = [
            "{} {} {}".format(stability[index]["source_id"], stability[index]["filter"],
                              stability[index]["method"])
            for index in order
        ]
        colors = [
            {"PASS": "tab:green", "WARN": "tab:orange", "FAIL": "tab:red"}.get(
                str(stability[index]["status"]), "0.5"
            ) for index in order
        ]
        positions = np.arange(len(values))
        stability_axis.bar(positions, values, color=colors, alpha=0.8)
        stability_axis.set_xticks(positions)
        stability_axis.set_xticklabels(labels, rotation=70, ha="right", fontsize=6)
        stability_axis.set_ylabel("RMS [mag]")
    else:
        stability_axis.text(0.5, 0.5, "No comparison-star light curves",
                            ha="center", va="center")
    stability_axis.grid(axis="y", alpha=0.2)

    group_axis = axes[1, 1]
    group_axis.set_title("Problem fraction by telescope, site, and filter")
    groups = products.get("group_summary")
    if groups is not None and len(groups):
        labels = ["{}:{}".format(row["group_type"], row["group_value"]) for row in groups]
        values = np.ma.asarray(groups["problem_fraction"], dtype=float).filled(np.nan)
        colors = [
            {"PASS": "tab:green", "WARN": "tab:orange", "FAIL": "tab:red"}.get(
                str(row["status"]), "0.5"
            ) for row in groups
        ]
        positions = np.arange(len(values))
        group_axis.bar(positions, values, color=colors, alpha=0.8)
        group_axis.set_xticks(positions)
        group_axis.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
        group_axis.set_ylim(0, 1)
        group_axis.set_ylabel("WARN + FAIL fraction")
    else:
        group_axis.text(0.5, 0.5, "No group summary", ha="center", va="center")
    group_axis.grid(axis="y", alpha=0.2)

    methods_axis = axes[2, 0]
    methods_axis.set_title("Multi-method target light curve")
    measurements = products.get("measurements")
    if measurements is not None and len(measurements):
        target = measurements[np.asarray(measurements["source_type"], dtype=str) == "target"]
        styles = {
            ("science", "small_aperture"): ("o", "tab:cyan"),
            ("science", "large_aperture"): ("s", "tab:blue"),
            ("science", "psf"): ("^", "navy"),
            ("difference", "small_aperture"): ("o", "tab:pink"),
            ("difference", "large_aperture"): ("s", "tab:red"),
            ("difference", "psf"): ("^", "darkred"),
        }
        for (kind, method), (marker, color) in styles.items():
            selection = (
                (np.asarray(target["image_kind"], dtype=str) == kind)
                & (np.asarray(target["method"], dtype=str) == method)
            )
            if not np.any(selection):
                continue
            rows = target[selection]
            magnitude_name = (
                "ensemble_corrected_magnitude"
                if "ensemble_corrected_magnitude" in rows.colnames
                else "calibrated_magnitude"
            )
            if magnitude_name not in rows.colnames:
                continue
            methods_axis.scatter(
                np.ma.asarray(rows["mjd_mid"], dtype=float).filled(np.nan),
                np.ma.asarray(rows[magnitude_name], dtype=float).filled(np.nan),
                marker=marker, color=color, label="{} {}".format(kind, method), s=30,
            )
        methods_axis.invert_yaxis()
        methods_axis.set_xlabel("MJD")
        methods_axis.set_ylabel("magnitude")
        handles, labels = methods_axis.get_legend_handles_labels()
        if handles:
            methods_axis.legend(handles, labels, fontsize=7, ncol=2)
    else:
        methods_axis.text(0.5, 0.5, "No target measurements", ha="center", va="center")
    methods_axis.grid(alpha=0.2)

    preferred_axis = axes[2, 1]
    preferred_axis.set_title("Final preferred light curve")
    preferred = products.get("preferred_light_curve")
    if preferred is not None and len(preferred):
        for included, marker, color, label in (
            (True, "o", "tab:blue", "included"),
            (False, "x", "tab:red", "retained but rejected"),
        ):
            selection = np.asarray(preferred["included_in_final"], dtype=bool) == included
            if not np.any(selection):
                continue
            rows = preferred[selection]
            preferred_axis.errorbar(
                np.ma.asarray(rows["mjd"], dtype=float).filled(np.nan),
                np.ma.asarray(rows["magnitude"], dtype=float).filled(np.nan),
                yerr=np.ma.asarray(rows["magnitude_uncertainty"], dtype=float).filled(np.nan),
                fmt=marker, color=color, label=label, capsize=2,
            )
        preferred_axis.invert_yaxis()
        preferred_axis.set_xlabel("MJD")
        preferred_axis.set_ylabel("preferred magnitude")
        preferred_axis.legend(fontsize=8)
    else:
        preferred_axis.text(0.5, 0.5, "No preferred light curve",
                            ha="center", va="center")
    preferred_axis.grid(alpha=0.2)
    fig.suptitle(
        "Batch consistency — status {} — unstable stars {} — failed epochs {} — outliers {}".format(
            products.get("status"), products.get("unstable_comparison_count", 0),
            products.get("failed_epoch_count", 0), products.get("measurement_outlier_count", 0),
        ),
        fontsize=15,
    )
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.25)
    if show:
        plt.show()
    return fig


__all__ = [
    "plot_background_diagnostics",
    "plot_batch_consistency_diagnostics",
    "plot_calibration_diagnostics",
    "plot_difference_photometry_diagnostics",
    "plot_astrometry_diagnostics",
    "plot_alignment_target_diagnostics",
    "plot_image_quality_diagnostics",
    "plot_image_usability_diagnostics",
    "plot_psf_diagnostics",
    "plot_science_photometry_diagnostics",
    "plot_subtraction_diagnostics",
    "plot_star_selection_diagnostics",
]
