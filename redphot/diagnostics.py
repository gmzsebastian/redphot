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
    """Plot source detections and Step 9 image-quality measurements.

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


__all__ = [
    "plot_background_diagnostics",
    "plot_image_quality_diagnostics",
]
