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


__all__ = ["plot_background_diagnostics"]
