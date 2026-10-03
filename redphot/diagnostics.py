"""Diagnostic figures for every redphot processing stage.

All figures share one visual language so a run can be judged at a glance:

* a header naming the stage and the image, its key metadata, a one-line
  "Check" telling you what a good result looks like, and a status badge;
* sky images shown with a robust z-scale stretch on a dark grayscale, with
  overlays in a small fixed palette;
* residual and difference images on a diverging map centered on zero, with
  symmetric limits, so over- and under-subtraction are equally visible;
* model maps on a perceptually uniform sequential map;
* a measurements panel listing each quantity next to the limit that decides
  PASS/WARN/FAIL, with a colored dot when a limit applies.

Every public ``plot_*`` function returns a Matplotlib figure and optionally
saves it (``output_path``) or shows it (``show``).
"""

from contextlib import contextmanager
from functools import wraps
import logging
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Visual language
# ---------------------------------------------------------------------------

INK = "#1f2328"
MUTED = "#57606a"
FAINT = "#8c959f"
RULE = "#d0d7de"
PANEL = "#f6f8fa"

BLUE = "#2f81f7"
TEAL = "#14b8a6"
AMBER = "#f59e0b"
RED = "#ef4444"
VIOLET = "#a78bfa"
PINK = "#ec4899"
GREEN = "#22c55e"
SLATE = "#94a3b8"

STATUS_COLORS = {
    "PASS": "#1a7f37",
    "APPROVED": "#1a7f37",
    "WARN": "#bf8700",
    "FAIL": "#cf222e",
    "REJECTED": "#cf222e",
    "SKIPPED": "#8c959f",
    "STALE": "#8c959f",
    "PENDING": "#0969da",
}

FILTER_COLORS = {
    "u": "#7c3aed", "g": "#16a34a", "r": "#dc2626", "i": "#9a3412",
    "z": "#475569", "y": "#1e293b", "B": "#2563eb", "V": "#65a30d",
    "R": "#b91c1c", "I": "#78350f", "G": "#0f766e",
}

METHOD_MARKERS = {
    "psf": "o", "small_aperture": "s", "large_aperture": "D",
}

MASK_COLORS = {
    "nonfinite": SLATE,
    "input": "#64748b",
    "saturation": RED,
    "bad_lines": AMBER,
    "amplifier": VIOLET,
    "trails": PINK,
    "manual": BLUE,
}

SKY_CMAP = "gray"
MAP_CMAP = "magma"
DIVERGING_CMAP = "RdBu_r"

_RC = {
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Helvetica Neue", "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans",
    ],
    "font.size": 9,
    "text.color": INK,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlepad": 6,
    "axes.titlecolor": INK,
    "axes.labelsize": 8.5,
    "axes.labelcolor": MUTED,
    "axes.edgecolor": RULE,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "grid.color": "#eaeef2",
    "grid.linewidth": 0.7,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "legend.frameon": False,
    "legend.fontsize": 7.5,
    "legend.handletextpad": 0.4,
    "image.interpolation": "nearest",
    "image.origin": "lower",
    "lines.linewidth": 1.4,
    "lines.markersize": 4,
    "scatter.edgecolors": "none",
}


@contextmanager
def diagnostic_style():
    """Apply the redphot figure style (fonts, colors, spines) temporarily."""

    import matplotlib.pyplot as plt
    from cycler import cycler

    font_logger = logging.getLogger("matplotlib.font_manager")
    level = font_logger.level
    font_logger.setLevel(logging.ERROR)
    rc = dict(_RC)
    rc["axes.prop_cycle"] = cycler(
        color=[BLUE, AMBER, TEAL, VIOLET, RED, PINK, GREEN, SLATE]
    )
    try:
        with plt.rc_context(rc):
            yield
    finally:
        font_logger.setLevel(level)


def _styled(function):
    """Run a plotting function inside :func:`diagnostic_style`."""

    @wraps(function)
    def wrapper(*args, **kwargs):
        with diagnostic_style():
            return function(*args, **kwargs)

    return wrapper


# ---------------------------------------------------------------------------
# Small numeric and formatting helpers
# ---------------------------------------------------------------------------

def _finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _fmt(value, spec="{:.3g}", unit="", missing="—"):
    """Format a number, returning an em dash for missing or non-finite values."""

    number = _finite(value)
    if number is None:
        if isinstance(value, str) and value:
            return value
        return missing
    text = spec.format(number)
    return "{} {}".format(text, unit).strip() if unit else text


def _column(table, name, default=np.nan):
    """Return a float array for a (possibly masked or missing) table column."""

    if table is None or not len(table) or name not in getattr(table, "colnames", []):
        return np.full(0 if table is None else len(table), default, dtype=float)
    values = np.ma.asarray(table[name])
    try:
        return np.asarray(np.ma.filled(values.astype(float), default), dtype=float)
    except (TypeError, ValueError):
        return np.full(len(table), default, dtype=float)


def _bool_column(table, name, default=False):
    if table is None or not len(table) or name not in getattr(table, "colnames", []):
        return np.full(0 if table is None else len(table), default, dtype=bool)
    return np.asarray(np.ma.filled(np.ma.asarray(table[name]), default), dtype=bool)


def _str_column(table, name):
    if table is None or not len(table) or name not in getattr(table, "colnames", []):
        return np.full(0 if table is None else len(table), "", dtype=object)
    return np.asarray([str(value) for value in table[name]], dtype=object)


def _robust(values):
    """Median and Gaussian-scaled MAD of the finite values."""

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None, None
    median = float(np.median(values))
    return median, float(1.4826 * np.median(np.abs(values - median)))


def _data(image):
    if image is None:
        return None
    return np.asarray(getattr(image, "data", image), dtype=float)


def _sample(values, limit=200000):
    values = np.asarray(values, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if values.size > limit:
        step = int(np.ceil(values.size / limit))
        values = values[::step]
    return values


def zscale_limits(data, contrast=0.25):
    """Robust display limits (IRAF z-scale) for a sky image."""

    values = _sample(data, 100000)
    if values.size < 10:
        return 0.0, 1.0
    try:
        from astropy.visualization import ZScaleInterval

        low, high = ZScaleInterval(contrast=contrast).get_limits(values)
    except Exception:
        low, high = np.percentile(values, [1.0, 99.5])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = np.percentile(values, [1.0, 99.5])
        if high <= low:
            high = low + 1.0
    return float(low), float(high)


def _symmetric_limit(values, percentile=99.0, floor=None):
    values = _sample(values)
    if values.size == 0:
        return 1.0
    limit = float(np.percentile(np.abs(values), percentile))
    if floor is not None:
        limit = max(limit, floor)
    return limit if limit > 0 else 1.0


def _status_color(status):
    return STATUS_COLORS.get(str(status).upper(), FAINT)


def _filter_color(name):
    return FILTER_COLORS.get(str(name), FILTER_COLORS.get(str(name).lower(), BLUE))


def _image_label(metadata, fallback=None):
    metadata = metadata or {}
    return str(metadata.get("filename") or fallback or "image")


def _image_subtitle(metadata):
    """One line of key metadata: filter, time, exposure, airmass, instrument."""

    metadata = metadata or {}
    parts = []
    if metadata.get("filter"):
        parts.append("{} band".format(metadata["filter"]))
    date = metadata.get("date_mid_utc") or metadata.get("date_obs")
    if date:
        parts.append(str(date).replace("T", " ")[:16] + " UT")
    if metadata.get("mjd_mid") is not None:
        parts.append("MJD {}".format(_fmt(metadata["mjd_mid"], "{:.3f}")))
    if metadata.get("exposure_time") is not None:
        parts.append("{} s".format(_fmt(metadata["exposure_time"], "{:g}")))
    if metadata.get("airmass") is not None:
        parts.append("airmass {}".format(_fmt(metadata["airmass"], "{:.2f}")))
    instrument = metadata.get("instrument") or metadata.get("telescope")
    if instrument:
        parts.append(str(instrument))
    return "  ·  ".join(parts)


# ---------------------------------------------------------------------------
# Figure scaffolding
# ---------------------------------------------------------------------------

def _new_figure(title, subtitle=None, status=None, check=None, size=(16, 9.4),
                rows=2, columns=3, height_ratios=None, width_ratios=None,
                status_note=None):
    """Create a figure with a header band and a body grid.

    Returns ``(figure, grid)`` where ``grid`` is a GridSpec for the body.
    """

    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=size, layout="constrained")
    figure.get_layout_engine().set(w_pad=0.06, h_pad=0.06, wspace=0.04, hspace=0.06)
    header_height = 0.105 * size[1] if check else 0.085 * size[1]
    outer = figure.add_gridspec(
        2, 1, height_ratios=[header_height, size[1] - header_height]
    )
    header = figure.add_subplot(outer[0])
    header.set_axis_off()
    header.text(0.0, 0.98, title, transform=header.transAxes, ha="left", va="top",
                fontsize=15, fontweight="bold", color=INK)
    positions = [0.50, 0.20] if check else [0.42]
    entries = []
    if subtitle:
        entries.append((subtitle, MUTED, 9.5))
    if check:
        entries.append(("Check:  " + check, INK, 9))
    for (text, color, points), y in zip(entries, positions):
        header.text(0.0, y, text, transform=header.transAxes, ha="left", va="top",
                    fontsize=points, color=color)
    if status:
        header.text(
            1.0, 0.92, " {} ".format(str(status).upper()), transform=header.transAxes,
            ha="right", va="top", fontsize=11, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.45,rounding_size=0.7",
                      facecolor=_status_color(status), edgecolor="none"),
        )
        if status_note:
            header.text(1.0, 0.36, status_note, transform=header.transAxes,
                        ha="right", va="top", fontsize=8, color=MUTED)
    header.plot([0, 1], [0.0, 0.0], transform=header.transAxes, color=RULE,
                linewidth=1.0, clip_on=False)
    body = outer[1].subgridspec(
        rows, columns, height_ratios=height_ratios, width_ratios=width_ratios
    )
    return figure, body


def _finish(figure, output_path=None, show=False, dpi=110):
    import matplotlib.pyplot as plt

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=dpi)
    if show:
        plt.show()
    return figure


def _empty(axis, message, title=None):
    """Replace a panel by a quiet 'not available' message."""

    axis.set_axis_off()
    if title:
        axis.set_title(title)
    axis.text(0.5, 0.5, message, transform=axis.transAxes, ha="center",
              va="center", fontsize=9, color=FAINT, wrap=True,
              bbox=dict(boxstyle="round,pad=0.8", facecolor=PANEL, edgecolor="none"))


def _colorbar(figure, image, axis, label=None):
    bar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.015, shrink=0.92)
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=7, length=2, color=FAINT)
    if label:
        bar.set_label(label, fontsize=7.5, color=MUTED)
    return bar


def _image_axes(axis, shape=None, labels=True):
    axis.tick_params(length=2, labelsize=7, color=FAINT)
    for spine in axis.spines.values():
        spine.set_visible(False)
    if labels:
        axis.set_xlabel("x [pixel]", fontsize=7.5)
        axis.set_ylabel("y [pixel]", fontsize=7.5)


def show_sky(axis, image, title=None, limits=None, cmap=SKY_CMAP, labels=True,
             extent=None):
    """Draw a sky image with a z-scale stretch. Returns the image artist."""

    data = _data(image)
    if data is None or data.ndim != 2 or not np.any(np.isfinite(data)):
        _empty(axis, "Image not available", title)
        return None
    low, high = limits if limits is not None else zscale_limits(data)
    artist = axis.imshow(data, cmap=cmap, vmin=low, vmax=high, extent=extent,
                         interpolation="nearest", origin="lower")
    if title:
        axis.set_title(title)
    _image_axes(axis, data.shape, labels)
    return artist


def show_map(figure, axis, image, title=None, cmap=MAP_CMAP, label=None,
             symmetric=False, limit=None, limits=None, labels=True, colorbar=True,
             log=False):
    """Draw a model, RMS or residual map with a slim colorbar."""

    data = _data(image)
    if data is None or data.ndim != 2 or not np.any(np.isfinite(data)):
        _empty(axis, "Not available", title)
        return None
    from matplotlib.colors import LogNorm

    norm = None
    if symmetric:
        bound = limit if limit is not None else _symmetric_limit(data)
        low, high = -bound, bound
        cmap = DIVERGING_CMAP if cmap == MAP_CMAP else cmap
    elif limits is not None:
        low, high = limits
    else:
        values = _sample(data)
        low, high = np.percentile(values, [0.5, 99.5]) if values.size else (0, 1)
        if high <= low:
            high = low + 1.0
    if log:
        positive = data[np.isfinite(data) & (data > 0)]
        if positive.size:
            norm = LogNorm(vmin=max(float(np.percentile(positive, 1)), 1e-12),
                           vmax=float(np.max(positive)))
    artist = axis.imshow(
        data, cmap=cmap, origin="lower", interpolation="nearest",
        **({"norm": norm} if norm is not None else {"vmin": low, "vmax": high})
    )
    if title:
        axis.set_title(title)
    _image_axes(axis, data.shape, labels)
    if colorbar:
        _colorbar(figure, artist, axis, label)
    return artist


def overlay_mask(axis, mask, color, alpha=0.55):
    """Tint the pixels where ``mask`` is True with one solid color."""

    from matplotlib.colors import to_rgba

    if mask is None:
        return
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return
    rgba = np.zeros(mask.shape + (4,), dtype=float)
    rgba[mask] = to_rgba(color, alpha)
    axis.imshow(rgba, origin="lower", interpolation="nearest")


def mark_target(axis, x, y, radius=14, color=TEAL, label="target"):
    """Mark a position with an open ring and four ticks (does not hide it)."""

    from matplotlib.patches import Circle

    if _finite(x) is None or _finite(y) is None:
        return
    axis.add_patch(Circle((x, y), radius, fill=False, edgecolor=color,
                          linewidth=1.3, label=label))
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        axis.plot([x + dx * radius * 1.25, x + dx * radius * 1.9],
                  [y + dy * radius * 1.25, y + dy * radius * 1.9],
                  color=color, linewidth=1.3)


def _grid(axis, axis_name="both"):
    axis.grid(True, axis=axis_name)


def _legend(axis, **kwargs):
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        options = {"loc": "best", "markerscale": 1.2}
        options.update(kwargs)
        axis.legend(**options)


def _image_legend(axis, loc="upper right", **kwargs):
    """Legend for panels that show an image: opaque white card, dark text.

    White text straight on a sky image disappears on bright stars and
    saturated regions, so image legends sit on their own card instead.
    """

    handles, labels = axis.get_legend_handles_labels()
    if not handles:
        return
    options = {
        "loc": loc, "frameon": True, "facecolor": "white", "framealpha": 0.92,
        "edgecolor": RULE, "fontsize": 8, "markerscale": 1.6, "borderpad": 0.55,
        "labelcolor": INK, "handlelength": 1.3, "handletextpad": 0.5,
    }
    options.update(kwargs)
    legend = axis.legend(**options)
    legend.get_frame().set_linewidth(0.6)


def _overlay_cut(axis, mask, color, alpha, extent):
    """Tint a cutout mask (already sliced) placed at ``extent``."""

    from matplotlib.colors import to_rgba

    if mask is None or not np.any(mask):
        return
    rgba = np.zeros(np.shape(mask) + (4,))
    rgba[np.asarray(mask, dtype=bool)] = to_rgba(color, alpha)
    axis.imshow(rgba, extent=extent, origin="lower", interpolation="nearest")


def _zoom_panel(axis, data, center_x, center_y, half, title, layers=(), target=None):
    """Sky cutout around a position with mask layers ``(mask, color, alpha)``."""

    if data is None or _finite(center_x) is None or _finite(center_y) is None:
        _empty(axis, "Position not available", title)
        return None
    ny, nx = data.shape
    x0, x1 = int(max(0, center_x - half)), int(min(nx, center_x + half))
    y0, y1 = int(max(0, center_y - half)), int(min(ny, center_y + half))
    if x1 <= x0 or y1 <= y0:
        _empty(axis, "Position outside the image", title)
        return None
    extent = (x0 - 0.5, x1 - 0.5, y0 - 0.5, y1 - 0.5)
    show_sky(axis, data[y0:y1, x0:x1], title, extent=extent, labels=False)
    for mask, color, alpha in layers:
        if mask is not None and np.shape(mask) == data.shape:
            _overlay_cut(axis, np.asarray(mask, dtype=bool)[y0:y1, x0:x1], color, alpha, extent)
    if target is not None:
        mark_target(axis, target[0], target[1], radius=max(4, half / 9))
    axis.set_xlim(extent[0], extent[1])
    axis.set_ylim(extent[2], extent[3])
    return extent


def metric_panel(axis, rows, title="Measurements", flags=None, note=None):
    """Render a clean two-column measurement list.

    ``rows`` is a sequence of ``(label, value, status)`` or
    ``(label, value, status, limit)``. ``status`` (PASS/WARN/FAIL or None) adds
    a colored dot; ``limit`` is shown in muted text after the value.
    """

    axis.set_axis_off()
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_title(title)
    rows = [tuple(row) + (None,) * (4 - len(row)) for row in rows]
    available = 0.97 if not flags else 0.80
    step = min(0.072, available / max(len(rows), 1))
    y = 0.985
    for label, value, status, limit in rows:
        if label is None:
            y -= step * 0.5
            continue
        middle = y - step * 0.45
        label_y = middle + (step * 0.16 if limit else 0.0)
        if status:
            axis.scatter([0.012], [label_y], s=26, color=_status_color(status), zorder=3,
                         clip_on=False)
        axis.text(0.04, label_y, str(label), ha="left", va="center", fontsize=8.5,
                  color=MUTED)
        if limit:
            axis.text(0.04, middle - step * 0.2, str(limit), ha="left", va="center",
                      fontsize=7, color=FAINT)
        axis.text(0.995, middle, str(value), ha="right", va="center", fontsize=8.8,
                  color=INK, fontweight="bold")
        axis.plot([0.0, 1.0], [y - step, y - step], color="#eaeef2", linewidth=0.6)
        y -= step
    if flags:
        y -= 0.02
        axis.text(0.0, y, "Flags", ha="left", va="top", fontsize=8.5,
                  color=MUTED, fontweight="bold")
        y -= 0.05
        x = 0.0
        for flag in flags:
            label = str(flag)
            width = 0.0125 * len(label) + 0.04
            if x + width > 1.0:
                x = 0.0
                y -= 0.055
            axis.text(x, y, label, ha="left", va="top", fontsize=7.3, color=INK,
                      bbox=dict(boxstyle="round,pad=0.3,rounding_size=0.4",
                                facecolor="#fff1e5" if "FAIL" not in label else "#ffebe9",
                                edgecolor="none"))
            x += width
    if note:
        axis.text(0.0, 0.0, note, ha="left", va="bottom", fontsize=7.5,
                  color=FAINT, wrap=True)


def _check_status(value, warn, fail, direction="high"):
    """PASS/WARN/FAIL for one value against optional warn/fail limits."""

    value = _finite(value)
    if value is None:
        return None
    warn, fail = _finite(warn), _finite(fail)
    if direction == "high":
        if fail is not None and value > fail:
            return "FAIL"
        if warn is not None and value > warn:
            return "WARN"
    else:
        if fail is not None and value < fail:
            return "FAIL"
        if warn is not None and value < warn:
            return "WARN"
    return "PASS" if warn is not None or fail is not None else None


def _limit_text(warn, fail, direction="high", spec="{:.3g}"):
    symbol = ">" if direction == "high" else "<"
    parts = []
    if _finite(warn) is not None:
        parts.append("warn {} {}".format(symbol, spec.format(float(warn))))
    if _finite(fail) is not None:
        parts.append("fail {} {}".format(symbol, spec.format(float(fail))))
    return ", ".join(parts)


def _threshold_lines(axis, warn=None, fail=None, orientation="vertical"):
    draw = axis.axvline if orientation == "vertical" else axis.axhline
    if _finite(warn) is not None:
        draw(float(warn), color=STATUS_COLORS["WARN"], linestyle="--", linewidth=1.0,
             label="warn limit")
    if _finite(fail) is not None:
        draw(float(fail), color=STATUS_COLORS["FAIL"], linestyle="--", linewidth=1.0,
             label="fail limit")


def _mosaic(cube, columns=None, gap=2, normalize=True):
    """Tile a cube of small images, each normalized to its own peak."""

    cube = np.asarray(cube, dtype=float)
    if cube.ndim != 3 or cube.shape[0] == 0:
        return None
    count = cube.shape[0]
    columns = columns or int(np.ceil(np.sqrt(count)))
    rows = int(np.ceil(count / columns))
    height, width = cube.shape[1:]
    mosaic = np.full((rows * height + (rows - 1) * gap,
                      columns * width + (columns - 1) * gap), np.nan)
    for index, image in enumerate(cube):
        row, column = divmod(index, columns)
        tile = np.array(image, dtype=float)
        if normalize:
            peak = np.nanmax(np.abs(tile)) if np.any(np.isfinite(tile)) else 0
            if peak > 0:
                tile = tile / peak
        y0 = (rows - 1 - row) * (height + gap)
        x0 = column * (width + gap)
        mosaic[y0:y0 + height, x0:x0 + width] = tile
    return mosaic


def _box_text(box):
    if box is None:
        return "—"
    values = list(box) if np.ndim(box) else [box, box]
    return "×".join("{:g}".format(float(value)) for value in values[:2]) + " px"


def _block_mean(values, exclude=None, block=8, minimum_fraction=0.3):
    """Mean of ``values`` in ``block``×``block`` cells, ignoring excluded pixels.

    Cells with fewer than ``minimum_fraction`` usable pixels are NaN.
    """

    values = np.asarray(values, dtype=float)
    usable = np.isfinite(values)
    if exclude is not None:
        usable &= ~np.asarray(exclude, dtype=bool)
    ny, nx = (np.array(values.shape) // block) * block
    values = np.where(usable, values, 0.0)[:ny, :nx]
    usable = usable[:ny, :nx]
    shape = (ny // block, block, nx // block, block)
    total = values.reshape(shape).sum(axis=(1, 3))
    count = usable.reshape(shape).sum(axis=(1, 3))
    with np.errstate(all="ignore"):
        mean = total / count
    mean[count < minimum_fraction * block * block] = np.nan
    return mean


def _compress_ranges(values, limit=6):
    """Summarize sorted integer indices as 'a-b, c, d-e'."""

    values = sorted(int(value) for value in values or [])
    if not values:
        return "none"
    ranges = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append((start, previous))
        start = previous = value
    ranges.append((start, previous))
    text = ", ".join(
        str(a) if a == b else "{}–{}".format(a, b) for a, b in ranges[:limit]
    )
    if len(ranges) > limit:
        text += " (+{} more)".format(len(ranges) - limit)
    return text


def _target_pixel(ccd, metadata):
    """Project the adopted target position onto an image, if possible."""

    metadata = metadata or {}
    wcs = getattr(ccd, "wcs", None)
    ra, dec = metadata.get("adopted_ra_deg"), metadata.get("adopted_dec_deg")
    if wcs is None or _finite(ra) is None or _finite(dec) is None:
        return None, None
    try:
        from astropy.coordinates import SkyCoord

        x, y = wcs.celestial.world_to_pixel(SkyCoord(float(ra), float(dec), unit="deg"))
        return float(x), float(y)
    except Exception:
        return None, None


def _compass(axis, ccd, length_fraction=0.09):
    """Draw N/E arrows in the image corner from the WCS, when available."""

    wcs = getattr(ccd, "wcs", None)
    data = _data(ccd)
    if wcs is None or data is None:
        return
    try:
        celestial = wcs.celestial
        ny, nx = data.shape
        x0, y0 = 0.08 * nx, 0.08 * ny
        sky = celestial.pixel_to_world(x0, y0)
        step = length_fraction * min(nx, ny)
        scale = np.mean(np.abs(np.diag(celestial.pixel_scale_matrix))) * 3600.0
        from astropy import units as u

        for name, offset in (("N", (0, 1)), ("E", (1, 0))):
            moved = sky.spherical_offsets_by(
                offset[0] * step * scale * u.arcsec, offset[1] * step * scale * u.arcsec
            )
            x1, y1 = celestial.world_to_pixel(moved)
            axis.annotate("", xy=(x1, y1), xytext=(x0, y0),
                          arrowprops=dict(arrowstyle="-|>", color="white",
                                          linewidth=1.0, shrinkA=0, shrinkB=0))
            axis.text(x1 + (x1 - x0) * 0.18, y1 + (y1 - y0) * 0.18, name,
                      color="white", fontsize=7.5, ha="center", va="center",
                      fontweight="bold")
    except Exception:
        return


# ---------------------------------------------------------------------------
# Generic cards
# ---------------------------------------------------------------------------

@_styled
def plot_stage_status(stage_title, status, image_label=None, subtitle=None,
                      reason=None, rows=None, check=None, output_path=None,
                      show=False):
    """A compact card for skipped, blocked, or failed stages.

    Shows the status, the recorded reason or the last lines of the error, and
    any available measurements, so every stage leaves something to inspect.
    """

    title = stage_title if not image_label else "{}  ·  {}".format(stage_title, image_label)
    figure, grid = _new_figure(title, subtitle, status, check, size=(12, 4.6),
                               rows=1, columns=2 if rows else 1,
                               width_ratios=[1.4, 1] if rows else None)
    text_axis = figure.add_subplot(grid[0, 0])
    text_axis.set_axis_off()
    text_axis.set_title("What happened")
    message = str(reason or "No reason recorded.")
    lines = [line for line in message.strip().splitlines() if line.strip()]
    if len(lines) > 14:
        lines = ["…"] + lines[-13:]
    text_axis.text(0.0, 0.97, "\n".join(line[:110] for line in lines),
                   ha="left", va="top", fontsize=8.5, color=INK,
                   family="monospace" if len(lines) > 2 else None,
                   transform=text_axis.transAxes)
    if rows:
        metric_panel(figure.add_subplot(grid[0, 1]), rows)
    return _finish(figure, output_path, show)


@_styled
def plot_stage_overview(stage_title, rows, metrics, output_path=None, show=False,
                        check=None):
    """One figure summarizing a stage across all images of a run.

    Parameters
    ----------
    stage_title : str
        Human-readable stage name.
    rows : sequence of mapping
        One mapping per image with ``label``, ``status``, ``filter`` and the
        metric values named in ``metrics``.
    metrics : sequence of mapping
        Each with ``key``, ``label`` and optional ``warn``, ``fail``,
        ``direction`` ("high" or "low") and ``unit``.
    """

    rows = list(rows)
    metrics = [metric for metric in metrics if any(
        _finite(row.get(metric["key"])) is not None for row in rows
    )][:6]
    counts = {}
    for row in rows:
        counts[str(row.get("status"))] = counts.get(str(row.get("status")), 0) + 1
    worst = next((name for name in ("FAIL", "REJECTED", "WARN", "PASS", "APPROVED", "SKIPPED")
                  if counts.get(name)), None)
    subtitle = "{} images  ·  ".format(len(rows)) + "  ".join(
        "{} {}".format(count, name) for name, count in sorted(counts.items())
    )
    panels = max(1, len(metrics))
    columns = min(3, panels)
    grid_rows = int(np.ceil(panels / columns))
    height = 1.3 + 3.0 * grid_rows + 0.12 * len(rows)
    figure, grid = _new_figure(
        "{}  ·  run overview".format(stage_title), subtitle, worst, check,
        size=(5.4 * columns + 1.0, height), rows=grid_rows, columns=columns,
    )
    labels = [str(row.get("label", "")) for row in rows]
    short = [label.split(".")[0] if len(label) > 14 else label for label in labels]
    positions = np.arange(len(rows))
    if not metrics:
        axis = figure.add_subplot(grid[0, 0])
        _empty(axis, "No numeric measurements for this stage.")
        return _finish(figure, output_path, show)
    for index, metric in enumerate(metrics):
        axis = figure.add_subplot(grid[index // columns, index % columns])
        values = np.array([_finite(row.get(metric["key"])) for row in rows], dtype=float)
        colors = [_status_color(row.get("status")) for row in rows]
        finite = values[np.isfinite(values)]
        # Quantities far from zero (magnitudes, sky levels) are shown as dots
        # around their median; small quantities as bars from zero.
        offset = finite.size and np.all(finite > 0) and finite.min() > 0.5 * finite.max()
        if offset:
            center = float(np.median(finite))
            axis.hlines(positions, center, np.where(np.isfinite(values), values, center),
                        color=RULE, linewidth=1.5)
            axis.scatter(np.where(np.isfinite(values), values, np.nan), positions, c=colors,
                         s=46, zorder=3)
            axis.axvline(center, color=FAINT, linewidth=0.8, linestyle=":")
        else:
            axis.barh(positions, np.nan_to_num(values, nan=0.0), color=colors,
                      alpha=0.85, height=0.62)
        for position, value in zip(positions, values):
            if np.isfinite(value):
                axis.annotate(_fmt(value, metric.get("spec", "{:.3g}")), (value, position),
                              xytext=(6, 0), textcoords="offset points", va="center",
                              ha="left", fontsize=7, color=MUTED)
            else:
                axis.text(0 if not offset else center, position, " —", va="center",
                          ha="left", fontsize=7, color=FAINT)
        _threshold_lines(axis, metric.get("warn"), metric.get("fail"))
        axis.set_yticks(positions)
        axis.set_yticklabels(short if index % columns == 0 else [], fontsize=7.5)
        if index % columns:
            axis.tick_params(axis="y", length=0)
        axis.invert_yaxis()
        unit = metric.get("unit")
        axis.set_title(metric["label"] + (" [{}]".format(unit) if unit else ""))
        if finite.size and offset:
            span = max(finite.max() - finite.min(), 1e-3 * abs(finite.max()))
            axis.set_xlim(finite.min() - 0.15 * span, finite.max() + 0.35 * span)
        elif finite.size:
            high = max(np.max(finite), _finite(metric.get("fail")) or -np.inf,
                       _finite(metric.get("warn")) or -np.inf)
            low = min(0.0, np.min(finite))
            axis.set_xlim(low * 1.25 if low < 0 else 0.0, high * 1.18 if high > 0 else 1.0)
        _grid(axis, "x")
    return _finish(figure, output_path, show)


# ---------------------------------------------------------------------------
# read
# ---------------------------------------------------------------------------

@_styled
def plot_read_diagnostics(ccd, metadata, output_path=None, show=False, status=None):
    """Raw frame, pixel distribution, edge profiles, and parsed metadata."""

    metadata = metadata or {}
    data = _data(ccd)
    status = status or metadata.get("metadata_status")
    figure, grid = _new_figure(
        "Read  ·  " + _image_label(metadata), _image_subtitle(metadata), status,
        "the frame looks like reduced sky; filter, exposure, gain and saturation "
        "were read correctly; few non-finite pixels.",
        rows=2, columns=3, width_ratios=[1.35, 1.0, 0.95],
    )
    image_axis = figure.add_subplot(grid[:, 0])
    show_sky(image_axis, data, "Frame as read (z-scale)")
    nonfinite_count = None
    if data is not None:
        bad = ~np.isfinite(data)
        nonfinite_count = int(bad.sum())
        overlay_mask(image_axis, bad, RED, 0.9)
        if bad.any():
            image_axis.plot([], [], "s", color=RED, markersize=6,
                            label="non-finite values (NaN or inf): {:,} px".format(nonfinite_count))
        x, y = _target_pixel(ccd, metadata)
        mark_target(image_axis, x, y)
        _compass(image_axis, ccd)
        _image_legend(image_axis)

    histogram = figure.add_subplot(grid[0, 1])
    histogram.set_title("Pixel values (log scale)")
    values = _sample(data, 400000) if data is not None else np.array([])
    if values.size:
        saturation = _finite(metadata.get("saturation"))
        positive = values[values > 0]
        if positive.size:
            low = max(float(np.percentile(positive, 0.1)), 1e-3)
            high = float(positive.max())
            if saturation is not None:
                high = max(high, saturation * 1.1)
            bins = np.geomspace(low, high, 160)
            histogram.hist(positive, bins=bins, color=BLUE, alpha=0.85, log=True)
            histogram.set_xscale("log")
        sky, noise = _robust(values)
        if sky is not None and sky > 0:
            histogram.axvline(sky, color=INK, linewidth=1.0,
                              label="median {:.0f}".format(sky))
        if saturation is not None:
            histogram.axvline(saturation, color=RED, linestyle="--", linewidth=1.0,
                              label="saturation {:.0f}".format(saturation))
        notes = []
        full = _data(ccd)
        if saturation is not None:
            notes.append("{} px ≥ saturation".format(int(np.count_nonzero(full >= saturation))))
        nonpositive = int(np.count_nonzero(full <= 0))
        if nonpositive:
            notes.append("{} px ≤ 0 (not shown)".format(nonpositive))
        if notes:
            histogram.text(0.98, 0.96, "\n".join(notes), transform=histogram.transAxes,
                           ha="right", va="top", fontsize=7.5, color=MUTED)
        histogram.set_xlabel("pixel value [{}]".format(getattr(ccd, "unit", "") or "adu"))
        histogram.set_ylabel("pixels")
        _legend(histogram, loc="upper left")
        _grid(histogram, "y")
    else:
        _empty(histogram, "No finite pixels")

    profiles = figure.add_subplot(grid[1, 1])
    profiles.set_title("Median column and row levels")
    if data is not None and np.any(np.isfinite(data)):
        with np.errstate(all="ignore"):
            columns = np.nanmedian(data, axis=0)
            rows = np.nanmedian(data, axis=1)
        profiles.plot(np.arange(columns.size), columns, color=BLUE, linewidth=1.0,
                      label="column medians (vs x)")
        profiles.plot(np.arange(rows.size), rows, color=AMBER, linewidth=1.0,
                      label="row medians (vs y)")
        center, spread = _robust(np.concatenate([columns, rows]))
        if center is not None and spread:
            profiles.set_ylim(center - 12 * spread, center + 12 * spread)
        profiles.set_xlabel("pixel index")
        profiles.set_ylabel("median value")
        _legend(profiles, loc="lower center")
        _grid(profiles)
    else:
        _empty(profiles, "Not available")

    table = figure.add_subplot(grid[:, 2])
    finite_fraction = _finite(metadata.get("finite_fraction"))
    rows = [
        ("Object", metadata.get("object") or "—", None),
        ("Filter", metadata.get("filter") or "—",
         None if metadata.get("filter") else "FAIL"),
        ("Exposure", _fmt(metadata.get("exposure_time"), "{:g}", "s"), None),
        ("Mid-exposure", str(metadata.get("date_mid_utc") or metadata.get("date_obs") or "—")[:19], None),
        ("MJD (mid)", _fmt(metadata.get("mjd_mid"), "{:.5f}"), None),
        ("Airmass", _fmt(metadata.get("airmass"), "{:.3f}"), None),
        ("Gain", _fmt(metadata.get("gain"), "{:.3g}", "e⁻/adu"), None),
        ("Read noise", _fmt(metadata.get("read_noise"), "{:.3g}", "e⁻"), None),
        ("Saturation", _fmt(metadata.get("saturation"), "{:.0f}"), None),
        ("Pixel scale", _fmt(metadata.get("pixel_scale"), "{:.3f}", "″/px"), None),
        ("Image size", "{} × {}".format(metadata.get("shape_x", "—"), metadata.get("shape_y", "—")), None),
        ("Data HDU", str(metadata.get("data_hdu", "—")), None),
        ("WCS", "valid" if metadata.get("wcs_valid") else "missing",
         "PASS" if metadata.get("wcs_valid") else "WARN"),
        ("Finite pixels", _fmt(None if finite_fraction is None else 100 * finite_fraction, "{:.2f}", "%"),
         _check_status(finite_fraction, 0.99, 0.90, "low")),
        ("Non-finite values", "—" if nonfinite_count is None else "{:,} px".format(nonfinite_count),
         None, "NaN or inf; usually overscan or flat holes"),
        ("Target from", str(metadata.get("adopted_position_source") or "—"), None),
    ]
    problems = list(metadata.get("quality_flags") or []) + [
        "MISSING:" + str(name) for name in metadata.get("missing_metadata") or []
    ] + ["CONFLICT:" + str(item.get("name", item)) if isinstance(item, dict) else "CONFLICT:" + str(item)
         for item in metadata.get("metadata_conflicts") or []]
    metric_panel(table, rows, "Header and metadata", flags=problems)
    return _finish(figure, output_path, show)


# ---------------------------------------------------------------------------
# region
# ---------------------------------------------------------------------------

@_styled
def plot_region_diagnostics(ccd, region, diagnostics=None, metadata=None,
                            output_path=None, show=False, status=None):
    """Valid-pixel region, header data section, trimmed edges, and crop."""

    from matplotlib.patches import Rectangle

    metadata = metadata or {}
    region = region or {}
    diagnostics = diagnostics or {}
    data = _data(ccd)
    flags = list(region.get("region_flags") or [])
    status = status or ("WARN" if flags else "PASS")
    figure, grid = _new_figure(
        "Region  ·  " + _image_label(metadata), _image_subtitle(metadata), status,
        "red/amber (hatched) pixels are not used: they should be only overscan, "
        "blank or damaged edges, and the target should sit well inside the rest.",
        rows=2, columns=3, width_ratios=[1.35, 1.0, 0.95],
    )
    image_axis = figure.add_subplot(grid[:, 0])
    show_sky(image_axis, data, "Full frame: what will and will not be used")
    valid = diagnostics.get("full_valid_mask")
    section = (region.get("header_section") or {})
    bounds = section.get("bounds")
    keyword = section.get("keyword") or "header"
    unused_columns = unused_rows = np.zeros(0, dtype=bool)
    if valid is not None and data is not None and np.shape(valid) == data.shape:
        valid = np.asarray(valid, dtype=bool)
        ny, nx = data.shape
        outside = np.zeros(data.shape, dtype=bool)
        if bounds and section.get("applied"):
            x0, x1, y0, y1 = bounds
            outside[:] = True
            outside[y0:y1, x0:x1] = False
        trimmed = ~valid & ~outside
        overlay_mask(image_axis, outside, RED, 0.55)
        overlay_mask(image_axis, trimmed, AMBER, 0.6)
        # Whole unused columns and rows can be a few pixels wide; outline them
        # with hatched boxes so they are visible at any zoom.
        unused_columns = ~np.any(valid, axis=0)
        unused_rows = ~np.any(valid, axis=1)
        for start, stop in _runs(unused_columns):
            image_axis.add_patch(Rectangle((start - 0.5, -0.5), stop - start + 1, ny,
                                           facecolor="none", edgecolor=RED, hatch="////",
                                           linewidth=1.4))
        for start, stop in _runs(unused_rows):
            image_axis.add_patch(Rectangle((-0.5, start - 0.5), nx, stop - start + 1,
                                           facecolor="none", edgecolor=RED, hatch="////",
                                           linewidth=1.4))
        # Call out each unused strip with a label, since a few pixels at the
        # frame edge are hard to see at this scale.
        callouts = [("columns", start, stop, ((start + stop) / 2.0, ny * 0.12))
                    for start, stop in _runs(unused_columns)]
        callouts += [("rows", start, stop, (nx * 0.3, (start + stop) / 2.0))
                     for start, stop in _runs(unused_rows)]
        for name, start, stop, (px, py) in callouts[:6]:
            text = "{} {} not used".format(name if stop > start else name[:-1],
                                           "{}–{}".format(start, stop) if stop > start else start)
            tx = px + (0.12 * nx if px < nx / 2 else -0.12 * nx) if name == "columns" else px
            ty = py + (0.08 * ny if py < ny / 2 else -0.08 * ny) if name == "rows" else py
            image_axis.annotate(text, xy=(px, py), xytext=(tx, ty), fontsize=7.5, color=INK,
                                ha="left" if tx > px else "right" if name == "columns" else "center",
                                va="center",
                                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                          edgecolor=RED, linewidth=0.8, alpha=0.95),
                                arrowprops=dict(arrowstyle="-|>", color=RED, linewidth=1.0))
        if outside.any():
            image_axis.plot([], [], "s", color=RED, markersize=7,
                            label="not used: outside {} ({:,} px)".format(keyword, int(outside.sum())))
        if trimmed.any():
            image_axis.plot([], [], "s", color=AMBER, markersize=7,
                            label="not used: edge trim / NaN ({:,} px)".format(int(trimmed.sum())))
    if bounds and section.get("applied"):
        x0, x1, y0, y1 = bounds
        image_axis.add_patch(Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0,
                                       fill=False, edgecolor=TEAL, linewidth=1.2,
                                       linestyle="--",
                                       label="{} (data section)".format(keyword)))
    slices = diagnostics.get("crop_slices")
    if slices:
        (y0, y1), (x0, x1) = slices
        image_axis.add_patch(Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0,
                                       fill=False, edgecolor=VIOLET, linewidth=1.4,
                                       label="processing crop"))
    tx, ty = region.get("target_x"), region.get("target_y")
    if slices and _finite(tx) is not None:
        tx, ty = tx + slices[1][0], ty + slices[0][0]
    mark_target(image_axis, tx, ty)
    _image_legend(image_axis)

    edges = region.get("empirical_edges") or {}
    for index, (axis_name, label) in enumerate((("x", "column"), ("y", "row"))):
        cell = grid[index, 1].subgridspec(2, 2, height_ratios=[2.0, 1.0])
        axis = figure.add_subplot(cell[0, :])
        axis.set_title("Median level of each {} (edge check)".format(label))
        if data is None:
            _empty(axis, "Not available")
            continue
        with np.errstate(all="ignore"):
            profile = np.nanmedian(data, axis=0 if axis_name == "x" else 1)
        unused = unused_columns if axis_name == "x" else unused_rows
        if unused.size != profile.size:
            unused = np.zeros(profile.size, dtype=bool)
        used_profile = np.where(unused, np.nan, profile)
        center, spread = _robust(used_profile[np.isfinite(used_profile)])
        if center is not None and spread:
            low, high = center - 15 * spread, center + 15 * spread
        else:
            low, high = np.nanmin(profile), np.nanmax(profile)
        section_pair = None
        if bounds and section.get("applied"):
            section_pair = (bounds[0], bounds[1]) if axis_name == "x" else (bounds[2], bounds[3])
        _edge_profile(axis, profile, unused, low, high, section_pair, legend=True)
        axis.set_xlabel("{} [pixel]".format(axis_name))
        axis.set_ylabel("median value")
        # Close-ups of the two ends, where unused lines are only a few pixels wide.
        width = int(max(30, 4 * max([stop - start + 1 for start, stop in _runs(unused)] or [0])))
        for column, (start, stop, side) in enumerate(((0, width, "first"),
                                                      (profile.size - width, profile.size, "last"))):
            zoom = figure.add_subplot(cell[1, column])
            zoom.set_title("{} {} {}s".format(side, width, label), fontsize=8.5)
            _edge_profile(zoom, profile, unused, low, high, section_pair,
                          window=(max(0, start), min(profile.size, stop)))

    table = figure.add_subplot(grid[:, 2])
    valid_fraction = _finite(region.get("valid_fraction_full"))
    trimmed = ", ".join(
        "{} {}".format(side, edges.get(side, 0))
        for side in ("bottom", "top", "left", "right") if edges.get(side)
    ) or "none"
    outside = ", ".join(
        "{} {}".format(side, edges.get(side + "_outside_section"))
        for side in ("bottom", "top", "left", "right")
        if edges.get(side + "_outside_section")
    ) or "none"
    crop = region.get("crop") or {}
    rows = [
        ("Usable fraction of frame", _fmt(None if valid_fraction is None else 100 * valid_fraction, "{:.1f}", "%"),
         _check_status(valid_fraction, 0.80, 0.50, "low")),
        ("Header section", "{} {}".format(section.get("keyword") or "", bounds or "").strip() or "none", None),
        ("Unused columns", _compress_ranges(np.flatnonzero(unused_columns).tolist()), None),
        ("Unused rows", _compress_ranges(np.flatnonzero(unused_rows).tolist()), None),
        ("Lines outside section", outside, None),
        ("Extra edge lines trimmed", trimmed, "WARN" if trimmed != "none" else "PASS"),
        ("Uniform border crop", _fmt(region.get("edge_crop_pixels"), "{:g}", "px"), None),
        ("Processing crop", "applied" if crop.get("applied") else str(crop.get("reason") or "none"), None),
        ("Target inside", "yes" if region.get("target_inside") else "no",
         "PASS" if region.get("target_inside") else "FAIL"),
        ("Target to edge", _fmt(region.get("target_edge_distance_pixels"), "{:.0f}", "px"), None),
    ]
    metric_panel(table, rows, "Region", flags=flags)
    return _finish(figure, output_path, show)


def _edge_profile(axis, profile, unused, low, high, section_pair=None, window=None,
                  legend=False):
    """Plot a row/column median profile with the unused lines made obvious.

    Used lines are a blue line; unused lines are gray points (clipped to the
    panel, with red triangles where they are off scale) over a red hatched
    band. ``window`` restricts the plot to a range of lines for edge close-ups
    (only that range is drawn, so nothing spills outside the panel).
    """

    profile = np.asarray(profile, dtype=float)
    unused = np.asarray(unused, dtype=bool)
    first, last = (0, profile.size) if window is None else (int(window[0]), int(window[1]))
    positions = np.arange(first, last)
    values = profile[first:last]
    hidden = unused[first:last]
    used_values = np.where(hidden, np.nan, values)
    if window is not None and np.any(np.isfinite(used_values)):
        local = used_values[np.isfinite(used_values)]
        pad = max(0.15 * (local.max() - local.min()), 1e-3 * abs(np.median(local)) + 1e-6)
        low, high = local.min() - pad, local.max() + pad
    axis.plot(positions, used_values, color=BLUE, linewidth=1.0, label="used")
    if hidden.any():
        clipped = np.clip(np.where(hidden, values, np.nan), low, high)
        off_scale = hidden & ((values < low) | (values > high) | ~np.isfinite(values))
        axis.plot(positions[hidden], clipped[hidden], ".", color=FAINT, markersize=3,
                  label="not used")
        if off_scale.any():
            axis.plot(positions[off_scale], np.full(int(off_scale.sum()), high), "^",
                      color=RED, markersize=4, label="not used, off scale")
        for start, stop in _runs(hidden):
            axis.axvspan(first + start - 0.5, first + stop + 0.5, facecolor=RED, alpha=0.2,
                         edgecolor=RED, hatch="////", linewidth=0)
    if section_pair is not None:
        for value in section_pair:
            if first <= value <= last:
                axis.axvline(value - 0.5, color=TEAL, linestyle="--", linewidth=1.0)
    axis.set_ylim(low, high)
    axis.set_xlim(first - 0.5, last - 0.5)
    if window is not None:
        axis.tick_params(labelsize=6.5)
    if legend:
        _legend(axis, loc="lower center", ncol=3, fontsize=7)
    _grid(axis)


def _runs(mask):
    """Start/stop index pairs of consecutive True values in a 1-D mask."""

    mask = np.asarray(mask, dtype=bool)
    runs = []
    start = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, mask.size - 1))
    return runs


# ---------------------------------------------------------------------------
# masks
# ---------------------------------------------------------------------------

_MASK_LABELS = {
    "nonfinite": "non-finite values",
    "input": "input mask",
    "saturation": "saturation (grown)",
    "bad_lines": "bad rows/columns",
    "amplifier": "amplifier seams",
    "trails": "trails",
    "manual": "manual",
}


@_styled
def plot_mask_diagnostics(ccd, components, info, metadata=None, output_path=None,
                          show=False, status=None):
    """Every mask component on the image, its pixel fraction, and close-ups."""

    metadata = metadata or {}
    components = components or {}
    info = info or {}
    data = _data(ccd)
    flags = list(info.get("flags") or [])
    status = status or ("WARN" if flags else "PASS")
    figure, grid = _new_figure(
        "Masks  ·  " + _image_label(metadata), _image_subtitle(metadata), status,
        "colored pixels are excluded from all measurements: they should cover "
        "saturated stars, bad lines and trails, not clean sky or the target.",
        rows=2, columns=3, width_ratios=[1.35, 1.0, 0.95],
    )
    image_axis = figure.add_subplot(grid[:, 0])
    show_sky(image_axis, data, "Mask components")
    fractions = info.get("component_fractions") or {}
    for name in ("nonfinite", "input", "saturation", "bad_lines", "amplifier", "trails", "manual"):
        mask = components.get(name)
        if mask is None or data is None or np.shape(mask) != data.shape:
            continue
        mask = np.asarray(mask, dtype=bool)
        overlay_mask(image_axis, mask, MASK_COLORS[name], 0.6)
        if mask.any():
            image_axis.plot([], [], "s", color=MASK_COLORS[name], markersize=7,
                            label="{} {:.2f}%".format(_MASK_LABELS[name], 100 * mask.mean()))
    target = info.get("target") or {}
    mark_target(image_axis, target.get("target_x"), target.get("target_y"))
    _image_legend(image_axis)

    bars = figure.add_subplot(grid[0, 1])
    bars.set_title("Fraction of pixels masked")
    names = [name for name in ("nonfinite", "input", "saturation", "bad_lines",
                               "amplifier", "trails", "manual") if name in fractions]
    values = [100 * float(fractions.get(name) or 0.0) for name in names]
    names.append("combined")
    values.append(100 * float(info.get("masked_fraction") or fractions.get("combined") or 0.0))
    colors = [MASK_COLORS.get(name, INK) for name in names[:-1]] + [INK]
    positions = np.arange(len(names))
    bars.barh(positions, values, color=colors, height=0.6)
    for position, value in zip(positions, values):
        bars.text(value, position, " {:.2f}%".format(value), va="center", fontsize=7.5,
                  color=MUTED)
    bars.set_yticks(positions)
    bars.set_yticklabels([_MASK_LABELS.get(name, name) for name in names])
    bars.invert_yaxis()
    bars.set_xlabel("% of pixels")
    bars.set_xlim(0, max(values + [1.0]) * 1.25)
    _grid(bars, "x")

    zoom = figure.add_subplot(grid[1, 1])
    trails = info.get("trail_list") or []
    cx, cy = target.get("target_x"), target.get("target_y")
    if data is not None and _finite(cx) is None:
        cy, cx = (np.array(data.shape) - 1) / 2.0
    layers = [
        (components.get(name), MASK_COLORS[name], 0.55)
        for name in ("nonfinite", "input", "saturation", "bad_lines", "amplifier", "trails", "manual")
    ]
    _zoom_panel(zoom, data, cx, cy, 80, "Target neighborhood (160 px)", layers,
                target=(target.get("target_x"), target.get("target_y"))
                if _finite(target.get("target_x")) is not None else None)

    saturation = info.get("saturation") or {}
    lines = info.get("bad_lines") or {}
    seams = info.get("amplifier") or {}
    masked = _finite(info.get("masked_fraction"))
    rows = [
        ("Masked overall", _fmt(None if masked is None else 100 * masked, "{:.2f}", "%"), None),
        ("Saturation level used", _fmt(saturation.get("effective_level"), "{:.0f}"), None),
        ("Saturated pixels", _fmt(None if saturation.get("saturated_fraction") is None
                                  else 100 * saturation["saturated_fraction"], "{:.3f}", "%"), None),
        ("Saturated stars masked", str(len(saturation.get("regions") or [])), None,
         "core grown to {:.0%} of saturation + {:.0f} px".format(
             saturation.get("region_fraction") or 0, saturation.get("buffer_pixels") or 0)
         if saturation.get("buffer_pixels") is not None else None),
        ("Bad rows", _compress_ranges(lines.get("bad_rows")), "WARN" if lines.get("bad_rows") else None),
        ("Bad columns", _compress_ranges(lines.get("bad_columns")), "WARN" if lines.get("bad_columns") else None),
        ("Seam columns", _compress_ranges(seams.get("seam_columns")), None),
        ("Trails", str(len(trails)), "WARN" if trails else None),
        ("Target masked", "yes" if target.get("target_masked") else "no",
         "FAIL" if target.get("target_masked") else "PASS"),
        ("Target on a trail", "yes" if target.get("target_trail") else "no",
         "FAIL" if target.get("target_trail") else "PASS"),
    ]
    for trail in trails[:3]:
        if _finite(trail.get("start_x")) is not None:
            where = "  trail ({:.0f},{:.0f})→({:.0f},{:.0f})".format(
                trail["start_x"], trail["start_y"], trail["end_x"], trail["end_y"])
        else:
            where = "  trail at ({:.0f}, {:.0f})".format(trail.get("centroid_x", 0),
                                                       trail.get("centroid_y", 0))
        rows.append((where, "{:.0f} × {:.0f} px".format(trail.get("length_pixels") or 0,
                                                       trail.get("width_pixels") or 0), None))
    bleeds = (info.get("trails") or {}).get("bleeds") or []
    if bleeds:
        rows.append(("Bleeds/spikes (in saturation)", str(len(bleeds)), None))
    metric_panel(figure.add_subplot(grid[:, 2]), rows, "Masks", flags=flags)
    return _finish(figure, output_path, show)


# ---------------------------------------------------------------------------
# cosmic rays and fringe
# ---------------------------------------------------------------------------

@_styled
def plot_cosmic_ray_diagnostics(ccd, products, info, metadata=None,
                                output_path=None, show=False, status=None):
    """Cosmic-ray mask on the image, the target neighborhood and the densest region.

    Pixels that were already masked by earlier stages (saturation, trails,
    edges) are shown in gray, so you can see which areas no longer matter.
    """

    metadata = metadata or {}
    info = info or {}
    products = products or {}
    if info.get("skipped"):
        return plot_stage_status(
            "Cosmic rays", status or "SKIPPED", _image_label(metadata),
            _image_subtitle(metadata),
            "Cosmic-ray cleaning was not run: {}.\nEnable it with "
            "masks.cosmic_rays.enabled (requires astroscrappy).".format(info.get("skipped")),
            output_path=output_path, show=show)
    data = _data(ccd)
    mask = products.get("cosmic_mask")
    existing = getattr(ccd, "mask", None)
    if existing is not None and data is not None and np.shape(existing) == data.shape:
        existing = np.asarray(existing, dtype=bool)
    else:
        existing = None
    figure, grid = _new_figure(
        "Cosmic rays  ·  " + _image_label(metadata), _image_subtitle(metadata),
        status or ("WARN" if info.get("flags") else "PASS"),
        "pink detections should be sharp single hits, never the cores of stars; gray "
        "areas were already masked by earlier stages.",
        rows=2, columns=3, width_ratios=[1.3, 0.85, 0.9], size=(16.5, 9.0))
    axis = figure.add_subplot(grid[:, 0])
    show_sky(axis, data, "Cosmic rays on the frame")
    overlay_mask(axis, existing, SLATE, 0.5)
    overlay_mask(axis, mask, PINK, 0.95)
    if existing is not None and existing.any():
        axis.plot([], [], "s", color=SLATE, markersize=7,
                  label="already masked ({:.1f}%)".format(100 * existing.mean()))
    if mask is not None and np.any(mask):
        axis.plot([], [], "s", color=PINK, markersize=7,
                  label="cosmic rays ({:,} px)".format(int(np.count_nonzero(mask))))
    tx, ty = _target_pixel(ccd, metadata)
    mark_target(axis, tx, ty)
    _image_legend(axis)

    layers = [(existing, SLATE, 0.45), (mask, PINK, 0.9)]
    target_zoom = figure.add_subplot(grid[0, 1])
    _zoom_panel(target_zoom, data, tx, ty, 50, "Target neighborhood (100 px)", layers,
                target=(tx, ty) if _finite(tx) is not None else None)
    dense_zoom = figure.add_subplot(grid[1, 1])
    if mask is not None and np.any(mask) and data is not None:
        from scipy import ndimage

        density = ndimage.uniform_filter(np.asarray(mask, dtype=float), 64)
        cy, cx = np.unravel_index(np.argmax(density), density.shape)
        _zoom_panel(dense_zoom, data, cx, cy, 50, "Densest region (100 px)", layers)
    else:
        _empty(dense_zoom, "No cosmic rays flagged", "Densest region")
    rows = [
        ("Mode", str(info.get("mode", "—")), None),
        ("Flagged pixels", _fmt(info.get("cosmic_pixel_count"), "{:.0f}"), None),
        ("Flagged fraction", _fmt(None if info.get("cosmic_pixel_fraction") is None
                                  else 100 * info["cosmic_pixel_fraction"], "{:.3f}", "%"), None),
        ("Already masked", _fmt(None if existing is None else 100 * existing.mean(), "{:.1f}", "%"),
         None, "earlier stages; not searched"),
        ("Touches target", str(info.get("target_overlap")), "WARN" if info.get("target_overlap") else None,
         "detections on the target were not masked ({} px)".format(info.get("target_protected_pixels"))
         if info.get("target_protected_pixels") else None),
    ]
    for name, key in (("sigclip", "sigclip"), ("objlim", "objlim"), ("Gain", "gain"),
                      ("Read noise", "readnoise")):
        value = (info.get("parameters") or {}).get(key)
        if value is not None:
            rows.append((name, _fmt(value, "{:.3g}"), None))
    metric_panel(figure.add_subplot(grid[:, 2]), rows, "Cosmic rays", flags=info.get("flags"))
    return _finish(figure, output_path, show)


@_styled
def plot_fringe_diagnostics(ccd, products, info, metadata=None, output_path=None,
                            show=False, status=None):
    """Scaled fringe model and the image before and after correction."""

    metadata = metadata or {}
    info = info or {}
    products = products or {}
    if info.get("skipped") or not info.get("applied"):
        return plot_stage_status(
            "Fringe", status or "SKIPPED", _image_label(metadata),
            _image_subtitle(metadata),
            "Fringe correction was not applied: {}.\nIt is normally only needed "
            "for red filters (i, z) with a fringe map configured in fringe.map_path."
            .format(info.get("skipped") or "not applied"),
            output_path=output_path, show=show)
    data = _data(ccd)
    figure, grid = _new_figure(
        "Fringe  ·  " + _image_label(metadata), _image_subtitle(metadata),
        status or ("WARN" if info.get("flags") else "PASS"),
        "the corrected image should show no fringe pattern; the scale should be stable night to night.",
        rows=1, columns=4, width_ratios=[1, 1, 1, 0.85], size=(17, 6.0))
    show_map(figure, figure.add_subplot(grid[0, 0]), products.get("fringe_model"),
             "Scaled fringe model", symmetric=True)
    corrected = products.get("corrected")
    before = data if corrected is None else data
    show_sky(figure.add_subplot(grid[0, 1]), before, "Before correction")
    show_sky(figure.add_subplot(grid[0, 2]), corrected if corrected is not None else data,
             "After correction")
    rows = [("Scale", _fmt(info.get("scale"), "{:.4g}"), None)]
    metric_panel(figure.add_subplot(grid[0, 3]), rows, "Fringe", flags=info.get("flags"))
    return _finish(figure, output_path, show)


# ---------------------------------------------------------------------------
# background
# ---------------------------------------------------------------------------

@_styled
def plot_background_diagnostics(ccd, products, info, metadata=None, output_path=None,
                                show=False, status=None, settings=None):
    """Background model, noise map, normalized residuals, and their statistics.

    The most direct test of a background model is the residual sky divided by
    the RMS map: away from sources it should look like featureless noise
    centered on zero with unit width.
    """

    from matplotlib.patches import Rectangle

    metadata = metadata or {}
    products = products or {}
    info = info or {}
    settings_background = (settings or {}).get("background", {})
    flags = list(info.get("flags") or [])
    status = status or ("WARN" if flags else "PASS")
    model = products.get("background")
    rms = products.get("background_rms")
    corrected = products.get("background_subtracted")
    source_mask = products.get("background_mask")
    data = _data(ccd)
    if data is None and corrected is not None and model is not None:
        data = np.asarray(corrected) + np.asarray(model)

    figure, grid = _new_figure(
        "Background  ·  " + _image_label(metadata), _image_subtitle(metadata), status,
        "the residual map (sky ÷ RMS) shows only noise — no gradients, rings or "
        "dark halos around stars — and its histogram matches the N(0, 1) curve.",
        rows=2, columns=4, size=(18, 9.6), width_ratios=[1, 1, 1, 0.85],
    )
    input_axis = figure.add_subplot(grid[0, 0])
    show_sky(input_axis, data, "Input with excluded sources", labels=False)
    if source_mask is not None and data is not None and np.shape(source_mask) == data.shape:
        overlay_mask(input_axis, np.asarray(source_mask, dtype=bool), AMBER, 0.35)

    model_axis = figure.add_subplot(grid[0, 1])
    show_map(figure, model_axis, model, "Background model", label="level", labels=False)
    excluded = products.get("mesh_excluded")
    box = info.get("effective_box_size")
    if model is not None and box is not None and excluded is not None:
        excluded = np.asarray(excluded, dtype=bool)
        by, bx = (box if np.ndim(box) else (box, box))[:2]
        for (row, column) in np.argwhere(excluded):
            model_axis.add_patch(Rectangle((column * bx - 0.5, row * by - 0.5), bx, by,
                                           fill=False, edgecolor="white", linewidth=0.5,
                                           hatch="////", alpha=0.6))
        if excluded.any():
            model_axis.text(0.01, 0.99, "hatched: {} of {} boxes interpolated (too many masked pixels)".format(
                                int(excluded.sum()), excluded.size),
                            transform=model_axis.transAxes,
                            ha="left", va="top", fontsize=7, color=INK,
                            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                                      edgecolor="none", alpha=0.85))

    show_map(figure, figure.add_subplot(grid[0, 2]), rms, "Background RMS",
             label="RMS", labels=False)

    normalized = None
    if corrected is not None and rms is not None:
        with np.errstate(all="ignore"):
            normalized = np.asarray(corrected, dtype=float) / np.asarray(rms, dtype=float)
    residual_axis = figure.add_subplot(grid[1, 0])
    if normalized is not None:
        blanked = np.asarray(source_mask, dtype=bool) if source_mask is not None and \
            np.shape(source_mask) == normalized.shape else None
        block = 16
        binned = _block_mean(normalized, blanked, block)
        ny, nx = normalized.shape
        artist = residual_axis.imshow(
            binned, cmap=DIVERGING_CMAP, vmin=-0.25, vmax=0.25, origin="lower",
            interpolation="nearest",
            extent=(-0.5, binned.shape[1] * block - 0.5, -0.5, binned.shape[0] * block - 0.5))
        residual_axis.set_xlim(-0.5, nx - 0.5)
        residual_axis.set_ylim(-0.5, ny - 0.5)
        residual_axis.set_title("Residual sky ÷ RMS ({0}×{0} px means, sources excluded)".format(block))
        _image_axes(residual_axis, labels=False)
        _colorbar(figure, artist, residual_axis, "mean residual [σ]")
    else:
        _empty(residual_axis, "Background was not subtracted", "Residual sky ÷ RMS")

    histogram = figure.add_subplot(grid[1, 1])
    histogram.set_title("Sky pixels before and after subtraction")
    residual_stats = (None, None)
    before_stats = (None, None)
    if normalized is not None:
        keep = np.isfinite(normalized)
        if source_mask is not None and np.shape(source_mask) == normalized.shape:
            keep &= ~np.asarray(source_mask, dtype=bool)
        values = _sample(normalized[keep])
        if values.size:
            bins = np.linspace(-6, 6, 121)
            grid_x = np.linspace(-6, 6, 400)
            residual_stats = _robust(values)
            if data is not None and np.shape(data) == normalized.shape:
                # "Before" = the same sky pixels minus one flat sky level (the
                # median of the model, so both curves share the same estimator)
                # in units of the same RMS map: the difference between the two
                # curves is what the spatial structure of the model achieved.
                raw = np.asarray(data, dtype=float)
                level = float(np.nanmedian(np.asarray(model, dtype=float)))
                with np.errstate(all="ignore"):
                    before = (raw - level) / np.asarray(rms, dtype=float)
                before_values = _sample(before[keep & np.isfinite(before)])
                if before_values.size:
                    before_stats = _robust(before_values)
                    histogram.hist(before_values, bins=bins, density=True,
                                   histtype="step", color=FAINT, linewidth=1.5,
                                   label="before: {:+.2f}σ, width {:.2f}σ".format(*before_stats))
            histogram.hist(values, bins=bins, density=True, color=BLUE, alpha=0.6,
                           label="after: {:+.2f}σ, width {:.2f}σ".format(*residual_stats))
            histogram.plot(grid_x, np.exp(-0.5 * grid_x ** 2) / np.sqrt(2 * np.pi),
                           color=INK, linewidth=1.2, label="N(0, 1)")
            histogram.axvline(residual_stats[0], color=AMBER, linewidth=1.0)
            histogram.set_yscale("log")
            histogram.set_ylim(1e-5, 1.0)
            histogram.set_xlabel("(sky − model) / RMS    [before: model = one flat level]")
            histogram.set_ylabel("density")
            _legend(histogram, loc="upper right", fontsize=7)
            _grid(histogram)
    else:
        _empty(histogram, "Not available")

    profile_axis = figure.add_subplot(grid[1, 2])
    profile_axis.set_title("Median profiles (each minus its median)")
    profiles = products.get("profiles") or {}
    plotted = False
    for prefix, style in (("row", "-"), ("column", "--")):
        for key, color, label in (("input", FAINT, "input"), ("background", AMBER, "model"),
                                  ("corrected", TEAL, "corrected")):
            values = profiles.get("{}_{}".format(prefix, key))
            if values is None:
                continue
            values = np.asarray(values, dtype=float)
            center = np.nanmedian(values) if np.any(np.isfinite(values)) else 0.0
            profile_axis.plot(values - center, style, color=color, linewidth=1.0,
                              label="{} ({}s)".format(label, prefix))
            plotted = True
    if plotted:
        profile_axis.axhline(0, color=INK, linewidth=0.6)
        profile_axis.set_xlabel("row (solid) / column (dashed) index")
        profile_axis.set_ylabel("level − median")
        _legend(profile_axis, loc="best", ncol=2, fontsize=6.8)
        _grid(profile_axis)
    else:
        _empty(profile_axis, "Profiles not available")

    before = info.get("gradient_before") or {}
    after = info.get("gradient_after") or {}
    preservation = info.get("source_preservation") or {}
    source_mask_info = info.get("source_mask") or {}
    reduction = _finite(info.get("gradient_reduction_fraction"))
    median_change = _finite(preservation.get("median_fractional_change"))
    sky = _robust(_sample(model))[0] if model is not None else None
    noise = _robust(_sample(rms))[0] if rms is not None else None
    width = residual_stats[1]
    rows = [
        ("Mode", str(info.get("mode", "—")), None),
        ("Mesh size", _box_text(info.get("effective_box_size")), None,
         "requested {}".format(_box_text(info.get("requested_box_size")))),
        ("Boxes interpolated", _fmt(None if info.get("excluded_mesh_fraction") is None
                                    else 100 * info["excluded_mesh_fraction"], "{:.0f}", "%"),
         _check_status(info.get("excluded_mesh_fraction"),
                       (settings_background or {}).get("excluded_mesh_warn_fraction", 0.5), None),
         "measured {} of {} boxes".format(info.get("measured_mesh_count", "—"),
                                         info.get("mesh_count", "—"))),
        ("Masked as sources", _fmt(None if source_mask_info.get("combined_mask_fraction") is None
                                   else 100 * source_mask_info["combined_mask_fraction"], "{:.1f}", "%"), None),
        ("Sky level (median)", _fmt(sky, "{:.1f}"), None),
        ("Sky RMS (median)", _fmt(noise, "{:.2f}"), None),
        ("Gradient before", _fmt(before.get("peak_to_peak"), "{:.3g}"), None),
        ("Gradient after", _fmt(after.get("peak_to_peak"), "{:.3g}"), None),
        ("Gradient removed", _fmt(None if reduction is None else 100 * reduction, "{:.0f}", "%"), None),
        ("Residual median", _fmt(residual_stats[0], "{:+.3f}", "σ"),
         _check_status(None if residual_stats[0] is None else abs(residual_stats[0]), 0.1, 0.3),
         "warn > 0.1σ; before {}".format(_fmt(before_stats[0], "{:+.3f}", "σ"))),
        ("Residual width", _fmt(width, "{:.3f}", "σ"),
         _check_status(None if width is None else abs(width - 1.0), 0.15, 0.4),
         "ideal 1.0σ; before {}".format(_fmt(before_stats[1], "{:.3f}", "σ"))),
        ("Source flux change", _fmt(None if median_change is None else 100 * median_change, "{:.2f}", "%"),
         _check_status(median_change, 0.02, None), "warn > 2%"),
    ]
    metric_panel(figure.add_subplot(grid[:, 3]), rows, "Background", flags=flags,
                 note="Residual median/width are diagnostic only and do not change the stage status.")
    return _finish(figure, output_path, show)


# ---------------------------------------------------------------------------
# source detection and image quality
# ---------------------------------------------------------------------------

@_styled
def plot_image_quality_diagnostics(ccd, sources, segmentation=None, info=None,
                                   metadata=None, output_path=None, show=False,
                                   settings=None, status=None):
    """Detected sources, seeing across the detector, shapes, and quality limits."""

    from matplotlib.patches import Circle

    metadata = metadata or {}
    info = info or {}
    quality = (settings or {}).get("image_quality", {})
    status = status or info.get("quality_status")
    flags = list(info.get("quality_flags") or [])
    figure, grid = _new_figure(
        "Sources and seeing  ·  " + _image_label(metadata), _image_subtitle(metadata),
        status,
        "seeing stars (teal) are round, unsaturated stars across the whole field; "
        "FWHM should not vary systematically across the detector.",
        rows=2, columns=4, size=(18, 9.6), width_ratios=[1.35, 1, 1, 0.9],
    )
    data = _data(ccd)
    image_axis = figure.add_subplot(grid[:, 0])
    show_sky(image_axis, data, "Detections")
    count = 0 if sources is None else len(sources)
    x = _column(sources, "x")
    y = _column(sources, "y")
    fwhm_pixels = _column(sources, "fwhm_pixels")
    fwhm = _column(sources, "fwhm_arcsec")
    unit = "″"
    if not np.any(np.isfinite(fwhm)):
        fwhm, unit = fwhm_pixels, " px"
    ellipticity = _column(sources, "ellipticity")
    orientation = _column(sources, "orientation_deg")
    good = _bool_column(sources, "good_for_seeing")
    saturated = _bool_column(sources, "saturated")
    if count:
        for selection, color, label in ((~good & ~saturated, AMBER, "not used for seeing"),
                                        (good, TEAL, "seeing sample")):
            image_axis.scatter(x[selection], y[selection], s=30, facecolors="none",
                               edgecolors=color, linewidths=0.7,
                               label="{} ({})".format(label, int(np.count_nonzero(selection))))
        if np.any(saturated):
            image_axis.scatter(x[saturated], y[saturated], marker="x", s=22, color=RED,
                               linewidths=0.9, label="saturated ({})".format(int(saturated.sum())))
    local = info.get("local_target_background") or {}
    if _finite(local.get("x")) is not None:
        for radius_key in ("inner_radius_pixels", "outer_radius_pixels"):
            if _finite(local.get(radius_key)) is not None:
                image_axis.add_patch(Circle((local["x"], local["y"]), local[radius_key],
                                            fill=False, edgecolor=VIOLET, linewidth=1.0,
                                            linestyle="--"))
        image_axis.plot([], [], "--", color=VIOLET, label="target sky annulus")
    if data is not None:
        image_axis.set_xlim(-0.5, data.shape[1] - 0.5)
        image_axis.set_ylim(-0.5, data.shape[0] - 0.5)
    _image_legend(image_axis)

    fwhm_warn = quality.get("fwhm_warn_arcsec") if unit == "″" else None
    fwhm_fail = quality.get("fwhm_fail_arcsec") if unit == "″" else None
    histogram = figure.add_subplot(grid[0, 1])
    histogram.set_title("FWHM of the seeing sample")
    sample = fwhm[good & np.isfinite(fwhm)]
    if sample.size:
        upper = np.percentile(sample, 99) * 1.3
        if _finite(fwhm_warn) is not None:
            upper = max(upper, float(fwhm_warn) * 1.1)
        histogram.hist(sample, bins=np.linspace(0, upper, 50), color=TEAL, alpha=0.85)
        median = _finite(info.get("fwhm_arcsec" if unit == "″" else "fwhm_pixels"))
        if median is not None:
            histogram.axvline(median, color=INK, linewidth=1.2,
                              label="median {:.2f}{}".format(median, unit))
        _threshold_lines(histogram, fwhm_warn, fwhm_fail)
        histogram.set_xlabel("FWHM [{}]".format(unit.strip() or "arcsec"))
        histogram.set_ylabel("stars")
        _legend(histogram, loc="upper right")
        _grid(histogram, "y")
    else:
        _empty(histogram, "No seeing stars")

    field = figure.add_subplot(grid[0, 2])
    field.set_title("FWHM across the detector")
    if sample.size and data is not None:
        center = np.nanmedian(sample)
        span = max(0.25 * center, np.nanpercentile(np.abs(sample - center), 90))
        points = field.scatter(x[good], y[good], c=fwhm[good], s=14, cmap="coolwarm",
                               vmin=center - span, vmax=center + span)
        _colorbar(figure, points, field, "FWHM [{}]".format(unit.strip() or "arcsec"))
        field.set_xlim(0, data.shape[1])
        field.set_ylim(0, data.shape[0])
        field.set_aspect("equal")
        field.set_xlabel("x [pixel]")
        field.set_ylabel("y [pixel]")
    else:
        _empty(field, "Not available")

    shape = figure.add_subplot(grid[1, 1])
    shape.set_title("Shape: ellipticity and orientation")
    if count and data is not None:
        use = good & np.isfinite(ellipticity) & np.isfinite(orientation)
        length = 0.06 * min(data.shape) * ellipticity[use] / 0.2
        angle = np.deg2rad(orientation[use])
        dx, dy = length * np.cos(angle) / 2, length * np.sin(angle) / 2
        shape.plot(np.vstack([x[use] - dx, x[use] + dx]), np.vstack([y[use] - dy, y[use] + dy]),
                   color=BLUE, linewidth=1.1)
        shape.plot([0.04 * data.shape[1], 0.04 * data.shape[1] + 0.06 * min(data.shape)],
                   [0.04 * data.shape[0]] * 2, color=INK, linewidth=2)
        shape.text(0.04 * data.shape[1], 0.07 * data.shape[0], "e = 0.2", fontsize=7, color=INK)
        shape.set_xlim(0, data.shape[1])
        shape.set_ylim(0, data.shape[0])
        shape.set_aspect("equal")
        shape.set_xlabel("x [pixel]")
        shape.set_ylabel("y [pixel]")
    else:
        _empty(shape, "Not available")

    brightness = figure.add_subplot(grid[1, 2])
    brightness.set_title("FWHM versus brightness")
    flux = _column(sources, "flux")
    if count:
        moment = _column(sources, "moment_fwhm_pixels")
        if moment.size == count and np.any(np.isfinite(moment)):
            with np.errstate(all="ignore"):
                factor = np.nanmedian(fwhm / fwhm_pixels) if unit == "″" else 1.0
            if not np.isfinite(factor):
                factor = 1.0
            shown = np.isfinite(flux) & (flux > 0) & np.isfinite(moment)
            brightness.scatter(flux[shown], moment[shown] * factor, s=5, color=SLATE, alpha=0.45,
                               label="footprint width (not used)")
        positive = np.isfinite(flux) & (flux > 0) & np.isfinite(fwhm)
        brightness.scatter(flux[positive & ~good], fwhm[positive & ~good], s=8,
                           color=AMBER, alpha=0.7, label="Gaussian fit, not used")
        brightness.scatter(flux[positive & good], fwhm[positive & good], s=8, color=TEAL,
                           alpha=0.8, label="Gaussian fit, seeing sample")
        brightness.scatter(flux[positive & saturated], fwhm[positive & saturated], s=16,
                           marker="x", color=RED, linewidths=0.8, label="saturated")
        brightness.set_xscale("log")
        if sample.size:
            brightness.set_ylim(0, np.percentile(sample, 99) * 2.0)
        brightness.set_xlabel("segment flux")
        brightness.set_ylabel("FWHM [{}]".format(unit.strip() or "arcsec"))
        _legend(brightness, loc="upper left")
        _grid(brightness)
    else:
        _empty(brightness, "No sources")

    def row(label, key, warn_key=None, fail_key=None, direction="high", spec="{:.3g}",
            unit_text="", scale=1.0):
        value = _finite(info.get(key))
        warn = quality.get(warn_key) if warn_key else None
        fail = quality.get(fail_key) if fail_key else None
        shown = None if value is None else value * scale
        return (label, _fmt(shown, spec, unit_text),
                _check_status(value, warn, fail, direction),
                _limit_text(None if warn is None else warn * scale,
                            None if fail is None else fail * scale, direction, spec))

    rows = [
        row("Sources detected", "source_count", "minimum_sources_warn",
            "minimum_sources_fail", "low", "{:.0f}"),
        ("Seeing sample", _fmt(info.get("seeing_source_count"), "{:.0f}"), None,
         "Gaussian fits to unsaturated stars"),
        row("FWHM", "fwhm_arcsec", "fwhm_warn_arcsec", "fwhm_fail_arcsec", "high",
            "{:.2f}", "″"),
        ("FWHM (pixels)", _fmt(info.get("fwhm_pixels"), "{:.2f}", "px"), None),
        row("FWHM scatter / FWHM", "fwhm_scatter_fraction", "fwhm_scatter_warn_fraction",
            "fwhm_scatter_fail_fraction", "high", "{:.2f}"),
        row("Ellipticity", "ellipticity", "ellipticity_warn", "ellipticity_fail", "high", "{:.3f}"),
        row("Ellipticity scatter", "ellipticity_scatter", "ellipticity_scatter_warn",
            "ellipticity_scatter_fail", "high", "{:.3f}"),
        ("Aligned elongation", "yes" if info.get("globally_elongated") else "no",
         "WARN" if info.get("globally_elongated") else None),
        row("Masked pixels", "masked_pixel_fraction", "maximum_masked_fraction_warn",
            "maximum_masked_fraction_fail", "high", "{:.1f}", "%", 100.0),
        row("Trail pixels", "trail_fraction", "maximum_trail_fraction_warn",
            "maximum_trail_fraction_fail", "high", "{:.2f}", "%", 100.0),
        ("Sky level", _fmt(info.get("background"), "{:.1f}"), None),
        ("Sky RMS", _fmt(info.get("background_rms"), "{:.2f}"), None),
        ("Saturated sources", _fmt(info.get("saturated_source_count"), "{:.0f}"), None),
    ]
    metric_panel(figure.add_subplot(grid[:, 3]), rows, "Image quality", flags=flags)
    return _finish(figure, output_path, show)


# ---------------------------------------------------------------------------
# astrometry
# ---------------------------------------------------------------------------

@_styled
def plot_astrometry_diagnostics(ccd, catalog, matches, info, metadata=None,
                                output_path=None, show=False, settings=None,
                                status=None):
    """Catalog overlay, residual vectors, residual scatter, and WCS checks."""

    metadata = metadata or {}
    info = info or {}
    astrometry = (settings or {}).get("astrometry", {})
    status = status or info.get("quality_status")
    figure, grid = _new_figure(
        "Astrometry  ·  " + _image_label(metadata), _image_subtitle(metadata), status,
        "teal (matched) rings sit on stars everywhere in the field; final residuals "
        "are small, round and show no pattern across the detector.",
        rows=2, columns=4, size=(18, 9.6), width_ratios=[1.35, 1, 1, 0.9],
    )
    data = _data(ccd)
    image_axis = figure.add_subplot(grid[:, 0])
    show_sky(image_axis, data, "Catalog projected with the adopted WCS")
    inside = _bool_column(catalog, "in_image", True)
    cx, cy = _column(catalog, "x"), _column(catalog, "y")
    inlier = _bool_column(matches, "inlier", True)
    mx, my = _column(matches, "x"), _column(matches, "y")
    if cx.size:
        # Unmatched catalog stars: small amber rings. Matched stars: larger,
        # thicker teal rings, so the two are easy to tell apart at a glance.
        image_axis.scatter(cx[inside], cy[inside], s=22, facecolors="none",
                           edgecolors=AMBER, linewidths=0.7, alpha=0.9,
                           label="{} catalog ({})".format(info.get("catalog_name", ""), int(inside.sum())))
    if mx.size:
        image_axis.scatter(mx[inlier], my[inlier], s=70, facecolors="none", edgecolors=TEAL,
                           linewidths=1.5, label="matched ({})".format(int(inlier.sum())))
        if np.any(~inlier):
            image_axis.scatter(mx[~inlier], my[~inlier], s=40, marker="x", color=RED,
                               linewidths=1.2, label="rejected ({})".format(int((~inlier).sum())))
    target = info.get("target_refined") or info.get("target_original") or {}
    mark_target(image_axis, target.get("x"), target.get("y"))
    _image_legend(image_axis)

    final_ra = _column(matches, "residual_ra_final_arcsec")
    final_dec = _column(matches, "residual_dec_final_arcsec")
    original_ra = _column(matches, "residual_ra_original_arcsec")
    original_dec = _column(matches, "residual_dec_original_arcsec")
    if not np.any(np.isfinite(final_ra)):
        final_ra, final_dec = original_ra, original_dec

    vectors = figure.add_subplot(grid[0, 1])
    vectors.set_title("Final residuals on the detector")
    if mx.size and data is not None:
        scale = 0.08 * min(data.shape)
        use = inlier & np.isfinite(final_ra)
        vectors.quiver(mx[use], my[use], -final_ra[use] * scale, final_dec[use] * scale,
                       angles="xy", scale_units="xy", scale=1, color=BLUE, width=0.004)
        if np.any(~inlier):
            vectors.scatter(mx[~inlier], my[~inlier], marker="x", s=12, color=RED, linewidths=0.8)
        vectors.quiver([0.08 * data.shape[1]], [0.06 * data.shape[0]], [0.5 * scale], [0],
                       angles="xy", scale_units="xy", scale=1, color=INK, width=0.006)
        vectors.text(0.08 * data.shape[1], 0.09 * data.shape[0], '0.5″', fontsize=7.5)
        vectors.set_xlim(0, data.shape[1])
        vectors.set_ylim(0, data.shape[0])
        vectors.set_aspect("equal")
        vectors.set_xlabel("x [pixel]")
        vectors.set_ylabel("y [pixel]")
    else:
        _empty(vectors, "No matches")

    scatter = figure.add_subplot(grid[0, 2])
    scatter.set_title("Sky residuals (catalog − detection)")
    if mx.size:
        scatter.scatter(original_ra, original_dec, s=7, color=FAINT, alpha=0.6, label="input WCS")
        scatter.scatter(final_ra[inlier], final_dec[inlier], s=8, color=TEAL, alpha=0.85,
                        label="adopted WCS")
        if np.any(~inlier):
            scatter.scatter(final_ra[~inlier], final_dec[~inlier], s=14, marker="x", color=RED,
                            linewidths=0.8, label="rejected")
        from matplotlib.patches import Circle

        rms = _finite(info.get("refined_rms_arcsec")) or _finite(info.get("original_rms_arcsec"))
        if rms:
            scatter.add_patch(Circle((0, 0), rms, fill=False, edgecolor=INK, linestyle="--",
                                     linewidth=0.9, label="RMS {:.2f}″".format(rms)))
        span = np.nanpercentile(np.abs(np.concatenate([original_ra, original_dec])), 98) if \
            np.any(np.isfinite(original_ra)) else 1.0
        span = max(span * 1.15, 0.5)
        scatter.set_xlim(-span, span)
        scatter.set_ylim(-span, span)
        scatter.set_aspect("equal")
        scatter.axhline(0, color=RULE, linewidth=0.8)
        scatter.axvline(0, color=RULE, linewidth=0.8)
        scatter.set_xlabel("ΔRA cos δ [arcsec]")
        scatter.set_ylabel("ΔDec [arcsec]")
        _legend(scatter, loc="upper right")
    else:
        _empty(scatter, "No matches")

    radial_axis = figure.add_subplot(grid[1, 1])
    radial_axis.set_title("Radial residual distribution")
    original = _column(matches, "separation_original_arcsec")
    final = _column(matches, "separation_final_arcsec")
    if original.size:
        upper = np.nanpercentile(original, 99) * 1.2 if np.any(np.isfinite(original)) else 2.0
        bins = np.linspace(0, max(upper, 0.5), 40)
        radial_axis.hist(original[np.isfinite(original)], bins=bins, histtype="step",
                         color=FAINT, linewidth=1.4, label="input WCS")
        if np.any(np.isfinite(final)):
            radial_axis.hist(final[inlier & np.isfinite(final)], bins=bins, color=TEAL,
                             alpha=0.75, label="adopted WCS (inliers)")
        _threshold_lines(radial_axis, astrometry.get("target_rms_arcsec"),
                         astrometry.get("warning_rms_arcsec"))
        radial_axis.set_xlabel("separation [arcsec]")
        radial_axis.set_ylabel("matches")
        _legend(radial_axis, loc="upper right")
        _grid(radial_axis, "y")
    else:
        _empty(radial_axis, "No matches")

    distortion = figure.add_subplot(grid[1, 2])
    distortion.set_title("Residual versus distance from center")
    if mx.size and data is not None:
        center = np.array(data.shape[::-1]) / 2.0
        radius = np.hypot(mx - center[0], my - center[1])
        radial = final if np.any(np.isfinite(final)) else original
        distortion.scatter(radius[inlier], radial[inlier], s=8, color=TEAL, alpha=0.8)
        if np.any(~inlier):
            distortion.scatter(radius[~inlier], radial[~inlier], s=14, marker="x", color=RED,
                               linewidths=0.8)
        distortion.set_xlabel("distance from image center [pixel]")
        distortion.set_ylabel("separation [arcsec]")
        distortion.set_ylim(0, np.nanpercentile(radial, 99) * 1.3 if np.any(np.isfinite(radial)) else 1)
        _grid(distortion)
    else:
        _empty(distortion, "No matches")

    target_original = info.get("target_original") or {}
    target_refined = info.get("target_refined") or {}
    final_rms = _finite(info.get("refined_rms_arcsec")) or _finite(info.get("original_rms_arcsec"))
    rows = [
        ("Catalog", "{} ({} rows, {} on image)".format(info.get("catalog_name", "—"),
                                                     info.get("catalog_row_count", "—"),
                                                     info.get("catalog_in_image_count", "—")), None),
        ("Matches", _fmt(info.get("match_count"), "{:.0f}"),
         _check_status(info.get("match_count"), astrometry.get("minimum_matches"), None, "low")),
        ("Inliers / rejected", "{} / {}".format(info.get("inlier_count", "—"),
                                                info.get("rejected_match_count", "—")), None),
        ("RMS, input WCS", _fmt(info.get("original_rms_arcsec"), "{:.3f}", "″"), None),
        ("RMS, adopted WCS", _fmt(final_rms, "{:.3f}", "″"),
         _check_status(final_rms, astrometry.get("target_rms_arcsec"),
                       astrometry.get("warning_rms_arcsec")),
         _limit_text(astrometry.get("target_rms_arcsec"), astrometry.get("warning_rms_arcsec"))),
        ("Refinement", "adopted" if info.get("refinement_adopted") else
         str(info.get("refinement_reason") or "not adopted"), None),
        ("Shift", _fmt(info.get("translation_pixels"), "{:.2f}", "px"), None),
        ("Rotation", _fmt(info.get("rotation_degrees"), "{:.4f}", "°"), None),
        ("Scale change", _fmt(None if info.get("scale_change_fraction") is None
                              else 100 * info["scale_change_fraction"], "{:.3f}", "%"), None),
        ("Target round trip", _fmt(target_refined.get("round_trip_error_arcsec",
                                                      target_original.get("round_trip_error_arcsec")),
                                   "{:.3f}", "″"), None),
        ("Plate solve", "used" if (info.get("fallback") or {}).get("succeeded") else "not needed", None),
    ]
    metric_panel(figure.add_subplot(grid[:, 3]), rows, "Astrometry", flags=info.get("flags"))
    return _finish(figure, output_path, show)


# ---------------------------------------------------------------------------
# star selection
# ---------------------------------------------------------------------------

_ROLE_STYLES = {
    "calibration": ("o", TEAL, 30),
    "psf": ("s", AMBER, 46),
    "ensemble": ("D", VIOLET, 26),
    "astrometry": ("o", BLUE, 14),
    "qc_anchor": ("*", PINK, 140),
}


@_styled
def plot_star_selection_diagnostics(ccd, measurements, image_id, summary=None,
                                    metadata=None, output_path=None, show=False,
                                    reference_info=None, status=None):
    """Role assignments on the image, rejection reasons, and catalog photometry."""

    metadata = metadata or {}
    summary = summary or {}
    rows_table = None
    if measurements is not None and len(measurements):
        rows_table = measurements[_str_column(measurements, "image_id") == str(image_id)]
    count = 0 if rows_table is None else len(rows_table)
    flags = list(summary.get("flags") or [])
    status = status or ("WARN" if flags else "PASS")
    figure, grid = _new_figure(
        "Star selection  ·  " + _image_label(metadata, image_id), _image_subtitle(metadata),
        status,
        "calibration and PSF stars are isolated, unsaturated and spread over the field; "
        "catalog vs instrumental magnitudes follow a tight line of slope 1.",
        rows=2, columns=3, width_ratios=[1.35, 1.0, 0.9],
    )
    data = _data(ccd)
    image_axis = figure.add_subplot(grid[:, 0])
    show_sky(image_axis, data, "Roles assigned in this image")
    if count:
        x, y = _column(rows_table, "x"), _column(rows_table, "y")
        accepted = _bool_column(rows_table, "image_accepted")
        image_axis.scatter(x[~accepted], y[~accepted], s=10, marker="x", color=RED,
                           linewidths=0.6, alpha=0.8, label="rejected ({})".format(int((~accepted).sum())))
        for role, (marker, color, size) in _ROLE_STYLES.items():
            selected = _bool_column(rows_table, "role_" + role)
            if not np.any(selected):
                continue
            image_axis.scatter(x[selected], y[selected], s=size, marker=marker,
                               facecolors="none" if role != "qc_anchor" else color,
                               edgecolors=color, linewidths=0.9,
                               label="{} ({})".format(role.replace("_", " "), int(selected.sum())))
    _image_legend(image_axis)

    reasons_axis = figure.add_subplot(grid[0, 1])
    reasons_axis.set_title("Why candidates were rejected")
    counts = {}
    for value in _str_column(rows_table, "rejection_reasons") if count else []:
        for reason in str(value).split(";"):
            if reason and reason not in ("--", "None"):
                counts[reason] = counts.get(reason, 0) + 1
    if counts:
        ordered = sorted(counts, key=counts.get)[-12:]
        positions = np.arange(len(ordered))
        reasons_axis.barh(positions, [counts[name] for name in ordered], color=RED, alpha=0.75,
                          height=0.6)
        reasons_axis.set_yticks(positions)
        reasons_axis.set_yticklabels([name.lower().replace("_", " ") for name in ordered], fontsize=7)
        reasons_axis.set_xlabel("candidates")
        _grid(reasons_axis, "x")
    else:
        _empty(reasons_axis, "No rejected candidates")

    photometry = figure.add_subplot(grid[1, 1])
    photometry.set_title("Catalog vs instrumental magnitude")
    if count:
        # Fixed-aperture fluxes: detection-footprint fluxes lose a growing
        # fraction of the light for fainter stars and bend this relation.
        flux = _column(rows_table, "aperture_flux")
        flux_label = "aperture flux"
        if not np.any(np.isfinite(flux)):
            flux, flux_label = _column(rows_table, "flux"), "detection flux"
        magnitude = _column(rows_table, "magnitude")
        with np.errstate(all="ignore"):
            instrumental = -2.5 * np.log10(flux)
        accepted = _bool_column(rows_table, "image_accepted")
        calibration = _bool_column(rows_table, "role_calibration")
        ok = np.isfinite(instrumental) & np.isfinite(magnitude)
        photometry.scatter(magnitude[ok & ~accepted], instrumental[ok & ~accepted], s=7,
                           color=FAINT, alpha=0.6, label="rejected")
        photometry.scatter(magnitude[ok & accepted], instrumental[ok & accepted], s=9,
                           color=BLUE, alpha=0.7, label="accepted")
        photometry.scatter(magnitude[ok & calibration], instrumental[ok & calibration], s=12,
                           color=TEAL, label="calibration role")
        use = ok & calibration
        if np.count_nonzero(use) > 3:
            offset = np.median(instrumental[use] - magnitude[use])
            span = np.array([np.nanmin(magnitude[ok]), np.nanmax(magnitude[ok])])
            photometry.plot(span, span + offset, color=INK, linewidth=0.9,
                            label="slope 1, median offset")
        bands = list(_str_column(rows_table, "magnitude_band")[ok])
        band = max(set(bands), key=bands.count) if bands else ""
        photometry.set_xlabel("catalog magnitude ({})".format(band))
        photometry.set_ylabel("−2.5 log₁₀({})".format(flux_label))
        photometry.invert_yaxis()
        _legend(photometry, loc="upper right")
        _grid(photometry)
    else:
        _empty(photometry, "No candidates")

    role_counts = summary.get("role_counts") or {}
    metric_rows = [
        ("Candidates", _fmt(summary.get("candidate_count", count), "{:.0f}"), None),
        ("Accepted", _fmt(summary.get("strictly_accepted_count"), "{:.0f}"), None),
        ("Rejected", _fmt(summary.get("rejected_count"), "{:.0f}"), None),
        (None, None, None),
    ]
    for role in ("calibration", "psf", "ensemble", "astrometry", "qc_anchor"):
        metric_rows.append(("  {} stars".format(role.replace("_", " ")),
                            _fmt(role_counts.get(role), "{:.0f}"), None))
    for name, entry in ((reference_info or {}).get("catalogs") or {}).items():
        metric_rows.append(("{} reference".format(name.upper()),
                            entry.get("error") or "{} matched ({})".format(
                                entry.get("matched", 0), entry.get("loaded_from", "")),
                            "FAIL" if entry.get("error") else ("PASS" if entry.get("matched") else "WARN")))
    metric_rows.append(("Image rejected", "yes" if summary.get("image_rejected") else "no",
                        "FAIL" if summary.get("image_rejected") else None))
    metric_panel(figure.add_subplot(grid[:, 2]), metric_rows, "Star selection", flags=flags)
    return _finish(figure, output_path, show)


# ---------------------------------------------------------------------------
# usability
# ---------------------------------------------------------------------------

@_styled
def plot_image_usability_diagnostics(ccd, decision, star_residuals=None, metadata=None,
                                     output_path=None, show=False, status=None):
    """Quick zeropoint, transparency, depth, and the effective review decision."""

    metadata = metadata or {}
    decision = decision or {}
    status = status or decision.get("status")
    note = "automatic {}  ·  {}".format(decision.get("automatic_status", "—"),
                                       str(decision.get("review_state") or "").lower() or "not reviewed")
    figure, grid = _new_figure(
        "Usability  ·  " + _image_label(metadata, decision.get("filename")),
        _image_subtitle(metadata), status,
        "the quick zeropoint is consistent with the other epochs, residuals show no "
        "cloud pattern, and the depth reaches the target.",
        rows=2, columns=3, width_ratios=[1.2, 1.0, 1.0], size=(17, 9.6),
        status_note=note,
    )
    data = _data(ccd)
    image_axis = figure.add_subplot(grid[0, 0])
    show_sky(image_axis, data, "Target and artifacts", labels=False)
    artifacts = decision.get("target_artifacts") or {}
    position = artifacts.get("position")
    if position is not None:
        mark_target(image_axis, position[0], position[1],
                    radius=max(10, artifacts.get("radius_pixels") or 10),
                    color=RED if artifacts.get("any") else TEAL)
        if artifacts.get("any"):
            image_axis.text(0.01, 0.99, "target overlaps: {}".format(
                ", ".join(str(name) for name, hit in (artifacts.get("overlaps") or {}).items() if hit)),
                transform=image_axis.transAxes, ha="left", va="top", color="white", fontsize=7.5)

    rows = star_residuals
    if rows is not None and len(rows) and "image_id" in rows.colnames:
        rows = rows[_str_column(rows, "image_id") == str(decision.get("image_id"))]
    spatial = figure.add_subplot(grid[0, 1])
    spatial.set_title("Zeropoint residuals across the field")
    if rows is not None and len(rows):
        residual = _column(rows, "spatial_residual_mag")
        inlier = _bool_column(rows, "inlier", True)
        use = inlier & np.isfinite(residual)
        limit = max(0.1, float(np.nanpercentile(np.abs(residual[use]), 95))) if use.any() else 0.2
        points = spatial.scatter(_column(rows, "x")[use], _column(rows, "y")[use], c=residual[use],
                                 cmap=DIVERGING_CMAP, vmin=-limit, vmax=limit, s=26,
                                 edgecolors=INK, linewidths=0.2)
        _colorbar(figure, points, spatial, "residual [mag]")
        if np.any(~inlier):
            spatial.scatter(_column(rows, "x")[~inlier], _column(rows, "y")[~inlier], marker="x",
                            s=14, color=FAINT, linewidths=0.7, label="clipped")
        if data is not None:
            spatial.set_xlim(0, data.shape[1])
            spatial.set_ylim(0, data.shape[0])
        spatial.set_aspect("equal")
        spatial.set_xlabel("x [pixel]")
        spatial.set_ylabel("y [pixel]")
        _legend(spatial, loc="upper right")
    else:
        _empty(spatial, "No stellar residuals")

    zeropoints = figure.add_subplot(grid[1, 0])
    zeropoints.set_title("Per-star quick zeropoint")
    if rows is not None and len(rows):
        magnitude = _column(rows, "catalog_magnitude")
        zeropoint = _column(rows, "quick_zeropoint_mag")
        inlier = _bool_column(rows, "inlier", True)
        zeropoints.scatter(magnitude[inlier], zeropoint[inlier], s=10, color=TEAL, label="used")
        zeropoints.scatter(magnitude[~inlier], zeropoint[~inlier], s=12, marker="x", color=FAINT,
                           linewidths=0.7, label="clipped")
        zp = _finite(decision.get("zeropoint_mag"))
        scatter = _finite(decision.get("zeropoint_scatter_mag"))
        if zp is not None:
            zeropoints.axhline(zp, color=INK, linewidth=1.0, label="ZP {:.3f}".format(zp))
            if scatter:
                zeropoints.axhspan(zp - scatter, zp + scatter, color=TEAL, alpha=0.12, linewidth=0)
            zeropoints.set_ylim(zp - 0.8, zp + 0.8)
        reference = _finite(decision.get("transparency_reference_mag"))
        if reference is not None:
            zeropoints.axhline(reference, color=AMBER, linestyle="--", linewidth=1.0,
                               label="best epoch {:.3f}".format(reference))
        zeropoints.set_xlabel("catalog magnitude")
        zeropoints.set_ylabel("zeropoint [mag]")
        _legend(zeropoints, loc="lower left", ncol=2)
        _grid(zeropoints)
    else:
        _empty(zeropoints, "No zeropoint stars")

    depth = figure.add_subplot(grid[1, 1])
    depth.set_title("Quick limiting depth")
    labels, values, colors = [], [], []
    for region, color in (("global", BLUE), ("local", VIOLET)):
        for sigma, value in (decision.get("{}_depths_mag".format(region)) or {}).items():
            if _finite(value) is not None:
                labels.append("{} {}".format(region, sigma))
                values.append(float(value))
                colors.append(color)
    if values:
        positions = np.arange(len(values))
        depth.bar(positions, values, color=colors, width=0.6)
        for position, value in zip(positions, values):
            depth.annotate("{:.2f}".format(value), (position, value), xytext=(0, 3),
                           textcoords="offset points", ha="center", va="bottom",
                           fontsize=7.5, color=INK)
        expected = _finite(decision.get("expected_target_magnitude"))
        if expected is not None:
            depth.axhline(expected, color=RED, linestyle="--", label="expected target")
        depth.set_xticks(positions)
        depth.set_xticklabels(labels)
        depth.set_ylim(max(values) + 1.0, min(values) - 1.5)
        depth.set_ylabel("limiting magnitude")
        _grid(depth, "y")
    else:
        _empty(depth, "Depth unavailable")

    check_rows = []
    for check in decision.get("checks") or []:
        check_rows.append((str(check.get("metric", "")).replace("_", " "),
                           _fmt(check.get("value"), "{:.3g}"), check.get("status"),
                           "limit {}".format(_fmt(check.get("threshold"), "{:.3g}"))))
    metric_rows = [
        ("Quick zeropoint", _fmt(decision.get("zeropoint_mag"), "{:.3f}", "mag"), None),
        ("Zeropoint scatter", _fmt(decision.get("zeropoint_scatter_mag"), "{:.3f}", "mag"), None),
        ("Transparency loss", _fmt(decision.get("transparency_attenuation_mag"), "{:.3f}", "mag"), None),
        ("Cloud amplitude", _fmt(decision.get("spatial_cloud_amplitude_mag"), "{:.3f}", "mag"), None),
        ("Calibration stars", "{} used, {} rejected".format(decision.get("calibration_star_count", "—"),
                                                            decision.get("calibration_rejected_star_count", "—")), None),
        ("Catalog recovery", _fmt(None if decision.get("catalog_recovery_fraction") is None
                                  else 100 * decision["catalog_recovery_fraction"], "{:.0f}", "%"), None),
        ("Seeing", _fmt(decision.get("fwhm_arcsec"), "{:.2f}", "″"), None),
        ("Use image", "yes" if decision.get("use_image") else "no",
         "PASS" if decision.get("use_image") else "FAIL"),
    ] + check_rows
    reasons = list(decision.get("reasons") or [])
    panel = figure.add_subplot(grid[:, 2])
    metric_panel(panel, metric_rows, "Decision inputs",
                 flags=list(decision.get("quality_flags") or []))
    if reasons:
        panel.text(0.0, -0.02, "\n".join("• " + str(reason) for reason in reasons[:6]),
                   transform=panel.transAxes, ha="left", va="top", fontsize=7.5, color=MUTED)
    return _finish(figure, output_path, show)


# ---------------------------------------------------------------------------
# alignment
# ---------------------------------------------------------------------------

@_styled
def plot_alignment_target_diagnostics(stacks, target_solution, target_candidates=None,
                                      projection_table=None, output_path=None, show=False,
                                      status=None):
    """Detection stacks at the frozen target position and alignment quality."""

    from astropy import units as u
    from astropy.coordinates import SkyCoord

    target_solution = target_solution or {}
    stacks = stacks or {}
    status = status or target_solution.get("status")
    items = sorted(stacks.items(), key=lambda item: (item[0] != "multifilter", item[0]))[:4]
    figure, grid = _new_figure(
        "Alignment and target position", "frozen at RA {:.6f}, Dec {:+.6f}  ·  ±{:.3f}″".format(
            target_solution.get("ra_deg", np.nan), target_solution.get("dec_deg", np.nan),
            target_solution.get("uncertainty_arcsec", np.nan)), status,
        "the cross sits on the transient in every stack; all epochs align to the "
        "reference with small common-star RMS.",
        rows=2, columns=max(4, len(items)), height_ratios=[1, 1.05], size=(18, 9.6),
    )
    final = None
    if _finite(target_solution.get("ra_deg")) is not None:
        final = SkyCoord(target_solution["ra_deg"], target_solution["dec_deg"], unit="deg")
    prior = None
    if _finite(target_solution.get("prior_ra_deg")) is not None:
        prior = SkyCoord(target_solution["prior_ra_deg"], target_solution["prior_dec_deg"], unit="deg")
    for index in range(max(4, len(items))):
        axis = figure.add_subplot(grid[0, index])
        if index >= len(items):
            axis.set_axis_off()
            continue
        name, product = items[index]
        data = _data(product.get("data"))
        wcs = product.get("wcs")
        if data is None or wcs is None or final is None:
            _empty(axis, "Stack not available", name)
            continue
        x, y = wcs.world_to_pixel(final)
        half = 30
        x0, x1 = int(max(0, x - half)), int(min(data.shape[1], x + half))
        y0, y1 = int(max(0, y - half)), int(min(data.shape[0], y + half))
        extent = (x0 - 0.5, x1 - 0.5, y0 - 0.5, y1 - 0.5)
        show_sky(axis, data[y0:y1, x0:x1], "{} stack ({} images)".format(
            name, len(product.get("contributors") or [])), extent=extent, labels=False)
        axis.scatter([x], [y], marker="+", s=140, color=TEAL, linewidths=1.4, label="frozen")
        if prior is not None:
            px, py = wcs.world_to_pixel(prior)
            axis.scatter([px], [py], marker="x", s=60, color=AMBER, linewidths=1.0, label="prior")
        if index == 0:
            _image_legend(axis)

    offsets = figure.add_subplot(grid[1, 0])
    offsets.set_title("Per-image centroids around the frozen position")
    if target_candidates is not None and len(target_candidates) and final is not None:
        ra, dec = _column(target_candidates, "ra_deg"), _column(target_candidates, "dec_deg")
        good = np.isfinite(ra) & np.isfinite(dec)
        coordinates = SkyCoord(ra[good] * u.deg, dec[good] * u.deg)
        dx, dy = final.spherical_offsets_to(coordinates)
        used = _bool_column(target_candidates, "used_in_solution")[good]
        accepted = _bool_column(target_candidates, "accepted")[good]
        names = _str_column(target_candidates, "source")[good]
        offsets.scatter(dx.arcsec[~accepted], dy.arcsec[~accepted], marker="x", color=RED, s=30,
                        label="rejected")
        offsets.scatter(dx.arcsec[accepted & ~used], dy.arcsec[accepted & ~used], s=26,
                        facecolors="none", edgecolors=BLUE, label="accepted")
        offsets.scatter(dx.arcsec[used], dy.arcsec[used], s=26, color=TEAL, label="used")
        for xi, yi, label in zip(dx.arcsec, dy.arcsec, names):
            offsets.annotate(str(label).split(":")[-1][:10], (xi, yi), xytext=(3, 3),
                             textcoords="offset points", fontsize=6, color=MUTED)
        span = max(0.5, float(np.nanmax(np.abs(np.concatenate([dx.arcsec, dy.arcsec])))) * 1.25)
        offsets.set_xlim(-span, span)
        offsets.set_ylim(-span, span)
        offsets.set_aspect("equal")
        offsets.axhline(0, color=RULE)
        offsets.axvline(0, color=RULE)
        offsets.set_xlabel("ΔRA cos δ [arcsec]")
        offsets.set_ylabel("ΔDec [arcsec]")
        _legend(offsets, loc="upper right")
    else:
        _empty(offsets, "No centroid candidates")

    projection = figure.add_subplot(grid[1, 1:3])
    projection.set_title("Relative alignment to the reference image")
    if projection_table is not None and len(projection_table):
        labels = [str(value).split(".")[0] for value in projection_table["image_id"]]
        values = _column(projection_table, "relative_alignment_rms_arcsec")
        colors = [_status_color(value) for value in _str_column(projection_table, "status")]
        positions = np.arange(len(labels))
        projection.bar(positions, np.nan_to_num(values), color=colors, width=0.6)
        projection.set_xticks(positions)
        projection.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
        projection.set_ylabel("common-star RMS [arcsec]")
        _grid(projection, "y")
    else:
        _empty(projection, "No alignment checks")

    rows = [
        ("Status", str(target_solution.get("status", "—")), target_solution.get("status")),
        ("Uncertainty", _fmt(target_solution.get("uncertainty_arcsec"), "{:.3f}", "″"), None),
        ("Prior source", str(target_solution.get("prior_source", "—")), None),
        ("Candidates used", "{} / {}".format(target_solution.get("used_candidate_count", "—"),
                                             target_solution.get("candidate_count", "—")), None),
        ("Frozen", "yes" if target_solution.get("frozen") else "no", None),
    ]
    if prior is not None and final is not None:
        rows.insert(2, ("Shift from prior", _fmt(final.separation(prior).arcsec, "{:.3f}", "″"), None))
    metric_panel(figure.add_subplot(grid[1, 3:]), rows, "Target solution",
                 flags=target_solution.get("flags"))
    return _finish(figure, output_path, show)


# ---------------------------------------------------------------------------
# PSF
# ---------------------------------------------------------------------------

def _radial_profile(image, center=None):
    image = np.asarray(image, dtype=float)
    cy, cx = ((np.array(image.shape) - 1) / 2.0) if center is None else center
    yy, xx = np.indices(image.shape, dtype=float)
    radius = np.hypot(xx - cx, yy - cy)
    return radius.ravel(), image.ravel()


@_styled
def plot_psf_diagnostics(result, output_path=None, show=False, metadata=None, status=None):
    """PSF stars, the normalized model, its profile, and the fit residuals."""

    result = result or {}
    metadata = metadata or {"filename": result.get("filename")}
    status = status or result.get("status")
    note = "automatic {}  ·  {}".format(result.get("automatic_status", "—"),
                                       str(result.get("review_state") or "").lower())
    figure, grid = _new_figure(
        "PSF  ·  " + _image_label(metadata, result.get("filename")), _image_subtitle(metadata),
        status,
        "PSF stars look alike and isolated; the model is smooth and round; residuals "
        "are small and show no repeated pattern (ring, dipole).",
        rows=2, columns=3, width_ratios=[1.1, 1.0, 0.9], size=(16, 9.6), status_note=note,
    )
    cutouts = result.get("cutouts")
    mosaic = _mosaic(cutouts) if cutouts is not None and np.size(cutouts) else None
    axis = figure.add_subplot(grid[0, 0])
    if mosaic is not None:
        show_map(figure, axis, mosaic, "PSF stars used ({}), each normalized".format(len(cutouts)),
                 cmap=MAP_CMAP, limits=(0, 1), label="relative", labels=False)
        axis.set_xticks([])
        axis.set_yticks([])
    else:
        _empty(axis, "No PSF stars", "PSF stars")

    model = result.get("model_native")
    model_axis = figure.add_subplot(grid[0, 1])
    show_map(figure, model_axis, model, "Model ({}), log stretch".format(result.get("model_type") or "—"),
             log=True, label="normalized")

    residual_axis = figure.add_subplot(grid[1, 0])
    residuals = result.get("residuals")
    residual_mosaic = None
    if residuals is not None and np.size(residuals) and cutouts is not None and np.size(cutouts):
        peaks = np.nanmax(np.abs(np.asarray(cutouts, dtype=float)), axis=(1, 2))
        peaks[~np.isfinite(peaks) | (peaks == 0)] = 1.0
        scaled = np.asarray(residuals, dtype=float) / peaks[:, None, None]
        residual_mosaic = _mosaic(scaled, normalize=False)
    if residual_mosaic is not None:
        show_map(figure, residual_axis, residual_mosaic, "Residuals (star − model) ÷ star peak",
                 symmetric=True, limit=0.1, label="fraction of peak", labels=False)
        residual_axis.set_xticks([])
        residual_axis.set_yticks([])
    else:
        _empty(residual_axis, "No residuals", "Residuals")

    profile = figure.add_subplot(grid[1, 1])
    profile.set_title("Radial profile")
    if model is not None:
        model_array = np.asarray(model, dtype=float)
        peak = np.nanmax(model_array) or 1.0
        if cutouts is not None and np.size(cutouts):
            for cutout in np.asarray(cutouts, dtype=float)[:40]:
                radius, value = _radial_profile(cutout)
                star_peak = np.nanmax(cutout) or 1.0
                profile.scatter(radius, value / star_peak, s=2, color=FAINT, alpha=0.25)
        radius, value = _radial_profile(model_array)
        order = np.argsort(radius)
        bins = np.arange(0, radius.max() + 1.0, 0.5)
        index = np.digitize(radius[order], bins)
        centers, means = [], []
        for i in range(1, len(bins)):
            selected = value[order][index == i]
            if selected.size:
                centers.append(float(np.mean(radius[order][index == i])))
                means.append(float(np.nanmean(selected)) / peak)
        profile.plot(centers, means, color=BLUE, linewidth=2.0, marker="o", markersize=3,
                     label="model")
        fwhm = _finite(result.get("fwhm_pixels"))
        if fwhm:
            profile.axvline(fwhm / 2.0, color=AMBER, linestyle="--", linewidth=1.0,
                            label="HWHM {:.2f} px (model fit)".format(fwhm / 2.0))
            profile.axhline(0.5, color=RULE, linewidth=0.8)
        star_fwhm = _finite(result.get("star_fwhm_pixels"))
        if star_fwhm:
            profile.axvline(star_fwhm / 2.0, color=FAINT, linestyle=":", linewidth=1.0,
                            label="star segment HWHM {:.2f} px".format(star_fwhm / 2.0))
        profile.set_xlim(0, radius.max())
        profile.set_ylim(-0.1, 1.1)
        profile.set_xlabel("radius [pixel]")
        profile.set_ylabel("value ÷ peak")
        _legend(profile, loc="upper right")
        _grid(profile)
    else:
        _empty(profile, "No model")

    pixel_scale = _finite(metadata.get("pixel_scale"))
    fwhm = _finite(result.get("fwhm_pixels"))
    settings = result.get("settings_used") or {}
    rows = [
        ("Model", str(result.get("model_type") or "failed"), None),
        ("Stars used", "{} of {}".format(result.get("star_count_used", "—"),
                                         result.get("star_count_considered", "—")),
         _check_status(result.get("star_count_used"), settings.get("minimum_stars"), None, "low")),
        ("FWHM", "{}{}".format(_fmt(fwhm, "{:.2f}", "px"),
                               "" if not (fwhm and pixel_scale) else "  ({:.2f}″)".format(fwhm * pixel_scale)),
         None),
        ("Star segment FWHM", _fmt(result.get("star_fwhm_pixels"), "{:.2f}", "px"), None,
         "moment width of PSF stars (reference)"),
        ("Ellipticity", _fmt(result.get("ellipticity"), "{:.3f}"), None),
        ("Median residual", _fmt(None if result.get("residual_median_fraction") is None
                                 else 100 * result["residual_median_fraction"], "{:.1f}", "%"),
         _check_status(result.get("residual_median_fraction"), settings.get("residual_warn_fraction"),
                       settings.get("residual_fail_fraction")),
         _limit_text(None if settings.get("residual_warn_fraction") is None else 100 * settings["residual_warn_fraction"],
                     None if settings.get("residual_fail_fraction") is None else 100 * settings["residual_fail_fraction"],
                     spec="{:.0f}%")),
        ("Max residual", _fmt(None if result.get("residual_maximum_fraction") is None
                              else 100 * result["residual_maximum_fraction"], "{:.1f}", "%"), None),
        ("Median correlation", _fmt(result.get("correlation_median"), "{:.4f}"),
         _check_status(result.get("correlation_median"), settings.get("minimum_correlation"), None, "low")),
        ("Spatial variation", "modeled" if result.get("spatial_support") else "constant", None),
        ("Approved for photometry", "yes" if result.get("approved_for_photometry") else "no",
         "PASS" if result.get("approved_for_photometry") else "FAIL"),
    ]
    if result.get("review_note"):
        rows.append(("Review note", str(result["review_note"])[:40], None))
    metric_panel(figure.add_subplot(grid[:, 2]), rows, "PSF",
                 flags=list(result.get("quality_flags") or []) + list(result.get("reasons") or []))
    return _finish(figure, output_path, show)


# ---------------------------------------------------------------------------
# science photometry
# ---------------------------------------------------------------------------

def _draw_apertures(axis, diagnostics, origin):
    from matplotlib.patches import Circle

    center = (diagnostics.get("fixed_x", 0.0) - origin[0],
              diagnostics.get("fixed_y", 0.0) - origin[1])
    for key, color, style, label in (
        ("small_radius_pixels", TEAL, "-", "small aperture"),
        ("large_radius_pixels", BLUE, "-", "large aperture"),
        ("sky_inner_radius_pixels", AMBER, "--", "sky annulus"),
        ("sky_outer_radius_pixels", AMBER, "--", None),
    ):
        radius = _finite(diagnostics.get(key))
        if radius:
            axis.add_patch(Circle(center, radius, fill=False, edgecolor=color, linewidth=1.1,
                                  linestyle=style, label=label))
    return center


@_styled
def plot_science_photometry_diagnostics(result, output_path=None, show=False,
                                        metadata=None, status=None):
    """Forced photometry at the frozen position: apertures, model, residual, fluxes."""

    result = result or {}
    metadata = metadata or {"filename": result.get("filename")}
    diagnostics = result.get("target_diagnostics") or {}
    table = result.get("measurements")
    flags = list(result.get("target_flags") or [])
    status = status or result.get("status")
    figure, grid = _new_figure(
        "Science photometry  ·  " + _image_label(metadata, result.get("filename")),
        _image_subtitle(metadata), status,
        "apertures are centered on the transient and the sky annulus avoids other "
        "sources; the PSF residual is consistent with noise.",
        rows=2, columns=4, size=(18, 9.0), width_ratios=[1.15, 1, 1, 0.95],
    )
    context = diagnostics.get("context_data")
    axis = figure.add_subplot(grid[:, 0])
    if context is not None:
        origin = diagnostics.get("context_origin", (0, 0))
        height, width = np.shape(context)
        extent = (-0.5, width - 0.5, -0.5, height - 0.5)
        show_sky(axis, context, "Target, apertures and sky annulus", extent=extent, labels=False)
        mask = diagnostics.get("context_mask")
        if mask is not None:
            overlay_mask(axis, np.asarray(mask, dtype=bool), RED, 0.35)
        _draw_apertures(axis, diagnostics, origin)
        _image_legend(axis)
    else:
        _empty(axis, "No target cutout")

    data = diagnostics.get("data")
    model = diagnostics.get("model")
    residual = diagnostics.get("residual")
    limits = zscale_limits(data) if data is not None else None
    show_sky(figure.add_subplot(grid[0, 1]), data, "Data at target", limits=limits, labels=False)
    model_axis = figure.add_subplot(grid[0, 2])
    if model is not None:
        show_sky(model_axis, model, "Forced PSF model (same stretch)", limits=limits, labels=False)
    else:
        _empty(model_axis, "No model", "Forced PSF model")
    residual_axis = figure.add_subplot(grid[1, 1])
    noise = None
    if table is not None and len(table):
        target_rows = table[_str_column(table, "source_type") == "target"]
        noise = _robust(_column(target_rows, "local_background_rms"))[0]
    show_map(figure, residual_axis, residual, "Data − model", symmetric=True,
             limit=None if noise is None else 4 * noise, label="residual", labels=False)

    flux_axis = figure.add_subplot(grid[1, 2])
    flux_axis.set_title("Target flux by method")
    target_rows = None
    if table is not None and len(table):
        target_rows = table[_str_column(table, "source_type") == "target"]
    if target_rows is not None and len(target_rows):
        methods = list(_str_column(target_rows, "method"))
        flux = _column(target_rows, "flux")
        error = _column(target_rows, "flux_uncertainty")
        snr = _column(target_rows, "snr")
        positions = np.arange(len(methods))
        flux_axis.errorbar(positions, flux, yerr=error, fmt="o", color=BLUE, capsize=4,
                           markersize=6)
        for position, value, ratio in zip(positions, flux, snr):
            if np.isfinite(value):
                flux_axis.annotate("S/N {:.1f}".format(ratio) if np.isfinite(ratio) else "S/N —",
                                   (position, value), xytext=(8, 0), textcoords="offset points",
                                   va="center", fontsize=7.5, color=MUTED)
        flux_axis.axhline(0, color=RULE)
        flux_axis.set_xticks(positions)
        flux_axis.set_xticklabels([name.replace("_", " ") for name in methods])
        flux_axis.set_xlim(-0.5, len(methods) - 0.2)
        flux_axis.set_ylabel("flux [{}]".format(result.get("flux_unit", "adu")))
        _grid(flux_axis, "y")
    else:
        _empty(flux_axis, "No target measurements")

    free = diagnostics.get("free_centroid") or {}
    rows = [
        ("Frozen position", "{:.6f}, {:+.6f}".format(result.get("fixed_target_ra_deg", np.nan),
                                                    result.get("fixed_target_dec_deg", np.nan)), None),
        ("Coordinate version", str(result.get("target_coordinate_version", "—")), None),
        ("PSF FWHM", _fmt(result.get("fwhm_pixels"), "{:.2f}", "px"), None),
        ("Free-centroid offset", _fmt(free.get("offset_arcsec") if free.get("offset_arcsec") is not None
                                      else free.get("offset_pixels"), "{:.2f}",
                                      "″" if free.get("offset_arcsec") is not None else "px"),
         _check_status(free.get("offset_arcsec"), 0.5, 1.5), "diagnostic only"),
        ("Sources measured", _fmt(result.get("source_count"), "{:.0f}"), None),
        ("Uncertainty from", str(result.get("uncertainty_source", "—")), None),
    ]
    if target_rows is not None:
        for method, snr in zip(_str_column(target_rows, "method"), _column(target_rows, "snr")):
            rows.append(("S/N, {}".format(method.replace("_", " ")), _fmt(snr, "{:.1f}"),
                         _check_status(snr, 5.0, 3.0, "low")))
    metric_panel(figure.add_subplot(grid[:, 3]), rows, "Forced photometry", flags=flags)
    return _finish(figure, output_path, show)


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------

def _preferred_method(table, preferred=("psf", "large_aperture", "small_aperture")):
    methods = list(dict.fromkeys(_str_column(table, "method")))
    for method in preferred:
        if method in methods:
            return method
    return methods[0] if methods else None


@_styled
def plot_calibration_diagnostics(products, output_path=None, show=False, status=None):
    """Run-level photometric calibration: zeropoints over time and residual trends."""

    products = products or {}
    zeropoints = products.get("zeropoints")
    stars = products.get("calibration_stars")
    limits = products.get("limits")
    status = status or products.get("status")
    figure, grid = _new_figure(
        "Calibration  ·  run summary", "catalogs: {}".format(
            ", ".join(products.get("catalogs_available") or []) or "none"), status,
        "zeropoints per filter are stable apart from cloudy epochs; residuals are flat "
        "against magnitude, color and position.",
        rows=2, columns=4, size=(18, 9.6), width_ratios=[1, 1, 1, 0.85],
    )
    axis = figure.add_subplot(grid[0, 0:2])
    axis.set_title("Zeropoint per image (bars: scatter of calibration stars)")
    if zeropoints is not None and len(zeropoints):
        image_ids = list(dict.fromkeys(_str_column(zeropoints, "image_id")))
        positions = {image: index for index, image in enumerate(image_ids)}
        offsets = {"small_aperture": -0.18, "psf": 0.0, "large_aperture": 0.18}
        for method in dict.fromkeys(_str_column(zeropoints, "method")):
            selected = zeropoints[_str_column(zeropoints, "method") == method]
            x = np.array([positions[value] for value in _str_column(selected, "image_id")], dtype=float)
            x += offsets.get(method, 0.0)
            colors = [_filter_color(value) for value in _str_column(selected, "filter")]
            axis.errorbar(x, _column(selected, "zeropoint_mag"),
                          yerr=_column(selected, "zeropoint_scatter_mag"), fmt="none",
                          ecolor=FAINT, elinewidth=0.8, capsize=0)
            axis.scatter(x, _column(selected, "zeropoint_mag"), c=colors,
                         marker=METHOD_MARKERS.get(method, "o"), s=26,
                         label=method.replace("_", " "))
        axis.set_xticks(range(len(image_ids)))
        axis.set_xticklabels([value.split(".")[0] for value in image_ids], rotation=30,
                             ha="right", fontsize=7)
        axis.set_ylabel("zeropoint [mag]")
        for band in dict.fromkeys(_str_column(zeropoints, "filter")):
            axis.plot([], [], "s", color=_filter_color(band), label="{} band".format(band))
        _legend(axis, loc="best", ncol=3)
        _grid(axis, "y")
    else:
        _empty(axis, "No zeropoints")

    inliers = None
    if stars is not None and len(stars):
        inliers = stars[_bool_column(stars, "inlier")]
        method = _preferred_method(inliers)
        if method is not None:
            inliers = inliers[_str_column(inliers, "method") == method]
    for (column, label), cell in zip(
        (("catalog_magnitude", "catalog magnitude"), ("catalog_color", "catalog color (g − r)"),
         ("x", "x [pixel]"), ("y", "y [pixel]")),
        (grid[0, 2], grid[1, 0], grid[1, 1], grid[1, 2]),
    ):
        axis = figure.add_subplot(cell)
        axis.set_title("Residual vs {}".format(label.split(" [")[0]))
        if inliers is not None and len(inliers) and column in inliers.colnames:
            residual = _column(inliers, "calibrated_residual")
            values = _column(inliers, column)
            colors = [_filter_color(value) for value in _str_column(inliers, "filter")]
            axis.scatter(values, residual, c=colors, s=6, alpha=0.6)
            ok = np.isfinite(values) & np.isfinite(residual)
            if np.count_nonzero(ok) > 10:
                edges = np.unique(np.percentile(values[ok], np.linspace(0, 100, 9)))
                index = np.digitize(values[ok], edges[1:-1])
                centers = [np.median(values[ok][index == i]) for i in range(len(edges) - 1)
                           if np.any(index == i)]
                medians = [np.median(residual[ok][index == i]) for i in range(len(edges) - 1)
                           if np.any(index == i)]
                axis.plot(centers, medians, color=INK, linewidth=1.6, marker="o", markersize=3,
                          label="binned median")
            axis.axhline(0, color=RULE)
            spread = _robust(residual)[1] or 0.05
            axis.set_ylim(-max(0.15, 6 * spread), max(0.15, 6 * spread))
            axis.set_xlabel(label)
            axis.set_ylabel("calibrated − catalog [mag]")
            _grid(axis)
        else:
            _empty(axis, "Not available")

    status_counts = {}
    for value in _str_column(zeropoints, "status") if zeropoints is not None else []:
        status_counts[value] = status_counts.get(value, 0) + 1
    rows = [
        ("Overall", str(products.get("status", "—")), products.get("status")),
        ("Zeropoints", ", ".join("{} {}".format(v, k) for k, v in sorted(status_counts.items())) or "none", None),
        ("Catalogs used", ", ".join(sorted(set(_str_column(zeropoints, "catalog_name")))) if zeropoints is not None else "—", None),
        ("Unstable stars", str(len(products.get("unstable_stars") or [])), None),
    ]
    if limits is not None and len(limits):
        depth = _column(limits, "empty_limit_5sigma_mag")
        if not np.any(np.isfinite(depth)):
            depth = _column(limits, "analytic_limit_5sigma_mag")
        rows.append(("5σ limit (median)", _fmt(np.nanmedian(depth) if np.any(np.isfinite(depth)) else None,
                                              "{:.2f}", "mag"), None))
    trends = products.get("trends")
    if trends is not None and len(trends):
        for row in trends:
            if str(row["status"]) != "PASS":
                rows.append(("Trend {} / {}".format(row["method"], row["variable"]),
                             _fmt(row["slope_mag_per_span"], "{:+.3f}", "mag"), str(row["status"])))
    metric_panel(figure.add_subplot(grid[:, 3]), rows, "Calibration")
    return _finish(figure, output_path, show)


@_styled
def plot_calibration_image_diagnostics(products, image_id, metadata=None, output_path=None,
                                       show=False, status=None):
    """Zeropoint fit for one image: per-star residuals and the zeropoint by method."""

    products = products or {}
    metadata = metadata or {"filename": image_id}
    stars = products.get("calibration_stars")
    zeropoints = products.get("zeropoints")
    image_stars = None if stars is None or not len(stars) else stars[
        _str_column(stars, "image_id") == str(image_id)]
    image_zeropoints = None if zeropoints is None or not len(zeropoints) else zeropoints[
        _str_column(zeropoints, "image_id") == str(image_id)]
    method = _preferred_method(image_stars) if image_stars is not None and len(image_stars) else None
    statuses = list(_str_column(image_zeropoints, "status")) if image_zeropoints is not None else []
    status = status or next((name for name in ("FAIL", "WARN", "PASS") if name in statuses), None)
    figure, grid = _new_figure(
        "Calibration  ·  " + _image_label(metadata, image_id), _image_subtitle(metadata), status,
        "calibration stars scatter evenly around zero with no trend in magnitude or "
        "position; clipped stars are genuine outliers.",
        rows=2, columns=3, width_ratios=[1, 1, 0.9], size=(16, 9.4),
    )
    if image_stars is None or not len(image_stars):
        _empty(figure.add_subplot(grid[:, 0:2]), "No calibration stars for this image")
        metric_panel(figure.add_subplot(grid[:, 2]), [], "Calibration")
        return _finish(figure, output_path, show)
    selected = image_stars[_str_column(image_stars, "method") == method]
    inlier = _bool_column(selected, "inlier")
    residual = _column(selected, "calibrated_residual")
    magnitude = _column(selected, "catalog_magnitude")
    instrumental = _column(selected, "instrumental_magnitude")
    zp_row = None
    if image_zeropoints is not None:
        rows_for_method = image_zeropoints[_str_column(image_zeropoints, "method") == method]
        zp_row = rows_for_method[0] if len(rows_for_method) else None
    zeropoint = None if zp_row is None else _finite(zp_row["zeropoint_mag"])
    scatter = None if zp_row is None else _finite(zp_row["zeropoint_scatter_mag"])

    fit = figure.add_subplot(grid[0, 0])
    fit.set_title("Instrumental vs catalog magnitude ({})".format(method.replace("_", " ")))
    fit.scatter(magnitude[~inlier], instrumental[~inlier], s=14, marker="x", color=RED,
                linewidths=0.8, label="clipped")
    fit.scatter(magnitude[inlier], instrumental[inlier], s=10, color=TEAL, label="used")
    if zeropoint is not None and np.any(np.isfinite(magnitude)):
        span = np.array([np.nanmin(magnitude), np.nanmax(magnitude)])
        fit.plot(span, span - zeropoint, color=INK, linewidth=1.0,
                 label="ZP = {:.3f}".format(zeropoint))
    fit.invert_yaxis()
    fit.set_xlabel("catalog magnitude")
    fit.set_ylabel("instrumental magnitude")
    _legend(fit, loc="lower right")
    _grid(fit)

    residual_axis = figure.add_subplot(grid[0, 1])
    residual_axis.set_title("Residual vs catalog magnitude")
    residual_axis.scatter(magnitude[~inlier], residual[~inlier], s=14, marker="x", color=RED,
                          linewidths=0.8)
    residual_axis.scatter(magnitude[inlier], residual[inlier], s=10, color=TEAL)
    if scatter:
        residual_axis.axhspan(-scatter, scatter, color=TEAL, alpha=0.12, linewidth=0,
                              label="±{:.3f} mag".format(scatter))
    residual_axis.axhline(0, color=RULE)
    spread = max(0.15, 6 * (scatter or 0.03))
    residual_axis.set_ylim(-spread, spread)
    residual_axis.set_xlabel("catalog magnitude")
    residual_axis.set_ylabel("calibrated − catalog [mag]")
    _legend(residual_axis, loc="upper left")
    _grid(residual_axis)

    field = figure.add_subplot(grid[1, 0])
    field.set_title("Residuals across the field")
    limit = max(0.05, 3 * (scatter or 0.03))
    points = field.scatter(_column(selected, "x")[inlier], _column(selected, "y")[inlier],
                           c=residual[inlier], cmap=DIVERGING_CMAP, vmin=-limit, vmax=limit,
                           s=26, edgecolors=INK, linewidths=0.2)
    _colorbar(figure, points, field, "residual [mag]")
    field.scatter(_column(selected, "x")[~inlier], _column(selected, "y")[~inlier], marker="x",
                  s=14, color=FAINT, linewidths=0.7)
    field.set_aspect("equal")
    field.set_xlabel("x [pixel]")
    field.set_ylabel("y [pixel]")

    methods_axis = figure.add_subplot(grid[1, 1])
    methods_axis.set_title("Zeropoint by photometry method")
    if image_zeropoints is not None and len(image_zeropoints):
        names = list(_str_column(image_zeropoints, "method"))
        values = _column(image_zeropoints, "zeropoint_mag")
        errors = _column(image_zeropoints, "zeropoint_scatter_mag")
        positions = np.arange(len(names))
        methods_axis.errorbar(positions, values, yerr=errors, fmt="o", color=BLUE, capsize=4)
        methods_axis.set_xticks(positions)
        methods_axis.set_xticklabels([name.replace("_", " ") for name in names])
        methods_axis.set_ylabel("zeropoint [mag]")
        _grid(methods_axis, "y")
    else:
        _empty(methods_axis, "No zeropoints")

    rows = [
        ("Method shown", method.replace("_", " "), None),
        ("Catalog", "{} ({})".format(zp_row["catalog_name"], zp_row["magnitude_band"]) if zp_row is not None else "—", None),
        ("Zeropoint", _fmt(zeropoint, "{:.3f}", "mag"), None),
        ("Uncertainty", _fmt(None if zp_row is None else zp_row["zeropoint_uncertainty_mag"], "{:.3f}", "mag"), None),
        ("Star scatter", _fmt(scatter, "{:.3f}", "mag"), None),
        ("Stars used / clipped", "{} / {}".format(int(inlier.sum()), int((~inlier).sum())), None),
        ("Aperture correction", _fmt(None if zp_row is None else zp_row["aperture_correction_mag"],
                                     "{:+.3f}", "mag"), None),
    ]
    limits = products.get("limits")
    if limits is not None and len(limits):
        mine = limits[(_str_column(limits, "image_id") == str(image_id)) &
                      (_str_column(limits, "method") == method)]
        if len(mine):
            rows.append(("5σ limit (empty apertures)", _fmt(mine[0]["empty_limit_5sigma_mag"], "{:.2f}", "mag"), None))
            rows.append(("5σ limit (analytic)", _fmt(mine[0]["analytic_limit_5sigma_mag"], "{:.2f}", "mag"), None))
    for row in image_zeropoints if image_zeropoints is not None else []:
        rows.append(("Status, {}".format(str(row["method"]).replace("_", " ")), str(row["status"]),
                     str(row["status"])))
    metric_panel(figure.add_subplot(grid[:, 2]), rows, "Zeropoint")
    return _finish(figure, output_path, show)


# ---------------------------------------------------------------------------
# subtraction and difference photometry
# ---------------------------------------------------------------------------

@_styled
def plot_subtraction_diagnostics(result, science_record=None, output_path=None, show=False,
                                 metadata=None, status=None):
    """Science, aligned template, difference, and subtraction quality."""

    result = result or {}
    metadata = metadata or (science_record or {}).get("metadata") or {"filename": result.get("image_id")}
    status = status or result.get("status")
    figure, grid = _new_figure(
        "Subtraction  ·  " + _image_label(metadata, result.get("image_id")),
        _image_subtitle(metadata), status,
        "stars vanish in the difference (no dipoles or rings) and the noise there matches "
        "the expectation; the transient remains.",
        rows=2, columns=4, size=(18, 9.4), width_ratios=[1, 1, 1, 0.85],
    )
    science = None
    if science_record is not None:
        for name in ("prepared_ccd", "working_ccd", "ccd"):
            if science_record.get(name) is not None:
                science = _data(science_record[name])
                break
    aligned = (result.get("aligned_template") or {}).get("data")
    difference = result.get("difference")
    limits = zscale_limits(science) if science is not None else None
    show_sky(figure.add_subplot(grid[0, 0]), science, "Science", limits=limits, labels=False)
    show_sky(figure.add_subplot(grid[0, 1]), aligned, "Aligned template", labels=False)
    noise = None
    quality = result.get("quality") or {}
    if difference is not None:
        noise = _robust(_sample(difference))[1]
    show_map(figure, figure.add_subplot(grid[0, 2]), difference, "Difference",
             symmetric=True, limit=None if not noise else 5 * noise, label="flux", labels=False)

    histogram = figure.add_subplot(grid[1, 0])
    histogram.set_title("Difference pixels ÷ robust σ")
    if difference is not None and noise:
        values = _sample(difference) / noise
        bins = np.linspace(-8, 8, 121)
        histogram.hist(values, bins=bins, density=True, color=BLUE, alpha=0.75)
        x = np.linspace(-8, 8, 400)
        histogram.plot(x, np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi), color=INK)
        histogram.set_yscale("log")
        histogram.set_ylim(1e-5, 1)
        _grid(histogram)
    else:
        _empty(histogram, "No difference image")

    residuals = quality.get("star_residuals")
    stars = figure.add_subplot(grid[1, 1])
    stars.set_title("Residual fraction on quality stars")
    if residuals is not None and len(residuals):
        stars.scatter(_column(residuals, "science_flux"), _column(residuals, "residual_fraction"),
                      c=_column(residuals, "dipole_fraction"), cmap=MAP_CMAP, s=14)
        stars.set_xscale("symlog")
        stars.axhline(0.10, color=STATUS_COLORS["WARN"], linestyle="--", linewidth=1.0)
        stars.set_xlabel("science flux")
        stars.set_ylabel("|residual| / flux")
        _grid(stars)
    else:
        _empty(stars, "No quality-star measurements")

    blank = np.asarray(quality.get("blank_aperture_fluxes", []), dtype=float)
    blank = blank[np.isfinite(blank)]
    blank_axis = figure.add_subplot(grid[1, 2])
    blank_axis.set_title("Blank-aperture fluxes")
    if blank.size:
        blank_axis.hist(blank, bins=min(30, max(5, blank.size // 2)), color=VIOLET, alpha=0.75)
        blank_axis.axvline(0, color=INK)
        blank_axis.set_xlabel("aperture flux")
        _grid(blank_axis, "y")
    else:
        _empty(blank_axis, "No blank apertures")

    parameters = result.get("parameters") or {}
    rows = [
        ("Backend", str(result.get("method", "—")), None),
        ("Convolved image", str(parameters.get("convolve", "—")), None),
        ("Template coverage", _fmt(None if (result.get("aligned_template") or {}).get("coverage_fraction") is None
                                   else 100 * result["aligned_template"]["coverage_fraction"], "{:.1f}", "%"), None),
        ("Median residual fraction", _fmt(quality.get("median_residual_fraction"), "{:.3f}"), None),
        ("Median dipole fraction", _fmt(quality.get("median_dipole_fraction"), "{:.3f}"), None),
        ("Noise ratio (measured / expected)", _fmt(quality.get("noise_ratio"), "{:.2f}"), None),
    ]
    if result.get("error"):
        rows.append(("Error", str(result["error"])[:60], "FAIL"))
    metric_panel(figure.add_subplot(grid[:, 3]), rows, "Subtraction", flags=result.get("flags"))
    return _finish(figure, output_path, show)


@_styled
def plot_difference_photometry_diagnostics(result, output_path=None, show=False,
                                           metadata=None, status=None):
    """Forced photometry on the difference image compared with the science image."""

    result = result or {}
    record = (result.get("difference_record") or {}).get("science_record") or {}
    metadata = metadata or record.get("metadata") or {"filename": result.get("image_id")}
    status = status or result.get("status")
    diagnostics = result.get("target_diagnostics") or {}
    figure, grid = _new_figure(
        "Difference photometry  ·  " + _image_label(metadata, result.get("image_id")),
        _image_subtitle(metadata), status,
        "the transient is centered in the apertures with no dipole; difference and "
        "science fluxes differ only by the host light.",
        rows=2, columns=4, size=(18, 9.0), width_ratios=[1.1, 1, 1, 0.9],
    )
    context = diagnostics.get("context_data")
    axis = figure.add_subplot(grid[:, 0])
    if context is not None:
        noise = _robust(_sample(context))[1]
        show_map(figure, axis, context, "Difference at target", symmetric=True,
                 limit=None if not noise else 5 * noise, label="flux", labels=False)
        _draw_apertures(axis, diagnostics, diagnostics.get("context_origin", (0, 0)))
        _legend(axis, loc="upper right")
    else:
        _empty(axis, "No difference cutout")
    show_map(figure, figure.add_subplot(grid[0, 1]), diagnostics.get("model"),
             "Forced PSF model", label="flux", labels=False)
    show_map(figure, figure.add_subplot(grid[0, 2]), diagnostics.get("residual"),
             "Difference − model", symmetric=True, labels=False)
    comparison = result.get("comparison")
    flux_axis = figure.add_subplot(grid[1, 1:3])
    flux_axis.set_title("Science vs difference flux by method")
    if comparison is not None and len(comparison):
        methods = list(_str_column(comparison, "method"))
        positions = np.arange(len(methods))
        flux_axis.errorbar(positions - 0.1, _column(comparison, "science_flux"),
                           yerr=_column(comparison, "science_uncertainty"), fmt="o", color=FAINT,
                           capsize=3, label="science (host included)")
        flux_axis.errorbar(positions + 0.1, _column(comparison, "difference_flux"),
                           yerr=_column(comparison, "difference_uncertainty"), fmt="o", color=BLUE,
                           capsize=3, label="difference (host removed)")
        flux_axis.set_xticks(positions)
        flux_axis.set_xticklabels([name.replace("_", " ") for name in methods])
        flux_axis.axhline(0, color=RULE)
        _legend(flux_axis, loc="best")
        _grid(flux_axis, "y")
    else:
        _empty(flux_axis, "No paired measurements")
    preferred = result.get("preferred_result") or {}
    dipole = result.get("dipole") or {}
    rows = [
        ("Preferred", "{} / {}".format(preferred.get("image_kind", "—"), preferred.get("method", "—")), None),
        ("Classification", str(preferred.get("classification", "—")), None),
        ("S/N", _fmt(preferred.get("snr"), "{:.1f}"), _check_status(preferred.get("snr"), 5.0, 3.0, "low")),
        ("Difference PSF", str(result.get("difference_psf_source", "—")), None),
        ("Dipole", "yes" if dipole.get("detected") else "no", "WARN" if dipole.get("detected") else None),
        ("Host light included", str(result.get("preferred_host_light_included")), None),
    ]
    metric_panel(figure.add_subplot(grid[:, 3]), rows, "Difference photometry",
                 flags=result.get("flags"))
    return _finish(figure, output_path, show)


# ---------------------------------------------------------------------------
# batch consistency
# ---------------------------------------------------------------------------

@_styled
def plot_batch_consistency_diagnostics(products, output_path=None, show=False, status=None):
    """Final light curve, epoch metrics over time, and comparison-star stability."""

    products = products or {}
    status = status or products.get("status")
    subtitle = "failed epochs {}  ·  measurement outliers {}  ·  unstable comparison stars {} of {} tested".format(
        products.get("failed_epoch_count", 0), products.get("measurement_outlier_count", 0),
        products.get("unstable_comparison_count", 0), products.get("tested_comparison_count", "—"))
    figure, grid = _new_figure(
        "Light curve and batch consistency", subtitle, status,
        "the preferred light curve is smooth within its errors; rejected epochs (hollow) "
        "have a clear reason in the epoch metrics.",
        rows=2, columns=4, size=(18, 9.8), width_ratios=[1.2, 1.2, 1, 0.85],
    )
    curve = figure.add_subplot(grid[0, 0:2])
    curve.set_title("Preferred light curve")
    preferred = products.get("preferred_light_curve")
    if preferred is not None and len(preferred):
        filters = _str_column(preferred, "filter")
        included = _bool_column(preferred, "included_in_final")
        mjd = _column(preferred, "mjd")
        magnitude = _column(preferred, "magnitude")
        error = _column(preferred, "magnitude_uncertainty")
        classification = _str_column(preferred, "classification")
        for band in dict.fromkeys(filters):
            color = _filter_color(band)
            for keep, face in ((True, color), (False, "white")):
                use = (filters == band) & (included == keep) & np.isfinite(magnitude)
                if not np.any(use):
                    continue
                curve.errorbar(mjd[use], magnitude[use], yerr=error[use], fmt="o", color=color,
                               mfc=face, mec=color, ms=6, capsize=0, elinewidth=1.0,
                               label="{}{}".format(band, "" if keep else " (excluded)"))
        limit = classification == "upper_limit"
        if np.any(limit):
            curve.scatter(mjd[limit], magnitude[limit], marker="v", color=FAINT, s=30,
                          label="upper limit")
        curve.invert_yaxis()
        curve.set_xlabel("MJD")
        curve.set_ylabel("magnitude")
        _legend(curve, loc="best", ncol=3)
        _grid(curve)
    else:
        _empty(curve, "No preferred light curve")

    epoch = products.get("epoch_metrics")
    specs = (("zeropoint_mag", "Zeropoint [mag]", grid[1, 0]),
             ("seeing_fwhm_arcsec", "Seeing [arcsec]", grid[1, 1]),
             ("depth_5sigma_mag", "5σ depth [mag]", grid[0, 2]),
             ("wcs_rms_arcsec", "WCS RMS [arcsec]", grid[1, 2]))
    for column, label, cell in specs:
        axis = figure.add_subplot(cell)
        axis.set_title(label)
        if epoch is not None and len(epoch) and column in epoch.colnames:
            mjd = _column(epoch, "mjd")
            values = _column(epoch, column)
            filters = _str_column(epoch, "filter")
            statuses = _str_column(epoch, "status")
            for band in dict.fromkeys(filters):
                use = filters == band
                axis.scatter(mjd[use], values[use], color=_filter_color(band), s=28, label=band,
                             edgecolors=[_status_color(s) if s != "PASS" else "none" for s in statuses[use]],
                             linewidths=1.6)
            if column == "depth_5sigma_mag":
                axis.invert_yaxis()
            axis.set_xlabel("MJD")
            _grid(axis)
        else:
            _empty(axis, "Not available")

    stability = products.get("comparison_stability")
    rows = [
        ("Status", str(products.get("status", "—")), products.get("status")),
        ("Failed epochs", str(products.get("failed_epoch_count", 0)), None),
        ("Outlier measurements", str(products.get("measurement_outlier_count", 0)), None),
        ("Comparison stars tested", str(products.get("tested_comparison_count", "—")), None,
         "{} more with too few epochs to test".format(products.get("untested_comparison_count", 0))),
        ("Unstable comparison stars", str(products.get("unstable_comparison_count", 0)), None,
         "scatter beyond errors (+{} floor)".format(
             _fmt(_finite((stability[0]["error_floor_mag"] if stability is not None and len(stability)
                           and "error_floor_mag" in stability.colnames else None)), "{:.2f}", " mag"))),
        ("Ensemble correction", "on" if products.get("ensemble_enabled") else "off", None),
    ]
    if preferred is not None and len(preferred):
        rows.append(("Light-curve points", "{} kept / {} total".format(
            int(_bool_column(preferred, "included_in_final").sum()), len(preferred)), None))
        for row in preferred:
            flags_text = str(row["flags"] or "").replace(";", ", ").lower().replace("_", " ")
            if not bool(row["included_in_final"]) or flags_text:
                rows.append((str(row["image_id"]).split(".")[0],
                             "kept" if bool(row["included_in_final"]) else "excluded",
                             "WARN" if bool(row["included_in_final"]) else "FAIL",
                             flags_text[:60]))
    metric_panel(figure.add_subplot(grid[:, 3]), rows, "Batch consistency")
    return _finish(figure, output_path, show)


__all__ = [
    "diagnostic_style",
    "metric_panel",
    "plot_alignment_target_diagnostics",
    "plot_astrometry_diagnostics",
    "plot_background_diagnostics",
    "plot_batch_consistency_diagnostics",
    "plot_calibration_diagnostics",
    "plot_calibration_image_diagnostics",
    "plot_cosmic_ray_diagnostics",
    "plot_difference_photometry_diagnostics",
    "plot_fringe_diagnostics",
    "plot_image_quality_diagnostics",
    "plot_image_usability_diagnostics",
    "plot_mask_diagnostics",
    "plot_psf_diagnostics",
    "plot_read_diagnostics",
    "plot_region_diagnostics",
    "plot_science_photometry_diagnostics",
    "plot_stage_overview",
    "plot_stage_status",
    "plot_star_selection_diagnostics",
    "plot_subtraction_diagnostics",
    "show_map",
    "show_sky",
    "zscale_limits",
]
