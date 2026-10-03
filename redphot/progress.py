"""Short progress messages printed while redphot runs.

Long stages (source detection, star selection, calibration, the output
reports) can take minutes on a large batch. The pipeline prints one line when
a stage starts, one line per image as it finishes, and a few lines from inside
the slow steps, so a run never sits silently. Messages go to standard output
and are switched off with ``{"pipeline": {"verbose": False}}``.
"""

import sys
import time


_STATE = {"enabled": True, "stage": None, "stage_start": None}


def configure(enabled=True):
    """Switch progress messages on or off for the rest of the session."""

    _STATE["enabled"] = bool(enabled)


def enabled():
    """True when progress messages are printed."""

    return bool(_STATE["enabled"])


def progress(message, indent=4):
    """Print one progress line (no-op when messages are switched off)."""

    if not _STATE["enabled"]:
        return
    print("{}{}".format(" " * int(indent), message), file=sys.stdout, flush=True)


def stage_started(stage, count=None, scope="image"):
    """Announce a stage and remember when it started."""

    _STATE["stage"] = stage
    _STATE["stage_start"] = time.perf_counter()
    if not _STATE["enabled"]:
        return
    if scope == "image" and count is not None:
        detail = "{} image{}".format(count, "" if count == 1 else "s")
    elif count is not None:
        detail = "all {} images together".format(count)
    else:
        detail = ""
    print("[{}] {}{}".format(time.strftime("%H:%M:%S"), stage,
                             "  ({})".format(detail) if detail else ""),
          file=sys.stdout, flush=True)


def stage_finished(message=""):
    """Close a stage with its total time."""

    start = _STATE.get("stage_start")
    if not _STATE["enabled"] or start is None:
        return
    elapsed = time.perf_counter() - start
    print("    {}finished in {}".format("{}; ".format(message) if message else "",
                                       format_seconds(elapsed)),
          file=sys.stdout, flush=True)


def format_seconds(seconds):
    """``4.2 s`` or ``3 min 05 s``."""

    seconds = float(seconds)
    if seconds < 60:
        return "{:.1f} s".format(seconds)
    minutes, rest = divmod(int(round(seconds)), 60)
    return "{} min {:02d} s".format(minutes, rest)


class Timer:
    """Elapsed time since creation, formatted for messages."""

    def __init__(self):
        self.start = time.perf_counter()

    def __str__(self):
        return format_seconds(time.perf_counter() - self.start)


__all__ = [
    "Timer",
    "configure",
    "enabled",
    "format_seconds",
    "progress",
    "stage_finished",
    "stage_started",
]
