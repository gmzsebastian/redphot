"""Run redphot one step at a time, and redo a step for chosen images.

Each of the 19 steps has its own function, named after the step and its
number, the same as the step's folder in ``<run_directory>/diagnostics``
(``06_background`` is :func:`background_06`)::

    from redphot.steps import start_run, read_01, region_02, background_06

    run = start_run("data/*.fits", "my_run", target=target)
    read_01(run)
    region_02(run)
    ...
    background_06(run, images=[3], box_size=[64, 64])    # redo step 6 for image 3

Keyword parameters are settings of the step, given by their name alone
(``box_size`` is ``background.box_size``). ``help(background_06)`` lists the
settings of a step with their defaults, ``show_parameters(background_06, run)``
with the values of a run. With ``images`` a change applies to those images
only, otherwise to every image. Results the change makes out of date are
redone when their steps run again (or right away with ``redo_later=True``),
everything else is kept.
"""

from collections.abc import Mapping
from datetime import datetime
from difflib import get_close_matches
from functools import lru_cache
import json
from pathlib import Path

from .config import merge_settings
from .pipeline import (
    _json_value,
    _stage_definitions,
    first_stage_reading,
    initialize_pipeline,
    load_pipeline_state,
    mark_pipeline_stale,
    pipeline_stage_names,
    review_image,
    run_pipeline_stage,
    save_pipeline_state,
    set_image_overrides,
    set_run_overrides,
    stage_reads_setting as _pipeline_stage_reads_setting,
)

# The stage definitions are fixed, so whether a stage reads a setting never
# changes; caching it keeps building the step help texts at import fast.
stage_reads_setting = lru_cache(maxsize=None)(_pipeline_stage_reads_setting)

# One line per step, shown by list_steps() and in each step's help.
STEP_SUMMARIES = {
    "read": "read the FITS files and their headers (filter, times, gain, saturation)",
    "region": "usable pixel region: data section, edge trimming, optional crop",
    "masks": "masks: saturation, bad rows/columns, amplifier seams, satellite trails",
    "cosmic_rays": "cosmic-ray masking (off unless masks.cosmic_rays.enabled)",
    "fringe": "fringe correction (off unless a fringe map is given)",
    "background": "model and subtract the sky background",
    "source_quality": "detect sources; seeing, ellipticity and image quality",
    "astrometry": "match to Gaia and check/refine the WCS",
    "star_selection": "choose zeropoint, PSF and comparison stars (all images together)",
    "usability": "quick zeropoint, depth and clouds; FAIL images are dropped (all images)",
    "alignment": "align all images and fix the target position (all images)",
    "psf": "build the PSF of each image; FAIL images are dropped",
    "science_photometry": "forced aperture and PSF photometry of the target and stars",
    "calibration": "zeropoints and limiting magnitudes (all images together)",
    "templates": "download/load the subtraction templates (all images)",
    "subtraction": "template subtraction (when subtraction.enabled)",
    "difference_photometry": "forced photometry on the difference images",
    "batch_consistency": "checks across the batch and the final light curve (all images)",
    "outputs": "write the final tables, FITS files, light-curve plot and PDFs",
}

# Where a step's keyword parameters are looked for first (dotted settings
# paths). A name found in several places goes to the first of these.
STEP_HOMES = {
    "read": ["metadata"],
    "region": ["crop"],
    "masks": ["masks"],
    "cosmic_rays": ["masks.cosmic_rays"],
    "fringe": ["fringe"],
    "background": ["background"],
    "source_quality": ["source_detection", "image_quality"],
    "astrometry": ["astrometry", "catalogs"],
    "star_selection": ["catalogs.comparison_stars", "psf", "catalogs"],
    "usability": ["image_quality.usability", "image_quality"],
    "alignment": ["target_position", "astrometry"],
    "psf": ["psf"],
    "science_photometry": ["apertures", "target_position", "background"],
    "calibration": ["calibration", "upper_limits"],
    "templates": ["subtraction"],
    "subtraction": ["subtraction.hotpants", "subtraction"],
    "difference_photometry": ["subtraction.photometry", "apertures", "upper_limits",
                              "subtraction"],
    "batch_consistency": ["batch_consistency"],
    "outputs": ["output", "diagnostics"],
}

# Nested settings whose entries can be given by name; any other dictionary
# setting is one value (pass it whole).
SETTING_GROUPS = (
    "masks.cosmic_rays",
    "catalogs.comparison_stars",
    "image_quality.usability",
    "subtraction.hotpants",
    "subtraction.pyzogy",
    "subtraction.photometry",
    "batch_consistency.ensemble_correction",
)

# Steps that run on all images together but can still use different
# settings for some images.
PER_IMAGE_BATCH_STEPS = {"star_selection", "usability", "calibration"}

_ALIASES = {
    "sources": "source_quality", "seeing": "source_quality", "stars": "star_selection",
    "photometry": "science_photometry", "zeropoints": "calibration",
    "difference": "difference_photometry", "consistency": "batch_consistency",
    "cosmics": "cosmic_rays", "cosmic": "cosmic_rays", "output": "outputs",
}


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------
class Run:
    """One run: its ``state`` and ``context``, saved in ``directory``.

    :func:`start_run` returns one, and every function here takes it as its
    first argument. The ``state, context`` of
    :func:`redphot.pipeline.run_batch` or
    :func:`redphot.pipeline.load_pipeline_state` become one with
    ``Run(state, context)``, and ``state, context = run`` gives them back for
    the functions of :mod:`redphot.pipeline`.
    """

    def __init__(self, state, context):
        self.state = state
        self.context = context

    @property
    def directory(self):
        """Folder the run is saved in."""

        return Path(self.state["run_directory"])

    def __iter__(self):
        return iter((self.state, self.context))

    def __repr__(self):
        reached = _reached_stages(self.state)
        return "<redphot run in {}: {} images, {}>".format(
            self.directory, len(self.state["images"]),
            "last step {}".format(function_name(reached[-1])) if reached else "no step run yet")


def _as_run(run):
    if isinstance(run, Run):
        return run
    if isinstance(run, (tuple, list)) and len(run) == 2:
        return Run(*run)
    raise TypeError(
        "Give the run from start_run as the first argument: run = start_run(...), "
        "then e.g. read_01(run). For the state, context of run_batch use "
        "Run(state, context).")


# ---------------------------------------------------------------------------
# Names and numbers
# ---------------------------------------------------------------------------
def step_name(step):
    """Stage name of a step given by its function (``background_06``), number
    (1-19), name (``"background"``), function name (``"background_06"``) or
    folder name (``"06_background"``)."""

    names = pipeline_stage_names()
    if callable(step) and hasattr(step, "stage"):
        return step.stage
    if isinstance(step, int) and not isinstance(step, bool):
        if not 1 <= step <= len(names):
            raise ValueError("Steps are numbered 1-{}; see list_steps()".format(len(names)))
        return names[step - 1]
    text = str(step).strip().lower()
    if text[:2].isdigit() and text[2:3] == "_":
        text = text[3:]  # "09_star_selection"
    elif text[-2:].isdigit() and text[-3:-2] == "_":
        text = text[:-3]  # "star_selection_09"
    if text.isdigit():
        return step_name(int(text))
    text = _ALIASES.get(text, text)
    if text not in names:
        close = get_close_matches(text, names, n=1)
        hint = " Did you mean {}?".format(function_name(close[0])) if close else ""
        raise ValueError("Unknown step {!r}.{} See list_steps().".format(step, hint))
    return text


def step_number(step):
    """Number (1-19) of a step."""

    return pipeline_stage_names().index(step_name(step)) + 1


def function_name(step):
    """Name of a step's function: ``function_name("background")`` is
    ``"background_06"``."""

    stage = step_name(step)
    return "{}_{:02d}".format(stage, step_number(stage))


def _scope(stage):
    return next(item["scope"] for item in _stage_definitions() if item["name"] == stage)


def list_steps():
    """Print the 19 step functions in order and what each one does."""

    for name in pipeline_stage_names():
        print("  {:25s} {}".format(function_name(name), STEP_SUMMARIES.get(name, "")))


def _image_ids(state, images=None):
    ids = list(state["images"])
    if images is None:
        return ids
    if isinstance(images, (list, tuple, set)):
        found = []
        for item in images:
            for image_id in _image_ids(state, item):
                if image_id not in found:
                    found.append(image_id)
        return found
    if isinstance(images, int) and not isinstance(images, bool):
        if not 1 <= images <= len(ids):
            raise ValueError("Image numbers are 1-{}; see list_images(run)".format(len(ids)))
        return [ids[images - 1]]
    text = str(images)
    if text in ids:
        return [text]
    matches = [image_id for image_id in ids if text in image_id]
    if not matches:
        raise ValueError("No image matches {!r}; see list_images(run)".format(images))
    return matches


def image_ids(run, images=None):
    """Image IDs from a number (as in :func:`list_images`), a name, part of a
    name, or a list of those. ``None`` means every image."""

    return _image_ids(_as_run(run).state, images)


# ---------------------------------------------------------------------------
# Settings by name
# ---------------------------------------------------------------------------
def _setting_paths(settings):
    """Every settings path a keyword can name: (dotted path, value)."""

    paths = []
    for section, values in settings.items():
        if not isinstance(values, Mapping):
            continue
        for key, value in values.items():
            path = "{}.{}".format(section, key)
            paths.append((path, value))
            if path in SETTING_GROUPS and isinstance(value, Mapping):
                paths.extend(("{}.{}".format(path, inner), item)
                             for inner, item in value.items())
    # Any header field can be replaced with metadata.<field>_override
    # (instrument_override, site_override, date_obs_override, ...).
    metadata = settings.get("metadata") or {}
    for field in metadata.get("keywords") or {}:
        key = "{}_override".format(field)
        if key not in metadata:
            paths.append(("metadata." + key, None))
    return paths


def _nested(path, value):
    nested = value
    for key in reversed(path.split(".")):
        nested = {key: nested}
    return nested


def _step_paths(stage, settings):
    return [(path, value) for path, value in _setting_paths(settings)
            if stage_reads_setting(stage, path)]


def find_setting(step, name, settings=None):
    """Full dotted path of the setting ``name`` of a step.

    Example: ``find_setting(star_selection_09, "minimum_snr")`` is
    ``"catalogs.comparison_stars.minimum_snr"``.
    """

    from .config import get_default_settings

    settings = settings or get_default_settings()
    return _resolve(step_name(step), name, _paths_by_name(settings))


def _paths_by_name(settings):
    """Setting name (last part of the path) -> every dotted path with it."""

    found = {}
    for path, _ in _setting_paths(settings):
        found.setdefault(path.split(".")[-1], []).append(path)
    return found


def _resolve(stage, name, by_name):
    """The path the keyword ``name`` means for ``stage`` (see find_setting)."""

    every = by_name.get(name, [])
    readable = [path for path in every if stage_reads_setting(stage, path)]
    if len(readable) == 1:
        return readable[0]
    if not readable:
        if every:
            path = every[0]
            user = function_name(first_stage_reading(_nested(path, None)))
            raise ValueError(
                "{} is not a setting of {}: {} is used from {} on, so give it "
                "there, e.g. {}(run, {}=...)".format(
                    name, function_name(stage), path, user, user, name))
        names = sorted(key for key, paths in by_name.items()
                       if any(stage_reads_setting(stage, path) for path in paths))
        close = get_close_matches(name, names, n=3)
        raise ValueError("{} has no setting {!r}.{} show_parameters({}) lists them.".format(
            function_name(stage), name,
            " Did you mean {}?".format(", ".join(close)) if close else "",
            function_name(stage)))
    for home in STEP_HOMES.get(stage, []):
        under = [path for path in readable if path.startswith(home + ".")]
        if under:
            depth = min(path.count(".") for path in under)
            best = [path for path in under if path.count(".") == depth]
            if len(best) == 1:
                return best[0]
            break
    raise ValueError(
        "{!r} is ambiguous for {}: {}. Give it nested instead, e.g. "
        "settings={}".format(name, function_name(stage), ", ".join(readable),
                             json.dumps(_nested(readable[0], "...")))
    )


def step_overrides(step, parameters=None, settings=None, base=None):
    """Nested settings overrides from a step's keyword parameters.

    ``parameters`` maps setting names (``box_size``), or section names with a
    dictionary (``background={"box_size": [64, 64]}``), to new values;
    ``settings`` is an optional nested dictionary merged on top.
    """

    from .config import get_default_settings

    base = base or get_default_settings()
    overrides = {}
    for name, value in (parameters or {}).items():
        if name in base and isinstance(base[name], Mapping) and isinstance(value, Mapping):
            piece = {name: value}
        else:
            piece = _nested(find_setting(step, name, base), value)
        overrides = merge_settings(overrides, piece)
    if settings:
        overrides = merge_settings(overrides, settings)
    return overrides


def _parameter_rows(stage, settings):
    """(name, value, dotted path, path when the name alone does not reach it)
    for every setting a step reads, in the order of its STEP_HOMES."""

    homes = STEP_HOMES.get(stage, [])

    def order(item):
        path = item[0]
        for index, home in enumerate(homes):
            if path.startswith(home + "."):
                return (index, -path.count("."))
        return (len(homes), 0)

    rows = []
    by_name = _paths_by_name(settings)
    for path, value in sorted(_step_paths(stage, settings), key=order):
        if path in SETTING_GROUPS:
            continue
        name = path.split(".")[-1]
        try:
            resolved = _resolve(stage, name, by_name)
        except ValueError:
            resolved = None
        rows.append((name, value, path, resolved == path))
    return rows


def _short_value(value, width=30):
    text = repr(_json_value(value))
    return text if len(text) <= width else text[:width - 3] + "..."


def show_parameters(step, run=None, image=None):
    """Print the settings a step reads, with their current values.

    ``show_parameters(background_06)`` shows the defaults,
    ``show_parameters(background_06, run)`` the values of a run, and
    ``image=`` one image's values (its overrides included). Any name printed
    can be given to the step as a keyword, e.g. ``background_06(run,
    box_size=[64, 64])``.
    """

    from .config import get_default_settings

    stage = step_name(step)
    settings = get_default_settings()
    where = "defaults"
    if run is not None:
        run = _as_run(run)
        settings = run.context.get("settings") or settings
        where = "run settings"
        if image is not None:
            image_id = _image_ids(run.state, image)[0]
            settings = run.context["images"][image_id]["settings"]
            where = image_id
    print("{}: {}".format(function_name(stage), STEP_SUMMARIES.get(stage, "")))
    print("Settings you can give by name, e.g. {}(run, name=value) (values: {}):".format(
        function_name(stage), where))
    for name, value, path, by_name in _parameter_rows(stage, settings):
        note = "" if by_name else "   (only as settings={})".format(
            json.dumps(_nested(path, "...")))
        print("  {:38s} {:30s} {}{}".format(name, _short_value(value), path, note))


# ---------------------------------------------------------------------------
# Start, list, status
# ---------------------------------------------------------------------------
def start_run(input_files, run_directory, target=None, settings=None, image_overrides=None,
              instrument=None, new=False):
    """Start a run, or reopen the one already saved in ``run_directory``.

    ``input_files`` is a folder, a glob pattern (``"data/*.fits"``) or a list of
    files. A run already saved in ``run_directory`` (by a step function or by
    :func:`redphot.pipeline.run_batch`) is reopened instead, so the same line
    continues a run in a new session. With ``new=True`` an existing run folder
    is renamed (not deleted) and a fresh run starts. Prints the numbered list
    of images and returns the :class:`Run`.
    """

    run_directory = Path(run_directory)
    saved = run_directory / "pipeline_state.json"
    if saved.exists() and new:
        backup = run_directory.with_name("{}_old_{}".format(
            run_directory.name, datetime.now().strftime("%Y%m%d_%H%M%S")))
        run_directory.rename(backup)
        print("Moved the previous run to", backup)
    if saved.exists():
        print("Reopening the run in {} (loading the saved images takes a moment)".format(
            run_directory))
        state, context = load_pipeline_state(run_directory)
        if settings is not None and _json_value(settings) != _json_value(
                state.get("run_settings") or {}):
            print("Note: the saved run has other run settings than the ones given here.\n"
                  "      Use new=True to start over, or change a setting with its step,\n"
                  "      e.g. background_06(run, box_size=[64, 64]).")
    else:
        if isinstance(input_files, (str, Path)):
            input_files = str(input_files)
        else:
            input_files = [str(path) for path in input_files]
        state, context = initialize_pipeline(
            input_files, settings=settings, instrument_name=instrument, target=target,
            image_overrides=image_overrides, run_directory=str(run_directory),
        )
        run_directory.mkdir(parents=True, exist_ok=True)
        save_pipeline_state(state, context)
        print("New run in", run_directory)
    run = Run(state, context)
    list_images(run)
    print("Plots and tables of each step go to", run_directory / "diagnostics")
    return run


def _reached_stages(state):
    """Stages any image (or the batch) has reached, in pipeline order."""

    names = pipeline_stage_names()
    reached = set(state.get("batch_stages", {}))
    for image in state["images"].values():
        reached.update(image.get("stages", {}))
    return [name for name in names if name in reached]


def _furthest_stage(state):
    reached = _reached_stages(state)
    return reached[-1] if reached else pipeline_stage_names()[0]


def _last_stage(state, image_id):
    stages = state["images"][image_id].get("stages", {})
    done = [name for name in pipeline_stage_names() if name in stages]
    return done[-1] if done else None


def list_images(run):
    """Numbered list of the images (the numbers ``images=`` accepts)."""

    state, context = _as_run(run)
    print("  #  {:44s} {:6s} {:10s} {:9s} {}".format("image", "filter", "MJD", "status",
                                                    "last step"))
    for number, image_id in enumerate(state["images"], 1):
        metadata = context["images"][image_id]["record"].get("metadata") or {}
        mjd = metadata.get("mjd_mid")
        last = _last_stage(state, image_id)
        print("{:3d}  {:44s} {:6s} {:10s} {:9s} {}".format(
            number, image_id[:44], str(metadata.get("filter") or "?"),
            "{:.3f}".format(mjd) if mjd else "?",
            str(state["images"][image_id].get("status")) if last else "new",
            function_name(last) if last else "-"))


def show_status(run):
    """Step-by-image table: P pass, W warn, F fail, A approved, R rejected,
    s skipped, ~ out of date (stale), . not run."""

    state = _as_run(run).state
    symbols = {"PASS": "P", "WARN": "W", "FAIL": "F", "APPROVED": "A", "REJECTED": "R",
               "SKIPPED": "s", "STALE": "~", None: "."}
    names = pipeline_stage_names()
    print("step  " + "".join("{:>3d}".format(i) for i in range(1, len(names) + 1)))
    for number, image_id in enumerate(state["images"], 1):
        row = ""
        for name in names:
            entry = state["images"][image_id].get("stages", {}).get(name)
            if entry is None and _scope(name) == "batch":
                entry = state.get("batch_stages", {}).get(name)
            row += "{:>3s}".format(symbols.get((entry or {}).get("status"), "?"))
        print("{:3d}   {}   {}".format(number, row, image_id[:34]))
    print("P pass, W warn, F fail, A approved, R rejected, s skipped, ~ out of date, "
          ". not run")


# ---------------------------------------------------------------------------
# Running steps
# ---------------------------------------------------------------------------
def _gates(context):
    return set(context["settings"].get("pipeline", {}).get("review_gates", []))


def _reopen_automatic_rejections(state, context, identifiers, from_stage):
    """Let the gates at or after ``from_stage`` decide again for these images.

    Only automatic rejections are undone; a rejection made with
    :func:`reject_images` or :func:`redphot.pipeline.review_image` stays.
    """

    names = pipeline_stage_names()
    start = names.index(from_stage)
    reopened = []
    for image_id in identifiers:
        image = state["images"][image_id]
        for gate in sorted(_gates(context), key=names.index):
            if names.index(gate) < start:
                continue
            decision = image.get("review_decisions", {}).get(gate) or {}
            if decision.get("decision") != "REJECTED" or decision.get("note") != "automatic review":
                continue
            image["review_decisions"].pop(gate)
            for name in names[names.index(gate):]:
                entry = image.get("stages", {}).get(name)
                if entry and (name == gate or entry.get("blocked")):
                    entry["status"] = "STALE"
                    entry["review_status"] = None
                    entry["blocked"] = False
            reopened.append(image_id)
    for image_id in reopened:
        from .pipeline import _update_image_status

        _update_image_status(state, image_id)
    return reopened


def _apply_changes(state, context, stage, identifiers, overrides, force, selected):
    """Store overrides and mark what they make out of date; return the first
    stage that has to run."""

    names = pipeline_stage_names()
    start = stage
    if overrides:
        first = first_stage_reading(overrides)
        if names.index(first) < names.index(stage):
            print("Note: this setting is first used by {}, so the steps run from "
                  "there.".format(function_name(first)))
            start = first
        if not selected:
            set_run_overrides(state, context, overrides, from_stage=start, save=False)
            print("Changed for every image: {}".format(json.dumps(_json_value(overrides))))
        else:
            if _scope(start) == "batch" and start not in PER_IMAGE_BATCH_STEPS:
                raise ValueError(
                    "{} uses the same settings for every image; leave out images= to "
                    "change it for all of them.".format(function_name(start)))
            for image_id in identifiers:
                set_image_overrides(state, context, image_id, overrides, from_stage=start,
                                    save=False)
            print("Changed for {}: {}".format(
                ", ".join(_short(state, image_id) for image_id in identifiers),
                json.dumps(_json_value(overrides))))
    if force:
        if _scope(start) == "batch" or not selected:
            mark_pipeline_stale(state, start, None, "redo requested")
        else:
            for image_id in identifiers:
                mark_pipeline_stale(state, start, image_id, "redo requested")
    if overrides or force:
        reopened = _reopen_automatic_rejections(state, context, identifiers, start)
        if reopened:
            print("Rejected automatically before; deciding again: {}".format(
                ", ".join(_short(state, image_id) for image_id in reopened)))
    return start


def _short(state, image_id):
    return "{} ({})".format(list(state["images"]).index(image_id) + 1, image_id)


def _earliest_stale(state, before):
    """First step before ``before`` that is out of date for any image."""

    names = pipeline_stage_names()
    for name in names[:names.index(before)]:
        if (state.get("batch_stages", {}).get(name) or {}).get("status") == "STALE":
            return name
        if any((image.get("stages", {}).get(name) or {}).get("status") == "STALE"
               for image in state["images"].values()):
            return name
    return None


def _run_one(state, context, stage, identifiers, selected):
    """Run one stage: image stages for the selected images, batch stages whole."""

    if _scope(stage) == "image" and selected:
        for image_id in identifiers:
            run_pipeline_stage(state, context, stage, image_id=image_id, save=False)
        save_pipeline_state(state, context)
    else:
        run_pipeline_stage(state, context, stage)


def run_step(run, step, images=None, *, redo_later=False, force=False, settings=None,
             **parameters):
    """Run one step, optionally with changed settings, and print the result.

    This is what every step function (:func:`read_01` ... :func:`outputs_19`)
    does; ``run_step(run, background_06, ...)``, ``run_step(run, 6, ...)``
    and ``background_06(run, ...)`` are the same. See :func:`background_06`
    for the parameters.
    """

    run = _as_run(run)
    state, context = run
    stage = step_name(step)
    names = pipeline_stage_names()
    selected = images is not None
    identifiers = _image_ids(state, images)
    overrides = step_overrides(stage, parameters, settings, context.get("settings"))
    start = _apply_changes(state, context, stage, identifiers, overrides, force, selected)
    earlier = _earliest_stale(state, start)
    if redo_later is False or redo_later is None:
        if earlier is not None:
            print("Note: {} is out of date for some images (an earlier change); run it "
                  "and the steps after it first, or add redo_later=True.".format(
                      function_name(earlier)))
        for name in names[names.index(start):names.index(stage) + 1]:
            _run_one(state, context, name, identifiers, selected)
        report(run, stage, identifiers if selected else None)
        return
    # Change and continue: run every step from here to the last one wanted
    # for all images; results the change did not affect are kept as they are.
    # Steps an earlier change left out of date are brought up to date first.
    if earlier is not None:
        print("Note: {} is out of date for some images (an earlier change), so the "
              "steps run from there.".format(function_name(earlier)))
        start = earlier
    last = _furthest_stage(state) if redo_later is True else step_name(redo_later)
    if names.index(last) < names.index(stage):
        last = stage
    for name in names[names.index(start):names.index(last) + 1]:
        run_pipeline_stage(state, context, name)
    report(run, stage, identifiers if selected else None)
    if last != stage:
        print("Redid {} to {}; later steps of other images were kept where their inputs "
              "did not change.".format(function_name(start), function_name(last)))
        if last == "outputs":
            report(run, "outputs")


def _step_docstring(stage):
    """Help text of a step function, with the settings it reads and their
    defaults."""

    from .config import get_default_settings

    number = step_number(stage)
    folder = "{:02d}_{}".format(number, stage)
    summary = STEP_SUMMARIES.get(stage, stage)
    if _scope(stage) == "image":
        scope = ("Runs on each image; with ``images`` only on those (the others keep "
                 "their results).")
    elif stage in PER_IMAGE_BATCH_STEPS:
        scope = ("Works on all images together, but each image can have its own "
                 "settings: with ``images`` a change applies to those images only.")
    else:
        scope = ("Works on all images together with one set of settings, so its "
                 "settings are changed without ``images``.")
    lines = [
        "Step {}: {}.".format(number, summary[0].upper() + summary[1:]),
        "",
        "{} Plots and ``summary.csv`` go to ``<run_directory>/diagnostics/{}``. "
        "Results this step makes out of date are redone when their steps run "
        "again; everything else is kept.".format(scope, folder),
        "",
        "Its settings are listed under Other Parameters with their defaults "
        "(before instrument profiles are applied); ``show_parameters({}, run)`` "
        "shows their values in a run.".format(function_name(stage)),
        "",
        "Parameters",
        "----------",
        "run : Run",
        "    The run, from :func:`start_run`.",
        "images : int, str or list, optional",
        "    Images whose settings change (numbers from :func:`list_images`, "
        "file names, parts of file names, or a list of those). Without it a "
        "change applies to every image.",
        "redo_later : bool or step, optional",
        "    ``True`` also redoes the later steps, up to the last step the run "
        "had reached (only where the change affects their results); a step "
        "function such as ``outputs_19`` stops at that step instead. Earlier "
        "steps that a previous change left out of date are redone first.",
        "force : bool, optional",
        "    Run the step again even though nothing changed (for example after "
        "editing the RedPhot code).",
        "settings : dict, optional",
        "    Nested settings to change, for names that are ambiguous.",
        "**parameters",
        "    Settings of this step by name, ``name=value``; see Other Parameters.",
        "",
    ]
    rows = _parameter_rows(stage, get_default_settings())
    if rows:
        lines += [
            "Other Parameters",
            "----------------",
        ]
        for name, value, path, by_name in rows:
            text = "``{}``, default ``{}``.".format(path, _short_value(value, 40))
            if not by_name:
                text += " Only as ``settings={}``.".format(json.dumps(_nested(path, "...")))
            lines += [name if by_name else path, "    " + text]
    return "\n".join(lines).rstrip() + "\n"


def _step_function(stage):
    """The function that runs one step, e.g. ``background_06``."""

    def step(run, images=None, *, redo_later=False, force=False, settings=None,
             **parameters):
        run_step(run, stage, images, redo_later=redo_later, force=force,
                 settings=settings, **parameters)

    step.__name__ = step.__qualname__ = function_name(stage)
    step.__module__ = __name__
    step.__doc__ = _step_docstring(stage)
    step.stage = stage
    return step


# One function per step, named after the step and its number (the same as the
# step's folder in diagnostics/).
read_01 = _step_function("read")
region_02 = _step_function("region")
masks_03 = _step_function("masks")
cosmic_rays_04 = _step_function("cosmic_rays")
fringe_05 = _step_function("fringe")
background_06 = _step_function("background")
source_quality_07 = _step_function("source_quality")
astrometry_08 = _step_function("astrometry")
star_selection_09 = _step_function("star_selection")
usability_10 = _step_function("usability")
alignment_11 = _step_function("alignment")
psf_12 = _step_function("psf")
science_photometry_13 = _step_function("science_photometry")
calibration_14 = _step_function("calibration")
templates_15 = _step_function("templates")
subtraction_16 = _step_function("subtraction")
difference_photometry_17 = _step_function("difference_photometry")
batch_consistency_18 = _step_function("batch_consistency")
outputs_19 = _step_function("outputs")

#: The 19 step functions in order.
STEPS = (read_01, region_02, masks_03, cosmic_rays_04, fringe_05, background_06,
         source_quality_07, astrometry_08, star_selection_09, usability_10, alignment_11,
         psf_12, science_photometry_13, calibration_14, templates_15, subtraction_16,
         difference_photometry_17, batch_consistency_18, outputs_19)


# ---------------------------------------------------------------------------
# Dropping images by hand
# ---------------------------------------------------------------------------
def _decide(run, images, decision, note):
    run = _as_run(run)
    state, context = run
    names = pipeline_stage_names()
    gates = []
    for image_id in _image_ids(state, images):
        stages = state["images"][image_id].get("stages", {})
        gate = next((name for name in ("psf", "usability") if name in stages), None)
        if gate is None:
            print("Skipping {}: it has not reached {} yet".format(
                image_id, function_name("usability")))
            continue
        review_image(state, context, image_id, gate, decision, note=note)
        gates.append(gate)
        print("{} {} at {}".format(_short(state, image_id), decision.lower(),
                                   function_name(gate)))
    if gates:
        after = names[min(names.index(gate) for gate in gates) + 1]
        print("Later steps of these images are out of date; run them again, or "
              "{}(run, redo_later=True).".format(function_name(after)))


def reject_images(run, images, note=None):
    """Drop images from the latest gate they reached (:func:`usability_10` or
    :func:`psf_12`) on.

    They stay in the run and keep everything computed so far; later steps
    skip them.
    """

    _decide(run, images, "REJECTED", note)


def keep_images(run, images, note=None):
    """Use images again after :func:`reject_images` or an automatic rejection."""

    _decide(run, images, "APPROVED", note)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def report(run, step, images=None):
    """Print one line per image for a step: status, key numbers, flags."""

    from .stage_reports import metric_specs, stage_directory, stage_summary_rows

    state, context = _as_run(run)
    stage = step_name(step)
    try:
        rows = stage_summary_rows(state, context, stage)
        specs = metric_specs(stage, context.get("settings") or {})[:4]
    except Exception:  # the summary must never stop a run; show statuses only
        specs = []
        rows = []
        for image_id, image in state["images"].items():
            entry = image.get("stages", {}).get(stage) or state.get(
                "batch_stages", {}).get(stage) or {}
            rows.append({"image_id": image_id, "label": image_id, "filter": "",
                         "status": entry.get("status") or "NOT RUN", "flags": "",
                         "note": str(entry.get("error") or "").strip()[-70:]})
    batch = state.get("batch_stages", {}).get(stage)
    print("\n{}: {}".format(function_name(stage), STEP_SUMMARIES.get(stage, "")))
    if batch is not None:
        line = "  all images together: {}".format(batch.get("status"))
        if batch.get("error"):
            line += "  " + str(batch["error"]).strip().splitlines()[-1][:100]
        print(line)
    if stage != "outputs":
        print("  #  {:30s} {:3s} {:9s}".format("image", "flt", "status") + "".join(
            " {:>13s}".format(spec["label"][:13]) for spec in specs) + "  flags / note")
        wanted = set(images) if images else None
        for row in rows:
            if wanted is not None and row["image_id"] not in wanted:
                continue
            values = ""
            for spec in specs:
                try:
                    values += " {:>13s}".format(spec.get("spec", "{:.3g}").format(
                        float(row.get(spec["key"]))))
                except (TypeError, ValueError):
                    values += " {:>13s}".format("-")
            print("{:3d}  {:30s} {:3s} {:9s}{}  {}".format(
                list(state["images"]).index(row["image_id"]) + 1, row["label"][:30],
                str(row.get("filter") or "")[:3], row["status"], values,
                (row.get("flags") or row.get("note") or "")[:70]))
    later = pipeline_stage_names()[pipeline_stage_names().index(stage) + 1:]
    stale = [name for name in later
             if any((image.get("stages", {}).get(name) or {}).get("status") == "STALE"
                    for image in state["images"].values())
             or (state.get("batch_stages", {}).get(name) or {}).get("status") == "STALE"]
    print("  plots and summary.csv:", stage_directory(state, stage))
    if stage == "outputs":
        result = context.get("shared", {}).get("outputs") or {}
        folder = (result.get("paths") or {}).get("manifest")
        if folder:
            print("  products in", Path(folder).parent)
    if stale:
        print("  out of date now: {}{} (run them again)".format(
            function_name(stale[0]), " and later steps" if len(stale) > 1 else ""))


__all__ = [
    "Run",
    "start_run",
    "read_01",
    "region_02",
    "masks_03",
    "cosmic_rays_04",
    "fringe_05",
    "background_06",
    "source_quality_07",
    "astrometry_08",
    "star_selection_09",
    "usability_10",
    "alignment_11",
    "psf_12",
    "science_photometry_13",
    "calibration_14",
    "templates_15",
    "subtraction_16",
    "difference_photometry_17",
    "batch_consistency_18",
    "outputs_19",
    "STEPS",
    "list_steps",
    "list_images",
    "show_parameters",
    "show_status",
    "reject_images",
    "keep_images",
    "run_step",
    "report",
    "image_ids",
    "find_setting",
    "step_name",
    "step_number",
    "function_name",
    "step_overrides",
]
