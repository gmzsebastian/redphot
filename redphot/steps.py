"""Run redphot one step at a time, and redo a step for chosen images.

Every function works on the ``state`` and ``context`` of a run, from
:func:`start_run`, :func:`redphot.pipeline.run_batch` or
:func:`redphot.pipeline.load_pipeline_state`. Steps are given by number
(1-19, the numbers of the ``diagnostics`` folders) or by name::

    from redphot.steps import start_run, run_step, rerun_from

    state, context = start_run("data/*.fits", "my_run", target=target)
    run_step(state, context, 1)                    # step 1, read
    run_step(state, context, "background", images=[3], box_size=[64, 64])
    rerun_from(state, context, 9, images=[3], minimum_snr=20)

Keyword parameters are settings of the step, given by their name alone
(``box_size`` is ``background.box_size``); :func:`show_parameters` lists them
with their current values. With ``images`` a change applies to those images
only, otherwise to every image. Results the change makes out of date are
redone, everything else is kept.
"""

from collections.abc import Mapping
from datetime import datetime
from difflib import get_close_matches
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
    stage_reads_setting,
)

# One line per step, shown by list_steps().
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
# Names and numbers
# ---------------------------------------------------------------------------
def step_name(step):
    """Stage name of a step given by number (1-19), name, or folder name."""

    names = pipeline_stage_names()
    if isinstance(step, int) and not isinstance(step, bool):
        if not 1 <= step <= len(names):
            raise ValueError("Steps are numbered 1-{}; see list_steps()".format(len(names)))
        return names[step - 1]
    text = str(step).strip().lower()
    if text[:2].isdigit() and text[2:3] == "_":
        text = text[3:]  # "09_star_selection"
    if text.isdigit():
        return step_name(int(text))
    text = _ALIASES.get(text, text)
    if text not in names:
        close = get_close_matches(text, names, n=1)
        hint = " Did you mean {!r}?".format(close[0]) if close else ""
        raise ValueError("Unknown step {!r}.{} See list_steps().".format(step, hint))
    return text


def step_number(step):
    """Number (1-19) of a step given by number or name."""

    return pipeline_stage_names().index(step_name(step)) + 1


def _scope(stage):
    return next(item["scope"] for item in _stage_definitions() if item["name"] == stage)


def _label(stage):
    return "step {} {}".format(step_number(stage), stage)


def list_steps():
    """Print the 19 steps with their numbers and what each one does."""

    for number, name in enumerate(pipeline_stage_names(), 1):
        print("{:3d}  {:22s} {}".format(number, name, STEP_SUMMARIES.get(name, "")))


def image_ids(state, images=None):
    """Image IDs from a number (as in :func:`list_images`), a name, part of a
    name, or a list of those. ``None`` means every image."""

    ids = list(state["images"])
    if images is None:
        return ids
    if isinstance(images, (list, tuple, set)):
        found = []
        for item in images:
            for image_id in image_ids(state, item):
                if image_id not in found:
                    found.append(image_id)
        return found
    if isinstance(images, int) and not isinstance(images, bool):
        if not 1 <= images <= len(ids):
            raise ValueError("Image numbers are 1-{}; see list_images(state)".format(len(ids)))
        return [ids[images - 1]]
    text = str(images)
    if text in ids:
        return [text]
    matches = [image_id for image_id in ids if text in image_id]
    if not matches:
        raise ValueError("No image matches {!r}; see list_images(state)".format(images))
    return matches


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

    Example: ``find_setting(9, "minimum_snr")`` is
    ``"catalogs.comparison_stars.minimum_snr"``.
    """

    from .config import get_default_settings

    stage = step_name(step)
    settings = settings or get_default_settings()
    every = [path for path, _ in _setting_paths(settings) if path.split(".")[-1] == name]
    readable = [path for path in every if stage_reads_setting(stage, path)]
    if len(readable) == 1:
        return readable[0]
    if not readable:
        if every:
            path = every[0]
            user = first_stage_reading(_nested(path, None))
            raise ValueError(
                "{} is not a setting of {}: {} is used from {} on, so give it "
                "there, e.g. run_step(state, context, {!r}, {}=...)".format(
                    name, _label(stage), path, _label(user), user, name))
        names = sorted({path.split(".")[-1] for path, _ in _step_paths(stage, settings)})
        close = get_close_matches(name, names, n=3)
        raise ValueError("{} has no setting {!r}.{} show_parameters({}) lists them.".format(
            _label(stage), name,
            " Did you mean {}?".format(", ".join(close)) if close else "",
            step_number(stage)))
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
        "settings={}".format(name, _label(stage), ", ".join(readable),
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


def show_parameters(step, state=None, context=None, image=None):
    """Print the settings a step reads, with their current values.

    Pass ``state, context`` for the values of a run, and ``image`` for one
    image's values (its overrides included). Any name printed can be passed to
    :func:`run_step` or :func:`rerun_from` as a keyword.
    """

    from .config import get_default_settings

    stage = step_name(step)
    settings = get_default_settings()
    where = "defaults"
    if context is not None:
        settings = context.get("settings") or settings
        where = "run settings"
        if image is not None:
            image_id = image_ids(state, image)[0]
            settings = context["images"][image_id]["settings"]
            where = image_id
    print("Step {}, {}: {}".format(step_number(stage), stage, STEP_SUMMARIES.get(stage, "")))
    print("Settings you can give by name (values: {}):".format(where))
    homes = STEP_HOMES.get(stage, [])

    def order(item):
        path = item[0]
        for index, home in enumerate(homes):
            if path.startswith(home + "."):
                return (index, -path.count("."))
        return (len(homes), 0)

    for path, value in sorted(_step_paths(stage, settings), key=order):
        if path in SETTING_GROUPS:
            continue
        name = path.split(".")[-1]
        try:
            resolved = find_setting(stage, name, settings)
        except ValueError:
            resolved = None
        text = repr(_json_value(value))
        if len(text) > 30:
            text = text[:27] + "..."
        note = "" if resolved == path else "   (only as settings={})".format(
            json.dumps(_nested(path, "...")))
        print("  {:38s} {:30s} {}{}".format(name, text, path, note))


# ---------------------------------------------------------------------------
# Start, list, status
# ---------------------------------------------------------------------------
def start_run(input_files, run_directory, target=None, settings=None, image_overrides=None,
              instrument=None, new=False):
    """Start a run, or reopen the one already saved in ``run_directory``.

    ``input_files`` is a folder, a glob pattern (``"data/*.fits"``) or a list of
    files. With ``new=True`` an existing run folder is renamed (not deleted)
    and a fresh run starts. Returns ``state, context``.
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
                  "      Use new=True to start over, or change them with run_step()/rerun_from().")
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
    list_images(state, context)
    print("Plots and tables of each step go to", run_directory / "diagnostics")
    return state, context


def _last_stage(state, image_id):
    stages = state["images"][image_id].get("stages", {})
    done = [name for name in pipeline_stage_names() if name in stages]
    return done[-1] if done else None


def list_images(state, context=None):
    """Numbered list of the images (the numbers ``images=`` accepts)."""

    print("  #  {:44s} {:6s} {:10s} {:9s} {}".format("image", "filter", "MJD", "status",
                                                    "last step"))
    for number, image_id in enumerate(state["images"], 1):
        metadata = {}
        if context is not None:
            metadata = context["images"][image_id]["record"].get("metadata") or {}
        mjd = metadata.get("mjd_mid")
        last = _last_stage(state, image_id)
        print("{:3d}  {:44s} {:6s} {:10s} {:9s} {}".format(
            number, image_id[:44], str(metadata.get("filter") or "?"),
            "{:.3f}".format(mjd) if mjd else "?",
            str(state["images"][image_id].get("status")) if last else "new",
            "{} {}".format(step_number(last), last) if last else "-"))


def show_status(state):
    """Step-by-image table: P pass, W warn, F fail, A approved, R rejected,
    s skipped, ~ out of date (stale), . not run."""

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
                  "there.".format(_label(first)))
            start = first
        if not selected:
            set_run_overrides(state, context, overrides, from_stage=start, save=False)
            print("Changed for every image: {}".format(json.dumps(_json_value(overrides))))
        else:
            if _scope(start) == "batch" and start not in PER_IMAGE_BATCH_STEPS:
                raise ValueError(
                    "{} uses the same settings for every image; leave out images= to "
                    "change it for all of them.".format(_label(start)))
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


def _run_one(state, context, stage, identifiers, selected):
    """Run one stage: image stages for the selected images, batch stages whole."""

    if _scope(stage) == "image" and selected:
        for image_id in identifiers:
            run_pipeline_stage(state, context, stage, image_id=image_id, save=False)
        save_pipeline_state(state, context)
    else:
        run_pipeline_stage(state, context, stage)


def run_step(state, context, step, images=None, settings=None, force=False, **parameters):
    r"""Run one step, optionally with changed settings, and print the result.

    Results the change makes out of date (later steps of the changed images)
    are marked stale; run the following steps again, or use
    :func:`rerun_from` to change and continue in one go.

    Parameters
    ----------
    step : int or str
        Step number (1-19) or name (``"background"``, ``"star_selection"``).
    images : int, str or list, optional
        Images to change and run (numbers from :func:`list_images`, names or
        parts of names). Steps that work on all images together (9, 10, 11,
        14, 15, 18, 19) always run on all of them; ``images`` then only says
        whose settings change.
    settings : dict, optional
        Nested settings to change, for names that are ambiguous.
    force : bool
        Run the step again even though nothing changed (for example after
        editing the redphot code).
    **parameters
        Settings of this step by name, e.g. ``box_size=[64, 64]``.
    """

    stage = step_name(step)
    selected = images is not None
    identifiers = image_ids(state, images)
    overrides = step_overrides(stage, parameters, settings, context.get("settings"))
    start = _apply_changes(state, context, stage, identifiers, overrides, force, selected)
    names = pipeline_stage_names()
    for name in names[names.index(start):names.index(stage) + 1]:
        _run_one(state, context, name, identifiers, selected)
    report(state, context, stage, identifiers if selected else None)
    return state, context


def _furthest_stage(state):
    names = pipeline_stage_names()
    reached = [names.index(name) for name in state.get("batch_stages", {})]
    for image in state["images"].values():
        reached.extend(names.index(name) for name in image.get("stages", {}))
    return names[max(reached)] if reached else names[0]


def rerun_from(state, context, step, images=None, through=None, settings=None, force=False,
               **parameters):
    """Change settings at a step (for some images or all) and redo the run from there.

    Runs the step and every later one up to ``through`` (default: the last
    step the run had reached, so a finished run is finished again). Steps
    before ``step`` are not redone. For images that are not selected the later
    steps are redone only where a step that works on all images together gave
    them a different result (for example a new target position); otherwise
    their results are kept as they are.

    Example: use a higher S/N cut for comparison stars (step 9) in images 3
    and 7 only, and bring the run up to date::

        rerun_from(state, context, 9, images=[3, 7], minimum_snr=20)
    """

    stage = step_name(step)
    names = pipeline_stage_names()
    last = step_name(through) if through is not None else _furthest_stage(state)
    selected = images is not None
    identifiers = image_ids(state, images)
    overrides = step_overrides(stage, parameters, settings, context.get("settings"))
    start = _apply_changes(state, context, stage, identifiers, overrides, force, selected)
    if names.index(last) < names.index(stage):
        last = stage
    for name in names[names.index(start):names.index(last) + 1]:
        run_pipeline_stage(state, context, name)
    report(state, context, stage, identifiers if selected else None)
    if last != stage:
        print("Redid steps {}-{} ({} to {}); later steps of other images were kept "
              "where their inputs did not change.".format(
                  step_number(start), step_number(last), start, last))
        if last == "outputs":
            report(state, context, "outputs")
    return state, context


# ---------------------------------------------------------------------------
# Dropping images by hand
# ---------------------------------------------------------------------------
def _decide(state, context, images, decision, note):
    for image_id in image_ids(state, images):
        stages = state["images"][image_id].get("stages", {})
        gate = next((name for name in ("psf", "usability") if name in stages), None)
        if gate is None:
            print("Skipping {}: it has not reached step 10 (usability) yet".format(image_id))
            continue
        review_image(state, context, image_id, gate, decision, note=note)
        print("{} {} at {}".format(_short(state, image_id), decision.lower(), _label(gate)))
    print("Later steps of these images are out of date; run them again "
          "(or rerun_from(state, context, 11)).")


def reject_images(state, context, images, note=None):
    """Drop images from the latest gate they reached (step 10 or 12) on.

    They stay in the run and keep everything computed so far; later steps
    skip them.
    """

    _decide(state, context, images, "REJECTED", note)


def keep_images(state, context, images, note=None):
    """Use images again after :func:`reject_images` or an automatic rejection."""

    _decide(state, context, images, "APPROVED", note)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def report(state, context, step, images=None):
    """One line per image for a step: status, key numbers, flags."""

    from .stage_reports import metric_specs, stage_directory, stage_summary_rows

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
    print("\n{}: {}".format(_label(stage), STEP_SUMMARIES.get(stage, "")))
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
        print("  out of date now (run them again): " + ", ".join(
            "{} {}".format(step_number(name), name) for name in stale))


__all__ = [
    "find_setting",
    "image_ids",
    "keep_images",
    "list_images",
    "list_steps",
    "reject_images",
    "report",
    "rerun_from",
    "run_step",
    "show_parameters",
    "show_status",
    "start_run",
    "step_name",
    "step_number",
    "step_overrides",
]
