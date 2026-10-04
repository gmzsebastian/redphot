# RedPhot

RedPhot is a function-based Python pipeline for robust time-domain optical
photometry, designed primarily for supernova observations from LCO and
KeplerCam. Input images must already be bias- and flat-corrected. RedPhot keeps
the original FITS files read-only and records failures instead of silently
discarding images.

The pipeline supports aperture and PSF photometry on science and difference
images, catalog calibration, limiting magnitudes, batch consistency checks,
diagnostic PDFs, resumable runs, and per-image review decisions.

## Installation

```bash
git clone https://github.com/gmzsebastian/redphot.git
cd redphot
python -m pip install -e .
```

Optional cosmic-ray cleaning requires:

```bash
python -m pip install -e '.[cosmic_rays]'
```

Hotpants is an external executable and is required only when Hotpants image
subtraction is enabled. IRAF and PyRAF are not dependencies.

Complete installation instructions, including CFITSIO and Hotpants builds for
Linux and macOS, are in
[`docs/installation.rst`](docs/installation.rst).

## Minimal batch

```python
from astropy.coordinates import SkyCoord
from redphot.config import resolve_settings
from redphot.pipeline import run_batch

target = SkyCoord("01:07:54.17", "+03:30:03.8", unit=("hourangle", "deg"))
settings = resolve_settings(
    instrument_name="KeplerCam",
    run_settings={
        "subtraction": {"enabled": False},
        "output": {"profile": "standard"},
    },
)

state, context = run_batch(
    "data/*.fits",
    settings=settings,
    target=target,
    run_directory="AT2024rmj_redphot",
    mode="automatic",
)
```

## Step by step

`redphot.steps` runs one step at a time, so each step's plots can be checked
before going on, and changes a step's settings by name for chosen images:

```python
from redphot.steps import start_run, run_step, rerun_from, show_parameters

state, context = start_run("data/*.fits", "AT2024rmj_redphot", target=target)
run_step(state, context, 1)                    # read; then 2, 3, ... 19
run_step(state, context, 6, images=[3], box_size=[64, 64])   # redo step 6 for image 3
show_parameters(9)                             # settings of step 9 and their values
```

After a run (all in one or step by step), change a setting of one step for
some images and continue from that step; steps before it are not redone, and
later steps are redone only for images whose results change:

```python
rerun_from(state, context, 9, images=[3, 7], minimum_snr=20)
```

`docs/tutorials/step_by_step.rst` explains every step, what to look at and
what to change.

## Stepwise runs with the lower-level functions

Stages can also be run one at a time (`run_pipeline_stage`) to look at each
stage's plots before going on. Nothing ever waits for an approval: at the
usability and PSF gates, images that FAIL are rejected automatically and
skipped by later stages, while PASS and WARN images continue. Any decision can
still be changed by hand:

```python
from redphot.pipeline import review_image, run_pipeline_through

review_image(state, context, "AT_2024rmj_r_FLWO_2024.1012.fits", "usability",
             "REJECTED", note="clouds")
state, context = run_pipeline_through(state, context)
```

Long stages print progress (one line per image and from inside the slow
steps); switch it off with `{"pipeline": {"verbose": False}}`.

Every stage writes plots and a `summary.csv` to
`<run_directory>/diagnostics/<NN>_<stage>/` as soon as it runs, so each step can
be checked before continuing (see `docs/outputs.rst`).

Runs can be resumed with `resume_pipeline("AT2024rmj_redphot")`. Configuration
changes made with `set_image_overrides` (or `redphot.steps`) mark only the
first stage that reads the changed setting and what follows it stale.

## Output size

`minimal` saves core tables, the final light-curve figure
(`<object>_lightcurve.png/.pdf`, every photometry method), configuration, log,
and manifest. `standard` also saves reports and, per image in `fits/`, the
PSF model, the difference image (when subtraction ran), and one final
processed image (cut to the usable area, background-subtracted, WCS aligned
to the reference, with MASK, BKG and BKGRMS extensions). `full` saves
every supplied derivative. Individual products can be changed with
`output.product_overrides`.

## Templates

With subtraction enabled, templates come from a user file
(`subtraction.template_path`) or are downloaded from the first survey in
`subtraction.template_survey_priority` that covers the field in that band:
Pan-STARRS1 (STScI cutout service), the DESI Legacy Surveys (`legacy`, or
`decam` for DECam-only data), or SDSS (through SkyView). Each survey's recipe
is in `subtraction.template_surveys`; set `template_source` to one survey name
to use only that one. Downloads are cached per survey, field and size; a
cached template of the same survey and band that covers the field is reused
even when the refined WCS has moved the field center slightly. The Templates
figure (`15_templates/batch.png`) shows each template with the outline of
every image that uses it.

```python
{"output": {
    "profile": "standard",
    "product_overrides": {
        "difference_image": False,
        "image_pdfs": False,
        "background_model": True,
    },
}}
```

See the documentation for worked examples, the equations and algorithms,
configuration precedence, the complete function API, output schemas,
instrument behavior, troubleshooting, and release validation.

## Release status

The current `0.1.x` line is a development release. A first stable release must
not be declared until the real-data validation checklist in
`docs/validation.rst` is complete, including a clean, repeatable multi-filter
supernova reduction and comparison with established `Phot_good.py` results.
