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

## Stepwise review

```python
from redphot.pipeline import review_image, run_batch, run_pipeline_through

state, context = run_batch(
    "data/*.fits*",
    settings=settings,
    target=target,
    run_directory="AT2024rmj_review",
    mode="stepwise",
)

review_image(
    state,
    context,
    "AT_2024rmj_r_FLWO_2024.1012.fits",
    "usability",
    "APPROVED",
    note="Quality and depth diagnostics inspected",
)

state, context = run_pipeline_through(state, context, mode="stepwise")
```

Runs can be resumed with `resume_pipeline("AT2024rmj_review")`. Configuration
changes made with `set_image_overrides` mark only affected and downstream
products stale.

## Output size

`minimal` saves core tables, configuration, log, and manifest. `standard` also
saves reports, PSF models, and difference images. `full` saves every supplied
derivative. Individual products can be changed with `output.product_overrides`.

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
