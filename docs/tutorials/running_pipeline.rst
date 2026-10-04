Running the Pipeline
====================

There are two ways to run RedPhot, and they share the same saved run, so they
can be mixed:

* **All in one**: :func:`redphot.pipeline.run_batch` runs every step on every
  image in one call (below).
* **Step by step**: :mod:`redphot.steps` runs one step at a time so each
  step's plots can be checked first (see :doc:`step_by_step`).

Either way, a setting of any step can be changed afterwards for some images
(or all) and the run continued from that step; see
:ref:`change-one-setting`.

All in one
----------

.. code-block:: python

   from pathlib import Path
   from astropy.coordinates import SkyCoord
   from redphot.pipeline import run_batch

   INPUT_DIR = Path("AT_2019stc/input_data")
   RUN_DIR = Path("AT_2019stc/full_run")
   TARGET = SkyCoord("06:54:23.103", "+17:29:31.35", unit=("hourangle", "deg"))
   RUN_SETTINGS = {
       "masks": {"cosmic_rays": {"enabled": True}},
       "subtraction": {"enabled": True, "cache_directory": "templates"},
       "catalogs": {"cache_directory": "catalogs"},
       "output": {"profile": "standard"},
   }
   IMAGE_OVERRIDES = {
       "AT2019stc_IMACS.fits": {"metadata": {"instrument_override": "IMACS",
                                             "site_override": "Las Campanas"}},
   }

   state, context = run_batch(str(INPUT_DIR), settings=RUN_SETTINGS, target=TARGET,
                              image_overrides=IMAGE_OVERRIDES,
                              run_directory=str(RUN_DIR))

``run_batch`` starts a new run in ``RUN_DIR`` and runs the 19 steps in order.
The instrument of each image is taken from its header unless
``instrument_name=`` (all images) or ``IMAGE_OVERRIDES`` (one image) says
otherwise; ``RUN_SETTINGS`` apply to every image and ``IMAGE_OVERRIDES`` to the
named file only. Nothing waits for a decision: at usability (step 10) and PSF
(step 12) images that FAIL are rejected automatically and skipped by later
steps, PASS and WARN images go on.

Each step writes its plots and a ``summary.csv`` to
``RUN_DIR/diagnostics/<NN>_<step>/`` as it finishes (see :doc:`../outputs`),
and the final products go to ``RUN_DIR/products/``. The run is saved after
every step (``pipeline_state.json`` and ``pipeline_context.pkl``); keep
``RUN_DIR`` on a local disk, the second file holds every image array.

While a step runs, RedPhot prints one line when it starts, one line per image
(status and time), lines from inside the slow parts, and the total time.
``{"pipeline": {"verbose": False}}`` switches this off.

.. _change-one-setting:

Change one setting and continue
-------------------------------

After a run (all in one or step by step), a setting of any step can be changed
for the images that need it, and the run continued from that step. Say the
plots of step 9 (``09_star_selection``) show that images 3 and 7 keep too many
faint comparison stars:

.. code-block:: python

   from redphot.pipeline import load_pipeline_state
   from redphot.steps import list_images, rerun_from

   state, context = load_pipeline_state(RUN_DIR)   # not needed if the run is still open
   list_images(state, context)                     # the image numbers
   rerun_from(state, context, 9, images=[3, 7], minimum_snr=20)

This

* stores ``catalogs.comparison_stars.minimum_snr = 20`` for images 3 and 7
  only (the other images keep 10);
* does not redo steps 1-8 for any image;
* redoes step 9 and every later step up to the last step the run had reached
  (``through=`` stops earlier, e.g. ``through=12``).

Steps 9, 10, 11, 14, 15, 18 and 19 work on all images together, so they run
again for the whole batch, but each image's own selection uses its own
settings. After each of them RedPhot compares every image's part of the new
result with the old one, and the image steps that follow (PSF, photometry,
subtraction, ...) are redone only for the images whose part changed. The other
images are kept as they were ("unchanged, kept" in the progress lines). When a
step that works on all images changes something every image uses, every image
is redone from there. The one exception is the fixed target position: it
counts as unchanged while it stays within
``pipeline.rerun_position_tolerance_mas`` (10 milliarcseconds, which changes
forced photometry by about 10⁻⁴ mag in 1-2 arcsec seeing; 0 compares it
exactly) of the position the kept images were measured at.

The keyword is the setting's name, without its section;
``show_parameters(9)`` lists every setting of step 9. Without ``images=`` the
change applies to every image:

.. code-block:: python

   rerun_from(state, context, 9, minimum_snr=20)        # all images
   rerun_from(state, context, 16, images=[11], kernel_order=0)
   rerun_from(state, context, 6, images="aa0629", box_size=[64, 64], through=8)

Settings of alignment (11), templates (15), batch consistency (18) and outputs
(19) are the same for all images, so they are changed without ``images=``.

To run only the step that changed and look at it before continuing, use
:func:`~redphot.steps.run_step` with the same arguments; later steps of the
changed images are then marked out of date (``~`` in
``show_status(state)``) until they are run:

.. code-block:: python

   from redphot.steps import run_step, show_status

   run_step(state, context, 9, images=[3, 7], minimum_snr=20)
   show_status(state)
   rerun_from(state, context, 10)        # continue from step 10 with no change

Changes are saved with the run: reopening it later (``load_pipeline_state`` or
``start_run``) keeps them, and each image's changes are listed under
``"overrides"`` in ``pipeline_state.json``.

Dropping an image by hand
-------------------------

.. code-block:: python

   from redphot.steps import keep_images, reject_images

   reject_images(state, context, [4], note="thin clouds")
   rerun_from(state, context, 11)
   keep_images(state, context, [4])      # undo

The image is dropped at the latest gate it reached (usability or PSF), keeps
everything computed before, and is skipped by later steps.
:func:`redphot.pipeline.review_image` does the same for one image and one gate.

Read and inspect one FITS file
------------------------------

Use the lower-level functions when inspecting ingestion before starting a
run. The filter is read from the FITS metadata; a ``filter_name`` argument is
only needed when deliberately overriding or testing configuration resolution.

.. code-block:: python

   from redphot.config import resolve_settings
   from redphot.image import read_fits_image

   filename = "AT_2024rmj_r_FLWO_2024.1012.fits"
   settings = resolve_settings(instrument_name="KeplerCam", image_name=filename)
   ccd, metadata = read_fits_image(filename, settings=settings)

   print(ccd.shape, metadata["data_hdu"])
   print(metadata["filter"], metadata["mjd_mid"])
   print(metadata["metadata_status"], metadata["quality_flags"])

For a mixed-filter pipeline run, pass user filter overrides through
``filter_settings`` on ``run_batch``; the controller applies them after reading
and normalizing each header filter.

Photometric reference catalogs
------------------------------

Astrometry uses Gaia. During star selection, each filter's calibration catalog
(``catalogs.photometry_catalog_by_filter``; PS1 for griz) is queried once,
cached next to the Gaia cache, and matched to the Gaia sources by position
(``catalogs.photometric_match_arcsec``). Use an absolute
``catalogs.cache_directory`` so the cache does not depend on the working
directory.

Lower-level control
-------------------

:mod:`redphot.steps` is built on these functions of :mod:`redphot.pipeline`,
which take stage names and nested settings:

.. code-block:: python

   from redphot.pipeline import (rerun_image, run_pipeline_stage, run_pipeline_through,
                                 set_image_overrides, set_run_overrides)

   run_pipeline_stage(state, context, "background")             # one stage
   set_image_overrides(state, context, "aa0629.fits",
                       {"background": {"box_size": [96, 96]}})  # one image
   set_run_overrides(state, context,
                     {"catalogs": {"comparison_stars": {"minimum_snr": 20}}})
   rerun_image(state, context, "aa0629.fits", from_stage="background",
               through_stage="psf")
   run_pipeline_through(state, context)                          # everything left

``set_image_overrides`` and ``set_run_overrides`` start from the first stage
that reads a changed setting (:func:`redphot.pipeline.first_stage_reading`).
A rerun always starts from the image that entered the stage (for background,
the output of the fringe stage), so repeating a stage never compounds its
correction.

Resume
------

.. code-block:: python

   from redphot.pipeline import resume_pipeline

   state, context = resume_pipeline("AT_2019stc/full_run")

Valid completed products are reused. Products whose inputs, settings or
earlier steps changed are marked stale and rebuilt. Original FITS files remain
unchanged.
