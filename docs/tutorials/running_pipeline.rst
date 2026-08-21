Running the Pipeline
====================

Minimal automatic batch
-----------------------

.. code-block:: python

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
       run_directory="AT2024rmj_run",
       mode="automatic",
   )

Stepwise review
---------------

.. code-block:: python

   from redphot.pipeline import review_image, run_batch, run_pipeline_through

   state, context = run_batch(
       "data/*.fits*",
       settings=settings,
       target=target,
       run_directory="AT2024rmj_review",
       mode="stepwise",
   )

   review_image(
       state, context,
       "AT_2024rmj_r_FLWO_2024.1012.fits",
       "usability", "APPROVED",
       note="Depth and quality diagnostics inspected",
   )
   state, context = run_pipeline_through(state, context, mode="stepwise")

Approve or reject the PSF in the same way when the second review gate is
reached. Rejected images are retained but blocked from downstream science
measurements.

Override and rerun one image
----------------------------

.. code-block:: python

   from redphot.pipeline import rerun_image, set_image_overrides

   set_image_overrides(
       state, context,
       "AT_2024rmj_r_FLWO_2024.1012.fits",
       {"background": {"box_size": [96, 96]}},
   )
   state, context = rerun_image(
       state, context,
       "AT_2024rmj_r_FLWO_2024.1012.fits",
       from_stage="background",
       through_stage="psf",
       mode="stepwise",
   )

Resume
------

.. code-block:: python

   from redphot.pipeline import resume_pipeline

   state, context = resume_pipeline("AT2024rmj_review", mode="stepwise")

Valid completed products are reused. Changed dependencies are marked stale and
rebuilt. Original FITS files remain unchanged.
