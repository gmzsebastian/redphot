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

Stepwise runs
-------------

.. code-block:: python

   from redphot.pipeline import review_image, run_pipeline_stage, run_pipeline_through

   run_pipeline_stage(state, context, "read")
   run_pipeline_stage(state, context, "region")
   # ... look at <run_directory>/diagnostics/ after each stage ...

Nothing waits for an approval. The usability and PSF gates
(``pipeline.review_gates``) decide automatically after they run: images that
FAIL are rejected and skipped by later stages, PASS and WARN images continue.
Rejected images keep everything computed before the rejection. A decision can
be changed at any time:

.. code-block:: python

   review_image(
       state, context,
       "AT_2024rmj_r_FLWO_2024.1012.fits",
       "usability", "REJECTED",
       note="thin clouds in the diagnostics",
   )
   state, context = run_pipeline_through(state, context)

While a stage runs, RedPhot prints one line when it starts, one line per image
(status and time), lines from inside the slow batch steps, and the total time.
``{"pipeline": {"verbose": False}}`` switches this off.

Diagnostics while you go
------------------------

Every stage writes plots and a ``summary.csv`` to
``<run_directory>/diagnostics/<NN>_<stage>/`` as soon as it runs (see
:doc:`../outputs`), so a stepwise run can be judged stage by stage before
continuing. Keep the run directory on a local disk: the checkpoint
(``pipeline_context.pkl``) holds the image arrays of every stage and is
rewritten after each one.

Photometric reference catalogs
------------------------------

Astrometry uses Gaia. During star selection, each filter's calibration catalog
(``catalogs.photometry_catalog_by_filter``; PS1 for griz) is queried once,
cached next to the Gaia cache, and matched to the Gaia sources by position
(``catalogs.photometric_match_arcsec``). Use an absolute
``catalogs.cache_directory`` so the cache does not depend on the working
directory.

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
   )

A rerun always starts from the image that entered the stage (for background,
the output of the fringe stage), so repeating a stage never compounds its
correction. To change a setting for every image, including settings read by
batch stages, use :func:`redphot.pipeline.set_run_overrides`; per-image
overrides keep precedence:

.. code-block:: python

   from redphot.pipeline import set_run_overrides

   set_run_overrides(state, context, {"catalogs": {"comparison_stars": {"minimum_snr": 20}}})
   state, context = run_pipeline_through(state, context)

Resume
------

.. code-block:: python

   from redphot.pipeline import resume_pipeline

   state, context = resume_pipeline("AT2024rmj_redphot")

Valid completed products are reused. Changed dependencies are marked stale and
rebuilt. Original FITS files remain unchanged.
