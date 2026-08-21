.. _reference:

Pipeline Reference
==================

RedPhot uses the following ordered stages: ``read``, ``region``, ``masks``,
``cosmic_rays``, ``fringe``, ``background``, ``source_quality``,
``astrometry``, ``star_selection``, ``usability``, ``alignment``, ``psf``,
``science_photometry``, ``calibration``, ``templates``, ``subtraction``,
``difference_photometry``, ``batch_consistency``, and ``outputs``.

Pipeline functions
------------------

``run_one_image`` and ``run_batch`` start new runs. ``run_pipeline_stage``
runs exactly one stage, while ``run_pipeline_through`` runs through a selected
stage. ``resume_pipeline`` continues a saved run. ``review_image``,
``set_image_overrides``, and ``rerun_image`` support review and correction.

States
------

``PASS`` and ``WARN`` are usable automatic results. ``FAIL`` records an error
or unusable image. ``APPROVED`` and ``REJECTED`` are review decisions.
``SKIPPED`` records disabled or intentionally omitted work. ``STALE`` means an
input, setting, dependency, or review decision changed.

Checkpoint safety
-----------------

``pipeline_state.json`` is readable. ``pipeline_context.pkl`` contains Astropy
objects needed for efficient resume and must only be loaded from a trusted
local RedPhot run.

Imports
-------

Functions are imported from their modules rather than re-exported from
``redphot.__init__``.

.. code-block:: python

   from redphot.image import read_fits_image
   from redphot.photometry import perform_science_image_photometry
   from redphot.pipeline import run_batch
