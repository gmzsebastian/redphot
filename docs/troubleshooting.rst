Troubleshooting
===============

No image data found
-------------------

Inspect the HDU list with ``astropy.io.fits.info``. Set ``input.data_hdu`` for
an unusual extension. Confirm that the selected HDU is a numeric 2-D array.

Duplicate metadata warning
--------------------------

Read ``metadata_conflicts`` and ``_sources`` in normalized metadata. RedPhot
does not hide duplicates. If the automatically selected value is wrong, use a
metadata override for that image.

Missing or poor WCS
-------------------

Check the catalog overlay and residual-vector plot. Increase matching tolerance
only when the initial WCS justifies it. Enable plate-solving fallback only when
Astrometry.net is installed and configured.

Too few PSF or calibration stars
--------------------------------

Inspect rejection reasons for saturation, masks, edge distance, crowding,
morphology, or catalog uncertainty. An analytic PSF fallback is expected in a
sparse field. Do not force contaminated stars through safety exclusions merely
to increase the count.

Background damages sources
--------------------------

Increase the mesh size or source-mask growth, protect the host, switch to
``measure_only`` or ``local_only``, or disable broad subtraction for that image.
Review source-preservation and row/column-profile diagnostics.

Hotpants not found or subtraction failed
-----------------------------------------

Confirm that the executable is on ``PATH``. Review the saved command,
parameters, and log. Check template coverage, filter, WCS, depth, saturation,
and pre-transient suitability. A generated difference is not accepted unless
its quality checks pass.

Resume marks products stale
---------------------------

This is expected after an input timestamp/size, relevant setting, dependency,
or review decision changes. Use ``rerun_image`` to rebuild one branch. A
missing checkpoint also marks completed work stale because arrays cannot be
recovered safely from JSON alone.

Output is too large
-------------------

Use the ``minimal`` profile or disable individual products with
``output.product_overrides``. Full-size templates, aligned templates,
background models, masks, and difference images are the largest products.

Warnings from Astropy during tests
----------------------------------

FITS/WCS fix warnings and the current Astropy ``TestRunner`` deprecation are
external warnings. RedPhot metadata or quality warnings should still be read;
they may represent intentional regression cases such as duplicate exposure
cards.
