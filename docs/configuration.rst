Configuration
=============

RedPhot settings are ordinary nested dictionaries. The precedence order is:

#. General defaults.
#. Instrument defaults.
#. Run settings.
#. Filter settings.
#. Individual-image overrides.

The result returned by ``resolve_settings`` is an independent dictionary.
Changing one image's settings does not modify defaults or another image.
In ``run_batch``, the FITS header supplies ``filter_name`` after ingestion, so
built-in and user ``filter_settings`` are applied automatically per image.

.. code-block:: python

   from redphot.config import resolve_settings

   image_overrides = {
       "difficult_epoch.fits": {
           "background": {"sigma_clip": 4.0},
           "psf": {"maximum_stars": 12},
       },
   }

   settings = resolve_settings(
       instrument_name="LCO",
       run_settings={"crop": {"size_arcmin": 12.0}},
       filter_name="i",
       image_name="difficult_epoch.fits",
       image_overrides=image_overrides,
   )

For a mixed-filter batch, pass filter settings separately so the controller can
route them using each normalized FITS filter:

.. code-block:: python

   from redphot.pipeline import run_batch

   state, context = run_batch(
       "data/*.fits*",
       instrument_name="LCO",
       settings={"crop": {"size_arcmin": 12.0}},
       filter_settings={
           "i": {"fringe": {"enabled": True, "map_path": "fringe_i.fits"}},
           "z": {"fringe": {"enabled": True, "map_path": "fringe_z.fits"}},
       },
       image_overrides=image_overrides,
       target=target,
       run_directory="AT2024rmj_run",
   )

Settings sections
-----------------

The main sections are deliberately aligned with the processing order:

``instrument``
   Instrument identity and detector fallbacks such as gain, read noise,
   saturation, nonlinearity, and nominal pixel scale.

``input`` and ``metadata``
   FITS suffixes, HDU selectors/search order, extension aliases, header keyword
   aliases, required values, time interpretation, and metadata conflict limits.

``crop``
   Angular processing size, center choice, valid-data sections, empirical edge
   detection, edge growth, and target-edge safety margin.

``masks``
   Input bad pixels, saturation/nonlinearity and halos, bad lines, amplifier
   seams, trail detection, manual regions, target protection, and optional
   L.A.Cosmic settings.

``fringe``
   Enabled filters, fringe-map paths, alignment mode, scale estimator,
   clipping, source masking, and accepted scale range.

``background``
   Processing mode, Background2D mesh/filter sizes, estimator, RMS estimator,
   clipping, excluded-pixel threshold, mask growth, target/host protection, and
   gradient/source-preservation checks.

``source_detection`` and ``image_quality``
   Detection/deblending thresholds and PASS/WARN/FAIL limits for source count,
   seeing, ellipticity, background, masking, trails, saturation, target-local
   conditions, upstream comparisons, and batch-relative deviations.

``catalogs`` and ``astrometry``
   Catalog choice/routing, query radius and cache, local-catalog mapping, epoch
   propagation, matching, WCS residual limits, allowed refinement terms, and
   optional plate-solving fallback.

``comparison_stars``
   Master-catalog and per-image screening, role-specific magnitude/S/N limits,
   crowding and artifact limits, spatial distribution, manual additions and
   removals, and usability thresholds.

``target_position``
   User/discovery prior, relative alignment, stack selection/normalization,
   host and filter-shift protection, centroid acceptance, and coordinate
   versioning.

``psf`` and ``apertures``
   PSF model/fallback, star cutouts, rejection, oversampling, review gate,
   aperture radii, annular sky, fractional pixels, mask coverage, centroid
   diagnostics, and uncertainty handling. Spatially varying PSFs and optimal
   extraction are rejected because they are not implemented.

``calibration`` and ``upper_limits``
   Catalog passband routing, calibration-star clipping, method-specific
   zeropoints, diagnostic aperture corrections and residual trends, detection
   classification, analytic limits, and empty-aperture sampling. Unsupported
   color/extinction corrections and injection/recovery are rejected rather
   than silently ignored.

``subtraction``
   Template source/path or survey, coverage margin and suitability limits,
   template resampling, scale/background matching, Hotpants/PyZOGY selection,
   kernel and executable options, subtraction validation, difference
   photometry, and preferred-result rules.

``batch_consistency``
   Comparison-star stability, isolated outliers, epoch/telescope/filter/site
   trends, optional simple ensemble offsets, and final-method priority.

``pipeline``
   Review gates, automatic/stepwise behavior, state saving, exception
   containment, and resume policy.

``diagnostics`` and ``output``
   Plot/report switches, output directory/overwrite behavior, storage profile,
   per-product overrides, tables, derivatives, logs, and manifest checksums.

Call ``validate_settings(settings)`` after constructing settings manually.
``resolve_settings`` performs validation automatically.

Changing a running image
------------------------

Use ``set_image_overrides`` after a review. The controller stores the change
and marks the affected stage and downstream dependencies ``STALE``.

.. code-block:: python

   from redphot.pipeline import set_image_overrides

   set_image_overrides(
       state,
       context,
       "difficult_epoch.fits",
       {"background": {"box_size": [96, 96]}},
   )

Optional stages
---------------

.. code-block:: python

   run_settings = {
       "masks": {"cosmic_rays": {"enabled": False, "mode": "off"}},
       "fringe": {"enabled": False},
       "background": {"enabled": False, "mode": "off"},
       "source_detection": {"enabled": True},
       "subtraction": {"enabled": False},
   }

Disabling a stage does not edit the input image. A disabled stage is recorded
as ``SKIPPED`` and later stages can continue when that operation is optional.

Output profiles
---------------

``minimal`` writes core tables, configuration, log, and manifest. ``standard``
adds reports, PSFs, and differences. ``full`` writes every supplied derivative.
``custom`` uses ``output.products`` exactly. ``output.product_overrides`` is
applied last under any profile.

For example, retain tables and reports but omit the largest pixel arrays:

.. code-block:: python

   run_settings = {
       "output": {
           "profile": "standard",
           "product_overrides": {
               "background_model": False,
               "aligned_template": False,
               "difference_image": False,
               "image_pdfs": True,
           },
       },
   }

Missing values
--------------

Configuration uses ``None``, floating arrays use ``NaN``, and tables use
masked values. Sentinel numbers such as ``-999`` and ``-1`` are not used for
missing scientific measurements.
