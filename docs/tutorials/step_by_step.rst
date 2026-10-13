Step by Step
============

RedPhot can be run one step at a time from a terminal, looking at each step's
plots before going on and redoing a step with other settings where needed.
Each of the 19 steps has its own function in :mod:`redphot.steps`, named after
the step and its number: ``read_01``, ``region_02``, ``masks_03``, ...
``outputs_19``. The name is the same as the step's folder of plots,
``<run_directory>/diagnostics/06_background/`` for ``background_06``.

A ready-to-copy version of everything below (one block per step) is the
script ``redphot_simple_steps.py``: paste the settings block, then each step
block in order. Every block has the same form::

   # Step 6: model and subtract the sky background ...
   # what to look at, and which setting to change when it goes wrong
   # e.g. background_06(run, images=[3], box_size=[64, 64])
   background_06(run)

The last line runs the step with the current settings. The ``e.g.`` line is
what to run instead when the plots show a problem.

Images and settings
-------------------

**Images** are picked with ``images=``: their number in the list
:func:`~redphot.steps.start_run` prints (also ``list_images(run)``), the file
name, part of the file name, or a list of those (``images=[3, "aa0629"]``).
Without ``images=`` a change applies to every image.

**Settings** are given as keywords by their name alone:
``box_size=[64, 64]`` in ``background_06`` is ``background.box_size``,
``minimum_snr=20`` in ``star_selection_09`` is
``catalogs.comparison_stars.minimum_snr``. To see every setting a step reads:

.. code-block:: python

   help(star_selection_09)                         # with their defaults
   show_parameters(star_selection_09, run)         # with the values of this run
   show_parameters(star_selection_09, run, 11)     # image 11, with its overrides

A name that the step does not read gives an error that says which step uses
it. A whole section can be given as a dictionary
(``background={"box_size": [64, 64], "filter_size": [5, 5]}``), and
``settings={...}`` takes any nested settings.

``list_steps()`` prints the 19 step functions with what each one does.

What a change redoes
--------------------

``background_06(run, images=..., setting=value)`` stores the new value (for
those images only, or for all), runs step 6 for those images, and marks their
later steps out of date (``~`` in ``show_status(run)``). Run the next steps as
usual afterwards; images whose results did not change are kept ("unchanged,
kept" in the progress lines). With ``redo_later=True`` the step redoes the
later steps right away, up to the last step the run had reached (see
:ref:`change-one-setting`); earlier steps that a previous change left out of
date are redone first, and a note says so:

.. code-block:: python

   star_selection_09(run, images=[11], minimum_snr=20, redo_later=True)
   psf_12(run, images=[4], box_size_pixels=31, redo_later=calibration_14)  # stop there

Steps 9, 10, 11, 14, 15, 18 and 19 work on all images together. They always
run for every image; ``images=`` then only says whose settings change.
``star_selection_09``, ``usability_10`` and ``calibration_14`` can use
different settings for some images. ``alignment_11``, ``templates_15``,
``batch_consistency_18`` and ``outputs_19`` use one set of settings for all
images, so their settings are changed without ``images=``.

When a setting is read by an earlier step than the one called (for example
``maximum_stars`` of the PSF, which star selection uses to pick the PSF
stars), the steps run from that earlier step and a note says so.

``force=True`` runs a step again even though nothing changed, for example
after editing the RedPhot code.

Start
-----

.. code-block:: python

   from pathlib import Path
   from astropy.coordinates import SkyCoord
   from redphot.steps import (start_run, read_01, region_02, masks_03, cosmic_rays_04,
                              fringe_05, background_06, source_quality_07, astrometry_08,
                              star_selection_09, usability_10, alignment_11, psf_12,
                              science_photometry_13, calibration_14, templates_15,
                              subtraction_16, difference_photometry_17,
                              batch_consistency_18, outputs_19, show_parameters,
                              list_images, show_status, reject_images, keep_images)

   INPUT_FILES = Path("AT_2019stc/input_data")      # a folder, "data/*.fits", or a list
   RUN_DIR = Path("AT_2019stc/simple_run")
   TARGET = SkyCoord("06:54:23.103", "+17:29:31.35", unit=("hourangle", "deg"))
   SETTINGS = {
       "masks": {"cosmic_rays": {"enabled": True}},
       "subtraction": {"enabled": True, "cache_directory": "templates"},
       "output": {"profile": "standard"},
   }
   IMAGE_OVERRIDES = {
       "AT2019stc_IMACS.fits": {"metadata": {"instrument_override": "IMACS",
                                             "site_override": "Las Campanas"}},
   }

   run = start_run(INPUT_FILES, RUN_DIR, TARGET, SETTINGS, IMAGE_OVERRIDES)

``start_run`` creates the run, or reopens the one already saved in
``RUN_DIR`` (so the blocks can be pasted again in a new terminal and continue
where the run stopped), and prints the numbered list of images. ``new=True``
renames an existing run folder and starts over. The instrument of each image
comes from its header unless ``IMAGE_OVERRIDES`` (or ``instrument=``) says
otherwise. Keep ``RUN_DIR`` on a local disk: the saved run
(``pipeline_context.pkl``) holds every image array and is rewritten after each
step.

``run`` holds the run's ``state`` and ``context`` (``run.state``,
``run.context``, or ``state, context = run``), for the functions of
:mod:`redphot.pipeline`.

The 19 steps
------------

Defaults are given in brackets.

Step 1 -- read_01
~~~~~~~~~~~~~~~~~

Reads each FITS file: the science array and its HDU, and from the header the
filter, exposure time, start/mid/end times (MJD), airmass, gain, read noise and
saturation level. Header values are cross-checked (duplicate cards, times
against exposure time).

Look at ``01_read``: one row per image. ``DEFAULT`` marks a value that came
from the instrument defaults (no header keyword), ``CONFLICT`` a header value
that disagrees with another one. For known instruments (LDSS3, IMACS,
KeplerCam, Binospec, ...; ``redphot.config.KNOWN_INSTRUMENTS``) the telescope
and site come from the instrument; without a pixel-scale keyword the pixel
scale comes from the WCS.

To change, per image, any header field with ``<field>_override``:
``saturation_override``, ``gain_override``, ``read_noise_override``,
``filter_override``, ``exposure_time_override``, ``instrument_override``,
``site_override``, ``mjd_override``, ... (``help(read_01)`` lists them).

.. code-block:: python

   read_01(run, images=[11], saturation_override=45000)

Step 2 -- region_02
~~~~~~~~~~~~~~~~~~~

Finds the usable part of each image: the header data section, then dead,
ringing or ramping rows and columns at the edges, trimmed to a rectangle.
Optionally crops a box around the target.

Look at ``02_region``: the red hatched strips are cut, the teal box is what
later steps use. Bad rows or columns left inside the box should be trimmed.

To change: ``edge_crop_pixels`` (0) trims a fixed number of pixels more from
every edge; ``size_arcmin`` (none) crops a square of that size around the
target (``center_on``, "target").

.. code-block:: python

   region_02(run, images=[4], edge_crop_pixels=20)

Step 3 -- masks_03
~~~~~~~~~~~~~~~~~~

Masks saturated pixels (grown by ``saturation_grow_pixels``, 2), bad rows and
columns, amplifier seams and satellite trails. Masked pixels are magenta in
every later plot.

Look at ``03_masks``: trails, saturated stars and bad columns should be
covered and nothing else.

To change: ``trail_sigma`` (5) -- lower finds fainter trails;
``trail_min_length_pixels`` (50); ``saturation_level`` (from the header or
step 1) -- the level above which pixels are saturated; ``manual_regions`` --
your own regions to mask.

.. code-block:: python

   masks_03(run, images=[2], trail_sigma=3.5)

Step 4 -- cosmic_rays_04
~~~~~~~~~~~~~~~~~~~~~~~~

Masks cosmic rays with L.A.Cosmic (``astroscrappy``). Off unless
``masks.cosmic_rays.enabled`` is True.

Look at ``04_cosmic_rays``: masked pixels per image. Detections touching the
target core (``target_core_radius_fwhm``, 1 FWHM) are most likely the
transient itself: they are left unmasked, drawn in amber and flagged
``TARGET_COSMIC_RAY``. Hits near the target but off its core are masked like
any other.

To change: ``objlim`` (5) -- raise it when star cores are masked as hits
(sharp seeing); ``sigclip`` (4.5) -- detection threshold; ``enabled``.

.. code-block:: python

   cosmic_rays_04(run, images=[5], objlim=10)

Step 5 -- fringe_05
~~~~~~~~~~~~~~~~~~~

Subtracts a scaled fringe map from red-filter images (``filters``,
``["i", "z"]``). Does nothing unless ``enabled`` and ``map_path`` are set.

.. code-block:: python

   fringe_05(run, images=[9], enabled=True, map_path="fringe_i.fits")

Step 6 -- background_06
~~~~~~~~~~~~~~~~~~~~~~~

Models the sky on a grid of boxes and subtracts it. Sources are masked first;
the brightest stars get a mask that grows outward until their halo is lost in
the sky.

Look at ``06_background``: the residual (image minus model, in units of the
sky RMS) should have a median near 0 and a width near 1, and the model should
have no bumps at bright stars (an RMS map that peaks at a star means its halo
got into the model).

To change: ``box_size`` (``[128, 128]``) -- larger when the model follows
stars, galaxies or halos, smaller when a gradient is left; ``filter_size``
(``[3, 3]``) -- boxes smoothed together; ``bright_star_count`` (5) and
``bright_star_halo_sigma`` (0.25) -- how many bright stars get a halo mask and
where the halo stops (in units of the sky RMS); ``exclude_percentile`` (50) --
boxes more masked than this are filled from their neighbors.

.. code-block:: python

   background_06(run, images=[3], box_size=[64, 64])

Step 7 -- source_quality_07
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Detects the sources and measures seeing (FWHM), ellipticity, background and
the number of saturated sources, and finds the flux above which stars get
broader (non-linear cores); those stars are kept out of the zeropoint and PSF.

Look at ``07_source_quality``: FWHM and ellipticity against their limits, and
the FWHM-versus-flux panel.

To change: ``threshold_sigma`` (5) -- detection threshold; ``fwhm_warn_arcsec``
/ ``fwhm_fail_arcsec`` (4 / 8), ``ellipticity_warn`` ... -- the limits;
``broadening_fraction`` (0.05) -- how much broader counts as broadened.

.. code-block:: python

   source_quality_07(run, threshold_sigma=3.0)

Step 8 -- astrometry_08
~~~~~~~~~~~~~~~~~~~~~~~

Matches the sources to Gaia and checks the WCS; refines it when the match
allows (translation, rotation and scale), then tries SIP distortion terms and
keeps them only when the data clearly need them: enough stars spread over the
frame and around the target, residuals on stars left out of the fit better by
at least 25 % and 0.05 arcsec, and a correction under 10 px everywhere. In
practice that is a few images from cameras with real distortion (LDSS3c,
maybe IMACS), and practically never KeplerCam.

Look at ``08_astrometry``: WCS RMS (aim below 0.5 arcsec), the number of
matched stars, and the residuals with and without distortion. Images with
distortion terms have the flag ``WCS_DISTORTION_FITTED`` and "SIP n" next to
their name in the overview.

To change: ``maximum_match_separation_arcsec`` (5) -- allow larger offsets for
a poor header WCS; ``distortion_mode`` ("auto") -- ``"never"`` or ``"always"``
to override the rule; ``distortion_sip_degrees`` ([2, 3]),
``distortion_minimum_stars`` (30), ``distortion_minimum_improvement_fraction``
(0.25), ``distortion_minimum_improvement_arcsec`` (0.05),
``distortion_maximum_shift_pixels`` (10); ``minimum_matches`` (6).

.. code-block:: python

   astrometry_08(run, images=[12], maximum_match_separation_arcsec=8.0)
   astrometry_08(run, images=[12], distortion_mode="never")

Step 9 -- star_selection_09
~~~~~~~~~~~~~~~~~~~~~~~~~~~

All images together. Matches the detections of every image into one source
list, matches it to the photometric catalog (PS1 for griz), and chooses in each
image the stars for the zeropoint, the PSF, the ensemble and the QC check.
Each rejected star has its reasons listed.

Look at ``09_star_selection``: the number of calibration and PSF stars in
each image and why the others were rejected.

Zeropoint stars closest to S/N 100 are taken first (so fainter stars, like
the science targets, are used rather than only the brightest), and stars whose
quick magnitude is off the catalog (``PHOTOMETRIC_OUTLIER``) are not used.

To change (per image when needed): ``minimum_snr`` (10),
``maximum_magnitude`` / ``minimum_magnitude`` (22 / 10),
``maximum_ellipticity`` (0.35), ``psf_minimum_snr`` (30),
``maximum_calibration_stars`` (200), ``calibration_preferred_snr`` (100),
``photometric_outlier_sigma`` (5), ``maximum_stars`` (20, PSF stars).

.. code-block:: python

   star_selection_09(run, images=[11], minimum_snr=5)

Step 10 -- usability_10
~~~~~~~~~~~~~~~~~~~~~~~

All images together. A quick zeropoint, the depth, the transparency compared
with the other images and the cloud pattern across the field. Images that
FAIL are dropped from here on (they keep everything computed so far).

Look at ``10_usability``: the quick zeropoint per star with the depth lines,
and the PASS/WARN/FAIL reasons.

To drop an image yourself, or bring one back:

.. code-block:: python

   reject_images(run, [4], note="clouds")
   keep_images(run, [4])

To change the limits: ``zeropoint_scatter_fail_mag`` (0.3),
``transparency_attenuation_fail_mag`` (1.5),
``cloud_spatial_amplitude_fail_mag`` (0.35), ``require_qc_anchor`` (True).

.. code-block:: python

   usability_10(run, zeropoint_scatter_fail_mag=0.5)

Step 11 -- alignment_11
~~~~~~~~~~~~~~~~~~~~~~~

All images together, one set of settings. Aligns every image to one reference
image and fixes the target position from the stacked images (forced
photometry is done at that position).

Look at ``11_alignment``: in ``batch.png`` the check stars should sit at the
center of every cutout; ``target_position.png`` shows where the target was
put (the dashed circle has a 0.1 arcsec radius).

To change: ``user_position_mode`` ("prior") -- "fixed" uses ``TARGET``
exactly instead of measuring it; ``centroid_search_radius_arcsec`` (3);
``relative_alignment_fail_rms_arcsec`` (1.5).

.. code-block:: python

   alignment_11(run, user_position_mode="fixed")

Step 12 -- psf_12
~~~~~~~~~~~~~~~~~

Builds the PSF of each image from its step-9 PSF stars. Images whose PSF
FAILs are dropped from here on.

Look at ``12_psf``: the model, the residual of each star and the residual
fraction.

To change: ``box_size_pixels`` (25) -- larger when the model is cut off;
``minimum_star_snr`` (30); ``model`` ("empirical") and ``fallback_model``
("moffat"). The number of PSF stars (``maximum_stars``) is chosen in step 9,
so changing it redoes the run from step 9.

.. code-block:: python

   psf_12(run, images=[4], box_size_pixels=31)

Step 13 -- science_photometry_13
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Forced photometry of the target and the selected stars at the fixed positions:
PSF fit and two apertures, with a local sky annulus. When the forced PSF S/N
of the target is at least 20, the PSF is also fitted with a free centroid
(method ``psf_recentered``, an extra light curve; the forced ones are not
changed); a move of more than 0.1 arcsec raises ``TARGET_RECENTERED`` (WARN)
and the figure gets a third row with the recentered fit.

Look at ``13_science_photometry``: the target magnitude of each method (with
a quick zeropoint from this step's stars; the final ones come from step 14),
data minus PSF model on the same stretch as the data, and the free-centroid
offset.

To change: ``small_radius_fwhm`` / ``large_radius_fwhm`` (1.0 / 2.5) --
aperture radii in units of the FWHM; ``sky_inner_radius_fwhm`` /
``sky_outer_radius_fwhm`` (4 / 7); ``recentered_psf_minimum_snr`` (20),
``recentered_psf_warn_offset_arcsec`` (0.1); ``psf_masked_weight_fraction``
(0.01) -- masked pixels flag the target only when they carry more than this
fraction of the PSF.

.. code-block:: python

   science_photometry_13(run, small_radius_fwhm=1.2)

Step 14 -- calibration_14
~~~~~~~~~~~~~~~~~~~~~~~~~

All images together. Zeropoint of each image and method from its calibration
stars (robust weights, sigma clipping, stars that vary across images removed),
aperture corrections, and limiting magnitudes.

Look at ``14_calibration``: the residual of every star against magnitude and
color, the zeropoint and the star scatter.

To change (per image when needed): ``minimum_star_snr`` (10), ``sigma_clip``
(3), ``maximum_catalog_error_mag`` (0.1), ``maximum_star_rms_mag`` (0.1).

.. code-block:: python

   calibration_14(run, images=[11], minimum_star_snr=20, sigma_clip=2.5)

Step 15 -- templates_15
~~~~~~~~~~~~~~~~~~~~~~~

All images together, one set of settings. Only with
``subtraction.enabled``. Downloads (once, into ``cache_directory``) or loads a
template per filter covering every image.

Look at ``15_templates``: each image's outline should land on real template
data.

To change: ``template_survey_priority`` (``["ps1", "legacy", "sdss"]``),
``template_path`` (your own template files).

.. code-block:: python

   templates_15(run, template_survey_priority=["legacy", "ps1"])

Step 16 -- subtraction_16
~~~~~~~~~~~~~~~~~~~~~~~~~

Registers the template to each image and subtracts it with Hotpants.

Look at ``16_subtraction``: science, template (as subtracted) and difference
on one stretch, the zoom on the target, the star residuals
(``maximum_residual_fraction``, 0.1) and the noise ratio
(``maximum_noise_ratio``, 2). ``save_subtraction=True`` also writes the three
images as FITS files to ``diagnostics/16_subtraction/fits/`` to blink them in
DS9 (without redoing the subtraction).

To change: ``kernel_order`` ("auto") -- 0 for an unstable kernel;
``stamp_count``, ``kernel_radius`` ("auto"); ``registration_order``
("auto").

.. code-block:: python

   subtraction_16(run, images=[11], kernel_order=0)
   subtraction_16(run, save_subtraction=True)

Step 17 -- difference_photometry_17
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Forced photometry of the target on each difference image, calibrated to
magnitudes, with limits from empty apertures.

To change: ``detection_sigma`` (3) -- below this S/N an upper limit is
reported.

.. code-block:: python

   difference_photometry_17(run, detection_sigma=5.0)

Step 18 -- batch_consistency_18
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All images together. Light curves of the comparison stars, comparison of the
photometry methods, outliers, and the final light curve: one measurement per
image, the first valid result of ``preferred_order`` (difference-image PSF,
then the difference-image apertures, then the science-image methods). The
figure shows it above the zeropoint, seeing, depth and WCS RMS of each epoch
on the same MJD axis.

To change: ``comparison_star_rms_warn_mag`` (0.05), ``preferred_order``,
``ensemble_correction`` (off).

.. code-block:: python

   batch_consistency_18(run, ensemble_correction={"enabled": True})

Step 19 -- outputs_19
~~~~~~~~~~~~~~~~~~~~~

Writes the tables, the FITS files of each image (PSF model, difference image,
processed image), the light-curve plot and the PDFs into ``products/``
(``products_v2``, ... when it already exists). See :doc:`../outputs`.

To change: ``profile`` ("standard"; "minimal" or "full").

.. code-block:: python

   outputs_19(run, profile="full")

Afterwards
----------

``show_status(run)`` prints a step-by-image table (``P`` pass, ``W`` warn,
``F`` fail, ``A`` approved, ``R`` rejected, ``s`` skipped, ``~`` out of date).
To change a step for some images and bring the whole run up to date in one go,
add ``redo_later=True`` to the step (see :ref:`change-one-setting`):

.. code-block:: python

   star_selection_09(run, images=[11], minimum_snr=20, redo_later=True)
