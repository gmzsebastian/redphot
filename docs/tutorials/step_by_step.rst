Step by Step
============

RedPhot can be run one step at a time from a terminal, looking at each step's
plots before going on and redoing a step with other settings where needed.
The functions for this are in :mod:`redphot.steps`. A ready-to-copy version of
everything below (one block per step) is the script
``redphot_simple_steps.py``: paste the settings block, then each step block in
order.

Every block has the same form::

   # Step 6: model and subtract the sky background ...
   # what to look at, and which setting to change when it goes wrong
   # e.g. run_step(state, context, 6, images=[3], box_size=[64, 64])
   run_step(state, context, 6)

The last line runs the step with the current settings. The ``e.g.`` line is
what to run instead when the plots show a problem.

How steps, images and settings are named
----------------------------------------

**Steps** are numbered 1-19, the numbers of the
``<run_directory>/diagnostics/<NN>_<step>/`` folders. A step can also be given
by name: ``run_step(state, context, "background")`` is step 6.
``list_steps()`` prints all of them.

**Images** are picked with ``images=``: their number in the list
:func:`~redphot.steps.start_run` prints (also ``list_images(state, context)``),
the file name, part of the file name, or a list of those
(``images=[3, "aa0629"]``). Without ``images=`` a change applies to every
image.

**Settings** are given as keywords by their name alone:
``box_size=[64, 64]`` in step 6 is ``background.box_size``, ``minimum_snr=20``
in step 9 is ``catalogs.comparison_stars.minimum_snr``. To see every setting a
step reads and its current value:

.. code-block:: python

   show_parameters(9)                        # defaults
   show_parameters(9, state, context)        # this run
   show_parameters(9, state, context, 11)    # image 11, with its overrides

A name that the step does not read gives an error that says which step uses
it. A whole section can be given as a dictionary
(``background={"box_size": [64, 64], "filter_size": [5, 5]}``), and
``settings={...}`` takes any nested settings.

What a change redoes
--------------------

``run_step(state, context, N, images=..., setting=value)`` stores the new
value (for those images only, or for all), runs step N for those images, and
marks their later steps out of date (``~`` in ``show_status(state)``). Run the
next steps as usual afterwards; images whose results did not change are kept
("unchanged, kept" in the progress lines).

Steps 9, 10, 11, 14, 15, 18 and 19 work on all images together. They always
run for every image; ``images=`` then only says whose settings change. Star
selection (9), usability (10) and calibration (14) can use different settings
for some images. Alignment (11), templates (15), batch consistency (18) and
outputs (19) use one set of settings for all images, so their settings are
changed without ``images=``.

When a setting is read by an earlier step than the one named (for example
``maximum_stars`` of the PSF, which star selection uses to pick the PSF
stars), the steps run from that earlier step and a note says so.

``run_step(..., force=True)`` runs a step again even though nothing changed,
for example after editing the RedPhot code.

Start
-----

.. code-block:: python

   from pathlib import Path
   from astropy.coordinates import SkyCoord
   from redphot.steps import (start_run, run_step, rerun_from, show_parameters,
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

   state, context = start_run(INPUT_FILES, RUN_DIR, TARGET, SETTINGS, IMAGE_OVERRIDES)

``start_run`` creates the run, or reopens the one already saved in
``RUN_DIR`` (so the blocks can be pasted again in a new terminal and continue
where the run stopped). ``new=True`` renames an existing run folder and starts
over. The instrument of each image comes from its header unless
``IMAGE_OVERRIDES`` (or ``instrument=``) says otherwise. Keep ``RUN_DIR`` on a
local disk: the saved run (``pipeline_context.pkl``) holds every image array
and is rewritten after each step.

The 19 steps
------------

Defaults are given in brackets.

Step 1 -- read
~~~~~~~~~~~~~~

Reads each FITS file: the science array and its HDU, and from the header the
filter, exposure time, start/mid/end times (MJD), airmass, gain, read noise and
saturation level. Header values are cross-checked (duplicate cards, times
against exposure time).

Look at ``01_read``: one row per image. ``DEFAULT`` marks a value that came
from the instrument defaults (no header keyword), ``CONFLICT`` a header value
that disagrees with another one.

To change, per image, any header field with ``<field>_override``:
``saturation_override``, ``gain_override``, ``read_noise_override``,
``filter_override``, ``exposure_time_override``, ``instrument_override``,
``site_override``, ``mjd_override``, ... (``show_parameters(1)`` lists them).

.. code-block:: python

   run_step(state, context, 1, images=[11], saturation_override=45000)

Step 2 -- region
~~~~~~~~~~~~~~~~

Finds the usable part of each image: the header data section, then dead,
ringing or ramping rows and columns at the edges, trimmed to a rectangle.
Optionally crops a box around the target.

Look at ``02_region``: the red hatched strips are cut, the teal box is what
later steps use. Bad rows or columns left inside the box should be trimmed.

To change: ``edge_crop_pixels`` (0) trims a fixed number of pixels more from
every edge; ``size_arcmin`` (none) crops a square of that size around the
target (``center_on``, "target").

.. code-block:: python

   run_step(state, context, 2, images=[4], edge_crop_pixels=20)

Step 3 -- masks
~~~~~~~~~~~~~~~

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

   run_step(state, context, 3, images=[2], trail_sigma=3.5)

Step 4 -- cosmic rays
~~~~~~~~~~~~~~~~~~~~~

Masks cosmic rays with L.A.Cosmic (``astroscrappy``). Off unless
``masks.cosmic_rays.enabled`` is True.

Look at ``04_cosmic_rays``: masked pixels per image; hits on the target are
flagged ``TARGET_COSMIC_RAY``.

To change: ``objlim`` (5) -- raise it when star cores are masked as hits
(sharp seeing); ``sigclip`` (4.5) -- detection threshold; ``enabled``.

.. code-block:: python

   run_step(state, context, 4, images=[5], objlim=10)

Step 5 -- fringe
~~~~~~~~~~~~~~~~

Subtracts a scaled fringe map from red-filter images (``filters``,
``["i", "z"]``). Does nothing unless ``enabled`` and ``map_path`` are set.

.. code-block:: python

   run_step(state, context, 5, images=[9], enabled=True, map_path="fringe_i.fits")

Step 6 -- background
~~~~~~~~~~~~~~~~~~~~

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

   run_step(state, context, 6, images=[3], box_size=[64, 64])

Step 7 -- sources and seeing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Detects the sources and measures seeing (FWHM), ellipticity, background and
the number of saturated sources, and finds the flux above which stars get
broader (non-linear cores); those stars are kept out of the zeropoint and PSF.

Look at ``07_source_quality``: FWHM and ellipticity against their limits, and
the FWHM-versus-flux panel.

To change: ``threshold_sigma`` (5) -- detection threshold; ``fwhm_warn_arcsec``
/ ``fwhm_fail_arcsec`` (4 / 8), ``ellipticity_warn`` ... -- the limits;
``broadening_fraction`` (0.05) -- how much broader counts as broadened.

.. code-block:: python

   run_step(state, context, 7, threshold_sigma=3.0)

Step 8 -- astrometry
~~~~~~~~~~~~~~~~~~~~

Matches the sources to Gaia and checks the WCS; refines it when the match
allows.

Look at ``08_astrometry``: WCS RMS (aim below 0.5 arcsec) and the number of
matched stars.

To change: ``maximum_match_separation_arcsec`` (5) -- allow larger offsets for
a poor header WCS; ``fit_distortion`` (False) -- fit distortion terms;
``minimum_matches`` (6).

.. code-block:: python

   run_step(state, context, 8, images=[12], maximum_match_separation_arcsec=8.0)

Step 9 -- star selection
~~~~~~~~~~~~~~~~~~~~~~~~

All images together. Matches the detections of every image into one source
list, matches it to the photometric catalog (PS1 for griz), and chooses in each
image the stars for the zeropoint, the PSF, the ensemble and the QC check.
Each rejected star has its reasons listed.

Look at ``09_star_selection``: the number of calibration and PSF stars in
each image and why the others were rejected.

To change (per image when needed): ``minimum_snr`` (10),
``maximum_magnitude`` / ``minimum_magnitude`` (22 / 10),
``maximum_ellipticity`` (0.35), ``psf_minimum_snr`` (30),
``maximum_calibration_stars`` (200), ``maximum_stars`` (20, PSF stars).

.. code-block:: python

   run_step(state, context, 9, images=[11], minimum_snr=5)

Step 10 -- usability
~~~~~~~~~~~~~~~~~~~~

All images together. A quick zeropoint, the depth, the transparency compared
with the other images and the cloud pattern across the field. Images that
FAIL are dropped from here on (they keep everything computed so far).

Look at ``10_usability``: the quick zeropoint per star with the depth lines,
and the PASS/WARN/FAIL reasons.

To drop an image yourself, or bring one back:

.. code-block:: python

   reject_images(state, context, [4], note="clouds")
   keep_images(state, context, [4])

To change the limits: ``zeropoint_scatter_fail_mag`` (0.3),
``transparency_attenuation_fail_mag`` (1.5),
``cloud_spatial_amplitude_fail_mag`` (0.35), ``require_qc_anchor`` (True).

.. code-block:: python

   run_step(state, context, 10, zeropoint_scatter_fail_mag=0.5)

Step 11 -- alignment
~~~~~~~~~~~~~~~~~~~~

All images together, one set of settings. Aligns every image to one reference
image and fixes the target position from the stacked images (forced
photometry is done at that position).

Look at ``11_alignment``: in ``batch.png`` the check stars should sit at the
center of every cutout; ``target_position.png`` shows where the target was
put.

To change: ``user_position_mode`` ("prior") -- "fixed" uses ``TARGET``
exactly instead of measuring it; ``centroid_search_radius_arcsec`` (3);
``relative_alignment_fail_rms_arcsec`` (1.5).

.. code-block:: python

   run_step(state, context, 11, user_position_mode="fixed")

Step 12 -- PSF
~~~~~~~~~~~~~~

Builds the PSF of each image from its step-9 PSF stars. Images whose PSF
FAILs are dropped from here on.

Look at ``12_psf``: the model, the residual of each star and the residual
fraction.

To change: ``box_size_pixels`` (25) -- larger when the model is cut off;
``minimum_star_snr`` (30); ``model`` ("empirical") and ``fallback_model``
("moffat"). The number of PSF stars (``maximum_stars``) is chosen in step 9,
so changing it redoes the run from step 9.

.. code-block:: python

   run_step(state, context, 12, images=[4], box_size_pixels=31)

Step 13 -- science photometry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Forced photometry of the target and the selected stars at the fixed positions:
PSF fit and two apertures, with a local sky annulus.

Look at ``13_science_photometry``: target S/N, and the free-centroid offset
(how far a free fit would move from the fixed position).

To change: ``small_radius_fwhm`` / ``large_radius_fwhm`` (1.0 / 2.5) --
aperture radii in units of the FWHM; ``sky_inner_radius_fwhm`` /
``sky_outer_radius_fwhm`` (4 / 7).

.. code-block:: python

   run_step(state, context, 13, small_radius_fwhm=1.2)

Step 14 -- calibration
~~~~~~~~~~~~~~~~~~~~~~

All images together. Zeropoint of each image and method from its calibration
stars (robust weights, sigma clipping, stars that vary across images removed),
aperture corrections, and limiting magnitudes.

Look at ``14_calibration``: the residual of every star against magnitude and
color, the zeropoint and the star scatter.

To change (per image when needed): ``minimum_star_snr`` (10), ``sigma_clip``
(3), ``maximum_catalog_error_mag`` (0.1), ``maximum_star_rms_mag`` (0.1).

.. code-block:: python

   run_step(state, context, 14, images=[11], minimum_star_snr=20, sigma_clip=2.5)

Step 15 -- templates
~~~~~~~~~~~~~~~~~~~~

All images together, one set of settings. Only with
``subtraction.enabled``. Downloads (once, into ``cache_directory``) or loads a
template per filter covering every image.

Look at ``15_templates``: each image's outline should land on real template
data.

To change: ``template_survey_priority`` (``["ps1", "legacy", "sdss"]``),
``template_path`` (your own template files).

.. code-block:: python

   run_step(state, context, 15, template_survey_priority=["legacy", "ps1"])

Step 16 -- subtraction
~~~~~~~~~~~~~~~~~~~~~~

Registers the template to each image and subtracts it with Hotpants.

Look at ``16_subtraction``: the zoom on the target, the star residuals
(``maximum_residual_fraction``, 0.1) and the noise ratio
(``maximum_noise_ratio``, 2).

To change: ``kernel_order`` ("auto") -- 0 for an unstable kernel;
``stamp_count``, ``kernel_radius`` ("auto"); ``registration_order``
("auto").

.. code-block:: python

   run_step(state, context, 16, images=[11], kernel_order=0)

Step 17 -- difference photometry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Forced photometry of the target on each difference image, calibrated to
magnitudes, with limits from empty apertures.

To change: ``detection_sigma`` (3) -- below this S/N an upper limit is
reported.

.. code-block:: python

   run_step(state, context, 17, detection_sigma=5.0)

Step 18 -- batch consistency
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All images together. Light curves of the comparison stars, comparison of the
photometry methods, outliers, and the preferred light curve.

To change: ``comparison_star_rms_warn_mag`` (0.05), ``preferred_order``,
``ensemble_correction`` (off).

.. code-block:: python

   run_step(state, context, 18, ensemble_correction={"enabled": True})

Step 19 -- outputs
~~~~~~~~~~~~~~~~~~

Writes the tables, the FITS files of each image (PSF model, difference image,
processed image), the light-curve plot and the PDFs into ``products/``
(``products_v2``, ... when it already exists). See :doc:`../outputs`.

To change: ``profile`` ("standard"; "minimal" or "full").

.. code-block:: python

   run_step(state, context, 19, profile="full")

Afterwards
----------

``show_status(state)`` prints a step-by-image table (``P`` pass, ``W`` warn,
``F`` fail, ``A`` approved, ``R`` rejected, ``s`` skipped, ``~`` out of date).
To change a step for some images and bring the whole run up to date in one go,
use :func:`~redphot.steps.rerun_from` (see
:ref:`change-one-setting`):

.. code-block:: python

   rerun_from(state, context, 9, images=[11], minimum_snr=20)
