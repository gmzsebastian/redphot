Outputs and Flags
=================

Core tables
-----------

``images.ecsv`` contains one row per input image. Important columns are
``image_id``, ``input_file``, ``object``, ``filter``, ``mjd``, ``telescope``,
``site``, ``instrument``, ``data_hdu``, ``quality_status``, ``failed_stage``,
``user_decision``, 3- and 5-sigma depths, ``flags``, ``config_sha256``, and
``run_id``.

``sources.ecsv`` contains persistent source identifiers, catalog coordinates,
catalog photometry, detector positions, screening results, rejection reasons,
and independent astrometry, PSF, calibration, ensemble, and QC roles.

``photometry.ecsv`` retains every measurement method. Core columns include:

* Input identity: ``image_id``, ``filename``, ``filter``, telescope/site,
  ``mjd_mid``, and exposure time.
* Source identity: ``source_id``, ``source_type``, roles, detector position,
  and sky position.
* Method: ``method``, aperture and sky radii, PSF model/version, fixed-position
  indicator, and coordinate version.
* Measurement: signed ``flux``, ``flux_uncertainty``, ``snr``, local
  background and uncertainty, coverage, unmasked weight, and ``valid``.
* Diagnostics: free-centroid position and offset, FWHM, flags, and uncertainty
  source.
* Calibration: instrumental and calibrated magnitudes, uncertainties,
  zeropoint, catalog/band/system, calibration state, and classification.
* Difference provenance: ``image_kind``, host-light indicator, preferred flag,
  and selection reason where applicable.
* Traceability: ``measurement_id``, ``input_file``, ``input_data_hdu``,
  ``image_layer``, ``config_sha256``, ``run_id``, and
  ``calibration_reference``.

``lightcurve.ecsv`` contains the preferred retained result per epoch while
preserving its source measurement ID, method, image layer, host-light state,
classification, inclusion decision, flags, and traceability fields.


Per-stage diagnostics
---------------------

As soon as each stage runs, RedPhot writes figures and a table into
``<run_directory>/diagnostics/<NN>_<stage>/`` (for example
``06_background``):

* ``<image>.png`` -- one figure per image for image stages, and for star
  selection, usability and calibration. Each figure states what a good result
  looks like ("Check: ...") and lists the measured values next to the limits
  that set PASS/WARN/FAIL.
* ``batch.png`` -- the run-level figure of a batch stage (alignment,
  calibration, batch consistency).
* ``overview.png`` -- every image of the run for that stage, with status
  colors and the key numbers against their limits.
* ``summary.csv`` -- one row per image: status, review state, flags, and the
  same key numbers.
* ``skipped.png`` -- one card when a stage is disabled for every image.

Figures are only redrawn for the entries a call changed; review decisions
refresh the affected figure and ``summary.csv``. Disable with
``{"diagnostics": {"save_stage_plots": False}}``; ``stage_plot_dpi`` sets the
resolution. The per-image PDF reports reuse these PNGs.

Other products
--------------

Depending on the output policy, RedPhot writes per-image diagnostic PDFs, a
batch PDF, FITS derivatives, PSF arrays, templates, differences,
``resolved_config.json``, ``run.log``, and a checksummed ``manifest.ecsv``.

Quality flags
-------------

Input and metadata flags include ``FITS_UNREADABLE``, ``NO_IMAGE_DATA``,
``MULTIPLE_IMAGE_HDUS``, ``METADATA_MISSING``, ``METADATA_CONFLICT``,
``EXPOSURE_TIME_CONFLICT``, ``TIME_CONFLICT``, ``FILTER_UNKNOWN``, and missing
gain/read-noise/saturation flags.

Image and astrometry flags include ``BAD_EDGES``, ``TARGET_NEAR_EDGE``,
``TARGET_MASKED``, ``TRAIL_PRESENT``, ``TARGET_TRAIL``,
``TARGET_COSMIC_RAY``, ``BACKGROUND_UNRELIABLE``, ``BACKGROUND_MESHES_EXCLUDED``,
``FRINGE_CORRECTION_FAILED``, ``WCS_MISSING``, ``WCS_POOR``, and relative
alignment failures.

Quality flags include ``TOO_FEW_SOURCES``, ``SEEING_POOR``, high ellipticity or
background, ``TRACKING_POOR``, ``QUALITY_BATCH_OUTLIER``, low catalog recovery,
nonuniform transparency, zeropoint scatter, and ``IMAGE_TOO_SHALLOW``.

Photometry flags include too few PSF/calibration stars, PSF failures,
incomplete apertures, insufficient unmasked pixels, target-fit or centroid
problems, bad local background, calibration failures, ``NONDETECTION``,
difference-image uncertainty/dipole/inversion flags, unstable comparisons,
batch epoch problems, and measurement outliers.

Subtraction flags cover missing or unsuitable templates, incomplete coverage,
filter/WCS/alignment problems, missing backends, failed subtraction, high
stellar residuals, dipoles, flux loss, and excess difference noise.

Flags never silently delete a row. Use the status, ``valid``, classification,
and final inclusion columns together when filtering results.
