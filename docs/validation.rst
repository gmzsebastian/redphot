Validation and Release Readiness
================================

Automated coverage currently available
--------------------------------------

The offline integration suite covers:

* The available KeplerCam regression image with duplicate exposure metadata,
  including confirmation that its bytes remain unchanged.
* Synthetic LCO single-HDU and tile-compressed ``.fits.fz`` ingestion.
* Independent disabling of cosmic-ray, fringe, broad-background, and source
  detection operations.
* Synthetic trail masking.
* Synthetic poor-seeing/high-noise batch rejection.
* Signed forced flux at a synthetic target nondetection.
* Synthetic passing and failing subtraction-quality results.
* Output profiles and measurement traceability.
* Restart, skip, reject, approve, override, stale propagation, and individual
  rerun behavior.

Synthetic tests verify deterministic software behavior; they do not establish
scientific performance on all observing conditions.

Required real-data validation
-----------------------------

The following items remain mandatory before declaring the first stable
release:

.. list-table:: Release gate
   :header-rows: 1
   :widths: 55 15 30

   * - Validation
     - State
     - Evidence to retain
   * - LCO single-HDU observational image
     - Pending
     - State, metadata, report, output tables
   * - LCO observational ``.fits.fz`` image
     - Pending
     - Selected HDU/WCS and byte checksum
   * - Genuine poor-seeing epoch
     - Pending
     - Quality PDF and configured decision
   * - Genuine cloudy or shallow epoch
     - Pending
     - Recovery, transparency, and depth diagnostics
   * - Genuine trail near target/comparison field
     - Pending
     - Mask and star-rejection diagnostics
   * - Real target nondetection
     - Pending
     - Signed fluxes and 3/5-sigma limits
   * - Passing and failing real subtractions
     - Pending
     - Hotpants/PyZOGY logs and residual diagnostics
   * - Comparison with ``Phot_good.py``
     - Pending
     - Per-image aperture/PSF differences and explanation
   * - Catalog-star aperture and PSF recovery
     - Pending
     - Residual table within expected uncertainty
   * - Visual review of every diagnostic PDF
     - Pending
     - Reviewer, date, and accepted/rejected notes
   * - Complete real multi-filter supernova batch
     - Pending
     - Final light curve and manifest
   * - Repeat from a clean output directory
     - Pending
     - Matching table rows, flags, preferred methods, and numerical tolerances

Repeatability procedure
-----------------------

#. Record the RedPhot version, dependency versions, input checksums, resolved
   configuration, catalog cache, template identifiers, and external subtraction
   executable version.
#. Run the complete batch into an empty output directory.
#. Preserve ``manifest.ecsv``, ``pipeline_state.json``, logs, reports, and core
   ECSV tables.
#. Run again from another empty directory with the same cached external data.
#. Compare image decisions, flags, source IDs, preferred methods, row counts,
   fluxes, uncertainties, zeropoints, limits, and final light curve within
   explicitly recorded tolerances.
#. Have a human inspect every per-image and batch PDF.
#. Mark this checklist complete only after discrepancies are understood.

Release rule
------------

Do not change the package to a stable release or describe it as scientifically
validated until every real-data item above is complete. The absence of a test
image is a pending validation item, not a passing result.
