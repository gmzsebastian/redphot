Algorithms and Measurement Conventions
======================================

This chapter describes the calculations performed by RedPhot. It is intended
to make a result scientifically reviewable, not merely to document function
names. Configuration values quoted here are defaults; the resolved
configuration saved with each run is the authority for a particular result.

Data and uncertainty conventions
--------------------------------

RedPhot treats the input FITS file as already bias- and flat-corrected. The
original file is opened read-only and is never modified. Processing products
are derived arrays with a copied header and provenance links to the input.

Pixel coordinates follow NumPy order ``data[y, x]``. Sky coordinates are ICRS
degrees internally, times are UTC MJD, exposure times are seconds, angular
sizes are arcseconds, and FWHM values are recorded in pixels and arcseconds
where possible. Fluxes retain the unit of the science array (normally ADU or
electrons), including negative measurements. Missing values are represented by
``None``, ``NaN``, or masked table entries rather than numerical sentinels.

Unless an uncertainty plane is supplied, an image variance is estimated from
the local or global background RMS. This empirical RMS already contains sky
and detector/read noise. When gain is known, source Poisson noise is added,

.. math::

   \sigma_{\rm pix}^2 = \sigma_{\rm background}^2 + \max(S,0)/g,

with signal :math:`S` and gain :math:`g`. Existing variance or standard
deviation planes take precedence and are not augmented again. Masked pixels
receive zero statistical weight; they are not silently replaced in scientific
measurements.

Robust location and scatter
---------------------------

Many quality checks use the median and the Gaussian-equivalent median absolute
deviation (MAD),

.. math::

   \hat\sigma_{\rm MAD} = 1.4826\,\mathrm{median}(|x_i-\mathrm{median}(x)|).

Sigma clipping is iterative and configurable. If the MAD collapses to zero,
RedPhot falls back to an ordinary standard deviation where appropriate. This
keeps a few outliers, cosmic rays, or bad detections from defining an entire
image's quality.

FITS selection, metadata, and time
----------------------------------

Astropy reads ``.fits``, ``.fit``, ``.fts``, ``.fits.gz``, and tiled-compressed
``.fits.fz`` files. An explicit HDU setting wins. Otherwise RedPhot examines
preferred HDUs (normally 0, 1, and 2), named science extensions, and then the
remaining extensions for a finite numeric two-dimensional array. If several
arrays are viable, an array whose valid WCS contains the target is preferred.
Associated mask and uncertainty extensions are attached when recognized.

Metadata keywords are searched through configured HDUs and aliases. RedPhot
records the HDU and keyword that supplied each value and reports conflicting
duplicate cards. Telescope pointing, header target coordinates, image WCS
center, user coordinates, and the final fixed target coordinate remain
separate quantities.

Start, midpoint, and end times are reconciled with the exposure. For an
exposure :math:`t_{\rm exp}` beginning at :math:`t_0`,

.. math::

   t_{\rm mid}=t_0+\frac{t_{\rm exp}}{2}, \qquad
   t_{\rm end}=t_0+t_{\rm exp}.

Header times and exposure-derived times are cross-checked. A discrepancy is
flagged rather than hidden; for example, duplicate KeplerCam ``EXPTIME`` cards
remain visible in ``metadata_conflicts`` even when start/end times identify the
usable value.

Processing region and masks
---------------------------

Header data sections and empirical row/column statistics define the usable
detector region. Optional cropping is evaluated in angular units about the
target or image center and translated through the WCS and pixel scale. The WCS
is sliced with the array, so sky coordinates remain valid.

The combined mask is the logical union of non-finite pixels, supplied bad-pixel
masks, invalid edges, saturation and bleed regions, bad rows or columns,
amplifier seams, trails, cosmic rays, and manual regions. Masks are kept as
separate components as well as a union so downstream flags can identify the
cause of an overlap. Target, host, and PSF-star protection regions are tracked
explicitly.

Optional cosmic-ray cleaning uses the L.A.Cosmic implementation in
``astroscrappy``. Its contrast, significance, iteration, gain, read-noise, and
saturation settings are configurable. The cleaned array is a derivative; the
cosmic-ray mask is retained, and protected target/PSF cores are not accepted as
ordinary clean pixels merely because interpolation produced a value.

Fringe correction
-----------------

For configured red filters, a supplied fringe map is aligned to the science
frame and a scalar amplitude is fitted on valid, source-masked pixels. In the
least-squares mode, after subtracting robust centers from science pixels
:math:`D_i` and fringe pixels :math:`F_i`,

.. math::

   a = \frac{\sum_i w_i F_iD_i}{\sum_i w_iF_i^2}, \qquad
   D'_i=D_i-aF_i.

Sigma clipping removes objects and artifacts from the fit. Control-region
pairs can be used instead. The scale and residual scatter are recorded; a
missing or unsuitable map causes a visible skip/failure according to the
configuration, never an unreported correction.

Broad and local background
--------------------------

An expanded segmentation/source mask is constructed before background
measurement, with configurable mask growth and explicit target/host
protection. ``photutils.background.Background2D`` estimates an additive
large-scale model and a background RMS map from sigma-clipped meshes. Mesh
size is required to be substantially broader than the stellar PSF. Meshes with
too many excluded pixels are ignored and the remaining mesh grid is filtered
smoothly before interpolation.

The modes are ``off``, ``measure_only``, ``subtract_broad``, ``local_only``,
and ``broad_plus_local``. Broad subtraction removes only the two-dimensional
instrumental structure. Local sky used by aperture or PSF photometry is
measured later in an annulus and is not double-counted. RedPhot records the
excluded-mesh fraction, residual planar gradient, row/column profiles, and
source-flux preservation statistics.

Detection and image quality
---------------------------

Sources are detected on the prepared image above a configurable multiple of
the background RMS. Photutils segmentation and deblending separate overlapping
islands. Measurements include centroids, flux and S/N, second-moment major and
minor widths, orientation, ellipticity,

.. math::

   e = 1-\frac{b}{a},

and a Gaussian-equivalent FWHM. Robust medians and scatters summarize the
stellar population. Saturated detections, masks, trails, target-local
background, and upstream LCO quality values are assessed separately.

PASS, WARN, and FAIL use both configured absolute limits and deviations from
the batch median. A coherent population of elongated sources indicates a
tracking problem; a single elongated detection is treated as a possible trail
or blend rather than proof of bad tracking.

Catalogs and astrometry
-----------------------

Catalog adapters normalize Gaia, Pan-STARRS1, SDSS, APASS, SkyMapper, and user
tables into common IDs, ICRS coordinates, proper motions, magnitudes, errors,
colors, morphology, and variability fields. Queries may be cached as ECSV.
For Gaia, coordinates are propagated from the catalog reference epoch to each
observation epoch using Astropy space-motion propagation when sufficient
astrometric fields exist.

Catalog and detected sources are paired by unique nearest neighbors within a
configured angular radius. After outlier rejection, WCS refinement fits a
two-dimensional similarity transform,

.. math::

   \begin{bmatrix}x'\\y'\end{bmatrix} =
   s\begin{bmatrix}\cos\theta&-\sin\theta\\
                    \sin\theta& \cos\theta\end{bmatrix}
   \begin{bmatrix}x\\y\end{bmatrix}+
   \begin{bmatrix}t_x\\t_y\end{bmatrix}.

Translation is always the least invasive candidate; rotation and scale are
accepted only with enough well-distributed matches and a justified reduction
in residual RMS. The refined WCS/header is derived metadata: science pixels
are not resampled. Astrometry.net is an optional configured fallback, not a
routine dependency.

Comparison and PSF-star selection
---------------------------------

A master catalog provides persistent IDs across epochs. Catalog-level screens
reject unsuitable magnitude or uncertainty, non-stellar morphology, excessive
proper motion, known variability, crowding, or disallowed color. Per-image
screens then apply edge, saturation/halo, mask/trail, S/N, shape, neighbor, and
detector-region criteria.

Roles are independent: astrometry, PSF, calibration, ensemble comparison, and
bright quality-control anchor. A star rejected for PSF shape can still be an
astrometric match. Candidate scores combine the relevant quality terms, while
grid-based selection prevents all PSF or calibration stars from occupying one
small detector region. Every rejection reason and manual global/per-image
addition or removal is retained.

Usability and limiting depth
----------------------------

The first review gate combines catalog recovery, recovery of a bright QC
anchor, approximate transparency and scatter, seeing, elongation, background,
global/local depth, cloud spatial structure, and target-artifact overlap.
Uniform cloud attenuation appears as a common magnitude residual; spatially
varying attenuation appears as a residual surface across detector position.

For a flux uncertainty :math:`\sigma_F`, exposure :math:`t`, zeropoint
:math:`ZP`, and significance :math:`n`, the quick analytic limit is

.. math::

   m_{n\sigma}=ZP-2.5\log_{10}\left(\frac{n\sigma_F}{t}\right).

Both 3-sigma and 5-sigma limits are stored. An expected transient magnitude is
only a decision aid. Automatic decisions and manual APPROVED/REJECTED states
are recorded without deleting the image.

Relative alignment and the fixed target
---------------------------------------

Common persistent stars refine each image relative to a chosen good reference
using the same robust translation/rotation/scale logic. Direct photometry uses
the resulting WCS on the original pixel grid. Only derived stack inputs are
reprojected.

Per-filter stacks are normalized by exposure and, when available, zeropoint;
masked sigma-clipped combination suppresses cosmic rays and trails. An optional
normalized multi-filter stack is a detection product, not a photometric image.
Target-position candidates can come from a user/discovery prior, good
individual images, and stacks. They are combined with inverse-variance weights
after rejecting large or filter-dependent offsets. Host-dominated centroids
are rejected or down-weighted. The adopted ICRS coordinate, uncertainty,
inputs, and version are frozen for all forced measurements. Free centroids are
diagnostics only.

PSF construction
----------------

Each selected PSF star is cut from the background-subtracted image with masks,
has a local border background removed, is centered, and is normalized to unit
flux. Bad residuals, blends, saturation, and artifacts are removed iteratively.
With enough stars, normalized cutouts are combined into an oversampled
empirical ePSF. Sparse images fall back to a unit-normalized Gaussian or
Moffat profile. A circular Gaussian is

.. math::

   P(r) = A\exp[-r^2/(2\sigma^2)], \qquad
   \mathrm{FWHM}=2\sqrt{2\ln 2}\,\sigma,

while a Moffat profile is

.. math::

   P(r)=A\left[1+(r/\alpha)^2\right]^{-\beta}, \qquad
   \mathrm{FWHM}=2\alpha\sqrt{2^{1/\beta}-1}.

All PSFs satisfy :math:`\sum P=1`. Spatial variation is deliberately disabled
unless a future implementation has adequate distributed stars; requesting it
currently raises a configuration error. Star residual RMS, encircled-energy
profiles, normalization, FWHM, ellipticity, model type, and dependency
signature support the PSF review gate and selective downstream invalidation.

Forced aperture and PSF photometry
----------------------------------

Small and reference apertures are centered at the fixed sky coordinate. Pixel
overlap weights account for fractional aperture boundaries. With local sky
:math:`B`, the aperture flux is

.. math::

   F=\sum_i w_i(D_i-B), \qquad
   \sigma_F^2=\sum_i w_i^2\sigma_i^2 +
   \left(\sum_iw_i\right)^2\sigma_B^2.

The local sky is a sigma-clipped annular estimator; too little clean annular
area or strong structure produces a flag. Masked aperture area is measured and
can invalidate a result rather than being implicitly filled.

For a fixed normalized PSF :math:`P_i`, data :math:`D_i`, background
:math:`B`, and inverse variances :math:`W_i`, the weighted least-squares flux
is

.. math::

   F=\frac{\sum_i W_iP_i(D_i-B)}{\sum_iW_iP_i^2}, \qquad
   \sigma_F=\left(\sum_iW_iP_i^2\right)^{-1/2}.

Target and comparison stars share this schema. Signed flux and uncertainty are
always retained. Free-centroid PSF fits report offsets but never replace the
fixed coordinate. Mask, cosmic-ray, trail, background, fit, and unmasked-area
flags remain attached to each method.

Photometric calibration
-----------------------

Catalog routing maps normalized instrument filters to catalog passbands. For
positive flux :math:`F` and exposure :math:`t`,

.. math::

   m_{\rm inst}=-2.5\log_{10}(F/t), \qquad
   \sigma_m=\frac{2.5}{\ln 10}\frac{\sigma_F}{F}.

Each calibration star gives :math:`ZP_i=m_{\rm cat,i}-m_{\rm inst,i}`.
Method- and image-specific zeropoints are inverse-variance weighted after
iterative robust clipping. Their uncertainty includes the formal weighted
error and observed residual scatter. The calibrated target is

.. math::

   m=m_{\rm inst}+ZP, \qquad
   \sigma_{m,\rm total}=\sqrt{\sigma_m^2+\sigma_{ZP}^2}.

Because each aperture and PSF method receives its own zeropoint, an aperture
correction is measured and reported as a diagnostic but is not applied a
second time. Residual trends with magnitude, color, S/N, detector position,
and airmass expose mismatch or systematic calibration errors. Color-term,
atmospheric-extinction, and Galactic-extinction transformations are not
silently approximated; unsupported requests are rejected by validation.

Final analytic limits use the equation above. Empty-aperture limits place the
same aperture in clean blank positions and use their robust flux scatter,
capturing correlated noise and subtraction residuals. A positive calibrated
flux can be a detection, a low-significance signed measurement a nondetection,
and an invalid fit a measurement failure; these states are distinct.

Templates and image subtraction
--------------------------------

User templates or supported survey tiles must cover the union of science WCS
footprints plus a margin. Tile mosaics and the aligned template are derived
products. Template coverage, filter, depth, seeing, saturation, WCS, and
pre-transient date are checked. The science grid remains fixed and only the
template is resampled.

Robust matched pixels determine a photometric scale and additive background.
For Hotpants, the approximate Gaussian matching width is

.. math::

   \sigma_K=\sqrt{|\sigma_{\rm sci}^2-\sigma_{\rm temp}^2|},

and the sharper image is convolved toward the broader PSF. The default kernel
basis uses Gaussian widths near :math:`0.5\sigma_K`, :math:`\sigma_K`, and
:math:`2\sigma_K`, with polynomial orders decreasing for broader components.
Kernel support, thresholds, masks, and saturation are derived from the measured
seeing and noise but remain configurable. The checked subprocess records its
exact command, parameters, stdout, and stderr. PyZOGY can be selected when its
required variance and PSF inputs and a configured runner are available.

A generated difference is not automatically accepted. Stellar residuals,
positive/negative dipoles, flux conservation, residual background, and
blank-aperture noise determine PASS/WARN/FAIL. Difference photometry then uses
the same fixed coordinate, methods, signed-flux schema, and limits as science
photometry. Preferred-result rules retain all alternatives and explicitly mark
whether host light is included.

Batch consistency and reported results
--------------------------------------

Persistent comparison-star measurements form light curves. Robust scatter
identifies unstable stars and isolated epoch outliers without deleting them.
Time-series tables track zeropoint, depth, seeing, background, and WCS by
filter, telescope, and site. Optional ensemble corrections solve only simple
robust group offsets and preserve both original and corrected values.

Science/difference and aperture/PSF measurements are compared under explicit
priority and quality rules. The final preferred light curve is therefore a
selection view, not a destructive reduction of the measurement table.
Configuration hashes, run IDs, image IDs, layer names, calibration methods,
flags, and source IDs trace each reported number to its input and settings.

Deliberate scope limits
-----------------------

RedPhot favors robust, inspectable reductions over millimagnitude complexity.
It does not perform bias/flat reduction, resample pixels for direct
photometry, silently fit spatial PSFs, apply undeclared color/extinction
transformations, or claim artificial-star completeness. Artificial-star
injection/recovery and optimal extraction remain possible future additions;
configuration validation prevents them from appearing enabled today.
