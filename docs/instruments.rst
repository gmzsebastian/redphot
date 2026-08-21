LCO and KeplerCam
=================

LCO
---

RedPhot accepts ordinary LCO FITS files and tile-compressed ``.fits.fz``
files. It searches named science extensions such as ``SCI`` and then numeric
two-dimensional image HDUs. LCO metadata aliases include telescope, site,
camera, exposure, filter, gain, noise, saturation, reduction level, and
upstream image-quality fields.

LCO images are often large. ``crop.size_arcmin`` limits the processing region
without changing the input file. The default LCO profile expects reduced data
and uses the selected science extension's WCS.

KeplerCam
---------

KeplerCam files commonly store science pixels in the primary HDU with extension
labels such as ``IM1`` or ``IM2``. RedPhot recognizes KeplerCam/FLWO aliases,
uses the configured gain/read-noise/saturation fallbacks when needed, and
normalizes Sloan filter labels.

Some KeplerCam headers contain duplicate cards. RedPhot records every duplicate
value and its HDU, warns on conflicts, and cross-checks exposure time against
start/end timestamps. Telescope pointing, header target coordinates, WCS
center, user coordinates, and the final frozen target position remain separate.

Both instruments
----------------

Science pixels may be found in HDU 0, 1, 2, or a later valid extension. An
explicit HDU override always wins. The selected data, header, WCS, mask, and
variance HDUs are recorded. WCS refinement changes only a derived WCS/header;
direct-photometry science pixels are not resampled.
