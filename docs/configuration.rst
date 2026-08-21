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

Missing values
--------------

Configuration uses ``None``, floating arrays use ``NaN``, and tables use
masked values. Sentinel numbers such as ``-999`` and ``-1`` are not used for
missing scientific measurements.
