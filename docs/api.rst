Function API
============

RedPhot intentionally exposes functions from their defining modules rather
than building a large top-level namespace. The most useful entry points are
``redphot.pipeline.run_batch`` (everything in one call) and ``redphot.steps``
(one step at a time, and changing a step for chosen images); lower-level
functions support interactive review and custom workflows.

Step by step
------------

.. automodule:: redphot.steps
   :members:
   :member-order: bysource

Configuration
-------------

.. automodule:: redphot.config
   :members:
   :member-order: bysource

FITS images, preparation, and quality
-------------------------------------

.. automodule:: redphot.image
   :members:
   :member-order: bysource

Metadata
--------

.. automodule:: redphot.metadata
   :members:
   :member-order: bysource

Catalogs, astrometry, and star selection
----------------------------------------

.. automodule:: redphot.catalogs
   :members:
   :member-order: bysource

Alignment and target position
-----------------------------

.. automodule:: redphot.alignment
   :members:
   :member-order: bysource

PSF, photometry, calibration, and limits
----------------------------------------

.. automodule:: redphot.photometry
   :members:
   :member-order: bysource

Templates and subtraction
-------------------------

.. automodule:: redphot.subtraction
   :members:
   :member-order: bysource

Pipeline and batch checks
-------------------------

.. automodule:: redphot.pipeline
   :members:
   :member-order: bysource

Diagnostics
-----------

.. automodule:: redphot.diagnostics
   :members:
   :member-order: bysource

Per-stage diagnostic files
--------------------------

.. automodule:: redphot.stage_reports
   :members:
   :member-order: bysource

Output products
---------------

.. automodule:: redphot.output
   :members:
   :member-order: bysource

Progress messages
-----------------

.. automodule:: redphot.progress
   :members:
   :member-order: bysource
