.. _installation:

Installation
============

RedPhot requires Python 3.9 or newer. A clean virtual environment is strongly
recommended, particularly when NumPy, Photutils, or scikit-image are already
installed in a system or Anaconda environment with incompatible binary builds.

Editable development installation
---------------------------------

.. code-block:: bash

   git clone https://github.com/gmzsebastian/redphot.git
   cd redphot
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -e .

For a non-editable installation from a local checkout, replace the last line
with ``python -m pip install .``.

Python dependencies
-------------------

The normal installation brings in:

* NumPy and SciPy for array operations, fitting, interpolation, and filtering.
* Astropy for FITS, WCS, coordinates, times, units, CCD containers, and ECSV.
* Photutils for segmentation, deblending, two-dimensional backgrounds, and
  aperture utilities.
* scikit-image for Photutils' deblending support.
* Astroquery for optional remote catalog and survey access.
* Matplotlib for diagnostic figures and PDF reports.

Catalog queries and survey downloads need network access when a matching cache
file is not already present. Direct FITS processing remains local.

Optional dependencies
---------------------

.. code-block:: bash

   python -m pip install -e '.[cosmic_rays]'
   python -m pip install -e '.[test]'

``astroscrappy`` supplies optional L.A.Cosmic cleaning. Tests use pytest. These
extras are not needed for a run in which the corresponding feature is off.

Installing Hotpants
-------------------

`Hotpants <https://github.com/acbecker/hotpants>`_ is an external C program,
not a Python package. It is needed only when
``subtraction.enabled`` is true and ``subtraction.method`` is ``hotpants``.
It requires a C compiler, ``make``, and the CFITSIO development library.

On Ubuntu or Debian, install the build requirements with:

.. code-block:: bash

   sudo apt-get update
   sudo apt-get install build-essential git libcfitsio-dev

On macOS, install Apple's command-line tools and CFITSIO with Homebrew:

.. code-block:: bash

   xcode-select --install
   brew install cfitsio

Then obtain the official source and compile it. The explicit compile command
below avoids machine-specific paths embedded in the upstream Makefile:

.. code-block:: bash

   git clone https://github.com/acbecker/hotpants.git
   cd hotpants
   cc -O3 -Wall -D_GNU_SOURCE -std=c99 \
      main.c functions.c vargs.c alard.c -lcfitsio -lm -o hotpants

If CFITSIO is in a nonstandard location, add its include and library paths. On
Homebrew macOS, for example:

.. code-block:: bash

   CFITSIO_PREFIX="$(brew --prefix cfitsio)"
   cc -O3 -Wall -D_GNU_SOURCE -std=c99 \
      -I"${CFITSIO_PREFIX}/include" -L"${CFITSIO_PREFIX}/lib" \
      main.c functions.c vargs.c alard.c -lcfitsio -lm -o hotpants

Place the resulting executable on ``PATH`` or configure its absolute path:

.. code-block:: bash

   mkdir -p "${HOME}/.local/bin"
   cp hotpants "${HOME}/.local/bin/hotpants"
   export PATH="${HOME}/.local/bin:${PATH}"
   command -v hotpants

.. code-block:: python

   run_settings = {
       "subtraction": {
           "enabled": True,
           "method": "hotpants",
           "hotpants_executable": "/absolute/path/to/hotpants",
       },
   }

The upstream repository includes a Makefile as an alternative; set its CFITSIO
include/library locations for the local installation before running ``make``.
RedPhot executes Hotpants as a checked subprocess and preserves its command,
parameters, standard output, and standard error.

Other external options
----------------------

`Astrometry.net <https://astrometry.net/>`_ is needed only for explicitly
enabled plate-solving fallback.
Install its command-line tools and index files using the Astrometry.net
instructions for the operating system, then configure the executable and index
location. Normal WCS verification/refinement does not require it.

PyZOGY is an optional subtraction backend. It is intentionally not installed as
a hard dependency: configure a compatible runner and provide the required
science/template variance and PSF information before selecting it.

IRAF and PyRAF
--------------

Neither IRAF nor PyRAF is required. Input images must already have detector
reduction such as bias subtraction and flat-fielding.

Quick verification
------------------

.. code-block:: bash

   python -c "from redphot.config import get_default_settings, validate_settings; validate_settings(get_default_settings())"
   pytest -q

Building the documentation
--------------------------

.. code-block:: bash

   python -m pip install -r docs/rtd-pip-requirements
   make -C docs html

Open ``docs/_build/html/index.html`` after the build completes. Warnings are
treated as documentation defects during release validation.
