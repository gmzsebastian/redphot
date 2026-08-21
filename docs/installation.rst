.. _installation:

Installation
============

RedPhot requires Python 3.9 or newer. A clean virtual environment is strongly
recommended.

Editable development installation
---------------------------------

.. code-block:: bash

   git clone https://github.com/gmzsebastian/redphot.git
   cd redphot
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -e .

Optional dependencies
---------------------

.. code-block:: bash

   python -m pip install -e '.[cosmic_rays]'
   python -m pip install -e '.[test]'

Hotpants is an external executable needed only when Hotpants subtraction is
enabled. Confirm that it is on ``PATH`` before an automatic subtraction run.

IRAF and PyRAF
--------------

Neither IRAF nor PyRAF is required. Input images must already have detector
reduction such as bias subtraction and flat-fielding.

Quick verification
------------------

.. code-block:: bash

   python -c "from redphot.config import get_default_settings, validate_settings; validate_settings(get_default_settings())"
   pytest -q
