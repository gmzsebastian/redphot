__version__ = "0.1.0"
__author__ = "Sebastian Gomez"
__email__ = "sgomez@cfa.harvard.edu"
__license__ = "MIT"
__description__ = "Robust time-domain optical photometry of supernovae"

# Photutils 3.0 renamed the centroid attributes and columns that RedPhot uses
# (SourceCatalog.xcentroid -> x_centroid, DAOStarFinder "xcentroid" ->
# "x_centroid"), with no alias for the old names. setup.py requires 3.0, but
# that is only checked by pip, so a run from a checkout or from another
# environment would otherwise fail midway through the pipeline.
MINIMUM_PHOTUTILS_VERSION = "3.0"


def _check_photutils_version():
    import photutils
    from packaging.version import Version

    found = photutils.__version__
    if Version(found) < Version(MINIMUM_PHOTUTILS_VERSION):
        raise ImportError(
            f"RedPhot requires photutils >= {MINIMUM_PHOTUTILS_VERSION}, "
            f"but photutils {found} is installed. Upgrade it in this "
            "environment with: python -m pip install --upgrade photutils"
        )


_check_photutils_version()
