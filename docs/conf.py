import os
import sys
from importlib.metadata import PackageNotFoundError, version

try:
    release = version("redphot")
except PackageNotFoundError:
    release = "unknown"

__version__ = release

sys.path.insert(0, os.path.abspath('../'))

# -- Project information -----------------------------------------------------

project = 'redphot'
copyright = '2024-2026, Sebastian Gomez'
author = 'Sebastian Gomez'

# The full version, including alpha/beta/rc tags
version = __version__
release = __version__


# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    '_build',
    'Thumbs.db',
    '.DS_Store',
]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# -- Extension configuration -------------------------------------------------

# Napoleon settings to support Google and NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# Additional options for LaTeX output, e.g., for PDF generation.
latex_elements = {
    # Additional stuff for the LaTeX preamble
    'preamble': r'''
        \usepackage{amsmath,amsfonts,amssymb,amsthm}
    ''',
}

# Add a logo to the sidebar
html_logo = 'images/redphot_small.png'
html_favicon = 'images/redphot_small.png'
