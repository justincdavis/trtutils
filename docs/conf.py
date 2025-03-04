# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join("..", "src")))
import trtutils

# -- Project information -----------------------------------------------------

project = "trtutils"
copyright = "2024, Justin Davis"
author = "Justin Davis"
version = "0.4.1"

assert version == trtutils.__version__  # Make sure version is consistent

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.todo",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_design",
    "myst_parser",
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "substitution",
]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["**/modules.rst"]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

autodoc_mock_imports = ["tensorrt", "cuda", "cuda-python"]

intersphinx_mapping = {
    "rtd": ("https://docs.readthedocs.io/en/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "cuda": ("https://nvidia.github.io/cuda-python/latest/", None),
    # "tensorrt": ("https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/index.html", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autodoc_default_options = {
    "member-order": "bysource",
    "show-inheritance": True,
    "separator": "---",
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "furo"
html_theme_options = {
    # "light_logo": "logo-light.png",
    # "dark_logo": "logo-dark.png",
    "sidebar_hide_name": False,
    # "sidebar_structure": "toc",
    # "toc_title": "Contents",
}
html_static_path = ["_static"]  # For additional CSS or custom static files.

# Global toctree depth setting
html_show_sourcelink = False
html_title = "trtutils"
html_copy_source = False

# Custom sidebar templates
html_sidebars = {
    "**": ["sidebar/scroll-start.html", "sidebar/brand.html", "sidebar/navigation.html", "sidebar/scroll-end.html"],
}

html_css_files = [
    "center.cs",  # Make sure this path is correct relative to _static
]
