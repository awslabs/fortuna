import os
import sys

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Fortuna"
copyright = "2022, AWS"
author = "Gianluca Detommaso"

sys.path.insert(0, os.path.abspath("../.."))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx_autodoc_typehints",
    "sphinx.ext.mathjax",
    "nbsphinx",
    "sphinx_gallery.load_style",
    "sphinxawesome_theme",
    "sphinx.ext.viewcode",
    "nbsphinx_link",
]

napoleon_google_docstring = False

templates_path = ["_templates"]
exclude_patterns = []

autodoc_inherit_docstrings = True
autodoc_preserve_defaults = True
autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "undoc-members": True,
}
autoclass_content = "both"

html_logo = "_static/fortuna_symbol_white.png"

nbsphinx_allow_errors = True
nbsphinx_execute = "never"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinxawesome_theme"
html_static_path = ["_static"]
html_css_files = ["sg_gallery.css"]
html_show_sourcelink = False
html_awesome_code_headers = False

html_title = "Fortuna's documentation"

# html_collapsible_definitions = True

pygments_style = "default"
