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
    "sphinx.ext.viewcode",
    "nbsphinx_link",
    "IPython.sphinxext.ipython_console_highlighting",
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

nbsphinx_allow_errors = True
nbsphinx_execute = "never"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_title = "Fortuna's documentation"
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_sidebars = {"**": ["sidebar-nav-bs"]}
html_theme_options = {
    "primary_sidebar_end": [],
    "footer_items": ["copyright"],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/awslabs/Fortuna",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        }
    ],
    "use_edit_page_button": False,
    "collapse_navigation": True,
    "logo": {
        "image_light": "fortuna_symbol.png",
        "image_dark": "fortuna_symbol_white.png",
        "text": html_title,
        "alt_text": "Fortuna's logo",
    },
}
html_context = {
    "github_user": "awslabs",
    "github_repo": "Fortuna",
    "github_version": "dev",
    "doc_path": "docs",
    "default_mode": "light",
}
htmlhelp_basename = "Fortuna's documentation"
html_show_sourcelink = False
