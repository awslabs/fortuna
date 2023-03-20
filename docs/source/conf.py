import os
import sys
import shutil

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
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..", "..")))

# Copy over the examples. Function copied from GPyTorch's conf.py
examples_source = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "examples")
)
examples_dest = os.path.abspath(os.path.join(os.path.dirname(__file__), "examples"))
if os.path.exists(examples_dest):
    shutil.rmtree(examples_dest)
os.mkdir(examples_dest)
for root, dirs, files in os.walk(examples_source):
    for dr in dirs:
        os.mkdir(os.path.join(root.replace(examples_source, examples_dest), dr))
    for fil in files:
        if os.path.splitext(fil)[1] in [".py", ".md", ".rst"]:
            source_filename = os.path.join(root, fil)
            dest_filename = source_filename.replace(examples_source, examples_dest)
            # If we're skipping examples, put a dummy file in place
            if os.getenv("SKIP_EXAMPLES"):
                if dest_filename.endswith("index.rst"):
                    shutil.copyfile(source_filename, dest_filename)
                else:
                    with open(os.path.splitext(dest_filename)[0] + ".rst", "w") as f:
                        basename = os.path.splitext(os.path.basename(dest_filename))[0]
                        f.write(f"{basename}\n" + "=" * 80)

            # Otherwise, copy over the real example files
            else:
                shutil.copyfile(source_filename, dest_filename)


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
    "nbsphinx",
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

nb_execution_mode = "auto"
nbsphinx_allow_errors = False
nbsphinx_custom_formats = {
    ".pct.py": ["jupytext.reads", {"fmt": "py:percent"}],
}
nbsphinx_execute_arguments = ["--InlineBackend.figure_formats={'svg', 'pdf'}"]
# If window is narrower than this, input/output prompts are on separate lines:
nbsphinx_responsive_width = "700px"
# List expensive-to-compute notebooks here:
# nb_execution_excludepatterns = ['list', 'of', '*patterns']
# Alias kernel names
nb_kernel_rgx_aliases = {"fortuna": "python3"}


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
