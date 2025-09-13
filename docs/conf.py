import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "AutoGMM"
author = "Tingshan Liu"
extensions = ["sphinx.ext.autodoc", "numpydoc"]
html_theme = "sphinx_rtd_theme"
autodoc_default_options = {"members": True, "inherited-members": True}

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "numpydoc",
]

autosummary_generate = True

numpydoc_show_class_members = False
numpydoc_class_members_toctree = False

suppress_warnings = ["ref.ref"]
