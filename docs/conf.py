import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "AutoGMM"
author = "Tingshan Liu"
extensions = ["sphinx.ext.autodoc", "numpydoc"]
html_theme = "sphinx_rtd_theme"
autodoc_default_options = {"members": True, "inherited-members": True}
