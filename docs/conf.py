# Configuration file for the Sphinx documentation builder.

import os
import sys
sys.path.insert(0, os.path.abspath(".."))  # Ensure Python files are found

# -- Project information -----------------------------------------------------
project = 'My Project'
author = 'Your Name'
release = '1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    "nbsphinx",  # Enables Jupyter notebooks
    "sphinx.ext.mathjax",  # LaTeX support for equations
    "sphinx.ext.autodoc",  # Auto-generate documentation from docstrings
    "sphinx.ext.napoleon",  # Support Google-style docstrings
]

# Ensure notebooks are never executed (change to "always" if needed)
nbsphinx_execute = "never"

# Exclude temporary or unnecessary files
exclude_patterns = ["_build", "**.ipynb_checkpoints"]

# -- HTML output -------------------------------------------------------------
html_theme = "sphinx_rtd_theme"  # Read the Docs theme

# -- Options for LaTeX (PDF) output ------------------------------------------
latex_elements = {
    'papersize': 'a4paper',
    'pointsize': '10pt',
}
