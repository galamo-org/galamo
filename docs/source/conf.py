# Configuration file for the Sphinx documentation builder.

# -- Project information -----------------------------------------------------

project = 'Galamo'
copyright = '2025, Jashanpreet Singh Dingra'
author = 'Jashanpreet Singh Dingra'
release = 'v0.0.8'

# -- General configuration ---------------------------------------------------

import os
import sys
sys.path.insert(0, os.path.abspath('..'))  # Ensures Sphinx can find galamo package

extensions = [
    'sphinx.ext.autodoc',       # Automatically include docstrings from your code
    'sphinx.ext.napoleon',      # Supports Google & NumPy style docstrings
    'sphinx.ext.viewcode'       # Adds links to highlighted source code
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_material'  # You can switch to 'sphinx_rtd_theme' later if you want
html_static_path = ['_static']