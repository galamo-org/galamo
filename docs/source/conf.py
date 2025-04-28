# Configuration file for the Sphinx documentation builder.

# -- Project information -----------------------------------------------------

project = 'Galamo'
copyright = '2025, Galamo'
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

html_theme = 'pydata_sphinx_theme'  # You can switch to 'sphinx_rtd_theme' later if you want
html_static_path = ['_static']


html_theme_options = {
    "logo": {
        "image_light": "_static/galamo_main.svg",   # Optional: Your logo path
        "image_dark": "_static/gray-logo.svg",
    },
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["search-field", "theme-switcher", "navbar-icon-links"],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/galamo-org/galamo",  # Replace with your repo
            "icon": "fab fa-github",
            "type": "fontawesome",
        },
        {
            "name": "X (Twitter)",
            "url": "https://x.com/galamo_org",  # Replace with your X.com handle
            "icon": "fab fa-x-twitter",         # FontAwesome updated icon for X
            "type": "fontawesome",
        },
    ],
}

html_sidebars = {
    "**": [],
}