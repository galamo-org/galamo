# galamo/__init__.py

"""
Galamo: Galaxy Morphology Predictor
===================================

A package to predict the morphological type and subclass of galaxies from images.
"""

# Suppress TensorFlow warnings for a cleaner user experience.
# This should be at the very top, before any tensorflow imports happen.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')


# --- Primary Imports ---
# When this line is executed, it runs the model.py script, which automatically
# handles the downloading, caching, and loading of the necessary models.
from .model import galaxy_morph, preprocess_image

# --- Package Metadata ---
__version__ = "1.1.2"
__author__ = "Jashanpreet Singh"
__email__ = "astrodingra@gmail.com"


# --- Public API ---
# This list defines what a user gets when they import `from galamo import *`.
# It's a good practice to explicitly define this to keep the package's
# namespace clean.
__all__ = [
    "galaxy_morph",
    "bpt",
    "psr"
]
