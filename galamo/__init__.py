import os
import tensorflow as tf
from .model import galaxy_morph, preprocess_image
from . import bpt  # BPT module
from . import psr  # PSR enhancement module

__version__ = "0.0.8"  # Updated version to reflect new module

# Suppress TensorFlow warnings and logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Model paths
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.keras")
MODEL_PATH2 = os.path.join(os.path.dirname(__file__), "model2.keras")

# Load models
if os.path.exists(MODEL_PATH) and os.path.exists(MODEL_PATH2):
    model1 = tf.keras.models.load_model(MODEL_PATH)
    model2 = tf.keras.models.load_model(MODEL_PATH2)
else:
    raise FileNotFoundError(f"One or more model files not found at {MODEL_PATH} or {MODEL_PATH2}")

# Expose key functions and modules
__all__ = [
    "galaxy_morph",
    "preprocess_image",
    "model1",
    "model2",
    "bpt",
    "psr"
]
