import os
from pathlib import Path
import logging
from typing import List, Dict, Union

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow import keras
import joblib
from tqdm import tqdm  # For progress bar

# -------------------------
# Logging setup
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# -------------------------
# Load models and encoder
# -------------------------
BASE_DIR = Path(__file__).parent
model1 = keras.models.load_model(BASE_DIR / "model.keras")
model2 = keras.models.load_model(BASE_DIR / "model2.keras")
encoder_path = BASE_DIR / "encoder.pkl"
label_encoder = joblib.load(encoder_path) if encoder_path.exists() else None

# -------------------------
# Class mappings
# -------------------------
CLASS_MAPPING_1 = {0: "Galaxy", 1: "Not a Galaxy"}
CLASS_MAPPING_2 = {
    0: ("Merger Galaxy", "Disturbed Galaxy"),
    1: ("Merger Galaxy", "Merging Galaxy"),
    2: ("Elliptical Galaxy", "Round Smooth Galaxy"),
    3: ("Elliptical Galaxy", "In-between Round Smooth Galaxy"),
    4: ("Elliptical Galaxy", "Cigar Shaped Smooth Galaxy"),
    5: ("Spiral Galaxy", "Barred Spiral Galaxy"),
    6: ("Spiral Galaxy", "Unbarred Tight Spiral Galaxy"),
    7: ("Spiral Galaxy", "Unbarred Loose Spiral Galaxy"),
    8: ("Spiral Galaxy", "Edge-on Galaxy without Bulge"),
    9: ("Spiral Galaxy", "Edge-on Galaxy with Bulge")
}

# -------------------------
# Utility functions
# -------------------------
def preprocess_image(image_path: Union[str, Path], target_size=(128, 128)) -> np.ndarray:
    """Load and preprocess an image for prediction."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = image / 255.0
    return np.expand_dims(image, axis=0)

def display_prediction(image_path: Path, prediction: Dict) -> None:
    """Display a single image with prediction overlay."""
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(5, 5))
    plt.imshow(image)
    plt.axis('off')
    
    text_lines = [f"Type: {prediction['Type']} ({prediction['Type Confidence (%)']:.1f}%)"]
    if prediction['Type'] == "Galaxy":
        text_lines.append(f"Subclass: {prediction['Subclass']}")
    
    for i, line in enumerate(text_lines):
        plt.text(5, 20 + i*20, line, color='yellow', fontsize=10, backgroundcolor='black')
    plt.title(Path(image_path).name, fontsize=12)
    plt.show()

def display_grid(images: List[Path], predictions: List[Dict], cols: int = 3) -> None:
    """Display multiple images in a grid with predictions."""
    rows = (len(images) + cols - 1) // cols
    plt.figure(figsize=(cols * 5, rows * 5))
    for i, (img_path, pred) in enumerate(zip(images, predictions)):
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.subplot(rows, cols, i+1)
        plt.imshow(image)
        plt.axis('off')
        text = f"{pred['Type']} ({pred['Type Confidence (%)']:.1f}%)"
        if pred['Type'] == "Galaxy":
            text += f"\n{pred['Subclass']}"
        plt.title(text, fontsize=10)
    plt.tight_layout()
    plt.show()

# -------------------------
# Galaxy Morphology Predictor
# -------------------------
class GalaxyMorphPredictor:
    def __init__(self):
        self.results: List[Dict] = []

    def __call__(self, path: Union[str, Path], show_preview: bool = True, grid_view: bool = False, recursive: bool = False):
        """
        Predict galaxy type and morphology for a single image, folder of images, or bulk upload.

        Args:
            path (str or Path): Path to image file or folder containing images.
            show_preview (bool): Whether to display preview of images.
            grid_view (bool): Display all images in a single grid if True.
            recursive (bool): Recursively search subfolders if True.
        """
        path = Path(path)
        images = self._get_images(path, recursive)
        if not images:
            logging.warning(f"No valid images found at {path}")
            return
        
        self.results.clear()
        logging.info(f"Running prediction on {len(images)} image(s).")
        
        for img_path in tqdm(images, desc="Processing images"):
            try:
                prediction = self._predict(img_path)
                self.results.append(prediction)
                
                if show_preview and not grid_view:
                    display_prediction(img_path, prediction)
                    
            except Exception as e:
                logging.error(f"Error processing {img_path}: {e}")
        
        if show_preview and grid_view:
            display_grid(images, self.results)

    def _get_images(self, path: Path, recursive: bool = False) -> List[Path]:
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        if path.is_dir():
            if recursive:
                return [f for f in path.rglob("*") if f.suffix.lower() in valid_exts]
            else:
                return [f for f in path.iterdir() if f.suffix.lower() in valid_exts]
        elif path.is_file() and path.suffix.lower() in valid_exts:
            return [path]
        return []

    def _predict(self, image_path: Path) -> Dict:
        image = preprocess_image(image_path)
        # Model 1: Galaxy vs Not
        pred1 = model1.predict(image, verbose=0)
        type_index = int(np.argmax(pred1))
        galaxy_type = CLASS_MAPPING_1.get(type_index, "Unknown")
        conf1 = float(np.max(pred1) * 100)
        
        # Model 2: Morphology if galaxy
        if galaxy_type == "Galaxy":
            pred2 = model2.predict(image, verbose=0)
            subclass_index = int(np.argmax(pred2))
            _, subclass = CLASS_MAPPING_2.get(subclass_index, ("Unknown", "Unknown"))
        else:
            subclass = "-"
        
        return {
            "Filename": image_path.name,
            "Type": galaxy_type,
            "Type Confidence (%)": round(conf1, 2),
            "Subclass": subclass,
            "Path": str(image_path)
        }

    def save_csv(self, filename: Union[str, Path] = "results.csv") -> None:
        if not self.results:
            logging.warning("No results to save. Run prediction first.")
            return
        df = pd.DataFrame(self.results)
        df.to_csv(filename, index=False)
        logging.info(f"Results saved to {filename}")


# -------------------------
# Export instance
# -------------------------
galaxy_morph = GalaxyMorphPredictor()
