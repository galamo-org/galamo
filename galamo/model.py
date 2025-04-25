import os
import cv2
import numpy as np
import pandas as pd
from tensorflow import keras
import joblib

# Load models
BASE_DIR = os.path.dirname(__file__)
model1 = keras.models.load_model(os.path.join(BASE_DIR, "model.keras"))      # Galaxy vs Not
model2 = keras.models.load_model(os.path.join(BASE_DIR, "model2.keras"))     # Morphology

# Load encoder (optional)
encoder_path = os.path.join(BASE_DIR, "encoder.pkl")
label_encoder = joblib.load(encoder_path) if os.path.exists(encoder_path) else None

# Class mappings
class_mapping1 = {0: "Galaxy", 1: "Not a Galaxy"}
class_mapping2 = {
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

def preprocess_image(image_path, target_size=(128, 128)):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = image / 255.0
    return np.expand_dims(image, axis=0)


class GalaxyMorphPredictor:
    def __init__(self):
        self.results = []

    def __call__(self, path):
        """Main callable function: galaxy_morph(path)"""
        if os.path.isdir(path):
            images = self._get_images_from_folder(path)
        elif os.path.isfile(path):
            images = [path]
        else:
            print(f"❌ Path not found: {path}")
            return

        self.results.clear()
        print(f"\n🪐 Running prediction on {len(images)} image(s):\n")
        for img_path in images:
            try:
                result = self._predict(img_path)
                self.results.append(result)
                self._print_result(result)
            except Exception as e:
                print(f"⚠️ Error with {img_path}: {e}")

    def _get_images_from_folder(self, folder):
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(valid_exts)]

    def _predict(self, image_path):
        image = preprocess_image(image_path)
        pred1 = model1.predict(image)
        type_index = np.argmax(pred1)
        galaxy_type = class_mapping1.get(type_index, "Unknown")
        conf1 = float(np.max(pred1) * 100)

        if galaxy_type == "Galaxy":
            pred2 = model2.predict(image)
            subclass_index = np.argmax(pred2)
            main_class, subclass = class_mapping2.get(subclass_index, ("Unknown", "Unknown"))
            conf2 = float(np.max(pred2) * 100)
        else:
            main_class, subclass, conf2 = "-", "-", 0.0

        return {
            "Filename": os.path.basename(image_path),
            "Type": galaxy_type,
            "Type Confidence (%)": round(conf1, 2),
            "Subclass": subclass,
            "Path": image_path
        }

    def _print_result(self, result):
        print(f"📂 {result['Filename']}")
        print(f"  → Type: {result['Type']} ({result['Type Confidence (%)']}%)")
        if result['Type'] == "Galaxy":
            print(f"  → Subclass: {result['Subclass']}")
        print("")

    def save_csv(self, filename="results.csv"):
        if not self.results:
            print("⚠️ No results to save. Run prediction first.")
            return
        df = pd.DataFrame(self.results)
        df.to_csv(filename, index=False)
        print(f"\n✅ Results saved to {filename}")


# Export for import
galaxy_morph = GalaxyMorphPredictor()
