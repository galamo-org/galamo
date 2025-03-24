import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import matplotlib.pyplot as plt
import joblib
import os
from tkinter import Tk, filedialog


MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.keras")
model = keras.models.load_model(MODEL_PATH)


label_encoder_path = 'encoder.pkl'
if os.path.exists(label_encoder_path):
    label_encoder = joblib.load(label_encoder_path)
else:
    label_encoder = None
    print("Label encoder not found. Predictions will be class indices.")


def preprocess_image(image_path, target_size=(128, 128)):
    """Loads and preprocesses an image for the model."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error: Could not load image from {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image


    
def galaxy_morph(image_path):
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    
    processed_image = preprocess_image(image_path)
    predictions = model.predict(processed_image)
    predicted_class_index = np.argmax(predictions)
    


    max_probability = np.max(predictions)
    predicted_class = np.argmax(predictions)

    class_mapping = {
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
      
    galaxy_type, subclass = class_mapping.get(predicted_class_index, ("Unknown", "Unknown"))
    
    
        
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.imshow(image)
    plt.title(f"Galaxy Type: {galaxy_type} \n\n Subclass: {subclass}", fontsize=14, pad=20)     
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    root = Tk()
    root.withdraw() 
    image_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp * .webp")]
    )

    if image_path:
        galaxy_morph(image_path)
    else:
        print("No image selected.")