import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def preprocess_image(image):
    return cv2.fastNlMeansDenoising(image, None, h=10, searchWindowSize=21, templateWindowSize=7)

def enhance_contrast(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

def sharpen_image(image):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)

def enhance(image_path, gamma=1.2, sharp=0.5, show=False, save_as=None):
    # Load in grayscale
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load image at {image_path}")

    # Enhance pipeline
    preprocessed = preprocess_image(image)
    contrast_enhanced = enhance_contrast(preprocessed)
    gamma_corrected = adjust_gamma(contrast_enhanced, gamma=gamma)
    sharpened = sharpen_image(gamma_corrected)
    final = cv2.addWeighted(gamma_corrected, 1 - sharp, sharpened, sharp, 0)
    final = enhance_contrast(final)

    # Show with matplotlib
    if show:
        plt.imshow(final, cmap='gray')
        plt.title("Enhanced Image")
        plt.axis('off')
        plt.show()

    # Save output
    if save_as:
        ext = Path(save_as).suffix.lower()
        if ext == '.pdf':
            plt.imsave(save_as, final, cmap='gray', format='pdf')
        else:
            cv2.imwrite(save_as, final)

    return final
