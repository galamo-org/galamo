# 🌌 Galamo - Galaxy Morphology Predictor



---

## 🚀 Features

✅ Pre-trained deep learning model for galaxy morphology classification  
✅ Automatic image preprocessing (resizing, normalization, and format conversion)  
✅ Simple and intuitive API requiring only an image file as input  
✅ Supports multiple galaxy morphology types  
✅ Compatible with Python 3.6+

---

## 📥 Installation

### Install from PyPI

To install the package using pip:

```bash
pip install galamo
```

### Install from Source

Alternatively, to install from source:

```bash
git clone https://github.com/jsdingra11/galamo.git
cd galamo
pip install .
```

---

## 📖 Usage Guide

### Import and Initialize the Model

```python
from galamo import galaxy_morph
```

### Predict Galaxy Morphology from an Image

```python
galaxy_morph("galaxy.jpg")
```

### Example Output

```
Predicted Morphology: Spiral Galaxy
```

---

## ⚙️ How It Works

1. Loads a pre-trained deep learning model for galaxy classification.
2. Preprocesses the input image (resizing, RGB conversion, and normalization).
3. Feeds the processed image into the neural network for prediction.
4. Converts the predicted class index to its corresponding galaxy morphology name.

---

## 📋 Requirements

Ensure the following dependencies are installed:

- Python 3.10+
- TensorFlow
- NumPy
- OpenCV
- Joblib
- Matplotlib

---

## 🧠 Model Details

- Trained on a dataset of galaxy images labeled with different morphology types.
- Utilizes a Convolutional Neural Network (CNN) to extract features and classify images.
- Uses a label encoder to map numerical predictions to meaningful class names (e.g., Spiral, Elliptical, Irregular, etc.).

---

## 🤝 Contributing

Galamo welcome contributions! To improve the model or add new features:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m 'Added new feature'`).
4. Push the branch (`git push origin feature-name`).
5. Create a pull request.

---

## 📜 License

This project is licensed under the MIT License – see the LICENSE file for details.

---

## 📬 Contact & Support

👨‍💻 **Author:** Jashanpreet Singh Dingra  
👨‍💻 **Co-Author:** Vikramjeet Singh  
📧 **Email:** [astrodingra@gmail.com](mailto:astrodingra@gmail.com)  
🔗 **GitHub:** [https://github.com/jsdingra11](https://github.com/jsdingra11)

For issues or feature requests, please open an issue on GitHub.

---

🌠 **Galamo - Unveiling the Universe, One Galaxy at a Time!**
