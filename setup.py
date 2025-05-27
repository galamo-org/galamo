from setuptools import setup, find_packages
import os

# Read the long description from README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Ensure both models and encoder files are packaged correctly
package_data = {}
if os.path.exists("galamo/model.keras") and os.path.exists("galamo/model2.keras"):
    package_data["galamo"] = ["model.keras", "model2.keras", "encoder.pkl"]

setup(
    name="galamo",
    version="0.0.8",
    author="Jashanpreet Singh Dingra",
    author_email="astrodingra@gmail.com",
    description="A Python package for comprehensive galaxy analysis, integrating machine learning and statistical methods. It provides automated tools for morphology classification, kinematics, photometry, and spectral analysis to aid astrophysical research.",
    long_description=long_description,  
    long_description_content_type="text/markdown",  
    url="https://www.galamo.org", 

    project_urls={
        "Source Code": "https://github.com/galamo-org/galamo",
        "Documentation": "https://galamo.readthedocs.io/en/latest/", 
        "Bug Tracker": "https://github.com/galamo-org/galamo/issues",
    },
    include_package_data=True,  
    package_data=package_data, 

    install_requires=[
        "tensorflow",
        "numpy",
        "opencv-python",
        "joblib",
        "matplotlib",
        "termcolor",
        "requests"
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
    python_requires=">=3.10",
)
