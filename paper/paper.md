---
title: 'Galamo: an open source python package for comprehensive galaxy analysis'
authors:
  - name: Jashanpreet Singh Dingra
    affiliation: 1
    orcid: 0009-0007-1311-3898
  - name: Virkamjeet Singh
    affiliation: 1
affiliations:
  - name: Guru Nanak Dev University, Amritsar, India
    index: 1
date: 2025-07-20
tags:
  - astronomy
  - galaxy classification
  - machine learning
  - spectral diagnostics
  - galaxy morphology
  - SDSS
  - open source
bibliography: paper.bib
---

# Summary

Classifying galaxies by their morphology and nuclear activity is a key step in understanding galaxy evolution, formation history, and the growth of supermassive black holes [@Conselice2014], [@Kormendy2013], [@Hopkins2006]. However, large sky surveys like SDSS [@York2000], [@Eisenstein2011] and HST generate millions of galaxy observations, making manual classification and custom spectral diagnostics infeasible at scale. To address this challenge, we present Galamo, a modular and open-source Python toolkit for machine-learning-assisted galaxy analysis. Galamo provides tools for automated morphological classification using photometric inputs, spectroscopic classification using emission-line ratios, and enhancement of low-light astronomical images. Designed for both educational and research applications, Galamo lowers the barrier to entry for reproducible galaxy science.

# Statement of Need

Galaxy morphology classification has traditionally depended on visual inspection, as seen in the Galaxy Zoo project [@Lintott2008],[@Willett2013], where volunteers labeled over a million galaxies. Similarly, spectral classification via the BPT diagram [@Baldwin1981] requires reliable computation of nebular emission line ratios. While tools like Astropy [@Astropy2013], [@Astropy2018] and Scikit-learn provide low-level capabilities, they do not offer unified, astronomy-specific pipelines for these critical tasks. Galamo addresses this need by offering domain-specific workflows for morphology prediction and spectral diagnostics in a single package, enabling students and researchers to perform analysis without writing long custom code. It supports batch processing of SDSS and HST data, and integrates machine learning with observational astronomy standards.

# Overview of Package Content 

Galamo is organized into three modules: `galaxy_morph`, `bpt`, and `psr`. 

The `galaxy_morph` module performs supervised morphological classification of galaxies into 36 detailed categories consistent with the Galaxy Zoo classification scheme. These categories reflect a comprehensive set of morphological features including bulge prominence, bar structure, spiral arm tightness and count, odd or peculiar features, and shape descriptors. Additionally, the module offers a simplified prediction mode that classifies galaxies into three major morphological classes: *spiral*, *elliptical*, and *merger*. 

The classification model mirrors the 3C3-MP-Dense256 convolutional neural network (CNN) architecture [@Dieleman2015] and is trained on the Galaxy Zoo dataset, publicly available on Kaggle ([https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/data](https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/data)). The dataset includes a training CSV file containing 38 columns: GalaxyID and 37 class probability columns named from Class1.1 to Class11.6, along with a folder of galaxy image cutouts associated by GalaxyID. These 37 labels represent a detailed taxonomy, starting with broad Hubble-type classes (Class1.1 for ellipticals, Class1.2 for spirals, and Class1.3 for irregulars), further divided into subtypes. Spiral classifications (e.g., Class2.1 and Class2.2) distinguish between barred and unbarred spirals, and the winding tightness is captured in Class3.1 and Class3.2. Elliptical galaxies are subdivided into round (Class4.1) and elongated (Class4.2) morphologies. Merger and irregular systems are represented by Classes 5.1 through 5.4, which include tidal remnants, dwarf irregulars, and major merger systems. Additional morphological nuances such as lenticulars, polar ring galaxies, cD galaxies, and ultra-diffuse galaxies are labeled under Classes 6.1 through 11.6. These include specialized types like barred/unbarred lenticulars, collisional rings, dust lane ellipticals, nucleated dwarf ellipticals, and orientation-dependent spiral subclasses.

For preprocessing, the image data (X) is aligned with GalaxyIDs and resized to 128×128 pixels, followed by rescaling and normalization. The label data (Y) is extracted from the CSV file and encoded into integer classes using label encoding. The dataset is partitioned into training and testing subsets using standard train-test splitting. Model compilation employs the Adam optimizer for adaptive learning rate adjustments, with categorical cross-entropy as the loss function and classification accuracy as the evaluation metric. The model training is accelerated using an NVIDIA GeForce GTX 1650 GPU. The best-performing model currently achieves an accuracy of 85.17% in version v0.0.9, with potential improvements expected in future releases.

To enhance interpretability, the `galaxy_morph` module also includes an alternative decision tree-based classification method grounded in the Galaxy Zoo hierarchical voting scheme ([https://data.galaxyzoo.org/gz\_trees/gz\_trees.html](https://data.galaxyzoo.org/gz_trees/gz_trees.html)). This model uses a branching series of binary questions such as “Is the galaxy smooth?”, “Is there a disk?”, and “Are there spiral arms?” to probabilistically determine morphological characteristics. It provides transparency into the logical flow behind classifications and allows users to visualize how different structural features contribute to the final classification.


The `bpt` module automates the classification of galaxies into AGN, composite, or star-forming based on the BPT diagram. It computes two key emission line ratios: log([O III]/Hβ) and log([N II]/Hα), and compares them against two well-known classification boundaries:

- Kauffmann et al. (2003):
  
  $$
  \log_{10} \left( \frac{[\text{O III}]}{\text{H}\beta} \right) = \frac{0.61}{\log_{10} \left( \frac{[\text{N II}]}{\text{H}\alpha} \right) - 0.05} + 1.3
  $$

- Kewley et al. (2001):
  
  $$
  \log_{10} \left( \frac{[\text{O III}]}{\text{H}\beta} \right) = \frac{0.61}{\log_{10} \left( \frac{[\text{N II}]}{\text{H}\alpha} \right) - 0.47} + 1.19
  $$


Galaxies below the Kauffmann line [@Kauffmann2003] are considered star-forming, those between the two lines are composites, and those above the Kewley line [@Kewley2001] are classified as AGNs. The module allows both manual input of line fluxes and automatic extraction from SDSS spectra. It also performs error checking on flux uncertainties and generates annotated plots.

The `psr` module in Galamo (Preprocessing and Signal Recovery) is designed to enhance the quality of low-light astronomical images, particularly those obtained from faint lunar and galaxies observations. Although the primary focus of Galamo is on the morphological and spectral classification of galaxies, this module plays a complementary role by improving image clarity prior to classification. The current implementation uses a classical computer vision pipeline based on OpenCV and NumPy libraries. The enhancement process begins with denoising using the Fast Non-Local Means Denoising algorithm, which reduces background noise while preserving fine structures. The denoised image is then subjected to contrast enhancement using CLAHE (Contrast Limited Adaptive Histogram Equalization), which improves the visibility of faint features across local regions of the image. This is followed by gamma correction to adjust the brightness non-linearly, helping to reveal structures in underexposed regions. Edge sharpening is applied using a high-pass kernel to enhance morphological boundaries, and the result is combined with the gamma-corrected image through weighted blending to maintain both smoothness and structural sharpness. A final contrast enhancement is applied to improve global balance. The enhancement function supports visualization using Matplotlib and allows saving the processed images in both raster and vector formats, including PNG and PDF. This lightweight and interpretable approach provides an effective preprocessing step that improves the performance of subsequent deep learning models or visual analysis.

# Functionality

Galamo is accessible via both command-line and importable Python modules. It includes example notebooks for morphology prediction, BPT diagram generation, and batch processing of SDSS DR17 cutouts using Astroquery. Outputs include classification labels, probabilities, diagnostic plots, and enhanced images. Users can customize model thresholds, switch between CNN and tree classifiers, and export results in CSV or FITS format. The code is written in Python 3.8+, and depends on Astropy, NumPy, Matplotlib, TensorFlow, Astroquery, and Scikit-learn. Galamo is publicly available at [https://github.com/galamo-org/galamo](https://github.com/galamo-org/galamo), with full documentation hosted at [https://galamo.org](https://galamo.org). The package is also distributed via the Python Package Index (PyPI) [https://pypi.org/project/galamo/](https://pypi.org/project/galamo/), allowing users to install it directly using standard package management tools.

# Acknowledgements

This work makes use of the Sloan Digital Sky Survey (SDSS) public archives. Funding for the Sloan Digital Sky Survey V has been provided by the Alfred P. Sloan Foundation, the Heising-Simons Foundation, the National Science Foundation, and the Participating Institutions. SDSS acknowledges support and resources from the Center for High-Performance Computing at the University of Utah. SDSS telescopes are located at Apache Point Observatory, funded by the Astrophysical Research Consortium and operated by New Mexico State University, and at Las Campanas Observatory, operated by the Carnegie Institution for Science. The SDSS web site is [https://www.sdss.org](www.sdss.org).

SDSS is managed by the Astrophysical Research Consortium for the Participating Institutions of the SDSS Collaboration, including the Carnegie Institution for Science, Chilean National Time Allocation Committee (CNTAC) ratified researchers, Caltech, the Gotham Participation Group, Harvard University, Heidelberg University, The Flatiron Institute, The Johns Hopkins University, L'Ecole polytechnique f\'{e}d\'{e}rale de Lausanne (EPFL), Leibniz-Institut f\"{u}r Astrophysik Potsdam (AIP), Max-Planck-Institut f\"{u}r Astronomie (MPIA Heidelberg), Max-Planck-Institut f\"{u}r Extraterrestrische Physik (MPE), Nanjing University, National Astronomical Observatories of China (NAOC), New Mexico State University, The Ohio State University, Pennsylvania State University, Smithsonian Astrophysical Observatory, Space Telescope Science Institute (STScI), the Stellar Astrophysics Participation Group, Universidad Nacional Aut\'{o}noma de M\'{e}xico, University of Arizona, University of Colorado Boulder, University of Illinois at Urbana-Champaign, University of Toronto, University of Utah, University of Virginia, Yale University, and Yunnan University.  

This research is also based on observations made with the NASA/ESA Hubble Space Telescope obtained from the Space Telescope Science Institute, which is operated by the Association of Universities for Research in Astronomy, Inc., under NASA contract NAS 5–26555.

We gratefully acknowledge the efforts of the Galaxy Zoo collaboration for releasing morphological labels and decision tree models, which enabled reproducible training and validation. Galamo was developed as part of an undergraduate research initiative supported by Guru Nanak Dev University.

# References
