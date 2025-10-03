# Lung Cancer Detection (Image Processing + Machine Learning)

## 📌 Overview

This project implements a **lung cancer detection pipeline** using medical images.
The workflow combines **classical image processing techniques** with **machine learning models** to classify images based on extracted texture features.
The objective is to showcase how traditional computer vision methods can complement machine learning for **medical image analysis**.

---

## ⚙️ Methodology

The pipeline follows a structured approach:

* **Image Preprocessing**

  * Resize, grayscale conversion, CLAHE enhancement, normalization, and sharpening

* **Segmentation**

  * Watershed algorithm to isolate lung regions

* **Feature Extraction**

  * Texture features using **Gray-Level Co-occurrence Matrix (GLCM)**

* **Model Training**

  * Machine learning classifiers are trained on extracted features, including:

    * Support Vector Machine (SVM)
    * Random Forest
    * Logistic Regression

* **Evaluation Metrics**

  * Accuracy
  * Confusion Matrix
  * Precision, Recall, and F1-score

---

## 📊 Results

* **SVM** generally performs best on texture-based features.
* **Random Forest** offers strong robustness with interpretable results.
* Logistic Regression provides a simpler baseline for comparison.

---

## 🚀 How to Run

1. Navigate to the `src/` directory:

   ```bash
   cd src
   ```

2. Run the training script:

   ```bash
   python train.py
   ```

👉 The script will:

* Extract features into `features.csv`
* Train a classifier and save it as `lung_model.pkl`
* Print evaluation metrics

---




