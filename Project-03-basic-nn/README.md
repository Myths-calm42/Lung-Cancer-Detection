#  Lung Cancer Detection (Image-Based Deep Learning)

##  Overview
This project focuses on **lung cancer detection** using **chest CT scan images**.  
The goal is to build and evaluate a **Convolutional Neural Network (CNN)** model to classify lung scans into **three categories**:
- Normal  
- Benign  
- Malignant  

By applying **K-Fold Cross Validation**, the project ensures robust evaluation of model performance across different data splits.

---

##  Methodology

### 1. Dataset
- Source: Local folder structure with subdirectories for each class  
  (`Dataset/normal/`, `Dataset/benign/`, `Dataset/malignant/`).  
- Images are resized to **128×128 pixels** and normalized (`rescale=1./255`).  

### 2. Preprocessing
- Image loading via `ImageDataGenerator`  
- Conversion to NumPy arrays (X for images, y for labels)  
- Dataset split using **K-Fold (5-fold)** cross-validation  

### 3. Model Architecture
A **Convolutional Neural Network (CNN)** built with TensorFlow/Keras:
- Conv2D → MaxPooling2D  
- Conv2D → MaxPooling2D  
- Flatten  
- Dense (128, ReLU)  
- Dropout (0.5)  
- Dense (Softmax, 3 classes)  

Optimizer: **Adam**  
Loss: **Sparse Categorical Crossentropy**  
Metric: **Accuracy**  

### 4. Evaluation
For each fold:
- Train on training subset  
- Validate on test subset  
- Report:
  - Accuracy  
  - Classification report (precision, recall, F1-score)  
  - Confusion matrix  

---
### Results

- The CNN achieves high accuracy in detecting lung cancer from CT scans.

- K-Fold validation provides reliable performance estimates.

- Misclassifications are mostly between benign vs malignant, while normal cases are detected more accurately.

---

##  How to Run

### 1. Clone and Install
```bash
git clone https://github.com/username/lung-cancer-detection.git
cd lung-cancer-detection
pip install -r requirements.txt
