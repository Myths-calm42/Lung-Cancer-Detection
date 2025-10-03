# Lung Cancer Detection (Deep Learning with Medical Images)

##  Overview
This project focuses on **lung cancer detection using medical images (Normal, Benign, Malignant)**.  
The pipeline applies **deep learning with transfer learning** using CNN backbones such as:
- ResNet50
- EfficientNetB3
- InceptionV3
- DenseNet121
- VGG16

The models classify lung scans into **three categories** and evaluate their accuracy using standard metrics.

---

##  Methodology
1. **Data Preprocessing**
   - Load images from dataset folder structure (`normal/`, `benign/`, `malignant/`).
   - Resize images to 256×256 pixels.
   - Apply data augmentation (flip, rotation, zoom, contrast).
   - Split into train (85%) and test (15%).

2. **Model Training**
   - Use pretrained CNN backbones (`ImageNet` weights).
   - Add global pooling, dropout, and softmax classification layers.
   - Train with **Adam optimizer** and **early stopping**.

3. **Evaluation**
   - Accuracy
   - Precision, Recall, F1-score
   - Confusion Matrix
   - Classification Report

---
### Results

- **ResNet50 and DenseNet121** performed the best, achieving high accuracy and balanced metrics.

- **EfficientNet** also showed robustness with fewer parameters.

- Simpler models like VGG16 were less effective for complex medical patterns.

---
##  How to Run

Navigate to the `src/` directory:

```bash
cd project-lung-cancer-dl/src
