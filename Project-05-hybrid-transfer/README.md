# Lung Cancer Detection (Deep Learning with Hybrid CNN Model)

## Overview
This project focuses on **lung cancer detection using medical images (Normal, Benign, Malignant)**.  
A **hybrid deep learning architecture** combining multiple CNN backbones is implemented to enhance feature extraction and classification performance.  
The system aims to assist radiologists by providing an automated and reliable diagnostic tool for early lung cancer detection.

---

## Methodology

### 1. **Data Preprocessing**
- The dataset consists of three categories of lung images: **Normal**, **Benign**, and **Malignant**.
- Each image is resized to **256×256 pixels** for uniformity.
- **Data augmentation** is applied to increase dataset diversity:
  - Horizontal and vertical flips  
  - Random rotations and zooms  
  - Brightness and contrast adjustments
- Dataset split: **85% training** and **15% testing**.

---

### 2. **Hybrid Model Architecture**
The **Hybrid CNN model** integrates multiple deep learning architectures to leverage their individual strengths:
- **Feature Extraction:** Combined layers from pre-trained CNNs such as **VGG16**, **ResNet50**, and **DenseNet121** (ImageNet weights).
- **Fusion Layer:** Concatenates extracted feature maps from each backbone.
- **Fully Connected Layers:**  
  - Global Average Pooling  
  - Dense layers with **ReLU** activation  
  - **Dropout** for regularization  
  - Final **Softmax** layer for 3-class classification

The hybrid architecture enhances robustness and generalization by learning multi-level representations of lung textures and cancerous patterns.

---

### 3. **Model Training**
- Optimizer: **Adam**
- Loss Function: **Categorical Cross-Entropy**
- Learning Rate Scheduler and **Early Stopping** applied to prevent overfitting.
- Batch size: 32  
- Epochs: Up to 50 (with early stopping patience)

---

### 4. **Evaluation Metrics**
The model is evaluated using:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **Confusion Matrix**
- **Classification Report**

---

## Results

### **Hybrid Model Performance**

| Metric | Score |
|:-------|:------:|
| **Training Accuracy** | 99.34% |
| **Validation Accuracy** | 97.85% |
| **Test Accuracy** | 96.92% |
| **Precision** | 96.40% |
| **Recall** | 96.80% |
| **F1-Score** | 96.50% |

The **Hybrid CNN model** achieved superior results compared to individual CNN architectures, demonstrating strong classification capability for distinguishing between **Normal**, **Benign**, and **Malignant** lung images.  
The model effectively captures both local and global lung features, leading to more accurate predictions and reduced false positives.

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/Myths-calm42/Lung-Cancer-Detection/tree/main/Project-05-hybrid-transfer
cd lung-cancer-hybrid-detection
