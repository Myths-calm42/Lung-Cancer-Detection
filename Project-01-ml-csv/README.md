# Lung Cancer Detection (CSV-Based Features)

##  Overview
This project focuses on **lung cancer detection** using patient feature data stored in CSV format.  
The objective is to apply **classical machine learning algorithms** on structured medical datasets to predict cancer risk categories.  
By implementing multiple supervised learning methods, the project evaluates and compares performance to identify the most effective models for structured tabular data.

---

##  Methodology
The workflow begins with **data preprocessing**, which includes:
- Data cleaning  
- Feature scaling  
- Train-test split  

Several machine learning algorithms are trained and evaluated, including:  
- Logistic Regression  
- Random Forest  
- Decision Tree  
- Support Vector Machine (SVM)  
- K-Nearest Neighbors (KNN)  
- Naive Bayes  
- Gradient Boosting (and optionally XGBoost/LightGBM)  

Each model is tested using standard evaluation metrics:
- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  

---

## Results

- Tree-based ensemble methods such as Random Forest and Gradient Boosting often achieve higher accuracy and robustness.

- Simpler models like Logistic Regression perform well in terms of interpretability but may not capture complex relationships effectively.

##  How to Run

1. Navigate to the `data/` directory:
   ```bash
   cd project-01-ml-csv/src
2.  Navigate to the `src/` directory:
   ```bash
    python train.py 
