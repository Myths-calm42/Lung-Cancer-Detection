import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.model import get_model
from src.features import extract_features_from_dataset
from src.evaluate import evaluate_model

def train_and_save(X, y, model_name="svm", save_path="lung_model.pkl"):
"""
Train a model and save it to disk.
"""
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42, stratify=y
)
model = get_model(model_name)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
joblib.dump(model, save_path)
return acc, model, (X_test, y_test)

if **name** == "**main**":
# Step 1: Extract features from segmented dataset
csv_path = "features.csv"
df = extract_features_from_dataset("data/segmentation", csv_path)

```
# Step 2: Split data
X = df.drop("Label", axis=1)
y = df["Label"]

# Step 3: Train model
acc, model, (X_test, y_test) = train_and_save(X, y, model_name="svm", save_path="lung_model.pkl")
print("Training accuracy:", acc)

# Step 4: Evaluate saved model
model = joblib.load("lung_model.pkl")
evaluate_model(model, X_test, y_test)
```
