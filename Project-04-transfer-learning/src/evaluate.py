import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from preprocess import create_dataset, load_dataset, train_test_split_data

MODEL_PATH = "lung_cancer_ResNet50.h5"
DATASET_DIR = r"C:\Users\vikra\Lungs_cancer_detection_using_deeplearning\Dataset"

# Load dataset
image_paths, labels = load_dataset(DATASET_DIR)
_, test_paths, _, test_labels = train_test_split_data(image_paths, labels)

test_ds = create_dataset(test_paths, test_labels)

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Evaluate
pred_probs = model.predict(test_ds)
pred_labels = np.argmax(pred_probs, axis=1)

print("Accuracy:", accuracy_score(test_labels, pred_labels))
print("\nClassification Report:\n", classification_report(test_labels, pred_labels))
print("\nConfusion Matrix:\n", confusion_matrix(test_labels, pred_labels))
