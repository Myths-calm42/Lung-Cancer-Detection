import argparse
import numpy as np
from tensorflow.keras.models import load_model
from preprocess import load_dataset
from sklearn.metrics import classification_report, confusion_matrix

def evaluate(model_path, data_dir, img_size=(128, 128), batch_size=32):
    """
    Evaluate a trained model on a dataset.
    """
    X, y, class_map = load_dataset(data_dir, img_size, batch_size)
    model = load_model(model_path)

    y_pred = np.argmax(model.predict(X), axis=1)

    print("✅ Evaluation Results")
    print("Classification Report:\n", classification_report(y, y_pred, target_names=class_map.keys()))
    print("Confusion Matrix:\n", confusion_matrix(y, y_pred))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to trained model file (.h5)")
    parser.add_argument("--data_dir", required=True, help="Path to dataset directory")
    parser.add_argument("--img_size", nargs=2, type=int, default=[128, 128], help="Image size (width height)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    evaluate(args.model, args.data_dir, tuple(args.img_size), args.batch_size)
