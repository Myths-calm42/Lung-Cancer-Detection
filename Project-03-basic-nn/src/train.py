import argparse
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from preprocess import load_dataset
from model import build_model

def train(data_dir, img_size=(128, 128), batch_size=32, n_splits=5, epochs=5):
    """
    Train CNN model using K-Fold Cross Validation.
    """
    X, y, class_map = load_dataset(data_dir, img_size, batch_size)

    print("Class mapping:", class_map)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_no = 1

    for train_idx, val_idx in kf.split(X, y):
        print(f"\n--- Fold {fold_no} ---")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = build_model(input_shape=(*img_size, 3), num_classes=len(class_map))
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                            validation_data=(X_val, y_val), verbose=1)

        # Predictions
        y_pred = np.argmax(model.predict(X_val), axis=1)

        print("Accuracy:", accuracy_score(y_val, y_pred))
        print("Classification Report:\n", classification_report(y_val, y_pred, target_names=class_map.keys()))
        print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))

        fold_no += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Path to dataset directory")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--img_size", nargs=2, type=int, default=[128, 128], help="Image size (width height)")
    args = parser.parse_args()

    train(args.data_dir, tuple(args.img_size), args.batch_size, n_splits=5, epochs=args.epochs)
