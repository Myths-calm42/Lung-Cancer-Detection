import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def load_dataset(data_dir, img_size=(128, 128), batch_size=32):
    """
    Load dataset from directory and return X (images), y (labels).
    Directory structure:
        data_dir/
            normal/
            benign/
            malignant/
    """
    datagen = ImageDataGenerator(rescale=1./255)

    dataset = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='sparse',
        shuffle=False
    )

    X, y = [], []
    for images, labels in dataset:
        X.extend(images)
        y.extend(labels)
        if len(X) >= dataset.samples:
            break

    X = np.array(X)
    y = np.array(y)

    print(f"✅ Dataset loaded | X shape: {X.shape}, y shape: {y.shape}")
    return X, y, dataset.class_indices
