import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Image settings
IMG_SIZE = 256
BATCH_SIZE = 16
CLASS_MAPPING = {"normal": 0, "benign": 1, "malignant": 2}

# Data augmentation pipeline
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.05),
    tf.keras.layers.RandomContrast(0.05),
])

def load_dataset(data_dir):
    image_paths, labels = [], []
    for class_name, label in CLASS_MAPPING.items():
        folder = os.path.join(data_dir, class_name)
        for fname in os.listdir(folder):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(folder, fname))
                labels.append(label)
    return np.array(image_paths), np.array(labels)

def parse_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    return image / 255.0, label

def create_dataset(paths, labels, shuffle=False, augment=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths))
    ds = ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x), y), 
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

def train_test_split_data(image_paths, labels, test_size=0.15):
    return train_test_split(image_paths, labels, test_size=test_size, 
                            stratify=labels, random_state=42)
