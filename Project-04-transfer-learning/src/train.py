import tensorflow as tf
from preprocess import load_dataset, train_test_split_data, create_dataset
from model import build_model

DATASET_DIR = r"C:\Users\vikra\Lungs_cancer_detection_using_deeplearning\Dataset"
MODEL_NAME = "ResNet50"

# Load dataset
image_paths, labels = load_dataset(DATASET_DIR)
train_paths, test_paths, train_labels, test_labels = train_test_split_data(image_paths, labels)

# Create datasets
train_ds = create_dataset(train_paths, train_labels, shuffle=True, augment=True)
test_ds = create_dataset(test_paths, test_labels)

# Build model
model = build_model(MODEL_NAME)

# Callbacks
early_stop = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(patience=3)

# Train model
history = model.fit(train_ds, validation_data=test_ds, epochs=20,
                    callbacks=[early_stop, reduce_lr])

# Save model
model.save(f"lung_cancer_{MODEL_NAME}.h5")
