from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50, EfficientNetB3, InceptionV3, DenseNet121, VGG16

# Dictionary of backbone models
BACKBONES = {
    "ResNet50": ResNet50,
    "EfficientNetB3": EfficientNetB3,
    "InceptionV3": InceptionV3,
    "DenseNet121": DenseNet121,
    "VGG16": VGG16,
}

def build_model(model_name="ResNet50", img_size=256, num_classes=3):
    if model_name not in BACKBONES:
        raise ValueError(f"Model {model_name} not supported.")
    
    base_model = BACKBONES[model_name](
        weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3)
    )
    base_model.trainable = False  # Transfer learning (freeze base)
    
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    model = models.Model(inputs=base_model.input, outputs=outputs)
    model.compile(optimizer="adam", 
                  loss="sparse_categorical_crossentropy", 
                  metrics=["accuracy"])
    return model
