from tensorflow.keras.applications import VGG16, DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Concatenate
from tensorflow.keras.models import Model

def hybrid_model(input_shape=(224,224,3), num_classes=3):
    vgg = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    dense = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in vgg.layers + dense.layers:
        layer.trainable = False
    combined = Concatenate()([GlobalAveragePooling2D()(vgg.output),
                              GlobalAveragePooling2D()(dense.output)])
    x = Dense(256, activation='relu')(combined)
    output = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=[vgg.input, dense.input], outputs=output)
