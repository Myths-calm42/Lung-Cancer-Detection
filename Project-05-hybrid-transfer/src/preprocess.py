import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_generator(img_size=(224,224), batch_size=32):
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )
    return datagen
