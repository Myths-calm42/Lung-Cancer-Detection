import tensorflow as tf
from sklearn.model_selection import KFold
from preprocess import get_data_generator
from model import hybrid_model

def train_model(dataset_path, epochs=40, k=5):
    datagen = get_data_generator()
    kf = KFold(n_splits=k, shuffle=True)
    model = hybrid_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # Add loop for KFold logic (load, train, save best model)
    model.fit(datagen.flow_from_directory(dataset_path, subset='training'),
              validation_data=datagen.flow_from_directory(dataset_path, subset='validation'),
              epochs=epochs)
    model.save("lung_cancer_hybrid_model.h5")
