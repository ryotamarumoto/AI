import tensorflow as tf
from keras.api._v2 import keras
from keras.models import Model
from keras.datasets import cifar10
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.utils import plot_model

import params
from preprocessing import preprocess_dataset
from model import CNNModel


def main():
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    train_images, train_labels = preprocess_dataset(images=train_images, labels=train_labels)

    model: Model = CNNModel().build()
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['acc'])
    model.summary()
    plot_model(model, to_file='model.pdf', show_shapes=True)


    callbacks = [
        EarlyStopping(patience=5),
        ModelCheckpoint(filepath=params.MODEL_FILE_PATH, save_best_only=True),
        TensorBoard(log_dir=params.LOG_DIR)]
    
    model.fit(
        x=train_images,
        y=train_labels,
        batch_size=params.BATCH_SIZE,
        epochs=params.EPOCHS,
        validation_split=params.VALIDATION_SPLIT,
        callbacks=callbacks)


if __name__ == '__main__':
    main()
    
