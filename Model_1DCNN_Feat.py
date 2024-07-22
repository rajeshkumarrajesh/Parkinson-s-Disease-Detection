import numpy as np
import tensorflow as tf
from keras.layers import Flatten, Dense, Conv1D, MaxPool1D, Dropout


def Model_1DCNN_Feat(Data, Target, sol=None):
    if sol is None:
        sol = [128, 5]

    Train_X = np.reshape(Data, (Data.shape[0], Data.shape[1], 1))
    cnn_model = Model(Train_X, Target)
    Weights = cnn_model.layers[-1].get_weights()[0]
    Feature = np.resize(Weights, (Data.shape[0], 100))
    return Feature


def Model(X, Y):
    batch = 16
    epochs = 10
    shape = np.size(X, 1)

    cnn_model = tf.keras.models.Sequential()
    # First CNN layer  with 32 filters, conv window 3, relu activation and same padding
    cnn_model.add(
        Conv1D(filters=32, kernel_size=(3,), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.001),
               input_shape=(X.shape[1], 1)))
    # Second CNN layer  with 64 filters, conv window 3, relu activation and same padding
    cnn_model.add(
        Conv1D(filters=64, kernel_size=(3,), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.001)))
    # Third CNN layer with 128 filters, conv window 3, relu activation and same padding
    cnn_model.add(
        Conv1D(filters=64, kernel_size=(3,), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.001)))
    # Fourth CNN layer with Max pooling
    cnn_model.add(MaxPool1D(pool_size=(3,), strides=2, padding='same'))
    cnn_model.add(Dropout(0.5))
    # Flatten the output
    cnn_model.add(Flatten())
    # Add a dense layer with 256 neurons
    cnn_model.add(Dense(units=256, activation=tf.keras.layers.LeakyReLU(alpha=0.001)))
    # Add a dense layer with 512 neurons
    cnn_model.add(Dense(units=512, activation=tf.keras.layers.LeakyReLU(alpha=0.001)))
    # Softmax as last layer with five outputs
    cnn_model.add(Dense(units=5, activation='softmax'))
    cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return cnn_model

