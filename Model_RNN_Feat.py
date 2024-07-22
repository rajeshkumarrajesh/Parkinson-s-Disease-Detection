import numpy as np
from keras.layers import Conv1D
# https://www.tensorflow.org/guide/keras/rnn
import keras
from tensorflow.python.keras.layers import LSTM, Dense
from tensorflow.python.keras.models import Sequential


def Model_RNN_Feat(Data, Target):
    Train_X = np.reshape(Data, (Data.shape[0], Data.shape[1], 1))
    rnn_Model = RNN_train(Train_X, Target)  # RNN
    Weights = rnn_Model.layers[-1].get_weights()[0]
    Feature = np.resize(Weights, (Data.shape[0], 100))
    return Feature


def RNN_train(Data, Target):
    trainX = np.reshape(Data, (Data.shape[0], 1, Data.shape[1]))
    model = Sequential()
    model = keras.models.Sequential()
    # Add a dilated convolutional layer
    model.add(Conv1D(filters=32, kernel_size=3, dilation_rate=2, activation='relu', input_shape=(100, 1)))
    model.add(LSTM(10, input_shape=(1, trainX.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
