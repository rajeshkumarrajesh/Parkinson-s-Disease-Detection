import numpy as np
import tensorflow as tf
from Evaluation import evaluation


def Model_NN(Train_Data, Train_Target, Test_Data, Test_Target):

    trainX = np.reshape(Train_Data, (Train_Data.shape[0], 1, Train_Data.shape[1]))
    testX = np.reshape(Test_Data, (Test_Data.shape[0], 1, Test_Data.shape[1]))
    # Define the neural network architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(1, trainX.shape[2])),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(trainX, Train_Target, epochs=10, batch_size=32, validation_split=0.1)

    pred = model.predict(testX)
    Eval = evaluation(pred, Test_Target)
    return pred, Eval

