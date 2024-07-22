from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Flatten, Activation, Dropout, Input, Add
from Evaluation import evaluation


# Define RC-LSTM model
def build_model(Feat_1, Feat_2, Feat_3, Feat_4, Feat_5, Tar, sol):
    model1 = Sequential()
    # Convolutional layers
    model1.add(Conv1D(64, 3, padding='same', input_shape=Feat_1))
    model2 = model1.add(Conv1D(64, 3, padding='same', input_shape=Feat_2))
    model3 = model2.add(Conv1D(64, 3, padding='same', input_shape=Feat_3))
    model4 = model3.add(Conv1D(64, 3, padding='same', input_shape=Feat_4))
    model5 = model4.add(Conv1D(64, 3, padding='same', input_shape=Feat_5))
    model = sol[0] * model1 + sol[1] * model2 + sol[2] * model3 + sol[3] * model4 + sol[4] *model5
    model.add(MaxPooling1D(pool_size=2))
    model.add(Activation('relu'))

    model.add(Conv1D(128, 3, padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Activation('relu'))

    # Residual connection
    residual = Conv1D(128, 1, padding='same')(model.layers[-1].output)
    model.add(Conv1D(128, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(Add()([residual, model.layers[-1].output]))

    # LSTM layer
    model.add(LSTM(64, return_sequences=True))

    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(Tar.shape[1]))
    model.add(Activation('softmax'))

    return model


def Model_ResLSTM(Feat_1, Feat_2, Feat_3, Feat_4, Feat_5, Target, sol=None):
    if sol is None:
        sol = [0.01, 0.01, 0.01, 0.01, 0.01]
    model = build_model(Feat_1, Feat_2, Feat_3, Feat_4, Feat_5, Target, sol)

    learnperc = round(Feat_1.shape[0] * 0.75)  # Split Training and Testing Datas
    train_data = Feat_1[:learnperc, :]
    train_target = Target[:learnperc, :]
    test_data = Feat_1[learnperc:, :]
    test_target = Target[learnperc:, :]

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_data, train_target, steps_per_epoch=2, epochs=10)
    pred = model.predict(test_data)

    testPredict = model.predict(test_target)
    Eval = evaluation(testPredict, test_data)

    return Eval, pred

