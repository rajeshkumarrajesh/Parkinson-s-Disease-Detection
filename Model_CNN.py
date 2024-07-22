import numpy as np
import tflearn
from keras.layers import Conv2D, Multiply
from tensorflow.python.framework import ops
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from Evaluation import evaluation


def attention_block(input_layer):
    """
    Attention block to compute attention scores.
    """
    filters = input_layer.shape[-1]
    attention_probs = Conv2D(filters, (1, 1), activation='softmax')(input_layer)
    output_layer = Multiply()([input_layer, attention_probs])
    return output_layer


def Model(X, Y, test_x, test_y, sol):
    LR = 1e-3
    IMG_SIZE = 20
    ops.reset_default_graph()
    convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

    convnet = conv_2d(convnet, 32, 5, name='layer-conv1', activation='linear')
    convnet = max_pool_2d(convnet, 5)

    convnet_2 = conv_2d(convnet, 64, 5, name='layer-conv2', activation='linear')
    convnet = max_pool_2d(convnet_2, 5)

    convnet = conv_2d(convnet, int(sol[0]), 5, name='layer-conv3', activation='linear')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 64, 5, name='layer-conv4', activation='linear')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 32, 5, name='layer-conv5', activation='linear')
    convnet = max_pool_2d(convnet, 5)

    convnet = attention_block(convnet)

    convnet1 = fully_connected(convnet, 1024, name='layer-conv', activation='linear')
    convnet2 = dropout(convnet1, 0.8)

    convnet3 = fully_connected(convnet2, Y.shape[1], name='layer-conv-before-softmax', activation='linear')

    regress = regression(convnet3, optimizer='sgd', learning_rate=sol[1],
                         loss='mean_square', name='target')

    model = tflearn.DNN(regress, tensorboard_dir='log')

    MODEL_NAME = 'test.model'.format(LR, '6conv-basic')
    model.fit({'input': X}, {'target': Y}, n_epoch=int(sol[2]),
              validation_set=({'input': test_x}, {'target': test_y}),
              snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

    pred = model.predict(test_x)

    return pred


def Model_CNN(train_data, train_target, test_data, test_target, sol=None):
    if sol is None:
        sol = [128, 0.01, 5]
    IMG_SIZE = 20
    Train_X = np.zeros((train_data.shape[0], IMG_SIZE, IMG_SIZE, 1))
    for i in range(train_data.shape[0]):
        temp = np.resize(train_data[i], (IMG_SIZE * IMG_SIZE, 1))
        Train_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 1))

    Test_X = np.zeros((test_data.shape[0], IMG_SIZE, IMG_SIZE, 1))
    for i in range(test_data.shape[0]):
        temp = np.resize(test_data[i], (IMG_SIZE * IMG_SIZE, 1))
        Test_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 1))
    pred = Model(Train_X, train_target, Test_X, test_target, sol)

    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0

    Eval = evaluation(pred, test_target)
    return Eval, pred
