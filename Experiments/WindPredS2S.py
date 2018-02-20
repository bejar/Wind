"""
.. module:: WindPrediction

WindPrediction
*************

:Description: WindPrediction

:Authors: bejar
    

:Version: 

:Created on: 06/09/2017 9:47 

"""

from __future__ import print_function
import numpy as np

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, RepeatVector, TimeDistributed
from keras.layers import LSTM, GRU, CuDNNGRU, CuDNNLSTM, Bidirectional
from keras.optimizers import RMSprop, SGD
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.regularizers import l1, l2
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

import json
import argparse
from time import time

from Wind.Util import load_config_file
from Wind.Data import generate_dataset
__author__ = 'bejar'

def architectureS2S(ahead, neurons, drop, nlayersE, nlayersD, activation, activation_r, rnntype, CuDNN=False,
                    bidirectional=False,
                    rec_reg='l1', rec_regw=0.1, k_reg='l1', k_regw=0.1):
    """

    :param neurons:
    :param drop:
    :param nlayers:
    :param activation:
    :param activation_r:
    :param rnntype:
    :param CuDNN:
    :param bidirectional:
    :param rec_reg:
    :param rec_regw:
    :param k_reg:
    :param k_regw:
    :return:
    """

    if rec_reg == 'l1':
        rec_regularizer = l1(rec_regw)
    elif rec_reg == 'l2':
        rec_regularizer = l2(rec_regw)
    else:
        rec_regularizer = None

    if k_reg == 'l1':
        k_regularizer = l1(k_regw)
    elif rec_reg == 'l2':
        k_regularizer = l2(k_regw)
    else:
        k_regularizer = None

    if CuDNN:
        RNN = CuDNNLSTM if rnntype == 'LSTM' else CuDNNGRU
        model = Sequential()
        if nlayersE == 1:

            model.add(
                RNN(neurons, input_shape=(train_x.shape[1], train_x.shape[2]),
                    recurrent_regularizer=rec_regularizer, kernel_regularizer=k_regularizer))
        else:
            model.add(RNN(neurons, input_shape=(train_x.shape[1], train_x.shape[2]), return_sequences=True,
                          recurrent_regularizer=rec_regularizer, kernel_regularizer=k_regularizer))
            for i in range(1, nlayersE - 1):
                model.add(RNN(neurons, return_sequences=True, recurrent_regularizer=rec_regularizer,
                              kernel_regularizer=k_regularizer))
            model.add(RNN(neurons, recurrent_regularizer=rec_regularizer, kernel_regularizer=k_regularizer))

        model.add(RepeatVector(ahead))

        for i in range(nlayersD):
            model.add(RNN(neurons, return_sequences=True, recurrent_regularizer=rec_regularizer,
                          kernel_regularizer=k_regularizer))

        model.add(TimeDistributed(Dense(1)))
    else:
        RNN = LSTM if rnntype == 'LSTM' else GRU
        model = Sequential()
        if nlayersE == 1:
            model.add(RNN(neurons, input_shape=(train_x.shape[1], train_x.shape[2]), implementation=impl,
                          recurrent_dropout=drop, activation=activation, recurrent_activation=activation_r,
                          recurrent_regularizer=rec_regularizer, kernel_regularizer=k_regularizer))
        else:
            model.add(RNN(neurons, input_shape=(train_x.shape[1], train_x.shape[2]), implementation=impl,
                          recurrent_dropout=drop, activation=activation, recurrent_activation=activation_r,
                          return_sequences=True, recurrent_regularizer=rec_regularizer,
                          kernel_regularizer=k_regularizer))
            for i in range(1, nlayersE - 1):
                model.add(RNN(neurons, recurrent_dropout=drop, implementation=impl,
                              activation=activation, recurrent_activation=activation_r, return_sequences=True,
                              recurrent_regularizer=rec_regularizer, kernel_regularizer=k_regularizer))
            model.add(RNN(neurons, recurrent_dropout=drop, activation=activation,
                          recurrent_activation=activation_r, implementation=impl,
                          recurrent_regularizer=rec_regularizer, kernel_regularizer=k_regularizer))

        model.add(RepeatVector(ahead))

        for i in range(nlayersD):
            model.add(RNN(neurons, recurrent_dropout=drop, implementation=impl,
                          activation=activation, recurrent_activation=activation_r,
                          return_sequences=True, recurrent_regularizer=rec_regularizer,
                          kernel_regularizer=k_regularizer))

        model.add(TimeDistributed(Dense(1)))

    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config2', help='Experiment configuration')
    parser.add_argument('--verbose', help="Verbose output (enables Keras verbose output)", action='store_true',
                        default=False)
    parser.add_argument('--gpu', help="Use LSTM/GRU gpu implementation", action='store_true', default=False)
    args = parser.parse_args()

    verbose = 1 if args.verbose else 0
    impl = 2 if args.gpu else 1

    config = load_config_file(args.config)
    ############################################
    # Data
    ahead = config['data']['ahead']

    train_x, train_y, val_x, val_y, test_x, test_y = generate_dataset(config['data'], ahead=ahead, s2s=True)
    print(test_y.shape)

    ############################################
    # Model

    neurons = config['arch']['neurons']
    drop = config['arch']['drop']
    nlayersE = config['arch']['nlayersE']  # >= 1
    nlayersD = config['arch']['nlayersD']  # >= 1

    activation = config['arch']['activation']
    activation_r = config['arch']['activation_r']
    rec_reg = config['arch']['rec_reg']
    rec_regw = config['arch']['rec_regw']
    k_reg = config['arch']['k_reg']
    k_regw = config['arch']['k_regw']

    print('lag: ', config['data']['lag'], 'Neurons: ', neurons, 'Layers: ', nlayersE, nlayersD, activation, activation_r)
    print('Tr:', train_x.shape, train_y.shape, 'Val:', val_x.shape, val_y.shape, 'Ts:', test_x.shape, test_y.shape)
    print()

    model = architectureS2S(ahead=ahead, neurons=neurons, drop=drop, nlayersE=nlayersE, nlayersD=nlayersD,
                            activation=activation,
                            activation_r=activation_r, rnntype=config['arch']['rnn'], CuDNN=config['arch']['CuDNN'],
                            rec_reg=rec_reg, rec_regw=rec_regw, k_reg=k_reg, k_regw=k_regw)

    model.summary()

    ############################################
    # Training
    tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
    mcheck = ModelCheckpoint(filepath='./model.h5', monitor='val_loss', verbose=0, save_best_only=True,
                             save_weights_only=False, mode='auto', period=1)
    # optimizer = RMSprop(lr=0.00001)
    optimizer = config['training']['optimizer']
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    batch_size = config['training']['batch']
    nepochs = config['training']['epochs']

    model.fit(train_x, train_y, batch_size=batch_size, epochs=nepochs, validation_data=(val_x, val_y),
              verbose=verbose, callbacks=[mcheck])

    ############################################
    # Results

    model = load_model('./model.h5')

    # score = model.evaluate(val_x, val_y, batch_size=batch_size, verbose=0)
    # print('MSE val= ', score)
    # print('MSE val persistence =', mean_squared_error(val_y[ahead:], val_y[0:-ahead]))

    val_yp = model.predict(val_x, batch_size=batch_size, verbose=0)
    for i in range(1, ahead+1):
        print('R2 val= ', i, r2_score(val_y[:, i-1, 0], val_yp[:, i-1, 0]))
        print('R2 val persistence =', i, r2_score(val_y[i:, 0, 0], val_y[0:-i, 0, 0]))

    # score = model.evaluate(test_x, test_y, batch_size=batch_size, verbose=0)
    print()
    # print('MSE test= ', score)
    # print('MSE test persistence =', mean_squared_error(test_y[ahead:], test_y[0:-ahead]))
    test_yp = model.predict(test_x, batch_size=batch_size, verbose=0)
    for i in range(1, ahead+1):
        print('R2 test= ', i, r2_score(test_y[:, i-1, 0], test_yp[:, i-1, 0]))
        print('R2 test persistence =', i, r2_score(test_y[i:, 0, 0], test_y[0:-i, 0, 0]))

    # fig = plt.figure()
    # plt.plot(val_y[1:201, 0, 0], color='r')
    # plt.plot(val_yp[1:201, 0, 0], color='b')
    # # plt.plot(val_y[:200, 0, 0], color='g')
    # plt.show()
    #
    # fig = plt.figure()
    # plt.plot(val_y[7:207, 7, 0], color='r')
    # plt.plot(val_yp[7:207, 7, 0], color='b')
    # # plt.plot(val_y[:200, 7, 0], color='g')
    # plt.show()
    #
    # fig = plt.figure()
    # plt.plot(val_y[11:211, 11, 0], color='r')
    # plt.plot(val_yp[11:211, 11, 0], color='b')
    # # plt.plot(val_y[:200, 11, 0], color='g')
    # plt.show()
