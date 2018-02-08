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
from keras.layers import Dense, Activation
from keras.layers import LSTM, GRU, CuDNNGRU, CuDNNLSTM, Bidirectional
from keras.optimizers import RMSprop, SGD
from keras.callbacks import TensorBoard, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf

import json
import argparse
from time import time

from Wind.Util import load_config_file



__author__ = 'bejar'


def lagged_vector(data, lag=1, ahead=0):
    """
    Returns a matrix with columns that are the steps of the lagged time series
    Last column is the value to predict
    :param data:
    :param lag:
    :return:
    """
    lvect = []
    for i in range(lag):
        lvect.append(data[i: -lag - ahead + i])
    lvect.append(data[lag + ahead:])
    return np.stack(lvect, axis=1)


def lagged_matrix(data, lag=1, ahead=0):
    """
    Returns a matrix with columns that are the steps of the lagged time series
    Last column is the value to predict
    :param data:
    :param lag:
    :return:
    """
    lvect = []
    for i in range(lag):
        lvect.append(data[i: -lag - ahead + i, :])
    lvect.append(data[lag + ahead:, :])
    return np.stack(lvect, axis=1)

def dataset():
    """
    Generates the dataset for training, test and validation

      0 = One site - wind
      1 = One site - all variables
      2 = All sites - wind
      3 = All sites - all variables
      4 = All sites - all variables stacked

    :return:
    """
    scaler = StandardScaler()
    if 0 <= config['dataset'] < 4:
        if config['dataset'] == 0:  # only windspeed
            wind1 = wind['90-45142']
            wind1 = wind1[:, 0].reshape(-1, 1)
        elif config['dataset'] == 1:  # One site all variables
            wind1 = wind['90-45142']
        elif config['dataset'] == 2:  # four sites windspeed
            wind1 = np.vstack((wind['90-45142'][:, 0],
                               wind['90-45143'][:, 0],
                               wind['90-45229'][:, 0],
                               wind['90-45230'][:, 0])
                              ).T
        elif config['dataset'] == 3:  # four sites all variables
            wind1 = np.hstack((wind['90-45142'],
                               wind['90-45143'],
                               wind['90-45229'],
                               wind['90-45230'])
                              )

        wind1 = scaler.fit_transform(wind1)

        print('DATA Dim =', wind1.shape)
        # print(wind2.shape)

        wind_train = wind1[:datasize, :]
        print('Train Dim =', wind_train.shape)

        if config['dataset'] == 0:
            train = lagged_vector(wind_train, lag=lag, ahead=ahead - 1)
        else:
            train = lagged_matrix(wind_train, lag=lag, ahead=ahead - 1)

        train_x, train_y = train[:, :lag], train[:, -1, 0]

        if config['dataset'] == 0:
            train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))

        if config['dataset'] == 0:
            wind_test = wind1[datasize:datasize + testsize, 0].reshape(-1, 1)
            test = lagged_vector(wind_test, lag=lag, ahead=ahead - 1)
        else:
            wind_test = wind1[datasize:datasize + testsize, :]
            test = lagged_matrix(wind_test, lag=lag, ahead=ahead - 1)

        half_test = int(test.shape[0] / 2)

        val_x, val_y = test[:half_test, :lag], test[:half_test, -1, 0]
        if config['dataset'] == 0:
            val_x = np.reshape(val_x, (val_x.shape[0], val_x.shape[1], 1))

        test_x, test_y = test[half_test:, :lag], test[half_test:, -1, 0]
        if config['dataset'] == 0:
            test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))
    elif config['dataset'] == 4:  # four sites all variables all sites stacked
        wind1 = wind['90-45142']
        wind2 = wind['90-45143']
        wind3 = wind['90-45229']
        wind4 = wind['90-45230']

        scaler = StandardScaler()
        wind1 = scaler.fit_transform(wind1)
        wind2 = scaler.fit_transform(wind2)
        wind3 = scaler.fit_transform(wind3)
        wind4 = scaler.fit_transform(wind4)

        print(wind1.shape)
        # print(wind2.shape)

        wind_train1 = wind1[:datasize, :]
        wind_train2 = wind2[:datasize, :]
        wind_train3 = wind3[:datasize, :]
        wind_train4 = wind4[:datasize, :]
        print(wind_train1.shape)

        train1 = lagged_matrix(wind_train1, lag=lag, ahead=ahead - 1)
        train_x1, train_y1 = train1[:, :lag], train1[:, -1, 0]
        train2 = lagged_matrix(wind_train2, lag=lag, ahead=ahead - 1)
        train_x2, train_y2 = train2[:, :lag], train2[:, -1, 0]
        train3 = lagged_matrix(wind_train3, lag=lag, ahead=ahead - 1)
        train_x3, train_y3 = train3[:, :lag], train3[:, -1, 0]
        train4 = lagged_matrix(wind_train4, lag=lag, ahead=ahead - 1)
        train_x4, train_y4 = train4[:, :lag], train4[:, -1, 0]

        train_x = np.vstack((train_x1, train_x2, train_x3, train_x4))
        train_y = np.hstack((train_y1, train_y2, train_y3, train_y4))

        print(train_x.shape, train_y.shape)

        wind_test = wind1[datasize:datasize + testsize, :]
        test = lagged_matrix(wind_test, lag=lag, ahead=ahead - 1)
        half_test = int(test.shape[0] / 2)

        val_x, val_y = test[:half_test, :lag], test[:half_test, -1, 0]
        test_x, test_y = test[half_test:, :lag], test[half_test:, -1, 0]

    return train_x, train_y, val_x, val_y, test_x, test_y


def architecture(neurons, drop, nlayers, activation, activation_r, rnntype, CuDNN=False, bidirectional=False):
    """
    RNN architecture

    :return:
    """

    if CuDNN:
        RNN = CuDNNLSTM if rnntype == 'LSTM' else CuDNNGRU
        model = Sequential()
        if nlayers == 1:
            model.add(RNN(neurons, input_shape=(train_x.shape[1], train_x.shape[2])))
        else:
            model.add(RNN(neurons, input_shape=(train_x.shape[1], train_x.shape[2]), return_sequences=True))
            for i in range(1, nlayers - 1):
                model.add(RNN(neurons, return_sequences=True))
            model.add(RNN(neurons))
        model.add(Dense(1))
    else:
        RNN = LSTM if rnntype == 'LSTM' else GRU
        model = Sequential()
        if bidirectional:
            if nlayers == 1:
                model.add(Bidirectional(RNN(neurons, input_shape=(train_x.shape[1], train_x.shape[2]), implementation=impl,
                              recurrent_dropout=drop, activation=activation, recurrent_activation=activation_r)))
            else:
                model.add(Bidirectional(RNN(neurons, input_shape=(train_x.shape[1], train_x.shape[2]), implementation=impl,
                              recurrent_dropout=drop, activation=activation, recurrent_activation=activation_r,
                              return_sequences=True)))
                for i in range(1, nlayers - 1):
                    model.add(Bidirectional(RNN(neurons, recurrent_dropout=drop, implementation=impl,
                                  activation=activation, recurrent_activation=activation_r, return_sequences=True)))
                model.add(Bidirectional(RNN(neurons, recurrent_dropout=drop, activation=activation,
                              recurrent_activation=activation_r, implementation=impl)))

        else:
            if nlayers == 1:
                model.add(RNN(neurons, input_shape=(train_x.shape[1], train_x.shape[2]), implementation=impl,
                              recurrent_dropout=drop, activation=activation, recurrent_activation=activation_r))
            else:
                model.add(RNN(neurons, input_shape=(train_x.shape[1], train_x.shape[2]), implementation=impl,
                              recurrent_dropout=drop, activation=activation, recurrent_activation=activation_r,
                              return_sequences=True))
                for i in range(1, nlayers - 1):
                    model.add(RNN(neurons, recurrent_dropout=drop, implementation=impl,
                                  activation=activation, recurrent_activation=activation_r, return_sequences=True))
                model.add(RNN(neurons, recurrent_dropout=drop, activation=activation,
                              recurrent_activation=activation_r, implementation=impl))
        model.add(Dense(1))

    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config1', help='Experiment configuration')
    parser.add_argument('--verbose', help="Verbose output (enables Keras verbose output)", action='store_true',
                        default=False)
    parser.add_argument('--gpu', help="Use LSTM/GRU gpu implementation", action='store_true', default=False)
    args = parser.parse_args()

    verbose = 1 if args.verbose else 0
    impl = 2 if args.gpu else 0

    config = load_config_file(args.config)
    ############################################
    # Data

    vars = {0: 'wind_speed', 1: 'air_density', 2: 'pressure'}

    wind = np.load('../Data/%s.npz' % config['datafile'])
    print(wind.files)

    # Size of the training and size for validatio+test set (half for validation, half for test)
    datasize = config['datasize']
    testsize = config['testsize']
    lag = config['lag']
    sahead = config['ahead']

    train_x, train_y, val_x, val_y, test_x, test_y = dataset()


    for ahead in range(1, sahead+1):
        print('-----------------------------------------------------------------------------')
        print('Steps Ahead = %d '%ahead)



        ############################################
        # Model

        neurons = config['neurons']
        drop = config['drop']
        nlayers = config['nlayers']  # >= 1

        activation = config['activation']
        activation_r = config['activation_r']


        model = architecture(neurons, drop, nlayers, activation, activation_r, config['rnn'], config['CuDNN'])

        print('lag: ', lag, 'Neurons: ', neurons, 'Layers: ', nlayers, activation, activation_r)
        print('Tr:', train_x.shape, train_y.shape, 'Val:', val_x.shape, val_y.shape, 'Ts:', test_x.shape, test_y.shape)
        print()

        ############################################
        # Training
        tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
        mcheck = ModelCheckpoint(filepath='./model.h5', monitor='val_loss', verbose=0, save_best_only=True,
                                 save_weights_only=False, mode='auto', period=1)
        # optimizer = RMSprop(lr=0.00001)
        model.compile(loss='mean_squared_error', optimizer='adam')

        batch_size = config['batch']
        nepochs = config['epochs']

        model.fit(train_x, train_y, batch_size=batch_size, epochs=nepochs, validation_data=(val_x, val_y),
                  verbose=verbose, callbacks=[mcheck])

        ############################################
        # Results

        model = load_model('./model.h5')

        score = model.evaluate(val_x, val_y, batch_size=batch_size, verbose=0)
        print('MSE val= ', score)
        print('MSE val persistence =', mean_squared_error(val_y[ahead:], val_y[0:-ahead]))
        val_yp = model.predict(val_x, batch_size=batch_size, verbose=0)
        print('R2 val= ', r2_score(val_y, val_yp))
        print('R2 val persistence =', r2_score(val_y[ahead:], val_y[0:-ahead]))

        score = model.evaluate(test_x, test_y, batch_size=batch_size, verbose=0)
        print()
        print('MSE test= ', score)
        print('MSE test persistence =', mean_squared_error(test_y[ahead:], test_y[0:-ahead]))
        test_yp = model.predict(test_x, batch_size=batch_size, verbose=0)
        print('R2 test= ', r2_score(test_y, test_yp))
        print('R2 test persistence =', r2_score(test_y[ahead:], test_y[0:-ahead]))

    # plt.subplot(2, 1, 1)
    # plt.plot(test_predict, color='r')
    # plt.plot(test_y, color='b')
    # plt.subplot(2, 1, 2)
    # plt.plot(test_y - test_predict, color='r')
    # plt.show()
    #
    # # step prediction
    #
    # obs = np.zeros((1, lag, 1))
    # pwindow = 5
    # lwpred = []
    # for i in range(0, 2000-(2*lag)-pwindow, pwindow):
    #     # copy the observations values
    #     for j in range(lag):
    #         obs[0, j, 0] = test_y[i+j]
    #
    #     lpred = []
    #     for j in range(pwindow):
    #         pred = model.predict(obs)
    #         lpred.append(pred)
    #         for k in range(lag-1):
    #             obs[0, k, 0] = obs[0, k+1, 0]
    #         obs[0, -1, 0] = pred
    #
    #
    #     lwpred.append((i, np.array(lpred)))
    #
    #
    # plt.subplot(1, 1, 1)
    # plt.plot(test_y[0:2100], color='b')
    # for i, (_, pred) in zip(range(0, 2000, pwindow), lwpred):
    #     plt.plot(range(i+lag, i+lag+pwindow), np.reshape(pred,pwindow), color='r')
    # plt.show()
