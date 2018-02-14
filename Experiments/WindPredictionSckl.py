"""
.. module:: WindPredictionKNN

WindPredictionKNN
*************

:Description: WindPredictionKNN

    

:Authors: bejar
    

:Version: 

:Created on: 13/02/2018 11:04 

"""

from __future__ import print_function
import numpy as np

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import os

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


def dataset(ahead):
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

        print(train.shape)
        train_x, train_y = train[:, :lag], train[:, -1, 0]

        if config['dataset'] == 0:
            train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1]))
        else:
            train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1]*train_x.shape[2]))


        if config['dataset'] == 0:
            wind_test = wind1[datasize:datasize + testsize, 0].reshape(-1, 1)
            test = lagged_vector(wind_test, lag=lag, ahead=ahead - 1)
        else:
            wind_test = wind1[datasize:datasize + testsize, :]
            test = lagged_matrix(wind_test, lag=lag, ahead=ahead - 1)

        half_test = int(test.shape[0] / 2)

        val_x, val_y = test[:half_test, :lag], test[:half_test, -1, 0]
        if config['dataset'] == 0:
            val_x = np.reshape(val_x, (val_x.shape[0], val_x.shape[1]))
        else:
            val_x = np.reshape(val_x, (val_x.shape[0], val_x.shape[1]*val_x.shape[2]))

        test_x, test_y = test[half_test:, :lag], test[half_test:, -1, 0]
        if config['dataset'] == 0:
            test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1]))
        else:
            test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1]*test_x.shape[2]))

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

        val_x, val_y = test[:half_test, :lag], test[:half_test, -1]
        test_x, test_y = test[half_test:, :lag], test[half_test:, -1]

    return train_x, train_y, val_x, val_y, test_x, test_y




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config1', help='Experiment configuration')
    # parser.add_argument('--verbose', help="Verbose output (enables Keras verbose output)", action='store_true',
    #                     default=False)
    # parser.add_argument('--gpu', help="Use LSTM/GRU gpu implementation", action='store_true', default=False)
    args = parser.parse_args()

    # verbose = 1 if args.verbose else 0
    # impl = 2 if args.gpu else 0

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

    for ahead in range(1, sahead + 1):
        print('-----------------------------------------------------------------------------')
        print('Steps Ahead = %d ' % ahead)

        train_x, train_y, val_x, val_y, test_x, test_y = dataset(ahead)
        print(train_x. shape, train_y.shape)

        ############################################
        # Model
        # knnr = KNeighborsRegressor(n_neighbors=27, weights='distance', n_jobs=-1)
        knnr = SVR(kernel='rbf', C=2, epsilon=0.1)
        knnr.fit(train_x, train_y)
        ############################################
        # Results


        val_yp = knnr.predict(val_x)
        print('R2 val= ', r2_score(val_y, val_yp))
        print('R2 val persistence =', r2_score(val_y[ahead:], val_y[0:-ahead]))

        test_yp = knnr.predict(test_x)
        print('R2 test= ', r2_score(test_y, test_yp))
        print('R2 test persistence =', r2_score(test_y[ahead:], test_y[0:-ahead]))

