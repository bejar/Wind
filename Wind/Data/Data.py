"""
.. module:: Data

Data
*************

:Description: Data

    Generates a dataset for the different experiments:




:Authors: bejar
    

:Version: 

:Created on: 19/02/2018 9:28 

"""

from __future__ import print_function
import numpy as np
from sklearn.preprocessing import StandardScaler
from netCDF4 import Dataset
from Wind.Config import wind_data_path
__author__ = 'bejar'


def lagged_vector(data, lag=1, ahead=0, mode=None):
    """
    Returns a matrix with columns that are the steps of the lagged time series
    Last column is the value to predict
    :param data:
    :param lag:
    :return:
    """
    lvect = []
    if mode in ['s2s', 'mlp']:
        for i in range(lag + ahead):
            lvect.append(data[i: -lag - ahead + i])
    else:
        ahead -= 1
        for i in range(lag):
            lvect.append(data[i: -lag - ahead + i])
        lvect.append(data[lag + ahead:])

    return np.stack(lvect, axis=1)


def lagged_matrix(data, lag=1, ahead=0, mode=None):
    """
    Returns a matrix with columns that are the steps of the lagged time series
    Last column is the value to predict
    :param data:
    :param lag:
    :return:
    """
    lvect = []

    if mode in ['s2s', 'mlp']:
        for i in range(lag + ahead):
            lvect.append(data[i: -lag - ahead + i, :])
    else:
        ahead -= 1
        for i in range(lag):
            lvect.append(data[i: -lag - ahead + i, :])
        lvect.append(data[lag + ahead:, :])
    return np.stack(lvect, axis=1)


def _generate_dataset_one_var(data, datasize, testsize, lag=1, ahead=1, mode=None):
    """
    Generates
    :return:
    """
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    # print('DATA Dim =', data.shape)

    wind_train =  data[:datasize, :]
    # print('Train Dim =', wind_train.shape)

    train = lagged_vector(wind_train, lag=lag, ahead=ahead, mode=mode)
    if mode == 's2s':
        train_x, train_y = train[:, :lag], train[:, -ahead:, 0]
        train_y = np.reshape(train_y, (train_y.shape[0], train_y.shape[1], 1))
    elif mode == 'mlp':
        train_x, train_y = train[:, :lag], train[:, -ahead:, 0]
        train_x = np.reshape(train_x, (train_y.shape[0], train_y.shape[1]))
    else:
        train_x, train_y = train[:, :lag], train[:, -1:, 0]

    wind_test = data[datasize:datasize + testsize, 0].reshape(-1, 1)
    test = lagged_vector(wind_test, lag=lag, ahead=ahead, mode=mode)
    half_test = int(test.shape[0] / 2)

    if mode == 's2s':
        val_x, val_y = test[:half_test, :lag], test[:half_test, -ahead:, 0]
        test_x, test_y = test[half_test:, :lag], test[half_test:, -ahead:, 0]
        val_y = np.reshape(val_y, (val_y.shape[0], val_y.shape[1], 1))
        test_y = np.reshape(test_y, (test_y.shape[0], test_y.shape[1], 1))
    elif mode == 'mlp':
        val_x, val_y = test[:half_test, :lag], test[:half_test, -ahead:, 0]
        test_x, test_y = test[half_test:, :lag], test[half_test:, -ahead:, 0]
        val_x = np.reshape(val_x, (val_x.shape[0], val_x.shape[1]))
        test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1]))
    else:
        val_x, val_y = test[:half_test, :lag], test[:half_test, -1:, 0]
        test_x, test_y = test[half_test:, :lag], test[half_test:, -1:, 0]

    return train_x, train_y, val_x, val_y, test_x, test_y


def _generate_dataset_multiple_var(data, datasize, testsize, lag=1, ahead=1, mode=None):
    """
    Generates
    :return:
    """
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    # print('DATA Dim =', data.shape)

    wind_train = data[:datasize, :]
    # print('Train Dim =', wind_train.shape)

    # Train
    train = lagged_matrix(wind_train, lag=lag, ahead=ahead, mode=mode)
    if mode == 's2s':
        train_x, train_y = train[:, :lag], train[:, -ahead:, 0]
        train_y = np.reshape(train_y, (train_y.shape[0], train_y.shape[1], 1))
    elif mode == 'mlp':
        train_x, train_y = train[:, :lag], train[:, -ahead:, 0]
        train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1] * train_x.shape[2]))
    else:
        train_x, train_y = train[:, :lag], train[:, -1:, 0]

    # Test and Val
    wind_test = data[datasize:datasize + testsize, :]

    test = lagged_matrix(wind_test, lag=lag, ahead=ahead, mode=mode)
    half_test = int(test.shape[0] / 2)

    if mode == 's2s':
        val_x, val_y = test[:half_test, :lag], test[:half_test, -ahead:, 0]
        test_x, test_y = test[half_test:, :lag], test[half_test:, -ahead:, 0]
        val_y = np.reshape(val_y, (val_y.shape[0], val_y.shape[1], 1))
        test_y = np.reshape(test_y, (test_y.shape[0], test_y.shape[1], 1))
    elif mode == 'mlp':
        val_x, val_y = test[:half_test, :lag], test[:half_test, -ahead:, 0]
        test_x, test_y = test[half_test:, :lag], test[half_test:, -ahead:, 0]
        val_x = np.reshape(val_x, (val_x.shape[0], val_x.shape[1] * val_x.shape[2]))
        test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1] * test_x.shape[2]))
    else:
        val_x, val_y = test[:half_test, :lag], test[:half_test, -1:, 0]
        test_x, test_y = test[half_test:, :lag], test[half_test:, -1:, 0]

    return train_x, train_y, val_x, val_y, test_x, test_y


def generate_dataset(config, ahead=1, mode=None, data_path=None):
    """
    Generates the dataset for training, test and validation

      0 = One site - wind
      1 = One site - all variables
      2 = All sites - wind
      3 = All sites - all variables
      4 = All sites - all variables stacked


    :param datanames: Name of the wind datafiles
    :param vars: List with the indices of the variables to use
    :param ahead: number of steps ahead for prediction
    :param mode: type of dataset
            None (recurrent one output regression)
            's2s' (recurrent multiple output regression)
            'mlp' (plain n layer MLP for regression)
    :return:
    """
    datanames = config['datanames']
    datasize = config['datasize']
    testsize = config['testsize']
    lag = config['lag']
    vars = config['vars']
    wind = {}

    # Reads numpy arrays for all sites and keep only selected columns
    for d in datanames:
        wind[d] = np.load(data_path + '/%s.npy' % d)
        if vars is not None:
            wind[d] = wind[d][:,vars]

    if config['dataset'] == 0:
        return _generate_dataset_one_var(wind[datanames[0]][:, 0].reshape(-1, 1), datasize, testsize,
                                         lag=lag, ahead=ahead, mode=mode)
    elif config['dataset'] == 1:
        return _generate_dataset_multiple_var(wind[datanames[0]], datasize, testsize,
                                              lag=lag, ahead=ahead, mode=mode)
    elif config['dataset'] == 2:
        stacked = np.vstack([wind[d][:,0] for d in datanames]).T
        return _generate_dataset_multiple_var(stacked, datasize, testsize,
                                              lag=lag, ahead=ahead, mode=mode)
    elif config['dataset'] == 3:
        stacked = np.hstack([wind[d] for d in datanames])
        return _generate_dataset_multiple_var(stacked, datasize, testsize,
                                              lag=lag, ahead=ahead, mode=mode)
    elif config['dataset'] == 4:
        stacked = [_generate_dataset_multiple_var(wind[d], datasize, testsize,
                                              lag=lag, ahead=ahead) for d in datanames]


        train_x = np.vstack([x[0] for x in stacked])
        train_y = np.vstack([x[1] for x in stacked])

        val_x = stacked[0][2]
        val_y = stacked[0][3]
        test_x = stacked[0][4]
        test_y = stacked[0][5]
        return train_x, train_y, val_x, val_y, test_x, test_y

    raise NameError('ERROR: No such dataset type')


if __name__ == '__main__':
    from Wind.Util import load_config_file
    config = load_config_file('../../Experiments/config2.json')

    # print(config)
    train_x, train_y, val_x, val_y, test_x, test_y = generate_dataset(config['data'], ahead=6, mode='mlp', data_path='../../Data')

    print(train_x.shape)
    # print(train_x[0:5,:])

    print(train_y.shape)
    # print(train_y[0:5,:])

    print(test_x.shape)
    print(test_y.shape)
    print(val_x.shape)
    print(val_y.shape)
