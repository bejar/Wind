"""
.. module:: Data

Data
*************

:Description: Data

    

:Authors: bejar
    

:Version: 

:Created on: 19/02/2018 9:28 

"""

from __future__ import print_function
import numpy as np
from sklearn.preprocessing import StandardScaler
from netCDF4 import Dataset
from Wind.Config import wind_data
__author__ = 'bejar'


def lagged_vector(data, lag=1, ahead=0, s2s=False):
    """
    Returns a matrix with columns that are the steps of the lagged time series
    Last column is the value to predict
    :param data:
    :param lag:
    :return:
    """
    lvect = []
    if s2s:
        for i in range(lag + ahead):
            lvect.append(data[i: -lag - ahead + i])
    else:
        ahead -= 1
        for i in range(lag):
            lvect.append(data[i: -lag - ahead + i])
        lvect.append(data[lag + ahead:])

    return np.stack(lvect, axis=1)


def lagged_matrix(data, lag=1, ahead=0, s2s=False):
    """
    Returns a matrix with columns that are the steps of the lagged time series
    Last column is the value to predict
    :param data:
    :param lag:
    :return:
    """
    lvect = []

    if s2s:
        for i in range(lag + ahead):
            lvect.append(data[i: -lag - ahead + i, :])
    else:
        ahead -= 1
        for i in range(lag):
            lvect.append(data[i: -lag - ahead + i, :])
        lvect.append(data[lag + ahead:, :])
    return np.stack(lvect, axis=1)


def _generate_dataset_one_var(data, datasize, testsize, lag=1, ahead=1, s2s=False):
    """
    Generates
    :return:
    """
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    wind_train =  data[:datasize, :]
    # print(wind_train[:15, 0])

    train = lagged_vector(wind_train, lag=lag, ahead=ahead, s2s=s2s)
    if s2s:
        train_x, train_y = train[:, :lag], train[:, -ahead:, 0]
    else:
        train_x, train_y = train[:, :lag], train[:, -1:, 0]
    # print(train_y.shape)
    # print(train_x[0:5,:])
    # print(train_y[0:5, :])

    wind_test = data[datasize:datasize + testsize, 0].reshape(-1, 1)
    test = lagged_vector(wind_test, lag=lag, ahead=ahead, s2s=s2s)
    half_test = int(test.shape[0] / 2)

    if s2s:
        val_x, val_y = test[:half_test, :lag], test[:half_test, -ahead, 0]
        test_x, test_y = test[half_test:, :lag], test[half_test:, -ahead, 0]
    else:
        val_x, val_y = test[:half_test, :lag], test[:half_test, -1, 0]
        test_x, test_y = test[half_test:, :lag], test[half_test:, -1, 0]

    return train_x, train_y, val_x, val_y, test_x, test_y


def _generate_dataset_multiple_var(data, datasize, testsize, lag=1, ahead=1, s2s=False):
    """
    Generates
    :return:
    """
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    wind_train = data[:datasize, :]

    # Train
    train = lagged_matrix(wind_train, lag=lag, ahead=ahead, s2s=s2s)
    if s2s:
        train_x, train_y = train[:, :lag], train[:, -ahead:, 0]
    else:
        train_x, train_y = train[:, :lag], train[:, -1:, 0]

    # Test and Val
    wind_test = data[datasize:datasize + testsize, :]

    test = lagged_matrix(wind_test, lag=lag, ahead=ahead, s2s=s2s)
    half_test = int(test.shape[0] / 2)

    if s2s:
        val_x, val_y = test[:half_test, :lag], test[:half_test, -ahead:, 0]
        test_x, test_y = test[half_test:, :lag], test[half_test:, -ahead:, 0]
    else:
        val_x, val_y = test[:half_test, :lag], test[:half_test, -1, 0]
        test_x, test_y = test[half_test:, :lag], test[half_test:, -1, 0]

    return train_x, train_y, val_x, val_y, test_x, test_y


def generate_dataset(config, s2s=False):
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
    :param type: type of dataset
    :return:
    """
    datanames = config['datanames']
    datasize = config['datasize']
    testsize = config['testsize']
    lag = config['lag']
    ahead = config['ahead']
    vars = config['vars']
    wind = {}

    # Reads numpy arrays for all sites and keep only selected columns
    for d in datanames:
        wind[d] = np.load(wind_data + '/%s.npy' % d)
        if vars is not None:
            wind[d] = wind[d][:,vars]
            # print(wind[d].shape)

    if config['dataset'] == 0:
        return _generate_dataset_one_var(wind[datanames[0]][:, 0].reshape(-1, 1), datasize, testsize,
                                         lag=lag, ahead=ahead, s2s=s2s)
    elif config['dataset'] == 1:
        return _generate_dataset_multiple_var(wind[datanames[0]], datasize, testsize,
                                              lag=lag, ahead=ahead, s2s=s2s)
    elif config['dataset'] == 2:
        stacked = np.vstack([wind[d][:,0] for d in datanames]).T
        return _generate_dataset_multiple_var(stacked, datasize, testsize,
                                              lag=lag, ahead=ahead, s2s=s2s)
    elif config['dataset'] == 3:
        stacked = np.hstack([wind[d] for d in datanames])
        return _generate_dataset_multiple_var(stacked, datasize, testsize,
                                              lag=lag, ahead=ahead, s2s=s2s)
    raise NameError('ERROR: No such dataset type')


if __name__ == '__main__':
    from Wind.Util import load_config_file
    config = load_config_file('../../Experiments/config2.json')

    print(config)
    ldata = generate_dataset(config['data'], s2s=True)

    for d in ldata:
        print(d.shape)