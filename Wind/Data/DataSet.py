"""
.. module:: DataSet

DataSet
*************

:Description: DataSet

    

:Authors: bejar
    

:Version: 

:Created on: 06/07/2018 11:11 

"""

from __future__ import print_function
import numpy as np
from sklearn.preprocessing import StandardScaler

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


class Dataset:
    train_x = None
    train_y = None
    val_x = None
    val_y = None
    test_x = None
    test_y = None
    data_path = None
    config = None

    def __init__(self, config, data_path):
        self.config = config
        self.data_path = data_path

    def _generate_dataset_one_var(self, data, datasize, testsize, lag=1, ahead=1, slice=1, mode=None):
        """
        Generates
        :return:
        """
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        # print('DATA Dim =', data.shape)

        wind_train = data[:datasize, :]
        # print('Train Dim =', wind_train.shape)

        train = lagged_vector(wind_train, lag=lag, ahead=ahead, mode=mode)
        if mode == 's2s':
            train_x, train_y = train[:, :lag], train[:, -slice:, 0]
            train_y = np.reshape(train_y, (train_y.shape[0], train_y.shape[1], 1))
        elif mode == 'mlp':
            train_x, train_y = train[:, :lag], train[:, -slice:, 0]
            train_x = np.reshape(train_x, (train_y.shape[0], train_y.shape[1]))
        elif mode == 'svm':
            train_x, train_y = train[:, :lag], np.ravel(train[:, -1:, 0])
            train_x = np.reshape(train_x, (train_y.shape[0], train_y.shape[1]))
        else:
            train_x, train_y = train[:, :lag], train[:, -1:, 0]

        wind_test = data[datasize:datasize + testsize, 0].reshape(-1, 1)
        test = lagged_vector(wind_test, lag=lag, ahead=ahead, mode=mode)
        half_test = int(test.shape[0] / 2)

        if mode == 's2s':
            val_x, val_y = test[:half_test, :lag], test[:half_test, -slice:, 0]
            test_x, test_y = test[half_test:, :lag], test[half_test:, -slice:, 0]
            val_y = np.reshape(val_y, (val_y.shape[0], val_y.shape[1], 1))
            test_y = np.reshape(test_y, (test_y.shape[0], test_y.shape[1], 1))
        elif mode == 'mlp':
            val_x, val_y = test[:half_test, :lag], test[:half_test, -slice:, 0]
            test_x, test_y = test[half_test:, :lag], test[half_test:, -slice:, 0]
            val_x = np.reshape(val_x, (val_x.shape[0], val_x.shape[1]))
            test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1]))
        elif mode == 'svm':
            val_x, val_y = test[:half_test, :lag], np.ravel(test[:half_test, -1:, 0])
            test_x, test_y = test[half_test:, :lag], np.ravel(test[half_test:, -1:, 0])
            val_x = np.reshape(val_x, (val_x.shape[0], val_x.shape[1]))
            test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1]))
        else:
            val_x, val_y = test[:half_test, :lag], test[:half_test, -1:, 0]
            test_x, test_y = test[half_test:, :lag], test[half_test:, -1:, 0]

        return train_x, train_y, val_x, val_y, test_x, test_y

    def _generate_dataset_multiple_var(self, data, datasize, testsize, lag=1, ahead=1, slice=1, mode=None):
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
        train = self.lagged_matrix(wind_train, lag=lag, ahead=ahead, mode=mode)
        if mode == 's2s':
            train_x, train_y = train[:, :lag], train[:, -slice:, 0]
            train_y = np.reshape(train_y, (train_y.shape[0], train_y.shape[1], 1))
        elif mode == 'mlp':
            train_x, train_y = train[:, :lag], train[:, -slice:, 0]
            train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1] * train_x.shape[2]))
        elif mode == 'svm':
            train_x, train_y = train[:, :lag], np.ravel(train[:, -1:, 0])
            train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1] * train_x.shape[2]))
        else:
            train_x, train_y = train[:, :lag], train[:, -1:, 0]

        # Test and Val
        wind_test = data[datasize:datasize + testsize, :]
        test = lagged_matrix(wind_test, lag=lag, ahead=ahead, mode=mode)
        half_test = int(test.shape[0] / 2)

        if mode == 's2s':
            val_x, val_y = test[:half_test, :lag], test[:half_test, -slice:, 0]
            test_x, test_y = test[half_test:, :lag], test[half_test:, -slice:, 0]
            val_y = np.reshape(val_y, (val_y.shape[0], val_y.shape[1], 1))
            test_y = np.reshape(test_y, (test_y.shape[0], test_y.shape[1], 1))
        elif mode == 'mlp':
            val_x, val_y = test[:half_test, :lag], test[:half_test, -slice:, 0]
            test_x, test_y = test[half_test:, :lag], test[half_test:, -slice:, 0]
            val_x = np.reshape(val_x, (val_x.shape[0], val_x.shape[1] * val_x.shape[2]))
            test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1] * test_x.shape[2]))
        elif mode == 'svm':
            val_x, val_y = test[:half_test, :lag], np.ravel(test[:half_test, -1:, 0])
            test_x, test_y = test[half_test:, :lag], np.ravel(test[half_test:, -1:, 0])
            val_x = np.reshape(val_x, (val_x.shape[0], val_x.shape[1] * val_x.shape[2]))
            test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1] * test_x.shape[2]))
        else:
            val_x, val_y = test[:half_test, :lag], test[:half_test, -1:, 0]
            test_x, test_y = test[half_test:, :lag], test[half_test:, -1:, 0]

        return train_x, train_y, val_x, val_y, test_x, test_y

    def generate_dataset(self, ahead=1, mode=None, ensemble=False, ens_slice=None):
        """
        Generates the dataset for training, test and validation

          0 = One site - wind
          1 = One site - all variables
          2 = All sites - wind
          3 = All sites - all variables
          4 = All sites - all variables stacked


        :param datanames: Name of the wind datafiles
        :param ahead: number of steps ahead for prediction
        :param mode: type of dataset
                None (recurrent one output regression)
                's2s' (recurrent multiple output regression)
                'mlp' (plain n layer MLP for regression)
        :return:
        """
        datanames = self.config['datanames']
        datasize = self.config['datasize']
        testsize = self.config['testsize']

        lag = self.config['lag']
        vars = self.config['vars']
        wind = {}

        if (mode == 's2s' or mode == 'mlp') and type(ahead) == list:
            dahead = ahead[1]
            slice = (ahead[1] - ahead[0]) + 1
        else:
            dahead = ahead
            slice = ahead

        # Reads numpy arrays for all sites and keep only selected columns
        for d in datanames:
            wind[d] = np.load(self.data_path + '/%s.npy' % d)
            if vars is not None:
                wind[d] = wind[d][:, vars]

        if self.config['dataset'] == 0:
            if not ensemble:
                self.train_x, self.train_y, self.val_x, self.val_y, self.test_x, self.test_y = \
                    self._generate_dataset_one_var(wind[datanames[0]][:, 0].reshape(-1, 1), datasize, testsize,
                                                   lag=lag, ahead=dahead, slice=slice, mode=mode)
            else:
                self.train_x, self.train_y, self.val_x, self.val_y, self.test_x, self.test_y = \
                    self._generate_dataset_one_var(wind[datanames[0]][ens_slice[0]::ens_slice[1], 0].reshape(-1, 1),
                                                   datasize, testsize,
                                                   lag=lag, ahead=dahead, slice=slice, mode=mode)

        elif self.config['dataset'] == 1:
            if not ensemble:
                self.train_x, self.train_y, self.val_x, self.val_y, self.test_x, self.test_y = \
                    self._generate_dataset_multiple_var(wind[datanames[0]], datasize, testsize,
                                                        lag=lag, ahead=dahead, slice=slice, mode=mode)
            else:
                self.train_x, self.train_y, self.val_x, self.val_y, self.test_x, self.test_y = \
                    self._generate_dataset_multiple_var(wind[datanames[0][ens_slice[0]::ens_slice[1], :]], datasize,
                                                        testsize,
                                                        lag=lag, ahead=dahead, slice=slice, mode=mode)

        elif self.config['dataset'] == 2:
            stacked = np.vstack([wind[d][:, 0] for d in datanames]).T
            return self._generate_dataset_multiple_var(stacked, datasize, testsize,
                                                       lag=lag, ahead=dahead, slice=slice, mode=mode)
        elif self.config['dataset'] == 3:
            stacked = np.hstack([wind[d] for d in datanames])
            self.train_x, self.train_y, self.val_x, self.val_y, self.test_x, self.test_y = \
                self._generate_dataset_multiple_var(stacked, datasize, testsize,
                                                    lag=lag, ahead=dahead, slice=slice, mode=mode)
        elif self.config['dataset'] == 4:
            stacked = [self._generate_dataset_multiple_var(wind[d], datasize, testsize,
                                                           lag=lag, ahead=dahead, slice=slice) for d in datanames]

            self.train_x = np.vstack([x[0] for x in stacked])
            self.train_y = np.vstack([x[1] for x in stacked])

            self.val_x = stacked[0][2]
            self.val_y = stacked[0][3]
            self.test_x = stacked[0][4]
            self.test_y = stacked[0][5]
            # return train_x, train_y, val_x, val_y, test_x, test_y

        raise NameError('ERROR: No such dataset type')

    def summary(self):
        """
        Dataset Summary

        :return:
        """
        if self.train_x is None:
            raise NameError('Data not loaded yet')
        else:
            print('Tr:', self.train_x.shape, self.train_y.shape)
            print('Val:', self.val_x.shape, self.val_y.shape)
            print('Ts:', self.test_x.shape, self.test_y.shape)
            print('Dtype=', self.config['dataset'])
            print('Lag=', self.config['lag'])
            print('Vars=', self.config['vars'])
