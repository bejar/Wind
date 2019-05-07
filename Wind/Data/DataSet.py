"""
.. module:: DataSet

DataSet
*************

:Description: DataSet

    Generates a dataset from the data matrix

:Authors: bejar

:Version: 

:Created on: 06/07/2018 11:11 

"""

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from Wind.Config.Paths import remote_data, remote_wind_data_path
from Wind.Spatial.Util import get_all_neighbors, get_closest_k_neighbors
from Wind.Preprocessing.Normalization import tanh_normalization
import numpy as np
import os

try:
    import pysftp
except Exception:
    pass

__author__ = 'bejar'


def lagged_vector(data, lag=1, ahead=0, mode=None):
    """
    Returns a matrix with columns that are the steps of the lagged time series
    Last column is the value to predict
    :param mode:
    :param ahead:
    :param data:
    :param lag:
    :return:
    """
    lvect = []
    # if mode in ['s2s', 'mlp', 'cnn']:
    if mode[1] in ['2D', '3D']:
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

    :param mode:
    :param ahead:
    :param data:
    :param lag:
    :return:
    """
    lvect = []

    # if mode in ['s2s', 'mlp', 'cnn']:
    if mode[1] in ['2D', '3D']:
        for i in range(lag + ahead):
            lvect.append(data[i: -lag - ahead + i, :])
    else:
        ahead -= 1
        for i in range(lag):
            lvect.append(data[i: -lag - ahead + i, :])

        lvect.append(data[lag + ahead:, :])

    return np.stack(lvect, axis=1)


class Dataset:
    """
    Class to generate the data matrices (train, validation and test)
    """

    ## Train X matrix
    train_x = None
    ## Train y matrix
    train_y = None
    ## Validation X matrix
    val_x = None
    ## Validation y matrix
    val_y = None
    ## Test X matrix
    test_x = None
    ## Test y matrix
    test_y = None
    ## Path to the datafiles
    data_path = None
    ## Section 'data' of the configuration file
    config = None
    ## Mode of the dataset
    mode = None
    ## Functions to use for scaling the data
    scalers = {'standard': StandardScaler(), 'minmax': MinMaxScaler(feature_range=(-1, 1)),
               'tanh': tanh_normalization()}
    ## Strings corresponding to the different dataset configurations
    dataset_type = ['onesiteonevar', 'onesitemanyvar', 'manysiteonevar', 'manysitemanyvar', 'manysitemanyvarstack',
                    'manysitemanyvarstackneigh']

    def __init__(self, config, data_path):
        """
        Initializes the object with the data configuration section of the configuration file and
        the path where the actual data is

        :param config:
        :param data_path:
        """
        self.config = config
        self.data_path = data_path

    def is_teacher_force(self):
        """
        Returns if the data matrix is configured for teaching force

        :return:
        """
        return self.config['dmatrix'] == 'teach_force'

    def is_dependent_auxiliary(self):
        """
        Returns if the data matrix is cofigured to separate dependent and independent variables

        :return:
        """
        return self.config['dmatrix'] == 'dep_aux'

    def _generate_dataset_one_var(self, data, datasize, testsize, lag=1, ahead=1, slice=1, mode=None):
        """
        Generates dataset matrices for one variable according to the lag and ahead horizon. The ahead horizon can be
        sliced to a subset of the horizon

        The dimensions of the matrix are adapted accordingly to the input and output dimensions of the model

        Input:
            By default is a 3D matrix - examples x variables x lag
            2D - examples x (variables * lag)
        Output:
            3D - examples x horizon x 1
            2D - examples x horizon
            1D - examples x 1 x 1
            0D - examples x 1

        'scaling' is obtained from the data section of the configuration
        'fraction' allows selecting only a part of the data, selects from the end

        :param data:
        :param datasize:
        :param testsize:
        :param lag:
        :param ahead:
        :param slice:
        :param mode:
        :return:
        :return:
        """
        if 'scaler' in self.config and self.config['scaler'] in self.scalers:
            scaler = self.scalers[self.config['scaler']]
            data = scaler.fit_transform(data)
        else:
            scaler = StandardScaler()
            data = scaler.fit_transform(data)

        mode_x, mode_y = mode

        if 'fraction' in self.config:
            isize = int((1 - self.config['fraction']) * datasize)
            wind_train = data[isize:datasize, :]
        else:
            wind_train = data[:datasize, :]

        train = lagged_vector(wind_train, lag=lag, ahead=ahead, mode=mode)
        train_x = train[:, :lag]

        #######################################
        if mode_x == '2D':
            train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1]))
        elif mode_x == '4D':
            raise NameError('4D is not possible when there is only a variable')
        # Default is '3D'

        if mode_y == '3D':
            train_y = train[:, -slice:, 0]
            train_y = np.reshape(train_y, (train_y.shape[0], train_y.shape[1], 1))
        elif mode_y == '2D':
            train_y = train[:, -slice:, 0]
            train_y = np.reshape(train_y, (train_y.shape[0], train_y.shape[1]))
        elif mode_y == '1D':
            train_y = train[:, -1:, 0]
        elif mode_y == '0D':
            train_y = np.ravel(train[:, -1:, 0])
        else:
            train_y = train[:, -1:, 0]

        wind_test = data[datasize:datasize + testsize, 0].reshape(-1, 1)
        test = lagged_vector(wind_test, lag=lag, ahead=ahead, mode=mode)
        half_test = int(test.shape[0] / 2)
        val_x = test[:half_test, :lag]
        test_x = test[half_test:, :lag]

        #######################################
        if mode_x == '2D':
            val_x = np.reshape(val_x, (val_x.shape[0], val_x.shape[1]))
            test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1]))
        elif mode_x == '4D':
            raise NameError('4D is not possible when there is only a variable')
        # Default is '3D'

        if mode_y == '3D':
            val_y = test[:half_test, -slice:, 0]
            test_y = test[half_test:, -slice:, 0]
            val_y = np.reshape(val_y, (val_y.shape[0], val_y.shape[1], 1))
            test_y = np.reshape(test_y, (test_y.shape[0], test_y.shape[1], 1))
        elif mode_y == '2D':
            val_y = test[:half_test, -slice:, 0]
            test_y = test[half_test:, -slice:, 0]
            val_y = np.reshape(val_y, (val_y.shape[0], val_y.shape[1]))
            test_y = np.reshape(test_y, (test_y.shape[0], test_y.shape[1]))
        elif mode_y == '1D':
            val_y = test[:half_test, -1:, 0]
            test_y = test[half_test:, -1:, 0]
        elif mode_y == '0D':
            val_y = np.ravel(test[:half_test, -1:, 0])
            test_y = np.ravel(test[half_test:, -1:, 0])
        else:  # Default is '1D'
            val_y = test[:half_test, -1:, 0]
            test_y = test[half_test:, -1:, 0]

        return train_x, train_y, val_x, val_y, test_x, test_y

    def _generate_dataset_multiple_var(self, data, datasize, testsize, lag=1, ahead=1, slice=1, mode=None):
        """
        Generates dataset matrices for one variable according to the lag and ahead horizon. The ahead horizon can be
        sliced to a subset of the horizon

        The dimensions of the matrix are adapted accordingly to the input and output dimensions of the model

        Input:
            By default is a 3D matrix - examples x lag x variables
            2D - examples x (lag * variables)
        Output:
            3D - examples x horizon x 1
            2D - examples x horizon
            1D - examples x 1 x 1
            0D - examples x 1

        'scaling' is obtained from the data section of the configuration
        'fraction' allows selecting only a part of the data, selects from the end

        :return:
        """
        if 'scaler' in self.config and self.config['scaler'] in self.scalers:
            scaler = self.scalers[self.config['scaler']]
            data = scaler.fit_transform(data)
        else:
            scaler = StandardScaler()
            data = scaler.fit_transform(data)
        # print('DATA Dim =', data.shape)

        mode_x, mode_y = mode

        if 'fraction' in self.config:
            isize = int((1 - self.config['fraction']) * datasize)
            wind_train = data[isize:datasize, :]
        else:
            wind_train = data[:datasize, :]
        # print('Train Dim =', wind_train.shape)

        # Train
        train = lagged_matrix(wind_train, lag=lag, ahead=ahead, mode=mode)
        train_x = train[:, :lag]

        #######################################
        if mode_x == '2D':
            # Interchange axes 1 and 2 so the variables values are contiguous in the 2D matrix
            train_x = np.swapaxes(train_x, 1, 2)
            train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1] * train_x.shape[2]))
        elif mode_x == '4D':
            # Add an extra dimension to simulate that we have only one channel
            train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], train_x.shape[2], 1))


        if mode_y == '3D':
            train_y = train[:, -slice:, 0]
            train_y = np.reshape(train_y, (train_y.shape[0], train_y.shape[1], 1))
        elif mode_y == '2D':
            train_y = train[:, -slice:, 0]
            train_y = np.reshape(train_y, (train_y.shape[0], train_y.shape[1]))
        elif mode_y == '1D':
            train_y = train[:, -1:, 0]
        elif mode_y == '0D':
            train_y = np.ravel(train[:, -1:, 0])
        else:
            train_y = train[:, -slice:, 0]

        # Test and Val
        wind_test = data[datasize:datasize + testsize, :]
        test = lagged_matrix(wind_test, lag=lag, ahead=ahead, mode=mode)
        half_test = int(test.shape[0] / 2)
        val_x = test[:half_test, :lag]
        test_x = test[half_test:, :lag]

        ########################################################
        if mode_x == '2D':
            val_x = np.swapaxes(val_x, 1, 2)
            val_x = np.reshape(val_x, (val_x.shape[0], val_x.shape[1] * val_x.shape[2]))
            test_x = np.swapaxes(test_x, 1, 2)
            test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1] * test_x.shape[2]))
        elif mode_x == '4D':
            # Add an extra dimension to simulate that we have only one channel
            val_x = np.reshape(val_x, (val_x.shape[0], val_x.shape[1], val_x.shape[2], 1))
            test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], test_x.shape[2], 1))

        if mode_y == '3D':
            val_y = test[:half_test, -slice:, 0]
            test_y = test[half_test:, -slice:, 0]
            val_y = np.reshape(val_y, (val_y.shape[0], val_y.shape[1], 1))
            test_y = np.reshape(test_y, (test_y.shape[0], test_y.shape[1], 1))
        elif mode_y == '2D':
            val_y = test[:half_test, -slice:, 0]
            test_y = test[half_test:, -slice:, 0]
            val_y = np.reshape(val_y, (val_y.shape[0], val_y.shape[1]))
            test_y = np.reshape(test_y, (test_y.shape[0], test_y.shape[1]))
        elif mode_y == '1D':
            val_y = test[:half_test, -1:, 0]
            test_y = test[half_test:, -1:, 0]
        elif mode_y == '0D':
            val_y = np.ravel(test[:half_test, -1:, 0])
            test_y = np.ravel(test[half_test:, -1:, 0])
        else:
            val_y = test[:half_test, -slice:, 0]
            test_y = test[half_test:, -slice:, 0]

        return train_x, train_y, val_x, val_y, test_x, test_y

    def generate_dataset(self, ahead=1, mode=None, ensemble=False, ens_slice=None, remote=None):
        """
        Generates the dataset for training, test and validation

          0 = One site - wind
          1 = One site - all variables
          2 = All sites - wind
          3 = All sites - all variables
          4 = All sites - all variables stacked
          5 = Uses neighbor sites around a radius

        :param ens_slice: (not yet used)
        :param remote: Use remote data
        :param ensemble: (not yet used)
        :param datanames: Name of the wind datafiles
        :param ahead: number of steps ahead for prediction
        :param mode: type of dataset (pair indicating the type of dimension for input and output)
        :return:
        """
        self.mode = mode

        datanames = self.config['datanames']
        datasize = self.config['datasize']
        testsize = self.config['testsize']

        lag = self.config['lag']
        vars = self.config['vars']
        wind = {}
        if 'angle' in self.config:
            angle = self.config['angle']
        else:
            angle = False

        #ahead = self.config['ahead'] if (type(self.config['ahead']) == list) else [1, self.config['ahead']]

        if type(ahead) == list:
            dahead = ahead[1]
            slice = (ahead[1] - ahead[0]) + 1
        else:
            dahead = ahead
            slice = ahead

        # Augment the dataset with the closest neighbors
        if self.config['dataset'] == 5 or self.config['dataset'] == 31:
            if 'radius' not in self.config:
                raise NameError("Radius missing for neighbours augmented dataset")
            else:
                radius = self.config['radius']
            if 'nneighbors' in self.config:
                datanames = get_closest_k_neighbors(datanames[0], radius, self.config['nneighbors'])
            else:
                datanames = get_all_neighbors(datanames[0], radius)
        # Reads numpy arrays for all sites and keeps only selected columns
        for d in datanames:
            if remote:
                srv = pysftp.Connection(host=remote_data[0], username=remote_data[1])
                srv.get(remote_wind_data_path + f"/{d}.npy", self.data_path + f"/{d}.npy")
                srv.close()
            if angle:
                wind[d] = np.load(self.data_path+'_angle' + f"/{d}.npy")
            else:
                wind[d] = np.load(self.data_path + f"/{d}.npy")
            if remote:
                os.remove(self.data_path + f"/{d}.npy")

            # If there is a list in vars attribute it should be a list of integers
            if type(vars) == list:
                for v in vars:
                    if type(v) != int or v > wind[d].shape[1]:
                        raise NameError('Error in variable selection')
                wind[d] = wind[d][:, vars]

        if (self.config['dataset'] == 0) or (self.config['dataset'] == 'onesiteonevar'):
            if not ensemble:
                self.train_x, self.train_y, self.val_x, self.val_y, self.test_x, self.test_y = \
                    self._generate_dataset_one_var(wind[datanames[0]][:, 0].reshape(-1, 1), datasize, testsize,
                                                   lag=lag, ahead=dahead, slice=slice, mode=mode)
            else:
                self.train_x, self.train_y, self.val_x, self.val_y, self.test_x, self.test_y = \
                    self._generate_dataset_one_var(wind[datanames[0]][ens_slice[0]::ens_slice[1], 0].reshape(-1, 1),
                                                   datasize, testsize,
                                                   lag=lag, ahead=dahead, slice=slice, mode=mode)

        elif (self.config['dataset'] == 1) or (self.config['dataset'] == 'onesitemanyvar'):
            if not ensemble:
                self.train_x, self.train_y, self.val_x, self.val_y, self.test_x, self.test_y = \
                    self._generate_dataset_multiple_var(wind[datanames[0]], datasize, testsize,
                                                        lag=lag, ahead=dahead, slice=slice, mode=mode)
            else:
                self.train_x, self.train_y, self.val_x, self.val_y, self.test_x, self.test_y = \
                    self._generate_dataset_multiple_var(wind[datanames[0][ens_slice[0]::ens_slice[1], :]], datasize,
                                                        testsize,
                                                        lag=lag, ahead=dahead, slice=slice, mode=mode)

        elif self.config['dataset'] == 2 or self.config['dataset'] == 'manysiteonevar':
            stacked = np.vstack([wind[d][:, 0] for d in datanames]).T
            self.train_x, self.train_y, self.val_x, self.val_y, self.test_x, self.test_y = \
                self._generate_dataset_multiple_var(stacked, datasize, testsize,
                                                    lag=lag, ahead=dahead, slice=slice, mode=mode)
        elif self.config['dataset'] == 3 or self.config['dataset'] == 31 or self.config['dataset'] == 'manysitemanyvar':
            stacked = np.hstack([wind[d] for d in datanames])
            self.train_x, self.train_y, self.val_x, self.val_y, self.test_x, self.test_y = \
                self._generate_dataset_multiple_var(stacked, datasize, testsize,
                                                    lag=lag, ahead=dahead, slice=slice, mode=mode)
        elif self.config['dataset'] == 4 or self.config['dataset'] == 5 or \
                self.config['dataset'] == 'manysitemanyvarstack':
            stacked = [self._generate_dataset_multiple_var(wind[d], datasize, testsize,
                                                           lag=lag, ahead=dahead, slice=slice, mode=mode) for d in
                       datanames]

            self.train_x = np.vstack([x[0] for x in stacked])
            self.train_y = np.vstack([x[1] for x in stacked])

            self.val_x = stacked[0][2]
            self.val_y = stacked[0][3]
            self.test_x = stacked[0][4]
            self.test_y = stacked[0][5]
        else:
            raise NameError('ERROR: No such dataset type')

    def get_data_matrices(self):
        """
        Returns the data matrices for training, validation and test

        :return:
        """

        if not 'dmatrix' in self.config or self.config['dmatrix'] == 'normal':
            return self.train_x, self.train_y, self.val_x, self.val_y, self.test_x, self.test_y
        elif self.config['dmatrix'] == 'teach_force':
            return self.teacher_forcing()
        elif self.config['dmatrix'] == 'dep_aux':
            return self.dependent_auxiliary()
        elif self.config['dmatrix'] == 'future':
            return self.auxiliary_future()
        else:
            raise NameError("DataSet: No such dmatrix type")

    def teacher_forcing(self):
        """
        returns data matrices for teacher forcing/attention assuming that data is for RNN

        :return:
        """
        # Use the last element of wind traininig data as the first of teacher forcing
        tmp = self.train_x[:, -1, 0]
        tmp = tmp.reshape(tmp.shape[0], 1, 1)
        train_y_tf = np.concatenate((tmp, self.train_y[:, :-1, :]), axis=1)

        tmp = self.test_x[:, -1, 0]
        tmp = tmp.reshape(tmp.shape[0], 1, 1)
        test_y_tf = np.concatenate((tmp, self.test_y[:, :-1, :]), axis=1)

        tmp = self.val_x[:, -1, 0]
        tmp = tmp.reshape(tmp.shape[0], 1, 1)
        val_y_tf = np.concatenate((tmp, self.val_y[:, :-1, :]), axis=1)
        return [self.train_x, train_y_tf], self.train_y, \
               [self.val_x, val_y_tf], self.val_y, \
               [self.test_x, test_y_tf], self.test_y

    def dependent_auxiliary(self):
        """
        Return data matrices separating dependent variable from the rest

        This is for two headed architecture with dependent and auxiliary
        variables in separated branches
        :return:
        """
        horizon = self.config['lag']
        if self.mode[1] != '2D':
            return [self.train_x[:, :, 0].reshape(self.train_x.shape[0], self.train_x.shape[1], 1),
                    self.train_x[:, :, 1:]], self.train_y, \
                   [self.val_x[:, :, 0].reshape(self.val_x.shape[0], self.val_x.shape[1], 1),
                    self.val_x[:, :, 1:]], self.val_y, \
                   [self.test_x[:, :, 0].reshape(self.test_x.shape[0], self.test_x.shape[1], 1),
                    self.test_x[:, :, 1:]], self.test_y
        else:
            return [self.train_x[:, :horizon].train_x[:, horizon:,]], self.train_y, \
                   [self.val_x[:, :horizon], self.val_x[:, :horizon]], self.val_y, \
                   [self.test_x[:, :horizon], self.test_x[:, :horizon]], self.test_y


    def auxiliary_future(self):
        """
        Returns data matrices adding a matrix for the future for a subset of the auxiliary matrices

        :return:
        """
        # Future variable, just one for now
        datalag = self.config['lag']
        future = self.config['varsf'][0]
        ahead = self.config['ahead'] if (type(self.config['ahead']) == list) else [1, self.config['ahead']]
        if type(ahead) == list:
            dahead = ahead[1]
            slice = (ahead[1] - ahead[0]) + 1
        else:
            dahead = ahead
            slice = ahead

        if self.mode[1] != '2D':
            # The values of the future variable are dahead positions from the start
            train_x_future = self.train_x[dahead:, -slice:, future]
            val_x_future = self.val_x[dahead:, -slice:, future]
            test_x_future = self.test_x[dahead:, -slice:, future]
        else:
            nvars = len(self.config['vars'])
            train_x_future = self.train_x[datalag-1:, (future*datalag)+ahead[0]:(future*datalag)+ahead[0]+slice]
            val_x_future = self.val_x[datalag-1:, (future*datalag)+ahead[0]:(future*datalag)+ahead[0]+slice]
            test_x_future = self.test_x[datalag-1:, (future*datalag)+ahead[0]:(future*datalag)+ahead[0]+slice]

        # We lose the last datalag-1 examples because we do not have their full future in the data matrix
        return [self.train_x[:-(datalag-1)], train_x_future], self.train_y[:-(datalag-1)], [self.val_x[:-(datalag-1)], val_x_future], self.val_y[:-(datalag-1)],\
               [self.test_x[:-(datalag-1)], test_x_future], self.test_y[:-(datalag-1)]

    def summary(self):
        """
        Dataset Summary of its characteristics

        :return:
        """
        if self.train_x is None:
            raise NameError('Data not loaded yet')
        else:
            print("--- Dataset Configuration-----------")
            print(f"Dataset name: {self.config['datanames']}")
            if 'fraction' in self.config:
                print(f"Data fraction: {self.config['fraction']}")
            else:
                print(f"Data fraction: 2")
            print(f"Training:   X={self.train_x.shape} Y={self.train_y.shape}")
            print(f"Validation: X={self.val_x.shape} Y={self.val_y.shape}")
            print(f"Tests:      X={self.test_x.shape} T={self.test_y.shape}")
            if type(self.config['dataset']) == int:
                print(f"Dataset type= {self.dataset_type[self.config['dataset']]}")
            else:
                print(f"Dataset type= {self.config['dataset']}")
            if 'scaler' in self.config:
                print(f"Scaler= {self.config['scaler']}")
            else:
                print(f"Scaler= standard")
            if 'dmatrix' in self.config:
                print(f"Data matrix configuration= {self.config['dmatrix']}")
            print(f"Vars= {self.config['vars']}")
            print(f"Lag= {self.config['lag']}")
            print(f"Ahead= {self.config['ahead']}")
            print("------------------------------------")


if __name__ == '__main__':
    from Wind.Misc import load_config_file
    from Wind.Config import wind_data_path
    import matplotlib.pyplot as plt

    cfile = "config_MLP_s2s_fut"
    config = load_config_file(f"../TestConfigs/{cfile}.json")

    # print(config)
    mode = ('2D', '2D')
    dataset = Dataset(config=config['data'], data_path=wind_data_path)

    dataset.generate_dataset(ahead=[1, 12], mode=mode)
    # dataset.summary()

    dm = dataset.get_data_matrices()

    trainx, trainx_f = dm[0]
    trainy = dm[1]
    valx, valx_f = dm[2]
    testx, testx_f = dm[4]

    print(trainx.shape)
    print(trainx_f.shape)
    print(trainy.shape)

    # print(trainx[2,:18])
    # print(trainx[2,18:36])
    # print(trainx_f[0,:])
    # print(testx[2,:18])
    # print(testx[2,18:36])
    # print(testx_f[0,:])
    # print(valx[2,:18])
    # print(valx[2,18:36])
    # print(valx_f[0,:])

