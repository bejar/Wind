"""
.. module:: MLPDirRegressionArchitecture

MLPDirRegressionArchitecture
*************

:Description: MLPDirRegressionArchitecture

    

:Authors: bejar
    

:Version: 

:Created on: 04/09/2018 7:58 

"""
from Wind.Architectures.NNArchitecture import NNArchitecture


from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
from keras.layers import Flatten


try:
    from keras.utils import multi_gpu_model
except ImportError:
    _has_multigpu = False
else:
    _has_multigpu = True

from keras.optimizers import RMSprop, SGD
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.regularizers import l1, l2
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score
from Wind.Data import generate_dataset
from Wind.Config import wind_data_path
from Wind.Training import updateprocess
from time import time, strftime
import os


__author__ = 'bejar'

class MLPDirRegressionArchitecture(NNArchitecture):

    modfile = None
    modname = 'MLPDir'

    def generate_model(self):
        """
        Model for MLP with direct regression

        :return:
        """

        activation = self.config['arch']['activation']

        dropout = self.config['arch']['drop']

        # Extra added from training function
        idimensions = self.config['idimensions']
        full_layers = self.config['arch']['full']

        model = Sequential()
        model.add(Dense(full_layers[0], input_shape=idimensions, activation=activation))
        model.add(Dropout(rate=dropout))
        for units in full_layers[1:]:
            model.add(Dense(units=units, activation=activation))
            model.add(Dropout(rate=dropout))
        model.add(Flatten())
        model.add(Dense(1, activation='linear'))

        return model

    def summary(self):
        self.model.summary()
        activation = self.config['arch']['activation']
        nlayers = self.config['arch']['nlayers']
        print('lag: ', self.config['data']['lag'], '/Layers: ', nlayers, '/Activation:', activation)
        print()

    def log_result(self, result):
        for i, r2val, r2test in result:
            print('%s | DNM= %s, DS= %d, V= %d, LG= %d, AH= %d, FL= %s, DR= %3.2f, AF= %s, '
                  'OPT= %s, R2V = %3.5f, R2T = %3.5f' %
                  (self.config['arch']['mode'],
                   self.config['data']['datanames'][0],
                   self.config['data']['dataset'],
                   self.len(self.config['data']['vars']),
                   self.config['data']['lag'],
                   i, str(self.config['arch']['full']),
                   self.config['arch']['drop'],
                   self.config['arch']['activation'],
                   self.config['training']['optimizer'],
                   r2val,
                   r2test,
                   ))