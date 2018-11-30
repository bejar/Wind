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
    data_mode = (False, False)  # False

    def generate_model(self):
        """
        Model for MLP with direct regression

        :return:
        """
        activation = self.config['arch']['activation']
        dropout = self.config['arch']['drop']
        full_layers = self.config['arch']['full']

        # Extra added from training function
        idimensions = self.config['idimensions']


        self.model = Sequential()
        self.model.add(Dense(full_layers[0], input_shape=idimensions, activation=activation))
        self.model.add(Dropout(rate=dropout))
        for units in full_layers[1:]:
            self.model.add(Dense(units=units, activation=activation))
            self.model.add(Dropout(rate=dropout))
        self.model.add(Flatten())
        self.model.add(Dense(1, activation='linear'))

    def summary(self):
        self.model.summary()
        activation = self.config['arch']['activation']
        print(
        f"lag: {self.config['data']['lag']} /Layers: {str(self.config['arch']['full'])} /Activation: {activation}")

        print()

    def log_result(self, result):
        for i, r2val, r2test in result:
            print(f"{self.config['arch']['mode']}|"
                  f"DNM={self.config['data']['datanames'][0]},"
                  f"DS={self.config['data']['dataset']},"
                  f"V={len(self.config['data']['vars'])},"
                  f"LG={self.config['data']['lag']},"
                  f"AH={i},"
                  f"FL={str(self.config['arch']['full'])},"
                  f"DR={self.config['arch']['drop']},"
                  f"AF={self.config['arch']['activation']},"
                  f"OPT={self.config['training']['optimizer']},"
                  f"R2V={r2val:3.5f},"
                  f"R2T={r2test:3.5f}"
                  )
            # print('%s | DNM= %s, DS= %d, V= %d, LG= %d, AH= %d, FL= %s, DR= %3.2f, AF= %s, '
            #       'OPT= %s, R2V = %3.5f, R2T = %3.5f' %
            #       (self.config['arch']['mode'],
            #        self.config['data']['datanames'][0],
            #        self.config['data']['dataset'],
            #        len(self.config['data']['vars']),
            #        self.config['data']['lag'],
            #        i, str(self.config['arch']['full']),
            #        self.config['arch']['drop'],
            #        self.config['arch']['activation'],
            #        self.config['training']['optimizer'],
            #        r2val,
            #        r2test,
            #        ))
