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

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Flatten


__author__ = 'bejar'


class MLPDirRegressionArchitecture(NNArchitecture):
    """
    Multilayer perceptron for direct regression

    """
    modfile = None
    modname = 'MLPDir'
    data_mode = ('3D', '1D')  # False

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

