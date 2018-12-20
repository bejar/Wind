"""
.. module:: MLPS2SArchitecture

MLPS2SArchitecture
*************

:Description: MLPS2SArchitecture

    

:Authors: bejar
    

:Version: 

:Created on: 04/09/2018 7:23 

"""

from Wind.Architectures.NNS2SArchitecture import NNS2SArchitecture
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l1, l2

try:
    from keras.utils import multi_gpu_model
except ImportError:
    _has_multigpu = False
else:
    _has_multigpu = True

__author__ = 'bejar'


class MLPS2SArchitecture(NNS2SArchitecture):
    """
    Multilayer perceptron sequence to sequence architecture
    """
    modfile = None
    modname = 'MLPS2S'
    ## Data mode 2 dimensional impur and 2 dimensional output
    data_mode = ('2D', '2D')  #'mlp'

    def generate_model(self):
        """
        Model for MLP multiple regression (s2s)

        :return:
        """

        activation = self.config['arch']['activation']
        dropout = self.config['arch']['drop']
        full_layers = self.config['arch']['full']

        # Extra added from training function
        idimensions = self.config['idimensions']
        odimension = self.config['odimensions']

        self.model = Sequential()
        self.model.add(Dense(full_layers[0], input_shape=idimensions, activation=activation))
        self.model.add(Dropout(rate=dropout))
        for units in full_layers[1:]:
            self.model.add(Dense(units=units, activation=activation))
            self.model.add(Dropout(rate=dropout))

        self.model.add(Dense(odimension, activation='linear'))


