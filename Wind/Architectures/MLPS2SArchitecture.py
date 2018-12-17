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
    modfile = None
    modname = 'MLPS2S'
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


    # def summary(self):
    #     self.model.summary()
    #     activation = self.config['arch']['activation']
    #     print(
    #     f"lag: {self.config['data']['lag']} /Layers: {str(self.config['arch']['full'])} /Activation: {activation}")
    #     print()

    # def log_result(self, result):
    #     for i, r2val, r2test in result:
    #         print(f"{self.config['arch']['mode']} |"
    #               f"DNM={self.config['data']['datanames'][0]},"
    #               f"DS={self.config['data']['dataset']},"
    #               f"V={len(self.config['data']['vars'])},"
    #               f"LG={self.config['data']['lag']},"
    #               f"AH={i},"
    #               f"FL={str(self.config['arch']['full'])},"
    #               f"DR={self.config['arch']['drop']},"
    #               f"AF={self.config['arch']['activation']},"
    #               f"OPT={self.config['training']['optimizer']},"
    #               f"R2V={r2val:3.5f},"
    #               f"R2T={r2test:3.5f}"
    #               )
