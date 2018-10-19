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
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
from keras.regularizers import l1, l2

try:
    from keras.utils import multi_gpu_model
except ImportError:
    _has_multigpu = False
else:
    _has_multigpu = True

from sklearn.metrics import mean_squared_error, r2_score


__author__ = 'bejar'


class MLPS2SArchitecture(NNS2SArchitecture):
    modfile = None
    modname = 'MLPS2S'
    data_mode = 'mlp'

    def generate_model(self):
        """
        Model for RNN multiple regression (s2s)

        :return:
        """

        activation = self.config['arch']['activation']

        dropout = self.config['arch']['drop']

        # Extra added from training function
        idimensions = self.config['idimensions']
        odimension = self.config['odimensions']
        full_layers = self.config['arch']['full']

        self.model = Sequential()
        self.model.add(Dense(full_layers[0], input_shape=idimensions, activation=activation))
        self.model.add(Dropout(rate=dropout))
        for units in full_layers[1:]:
            self.model.add(Dense(units=units, activation=activation))
            self.model.add(Dropout(rate=dropout))

        self.model.add(Dense(odimension, activation='linear'))


    def summary(self):
        self.model.summary()
        neurons = self.config['arch']['neurons']
        activation = self.config['arch']['activation']
        nlayers = self.config['arch']['nlayers']
        print(
        'lag: ', self.config['data']['lag'], '/Neurons: ', neurons, '/Layers: ', nlayers, '/Activation:', activation)
        print('/full layers:', self.config['arch']['full'])
        print()

    def log_result(self, result):
        for i, r2val, r2test in result:
            print('%s | DNM= %s, DS= %d, V= %d, LG= %d, AH= %d, FL= %s, DR= %3.2f, AF= %s, '
                  'OPT= %s, R2V = %3.5f, R2T = %3.5f' %
                  (self.config['arch']['mode'],
                   self.config['data']['datanames'][0],
                   self.config['data']['dataset'],
                   len(self.config['data']['vars']),
                   self.config['data']['lag'],
                   i, str(self.config['arch']['full']),
                   self.config['arch']['drop'],
                   self.config['arch']['activation'],
                   self.config['training']['optimizer'],
                   r2val,
                   r2test,
                   ))
