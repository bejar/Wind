"""
.. module:: MLPS2SArchitecture

MLPS2SArchitecture
*************

:Description: MLPS2SArchitecture

    

:Authors: bejar
    

:Version: 

:Created on: 04/09/2018 7:23 

"""

from tensorflow.keras.layers import Dense, Dropout, GaussianNoise, Input, BatchNormalization, Concatenate
from tensorflow.keras.models import Model

from Wind.Architectures.NNS2SArchitecture import NNS2SArchitecture
from Wind.Train.Activations import generate_activation

__author__ = 'bejar'


class MLPCascadeS2SArchitecture(NNS2SArchitecture):
    """
    Multilayer perceptron sequence to sequence architecture
    """
    modfile = None
    modname = 'MLPCASS2S'
    ## Data mode 2 dimensional input and 2 dimensional output
    data_mode = ('2D', '2D')  # 'mlp'

    def generate_model(self):
        """
        Model for MLP multiple regression (s2s)

        :return:
        """

        activation = self.config['arch']['activation']
        dropout = self.config['arch']['drop']
        full_layers = self.config['arch']['full']

        # Adding the possibility of using a GaussianNoise Layer for regularization
        if 'noise' in self.config['arch']:
            noise = self.config['arch']['noise']
        else:
            noise = 0

        if 'batchnorm' in self.config['arch']:
            bnorm = self.config['arch']['batchnorm']
        else:
            bnorm = False

        # Extra added from training function
        idimensions = self.config['idimensions']
        odimension = self.config['odimensions']

        data_input = Input(shape=(idimensions))

        if noise != 0:
            layer = GaussianNoise(noise)(data_input)
            layer = Dense(full_layers[0])(layer)

            layer = generate_activation(activation)(layer)
            layer = Dropout(rate=dropout)(layer)
        else:
            layer = Dense(full_layers[0])(data_input)
            if bnorm:
                layer = BatchNormalization()(layer)
            layer = generate_activation(activation)(layer)
            layer = Dropout(rate=dropout)(layer)

        for units in full_layers[1:]:
            layer = Concatenate()([data_input, layer])
            layer = Dense(units=units)(layer)
            if bnorm:
                layer = BatchNormalization()(layer)
            layer = generate_activation(activation)(layer)

            layer = Dropout(rate=dropout)(layer)

        output = Dense(odimension, activation='linear')(layer)

        self.model = Model(inputs=data_input, outputs=output)
