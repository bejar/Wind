"""
.. module:: MLPS2SSeparateArchitecture

MLPS2SArchitecture
*************

:Description: MLPS2SSeparateArchitecture

    Data has multiple data sites, the additional sites are processed by a different branch and then combined.

:Authors: bejar
    

:Version: 

:Created on: 04/09/2018 7:23 

"""

from tensorflow.keras.layers import Dense, Dropout, Input, concatenate
from tensorflow.keras.models import Model

from Wind.Architectures.NNS2SArchitecture import NNS2SArchitecture
from Wind.Train.Activations import generate_activation

__author__ = 'bejar'


class MLPS2SSeparateArchitecture(NNS2SArchitecture):
    """
    Multilayer perceptron sequence to sequence architecture
    with additional sites
    """
    modfile = None
    modname = 'MLPS2SSeparate'
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
        fulladd_layers = self.config['arch']['fulladd']

        # Extra added from training function
        idimensions = self.config['idimensions']
        odimension = self.config['odimensions']

        # Dependent variable input
        data_input = Input(shape=(idimensions[0]))

        mlp_layers = Dense(full_layers[0])(data_input)
        mlp_layers = generate_activation(activation)(mlp_layers)
        mlp_layers = Dropout(rate=dropout)(mlp_layers)
        for units in full_layers[1:]:
            mlp_layers = Dense(units=units)(mlp_layers)
            mlp_layers = generate_activation(activation)(mlp_layers)
            mlp_layers = Dropout(rate=dropout)(mlp_layers)

        dataadd_input = Input(shape=(idimensions[1]))

        mlpadd_layers = Dense(full_layers[0])(dataadd_input)
        mlpadd_layers = generate_activation(activation)(mlpadd_layers)
        mlpadd_layers = Dropout(rate=dropout)(mlpadd_layers)
        for units in fulladd_layers[1:]:
            mlpadd_layers = Dense(units=units)(mlpadd_layers)
            mlpadd_layers = generate_activation(activation)(mlpadd_layers)
            mlpadd_layers = Dropout(rate=dropout)(mlpadd_layers)

        fusion = self.config['arch']['funits']
        mlp_layers = concatenate([mlp_layers, mlpadd_layers])

        mlp_layers = Dense(units=fusion)(mlp_layers)
        mlp_layers = generate_activation(activation)(mlp_layers)
        mlp_layers = Dropout(rate=dropout)(mlp_layers)

        output = Dense(odimension, activation='linear')(mlp_layers)
        self.model = Model(inputs=[data_input, dataadd_input], outputs=output)
