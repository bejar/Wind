"""
.. module:: MLPS2SArchitecture

MLPS2SArchitecture
*************

:Description: MLPS2SArchitecture

    

:Authors: bejar
    

:Version: 

:Created on: 04/09/2018 7:23 

"""

from tensorflow.keras.layers import Dense, Dropout, Input, concatenate
from tensorflow.keras.models import Model

from Wind.Architectures.NNS2SArchitecture import NNS2SArchitecture
from Wind.Train.Activations import generate_activation

__author__ = 'bejar'


class MLPS2SFutureArchitecture(NNS2SArchitecture):
    """
    Multilayer perceptron sequence to sequence architecture
    with future values variables
    """
    modfile = None
    modname = 'MLPS2SFuture'
    ## Data mode 2 dimensional input and 2 dimensional output
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


        # Dependent variable input
        data_input = Input(shape=(idimensions[0]))
        future_input = Input(shape=(idimensions[1]))

        to_mlp = concatenate([data_input,future_input])

        mlp_layers =  Dense(full_layers[0])(to_mlp)
        mlp_layers = generate_activation(activation)(mlp_layers)
        mlp_layers = Dropout(rate=dropout)(mlp_layers)
        for units in full_layers[1:]:
            mlp_layers = Dense(units=units)(mlp_layers)
            mlp_layers = generate_activation(activation)(mlp_layers)
            mlp_layers = Dropout(rate=dropout)(mlp_layers)

        output = Dense(odimension, activation='linear')(mlp_layers)
        self.model = Model(inputs=[data_input, future_input], outputs=output)



