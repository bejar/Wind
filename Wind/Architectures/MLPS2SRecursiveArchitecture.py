"""
.. module:: MLPS2SRecursiveArchirecture

MLPS2SRecursiveArchirecture
*************

:Description: MLPS2SRecursiveArchirecture

    

:Authors: bejar
    

:Version: 

:Created on: 30/11/2018 13:20 

"""

from tensorflow.keras.layers import Dense, Dropout, Input, concatenate
from tensorflow.keras.models import load_model, Model

from Wind.Architectures.NNS2SArchitecture import NNS2SArchitecture
from Wind.Train.Activations import generate_activation

__author__ = 'bejar'

class MLPS2SRecursiveArchitecture(NNS2SArchitecture):
    """
    Mutitlayer perceptron with sequence to sequence architecture for recursive training

    """
    modfile = None
    modname = 'MLPS2SREC'
    data_mode = ('2D', '2D')  #'mlp'

    def generate_model(self):
        """
        Model for MLP recursive multiple regression (s2s)

        It takes as inputs the data and the predictions of the previous step

        :return:
        """

        activation = self.config['arch']['activation']
        dropout = self.config['arch']['drop']
        full_layers = self.config['arch']['full']

        # Extra added from training function
        idimensions = self.config['idimensions']
        odimensions = self.config['odimensions']
        rdimensions = self.config['rdimensions']

        input = Input(shape=(idimensions))
        # If there are predictions from the previous step the NN has to heads, one for the data, other for
        # the predictions
        if rdimensions > 0:
            # The dimensions of recursive input depend on the recursive steps but it is a matrix (batch, recpred)
            rinput = Input(shape=(rdimensions,))
            recinput = concatenate([input, rinput])
        else:
            recinput = input

        # Dense layers to process the input
        model = Dense(full_layers[0])(recinput)
        model = generate_activation(activation)(model)
        model = Dropout(rate=dropout)(model)

        for units in full_layers[1:]:
            model = Dense(units=units)(model)
            model = generate_activation(activation)(model)
            model = Dropout(rate=dropout)(model)

        output = Dense(odimensions, activation='linear')(model)

        if rdimensions > 0:
            self.model = Model(inputs=[input, rinput], outputs=output)
        else:
            self.model = Model(inputs=input, outputs=output)

    def predict(self, val_x):
        """
        Returns the predictions of the model for some data

        :param val_x:
        :param val_y:
        :return:
        """
        batch_size = self.config['training']['batch']

        if self.runconfig.best:
            self.model = load_model(self.modfile)

        return self.model.predict(val_x, batch_size=batch_size, verbose=0)
