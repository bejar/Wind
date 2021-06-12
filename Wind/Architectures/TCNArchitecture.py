"""
.. module:: TCNArchitecture

TCNArchitecture
******

:Description: TCNArchitecture

    Class for Temporal Convolutional Network with sequence to sequence architecture

:Authors:
    bejar

:Version: 

:Date:  23/03/2021
"""

__author__ = 'bejar'

from tensorflow.keras.layers import Dense, Dropout, Conv1D, Flatten, Input, BatchNormalization, Add
from tensorflow.keras.models import load_model, Model

from Wind.Architectures.NNS2SArchitecture import NNS2SArchitecture
from Wind.Train.Activations import generate_activation
from Wind.Train.Layers import squeeze_and_excitation

__author__ = 'bejar'


class TCNArchitecture(NNS2SArchitecture):
    """
    Class for Tempral Convolutional Network with sequence to sequence architecture

    """
    modfile = None
    modname = 'TCN'
    data_mode = ('3D', '2D')  # 'cnn'

    def residual_block(self, input, kernel_size, strides, dilation, filters, padding, activation,drop):

        ## Residual connection
        res =  Conv1D(filters, kernel_size=1, strides=strides,
                       padding=padding, dilation_rate=dilation)(input)

        ## Convolutions
        model = Conv1D(filters, kernel_size=kernel_size, strides=strides,
                       padding=padding, dilation_rate=dilation)(input)
        model = BatchNormalization()(model)
        model = generate_activation(activation)(model)

        model = Conv1D(filters, kernel_size=kernel_size, strides=strides,
                       padding=padding, dilation_rate=dilation)(model)

        model = Add()([model, res])
        model = BatchNormalization()(model)

        model = generate_activation(activation)(model)
        model = Dropout(rate=drop)(model)
        return model



    def generate_model(self):
        """
        Time Convolutional Network

        """
        drop = self.config['arch']['drop']
        filters = self.config['arch']['filters']
        kernel_size = self.config['arch']['kernel_size']

        padding = 'causal' if 'padding' not in self.config['arch'] else self.config['arch']['padding']
        # If there is a dilation field and it is true the strides field is the dilation rates
        # and the strides are all 1's
        if 'dilation' in self.config['arch'] and self.config['arch']['dilation']:
            dilation = self.config['arch']['strides']
            strides = [1] * len(dilation)
        else:
            strides = self.config['arch']['strides']
            dilation = [1] * len(strides)

        activation = self.config['arch']['activation']

        full_layers = self.config['arch']['full']
        activationfl = self.config['arch']['activation_full']
        fulldrop = self.config['arch']['fulldrop']

        # Extra added from training function
        idimensions = self.config['idimensions']
        odimensions = self.config['odimensions']

        input = Input(shape=(idimensions))
        model = Conv1D(filters[0], input_shape=(idimensions), kernel_size=kernel_size[0], strides=strides[0],
                       padding=padding, dilation_rate=dilation[0])(input)
        model = generate_activation(activation)(model)

        for i in range(1, len(filters)):
            model = self.residual_block(model, kernel_size[0], strides[0], dilation[0], filters[i], padding, activation,drop)

        model = Flatten()(model)

        model = Dense(full_layers[0])(model)
        model = generate_activation(activationfl)(model)
        model = Dropout(rate=fulldrop)(model)

        output = Dense(odimensions, activation='linear')(model)

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
