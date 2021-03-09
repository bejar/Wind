"""
.. module:: CNNMIMOOSkipArchitecture

CNNS2SArchitecture
*************


:Description: CNNMIMOSkipArchitecture

    Class for convolutional sequence to sequence architecture with residual connections


:Authors: bejar
    

:Version: 

:Created on: 24/10/2018 8:10 

"""

from tensorflow.keras.layers import Dense, Dropout, Conv1D, Flatten, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1, l2

from Wind.Architectures.NNS2SArchitecture import NNS2SArchitecture
from Wind.Train.Activations import generate_activation

__author__ = 'bejar'


class CNNMIMOSkipArchitecture(NNS2SArchitecture):
    """
    Class for convolutional sequence to sequence architecture with skip/residual
    """
    modfile = None
    modname = 'CNNMIMOSKIP'
    data_mode = ('3D', '2D')  # 'cnn'

    def generate_model(self):
        """
        Model for CNN with Encoder Decoder for S2S

        json config:

        "arch": {
            "filters": [32],
            "strides": [1],
            "dilation": false,
            "kernel_size": [3],
            "k_reg": "None",
            "k_regw": 0.1,
            "rec_reg": "None",
            "rec_regw": 0.1,
            "drop": 0,
            "activation": "relu",
            "activation_full": "linear",
            "full": [16,8],
            "fulldrop": 0,
            "mode":"CNN_s2s"
        }

        :return:
        """
        drop = self.config['arch']['drop']
        filters = self.config['arch']['filters']
        kernel_size = self.config['arch']['kernel_size']
        # If there is a dilation field and it is true the strides field is the dilation rates
        # and the strides are all 1's
        if 'dilation' in self.config['arch'] and self.config['arch']['dilation']:
            dilation = self.config['arch']['strides']
            strides = [1] * len(dilation)
        else:
            strides = self.config['arch']['strides']
            dilation = [1] * len(strides)
        activationfl = self.config['arch']['activation_full']
        fulldrop = self.config['arch']['fulldrop']
        full_layers = self.config['arch']['full']

        activation = self.config['arch']['activation']

        k_reg = self.config['arch']['k_reg']
        k_regw = self.config['arch']['k_regw']

        # Extra added from training function
        idimensions = self.config['idimensions']
        odimensions = self.config['odimensions']

        if k_reg == 'l1':
            k_regularizer = l1(k_regw)
        elif k_reg == 'l2':
            k_regularizer = l2(k_regw)
        else:
            k_regularizer = None

        input = Input(shape=(idimensions))
        model = Conv1D(filters[0], input_shape=(idimensions), kernel_size=kernel_size[0], strides=strides[0],
                       padding='causal', dilation_rate=dilation[0],
                       kernel_regularizer=k_regularizer)(input)
        model = generate_activation(activation)(model)

        if drop != 0:
            model = Dropout(rate=drop)(model)

        last2 = model  # keep for generating the skip connections
        last1 = input
        for i in range(1, len(filters)):
            model = Concatenate()([model, last1])
            model = Conv1D(filters[i], kernel_size=kernel_size[i], strides=strides[i],
                           padding='causal', dilation_rate=dilation[i],
                           kernel_regularizer=k_regularizer)(model)
            model = generate_activation(activation)(model)

            if drop != 0:
                model = Dropout(rate=drop)(model)
            last1 = last2
            last2 = model

        model = Concatenate()([model, last1])
        # model = Concatenate()([Flatten()(input), Flatten()(model)])
        model = Flatten()(model)
        for l in full_layers:
            model = Dense(l)(model)
            model = generate_activation(activationfl)(model)
            if fulldrop != 0:
                model = Dropout(rate=fulldrop)(model)

        output = Dense(odimensions, activation='linear')(model)

        self.model = Model(inputs=input, outputs=output)
