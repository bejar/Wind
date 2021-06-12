"""
.. module:: CNNS2SArchitecture

CNNS2SArchitecture
*************


:Description: CNNS2SArchitecture

    Class for convolutional sequence to sequence architecture


:Authors: bejar
    

:Version: 

:Created on: 24/10/2018 8:10 

"""

from tensorflow.keras.layers import Dense, Dropout, SeparableConv1D, Flatten, Input, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1, l2

from Wind.Architectures.NNS2SArchitecture import NNS2SArchitecture
from Wind.Train.Activations import generate_activation

__author__ = 'bejar'


class CNNSeparableS2SArchitecture(NNS2SArchitecture):
    """
    Class for separable convolutional sequence to sequence architecture

    """
    modfile = None
    modname = 'CNNS2S'
    data_mode = ('3D', '2D')  # 'cnn'

    def generate_model(self):
        """
        Model for separable CNN for S2S

        json config:

        "arch": {
            "filters": [32],
            "strides": [1],
            "dilation": false,
            "kernel_size": [3],
            "depth_multiplier": 1,
            "k_reg": "None",
            "k_regw": 0.1,
            "rec_reg": "None",
            "rec_regw": 0.1,
            "drop": 0,
            "activation": "relu",
            "activation_full": "linear",
            "full": [16,8],
            "fulldrop": 0,
            "mode":"CNN_sep_s2s"
        }

        :return:
        """
        drop = self.config['arch']['drop']
        filters = self.config['arch']['filters']
        padding = self.config['arch']['padding']
        kernel_size = self.config['arch']['kernel_size']
        # If there is a dilation field and it is true the strides field is the dilation rates
        # and the strides are all 1's
        if 'dilation' in self.config['arch'] and self.config['arch']['dilation']:
            dilation = self.config['arch']['strides']
            strides = [1] * len(dilation)
        else:
            strides = self.config['arch']['strides']
            dilation = [1] * len(strides)

        depth_multiplier = self.config['arch']['depth_multiplier']
        activationfl = self.config['arch']['activation_full']
        fulldrop = self.config['arch']['fulldrop']
        full_layers = self.config['arch']['full']

        activation = self.config['arch']['activation']

        k_reg = self.config['arch']['k_reg']
        k_regw = self.config['arch']['k_regw']

        # Extra added from training function
        idimensions = self.config['idimensions']
        odimensions = self.config['odimensions']

        if 'batchnorm' in self.config['arch']:
            bnorm = self.config['arch']['batchnorm']
        else:
            bnorm = False

        if k_reg == 'l1':
            k_regularizer = l1(k_regw)
        elif k_reg == 'l2':
            k_regularizer = l2(k_regw)
        else:
            k_regularizer = None

        input = Input(shape=(idimensions))
        model = SeparableConv1D(filters[0], input_shape=(idimensions), kernel_size=kernel_size[0], strides=strides[0],
                                padding=padding, dilation_rate=dilation[0], depth_multiplier=depth_multiplier,
                                kernel_regularizer=k_regularizer)(input)
        model = generate_activation(activation)(model)
        if bnorm:
            model = BatchNormalization()(model)

        if drop != 0:
            model = Dropout(rate=drop)(model)

        for i in range(1, len(filters)):
            model = SeparableConv1D(filters[i], kernel_size=kernel_size[i], strides=strides[i],
                                    padding=padding, dilation_rate=dilation[i], depth_multiplier=depth_multiplier,
                                    kernel_regularizer=k_regularizer)(model)
            model = generate_activation(activation)(model)
            if bnorm:
                model = BatchNormalization()(model)

            if drop != 0:
                model = Dropout(rate=drop)(model)

        model = Flatten()(model)
        for l in full_layers:
            model = Dense(l)(model)
            model = generate_activation(activationfl)(model)
            if bnorm:
                model = BatchNormalization()(model)
            if fulldrop != 0:
                model = Dropout(rate=fulldrop)(model)

        output = Dense(odimensions, activation='linear')(model)

        self.model = Model(inputs=input, outputs=output)
