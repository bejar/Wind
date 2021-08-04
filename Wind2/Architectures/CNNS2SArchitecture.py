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

from Wind2.Architectures.NNS2SArchitecture import NNS2SArchitecture
from Wind2.Train.Activations import generate_activation
from Wind2.Train.Layers import squeeze_and_excitation

from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Conv1D, Flatten, Input, BatchNormalization, GlobalAveragePooling1D
from keras.regularizers import l1, l2

__author__ = 'bejar'


class CNNS2SArchitecture(NNS2SArchitecture):
    """
    Class for convolutional sequence to sequence architecture

    """
    modfile = None
    modname = 'CNNS2S'
    data_mode = ('3D', '2D') #'cnn'

    def generate_model(self):
        """
        Model for CNN  for S2S

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
            "squeeze":ratio,
            "padding":"causal/same/valid",
            "bias": true/false,
            "batchnorm":true/false,
            "activation_full": "linear",
            "full": [16,8],
            "fulldrop": 0,
            "fulltype": "mlp/conv"
            "mode":"CNN_s2s"
        }

        :return:
        """
        drop = self.config['arch']['drop']
        filters = self.config['arch']['filters']
        kernel_size = self.config['arch']['kernel_size']

        padding = 'causal' if 'padding' not in self.config['arch'] else  self.config['arch']['padding']
        # If there is a dilation field and it is true the strides field is the dilation rates
        # and the strides are all 1's
        if 'dilation' in self.config['arch'] and self.config['arch']['dilation']:
            dilation = self.config['arch']['strides']
            strides = [1] * len(dilation)
        else:
            strides = self.config['arch']['strides']
            dilation = [1] * len(strides)

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
        bias = True if 'bias' not in self.config['arch'] else self.config['arch']['bias']

        if 'squeeze' in self.config['arch']:
            squeeze = self.config['arch']['squeeze']
        else:
            squeeze = None

        if k_reg == 'l1':
            k_regularizer = l1(k_regw)
        elif k_reg == 'l2':
            k_regularizer = l2(k_regw)
        else:
            k_regularizer = None

        input = Input(shape=(idimensions))
        model = Conv1D(filters[0], input_shape=(idimensions), kernel_size=kernel_size[0], strides=strides[0],
                              padding=padding, dilation_rate=dilation[0], use_bias=bias,
                              kernel_regularizer=k_regularizer)(input)
        model = generate_activation(activation)(model)
        if bnorm:
            model = BatchNormalization()(model)

        if drop != 0:
            model = Dropout(rate=drop)(model)

        if squeeze is not None:
            model = squeeze_and_excitation(model, ratio=squeeze)

        for i in range(1, len(filters)):
            model = Conv1D(filters[i], kernel_size=kernel_size[i], strides=strides[i],
                              padding=padding, dilation_rate=dilation[i], use_bias=bias,
                              kernel_regularizer=k_regularizer)(model)
            model = generate_activation(activation)(model)
            if bnorm:
                model = BatchNormalization()(model)

            if drop != 0:
                model = Dropout(rate=drop)(model)

            if squeeze is not None:
                model = squeeze_and_excitation(model, ratio=squeeze)

        if 'fulltype' not in self.config['arch']:
            # MLP output, full/fulldrop/activation are used to build a MLP with a final layer that is linear with odimensions
            activationfl = self.config['arch']['activation_full']
            fulldrop = self.config['arch']['fulldrop']
            full_layers = self.config['arch']['full']
            model = Flatten()(model)

            for l in full_layers:
                model= Dense(l)(model)
                model = generate_activation(activationfl)(model)
                if bnorm:
                    model = BatchNormalization()(model)

                if fulldrop != 0:
                    model = Dropout(rate=fulldrop)(model)

            output = Dense(odimensions, activation='linear')(model)
        elif self.config['arch']['fulltype'] == 'mlp':
            # MLP output, full/fulldrop/activation are used to build a MLP with a final layer that is linear with odimensions
            activationfl = self.config['arch']['activation_full']
            fulldrop = self.config['arch']['fulldrop']
            full_layers = self.config['arch']['full']
            model = Flatten()(model)

            for l in full_layers:
                model= Dense(l)(model)
                model = generate_activation(activationfl)(model)
                if bnorm:
                    model = BatchNormalization()(model)

                if fulldrop != 0:
                    model = Dropout(rate=fulldrop)(model)

            output = Dense(odimensions, activation='linear')(model)
        elif self.config['arch']['fulltype'] == 'conv': # "conv"
            # Fully convolutional output, size 1 stride 1 convolution with odimensions filters
            activationfl = self.config['arch']['activation_full']
            fulldrop = self.config['arch']['fulldrop']
            model = Conv1D(odimensions, kernel_size=1, strides=1)(model)
            model = generate_activation(activationfl)(model)
            if fulldrop != 0:
                model = Dropout(rate=fulldrop)(model)
            model = Flatten()(model)
            # model = GlobalAveragePooling1D()(model)
            output = Dense(odimensions, activation='linear')(model)
        else:
            model = Flatten()(model)
            output = Dense(odimensions, activation='linear')(model)


        self.model = Model(inputs=input, outputs=output)


    def predict(self, val_x):
        """
        Returns the predictions of the model for some data

        :param val_x:
        :return:
        """
        batch_size = self.config['training']['batch']

        if self.runconfig.best:
            self.model = load_model(self.modfile)

        return self.model.predict(val_x, batch_size=batch_size, verbose=0)
