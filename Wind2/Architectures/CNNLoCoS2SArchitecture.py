"""
.. module:: CNNLoCoS2SArchitecture

CNNS2SArchitecture
*************


:Description: CNNS2SArchitecture

    Class for convolutional Locally Connected sequence to sequence architecture


:Authors: bejar
    

:Version: 

:Created on: 24/10/2018 8:10 

"""

from keras.layers import Dense, Dropout, Flatten, Input, LocallyConnected1D
from keras.models import load_model, Model
from keras.regularizers import l1, l2

from Wind.Architectures.NNS2SArchitecture import NNS2SArchitecture
from Wind.Train.Activations import generate_activation

__author__ = 'bejar'


class CNNLoCoS2SArchitecture(NNS2SArchitecture):
    """
    Class for convolutional Locally Connected sequence to sequence architecture

    """
    modfile = None
    modname = 'CNNLCS2S'
    data_mode = ('3D', '2D') #'cnn'

    def generate_model(self):
        """
        Model for CNN LoCo for S2S

        json config:

        "arch": {
            "filters": [32],
            "strides": [1],
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

        # padding = 'causal' if 'padding' not in self.config['arch'] else  self.config['arch']['padding']
        padding = 'valid' # LocallyConnected only supports valid padding

        strides = self.config['arch']['strides']

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
        model = LocallyConnected1D(filters[0], input_shape=(idimensions), kernel_size=kernel_size[0], strides=strides[0],
                              padding=padding, data_format='channels_last',
                              kernel_regularizer=k_regularizer)(input)
        model = generate_activation(activation)(model)

        if drop != 0:
            model = Dropout(rate=drop)(model)

        for i in range(1, len(filters)):
            model = LocallyConnected1D(filters[i], kernel_size=kernel_size[i], strides=strides[i],
                              padding=padding, data_format='channels_last',
                              kernel_regularizer=k_regularizer)(model)
            model = generate_activation(activation)(model)

            if drop != 0:
                model = Dropout(rate=drop)(model)

        model = Flatten()(model)
        for l in full_layers:
            model= Dense(l)(model)
            model = generate_activation(activationfl)(model)
            if fulldrop != 0:
                model = Dropout(rate=fulldrop)(model)

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
