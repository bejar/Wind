"""
.. module:: CNNSeparable2LS2SArchitecture

CNNS2SArchitecture
*************


:Description: CNNSeparable2LS2SArchitecture

    Class for separable convolutional sequence to sequence architecture

     Hack to experiment with a 2 layers architecture


:Authors: bejar
    

:Version: 

:Created on: 24/10/2018 8:10 

"""

from Wind.Architectures.NNS2SArchitecture import NNS2SArchitecture
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, SeparableConv1D, Flatten, Input
from sklearn.metrics import r2_score
from Wind.Train.Activations import generate_activation

from keras.regularizers import l1, l2

__author__ = 'bejar'


class CNNSeparable3LS2SArchitecture(NNS2SArchitecture):
    """
    Class for 2 layers separable convolutional sequence to sequence architecture

    """
    modfile = None
    modname = 'CNNS2S'
    data_mode = ('3D', '2D') #'cnn'

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
            "activation": "relu",
            "drop": 0,
            "k_reg": "None",
            "k_regw": 0.1,
            "rec_reg": "None",
            "rec_regw": 0.1,
            "activation_full": "linear",
            "full": [16,8],
            "fulldrop": 0,
            "mode":"CNN_sep_2l_s2s"
        }

        :return:
        """

        # 1st Layer
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

        depth_multiplier = self.config['arch']['depth_multiplier']
        activation = self.config['arch']['activation']


        # 2nd Layer
        drop2 = self.config['arch']['drop2']
        filters2 = self.config['arch']['filters2']
        kernel_size2 = self.config['arch']['kernel_size2']
        # If there is a dilation field and it is true the strides field is the dilation rates
        # and the strides are all 1's
        if 'dilation' in self.config['arch'] and self.config['arch']['dilation']:
            dilation2 = self.config['arch']['strides2']
            strides2 = [1] * len(dilation)
        else:
            strides2 = self.config['arch']['strides2']
            dilation2 = [1] * len(strides2)

        depth_multiplier2 = self.config['arch']['depth_multiplier2']
        activation2 = self.config['arch']['activation']

        # 3nd Layer
        drop3 = self.config['arch']['drop3']
        filters3 = self.config['arch']['filters3']
        kernel_size3 = self.config['arch']['kernel_size3']
        # If there is a dilation field and it is true the strides field is the dilation rates
        # and the strides are all 1's
        if 'dilation' in self.config['arch'] and self.config['arch']['dilation']:
            dilation3 = self.config['arch']['strides3']
            strides3 = [1] * len(dilation)
        else:
            strides3 = self.config['arch']['strides3']
            dilation3 = [1] * len(strides3)

        depth_multiplier3 = self.config['arch']['depth_multiplier3']
        activation2 = self.config['arch']['activation']


        activationfl = self.config['arch']['activation_full']
        fulldrop = self.config['arch']['fulldrop']
        full_layers = self.config['arch']['full']

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
        model = SeparableConv1D(filters[0], input_shape=(idimensions), kernel_size=kernel_size[0], strides=strides[0],
                              padding='same', dilation_rate=dilation[0],depth_multiplier=depth_multiplier,
                              kernel_regularizer=k_regularizer)(input)
        model = generate_activation(activation)(model)

        if drop != 0:
            model = Dropout(rate=drop)(model)


        model = SeparableConv1D(filters2[0], kernel_size=kernel_size2[0], strides=strides2[0],
                          padding='same', dilation_rate=dilation2[0],depth_multiplier=depth_multiplier2,
                          kernel_regularizer=k_regularizer)(model)
        model = generate_activation(activation)(model)

        if drop != 0:
            model = Dropout(rate=drop2)(model)

        model = SeparableConv1D(filters3[0], kernel_size=kernel_size3[0], strides=strides3[0],
                          padding='same', dilation_rate=dilation3[0],depth_multiplier=depth_multiplier3,
                          kernel_regularizer=k_regularizer)(model)
        model = generate_activation(activation)(model)

        if drop != 0:
            model = Dropout(rate=drop3)(model)


        model = Flatten()(model)
        for l in full_layers:
            model= Dense(l)(model)
            model = generate_activation(activationfl)(model)
            if fulldrop != 0:
                model = Dropout(rate=fulldrop)(model)

        output = Dense(odimensions, activation='linear')(model)

        self.model = Model(inputs=input, outputs=output)


