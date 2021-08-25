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
from tensorflow.keras.layers import Dense, Dropout, SeparableConv1D, Flatten, Input, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1, l2

from Wind.Architectures.NNS2SArchitecture import NNS2SArchitecture
from Wind.Train.Activations import generate_activation

__author__ = 'bejar'


class CNNSeparable2LS2SSeparateArchitecture(NNS2SArchitecture):
    """
    Class for 2 layers separable convolutional sequence to sequence architecture

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

        # Principal Branch
        # 1st Layer
        drop = self.config['arch']['drop']
        filters = self.config['arch']['filters']
        kernel_size = self.config['arch']['kernel_size']
        padding = self.config['arch']['padding']


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
        activation2 = self.config['arch']['activation2']


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

        input = Input(shape=(idimensions[0]))
        model = SeparableConv1D(filters[0], input_shape=(idimensions[0]), kernel_size=kernel_size[0], strides=strides[0],
                                padding=padding, dilation_rate=dilation[0], depth_multiplier=depth_multiplier,
                                kernel_regularizer=k_regularizer)(input)
        model = generate_activation(activation)(model)

        if drop != 0:
            model = Dropout(rate=drop)(model)

        model = SeparableConv1D(filters2[0], kernel_size=kernel_size2[0], strides=strides2[0],
                                padding=padding, dilation_rate=dilation2[0], depth_multiplier=depth_multiplier2,
                                kernel_regularizer=k_regularizer)(model)
        model = generate_activation(activation2)(model)
        if drop != 0:
            model = Dropout(rate=drop2)(model)

        # Additional Branch

        # First Layer
        dropa = self.config['arch']['dropa']
        filtersa = self.config['arch']['filtersa']
        kernel_sizea = self.config['arch']['kernel_sizea']

        if 'dilation' in self.config['arch'] and self.config['arch']['dilation']:
            dilationa = self.config['arch']['strides']
            stridesa = [1] * len(dilationa)
        else:
            stridesa = self.config['arch']['stridesa']
            dilationa = [1] * len(stridesa)

        if 'dilation' in self.config['arch'] and self.config['arch']['dilation']:
            dilationa2 = self.config['arch']['stridesa2']
            stridesa2 = [1] * len(dilation)
        else:
            stridesa2 = self.config['arch']['stridesa2']
            dilationa2 = [1] * len(stridesa2)

        depth_multipliera = self.config['arch']['depth_multipliera']
        activationa = self.config['arch']['activationa']
        depth_multipliera2 = self.config['arch']['depth_multipliera2']
        activationa2 = self.config['arch']['activationa2']


        # Second Layer
        dropa2 = self.config['arch']['dropa2']
        filtersa2 = self.config['arch']['filtersa2']
        kernel_sizea2 = self.config['arch']['kernel_sizea2']

        inputa = Input(shape=(idimensions[1]))
        modela = SeparableConv1D(filtersa[0], input_shape=(idimensions[1]), kernel_size=kernel_sizea[0], strides=stridesa[0],
                                padding=padding, dilation_rate=dilationa[0], depth_multiplier=depth_multipliera,
                                kernel_regularizer=k_regularizer)(inputa)
        modela = generate_activation(activationa)(modela)

        if drop != 0:
            modela = Dropout(rate=dropa)(modela)

        modela = SeparableConv1D(filtersa2[0], kernel_size=kernel_sizea2[0], strides=stridesa2[0],
                                padding=padding, dilation_rate=dilationa2[0], depth_multiplier=depth_multipliera2,
                                kernel_regularizer=k_regularizer)(modela)
        modela = generate_activation(activationa2)(modela)
        if drop != 0:
            modela = Dropout(rate=dropa2)(modela)


        # Merge

        activationfl = self.config['arch']['activation_full']
        fulldrop = self.config['arch']['fulldrop']
        full_layers = self.config['arch']['full']


        model = Flatten()(model)
        modela = Flatten()(modela)
        modelf = concatenate([model, modela])
        for l in full_layers:
            modelf = Dense(l)(modelf)
            modelf = generate_activation(activationfl)(modelf)
            if fulldrop != 0:
                modelf = Dropout(rate=fulldrop)(modelf)

        output = Dense(odimensions, activation='linear')(modelf)

        self.model = Model(inputs=[input,inputa], outputs=output)
