"""
.. module:: TimeInceptionArchitecture

CNNS2SArchitecture
*************


:Description: TimeInceptionArchitecture

    Class for Time Inception architecture


:Authors:
    
Borrowed from "InceptionTime: Finding AlexNet for Time Series Classification" https://arxiv.org/pdf/1909.04939.pdf
https://github.com/hfawaz/InceptionTime

:Version: 

:Created on: 24/10/2018 8:10 

"""

from Wind.Architectures.NNS2SArchitecture import NNS2SArchitecture
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Conv1D, Flatten, Input, BatchNormalization, \
    GlobalAveragePooling1D, Concatenate, MaxPool1D, Add
from Wind.Train.Activations import generate_activation

__author__ = 'bejar'


class TimeInceptionArchitecture(NNS2SArchitecture):
    """
    Class for convolutional sequence to sequence architecture

    """
    modfile = None
    modname = 'TimeIncep'
    data_mode = ('3D', '2D')  # 'cnn'

    def generate_model(self):
        """
        Model for TimeInception

        json config:

        "arch": {
            "filters": [32],
            "residual": true/false,
            "bottleneck": true/false,
            "kernel_size": [3],
            "drop": 0,
            "activation": "relu",
            "padding":"causal/same/valid",
            "bias": true/false,
            "batchnorm":true/false,
            "depth": n
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

        residual = self.config['arch']['residual']
        bottle = self.config['arch']['bottleneck']
        depth = self.config['arch']['depth']

        # Extra added from training function
        idimensions = self.config['idimensions']
        odimensions = self.config['odimensions']

        if 'batchnorm' in self.config['arch']:
            bnorm = self.config['arch']['batchnorm']
        else:
            bnorm = False
        bias = True if 'bias' not in self.config['arch'] else self.config['arch']['bias']

        input = Input(shape=(idimensions))

        x = input
        input_res = input

        for d in range(depth):

            x = self._inception_module(x, filters, kernel_size, activation, bottle)

            if residual and d % 3 == 2:
                x = self.shortcut_layer(input_res, x)
                input_res = x

        gap_layer = GlobalAveragePooling1D()(x)

        activationfl = self.config['arch']['activation_full']
        fulldrop = self.config['arch']['fulldrop']
        full_layers = self.config['arch']['full']

        model = gap_layer

        for l in full_layers:
            model = Dense(l)(model)
            model = generate_activation(activationfl)(model)
            if bnorm:
                model = BatchNormalization()(model)

            if fulldrop != 0:
                model = Dropout(rate=fulldrop)(model)

        output = Dense(odimensions, activation='linear')(model)

        self.model = Model(inputs=input, outputs=output)

    def _inception_module(self, input_tensor, filters, kernel_size, activation, bottleneck, stride=1):

        if bottleneck and int(input_tensor.shape[-1]) > 1:
            input_inception = Conv1D(filters=self.bottleneck_size, kernel_size=1,
                                     padding='same', activation=activation, use_bias=False)(input_tensor)
            input_inception = generate_activation(activation)(input_inception)
        else:
            input_inception = input_tensor

        conv_list = []

        for i in range(len(kernel_size)):
            layer = Conv1D(filters=filters, kernel_size=kernel_size[i],
                                    strides=stride, padding='same',
                                    use_bias=False)(input_inception)
            conv_list.append(generate_activation(activation)(layer))

        max_pool_1 = MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

        conv_6 = Conv1D(filters=filters, kernel_size=1,
                        padding='same', activation=activation, use_bias=False)(max_pool_1)
        conv6 = generate_activation(activation)(conv_6)

        conv_list.append(conv_6)

        x = Concatenate(axis=2)(conv_list)
        x = BatchNormalization()(x)
        x = generate_activation(activation)(x)
        return x

    def _shortcut_layer(self, input_tensor, out_tensor):
        shortcut_y = Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                            padding='same', use_bias=False)(input_tensor)
        shortcut_y = BatchNormalization()(shortcut_y)

        x = Add()([shortcut_y, out_tensor])
        x = generate_activation(activation)(x)
        return x

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
