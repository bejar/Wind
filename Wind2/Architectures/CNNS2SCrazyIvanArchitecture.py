"""
.. module:: CNNS2SCrazyIvanArchitecture

CNNS2SCrazyIvanArchitecture
*************

:Description: CNNS2SCrazyIvanArchitecture

    Imaginative versions of CNN

      - First try "a la inception" with multiple heads using different kernel sizes

:Authors: bejar
    

:Version: 

:Created on: 25/03/2019 16:31 

"""

__author__ = 'bejar'


from Wind.Architectures.NNS2SArchitecture import NNS2SArchitecture
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Conv1D, Flatten, Concatenate, Input
from sklearn.metrics import r2_score
from Wind.Train.Activations import generate_activation

from keras.regularizers import l1, l2

__author__ = 'bejar'


class CNNS2SCrazyIvanArchitecture(NNS2SArchitecture):
    """
    Class for Multiple head convolutional sequence to sequence architecture

    """
    modfile = None
    modname = 'CNNCIS2S'
    data_mode = ('3D', '2D') #'cnn'

    def generate_model(self):
        """
        Model for CNN with Encoder Decoder for S2S

        json config:

        "arch": {
            "filters": 32,
            "strides": 1,
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
        if type(kernel_size) != list:
            raise NameError('kernel size must be a list')
        elif len(kernel_size) < 1:
            raise NameError('kernel size list must have more than one element')

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

        lconv = []

        # Assumes several kernel sizes but only one layer for head
        for k in kernel_size:
            convomodel = Conv1D(filters[0], input_shape=(idimensions), kernel_size=k, strides=strides[0],
                              padding='causal', kernel_regularizer=k_regularizer)(input)
            convomodel = generate_activation(activation)(convomodel)

            if drop != 0:
                convomodel = Dropout(rate=drop)(convomodel)
            lconv.append(convomodel)

        convoout = Concatenate()(lconv)
        fullout = Dense(full_layers[0])(convoout)
        fullout = generate_activation(activationfl)(fullout)
        fullout = Dropout(rate=fulldrop)(fullout)

        for l in full_layers[1:]:
            fullout = Dense(l)(fullout)
            fullout = generate_activation(activationfl)(fullout)
            fullout = Dropout(rate=fulldrop)(fullout)

        fullout = Flatten()(fullout)

        output = Dense(odimensions, activation='linear')(fullout)

        self.model = Model(inputs=input, outputs=output)
