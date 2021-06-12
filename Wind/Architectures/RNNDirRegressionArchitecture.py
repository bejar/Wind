"""
.. module:: RNNDirRegressionArchitecture

RNNDirRegressionArchitecture
*************

:Description: RNNDirRegressionArchitecture

    Recurrent architecture for direct regression

:Authors: bejar
    

:Version: 

:Created on: 06/07/2018 8:27 

"""

from tensorflow.keras.layers import LSTM, GRU, Bidirectional, Dense
from tensorflow.keras.models import Sequential

from Wind.Architectures.NNArchitecture import NNArchitecture

try:
    from keras.layers import CuDNNGRU, CuDNNLSTM
except ImportError:
    _has_CuDNN = False
else:
    _has_CuDNN = True

from keras.regularizers import l1, l2

__author__ = 'bejar'


class RNNDirRegressionArchitecture(NNArchitecture):
    """
    Recurrent architecture for direct regression

    """
    modfile = None
    modname = 'RNNDir'
    data_mode = ('3D', '1D') # False

    def generate_model(self):
        """
        Model for RNN with direct regression

        -------------
        json config:


       "arch": {
            "neurons": 16,
            "k_reg": "None",
            "k_regw": 0.1,
            "rec_reg": "None",
            "rec_regw": 0.1,
            "drop": 0.3,
            "nlayers": 1,
            "activation": "relu",
            "activation_r": "hard_sigmoid",
            "CuDNN": false,
            "bidirectional": false,
            "bimerge": "ave",
            "rnn": "GRU",
            "full": [1],
            "mode": "RNN_dir_reg"
        }

        :return:
        """
        neurons = self.config['arch']['neurons']
        drop = self.config['arch']['drop']
        nlayers = self.config['arch']['nlayers']  # >= 1

        activation = self.config['arch']['activation']
        activation_r = self.config['arch']['activation_r']
        rec_reg = self.config['arch']['rec_reg']
        rec_regw = self.config['arch']['rec_regw']
        k_reg = self.config['arch']['k_reg']
        k_regw = self.config['arch']['k_regw']
        bidirectional = self.config['arch']['bidirectional']
        bimerge = self.config['arch']['bimerge']

        rnntype = self.config['arch']['rnn']
        CuDNN = self.config['arch']['CuDNN']
        full = self.config['arch']['full']

        # Extra added from training function
        idimensions = self.config['idimensions']
        impl = self.runconfig.impl

        if rec_reg == 'l1':
            rec_regularizer = l1(rec_regw)
        elif rec_reg == 'l2':
            rec_regularizer = l2(rec_regw)
        else:
            rec_regularizer = None

        if k_reg == 'l1':
            k_regularizer = l1(k_regw)
        elif rec_reg == 'l2':
            k_regularizer = l2(k_regw)
        else:
            k_regularizer = None

        if CuDNN:
            RNN = CuDNNLSTM if rnntype == 'LSTM' else CuDNNGRU
            self.model = Sequential()
            if nlayers == 1:
                self.model.add(
                    RNN(neurons, input_shape=(idimensions), recurrent_regularizer=rec_regularizer,
                        kernel_regularizer=k_regularizer))
            else:
                self.model.add(RNN(neurons, input_shape=(idimensions), return_sequences=True,
                                   recurrent_regularizer=rec_regularizer, kernel_regularizer=k_regularizer))
                for i in range(1, nlayers - 1):
                    self.model.add(RNN(neurons, return_sequences=True, recurrent_regularizer=rec_regularizer,
                                       kernel_regularizer=k_regularizer))
                self.model.add(RNN(neurons, recurrent_regularizer=rec_regularizer, kernel_regularizer=k_regularizer))
            for l in full:
                self.model.add(Dense(l))
        else:
            RNN = LSTM if rnntype == 'LSTM' else GRU
            self.model = Sequential()
            if bidirectional:
                if nlayers == 1:
                    self.model.add(
                        Bidirectional(RNN(neurons, implementation=impl,
                                          recurrent_dropout=drop,
                                          activation=activation,
                                          recurrent_activation=activation_r,
                                          recurrent_regularizer=rec_regularizer,
                                          kernel_regularizer=k_regularizer),
                                      input_shape=(idimensions), merge_mode=bimerge))
                else:
                    self.model.add(
                        Bidirectional(RNN(neurons, implementation=impl,
                                          recurrent_dropout=drop,
                                          activation=activation,
                                          recurrent_activation=activation_r,
                                          return_sequences=True,
                                          recurrent_regularizer=rec_regularizer,
                                          kernel_regularizer=k_regularizer),
                                      input_shape=(idimensions), merge_mode=bimerge))
                    for i in range(1, nlayers - 1):
                        self.model.add(Bidirectional(RNN(neurons, recurrent_dropout=drop, implementation=impl,
                                                         activation=activation, recurrent_activation=activation_r,
                                                         return_sequences=True,
                                                         recurrent_regularizer=rec_regularizer,
                                                         kernel_regularizer=k_regularizer), merge_mode=bimerge))
                    self.model.add(Bidirectional(RNN(neurons, recurrent_dropout=drop, activation=activation,
                                                     recurrent_activation=activation_r, implementation=impl,
                                                     recurrent_regularizer=rec_regularizer,
                                                     kernel_regularizer=k_regularizer),
                                                 merge_mode=bimerge))
                self.model.add(Dense(1))
            else:
                if nlayers == 1:
                    self.model.add(RNN(neurons, input_shape=(idimensions), implementation=impl,
                                       recurrent_dropout=drop, activation=activation, recurrent_activation=activation_r,
                                       recurrent_regularizer=rec_regularizer, kernel_regularizer=k_regularizer))
                else:
                    self.model.add(RNN(neurons, input_shape=(idimensions), implementation=impl,
                                       recurrent_dropout=drop, activation=activation, recurrent_activation=activation_r,
                                       return_sequences=True, recurrent_regularizer=rec_regularizer,
                                       kernel_regularizer=k_regularizer))
                    for i in range(1, nlayers - 1):
                        self.model.add(RNN(neurons, recurrent_dropout=drop, implementation=impl,
                                           activation=activation, recurrent_activation=activation_r,
                                           return_sequences=True,
                                           recurrent_regularizer=rec_regularizer, kernel_regularizer=k_regularizer))
                    self.model.add(RNN(neurons, recurrent_dropout=drop, activation=activation,
                                       recurrent_activation=activation_r, implementation=impl,
                                       recurrent_regularizer=rec_regularizer, kernel_regularizer=k_regularizer))
                for l in full:
                    self.model.add(Dense(l))


