"""
.. module:: RNNDirRegressionArchitecture

RNNDirRegressionArchitecture
*************

:Description: RNNDirRegressionArchitecture

    

:Authors: bejar
    

:Version: 

:Created on: 06/07/2018 8:27 

"""

from Wind.Architectures.NNArchitecture import NNArchitecture
from keras.models import Sequential
from keras.layers import LSTM, GRU, Bidirectional, Dense

try:
    from keras.layers import CuDNNGRU, CuDNNLSTM
except ImportError:
    _has_CuDNN = False
else:
    _has_CuDNN = True

from keras.regularizers import l1, l2

try:
    from keras.utils import multi_gpu_model
except ImportError:
    _has_multigpu = False
else:
    _has_multigpu = True

__author__ = 'bejar'


class RNNDirRegressionArchitecture(NNArchitecture):
    modfile = None
    modname = 'RNNDir'
    data_mode = False

    def generate_model(self):
        """
        Model for RNN with direct regression

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

        # return model

    def summary(self):
        self.model.summary()
        neurons = self.config['arch']['neurons']
        nlayers = self.config['arch']['nlayers']  # >= 1
        activation = self.config['arch']['activation']
        activation_r = self.config['arch']['activation_r']
        print('lag: ', self.config['data']['lag'], '/Neurons: ', neurons, '/Layers: ', nlayers, '/Activation:',
              activation, activation_r)

    def log_result(self, result):
        for r in result:
            print(
                        '%s | DNM= %s, DS= %d, V= %d, LG= %d, AH= %d, RNN= %s, Bi=%s, LY= %d, NN= %d, DR= %3.2f, AF= %s, RAF= %s, '
                        'OPT= %s, R2V = %3.5f, R2T = %3.5f' %
                        (self.config['arch']['mode'],
                         self.config['data']['datanames'][0],
                         self.config['data']['dataset'],
                         len(self.config['data']['vars']),
                         self.config['data']['lag'],
                         r[0],
                         self.config['arch']['rnn'],
                         self.config['arch']['bimerge'] if self.config['arch']['bidirectional'] else 'no',
                         self.config['arch']['nlayers'],
                         self.config['arch']['neurons'],
                         self.config['arch']['drop'],
                         self.config['arch']['activation'],
                         self.config['arch']['activation_r'],
                         self.config['training']['optimizer'],
                         r[1], r[2]
                         ))

    # def save(self, postfix):
    #     if not self.runconfig.save and self.runconfig.best:
    #         try:
    #             os.remove(self.modfile)
    #         except OSError:
    #             pass
    #     else:
    #         os.rename(self.modfile, 'modelRNNDir-S%s5s.h5'%(self.config['data']['datanames'][0], postfix))
