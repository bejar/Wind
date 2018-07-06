"""
.. module:: RNNDirRegressionArchitecture

RNNDirRegressionArchitecture
*************

:Description: RNNDirRegressionArchitecture

    

:Authors: bejar
    

:Version: 

:Created on: 06/07/2018 8:27 

"""

from Wind.Architectures.Architecture import Architecture
from keras.models import Sequential, load_model
from keras.layers import LSTM, GRU,  Bidirectional, Dense

try:
    from keras.layers import CuDNNGRU, CuDNNLSTM
except ImportError:
    _has_CuDNN = False
else:
    _has_CuDNN = True

from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.regularizers import l1, l2

try:
    from keras.utils import multi_gpu_model
except ImportError:
    _has_multigpu = False
else:
    _has_multigpu = True

from sklearn.metrics import r2_score
from time import time
import os

__author__ = 'bejar'


class RNNDirRegressionArchitecture(Architecture):

    modfile = None

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

        rnntype=self.config['arch']['rnn']
        CuDNN=self.config['arch']['CuDNN']
        full=self.config['arch']['full']

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
                                                recurrent_regularizer=rec_regularizer, kernel_regularizer=k_regularizer),
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
                                      activation=activation, recurrent_activation=activation_r, return_sequences=True,
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


    def train(self, train_x, train_y, val_x, val_y):
        batch_size = self.config['training']['batch']
        nepochs = self.config['training']['epochs']
        optimizer = self.config['training']['optimizer']

        cbacks = []
        if self.runconfig.tboard:
            tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
            cbacks.append(tensorboard)

        if self.runconfig.best:
            self.modfile = './model%d.h5' % int(time())
            mcheck = ModelCheckpoint(filepath=self.modfile, monitor='val_loss', verbose=0, save_best_only=True,
                                     save_weights_only=False, mode='auto', period=1)
            cbacks.append(mcheck)

        if self.runconfig.early:
            early = EarlyStopping(monitor='val_loss', patience=5, verbose=0)
            cbacks.append(early)


        if optimizer == 'rmsprop':
            if 'lrate' in self.config['training']:
                optimizer = RMSprop(lr=self.config['training']['lrate'])
            else:
                optimizer = RMSprop(lr=0.001)

        if self.runconfig.multi == 1:
            self.model.compile(loss='mean_squared_error', optimizer=optimizer)
        else:
            pmodel = multi_gpu_model(self.model, gpus=self.runconfig.multi)
            pmodel.compile(loss='mean_squared_error', optimizer=optimizer)

        if self.runconfig.multi == 1:
            self.model.fit(train_x, train_y, batch_size=batch_size, epochs=nepochs, validation_data=(val_x, val_y),
                  verbose=self.runconfig.verbose, callbacks=cbacks)
        else:
            pmodel.fit(train_x, train_y, batch_size=batch_size, epochs=nepochs, validation_data=(val_x, val_y),
                  verbose=self.runconfig.verbose, callbacks=cbacks)


    def evaluate(self, val_x, val_y, test_x, test_y):
        batch_size = self.config['training']['batch']

        if self.runconfig.best:
            self.model = load_model(self.modfile)

        val_yp = self.model.predict(val_x, batch_size=batch_size, verbose=0)
        r2val = r2_score(val_y, val_yp)

        test_yp = self.model.predict(test_x, batch_size=batch_size, verbose=0)
        r2test = r2_score(test_y, test_yp)

        return r2val, r2test


    def log_result(self, result):
        for r in result:
            print('%s | DNM= %s, DS= %d, V= %d, LG= %d, AH= %d, RNN= %s, Bi=%s, LY= %d, NN= %d, DR= %3.2f, AF= %s, RAF= %s, '
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

    def save(self, postfix):
        if not self.runconfig.save and self.runconfig.best:
            try:
                os.remove(self.modfile)
            except OSError:
                pass
        else:
            os.rename(self.modfile, 'modelRNNDir-S%s5s.h5'%(self.config['data']['datanames'][0], postfix))
