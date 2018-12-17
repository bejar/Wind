"""
.. module:: NNArchitecture

NNArchitecture
******

:Description: NNArchitecture

    Different Auxiliary functions used for different purposes

:Authors:
    bejar

:Version: 

:Date:  13/07/2018
"""

from Wind.Architectures.Architecture import Architecture
from keras.models import load_model
from keras import backend as K

#try:
#    from keras.utils import plot_model
#except ImportError:
#    _has_pydot = False
#else:
#    _has_pydot = True

try:
    from keras.layers import CuDNNGRU, CuDNNLSTM
except ImportError:
    _has_CuDNN = False
else:
    _has_CuDNN = True

from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

try:
    from keras.utils import multi_gpu_model
except ImportError:
    _has_multigpu = False
else:
    _has_multigpu = True

from sklearn.metrics import r2_score
from time import time
import os

from Wind.Train.Losses import linear_weighted_mse


__author__ = 'bejar'


class NNArchitecture(Architecture):
    modfile = None
    modname = ''

    def train(self, train_x, train_y, val_x, val_y):
        """
        Trainin process for a NN

        :param train_x:
        :param train_y:
        :param val_x:
        :param val_y:
        :return:
        """
        batch_size = self.config['training']['batch']
        nepochs = self.config['training']['epochs']
        optimizer = self.config['training']['optimizer']

        cbacks = []
        if self.runconfig.tboard:
            tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
            cbacks.append(tensorboard)

        if self.runconfig.best:
            self.modfile = f"./model{int(time())}- {self.config['data']['datanames'][0]}.h5"
            mcheck = ModelCheckpoint(filepath=self.modfile, monitor='val_loss', verbose=0, save_best_only=True,
                                     save_weights_only=False, mode='auto', period=1)
            cbacks.append(mcheck)

        if self.runconfig.early:
            patience = self.config['training']['patience'] if 'patience' in self.config['training'] else 5

            early = EarlyStopping(monitor='val_loss', patience=patience, verbose=0)
            cbacks.append(early)

        if optimizer == 'rmsprop':
            if 'lrate' in self.config['training']:
                optimizer = RMSprop(lr=self.config['training']['lrate'])
            else:
                optimizer = RMSprop(lr=0.001)

        if 'loss' in self.config['training']:
            if self.config['training']['loss'] == 'wmse':
                loss = linear_weighted_mse(self.config['odimensions'])
            else:
                loss = 'mean_squared_error'
        else:
            loss = 'mean_squared_error'

        if self.runconfig.multi == 1:
            # self.model.compile(loss='mean_squared_error', optimizer=optimizer)
            self.model.compile(loss=loss, optimizer=optimizer)

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
        """
        Evaluates a trained model, loads the best if it is configured to do so
        Computes the R² for validation and test

        :param val_x:
        :param val_y:
        :param test_x:
        :param test_y:
        :return:
        """
        batch_size = self.config['training']['batch']

        if self.runconfig.best:
            self.model = load_model(self.modfile)

        val_yp = self.model.predict(val_x, batch_size=batch_size, verbose=0)
        r2val = r2_score(val_y, val_yp)

        test_yp = self.model.predict(test_x, batch_size=batch_size, verbose=0)
        r2test = r2_score(test_y, test_yp)

        return r2val, r2test

    def save(self, postfix):
        """
        Saves and renames the last/best model if it is configured to do so, otherwise the file is deleted
        :param postfix:
        :return:
        """
        if not self.runconfig.save:# or not self.runconfig.best):
            try:
                os.remove(self.modfile)
            except OSError:
                pass
        else:
            os.rename(self.modfile, 'model%s-S%s%s.h5' % (self.modname, self.config['data']['datanames'][0], postfix))

    def plot(self):
        """
        Plots the model as a png file
        :return:
        """
        pass
        #if _has_pydot:
        #    plot_model(self.model, show_shapes=True, to_file=f'{self.modname}.png')


