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

from tensorflow.keras.models import load_model
import shutil

from Wind.Architectures.Architecture import Architecture

try:
   from keras.utils import plot_model
   import pydot
except ImportError:
   _has_pydot = False
else:
   _has_pydot = True

from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, ReduceLROnPlateau

try:
    from keras.utils import multi_gpu_model
except ImportError:
    _has_multigpu = False
else:
    _has_multigpu = True

from time import time
import os
import h5py

from Wind.Train.Losses import regression_losses
from Wind.ErrorMeasure import ErrorMeasure

__author__ = 'bejar'


class NNArchitecture(Architecture):
    """
    Class for all the Neural Network architectures

    """
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
            self.modfile = f"./model{int(time()*100)}-{self.config['data']['datanames'][0]}"
            mcheck = ModelCheckpoint(filepath=self.modfile, monitor='val_loss', verbose=0, save_best_only=True,
                                     save_weights_only=False, mode='auto', period=1)
            cbacks.append(mcheck)

        if 'RLROP' in self.config['training']:
            rlrop = ReduceLROnPlateau(monitor='val_loss', factor=self.config['training']['RLROP']['factor'],
                                             patience=self.config['training']['RLROP']['patience'])
            cbacks.append(rlrop)

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
            if self.config['training']['loss'] in regression_losses:
                loss = regression_losses[self.config['training']['loss']](self.config['odimensions'])
            else:
                loss = 'mean_squared_error'
        else:
            loss = 'mean_squared_error'

        if self.runconfig.multi == 1:
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

    def evaluate(self, val_x, val_y, test_x, test_y, scaler=None, save_errors=None):
        """
        Evaluates a trained model, loads the best if it is configured to do so
        Computes the R² for validation and test

        :param save_errors:
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
        test_yp = self.model.predict(test_x, batch_size=batch_size, verbose=0)

        if save_errors is not None:
            f = h5py.File(f'errors{self.modname}-S{self.config["data"]["datanames"][0]}{save_errors}.hdf5', 'w')
            dgroup = f.create_group('errors')
            dgroup.create_dataset('val_y', val_y.shape, dtype='f', data=val_y, compression='gzip')
            dgroup.create_dataset('val_yp', val_yp.shape, dtype='f', data=val_yp, compression='gzip')
            dgroup.create_dataset('test_y', test_y.shape, dtype='f', data=test_y, compression='gzip')
            dgroup.create_dataset('test_yp', test_yp.shape, dtype='f', data=test_y, compression='gzip')
            if scaler is not None:
                # Unidimensional vectors
                dgroup.create_dataset('val_yu', val_y.shape, dtype='f', data=scaler.inverse_transform(val_y.reshape(-1, 1)), compression='gzip')
                dgroup.create_dataset('val_ypu', val_yp.shape, dtype='f', data=scaler.inverse_transform(val_yp.reshape(-1, 1)), compression='gzip')
                dgroup.create_dataset('test_yu', test_y.shape, dtype='f', data=scaler.inverse_transform(test_y.reshape(-1, 1)), compression='gzip')
                dgroup.create_dataset('test_ypu', test_yp.shape, dtype='f', data=scaler.inverse_transform(test_yp.reshape(-1, 1)), compression='gzip')

        return ErrorMeasure().compute_errors(val_y, val_yp, test_y, test_yp, scaler=scaler)


    def save(self, postfix):
        """
        Saves and renames the last/best model if it is configured to do so, otherwise the file is deleted
        :param postfix:
        :return:
        """
        print('SAVING MODEL')
        if not self.runconfig.save:# or not self.runconfig.best):
            print("!!!!SAVING MODEL!!!!")
            try:
                shutil.rmtree(self.modfile)
                print(f'erasing {self.modfile} suceeded')
                # os.remove(self.modfile)
            except Exception:
                print(f'erasing {self.modfile} failed')
                #pass
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        else:
            print('WTF!!!!!!!!')
            os.rename(self.modfile, f'model{self.modname}-S{self.config["data"]["datanames"][0]}{postfix}.h5')
        print('SAVE ENDS!!!!!!!!!')

    def plot(self):
        """
        Plots the model as a png file
        :return:
        """
        # if _has_pydot:
        #    plot_model(self.model, show_shapes=True, to_file=f'{self.modname}.png')
        pass

