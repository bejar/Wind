"""
.. module:: MLPRegressionDir

MLPRegressionDir
*************

:Description: MLPRegressionDir

    Multilater perceptron with direct regresion horizon

:Authors: bejar
    

:Version: 

:Created on: 24/04/2018 12:17 

"""

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM, GRU, CuDNNGRU, CuDNNLSTM, Bidirectional, TimeDistributed, Flatten, RepeatVector
from keras.optimizers import RMSprop, SGD
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.regularizers import l1, l2
from keras.utils import multi_gpu_model
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score
from Wind.Data import generate_dataset
from Wind.Config import wind_data_path
from Wind.Training import updateprocess
from time import time, strftime
import os

__author__ = 'bejar'


def architectureMLP_dirreg(idimensions, activation='linear', rec_reg='l1', rec_regw=0.1, k_reg='l1', k_regw=0.1,
                       dropout=0.0, full_layers=[128]):
    """
    Arquitecture with direct regression using MLP

    :return:
    """
    model = Sequential()
    model.add(Dense(full_layers[0], input_shape=idimensions, activation=activation))
    model.add(Dropout(rate=dropout))
    for units in full_layers[1:]:
        model.add(Dense(units=units, activation=activation))
        model.add(Dropout(rate=dropout))
    model.add(Flatten())
    model.add(Dense(1,activation='linear'))

    return model

def train_MLP_dirreg_architecture(config, verbose, tboard, best, early, multi=1):
    """
    Training process for architecture with direct regression of ahead time steps

    :return:
    """

    if type(config['data']['ahead']) == list:
        iahead, sahead = config['data']['ahead']
    else:
        iahead, sahead = 1, config['data']['ahead']

    lresults = []
    for ahead in range(iahead, sahead + 1):

        if verbose:
            print('-----------------------------------------------------------------------------')
            print('Steps Ahead = %d ' % ahead)

        train_x, train_y, val_x, val_y, test_x, test_y = generate_dataset(config['data'], ahead=ahead, mode=False,
                                                                          data_path=wind_data_path)

        ############################################
        # Model

        drop = config['arch']['drop']
        nlayers = config['arch']['nlayers']  # >= 1

        activation = config['arch']['activation']
        rec_reg = config['arch']['rec_reg']
        rec_regw = config['arch']['rec_regw']
        k_reg = config['arch']['k_reg']
        k_regw = config['arch']['k_regw']


        if multi == 1:
            model = architectureMLP_dirreg(idimensions=train_x.shape[1:], activation=activation,
                                   rec_reg=rec_reg, rec_regw=rec_regw, k_reg=k_reg, k_regw=k_regw, dropout=drop,
                                   full_layers=config['arch']['full'])
        else:
            with tf.device('/cpu:0'):
                model = architectureMLP_dirreg(idimensions=train_x.shape[1:], activation=activation,
                                   rec_reg=rec_reg, rec_regw=rec_regw, k_reg=k_reg, k_regw=k_regw, dropout=drop,
                                   full_layers=config['arch']['full'])

        if verbose:
            model.summary()
            print('lag: ', config['data']['lag'], '/Layers: ', nlayers, '/Activation:', activation)
            print('Tr:', train_x.shape, train_y.shape, 'Val:', val_x.shape, val_y.shape, 'Ts:', test_x.shape, test_y.shape)
            print()

        ############################################
        # Training
        cbacks = []
        if tboard:
            tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
            cbacks.append(tensorboard)

        if best:
            modfile = './model%d.h5' % int(time())
            mcheck = ModelCheckpoint(filepath=modfile, monitor='val_loss', verbose=0, save_best_only=True,
                                     save_weights_only=False, mode='auto', period=1)
            cbacks.append(mcheck)

        if early:
            early = EarlyStopping(monitor='val_loss', patience=5, verbose=0)
            cbacks.append(early)

        optimizer = config['training']['optimizer']
        if optimizer == 'rmsprop':
            if 'lrate' in config['training']:
                optimizer = RMSprop(lr=config['training']['lrate'])
            else:
                optimizer = RMSprop(lr=0.001)

        if multi == 1:
            model.compile(loss='mean_squared_error', optimizer=optimizer)
        else:
            pmodel = multi_gpu_model(model, gpus=multi)
            pmodel.compile(loss='mean_squared_error', optimizer=optimizer)


        batch_size = config['training']['batch']
        nepochs = config['training']['epochs']

        if multi == 1:
            model.fit(train_x, train_y, batch_size=batch_size, epochs=nepochs, validation_data=(val_x, val_y),
                  verbose=verbose, callbacks=cbacks)
        else:
            pmodel.fit(train_x, train_y, batch_size=batch_size, epochs=nepochs, validation_data=(val_x, val_y),
                  verbose=verbose, callbacks=cbacks)


        ############################################
        # Results

        if best:
            model = load_model(modfile)

        # score = model.evaluate(val_x, val_y, batch_size=batch_size, verbose=0)
        # print('MSE val= ', score)
        # print('MSE val persistence =', mean_squared_error(val_y[ahead:], val_y[0:-ahead]))
        val_yp = model.predict(val_x, batch_size=batch_size, verbose=0)
        r2val = r2_score(val_y, val_yp)
        # r2persV = r2_score(val_y[ahead:], val_y[0:-ahead])
        # print('R2 val= ', r2val)
        # print('R2 val persistence =', r2persV)

        # score = model.evaluate(test_x, test_y, batch_size=batch_size, verbose=0)
        # print()
        # print('MSE test= ', score)
        # print('MSE test persistence =', mean_squared_error(test_y[ahead:], test_y[0:-ahead]))
        test_yp = model.predict(test_x, batch_size=batch_size, verbose=0)
        r2test = r2_score(test_y, test_yp)
        # r2persT = r2_score(test_y[ahead:, 0], test_y[0:-ahead, 0])
        # print('R2 test= ', r2test)
        # print('R2 test persistence =', r2persT)

        lresults.append((ahead, r2val,  r2test))

        print('%s | DNM= %s, DS= %d, V= %d, LG= %d, AH= %d, FL= %s, DR= %3.2f, AF= %s, '
              'OPT= %s, R2V = %3.5f, R2T = %3.5f' %
              (config['arch']['mode'],
               config['data']['datanames'][0],
               config['data']['dataset'],
               len(config['data']['vars']),
               config['data']['lag'],
               ahead, str(config['arch']['full']),
               config['arch']['drop'],
               config['arch']['activation'],
               config['training']['optimizer'],
               r2val, r2test
               ))

        print(strftime('%Y-%m-%d %H:%M:%S'))

        # Update result in db
        if config is not None:
            updateprocess(config, ahead)

        try:
            os.remove(modfile)
        except OSError:
            pass

        del train_x, train_y, test_x, test_y, val_x, val_y
        del model

    return lresults


if __name__ == '__main__':

    from Wind.Util import load_config_file
    config = load_config_file("configmlpdir.json")
    wind_data_path = '../../Data'
    lresults = train_MLP_dirreg_architecture(config, False, False, True, True)