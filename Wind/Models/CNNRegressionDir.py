"""
.. module:: ConvoRegression

ConvoRegression
*************

:Description: ConvoRegression

    Direct Regression with CNN

:Authors: bejar
    

:Version: 

:Created on: 23/04/2018 13:31 

"""

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout, Conv1D
from keras.layers import Flatten
from keras.optimizers import RMSprop, SGD
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.regularizers import l1, l2
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score
from Wind.Data import generate_dataset
from Wind.Config import wind_data_path
from Wind.Training import updateprocess
from time import time, strftime
import os


try:
    from keras.utils import multi_gpu_model
except ImportError:
    _has_multigpu = False
else:
    _has_multigpu = True

__author__ = 'bejar'


def architectureConvDirRegression(idimensions, filters, kernel_size, strides, drop, activation, activationfl,
                                  act_reg='l1', act_regw=0.1, k_reg='l1', k_regw=0.1, full=[1]):
    """
    Regression RNN architecture

    :return:
    """
    if act_reg == 'l1':
        act_regularizer = l1(act_regw)
    elif act_reg == 'l2':
        act_regularizer = l2(act_regw)
    else:
        act_regularizer = None

    if k_reg == 'l1':
        k_regularizer = l1(k_regw)
    elif act_reg == 'l2':
        k_regularizer = l2(k_regw)
    else:
        k_regularizer = None

    model = Sequential()


    model.add(Conv1D(filters[0], input_shape=(idimensions), kernel_size=kernel_size[0], strides=strides[0],
                     activation=activation, padding='causal',
                     activity_regularizer=act_regularizer,
                     kernel_regularizer=k_regularizer))

    if drop != 0:
        model.add(Dropout(rate=drop))
    for i in range(1, len(filters)):
        model.add(Conv1D(filters[i], kernel_size=kernel_size[i], strides=strides[i],
                         activation=activation, padding='causal',
                         kernel_regularizer=k_regularizer))


        if drop != 0:
            model.add(Dropout(rate=drop))

    model.add(Flatten())

    for l in full:
        model.add(Dense(l, activation=activationfl))

    model.add(Dense(1, activation='linear'))

    return model


def train_convdirregression_architecture(config, verbose, tboard, best, early, multi=1, remote=False):
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
                                                                          data_path=wind_data_path, remote=remote)

        ############################################
        # Model

        filters = config['arch']['filters']
        strides = config['arch']['strides']
        kernel_size = config['arch']['kernel_size']

        drop = config['arch']['drop']

        activation = config['arch']['activation']
        activationfl = config['arch']['activationfl']
        rec_reg = config['arch']['rec_reg']
        rec_regw = config['arch']['rec_regw']
        k_reg = config['arch']['k_reg']
        k_regw = config['arch']['k_regw']

        if multi == 1:
            model = architectureConvDirRegression(idimensions=train_x.shape[1:], filters=filters, drop=drop,
                                                  activation=activation, activationfl=activationfl,
                                                   strides=strides,
                                                  kernel_size=kernel_size,
                                                  act_reg=rec_reg, act_regw=rec_regw, k_reg=k_reg, k_regw=k_regw,
                                                  full=config['arch']['full'])
        else:
            with tf.device('/cpu:0'):
                model = architectureConvDirRegression(idimensions=train_x.shape[1:], filters=filters, drop=drop,
                                                      activation=activation, activationfl=activationfl,
                                                      strides=strides,
                                                      kernel_size=kernel_size,
                                                      act_reg=rec_reg, act_regw=rec_regw, k_reg=k_reg, k_regw=k_regw,
                                                      full=config['arch']['full'])

        if verbose:
            model.summary()

            print('lag: ', config['data']['lag'], '/Filters: ', filters, '/Activation:',
                  activation, 'K Size', kernel_size, '/Strides', strides)
            print('Tr:', train_x.shape, train_y.shape, 'Val:', val_x.shape, val_y.shape, 'Ts:', test_x.shape,
                  test_y.shape)
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
        r2persV = r2_score(val_y[ahead:], val_y[0:-ahead])
        # print('R2 val= ', r2val)
        # print('R2 val persistence =', r2persV)

        # score = model.evaluate(test_x, test_y, batch_size=batch_size, verbose=0)
        # print()
        # print('MSE test= ', score)
        # print('MSE test persistence =', mean_squared_error(test_y[ahead:], test_y[0:-ahead]))
        test_yp = model.predict(test_x, batch_size=batch_size, verbose=0)
        r2test = r2_score(test_y, test_yp)
        r2persT = r2_score(test_y[ahead:, 0], test_y[0:-ahead, 0])
        # print('R2 test= ', r2test)
        # print('R2 test persistence =', r2persT)

        lresults.append((ahead, r2val, r2persV, r2test, r2persT))
        print('%s | DNM= %s, DS= %d, V= %d, LG= %d, AH= %d, ST= %s, KS= %s, LY= %d, FLT= %s, DR= %3.2f, AF= %s, '
              'OPT= %s, R2V = %3.5f, R2PV = %3.5f, R2T = %3.5f, R2PT = %3.5f' %
              (config['arch']['mode'],
               config['data']['datanames'][0],
               config['data']['dataset'],
               len(config['data']['vars']),
               config['data']['lag'],
               ahead,
               str(config['arch']['strides']),
               str(config['arch']['kernel_size']),
               len(config['arch']['kernel_size']),
               str(config['arch']['filters']),
               config['arch']['drop'],
               config['arch']['activation'],
               config['training']['optimizer'],
               r2val, r2persV, r2test, r2persT
               ))
        print(strftime('%Y-%m-%d %H:%M:%S'))

        # Update result in db
        if config is None:
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
    config = load_config_file("configcnv.json")
    wind_data_path = '../../Data'
    lresults = train_convdirregression_architecture(config, False, False, True, True)