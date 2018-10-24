"""
.. module:: MLPRegressions2s

MLPRegression
*************

:Description: MLPRegressions2s

 Sequence to seguence Direct regression with MLP

:Authors: bejar
    

:Version: 

:Created on: 06/04/2018 14:24 

"""

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout, Conv1D, Flatten
from keras.optimizers import RMSprop, SGD
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.regularizers import l1, l2
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score
from Wind.Data import generate_dataset
from Wind.Config import wind_data_path
from time import time, strftime
import numpy as np
import os


try:
    from keras.utils import multi_gpu_model
except ImportError:
    _has_multigpu = False
else:
    _has_multigpu = True

__author__ = 'bejar'


def architectureConvos2s(idimensions, odimension, filters, kernel_size, strides, drop, activation, activationfl,
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

    model.add(Dense(odimension, activation='linear'))

    return model



def train_convo_regs2s_architecture(config, verbose, tboard, best, early, multi=1, save=False, remote=False):
    """
     Training process for MLP architecture with regression of ahead time steps

    :param config:
    :param impl:
    :param verbose:
    :return:
    """
    ahead = config['data']['ahead']

    train_x, train_y, val_x, val_y, test_x, test_y = generate_dataset(config['data'], ahead=ahead, mode='cnn',
                                                                      data_path=wind_data_path, remote=remote)


    # train_y = np.reshape(train_y, (train_y.shape[0], train_y.shape[1]))
    # test_y = np.reshape(test_y, (test_y.shape[0], test_y.shape[1]))
    # val_y = np.reshape(val_y, (val_y.shape[0], val_y.shape[1]))

    ############################################
    # Model

    filters = config['arch']['filters']
    strides = config['arch']['strides']
    kernel_size = config['arch']['kernel_size']

    activation = config['arch']['activation']
    activationfl = config['arch']['activationfl']

    rec_reg = config['arch']['rec_reg']
    rec_regw = config['arch']['rec_regw']
    k_reg = config['arch']['k_reg']
    k_regw = config['arch']['k_regw']

    dropout = config['arch']['drop']
    if multi == 1:
        model = architectureConvos2s(idimensions=train_x.shape[1:], odimension=config['data']['ahead'], activation=activation,
                                    activationfl=activationfl, k_reg=k_reg, k_regw=k_regw, drop=dropout,
                                   full=config['arch']['full'], filters=filters, kernel_size=kernel_size, strides=strides)
    else:
        with tf.device('/cpu:0'):
            model = architectureConvos2s(idimensions=train_x.shape[1:], odimension=config['data']['ahead'], activation=activation,
                                        k_reg=k_reg, k_regw=k_regw, drop=dropout,
                                       full=config['arch']['full'], kernel_size=kernel_size, strides=strides)

    if verbose:
        model.summary()
        print('lag: ', config['data']['lag'],  '/Filters: ', config['arch']['filters'], '/Activation:', activation)
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
        early = EarlyStopping(monitor='val_loss', patience=10, verbose=0)
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

    val_yp = model.predict(val_x, batch_size=batch_size, verbose=0)
    test_yp = model.predict(test_x, batch_size=batch_size, verbose=0)

    lresults = []

    for i in range(1, ahead + 1):
        lresults.append((i,
                         r2_score(val_y[:, i - 1], val_yp[:, i - 1]),
                         r2_score(val_y[i:, 0], val_y[0:-i, 0]),
                         r2_score(test_y[:, i - 1], test_yp[:, i - 1]),
                         r2_score(test_y[i:, 0], test_y[0:-i, 0])))


    for i, r2val, r2persV, r2test, r2persT in lresults:
        print('%s | DNM= %s, DS= %d, V= %d, LG= %d, ST= %s, KS= %s, FLT= %s, AH= %d, FL= %s, DR= %3.2f, AF= %s, '
              'OPT= %s, R2V = %3.5f, R2PV = %3.5f, R2T = %3.5f, R2PT = %3.5f' %
              (config['arch']['mode'],
               config['data']['datanames'][0],
               config['data']['dataset'],
               len(config['data']['vars']),
               config['data']['lag'],
               str(config['arch']['strides']),
               str(config['arch']['kernel_size']),
               str(config['arch']['filters']),
               i,
               str(config['arch']['full']),
               config['arch']['drop'],
               config['arch']['activation'],
               config['training']['optimizer'],
               r2val, r2persV, r2test, r2persT
               ))

    try:
        os.remove(modfile)
    except OSError:
        pass

    return lresults
