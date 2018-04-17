"""
.. module:: MLPRegression

MLPRegression
*************

:Description: MLPRegression

 Multiple Direct regression with MLP

:Authors: bejar
    

:Version: 

:Created on: 06/04/2018 14:24 

"""

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import RMSprop, SGD
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.regularizers import l1, l2
from keras.utils import multi_gpu_model
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score
from Wind.Data import generate_dataset
from Wind.Config import wind_data_path
from time import time, strftime
import os

__author__ = 'bejar'


def architectureMLP(idimensions, odimension, activation='linear', rec_reg='l1', rec_regw=0.1, k_reg='l1', k_regw=0.1,
                    dropout=0.0, full_layers=[128]):
    """
    Arquitecture with direct regression using MLP

    :return:
    """
    model = Sequential()
    model.add(Dense(full_layers[0], input_shape=idimensions))
    model.add(Dropout(rate=dropout))
    for units in full_layers[1:]:
        model.add(Dense(units=units, activation=activation))
        model.add(Dropout(rate=dropout))

    model.add(Dense(odimension))

    return model


def train_MLP_regdir_architecture(config, verbose, tboard, best, early, multi=1):
    """
     Training process for MLP architecture with regression of ahead time steps

    :param config:
    :param impl:
    :param verbose:
    :return:
    """
    ahead = config['data']['ahead']

    train_x, train_y, val_x, val_y, test_x, test_y = generate_dataset(config['data'], ahead=ahead, mode='mlp',
                                                                      data_path=wind_data_path)

    ############################################
    # Model

    neurons = config['arch']['neurons']
    drop = config['arch']['drop']
    nlayers = config['arch']['nlayers']  # >= 1

    activation = config['arch']['activation']

    rec_reg = config['arch']['rec_reg']
    rec_regw = config['arch']['rec_regw']
    k_reg = config['arch']['k_reg']
    k_regw = config['arch']['k_regw']

    dropout = config['arch']['drop']
    if multi == 1:
        model = architectureMLP(idimensions=train_x.shape[1:], odimension=config['data']['ahead'], activation=activation,
                            rec_reg=rec_reg, rec_regw=rec_regw, k_reg=k_reg, k_regw=k_regw, dropout=dropout,
                            full_layers=config['arch']['full'])
    else:
        with tf.device('/cpu:0'):
            model = architectureMLP(idimensions=train_x.shape[1:], odimension=config['data']['ahead'], activation=activation,
                            rec_reg=rec_reg, rec_regw=rec_regw, k_reg=k_reg, k_regw=k_regw, dropout=dropout,
                            full_layers=config['arch']['full'])

    if verbose:
        model.summary()
        print('lag: ', config['data']['lag'], '/Neurons: ', neurons, '/Layers: ', nlayers, '/Activation:', activation)
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
        print('DNM= %s, DS= %d, V= %d, LG= %d, AH= %d, FL= %s, DR= %3.2f, AF= %s, '
              'OPT= %s, R2V = %3.5f, R2PV = %3.5f, R2T = %3.5f, R2PT = %3.5f' %
              (config['data']['datanames'][0],
               config['data']['dataset'],
               len(config['data']['vars']),
               config['data']['lag'],
               i,str(config['arch']['full']),
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
