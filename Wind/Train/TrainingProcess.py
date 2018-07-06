"""
.. module:: TrainingProcess

TrainingProcess
*************

:Description: TrainingProcess

    

:Authors: bejar
    

:Version: 

:Created on: 06/07/2018 7:53 

"""
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM, GRU, Bidirectional

try:
    from keras.layers import CuDNNGRU, CuDNNLSTM
except ImportError:
    _has_CuDNN = False
else:
    _has_CuDNN = True

from keras.optimizers import RMSprop, SGD
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.regularizers import l1, l2

try:
    from keras.utils import multi_gpu_model
except ImportError:
    _has_multigpu = False
else:
    _has_multigpu = True

import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score
from Wind.Data import generate_dataset
from Wind.Config import wind_data_path
from time import time, strftime
import os
from Wind.Training import updateprocess
from sklearn.svm import SVR

__author__ = 'bejar'


def train_dirregression(architecture, config, runconfig):
    """
    Training process for architecture with direct regression of ahead time steps

    :return:
    """

    if type(config['data']['ahead']) == list:
        iahead, sahead = config['data']['ahead']
    else:
        iahead, sahead = 1, config['data']['ahead']

    lresults = []
    if 'iter' in config['training']:
        niter = config['training']['iter']
    else:
        niter = 1

    for iter in range(niter):

        for ahead in range(iahead, sahead + 1):

            if runconfig.verbose:
                print('-----------------------------------------------------------------------------')
                print('Steps Ahead = %d ' % ahead)

            train_x, train_y, val_x, val_y, test_x, test_y = generate_dataset(config['data'], ahead=ahead, mode=False,
                                                                              data_path=wind_data_path)

            ############################################
            # Model



            config['idimensions'] = train_x.shape[1:]

            arch = architecture(config, runconfig)

            if runconfig.multi == 1:
                arch.generate_model()
            else:
                with tf.device('/cpu:0'):
                    arch.generate_model()

            if runconfig.verbose:
                arch.summary()
                print('Tr:', train_x.shape, train_y.shape, 'Val:', val_x.shape, val_y.shape, 'Ts:', test_x.shape,
                      test_y.shape)
                print()

            ############################################
            # Training
            arch.train(train_x, train_y, val_x, val_y)

            ############################################
            # Results

            lresults.append((ahead, arch.evaluate(val_x, val_y, test_x, test_y)))

            print(strftime('%Y-%m-%d %H:%M:%S'))

            # Update result in db
            if config is not None and not runconfig.proxy:
                from Wind.Training import updateprocess
                updateprocess(config, ahead)

            arch.save('-A%d-R%02d' % (ahead, iter))
            del train_x, train_y, test_x, test_y, val_x, val_y

    arch.log_results(lresults)

    return lresults


def train_persistence(architecture, config, runconfig):
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

        if runconfig.verbose:
            print('-----------------------------------------------------------------------------')
            print('Steps Ahead = %d ' % ahead)

        train_x, train_y, val_x, val_y, test_x, test_y = generate_dataset(config['data'], ahead=ahead, mode=False,
                                                                          data_path=wind_data_path)

        r2persV = r2_score(val_y[ahead:], val_y[0:-ahead])

        r2persT = r2_score(test_y[ahead:, 0], test_y[0:-ahead, 0])

        lresults.append((ahead, r2persV, r2persT))

        print('%s | DNM= %s, DS= %d, AH= %d, R2PV = %3.5f, R2PT = %3.5f' %
              (config['arch']['mode'],
               config['data']['datanames'][0],
               config['data']['dataset'],
               ahead,
               r2persV, r2persT
               ))

        print(strftime('%Y-%m-%d %H:%M:%S'))

        # Update result in db
        if config is not None:
            updateprocess(config, ahead)

        del train_x, train_y, test_x, test_y, val_x, val_y

    return lresults


def train_svm_dirregression(arcchitecture, config, runconfig):
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

        if runconfig.verbose:
            print('-----------------------------------------------------------------------------')
            print('Steps Ahead = %d ' % ahead)

        train_x, train_y, val_x, val_y, test_x, test_y = generate_dataset(config['data'], ahead=ahead, mode='svm',
                                                                          data_path=wind_data_path)

        # train_x = np.squeeze(train_x, axis=2)
        # val_x = np.squeeze(val_x, axis=2)
        # test_x = np.squeeze(test_x, axis=2)

        ############################################
        # Model

        kernel = config['arch']['kernel']
        C = config['arch']['C']
        epsilon = config['arch']['epsilon']
        degree = config['arch']['degree']
        coef0 = config['arch']['coef0']

        if runconfig.verbose:
            print(
            'lag: ', config['data']['lag'], '/kernel: ', kernel, '/C: ', C, '/epsilon:', epsilon, '/degree:', degree)
            print('Tr:', train_x.shape, train_y.shape, 'Val:', val_x.shape, val_y.shape, 'Ts:', test_x.shape,
                  test_y.shape)
            print()

        ############################################
        # Training

        svmr = SVR(kernel=kernel, C=C, epsilon=epsilon, degree=degree, coef0=coef0)
        svmr.fit(train_x, train_y)

        ############################################
        # Results

        val_yp = svmr.predict(val_x)

        r2val = r2_score(val_y, val_yp)
        r2persV = r2_score(val_y[ahead:], val_y[0:-ahead])

        test_yp = svmr.predict(test_x)
        r2test = r2_score(test_y, test_yp)
        r2persT = r2_score(test_y[ahead:], test_y[0:-ahead])

        lresults.append((ahead, r2val, r2persV, r2test, r2persT))
        print(
                    '%s |  AH=%d, KRNL= %s, C= %3.5f, EPS= %3.5f, DEG=%d, COEF0= %d, R2V = %3.5f, R2PV = %3.5f, R2T = %3.5f, R2PT = %3.5f' %
                    (config['arch']['mode'], ahead,
                     config['arch']['kernel'],
                     config['arch']['C'],
                     config['arch']['epsilon'],
                     config['arch']['degree'],
                     config['arch']['coef0'],
                     r2val, r2persV, r2test, r2persT
                     ))
        print(strftime('%Y-%m-%d %H:%M:%S'))

        # Update result in db
        if config is not None:
            updateprocess(config, ahead)

    return lresults
