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
from Wind.Data.DataSet import Dataset
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

            dataset = Dataset(config=config['data'], data_path=wind_data_path)
            dataset.generate_dataset(ahead=ahead,mode=False)

            # train_x, train_y, val_x, val_y, test_x, test_y = generate_dataset(config['data'], ahead=ahead, mode=False,
            #                                                                   data_path=wind_data_path)

            ############################################
            # Model

            config['idimensions'] = dataset.train_x.shape[1:]

            arch = architecture(config, runconfig)

            if runconfig.multi == 1:
                arch.generate_model()
            else:
                with tf.device('/cpu:0'):
                    arch.generate_model()

            if runconfig.verbose:
                arch.summary()
                dataset.summary()
                print()

            ############################################
            # Training
            arch.train(dataset.train_x, dataset.train_y, dataset.val_x, dataset.val_y)

            ############################################
            # Results

            lresults.append((ahead, arch.evaluate(dataset.val_x, dataset.val_y, dataset.test_x, dataset.test_y)))

            print(strftime('%Y-%m-%d %H:%M:%S'))

            # Update result in db
            if config is not None and not runconfig.proxy:
                from Wind.Training import updateprocess
                updateprocess(config, ahead)

            arch.save('-A%d-R%02d' % (ahead, iter))
            del dataset

    arch.log_result(lresults)

    return lresults

def train_sequence2sequence(architecture, config, runconfig):
    """
    Training process for sequence 2 sequence architectures

    :param architecture:
    :param config:
    :param runconfig:
    :return:
    """
    ahead = config['data']['ahead']

    if not type(ahead) == list:
        ahead = [1, ahead]

    dataset = Dataset(config=config['data'], data_path=wind_data_path)
    dataset.generate_dataset(ahead=ahead, mode=False)

    # train_x, train_y, val_x, val_y, test_x, test_y = generate_dataset(config['data'], ahead=ahead, mode='s2s',
    #                                                                   data_path=wind_data_path)

    batch_size = config['training']['batch']
    nepochs = config['training']['epochs']
    optimizer = config['training']['optimizer']

    if 'iter' in config['training']:
        niter = config['training']['iter']
    else:
        niter = 1

    if type(ahead) == list:
        odimensions = ahead[1] - ahead[0] +1
    else:
        odimensions = ahead

    lresults = []
    for iter in range(niter):

        config['idimensions'] = dataset.train_x.shape[1:]
        config['odimensions'] = odimensions
        arch = architecture(config, runconfig)

        if runconfig.multi == 1:
            arch.generate_model()
        else:
            with tf.device('/cpu:0'):
                arch.generate_model()

        # model = architectureS2S(idimensions=train_x.shape[1:], odimensions=odimensions, neurons=neurons,
        #                         neuronsD=config['arch']['neuronsD'], drop=drop, nlayersE=nlayersE, nlayersD=nlayersD,
        #                         activation=activation, activation_r=activation_r, rnntype=config['arch']['rnn'],
        #                         impl=impl, CuDNN=config['arch']['CuDNN'], rec_reg=rec_reg, rec_regw=rec_regw,
        #                         k_reg=k_reg, k_regw=k_regw)
        if runconfig.verbose:
            arch.summary()
            dataset.summary()
            print()

        ############################################
        # Training
        arch.train(dataset.train_x, dataset.train_y, dataset.val_x, dataset.val_y)


        ############################################
        # Results

        lresults.append((ahead, arch.evaluate(dataset.val_x, dataset.val_y, dataset.test_x, dataset.test_y)))

        print(strftime('%Y-%m-%d %H:%M:%S'))

        arch.save('-A%d-%d-R%02d' % (ahead[0], ahead[1], iter))

    arch.log_result(lresults)

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

        dataset = Dataset(config=config['data'], data_path=wind_data_path)
        dataset.generate_dataset(ahead=ahead, mode=False)

        # train_x, train_y, val_x, val_y, test_x, test_y = generate_dataset(config['data'], ahead=ahead, mode=False,
        #                                                                   data_path=wind_data_path)

        arch = architecture(config, runconfig)
        lresults.append((ahead, arch.evaluate(dataset.val_x, dataset.val_y, dataset.test_x, dataset.test_y)))

        print(strftime('%Y-%m-%d %H:%M:%S'))

        # Update result in db
        if config is not None:
            updateprocess(config, ahead)

        del dataset

    arch.log_result(lresults)
    return lresults


def train_svm_dirregression(architecture, config, runconfig):
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

        dataset = Dataset(config=config['data'], data_path=wind_data_path)
        dataset.generate_dataset(ahead=ahead, mode='svm')

        # train_x, train_y, val_x, val_y, test_x, test_y = generate_dataset(config['data'], ahead=ahead, mode='svm',
        #                                                                   data_path=wind_data_path)

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

        arch = architecture(config, runconfig)

        if runconfig.verbose:
            arch.summary()
            dataset.summary()

            print()

        ############################################
        # Training

        arch.train(dataset.train_x, dataset.train_y)
        svmr = SVR(kernel=kernel, C=C, epsilon=epsilon, degree=degree, coef0=coef0)
        svmr.fit(dataset.train_x, dataset.train_y)

        ############################################
        # Results

        lresults.append((ahead, arch.evaluate(dataset.val_x, dataset.val_y, dataset.test_x, dataset.test_y)))

        val_yp = svmr.predict(dataset.val_x)

        r2val = r2_score(dataset.val_y, val_yp)
        r2persV = r2_score(dataset.val_y[ahead:], dataset.val_y[0:-ahead])

        test_yp = svmr.predict(dataset.test_x)
        r2test = r2_score(dataset.test_y, test_yp)
        r2persT = r2_score(dataset.test_y[ahead:], dataset.test_y[0:-ahead])

        lresults.append((ahead, r2val, r2persV, r2test, r2persT))

        print(strftime('%Y-%m-%d %H:%M:%S'))

        if config is not None:
            updateprocess(config, ahead)

    arch.log_result(lresults)

    return lresults
