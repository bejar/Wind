"""
.. module:: WindPrediction

WindPrediction
*************

:Description: WindPrediction

:Authors: bejar
    

:Version: 

:Created on: 06/09/2017 9:47 

"""

from __future__ import print_function
import numpy as np

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.layers import LSTM, GRU, CuDNNGRU, CuDNNLSTM, Bidirectional, TimeDistributed, Flatten
from keras.optimizers import RMSprop, SGD
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.regularizers import l1, l2
#import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

import argparse
from time import time

from Wind.Util import load_config_file
from Wind.Data import generate_dataset
from Wind.Config import wind_data_path
import itertools
from copy import deepcopy
import pprint
import sys

__author__ = 'bejar'


def architecture(neurons, drop, nlayers, activation, activation_r, rnntype, CuDNN=False, bidirectional=False, bimerge='sum',
                 rec_reg='l1', rec_regw=0.1, k_reg='l1', k_regw=0.1, full=[1]):
    """
    RNN architecture

    :return:
    """
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
        model = Sequential()
        if nlayers == 1:
            model.add(
                RNN(neurons, input_shape=(train_x.shape[1], train_x.shape[2]), recurrent_regularizer=rec_regularizer,
                    kernel_regularizer=k_regularizer))
        else:
            model.add(RNN(neurons, input_shape=(train_x.shape[1], train_x.shape[2]), return_sequences=True,
                          recurrent_regularizer=rec_regularizer, kernel_regularizer=k_regularizer))
            for i in range(1, nlayers - 1):
                model.add(RNN(neurons, return_sequences=True, recurrent_regularizer=rec_regularizer,
                              kernel_regularizer=k_regularizer))
            model.add(RNN(neurons, recurrent_regularizer=rec_regularizer, kernel_regularizer=k_regularizer))
        for l in full:
            model.add(Dense(l))
    else:
        RNN = LSTM if rnntype == 'LSTM' else GRU
        model = Sequential()
        if bidirectional:
            if nlayers == 1:
                model.add(
                    Bidirectional(RNN(neurons, implementation=impl,
                                      recurrent_dropout=drop,
                                      activation=activation,
                                      recurrent_activation=activation_r,
                                      recurrent_regularizer=rec_regularizer,
                                      kernel_regularizer=k_regularizer),
                                  input_shape=(train_x.shape[1], train_x.shape[2]), merge_mode=bimerge))
            else:
                model.add(
                    Bidirectional(RNN(neurons, implementation=impl,
                                      recurrent_dropout=drop,
                                      activation=activation,
                                      recurrent_activation=activation_r,
                                      return_sequences=True,
                                      recurrent_regularizer=rec_regularizer,
                                      kernel_regularizer=k_regularizer),
                                  input_shape=(train_x.shape[1], train_x.shape[2]), merge_mode=bimerge))
                for i in range(1, nlayers - 1):
                    model.add(Bidirectional(RNN(neurons, recurrent_dropout=drop, implementation=impl,
                                                activation=activation, recurrent_activation=activation_r,
                                                return_sequences=True,
                                                recurrent_regularizer=rec_regularizer,
                                                kernel_regularizer=k_regularizer), merge_mode=bimerge))
                model.add(Bidirectional(RNN(neurons, recurrent_dropout=drop, activation=activation,
                                            recurrent_activation=activation_r, implementation=impl,
                                            recurrent_regularizer=rec_regularizer, kernel_regularizer=k_regularizer), merge_mode=bimerge))
            model.add(Dense(1))
        else:
            if nlayers == 1:
                model.add(RNN(neurons, input_shape=(train_x.shape[1], train_x.shape[2]), implementation=impl,
                              recurrent_dropout=drop, activation=activation, recurrent_activation=activation_r,
                              recurrent_regularizer=rec_regularizer, kernel_regularizer=k_regularizer))
            else:
                model.add(RNN(neurons, input_shape=(train_x.shape[1], train_x.shape[2]), implementation=impl,
                              recurrent_dropout=drop, activation=activation, recurrent_activation=activation_r,
                              return_sequences=True, recurrent_regularizer=rec_regularizer,
                              kernel_regularizer=k_regularizer))
                for i in range(1, nlayers - 1):
                    model.add(RNN(neurons, recurrent_dropout=drop, implementation=impl,
                                  activation=activation, recurrent_activation=activation_r, return_sequences=True,
                                  recurrent_regularizer=rec_regularizer, kernel_regularizer=k_regularizer))
                model.add(RNN(neurons, recurrent_dropout=drop, activation=activation,
                              recurrent_activation=activation_r, implementation=impl,
                              recurrent_regularizer=rec_regularizer, kernel_regularizer=k_regularizer))
            for l in full:
                model.add(Dense(l))

    return model

def generate_configs(config):
    """
    Generates all possible individual configs from the fields with multiple values
    :param config:
    :return:
    """
    lconf = [{}]

    for f1 in config:
        for f2 in config[f1]:
            lnconf = []
            for v in config[f1][f2]:
                for c in lconf:
                    cp = deepcopy(c)
                    if f1 in cp:
                        cp[f1][f2] = v
                    else:
                        cp[f1] = {f2: v}
                    lnconf.append(cp)
            lconf = lnconf

    return lconf


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configBatch1', help='Experiment configuration')
    parser.add_argument('--verbose', help="Verbose output (enables Keras verbose output)", action='store_true',
                        default=False)
    parser.add_argument('--gpu', help="Use LSTM/GRU gpu implementation", action='store_true', default=False)
    parser.add_argument('--best', help="Save weights best in test", action='store_true', default=False)
    parser.add_argument('--tboard', help="Save log for tensorboard", action='store_true', default=False)
    parser.add_argument('--test', help="Print configurations and stop", action='store_true', default=False)
    args = parser.parse_args()

    verbose = 1 if args.verbose else 0
    impl = 2 if args.gpu else 1

    configB = load_config_file(args.config)
    if args.test:
        lconf = generate_configs(configB)
        print(len(lconf))
        sys.exit(0)

    rescode = int(time())
    for dname in configB['data']['datanames']:
        resfile = open('result-%d-%s.txt'% (rescode, dname[0]), 'a')
        resfile.write('DNAME, DATAS, VARS, LAG, AHEAD, RNN, Bi, NLAY, NNEUR, DROP, ACT, RACT, '
                      'OPT, R2Val, R2persV, R2Test, R2persT\n')
        resfile.close()

    ############################################
    # Data

    for config in generate_configs(configB):

        sahead = config['data']['ahead']

        for ahead in range(1, sahead + 1):

            if args.verbose:
                print('-----------------------------------------------------------------------------')
                print('Steps Ahead = %d ' % ahead)

            train_x, train_y, val_x, val_y, test_x, test_y = generate_dataset(config['data'], ahead=ahead, mode=False,
                                                                              data_path=wind_data_path)

            ############################################
            # Model

            neurons = config['arch']['neurons']
            drop = config['arch']['drop']
            nlayers = config['arch']['nlayers']  # >= 1

            activation = config['arch']['activation']
            activation_r = config['arch']['activation_r']
            rec_reg = config['arch']['rec_reg']
            rec_regw = config['arch']['rec_regw']
            k_reg = config['arch']['k_reg']
            k_regw = config['arch']['k_regw']
            bidirectional = config['arch']['bidirectional']
            bimerge = config['arch']['bimerge']

            model = architecture(neurons=neurons, drop=drop, nlayers=nlayers, activation=activation,
                                 activation_r=activation_r, rnntype=config['arch']['rnn'], CuDNN=config['arch']['CuDNN'],
                                 rec_reg=rec_reg, rec_regw=rec_regw, k_reg=k_reg, k_regw=k_regw,
                                 bidirectional=bidirectional, bimerge=bimerge,
                                 full=config['arch']['full'])
            if args.verbose:
                model.summary()

                print('lag: ', config['data']['lag'], '/Neurons: ', neurons, '/Layers: ', nlayers, '/Activation:',
                      activation, activation_r)
                print('Tr:', train_x.shape, train_y.shape, 'Val:', val_x.shape, val_y.shape, 'Ts:', test_x.shape,
                      test_y.shape)
                print()

            ############################################
            # Training
            cbacks = []
            if args.tboard:
                tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
                cbacks.append(tensorboard)

            if args.best:
                modfile = './model%d.h5' % int(time())
                mcheck = ModelCheckpoint(filepath=modfile, monitor='val_loss', verbose=0, save_best_only=True,
                                         save_weights_only=False, mode='auto', period=1)
                cbacks.append(mcheck)

            optimizer = config['training']['optimizer']
            if optimizer == 'rmsprop':
                if 'lrate' in config['training']:
                    optimizer = RMSprop(lr=config['training']['lrate'])
                else:
                    optimizer = RMSprop(lr=0.001)

            model.compile(loss='mean_squared_error', optimizer=optimizer)

            batch_size = config['training']['batch']
            nepochs = config['training']['epochs']

            model.fit(train_x, train_y, batch_size=batch_size, epochs=nepochs, validation_data=(val_x, val_y),
                      verbose=verbose, callbacks=cbacks)

            ############################################
            # Results

            if args.best:
                model = load_model(modfile)

            score = model.evaluate(val_x, val_y, batch_size=batch_size, verbose=0)
            # print('MSE val= ', score)
            # print('MSE val persistence =', mean_squared_error(val_y[ahead:], val_y[0:-ahead]))
            val_yp = model.predict(val_x, batch_size=batch_size, verbose=0)
            r2val = r2_score(val_y, val_yp)
            r2persV = r2_score(val_y[ahead:], val_y[0:-ahead])
            # print('R2 val= ', r2val)
            # print('R2 val persistence =', r2persV)

            score = model.evaluate(test_x, test_y, batch_size=batch_size, verbose=0)
            # print()
            # print('MSE test= ', score)
            # print('MSE test persistence =', mean_squared_error(test_y[ahead:], test_y[0:-ahead]))
            test_yp = model.predict(test_x, batch_size=batch_size, verbose=0)
            r2test = r2_score(test_y, test_yp)
            r2persT = r2_score(test_y[ahead:, 0], test_y[0:-ahead, 0])
            # print('R2 test= ', r2test)
            # print('R2 test persistence =', r2persT)

            resfile = open('result-%d-%s.txt'%(rescode, config['data']['datanames'][0]), 'a')
            resfile.write('%s, %d, %d, %d, %d, %s, %s, %d, %d, %3.2f, %s, %s, '
                          '%s, %3.5f, %3.5f, %3.5f, %3.5f\n' %
                          (config['data']['datanames'][0],
                           config['data']['dataset'],
                           len(config['data']['vars']),
                           config['data']['lag'],
                           ahead,
                           config['arch']['rnn'],
                           config['arch']['bimerge'] if config['arch']['bidirectional'] else 'no',
                           config['arch']['nlayers'],
                           config['arch']['neurons'],
                           config['arch']['drop'],
                           config['arch']['activation'],
                           config['arch']['activation_r'],
                           config['training']['optimizer'],
                           r2val, r2persV, r2test, r2persT
                           ))
            resfile.close()
            print('DNM= %s, DS= %d, V= %d, LG= %d, AH= %d, RNN= %s, Bi=%s, LY= %d, NN= %d, DR= %3.2f, AF= %s, RAF= %s, '
                      'OPT= %s, R2V = %3.5f, R2PV = %3.5f, R2T = %3.5f, R2PT = %3.5f' %
                      (config['data']['datanames'][0],
                       config['data']['dataset'],
                       len(config['data']['vars']),
                       config['data']['lag'],
                       ahead,
                       config['arch']['rnn'],
                       config['arch']['bimerge'] if config['arch']['bidirectional'] else 'no',
                       config['arch']['nlayers'],
                       config['arch']['neurons'],
                       config['arch']['drop'],
                       config['arch']['activation'],
                       config['arch']['activation_r'],
                       config['training']['optimizer'],
                       r2val, r2persV, r2test, r2persT
                           ))

            try:
                os.remove(modfile)
            except OSError:
                pass
    # ----------------------------------------------
    # plt.subplot(2, 1, 1)
    # plt.plot(test_predict, color='r')
    # plt.plot(test_y, color='b')
    # plt.subplot(2, 1, 2)
    # plt.plot(test_y - test_predict, color='r')
    # plt.show()
    #
    # # step prediction
    #
    # obs = np.zeros((1, lag, 1))
    # pwindow = 5
    # lwpred = []
    # for i in range(0, 2000-(2*lag)-pwindow, pwindow):
    #     # copy the observations values
    #     for j in range(lag):
    #         obs[0, j, 0] = test_y[i+j]
    #
    #     lpred = []
    #     for j in range(pwindow):
    #         pred = model.predict(obs)
    #         lpred.append(pred)
    #         for k in range(lag-1):
    #             obs[0, k, 0] = obs[0, k+1, 0]
    #         obs[0, -1, 0] = pred
    #
    #
    #     lwpred.append((i, np.array(lpred)))
    #
    #
    # plt.subplot(1, 1, 1)
    # plt.plot(test_y[0:2100], color='b')
    # for i, (_, pred) in zip(range(0, 2000, pwindow), lwpred):
    #     plt.plot(range(i+lag, i+lag+pwindow), np.reshape(pred,pwindow), color='r')
    # plt.show()
