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
from time import time, strftime

from Wind.Data import generate_dataset
from Wind.Config import wind_data_path
import sys
from Wind.Private.DBConfig import mongoconnection
from copy import deepcopy
from pymongo import MongoClient
import requests
import json
import socket

__author__ = 'bejar'


def architectureREG(idimensions, neurons, drop, nlayers, activation, activation_r, rnntype, CuDNN=False, bidirectional=False, bimerge='sum',
                    rec_reg='l1', rec_regw=0.1, k_reg='l1', k_regw=0.1, full=[1], impl=1):
    """
    Regression RNN architecture

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
                RNN(neurons, input_shape=(idimensions), recurrent_regularizer=rec_regularizer,
                    kernel_regularizer=k_regularizer))
        else:
            model.add(RNN(neurons, input_shape=(idimensions), return_sequences=True,
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
                                  input_shape=(idimensions), merge_mode=bimerge))
            else:
                model.add(
                    Bidirectional(RNN(neurons, implementation=impl,
                                      recurrent_dropout=drop,
                                      activation=activation,
                                      recurrent_activation=activation_r,
                                      return_sequences=True,
                                      recurrent_regularizer=rec_regularizer,
                                      kernel_regularizer=k_regularizer),
                                  input_shape=(idimensions), merge_mode=bimerge))
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
                model.add(RNN(neurons, input_shape=(idimensions), implementation=impl,
                              recurrent_dropout=drop, activation=activation, recurrent_activation=activation_r,
                              recurrent_regularizer=rec_regularizer, kernel_regularizer=k_regularizer))
            else:
                model.add(RNN(neurons, input_shape=(idimensions), implementation=impl,
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


def getconfig(proxy=False):
    """
    Gets a config from the database
    :return:
    """
    if not proxy:
        client = MongoClient(mongoconnection.server)
        db = client[mongoconnection.db]
        db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
        col = db[mongoconnection.col]
        config = col.find_one({'status': 'pending'})
        if config is not None:
            col.update({'_id': config['_id']}, {'$set': {'status': 'working'}})
            col.update({'_id': config['_id']}, {'$set': {'btime': strftime('%Y-%m-%d %H:%M:%S')}})
            col.update({'_id': config['_id']}, {'$set': {'host': socket.gethostname().split('.')[0]}})
        return config
    else:
        return requests.get('http://polaris.cs.upc.edu:9000/Proxy').json()



def saveconfig(config, lresults, proxy=False):
    """
    Saves a config in the database
    :param proxy:
    :return:
    """

    if not proxy:
        client = MongoClient(mongoconnection.server)
        db = client[mongoconnection.db]
        db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
        col = db[mongoconnection.col]
        col.update({'_id': config['_id']}, {'$set': {'status': 'done'}})
        col.update({'_id': config['_id']}, {'$set': {'result': lresults}})
        col.update({'_id': config['_id']}, {'$set': {'etime': strftime('%Y-%m-%d %H:%M:%S')}})
    else:
        config['results'] = lresults
        requests.post('http://polaris.cs.upc.edu:9000/Proxy', params={'res': json.dumps(config)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', help="Verbose output (enables Keras verbose output)", action='store_true',
                        default=False)
    parser.add_argument('--gpu', help="Use LSTM/GRU gpu implementation", action='store_true', default=False)
    parser.add_argument('--best', help="Save weights best in test", action='store_true', default=False)
    parser.add_argument('--tboard', help="Save log for tensorboard", action='store_true', default=False)
    parser.add_argument('--nbatches', help="Number of configurations to run", default=1, type=int)
    parser.add_argument('--proxy', help="Access configurations throught proxy", action='store_true', default=False)
    args = parser.parse_args()

    verbose = 1 if args.verbose else 0
    impl = 2 if args.gpu else 1




    config = getconfig(proxy=args.proxy)
    if config is not None:

        ############################################
        # Data

        print('Running job %s %s' % (config['_id'], strftime('%Y-%m-%d %H:%M:%S')))

        sahead = config['data']['ahead']
        lresults = []
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

            model = architectureREG(idimensions=train_x.shape[1:], neurons=neurons, drop=drop, nlayers=nlayers, activation=activation,
                                    activation_r=activation_r, rnntype=config['arch']['rnn'], CuDNN=config['arch']['CuDNN'],
                                    rec_reg=rec_reg, rec_regw=rec_regw, k_reg=k_reg, k_regw=k_regw,
                                    bidirectional=bidirectional, bimerge=bimerge,
                                    full=config['arch']['full'], impl=impl)
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

            # Update result in db
            lresults.append((ahead, r2val, r2persV, r2test, r2persT))
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
            print(strftime('%Y-%m-%d %H:%M:%S'))

            try:
                os.remove(modfile)
            except OSError:
                pass

            del train_x, train_y, test_x, test_y, val_x, val_y
            del model

        saveconfig(config, lresults, proxy=args.proxy)
