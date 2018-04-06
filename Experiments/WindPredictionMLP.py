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
from keras.layers import Dense, Activation, Input, Flatten, Reshape, Dropout
from keras.layers import LSTM, GRU, CuDNNGRU, CuDNNLSTM, Bidirectional
from keras.optimizers import RMSprop, SGD
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.regularizers import l1, l2
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

import argparse
from time import time

from Wind.Util import load_config_file
from Wind.Data import generate_dataset
from Wind.Config import wind_data_path

__author__ = 'bejar'


def architectureMLP(idimensions, odimension, activation='linear', rec_reg='l1', rec_regw=0.1, k_reg='l1', k_regw=0.1,
                    dropout=0.0, full_layers=[128]):
    """
    MLP architectureDirRegression

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config2', help='Experiment configuration')
    parser.add_argument('--verbose', help="Verbose output (enables Keras verbose output)", action='store_true',
                        default=False)
    parser.add_argument('--gpu', help="Use LSTM/GRU gpu implementation", action='store_true', default=False)
    args = parser.parse_args()

    verbose = 0 if args.verbose else 1
    impl = 1 if args.gpu else 1

    config = load_config_file(args.config)
    ############################################
    # Data

    ahead = config['data']['ahead']

    train_x, train_y, val_x, val_y, test_x, test_y = generate_dataset(config['data'], ahead=ahead, mode='mlp',
                                                                      data_path=wind_data_path)
    print('Tr:', train_x.shape, train_y.shape, 'Val:', val_x.shape, val_y.shape, 'Ts:', test_x.shape, test_y.shape)

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
    dropout = config['arch']['drop']

    model = architectureMLP(idimensions=train_x.shape[1:], odimension=config['data']['ahead'], activation='sigmoid',
                            rec_reg=rec_reg, rec_regw=rec_regw, k_reg=k_reg, k_regw=k_regw, dropout=dropout,
                            full_layers=config['arch']['full'])

    model.summary()

    print('lag: ', config['data']['lag'], '/Neurons: ', neurons, '/Layers: ', nlayers, '/Activation:', activation,
          activation_r)
    print('Tr:', train_x.shape, train_y.shape, 'Val:', val_x.shape, val_y.shape, 'Ts:', test_x.shape, test_y.shape)
    print()

    ############################################
    # Training
    tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
    mcheck = ModelCheckpoint(filepath='./model.h5', monitor='val_loss', verbose=0, save_best_only=True,
                             save_weights_only=False, mode='auto', period=1)
    # optimizer = RMSprop(lr=0.00001)
    optimizer = config['training']['optimizer']
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    batch_size = config['training']['batch']
    nepochs = config['training']['epochs']

    model.fit(train_x, train_y, batch_size=batch_size, epochs=nepochs, validation_data=(val_x, val_y),
              verbose=verbose, callbacks=[mcheck])

    ############################################
    # Results

    model = load_model('./model.h5')

    val_yp = model.predict(val_x, batch_size=batch_size, verbose=0)
    for i in range(1, ahead + 1):
        print('R2 val= ', i, r2_score(val_y[:, i - 1], val_yp[:, i - 1]))
        print('R2 val persistence =', i, r2_score(val_y[i:, 0], val_y[0:-i, 0]))

    # score = model.evaluate(test_x, test_y, batch_size=batch_size, verbose=0)
    print()
    # print('MSE test= ', score)
    # print('MSE test persistence =', mean_squared_error(test_y[ahead:], test_y[0:-ahead]))
    test_yp = model.predict(test_x, batch_size=batch_size, verbose=0)
    for i in range(1, ahead + 1):
        print('R2 test= ', i, r2_score(test_y[:, i - 1], test_yp[:, i - 1]))
        print('R2 test persistence =', i, r2_score(test_y[i:, 0], test_y[0:-i, 0]))

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
