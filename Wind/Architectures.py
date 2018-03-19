"""
.. module:: Architectures

Architectures
*************

:Description: Architectures

    

:Authors: bejar
    

:Version: 

:Created on: 19/03/2018 13:25 

"""

from __future__ import print_function
import numpy as np

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout, RepeatVector, TimeDistributed
from keras.layers import LSTM, GRU, CuDNNGRU, CuDNNLSTM, Bidirectional, TimeDistributed, Flatten
from keras.optimizers import RMSprop, SGD
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.regularizers import l1, l2

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

def architectureS2S(idimensions, ahead, neurons, drop, nlayersE, nlayersD, activation, activation_r, rnntype, CuDNN=False,
                    bidirectional=False,
                    rec_reg='l1', rec_regw=0.1, k_reg='l1', k_regw=0.1, impl=1):
    """
    Sequence to Sequence Architecture

    :param neurons:
    :param drop:
    :param nlayers:
    :param activation:
    :param activation_r:
    :param rnntype:
    :param CuDNN:
    :param bidirectional:
    :param rec_reg:
    :param rec_regw:
    :param k_reg:
    :param k_regw:
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
        if nlayersE == 1:

            model.add(
                RNN(neurons, input_shape=(idimensions),
                    recurrent_regularizer=rec_regularizer, kernel_regularizer=k_regularizer))
        else:
            model.add(RNN(neurons, input_shape=(idimensions), return_sequences=True,
                          recurrent_regularizer=rec_regularizer, kernel_regularizer=k_regularizer))
            for i in range(1, nlayersE - 1):
                model.add(RNN(neurons, return_sequences=True, recurrent_regularizer=rec_regularizer,
                              kernel_regularizer=k_regularizer))
            model.add(RNN(neurons, recurrent_regularizer=rec_regularizer, kernel_regularizer=k_regularizer))

        model.add(RepeatVector(ahead))

        for i in range(nlayersD):
            model.add(RNN(neurons, return_sequences=True, recurrent_regularizer=rec_regularizer,
                          kernel_regularizer=k_regularizer))

        model.add(TimeDistributed(Dense(1)))
    else:
        RNN = LSTM if rnntype == 'LSTM' else GRU
        model = Sequential()
        if nlayersE == 1:
            model.add(RNN(neurons, input_shape=(idimensions), implementation=impl,
                          recurrent_dropout=drop, activation=activation, recurrent_activation=activation_r,
                          recurrent_regularizer=rec_regularizer, kernel_regularizer=k_regularizer))
        else:
            model.add(RNN(neurons, input_shape=(idimensions), implementation=impl,
                          recurrent_dropout=drop, activation=activation, recurrent_activation=activation_r,
                          return_sequences=True, recurrent_regularizer=rec_regularizer,
                          kernel_regularizer=k_regularizer))
            for i in range(1, nlayersE - 1):
                model.add(RNN(neurons, recurrent_dropout=drop, implementation=impl,
                              activation=activation, recurrent_activation=activation_r, return_sequences=True,
                              recurrent_regularizer=rec_regularizer, kernel_regularizer=k_regularizer))
            model.add(RNN(neurons, recurrent_dropout=drop, activation=activation,
                          recurrent_activation=activation_r, implementation=impl,
                          recurrent_regularizer=rec_regularizer, kernel_regularizer=k_regularizer))

        model.add(RepeatVector(ahead))

        for i in range(nlayersD):
            model.add(RNN(neurons, recurrent_dropout=drop, implementation=impl,
                          activation=activation, recurrent_activation=activation_r,
                          return_sequences=True, recurrent_regularizer=rec_regularizer,
                          kernel_regularizer=k_regularizer))

        model.add(TimeDistributed(Dense(1)))

    return model


def architectureMLP(idimensions, odimension, activation='linear', rec_reg='l1', rec_regw=0.1, k_reg='l1', k_regw=0.1,
                    dropout=0.0, full_layers=[128]):
    """
    MLP architectureREG

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
