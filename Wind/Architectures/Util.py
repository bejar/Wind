"""
.. module:: Util

Util
*************

:Description: Util

    Some utility functions

:Authors: bejar
    

:Version: 

:Created on: 03/12/2018 11:43 

"""
from keras.layers import Bidirectional

__author__ = 'bejar'

def recurrent_encoder_functional(RNN, nlayers, neurons, impl, drop, activation, activation_r, rec_regularizer, k_regularizer, input):
    """
    Returns a functional for several layers of encoder Recurrent layers
    :return:
    """

    if nlayers == 1:
        encoder = RNN(neurons, implementation=impl,
                      recurrent_dropout=drop, activation=activation, recurrent_activation=activation_r,
                      recurrent_regularizer=rec_regularizer, return_sequences=False, kernel_regularizer=k_regularizer)(input)
    else:
        encoder = RNN(neurons, implementation=impl,
                      recurrent_dropout=drop, activation=activation, recurrent_activation=activation_r,
                      recurrent_regularizer=rec_regularizer, return_sequences=True, kernel_regularizer=k_regularizer)(input)

        for i in range(1, nlayers-1):
            encoder = RNN(neurons, implementation=impl,
                          recurrent_dropout=drop, activation=activation, recurrent_activation=activation_r,
                          recurrent_regularizer=rec_regularizer, return_sequences=True,
                          kernel_regularizer=k_regularizer)(encoder)

        encoder = RNN(neurons, implementation=impl,
                          recurrent_dropout=drop, activation=activation, recurrent_activation=activation_r,
                          recurrent_regularizer=rec_regularizer, return_sequences=False,
                          kernel_regularizer=k_regularizer)(encoder)



    return encoder


def recurrent_decoder_functional(RNN, nlayers, neurons, impl, drop, activation, activation_r, rec_regularizer,
                                 k_regularizer, input):
    """
    Returns a functional for several layers of decoder Recurrent layers
    :return:
    """

    decoder = RNN(neurons, implementation=impl,
                  recurrent_dropout=drop, activation=activation, recurrent_activation=activation_r,
                  recurrent_regularizer=rec_regularizer, return_sequences=True, kernel_regularizer=k_regularizer)(input)

    for i in range(1, nlayers):
        decoder = RNN(neurons, implementation=impl,
                      recurrent_dropout=drop, activation=activation, recurrent_activation=activation_r,
                      recurrent_regularizer=rec_regularizer, return_sequences=True,
                      kernel_regularizer=k_regularizer)(decoder)

    return decoder


def recurrent_layer(rnntype, neurons, input_shape, implementation,return_sequences, recurrent_dropout, activation,
                    recurrent_activation, recurrent_regularizer, kernel_regularizer, bidir, bimerge):
    """
    Returns an recurrent layer with the given parameters

    :param rnntype:
    :param neurons:
    :param input_shape:
    :param implementation:
    :param return_sequences:
    :param recurrent_dropout:
    :param activation:
    :param recurrent_activation:
    :param recurrent_regularizer:
    :param kernel_regularizer:
    :param bidir:
    :param bimerge:
    :return:
    """


    if bidir:
        return Bidirectional(
                   rnntype(neurons,
                           input_shape=input_shape,
                           implementation=implementation,
                           return_sequences=return_sequences,
                           recurrent_dropout=recurrent_dropout,
                           activation=activation,
                           recurrent_activation=recurrent_activation,
                           recurrent_regularizer=recurrent_regularizer,
                           kernel_regularizer=kernel_regularizer),
                merge_mode=bimerge
                )
    else:
        return rnntype(neurons,
                           input_shape=input_shape,
                           implementation=implementation,
                           return_sequences=return_sequences,
                           recurrent_dropout=recurrent_dropout,
                           activation=activation,
                           recurrent_activation=recurrent_activation,
                           recurrent_regularizer=recurrent_regularizer,
                           kernel_regularizer=kernel_regularizer)