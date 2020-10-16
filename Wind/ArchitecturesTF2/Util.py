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

