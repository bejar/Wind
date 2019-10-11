"""
.. module:: Layers

Layers
*************

:Description: Layers

    Different layers and layer transformations

:Authors: bejar
    

:Version: 

:Created on: 18/09/2019 12:46 

"""
from keras.layers import GlobalAveragePooling1D, Multiply, Dense, LSTM, GRU, Flatten, Dropout, Bidirectional, Input
from keras import backend as K

__author__ = 'bejar'


def squeeze_and_excitation(tensor, ratio=16):
    """
    HU, Shen, Sun "Squeeze and Excitation Networks" https://arxiv.org/abs/1709.01507

    1D version (the original is for 2D convolutions)

    :param x:
    :param ratio:
    :return:
    """
    nb_channel = K.int_shape(tensor)[-1]

    x = GlobalAveragePooling1D()(tensor)
    x = Dense(nb_channel // ratio, activation='relu')(x)
    x = Dense(nb_channel, activation='sigmoid')(x)

    x = Multiply()([tensor, x])
    return x


def generate_recurrent_layer(neurons, impl, drop, activation_r, rec_regularizer, k_regularizer, backwards,
                             rnntype, after, bidir, bimerge, rseq):
    """
    Utility function for generating recurrent layers
    :return:
    """
    RNN = LSTM if rnntype == 'LSTM' else GRU

    if after and rnntype == 'GRU':
        layer = RNN(neurons,
            implementation=impl, return_sequences=rseq,
            recurrent_dropout=drop,
            recurrent_activation=activation_r,
            recurrent_regularizer=rec_regularizer,
            kernel_regularizer=k_regularizer, go_backwards=backwards,
            reset_after=True)
    else:
        layer = RNN(neurons,
            implementation=impl, return_sequences=rseq,
            recurrent_dropout=drop,
            recurrent_activation=activation_r,
            recurrent_regularizer=rec_regularizer,
            kernel_regularizer=k_regularizer, go_backwards=backwards)

    if bidir:
        return Bidirectional(layer, merge_mode=bimerge)
    else:
        return layer
