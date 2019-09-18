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
from keras.layers import GlobalAveragePooling1D, Multiply, Dense
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