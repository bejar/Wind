"""
.. module:: Losses

Losses
*************

:Description: Losses

    Special losses for multiple regression in time series

:Authors: bejar
    

:Version: 

:Created on: 05/12/2018 6:10 

"""

from keras import backend as K
from tensorflow import losses

import functools
from functools import partial, update_wrapper

def wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func



__author__ = 'bejar'

def linear_weighted_mse(odimensions):
    """
    Computes MSE but weighting the error using the distance in time

    :param y_true:
    :param y_pred:
    :return:
    """
    # weights = K.ones(odimensions)
    # l = ([1]*(odimensions-1)) + [odimensions]
    weights = K.constant(([1]*(odimensions-3)) + ([odimensions/2, odimensions/2, odimensions]))
    weights = K.reshape(weights, (1,-1, 1))
    return wrapped_partial(losses.mean_squared_error, weights=weights)