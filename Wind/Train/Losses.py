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

from functools import partial, update_wrapper

import tensorflow as tf
from tensorflow import losses


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
    #weights = K.constant(([1]*(odimensions-3)) + ([odimensions/2, odimensions/2, odimensions/4]))
    weights = tf.range(1,odimensions+1)
    weights = tf.reshape(weights, (1,-1))
    return wrapped_partial(losses.mean_squared_error, weights=weights)

def linear_weighted_mse1(odimensions):
    """
    Computes MSE but weighting the error using the distance in time

    :param y_true:
    :param y_pred:
    :return:
    """
    weights = tf.constant(([1]*(odimensions-3)) + ([odimensions/2, odimensions/2, odimensions/4]))
    weights = tf.reshape(weights, (1,-1, 1))
    return wrapped_partial(losses.mean_squared_error, weights=weights)

def squared_weighted_mse(odimensions):
    """
    Computes MSE but weighting the error using the distance in time

    :param y_true:
    :param y_pred:
    :return:
    """
    weights = tf.range(1,odimensions+1)
    weights = tf.multiply(weights, weights)
    weights = tf.reshape(weights, (1,-1))
    return wrapped_partial(losses.mean_squared_error, weights=weights)

regression_losses = {'wmse':linear_weighted_mse1,
                     'wmse_linear':linear_weighted_mse,
                     'wmse_squared':squared_weighted_mse,
                     }
