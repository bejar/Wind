"""
.. module:: Activations

Activations
*************

:Description: Activations

    

:Authors: bejar
    

:Version: 

:Created on: 07/03/2019 12:21 

"""

from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import LeakyReLU, PReLU, ELU
from tensorflow.keras.layers import Activation
from tensorflow.keras import backend as K
from tensorflow.keras.utils import get_custom_objects
import tensorflow as tf

__author__ = 'bejar'


def generate_activation(act_par):
    """
    Uses the value in the configuration to generate the activation layer

    :param act_par:
    :return:
    """

    if type(act_par) == list:
        if len(act_par) == 2:
            atype, par = act_par
            if atype == 'elu':
                return ELU(alpha=par)
            elif atype == 'leaky':
                return LeakyReLU(alpha=par)
            elif atype == 'prelu':
                return PReLU()
            else:
                raise NameError("No such Activation layer")
        elif len(act_par) == 1:
            if act_par[0] == 'snake':
                return Activation(snake)
            elif act_par[0] == 'snakeh2':
                return Activation(snakeh2)
            elif act_par[0] == 'snake2':
                return Activation(snake2)
            elif act_par[0] == 'xsin':
                return Activation(xsin)
            elif act_par[0] == 'swish':
                return Activation(swish)
            else:
                return Activation(act_par[0])
        else:
            raise NameError("No such Activation layer")
    elif type(act_par) == str:
        return Activation(act_par)
    else:
        raise NameError("Wrong parameters for activation layer")


def snake(x):
    """
    Snake activation function

      f(x) = x + sin(x)**2

      The function is computed used the first terms of the Taylor series decomposition
    :param x:
    :return:
    """
    return x + (tf.sin(x) * tf.sin(x))
    # return x + (x*x) - (x*x*x/3)


def snakeh2(x):
    """
    Snake activation function

      f(x) = x + sin(x)**2

      The function is computed used the first terms of the Taylor series decomposition
    :param x:
    :return:
    """
    return x + (2 * tf.sin(0.5 * x) * tf.sin(0.5 * x))


def snake2(x):
    """
    Snake activation function

      f(x) = x + sin(x)**2

      The function is computed used the first terms of the Taylor series decomposition
    :param x:
    :return:
    """
    return x + (0.5 * tf.sin(2 * x) * tf.sin(2 * x))


def xsin(x):
    """
    Snake activation function

      f(x) = x + sin(x)

      The function is computed used the first terms of the Taylor series decomposition
    :param x:
    :return:
    """
    return x + tf.sin(x)


def swish(x):
    """
    Swish activation function
    f(x) = x * sigmoid(x)
    :param x:
    :return:
    """
    return x * tf.sigmoid(x)


get_custom_objects().update({'swish': Activation(swish)})
get_custom_objects().update({'snake': Activation(snake)})
get_custom_objects().update({'xsin': Activation(xsin)})
get_custom_objects().update({'snakeh2': Activation(snakeh2)})
get_custom_objects().update({'snake2': Activation(snake2)})
