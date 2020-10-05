"""
.. module:: Activations

Activations
*************

:Description: Activations

    

:Authors: bejar
    

:Version: 

:Created on: 07/03/2019 12:21 

"""

from keras.layers import Activation
from keras.layers import LeakyReLU, PReLU, ELU
from keras.layers import Activation
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
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
                return Snake(snake)
            elif act_par[0] == 'snakeh2':
                return Snakeh2(snakeh2)
            elif act_par[0] == 'snake2':
                return Snake2(snake2)
            elif act_par[0] == 'snake3':
                return Snake3(snake3)
            elif act_par[0] == 'snake4':
                return Snake4(snake4)
            elif act_par[0] == 'snake5':
                return Snake5(snake5)
            elif act_par[0] == 'xsin':
                return Xsin(xsin)
            elif act_par[0] == 'swish':
                return Swish(swish)
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
    :param X:
    :return:
    """
    return x + (tf.sin(x) * tf.sin(x))
    #return x + (x*x) - (x*x*x/3)

def snakeh2(x):
    """
    Snake activation function

      f(x) = x + sin(x)**2

      The function is computed used the first terms of the Taylor series decomposition
    :param X:
    :return:
    """
    return x + (2 * tf.sin(0.5*x) * tf.sin(0.5*x))

def snake2(x):
    """
    Snake activation function

      f(x) = x + sin(x)**2

      The function is computed used the first terms of the Taylor series decomposition
    :param X:
    :return:
    """
    return x + (0.5 * tf.sin(2*x) * tf.sin(2*x))

def snake3(x):
    """
    Snake activation function

      f(x) = x + sin(x)**2

      The function is computed used the first terms of the Taylor series decomposition
    :param X:
    :return:
    """
    return x + (0.33333333 * tf.sin(3*x) * tf.sin(3*x))

def snake4(x):
    """
    Snake activation function

      f(x) = x + sin(x)**2

      The function is computed used the first terms of the Taylor series decomposition
    :param X:
    :return:
    """
    return x + (0.25 * tf.sin(4*x) * tf.sin(4*x))

def snake5(x):
    """
    Snake activation function

      f(x) = x + sin(x)**2

      The function is computed used the first terms of the Taylor series decomposition
    :param X:
    :return:
    """
    return x + (0.2 * tf.sin(5*x) * tf.sin(5*x))

def xsin(x):
    """
    Snake activation function

      f(x) = x + sin(x)

      The function is computed used the first terms of the Taylor series decomposition
    :param X:
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



class Swish(Activation):
    def __init__(self, activation, **kwargs):
        super(Swish, self).__init__(activation, **kwargs)
        self.__name__ = 'swish'

class Snake(Activation):
    def __init__(self, activation, **kwargs):
        super(Snake, self).__init__(activation, **kwargs)
        self.__name__ = 'snake'

class Snake2(Activation):
    def __init__(self, activation, **kwargs):
        super(Snake2, self).__init__(activation, **kwargs)
        self.__name__ = 'snake2'

class Snake3(Activation):
    def __init__(self, activation, **kwargs):
        super(Snake3, self).__init__(activation, **kwargs)
        self.__name__ = 'snake3'


class Snake4(Activation):
    def __init__(self, activation, **kwargs):
        super(Snake4, self).__init__(activation, **kwargs)
        self.__name__ = 'snake4'


class Snake5(Activation):
    def __init__(self, activation, **kwargs):
        super(Snake5, self).__init__(activation, **kwargs)
        self.__name__ = 'snake5'


class Snakeh2(Activation):
    def __init__(self, activation, **kwargs):
        super(Snakeh2, self).__init__(activation, **kwargs)
        self.__name__ = 'snakeh2'

class Xsin(Activation):
    def __init__(self, activation, **kwargs):
        super(Xsin, self).__init__(activation, **kwargs)
        self.__name__ = 'xsin'


get_custom_objects().update({'swish': Swish(swish)})
get_custom_objects().update({'snake': Snake(snake)})
get_custom_objects().update({'snakeh2': Snakeh2(snakeh2)})
get_custom_objects().update({'snake2': Snake2(snake2)})
get_custom_objects().update({'snake3': Snake3(snake3)})
get_custom_objects().update({'snake4': Snake4(snake4)})
get_custom_objects().update({'snake5': Snake5(snake5)})
get_custom_objects().update({'xsin': Xsin(xsin)})
