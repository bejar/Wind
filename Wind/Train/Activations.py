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

__author__ = 'bejar'


def generate_activation(act_par):
    """
    Uses the value in the configuration to generate the activation layer

    :param act_par:
    :return:
    """

    if type(act_par) == list:
        atype, par = act_par
        if atype == 'elu':
            return ELU(alpha=par)
        elif atype == ' leaky':
            return LeakyReLU(alpha=0.3)
        elif atype == 'prelu':
            return PReLU()
    elif type(act_par) == str:
        return Activation(act_par)
    else:
        raise NameError("Wrong parameter for activation layer")

