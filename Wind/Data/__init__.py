"""
.. module:: __init__.py

__init__.py
*************

:Description: __init__.py

    

:Authors: bejar
    

:Version: 

:Created on: 19/02/2018 9:28 

"""

__author__ = 'bejar'

from .Data import lagged_matrix, lagged_vector, generate_dataset

__all__ = ['lagged_vector', 'lagged_matrix', 'generate_dataset']