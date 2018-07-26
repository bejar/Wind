"""
.. module:: __init__.py

__init__.py
*************

:Description: __init__.py

    

:Authors: bejar
    

:Version: 

:Created on: 06/07/2018 7:53 

"""

from .TrainDispatch import TrainDispatch
from .TrainingProcess import train_dirregression, train_persistence, train_sequence2sequence, train_svm_dirregression
from .RunConfig import RunConfig

__author__ = 'bejar'


__all__ = ['TrainDispatch',
           'RunConfig',
           'train_persistence', 'train_dirregression', 'train_sequence2sequence', 'train_svm_dirregression']