"""
.. module:: __init__.py

__init__.py
*************

:Description: __init__.py

    

:Authors: bejar
    

:Version: 

:Created on: 06/04/2018 14:15 

"""

from .DirectRegression import architectureDirRegression, train_dirregression_architecture
from .MLPRegression import architectureMLP, train_MLP_regdir_architecture
from .Seq2SeqRegression import architectureS2S, train_seq2seq_architecture
from .Ensemble import train_ensemble_architecture
from .ConvoRegression import train_convdirregression_architecture, architectureConvDirRegression
__author__ = 'bejar'

__all__ = ['architectureDirRegression', 'train_dirregression_architecture',
           'architectureMLP', 'train_MLP_regdir_architecture',
           'architectureS2S', 'train_seq2seq_architecture',
           'train_ensemble_architecture',
           'train_convdirregression_architecture', 'architectureConvDirRegression']