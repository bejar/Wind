"""
.. module:: __init__.py

__init__.py
*************

:Description: __init__.py

    

:Authors: bejar
    

:Version: 

:Created on: 06/04/2018 14:15 

"""

from .RNNRegressionDir import architectureDirRegression, train_dirregression_architecture
from .MLPRegressionS2S import architectureMLPs2s, train_MLP_regs2s_architecture
from .MLPRegressionDir import architectureMLP_dirreg, train_MLP_dirreg_architecture
from .RNNRegressionS2S import architectureS2S, train_seq2seq_architecture
from .Ensemble import train_ensemble_architecture
from .CNNRegressionDir import train_convdirregression_architecture, architectureConvDirRegression
from .SVMRegressionDir import train_svm_dirregression_architecture
from .CNNRegressionS2S import train_convo_regs2s_architecture, architectureConvos2s
from .Persistence import train_persistence
from .Cascade.CascadeRNNRegressionS2S import train_cascade_seq2seq_architecture

__author__ = 'bejar'

__all__ = ['architectureDirRegression', 'train_dirregression_architecture',
           'architectureMLPs2s', 'train_MLP_regs2s_architecture',
           'architectureS2S', 'train_seq2seq_architecture',
           'train_ensemble_architecture',
           'train_convdirregression_architecture', 'architectureConvDirRegression',
           'architectureMLP_dirreg', 'train_MLP_dirreg_architecture',
           'train_svm_dirregression_architecture',
           'train_convo_regs2s_architecture', 'architectureConvos2s',
           'train_persistence',
           'train_cascade_seq2seq_architecture']