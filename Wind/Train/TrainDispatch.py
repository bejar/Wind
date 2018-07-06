"""
.. module:: Dispatch

Dispatch
*************

:Description: Dispatch

  Selects the model and the function to run it

:Authors: bejar
    

:Version: 

:Created on: 06/07/2018 7:47 

"""

__author__ = 'bejar'

from Wind.Train.TrainingProcess import train_dirregression, train_persistence
from Wind.Models import RNNDirRegressionArchitecture

class TrainDispatch:

    model_dict = {}

    def __init__(self):
        self.model_dict['RNN_dir_reg'] = (train_dirregression, RNNDirRegressionArchitecture)
        self.model_dict['regdir'] = (train_dirregression, RNNDirRegressionArchitecture)
        self.model_dict['persistence'] = (train_persistence, None)

    def dispatch(self, mode):
        print(mode)
        if mode in self.model_dict:
            return self.model_dict[mode]
        else:
            raise NameError('ERROR: No such mode')