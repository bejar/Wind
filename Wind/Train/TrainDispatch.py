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

from Wind.Deprecated.Train.TrainingProcess import train_dirregression, train_persistence, train_svm_dirregression, \
    train_sequence2sequence
from Wind.Architectures import RNNDirRegressionArchitecture, SVMDirRegressionArchitecture, \
    PersistenceArchitecture, RNNEncoderDecoderS2SArchitecture, MLPS2SArchitecture, MLPDirRegressionArchitecture, \
    CNNS2SArchitecture

class TrainDispatch:

    model_dict = {}

    def __init__(self):
        self.model_dict['RNN_dir_reg'] = (train_dirregression, RNNDirRegressionArchitecture)
        self.model_dict['regdir'] = (train_dirregression, RNNDirRegressionArchitecture)

        self.model_dict['SVM_dir_reg'] = (train_svm_dirregression, SVMDirRegressionArchitecture)
        self.model_dict['svm'] = (train_svm_dirregression, SVMDirRegressionArchitecture)

        self.model_dict['RNN_ED_s2s'] = (train_sequence2sequence, RNNEncoderDecoderS2SArchitecture)
        self.model_dict['seq2seq'] = (train_sequence2sequence, RNNEncoderDecoderS2SArchitecture)

        self.model_dict['MLP_s2s'] = (train_sequence2sequence, MLPS2SArchitecture)
        self.model_dict['mlps2s'] = (train_sequence2sequence, MLPS2SArchitecture)

        self.model_dict['MLP_dir_reg'] = (train_dirregression, MLPDirRegressionArchitecture)
        self.model_dict['mlpdir'] = (train_dirregression, MLPDirRegressionArchitecture)

        self.model_dict['persistence'] = (train_persistence, PersistenceArchitecture)

        self.model_dict['convos2s'] = (train_sequence2sequence, CNNS2SArchitecture)
        self.model_dict['CNN_s2s'] = (train_sequence2sequence, CNNS2SArchitecture)


    def dispatch(self, mode):
        if mode in self.model_dict:
            return self.model_dict[mode]
        else:
            raise NameError('ERROR: No such mode')