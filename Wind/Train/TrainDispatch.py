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

from Wind.Train.TrainingProcess import train_dirregression, train_persistence, train_sckit_dirregression, \
    train_sequence2sequence,train_sequence2sequence_tf, train_recursive_multi_sequence2sequence
from Wind.Architectures import RNNDirRegressionArchitecture, SVMDirRegressionArchitecture, \
    PersistenceArchitecture, RNNEncoderDecoderS2SArchitecture, MLPS2SArchitecture, MLPDirRegressionArchitecture, \
    CNNS2SArchitecture, RNNS2SArchitecture, RNNEncoderDecoderS2SAttentionArchitecture, MLPS2SRecursiveArchitecture, \
    RNNEncoderDecoderS2SDepArchitecture, KNNDirRegressionArchitecture, PersistenceMeanArchitecture, \
    CNNS2SCrazyIvanArchitecture

class TrainDispatch:

    model_dict = {}

    def __init__(self):
        """
        Fills the model dictionary with pairs (training algorithm, architecture)

        Keep the old model names for now, but deprecate them in the near future
        """

        self.model_dict['persistence'] = (train_sequence2sequence, PersistenceArchitecture)
        self.model_dict['persistencemean'] = (train_sequence2sequence, PersistenceMeanArchitecture)

        # Scikit learn models
        self.model_dict['KNN_dir_reg'] = (train_sckit_dirregression, KNNDirRegressionArchitecture)

        self.model_dict['SVM_dir_reg'] = self.model_dict['svm'] = (train_sckit_dirregression, SVMDirRegressionArchitecture)

        # RNN models

        self.model_dict['RNN_dir_reg'] = self.model_dict['regdir'] = (train_dirregression, RNNDirRegressionArchitecture)

        self.model_dict['RNN_ED_s2s'] = self.model_dict['seq2seq'] = (train_sequence2sequence, RNNEncoderDecoderS2SArchitecture)

        # self.model_dict['RNN_ED_s2s'] = (train_sequence2sequence, RNNEncoderDecoderS2SArchitecture)

        self.model_dict['RNN_ED_s2s_dep'] = (train_sequence2sequence, RNNEncoderDecoderS2SDepArchitecture)

        self.model_dict['RNN_ED_s2s_att'] = (train_sequence2sequence_tf, RNNEncoderDecoderS2SAttentionArchitecture)

        self.model_dict['RNN_s2s'] = (train_sequence2sequence, RNNS2SArchitecture)

        # MLP models

        self.model_dict['MLP_s2s'] = self.model_dict['mlps2s'] = (train_sequence2sequence, MLPS2SArchitecture)

        self.model_dict['MLP_dir_reg'] = self.model_dict['mlpdir'] = (train_dirregression, MLPDirRegressionArchitecture)

        self.model_dict['MLP_s2s_rec'] = (train_recursive_multi_sequence2sequence, MLPS2SRecursiveArchitecture)

        # Convolutional models

        self.model_dict['CNN_s2s'] = self.model_dict['convos2s'] = (train_sequence2sequence, CNNS2SArchitecture)
        self.model_dict['CNN_CI_s2s'] =(train_sequence2sequence, CNNS2SCrazyIvanArchitecture)



    def dispatch(self, mode):
        """
        Returns the corresponding (training algorithm, architecture)
        :param mode:
        :return:
        """
        if mode in self.model_dict:
            return self.model_dict[mode]
        else:
            raise NameError('ERROR: No such mode')