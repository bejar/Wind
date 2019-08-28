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
    train_sequence2sequence,train_sequence2sequence_tf, train_recursive_multi_sequence2sequence, \
    train_sckit_sequence2sequence, train_sjoint_sequence2sequence

from Wind.Architectures import PersistenceArchitecture, PersistenceMeanArchitecture

from Wind.Architectures import MLPDirRegressionArchitecture, MLPS2SArchitecture, MLPS2SFutureArchitecture, MLPS2SRecursiveArchitecture, \
    MLPCascadeS2SArchitecture

from Wind.Architectures import CNNS2SArchitecture, CNNS2SCrazyIvanArchitecture, CNNS2SCrazyIvan2HArchitecture, CNNS2S2DArchitecture,  \
    CNNS2SSkipArchitecture, CNNSeparableS2SArchitecture, CNNSeparable2LS2SArchitecture, CNNSeparable3LS2SArchitecture,\
    CNNSeparable4LS2SArchitecture, CNN2LS2SArchitecture, CNN3LS2SArchitecture, CNN4LS2SArchitecture

from Wind.Architectures import RNNDirRegressionArchitecture, RNNEncoderDecoderS2SArchitecture, RNNS2SArchitecture, \
    RNNEncoderDecoderS2SAttentionArchitecture, RNNS2SSelfAttentionArchitecture, RNNEncoderDecoderS2SDepArchitecture

from Wind.Architectures import KNNDirRegressionArchitecture, RandomForestDirRegressionArchitecture, SVMDirRegressionArchitecture, \
    AdaBoostDirRegressionArchitecture, KNNS2SArchitecture, ElasticNetS2SArchitecture, XGBoostDirRegressionArchitecture, \
    RandomForestS2SArchitecture, ExtraTreesS2SArchitecture



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
        self.model_dict['KNN_s2s'] = (train_sckit_sequence2sequence, KNNS2SArchitecture)

        self.model_dict['SVM_dir_reg'] = self.model_dict['svm'] = (train_sckit_dirregression, SVMDirRegressionArchitecture)
        self.model_dict['RF_dir_reg'] =  (train_sckit_dirregression, RandomForestDirRegressionArchitecture)
        self.model_dict['RF_s2s'] =  (train_sckit_sequence2sequence, RandomForestS2SArchitecture)
        self.model_dict['AB_dir_reg'] =  (train_sckit_dirregression, AdaBoostDirRegressionArchitecture)
        self.model_dict['ET_s2s'] =  (train_sckit_sequence2sequence, ExtraTreesS2SArchitecture)
        self.model_dict['XGB_dir_reg'] =  (train_sckit_dirregression, XGBoostDirRegressionArchitecture)
        self.model_dict['ENet_s2s'] =  (train_sckit_sequence2sequence, ElasticNetS2SArchitecture)

        # RNN models

        self.model_dict['RNN_dir_reg'] = self.model_dict['regdir'] = (train_dirregression, RNNDirRegressionArchitecture)
        self.model_dict['RNN_ED_s2s'] = self.model_dict['seq2seq'] = (train_sequence2sequence, RNNEncoderDecoderS2SArchitecture)

        # self.model_dict['RNN_ED_s2s'] = (train_sequence2sequence, RNNEncoderDecoderS2SArchitecture)

        self.model_dict['RNN_ED_s2s_dep'] = (train_sequence2sequence, RNNEncoderDecoderS2SDepArchitecture)

        self.model_dict['RNN_ED_s2s_att'] = (train_sequence2sequence_tf, RNNEncoderDecoderS2SAttentionArchitecture)

        self.model_dict['RNN_s2s'] = (train_sequence2sequence, RNNS2SArchitecture)
        self.model_dict['RNN_s2s_att'] = (train_sequence2sequence, RNNS2SSelfAttentionArchitecture)

        # MLP models

        self.model_dict['MLP_s2s'] = self.model_dict['mlps2s'] = (train_sequence2sequence, MLPS2SArchitecture)
        self.model_dict['MLP_s2s_cas'] =  (train_sequence2sequence, MLPCascadeS2SArchitecture)
        self.model_dict['MLP_s2s_fut'] = (train_sequence2sequence, MLPS2SFutureArchitecture)

        self.model_dict['MLP_dir_reg'] = self.model_dict['mlpdir'] = (train_dirregression, MLPDirRegressionArchitecture)

        self.model_dict['MLP_s2s_rec'] = (train_recursive_multi_sequence2sequence, MLPS2SRecursiveArchitecture)
        self.model_dict['MLP_s2s_sjoint'] = (train_sjoint_sequence2sequence, MLPS2SArchitecture)

        # Convolutional models

        self.model_dict['CNN_s2s'] = self.model_dict['convos2s'] = (train_sequence2sequence, CNNS2SArchitecture)
        self.model_dict['CNN_2l_s2s'] = (train_sequence2sequence, CNN2LS2SArchitecture)
        self.model_dict['CNN_3l_s2s'] = (train_sequence2sequence, CNN3LS2SArchitecture)
        self.model_dict['CNN_4l_s2s'] = (train_sequence2sequence, CNN4LS2SArchitecture)
        self.model_dict['CNN_2D_s2s'] = (train_sequence2sequence, CNNS2S2DArchitecture)
        self.model_dict['CNN_Skip_s2s'] = (train_sequence2sequence, CNNS2SSkipArchitecture)
        self.model_dict['CNN_sep_s2s'] = (train_sequence2sequence, CNNSeparableS2SArchitecture)
        self.model_dict['CNN_sep_2l_s2s'] = (train_sequence2sequence, CNNSeparable2LS2SArchitecture)
        self.model_dict['CNN_sep_3l_s2s'] = (train_sequence2sequence, CNNSeparable3LS2SArchitecture)
        self.model_dict['CNN_sep_4l_s2s'] = (train_sequence2sequence, CNNSeparable4LS2SArchitecture)

        self.model_dict['CNN_CI_s2s'] =(train_sequence2sequence, CNNS2SCrazyIvanArchitecture)
        self.model_dict['CNN_CI_2H_s2s'] =(train_sequence2sequence, CNNS2SCrazyIvan2HArchitecture)

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