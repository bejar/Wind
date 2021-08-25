"""
.. module:: __init__.py

__init__.py
*************

:Description: __init__.py

    

:Authors: bejar
    

:Version: 

:Created on: 06/07/2018 10:53 

"""

from Wind.Architectures.NNArchitecture import NNArchitecture
from .AdaBoostDirRegressionArchitecture import AdaBoostDirRegressionArchitecture
from .Architecture import Architecture
from .CNN2LS2SArchitecture import CNN2LS2SArchitecture
from .CNN3LS2SArchitecture import CNN3LS2SArchitecture
from .CNN4LS2SArchitecture import CNN4LS2SArchitecture
from .CNNLoCoS2SArchitecture import CNNLoCoS2SArchitecture
from .CNNMIMOResidualArchitecture import CNNMIMOResidualArchitecture
from .CNNMIMOSkipArchitecture import CNNMIMOSkipArchitecture
from .CNNS2S2DArchitecture import CNNS2S2DArchitecture
from .CNNS2SArchitecture import CNNS2SArchitecture
from .CNNS2SCrazyIvan2HArchitecture import CNNS2SCrazyIvan2HArchitecture
from .CNNS2SCrazyIvan3HArchitecture import CNNS2SCrazyIvan3HArchitecture
from .CNNS2SCrazyIvanArchitecture import CNNS2SCrazyIvanArchitecture
from .CNNS2SSkipArchitecture import CNNS2SSkipArchitecture
from .CNNSeparable2LS2SArchitecture import CNNSeparable2LS2SArchitecture
from .CNNSeparable3LS2SArchitecture import CNNSeparable3LS2SArchitecture
from .CNNSeparable4LS2SArchitecture import CNNSeparable4LS2SArchitecture
from .CNNSeparableS2SArchitecture import CNNSeparableS2SArchitecture
from .CNNSeparableS2SSeparateArchitecture import CNNSeparableS2SSeparateArchitecture
from .CNNSeparable2LS2SSeparateArchitecture import CNNSeparable2LS2SSeparateArchitecture
from .CNNSeparableS2SSeparateBArchitecture import CNNSeparableS2SSeparateBArchitecture
from .ElasticNetS2SArchitecture import ElasticNetS2SArchitecture
from .ExtraTreesS2SArchitecture import ExtraTreesS2SArchitecture
from .KNNDirRegressionArchitecture import KNNDirRegressionArchitecture
from .KNNS2SArchitecture import KNNS2SArchitecture
from .MLPCascadeS2SArchitecture import MLPCascadeS2SArchitecture
from .MLPDirRegressionArchitecture import MLPDirRegressionArchitecture
from .MLPS2SArchitecture import MLPS2SArchitecture
from .MLPS2SFutureArchitecture import MLPS2SFutureArchitecture
from .MLPS2SSeparateArchitecture import MLPS2SSeparateArchitecture
from .MLPS2SRecursiveArchitecture import MLPS2SRecursiveArchitecture
from .NNS2SArchitecture import NNS2SArchitecture
from .PersistenceArchitecture import PersistenceArchitecture
from .PersistenceMeanArchitecture import PersistenceMeanArchitecture
from .RNNDirRegressionArchitecture import RNNDirRegressionArchitecture
#from .RNNEncoderDecoderAttentionArchitecture import RNNEncoderDecoderAttentionArchitecture
from .RNNEncoderDecoderS2SArchitecture import RNNEncoderDecoderS2SArchitecture
from .RNNEncoderDecoderS2SAttentionArchitecture import RNNEncoderDecoderS2SAttentionArchitecture
from .RNNEncoderDecoderS2SDepArchitecture import RNNEncoderDecoderS2SDepArchitecture
from .RNNS2SArchitecture import RNNS2SArchitecture
from .RNNS2SSelfAttentionArchitecture import RNNS2SSelfAttentionArchitecture
from .RandomForestDirRegressionArchitecture import RandomForestDirRegressionArchitecture
from .RandomForestS2SArchitecture import RandomForestS2SArchitecture
from .SCKArchitecture import SCKArchitecture
from .SCKS2SArchitecture import SCKS2SArchitecture
from .SVMDirRegressionArchitecture import SVMDirRegressionArchitecture
from .TimeInceptionArchitecture import TimeInceptionArchitecture
from .XGBoostDirRegressionArchitecture import XGBoostDirRegressionArchitecture
from .TCNArchitecture import TCNArchitecture
from .NBeatsArchitecture import NBeatsArchitecture
from .NaivePeriodicArchitecture import NaivePeriodicArchitecture

__author__ = 'bejar'

__all__ = ['Architecture',
           'RNNDirRegressionArchitecture',
           'SVMDirRegressionArchitecture',
           'RNNEncoderDecoderS2SArchitecture',
           'PersistenceArchitecture',
           'PersistenceMeanArchitecture',
           'MLPS2SArchitecture',
           'MLPCascadeS2SArchitecture',
           'MLPS2SFutureArchitecture',
           'MLPS2SSeparateArchitecture',
           'MLPDirRegressionArchitecture',
           'CNNS2SArchitecture',
           'CNN2LS2SArchitecture',
           'CNN3LS2SArchitecture',
           'CNN4LS2SArchitecture',
           'CNNLoCoS2SArchitecture',
           'RNNS2SArchitecture',
           'RNNS2SSelfAttentionArchitecture',
           'RNNEncoderDecoderS2SAttentionArchitecture',
#           'RNNEncoderDecoderAttentionArchitecture',
           'MLPS2SRecursiveArchitecture',
           'RNNEncoderDecoderS2SDepArchitecture',
           'KNNDirRegressionArchitecture',
           'KNNS2SArchitecture',
           'NNArchitecture',
           'NNS2SArchitecture',
           'SCKArchitecture',
           'SCKS2SArchitecture',
           'CNNS2SCrazyIvanArchitecture',
           'CNNS2SCrazyIvan2HArchitecture',
           'CNNS2SCrazyIvan3HArchitecture',
           'CNNS2S2DArchitecture',
           'CNNSeparableS2SArchitecture',
           'CNNSeparableS2SSeparateArchitecture',
           'CNNSeparable2LS2SSeparateArchitecture',
           'CNNSeparableS2SSeparateBArchitecture',
           'CNNSeparable2LS2SArchitecture',
           'CNNSeparable3LS2SArchitecture',
           'CNNSeparable4LS2SArchitecture',
           'CNNS2SSkipArchitecture',
           'CNNMIMOSkipArchitecture',
           'CNNMIMOResidualArchitecture',
           'RandomForestDirRegressionArchitecture',
           'RandomForestS2SArchitecture',
           'AdaBoostDirRegressionArchitecture',
           'ExtraTreesS2SArchitecture',
           'ElasticNetS2SArchitecture',
           'TimeInceptionArchitecture',
           'XGBoostDirRegressionArchitecture',
           'TCNArchitecture',
           'NBeatsArchitecture',
           'NaivePeriodicArchitecture'
           ]
