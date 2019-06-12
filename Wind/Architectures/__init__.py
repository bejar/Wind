"""
.. module:: __init__.py

__init__.py
*************

:Description: __init__.py

    

:Authors: bejar
    

:Version: 

:Created on: 06/07/2018 10:53 

"""

from .Architecture import Architecture
from Wind.Architectures.NNArchitecture import NNArchitecture
from .NNS2SArchitecture import NNS2SArchitecture
from .SCKArchitecture import SCKArchitecture
from .SCKS2SArchitecture import SCKS2SArchitecture

from .PersistenceArchitecture import PersistenceArchitecture
from .PersistenceMeanArchitecture import PersistenceMeanArchitecture

from .SVMDirRegressionArchitecture import SVMDirRegressionArchitecture
from .KNNDirRegressionArchitecture import KNNDirRegressionArchitecture
from .KNNS2SArchitecture import KNNS2SArchitecture
from .RandomForestDirRegressionArchitecture import RandomForestDirRegressionArchitecture
from .RandomForestS2SArchitecture import RandomForestS2SArchitecture
from .AdaBoostDirRegressionArchitecture import AdaBoostDirRegressionArchitecture
from .AdaBoostS2SArchitecture import AdaBoostS2SArchitecture
from .XGBoostDirRegressionArchitecture import XGBoostDirRegressionArchitecture
from .ElasticNetS2SArchitecture import ElasticNetS2SArchitecture

from .RNNDirRegressionArchitecture import RNNDirRegressionArchitecture
from .RNNEncoderDecoderS2SArchitecture import RNNEncoderDecoderS2SArchitecture
from .RNNS2SArchitecture import RNNS2SArchitecture
from .RNNS2SSelfAttentionArchitecture import RNNS2SSelfAttentionArchitecture
from .RNNEncoderDecoderS2SAttentionArchitecture import RNNEncoderDecoderS2SAttentionArchitecture
from .RNNEncoderDecoderS2SDepArchitecture import RNNEncoderDecoderS2SDepArchitecture

from .MLPS2SArchitecture import MLPS2SArchitecture
from .MLPCascadeS2SArchitecture import MLPCascadeS2SArchitecture
from .MLPDirRegressionArchitecture import MLPDirRegressionArchitecture
from .MLPS2SRecursiveArchitecture import MLPS2SRecursiveArchitecture
from .MLPS2SFutureArchitecture import MLPS2SFutureArchitecture

from .CNNS2SArchitecture import CNNS2SArchitecture
from .CNNS2SSkipArchitecture import CNNS2SSkipArchitecture
from .CNNS2SCrazyIvanArchitecture import CNNS2SCrazyIvanArchitecture
from .CNNS2S2DArchitecture import CNNS2S2DArchitecture

__author__ = 'bejar'

__all__ = ['Architecture',
           'RNNDirRegressionArchitecture',
           'SVMDirRegressionArchitecture',
           'RNNEncoderDecoderS2SArchitecture',
           'PersistenceArchitecture',
           'PersistenceMeanArchitecture',
           'PersistenceMeanArchitecture2',
           'MLPS2SArchitecture',
           'MLPCascadeS2SArchitecture',
           'MLPS2SFutureArchitecture',
           'MLPDirRegressionArchitecture',
           'CNNS2SArchitecture',
           'RNNS2SArchitecture',
           'RNNS2SSelfAttentionArchitecture',
           'RNNEncoderDecoderS2SAttentionArchitecture',
           'MLPS2SRecursiveArchitecture',
           'RNNEncoderDecoderS2SDepArchitecture',
           'KNNDirRegressionArchitecture',
           'KNNS2SArchitecture',
           'NNArchitecture',
           'NNS2SArchitecture',
           'SCKArchitecture',
           'SCKS2SArchitecture',
           'CNNS2SCrazyIvanArchitecture',
           'CNNS2S2DArchitecture',
           'CNNS2SSkipArchitecture',
           'RandomForestDirRegressionArchitecture',
           'RandomForestS2SArchitecture',
           'AdaBoostDirRegressionArchitecture',
           'AdaBoostS2SArchitecture',
           'ElasticNetS2SArchitecture'
          ]