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

from .PersistenceArchitecture import PersistenceArchitecture
from .PersistenceMeanArchitecture import PersistenceMeanArchitecture

from .SVMDirRegressionArchitecture import SVMDirRegressionArchitecture
from .KNNDirRegressionArchitecture import KNNDirRegressionArchitecture
from .RandomForestDirRegressionArchitecture import RandomForestDirRegressionArchitecture
from .AdaBoostDirRegressionArchitecture import AdaBoostDirRegressionArchitecture

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
           'NNArchitecture',
           'NNS2SArchitecture',
           'SCKArchitecture',
           'CNNS2SCrazyIvanArchitecture',
           'CNNS2S2DArchitecture',
           'CNNS2SSkipArchitecture',
           'RandomForestDirRegressionArchitecture',
           'AdaBoostDirRegressionArchitecture'
           ]