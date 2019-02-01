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

from .RNNDirRegressionArchitecture import RNNDirRegressionArchitecture
from .RNNEncoderDecoderS2SArchitecture import RNNEncoderDecoderS2SArchitecture
from .RNNS2SArchitecture import RNNS2SArchitecture
from .RNNEncoderDecoderS2SAttentionArchitecture import RNNEncoderDecoderS2SAttentionArchitecture
from .RNNEncoderDecoderS2SDepArchitecture import RNNEncoderDecoderS2SDepArchitecture

from .MLPS2SArchitecture import MLPS2SArchitecture
from .MLPDirRegressionArchitecture import MLPDirRegressionArchitecture
from .MLPS2SRecursiveArchitecture import MLPS2SRecursiveArchitecture

from .CNNS2SArchitecture import CNNS2SArchitecture

__author__ = 'bejar'

__all__ = ['Architecture',
           'RNNDirRegressionArchitecture',
           'SVMDirRegressionArchitecture',
           'RNNEncoderDecoderS2SArchitecture',
           'PersistenceArchitecture',
           'PersistenceMeanArchitecture',
           'PersistenceMeanArchitecture2',
           'MLPS2SArchitecture',
           'MLPDirRegressionArchitecture',
           'CNNS2SArchitecture',
           'RNNS2SArchitecture',
           'RNNEncoderDecoderS2SAttentionArchitecture',
           'MLPS2SRecursiveArchitecture',
           'RNNEncoderDecoderS2SDepArchitecture',
           'KNNDirRegressionArchitecture',
           'NNArchitecture',
           'NNS2SArchitecture',
           'SCKArchitecture'
           ]