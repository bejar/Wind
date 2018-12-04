"""
.. module:: __init__.py

__init__.py
*************

:Description: __init__.py

    

:Authors: bejar
    

:Version: 

:Created on: 06/07/2018 10:53 

"""

from .RNNDirRegressionArchitecture import RNNDirRegressionArchitecture
from .SVMDirRegressionArchitecture import SVMDirRegressionArchitecture
from .RNNEncoderDecoderS2SArchitecture import RNNEncoderDecoderS2SArchitecture
from .PersistenceArchitecture import PersistenceArchitecture
from .MLPS2SArchitecture import MLPS2SArchitecture
from .MLPDirRegressionArchitecture import MLPDirRegressionArchitecture
from .CNNS2SArchitecture import CNNS2SArchitecture
from .RNNS2SArchitecture import RNNS2SArchitecture
from .RNNEncoderDecoderS2SAttentionArchitecture import RNNEncoderDecoderS2SAttentionArchitecture
from .MLPS2SRecursiveArchirecture import MLPS2SRecursiveArchitecture
from .RNNEncoderDecoderS2SDepArchitecture import RNNEncoderDecoderS2SDepArchitecture
from .KNNDirRegressionArchitecture import KNNDirRegressionArchitecture

__author__ = 'bejar'

__all__ = [
            'RNNDirRegressionArchitecture',
            'SVMDirRegressionArchitecture',
            'RNNEncoderDecoderS2SArchitecture',
            'PersistenceArchitecture',
            'MLPS2SArchitecture',
            'MLPDirRegressionArchitecture',
            'CNNS2SArchitecture',
            'RNNS2SArchitecture',
            'RNNEncoderDecoderS2SAttentionArchitecture',
            'MLPS2SRecursiveArchirecture',
            'RNNEncoderDecoderS2SDepArchitecture',
            'KNNDirRegressionArchitecture'
            ]