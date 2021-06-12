"""
.. module:: ElasticNetS2SArchitecture

KNNDirRegressionArchitecture
*************

:Description: ElasticNetS2SArchitecture

    Multi output ElasticNet

:Authors: bejar
    

:Version: 

:Created on: 04/12/2018 7:12 

"""

from Wind.Architectures.SCKS2SArchitecture import SCKS2SArchitecture
from sklearn.linear_model import ElasticNet

__author__ = 'bejar'

class ElasticNetS2SArchitecture(SCKS2SArchitecture):
    """S2S regression architecture based on ElasticNet
    """
    data_mode = ('2D', '2D')  #
    modname = 'ENS2S'

    def generate_model(self):
        """
        Generates the model

        -------------
        json config:

        "arch": {
            "alpha" : penalization
            "l1_ratio": L1 to L2 ratio
            "mode": "ENet_s2s"
        }

        The rest of the parameters are the defaults of scikit-learn
        -------------

        :return:
        """
        self.model = ElasticNet(alpha=self.config['arch']['alpha'],
                                         l1_ratio=self.config['arch']['l1_ratio'],
                                max_iter=10000)



