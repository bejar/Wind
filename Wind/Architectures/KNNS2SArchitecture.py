"""
.. module:: KNNS2SArchitecture

KNNS2SArchitecture
*************

:Description: KNNS2SArchitecture

    S2S architecture based on K-nearest neigbours

:Authors: bejar
    

:Version: 

:Created on: 04/12/2018 7:12 

"""

from Wind.Architectures.SCKS2SArchitecture import SCKS2SArchitecture
from sklearn.neighbors import KNeighborsRegressor

__author__ = 'bejar'



class KNNS2SArchitecture(SCKS2SArchitecture):
    """S2S regression architecture based on K-nearest neigbours
    """
    data_mode = ('2D', '2D')  #
    modname = 'KNNS2S'

    def generate_model(self):
        """
        Generates the model

        -------------
        json config:

        "arch": {
            "n_neighbors" : number of neighbors,
            "weights": weights applied to the neighbors, values in ["distance", "uniform"],
            "mode": "KNN_s2s_reg"
        }

        The rest of the parameters are the defaults of scikit-learn
        -------------

        :return:
        """
        self.model = KNeighborsRegressor(n_neighbors=self.config['arch']['n_neighbors'],
                                         weights=self.config['arch']['weights'])


