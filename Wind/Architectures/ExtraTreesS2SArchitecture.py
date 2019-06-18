"""
.. module:: AdaBoostDirRegressionArchitecture

RandomForestDirRegressionArchitecture
*************

:Description: AdaBoostDirRegressionArchitecture

    

:Authors: bejar
    

:Version: 

:Created on: 29/05/2019 12:21 

"""


from Wind.Architectures.SCKS2SArchitecture import SCKS2SArchitecture
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import ExtraTreeRegressor

__author__ = 'bejar'

class ExtraTreesS2SArchitecture(SCKS2SArchitecture):
    """Direct regression architecture based on random forest
    """
    data_mode = ('2D', '2D')  #
    modname = 'ETS2S'

    def generate_model(self):
        """
        Generates the model

        -------------
        json config:

        "arch": {
            "max_depth" : depth of the ExtraTreesRegressor
            "n_estimators" : number of trees
            "max_features": number of features to consider when looking for the best split
            "mode": "ET_dir_reg"

        }

        The rest of the parameters are the defaults of scikit-learn
        -------------

        :return:
        """
        self.model = ExtraTreesRegressor(max_depth=self.config['arch']['max_depth'],
                                         n_estimators=self.config['arch']['n_estimators'],
                                         max_features=self.config['arch']['max_features'],n_jobs=-1)
