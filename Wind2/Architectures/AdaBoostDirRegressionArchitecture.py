"""
.. module:: AdaBoostDirRegressionArchitecture

RandomForestDirRegressionArchitecture
*************

:Description: AdaBoostDirRegressionArchitecture

    

:Authors: bejar
    

:Version: 

:Created on: 29/05/2019 12:21 

"""


from Wind2.Architectures.SCKArchitecture import SCKArchitecture
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

__author__ = 'bejar'

class AdaBoostDirRegressionArchitecture(SCKArchitecture):
    """Direct regression architecture based on random forest
    """
    data_mode = ('2D', '0D')  #
    modname = 'ABDIRREG'

    def generate_model(self):
        """
        Generates the model

        -------------
        json config:

        "arch": {
            "max_depth" : depth of the ecisionTreeRegressor
            "n_estimators" : number of models
            "learning_rate": Learning rate shrinks the contribution of each regressor
            "loss": The loss function to use when updating the weights
            "mode": "AB_dir_reg"
        }

        The rest of the parameters are the defaults of scikit-learn
        -------------

        :return:
        """
        self.model = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=self.config['arch']['max_depth']),
                                       n_estimators=self.config['arch']['n_estimators'],
                                         learning_rate=self.config['arch']['learning_rate'],
                                         loss=self.config['arch']['loss'])
