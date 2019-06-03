"""
.. module:: XGBoostDirRegressionArchitecture

RandomForestDirRegressionArchitecture
*************

:Description: XGBoostDirRegressionArchitecture

    

:Authors: bejar
    

:Version: 

:Created on: 29/05/2019 12:21 

"""


from Wind.Architectures.SCKArchitecture import SCKArchitecture
import xgboost as xgb

__author__ = 'bejar'

class XGBoostDirRegressionArchitecture(SCKArchitecture):
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
            "n_estimators" : number of models
            "learning_rate": Learning rate shrinks the contribution of each regressor
            "mode": "AB_dir_reg"
        }

        The rest of the parameters are the defaults of xgboost
        -------------

        :return:
        """
        self.model = xgb.XGBRegressor(n_estimators=self.config['arch']['n_estimators'],
                                      learning_rate=self.config['arch']['learning_rate'],
                                      objective='reg:squarederror',
                                      njobs=-1)
