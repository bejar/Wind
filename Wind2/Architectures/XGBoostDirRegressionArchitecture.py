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
    modname = 'XGBDIRREG'

    def generate_model(self):
        """
        Generates the model

        -------------
        json config:

        "arch": {
            "max_depth": max depth of the estimators
            "n_estimators" : number of models
            "learning_rate": Learning rate shrinks the contribution of each regressor
            "lambda": L2 regularization
            "alpha": L1 regularization
            "mode": "XGB_dir_reg"
        }

        The rest of the parameters are the defaults of xgboost
        -------------

        :return:
        """
        self.model = xgb.XGBRegressor(n_estimators=self.config['arch']['n_estimators'],
                                      learning_rate=self.config['arch']['learning_rate'],
                                      max_depth=self.config['arch']['max_depth'],
                                      objective='reg:squarederror',
                                      reg_lambda=self.config['arch']['lambda'],
                                      alpha=self.config['arch']['alpha'],
                                      n_jobs=4)
