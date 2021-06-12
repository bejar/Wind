"""
.. module:: SVMDirRegressionArchitecture

SVMDirRegressionArchitecture
*************

:Description: SVMDirRegressionArchitecture

    SVM direct regression architecture

:Authors: bejar
    

:Version: 

:Created on: 06/07/2018 8:33 

"""

from sklearn.svm import SVR

from Wind.Architectures.SCKArchitecture import SCKArchitecture

__author__ = 'bejar'


class SVMDirRegressionArchitecture(SCKArchitecture):
    """SVM direct regression architecture

    """
    data_mode = ('2D', '0D') #'svm'
    modname = 'SVMDIRREG'

    def generate_model(self):
        """
        Generates the model
         -------------
        json config:

        "arch": {
            "kernel" : "rbf",
            "C" : 0.01,
            "epsilon": 0.01,
            "degree": 2,
            "coef0": 1,
            "mode": "SVM_dir_reg"
        }

        The rest of the parameters are the defaults of scikit-learn
        -------------
        :return:
        """
        kernel = self.config['arch']['kernel']
        C = self.config['arch']['C']
        epsilon = self.config['arch']['epsilon']
        degree = self.config['arch']['degree']
        coef0 = self.config['arch']['coef0']

        self.model = SVR(kernel=kernel, C=C, epsilon=epsilon, degree=degree, coef0=coef0)

