"""
.. module:: SCKArchitecture

SCKArchitecture
*************

:Description: SCKArchitecture

 Metaclass for scikit learn classifiers using direct regression

:Authors: bejar
    

:Version: 

:Created on: 04/12/2018 7:46 

"""

from Wind.Architectures.Architecture import Architecture
from sklearn.metrics import r2_score

__author__ = 'bejar'


class SCKArchitecture(Architecture):
    """
    Class for all the scikit models using direct regression

    """
    ## data mode 2 dimensional input and only one output
    data_mode = ('2D', '0D') #'svm'
    modname = 'SCKDIRREG'

    def train(self, train_x, train_y, val_x, val_y):
        """
        Trains the model

        :return:
        """
        self.model.fit(train_x, train_y)

    def evaluate(self, val_x, val_y, test_x, test_y):
        """
        Evaluates the training
        :return:
        """
        val_yp = self.model.predict(val_x)

        r2val = r2_score(val_y, val_yp)

        test_yp = self.model.predict(test_x)
        r2test = r2_score(test_y, test_yp)

        return r2val, r2test
