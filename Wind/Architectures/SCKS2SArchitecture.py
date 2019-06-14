"""
.. module:: NNS2SArchitecture

NNS2SArchitecture
*************

:Description: SCKS2SArchitecture

    

:Authors: bejar
    

:Version: 

:Created on: 19/10/2018 10:32 

"""

from sklearn.metrics import r2_score, mean_squared_error
from keras.models import load_model

from Wind.Architectures.SCKArchitecture import SCKArchitecture
from Wind.ErrorMeasure import ErrorMeasure

__author__ = 'bejar'


class SCKS2SArchitecture(SCKArchitecture):
    """
    Class for all the neural networks models based on sequence to sequence

    """
    def evaluate(self, val_x, val_y, test_x, test_y):
        """
        Evaluates the trained model with validation and test

        Overrides parent function

        :param val_x:
        :param val_y:
        :param test_x:
        :param test_y:
        :return:
        """

        val_yp = self.model.predict(val_x)
        test_yp = self.model.predict(test_x)

        # Maintained to be compatible with old configuration files
        if type(self.config['data']['ahead'])==list:
            iahead = self.config['data']['ahead'][0]
            ahead = (self.config['data']['ahead'][1] - self.config['data']['ahead'][0]) + 1
        else:
            iahead = 1
            ahead = self.config['data']['ahead']

        lresults = []
        for i, p in zip(range(1, ahead + 1), range(iahead, self.config['data']['ahead'][1]+1)):
            lresults.append([p] + ErrorMeasure().compute_errors(val_y[:, i - 1],
                                                               val_yp[:, i - 1],
                                                               test_y[:, i - 1],
                                                               test_yp[:, i - 1]))

        return lresults
