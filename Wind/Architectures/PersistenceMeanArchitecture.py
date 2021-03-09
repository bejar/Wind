"""
.. module:: PersistenceArchitecture

PersistenceArchitecture
******

:Description: PersistenceArchitecture

    Class for persistence model

:Authors:
    bejar

:Version: 

:Date:  13/07/2018
"""

import numpy as np

from Wind.Architectures.PersistenceArchitecture import PersistenceArchitecture
from Wind.ErrorMeasure import ErrorMeasure

__author__ = 'bejar'


class PersistenceMeanArchitecture(PersistenceArchitecture):
    """Class for persistence model plus the mean

     """
    ## Data mode default for input, 1 dimensional output
    data_mode = ('2D', '2D')
    modname = 'PersistenceMean'

    def generate_model(self):
        """
        Generates the model

        :return:
        """
        for v in self.config['arch']['alpha']:
            if not (0 <= v <= 1):
                raise NameError(f"Alpha parameter value {self.config['arch']['alpha']} not valid")

    def train(self, train_x, train_y, val_x, val_y):
        """
        Trains the model

        :return:
        """
        pass


    def evaluate(self, val_x, val_y, test_x, test_y, scaler=None, save_errors=None):
        """
        Evaluates the training
        :param save_errors:
        :return:
        """

        alpha = self.config['arch']['alpha']
        if len(alpha) < val_y.shape[1]:
            alpha.extend([alpha[-1]] * (val_y.shape[1] - len(alpha)))

        if type(self.config['data']['ahead']) == list:
            ahead = self.config['data']['ahead'][1]
        else:
            ahead = self.config['data']['ahead']

        lresults = []
        for a, i in zip(alpha, range(1, ahead + 1)):
            lresults.append([i] +
                            ErrorMeasure().compute_errors((val_x[:, -1] * a) + ((1 - a) * np.mean(val_x, axis=1)),
                                                        val_y[:, i - 1],
                                                        (test_x[:, -1] * a) + ((1 - a) * np.mean(test_x, axis=1)),
                                                        test_y[:, i - 1], scaler=scaler))

        return lresults


