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

from Wind.Architectures.Architecture import Architecture
from sklearn.metrics import r2_score

__author__ = 'bejar'


class PersistenceArchitecture(Architecture):
    """Class for persistence model

    """
    ## Data mode default for input, 1 dimensional output
    data_mode = ('2D', '2D')

    def generate_model(self):
        """
        Generates the model

        :return:
        """
        pass

    def train(self, train_x, train_y, val_x, val_y):
        """
        Trains the model

        :return:
        """
        pass

    def summary(self):
        """
        Model summary
        :return:
        """
        print("Persitence")

    def evaluate(self, val_x, val_y, test_x, test_y):
        """
        Evaluates the training
        :return:
        """

        if type(self.config['data']['ahead']) == list:
            ahead = self.config['data']['ahead'][1]
        else:
            ahead = self.config['data']['ahead']

        lresults = []
        for i in range(1, ahead + 1):
            lresults.append((i,
                             r2_score(val_x[:, -1] , val_y[:, i - 1]),
                             r2_score(test_x[:, -1] , test_y[:, i - 1])
                             ))

        return lresults

        #
        # r2val = r2_score(val_x[:, -1], val_y[:, 0])
        # r2test = r2_score(test_x[:, -1], test_y[:, 0])
        #
        # return r2val, r2test


